"""Durable headless session workflow for live block capture.

The public ``SessionWorkflow`` interface is intentionally small. SQLite session
state, Pi outbox publication, HTTP transfer, and background preprocessing sit
behind adapters so a console or touchscreen only consumes commands and events.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from time import perf_counter, sleep
from typing import Callable, Protocol, TypedDict
import cv2
import numpy as np

from capture_storage import CaptureRecord
from camera_calibration import (
    ActivatedCameraMode,
    CalibrationQuality,
    LockedCameraControls,
    PhaseCameraCalibration,
)
from capture_session import (
    CaptureSession,
    SessionEvent as CaptureSessionEvent,
)
from slide.qr import (
    DECODE_BUDGET_SECONDS,
    SlideQRResult,
    decode_slide_identity,
    scanner_identity,
)
from session.atomic_io import (  # noqa: F401
    atomic_json as _atomic_json,
    sha256 as _sha256,
)
from session.outbox_transport import HttpCaptureClient, PiOutbox
from session.outbox_transport import (  # noqa: F401
    default_debug_snap_dir,
    open_saved_image,
    save_debug_snap,
)
from session.rpc_server import (  # noqa: F401
    LoopbackCaptureReceiver,
    _RPC_ARITY,
    _RPC_METHODS,
)
from session.session_mode import SessionMode
from session.processing_store import (  # noqa: F401
    ProcessingStore,
    default_work_order_scorer,
    format_profile_summary_row,
)
from session.profile_report import ProfileRow, project_profile_rows
from verify.slide_image_overlay import build_slide_image_overlay  # noqa: F401

from session.workflow_types import (
    BlockReadiness,
    FailedBlockWarning,
    FramingCalibration,
    ScanOutcome,
    SessionIdentity,
    SessionSummary,
    SlideStageTimings,
    UploadReceipt,
    WorkflowEvent,
    WorkflowSnapshot,
)
from session.workflow_types import (  # noqa: F401
    ClaimOutcome,
    HybridPoolFreezeResult,
    RecaptureOutcome,
    WorkOrderScoringResult,
)

log = logging.getLogger(__name__)


class HybridProfileStatus(TypedDict):
    """#258: ``hybrid_profile_status``'s return shape -- a plain dict at
    runtime (so existing callers' ``status["rows"]``/``["queue_count"]`` and
    every test fake's bare dict literal keep working unchanged), but typed
    precisely so callers narrow ``rows``/``queue_count`` correctly instead
    of passing ``object`` straight into ``format_profile_console``/
    ``profile_screen_fields``."""

    queue_count: int
    rows: tuple[ProfileRow, ...]


class PhaseCamera(Protocol):
    def activate_mode(self, mode: str) -> ActivatedCameraMode: ...


class _HeadlessPhaseCamera:
    """No-hardware adapter with deterministic mode-bound activation state."""

    def activate_mode(self, mode: str) -> ActivatedCameraMode:
        controls = LockedCameraControls(
            exposure_time_us=1,
            analogue_gain=1.0,
            colour_gains=(1.0, 1.0),
        )
        quality = CalibrationQuality(
            stable=True,
            sample_count=1,
            settling_frames=0,
            exposure_cv=0.0,
            gain_cv=0.0,
            red_gain_cv=0.0,
            blue_gain_cv=0.0,
            background_luma_median=220.0,
            clipped_high_fraction=0.0,
            clipped_low_fraction=0.0,
        )
        calibration = PhaseCameraCalibration(
            mode=mode,
            controls=controls,
            quality=quality,
            metadata_samples=(),
        )
        baseline = np.full((480, 640, 3), 220, dtype=np.uint8)
        return ActivatedCameraMode(calibration, baseline)


class FramingCalibrationStore:
    """Durable physical alignment approval, independent of session baselines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def view(self) -> FramingCalibration | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return FramingCalibration(
            Path(payload["image_path"]),
            datetime.fromisoformat(payload["approved_at"]),
        )

    def approve(
        self, image_path: str | Path, *, approved_at: datetime
    ) -> FramingCalibration:
        if approved_at.tzinfo is None:
            raise ValueError("approved_at must be timezone-aware")
        calibration = FramingCalibration(
            Path(image_path), approved_at.astimezone(timezone.utc)
        )
        _atomic_json(
            self.path,
            {
                "image_path": str(calibration.image_path),
                "approved_at": calibration.approved_at.isoformat(),
            },
        )
        return calibration

    def recalibrate(
        self, image_path: str | Path, *, approved_at: datetime
    ) -> FramingCalibration:
        return self.approve(image_path, approved_at=approved_at)


class SessionWorkflow:
    """Permanent command/event interface consumed by future presentation code."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        store: ProcessingStore,
        outbox: PiOutbox,
        transport: HttpCaptureClient,
        camera: PhaseCamera | None = None,
        framing_calibration: FramingCalibrationStore | None = None,
        slide_decoder: Callable[[np.ndarray], SlideQRResult] | None = None,
        clock: Callable[[], float] | None = None,
        session_mode: SessionMode = SessionMode.NORMAL,
        hybrid_descriptor_names: tuple[str, ...] = (),
        hybrid_candidate_configuration: dict[str, object] | None = None,
    ):
        self.session = session
        self.store = store
        self.outbox = outbox
        self.transport = transport
        self.camera = camera or _HeadlessPhaseCamera()
        self.slide_decoder = slide_decoder or decode_slide_identity
        self._clock = clock or perf_counter
        # #250: NORMAL (the default) and OPEN_RETRIEVAL make `poll_drain`
        # behave exactly as it always has -- only HYBRID/HYBRID_SHADOW take
        # the Hybrid Candidate Pool freeze branch below.
        self.session_mode = session_mode
        self.hybrid_descriptor_names = tuple(hybrid_descriptor_names)
        self.hybrid_candidate_configuration = hybrid_candidate_configuration
        self.last_slide_stage_timings: SlideStageTimings | None = None
        # #257: pure in-process flag for "View Results while a slide-capture
        # work order is open" -- pause_capture/resume_capture below only ever
        # flip this bool. No store I/O, no job wait/cancel: background
        # scoring and queued jobs are completely unaffected by entering or
        # leaving Results. The automatic capture loop (PiCaptureRuntime /
        # tools/run_pi_session.py, outside this module) is expected to read
        # this flag before firing an automatic shot; see the methods below.
        self.capture_paused: bool = False
        # #256 follow-up: the operator-armed recapture target, in-memory
        # only (no migration -- see `arm_hybrid_recapture` below for why).
        # Read-and-cleared by the NEXT `capture_slide` call, matching or
        # not, so a stale arm can never survive past one physical capture.
        # Lost on restart exactly like `capture_paused` above; the operator
        # simply re-arms.
        self._pending_recapture_capture_id: str | None = None
        if framing_calibration is not None:
            self.framing_calibration = framing_calibration
        else:
            # `store.root` is a processing-computer path; it is meaningless
            # (and, over the wire, absent) on the Pi. Only fall back to it
            # when `store` really is the local `ProcessingStore` test seam --
            # detected structurally via `.root`, not an isinstance check on
            # `RemoteProcessingStore` (that would import remote_store here
            # and risk a cycle). See ADR 0002 boundary rule #1: framing
            # calibration is Pi-local, `store.root` never crosses the wire.
            root = getattr(self.store, "root", None)
            if root is None:
                raise ValueError(
                    "a remote store requires an explicit Pi-local "
                    "framing_calibration (SessionWorkflow cannot default "
                    "one from store.root over the wire)"
                )
            self.framing_calibration = FramingCalibrationStore(
                root / "framing_calibration.json"
            )
        self._calibrations: dict[str, object] = {}
        self._baselines: dict[str, object] = {}
        self._active_camera_mode: str | None = None
        self._slide_capture_session: CaptureSession | None = None
        self._slide_recovery_state = self.store.slide_recovery_state(
            self.session.number
        )

    def scan_block(self, block_id: str) -> ScanOutcome:
        return self.store.scan_block(self.session.number, block_id)

    def precheck_slide_scan(self, payload: str) -> bool:
        """True if a handheld slide ``payload`` may be stashed for the next
        capture; False if its block already has a durable verdict (a duplicate
        re-scan), in which case the store emits ``duplicate_slide_scan``.

        The payload is resolved through the same scanner grammar
        ``capture_slide`` uses. An unresolvable payload passes through (True) so
        the existing capture/reposition flow surfaces the failure -- the guard
        only fast-fails confident duplicates.
        """
        result = scanner_identity(payload)
        if not result.success or not result.block_id:
            return True
        return self.store.precheck_slide_scan(self.session.number, result.block_id)

    def awaiting_capture_blocks(self) -> tuple[str, ...]:
        return self.store.awaiting_capture_blocks(self.session.number)

    def unscan_block(self, block_id: str) -> bool:
        return self.store.unscan_block(self.session.number, block_id)

    def capture_block(
        self, block_id: str, source: str | Path, *, captured_at: datetime
    ) -> UploadReceipt | None:
        scan = self.scan_block(block_id)
        if not scan.accepted:
            raise ValueError(scan.message)
        return self.publish_scanned_block(block_id, source, captured_at=captured_at)

    def publish_scanned_block(
        self, block_id: str, source: str | Path, *, captured_at: datetime,
        profile: bool = False,
    ) -> UploadReceipt | None:
        """Publish/upload a block whose operator scan was already accepted."""
        capture = self.outbox.publish_block(
            source, block_id, captured_at, profile=profile
        )
        receipts = self.outbox.replay(self.session.number, self.transport)
        return next(
            (receipt for receipt in receipts if receipt.capture_id == capture.capture_id),
            None,
        )

    def poll_status(self) -> tuple[UploadReceipt, ...]:
        """One presentation-loop poll that also drains recovered connectivity."""
        receipts = self.outbox.replay(self.session.number, self.transport)
        self.outbox.replay_slides(self.session.number, self.store)
        if self.store.snapshot(self.session).phase == "draining_blocks":
            self.poll_drain(replay=False)
        return receipts

    def finish_blocks(self) -> WorkflowSnapshot:
        self.store.begin_block_drain(self.session.number)
        return self.poll_drain()

    def poll_drain(self, *, replay: bool = True) -> WorkflowSnapshot:
        if replay:
            self.outbox.replay(self.session.number, self.transport)
        snapshot = self.snapshot()
        if snapshot.phase != "draining_blocks":
            return snapshot
        self.store.record_event(
            self.session.number,
            "block_drain_progress",
            f"{snapshot.pending_transfers} transfers and "
            f"{snapshot.preprocessing_pending} preprocessing jobs pending; "
            f"{snapshot.unresolved_blocks} blocks unresolved",
        )
        # #250: HYBRID/HYBRID_SHADOW freeze the Hybrid Candidate Pool (or, on
        # <2 usable blocks, bounce back to the blocks phase) instead of the
        # unconditional try_enter_slides transition every other mode uses.
        # `freeze_hybrid_pool` never raises at all -- an unknown session
        # returns a not-frozen result, exactly like `try_enter_slides`'s own
        # `bool(row and ...)` contract -- so this stays safe on the
        # background poll tick that also reaches this method.
        if self.session_mode in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
            self.store.freeze_hybrid_pool(
                self.session.number,
                descriptor_names=self.hybrid_descriptor_names,
                candidate_configuration=self.hybrid_candidate_configuration,
            )
        else:
            self.store.try_enter_slides(self.session.number)
        return self.snapshot()

    def prepare_empty_backlight(self, mode: str) -> None:
        """Camera Calibration for ``mode`` (Empty-Backlight Setup, first half)."""
        self._activate_camera_mode(mode)
        if mode == "slide":
            self.store.record_event(
                self.session.number,
                "slide_mode_entered",
                "Slide camera calibrated, controls locked, and "
                "empty-backlight baseline collected",
            )

    def require_slide_mode(self) -> None:
        if self.snapshot().phase != "slides":
            raise RuntimeError("slide actions require slide mode")
        if self._active_camera_mode != "slide":
            raise RuntimeError("slide camera calibration is not active")

    def capture_slide(
        self,
        source: str | Path,
        *,
        captured_at: datetime,
        scanned_payload: str | None = None,
        profile: bool = False,
    ) -> SlideQRResult:
        """Persist and decode one still requested by the settled-slide lifecycle.

        ``profile`` (#258) mirrors ``publish_scanned_block``'s own ``profile``
        parameter for blocks: threaded straight through to
        ``PiOutbox.publish_slide`` regardless of ``result.success``, so an
        unreadable slide's outbox entry still carries the flag durably (the
        main computer's own ``record_slide_capture`` is what actually gates
        collection on an accepted, in-pool Hybrid claim).

        #256 follow-up: read-and-clear whatever ``arm_hybrid_recapture``
        armed, BEFORE this capture's outcome is even known. This is the ONE
        place a physical capture is routed: the durable outbox entry below
        carries the result as ``supersedes`` so ``PiOutbox.replay_slides``
        calls ``store.recapture_hybrid_slide`` for THIS capture instead of
        the ordinary ``store.record_slide_capture`` -- whether the decode
        matches the armed row's own claim or not (the store method itself
        is what decides match vs. mismatch; this method never duplicates
        that check). Reading-and-clearing here, rather than inside the
        outbox/replay layer, is what guarantees the arm consumes exactly
        ONE physical capture: a failed or mismatched attempt does not leave
        it armed for a later, unrelated slide.
        """
        self.require_slide_mode()
        supersedes = self._pending_recapture_capture_id
        self._pending_recapture_capture_id = None
        if scanned_payload is not None:
            # Intentionally skips both the raster decode and its readability
            # guard: identity resolves from the payload alone, so an
            # unreadable/missing still no longer raises here. The still is
            # still durably queued to this machine's outbox first; an
            # unreadable image is caught by the main computer's own
            # `record_slide_capture` readability guard when the capture is
            # replayed (ADR-0014 / #185).
            started = self._clock()
            result = scanner_identity(scanned_payload)
        else:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("slide capture could not be read")
            started = self._clock()
            result = self.slide_decoder(image)
            del image
        decode_end = self._clock()
        elapsed = decode_end - started
        decode_ms = elapsed * 1000.0
        if not result.success and elapsed < DECODE_BUDGET_SECONDS:
            sleep(DECODE_BUDGET_SECONDS - elapsed)
            duration_ms = (self._clock() - started) * 1000.0
        else:
            duration_ms = decode_ms
        outbox_ms: float | None = None
        send_ms: float | None = None
        slide_capture_id: str | None = None
        if result.success:
            outbox_start = self._clock()
            published = self.outbox.publish_slide(
                source, captured_at, result=result, duration_ms=duration_ms,
                profile=profile, supersedes=supersedes,
            )
            slide_capture_id = published.capture_id
            outbox_end = self._clock()
            outbox_ms = (outbox_end - outbox_start) * 1000.0
            send_start = self._clock()
            self.outbox.replay_slides(self.session.number, self.store)
            send_end = self._clock()
            send_ms = (send_end - send_start) * 1000.0
        else:
            published = self.outbox.publish_slide(
                source, captured_at, result=result, duration_ms=duration_ms,
                profile=profile, supersedes=supersedes,
            )
            slide_capture_id = published.capture_id
            self.outbox.replay_slides(self.session.number, self.store)
        self.last_slide_stage_timings = SlideStageTimings(
            decode_ms=decode_ms, outbox_ms=outbox_ms, send_ms=send_ms,
            capture_id=slide_capture_id,
        )
        self._slide_recovery_state = (
            "waiting_for_removal" if result.success else "reposition"
        )
        return result

    def consume_slide_capture(self, capture: CaptureRecord) -> SlideQRResult:
        """Bridge one automatic slide capture into identity processing."""
        if capture.role != "slide":
            raise ValueError("slide identity requires a slide capture")
        return self.capture_slide(
            capture.path, captured_at=capture.captured_at
        )

    def restore_slide_capture_session(self, capture_session: CaptureSession) -> None:
        """Restore durable unreadable-slide recovery into the frame state machine."""
        self.require_slide_mode()
        self._slide_capture_session = capture_session
        capture_session.add_event_listener(self._on_slide_capture_event)
        state = self.store.slide_recovery_state(self.session.number)
        self._slide_recovery_state = state
        if state == "reposition":
            capture_session.restore_unreadable_slide()
        elif state == "waiting_for_removal":
            capture_session.restore_waiting_for_removal()

    def _on_slide_capture_event(self, event: CaptureSessionEvent) -> None:
        try:
            if event.kind == "removal_confirmed":
                self.store.mark_waiting_for_slide(self.session.number)
                self._slide_recovery_state = "waiting"
            elif event.kind == "slide_reposition_required":
                self.store.record_event(
                    self.session.number,
                    "slide_reposition_required",
                    "Reposition slide",
                )
        except (OSError, sqlite3.Error):
            # The stored state remains fail-closed and will be retried/recovered
            # on restart; a transient persistence failure must not stop preview.
            if (
                event.kind == "removal_confirmed"
                and self._slide_capture_session is not None
            ):
                if self._slide_recovery_state == "reposition":
                    self._slide_capture_session.restore_unreadable_slide()
                else:
                    self._slide_capture_session.restore_waiting_for_removal()
            return

    def skip_unreadable_slide(self, capture_session: CaptureSession) -> None:
        """Durably skip the current unidentified slide."""
        self.require_slide_mode()
        if not capture_session.unreadable_slide_can_be_skipped:
            raise RuntimeError("Skip is only valid for an unreadable slide")
        self.store.skip_unreadable_slide(self.session.number)
        self._slide_recovery_state = "waiting_for_removal"
        capture_session.skip_unreadable_slide()

    def baseline_for(self, mode: str) -> object:
        if mode not in ("block", "slide"):
            raise ValueError("mode must be 'block' or 'slide'")
        if mode not in self._baselines:
            raise RuntimeError(f"no {mode} baseline has been collected")
        return self._baselines[mode]

    def calibration_for(self, mode: str) -> object:
        if mode not in ("block", "slide"):
            raise ValueError("mode must be 'block' or 'slide'")
        if mode not in self._calibrations:
            raise RuntimeError(f"no {mode} calibration is active")
        return self._calibrations[mode]

    def view_framing_calibration(self) -> FramingCalibration | None:
        return self.framing_calibration.view()

    def approve_framing_calibration(
        self, image_path: str | Path, *, approved_at: datetime
    ) -> FramingCalibration:
        calibration = self.framing_calibration.approve(
            image_path, approved_at=approved_at
        )
        self.store.record_event(
            self.session.number,
            "framing_calibration_approved",
            f"Framing calibration approved: {calibration.image_path}",
        )
        return calibration

    def recalibrate_framing(
        self, image_path: str | Path, *, approved_at: datetime
    ) -> FramingCalibration:
        calibration = self.framing_calibration.recalibrate(
            image_path, approved_at=approved_at
        )
        self.store.record_event(
            self.session.number,
            "framing_recalibrated",
            f"Framing calibration replaced: {calibration.image_path}",
        )
        return calibration

    def _activate_camera_mode(self, mode: str) -> None:
        self._active_camera_mode = None
        self._calibrations.clear()
        self._baselines.clear()
        activated = self.camera.activate_mode(mode)
        self._persist_camera_calibration(activated.calibration)
        # Atomically publish only the newly activated mode. A baseline and its
        # locked controls are one unit and must never survive a mode switch.
        self._calibrations = {mode: activated.calibration}
        self._baselines = {mode: activated.baseline}
        self._active_camera_mode = mode

    def _persist_camera_calibration(self, calibration: PhaseCameraCalibration) -> None:
        controls = calibration.controls
        quality = calibration.quality
        _atomic_json(
            self.outbox.directory / f"camera_calibration_{calibration.mode}.json",
            {
                "mode": calibration.mode,
                "calibration_id": calibration.calibration_id,
                "calibrated_at": calibration.calibrated_at.isoformat(),
                "controls": {
                    "exposure_time_us": controls.exposure_time_us,
                    "analogue_gain": controls.analogue_gain,
                    "colour_gains": list(controls.colour_gains),
                    "frame_duration_us": controls.frame_duration_us,
                },
                "quality": {
                    "stable": quality.stable,
                    "sample_count": quality.sample_count,
                    "settling_frames": quality.settling_frames,
                    "exposure_cv": quality.exposure_cv,
                    "gain_cv": quality.gain_cv,
                    "red_gain_cv": quality.red_gain_cv,
                    "blue_gain_cv": quality.blue_gain_cv,
                    "background_luma_median": quality.background_luma_median,
                    "clipped_high_fraction": quality.clipped_high_fraction,
                    "clipped_low_fraction": quality.clipped_low_fraction,
                    "failure_reason": quality.failure_reason,
                },
            },
        )

    def wait_for_block_jobs(self) -> None:
        self.store.wait_for_jobs()

    def recapture_block(
        self, block_id: str, source: str | Path, *, captured_at: datetime
    ) -> UploadReceipt | None:
        warning_ids = {warning.block_id for warning in self.active_warnings()}
        if block_id not in warning_ids:
            raise ValueError("only the intended failed block can be recaptured")
        capture = self.outbox.publish_block(
            source, block_id, captured_at, recapture=True
        )
        receipts = self.outbox.replay(self.session.number, self.transport)
        return next(
            (receipt for receipt in receipts if receipt.capture_id == capture.capture_id),
            None,
        )

    def dismiss_block(self, block_id: str, *, reason: str) -> None:
        self.store.dismiss_block(self.session.number, block_id, reason=reason)

    def active_warnings(self) -> tuple[FailedBlockWarning, ...]:
        return self.store.active_warnings(self.session.number)

    def block_readiness(self, block_id: str) -> BlockReadiness:
        return self.store.block_readiness(self.session.number, block_id)

    def events(self) -> tuple[WorkflowEvent, ...]:
        return self.store.events(self.session.number)

    def start_work_order(self) -> int:
        return self.store.start_work_order(self.session.number)

    def finish_work_order(self) -> int:
        return self.store.finish_work_order(self.session.number)

    def open_work_order_id(self) -> int | None:
        return self.store.open_work_order_id(self.session.number)

    def has_work_orders(self) -> bool:
        return self.store.has_work_orders(self.session.number)

    def list_results_ready_work_orders(self) -> tuple[dict[str, object], ...]:
        return self.store.list_results_ready_work_orders(self.session.number)

    def retry_hybrid_slide(
        self, capture_id: str, *, request_id: str | None = None,
    ) -> bool:
        """#256: operator-triggered retry of a Hybrid Processing Error --
        re-runs scoring against the durably saved capture, no new photo."""
        return self.store.retry_hybrid_slide(
            self.session.number, capture_id, request_id=request_id,
        )

    def recapture_hybrid_slide(
        self,
        superseded_capture_id: str,
        source: str | Path,
        *,
        captured_at: datetime,
        result: SlideQRResult,
        duration_ms: float,
        request_id: str | None = None,
        source_token: str | None = None,
    ) -> RecaptureOutcome:
        """#256: an accepted recapture. Supersedes `superseded_capture_id`
        only when `result`'s decoded claim matches its own claimed block."""
        return self.store.recapture_hybrid_slide(
            self.session.number, superseded_capture_id, source,
            captured_at=captured_at, result=result, duration_ms=duration_ms,
            request_id=request_id, source_token=source_token,
        )

    def arm_hybrid_recapture(self, capture_id: str) -> None:
        """#256 follow-up: arm the runtime so the NEXT physically captured
        slide routes through `recapture_hybrid_slide` for `capture_id`
        instead of the ordinary `record_slide_capture` path -- the missing
        link between the passive attention banner and a real recapture.

        Eligibility is the EXACT SAME `can_recapture` gate the attention
        banner itself already computes (`kiosk.attention.
        project_attention_banner`, over the SAME `kiosk.results_table.
        project_results_table`-projected rows the kiosk relay builds) --
        called here, never re-derived, so arming can never diverge from
        what the operator was shown. Raises ``ValueError`` (never silently
        no-ops) when `capture_id` is not the current attention item, or
        when it is but a DIFFERENT work order is actively capturing right
        now (the same "wait for an available transition" case the banner
        itself renders as `can_recapture: false`).

        Held IN MEMORY ONLY, deliberately -- no migration, no new column
        (mirrors `capture_paused`'s own in-process-only shape): losing an
        armed recapture across a restart just means the operator re-arms:
        arming is a short-lived "the next shot is special" intent, not
        durable session state. See `capture_slide` for where this is read
        and cleared, and `disarm_hybrid_recapture` for the cancel path.
        """
        from kiosk.attention import project_attention_banner
        from kiosk.results_table import project_results_table

        rows = project_results_table(self.results_status().get("rows") or [])
        open_work_order_id = self.open_work_order_id()
        banner = project_attention_banner(
            rows, work_order_open=open_work_order_id is not None,
            open_work_order_id=open_work_order_id,
        )
        if banner is None or banner.get("capture_id") != capture_id:
            raise ValueError(
                f"{capture_id} is not the current Hybrid attention item"
            )
        if not banner.get("can_recapture"):
            raise ValueError(
                "recapture is not available while a different work order "
                "is actively capturing"
            )
        self._pending_recapture_capture_id = capture_id

    def disarm_hybrid_recapture(self) -> None:
        """Cancel a pending #256 follow-up recapture arm.

        A harmless no-op when nothing is armed -- mirrors `resume_capture`'s
        own unconditional-clear shape. Exists so an operator who armed by
        mistake (or changed their mind before placing the next slide) has
        an explicit way back, without waiting for an unrelated capture to
        consume (and mismatch-reject) the arm instead.
        """
        self._pending_recapture_capture_id = None

    def pause_capture(self) -> None:
        """#257: entering Results while a slide-capture work order is open
        pauses AUTOMATIC CAPTURE ONLY. Sets the `capture_paused` in-process
        flag the capture loop is expected to honor before firing an
        automatic shot; touches no store/job state, so background scoring
        and queued jobs are completely unaffected."""
        self.capture_paused = True

    def resume_capture(self) -> None:
        """Clear the #257 capture pause (Go Back from Results). Does not
        finish the work order, end the session, or otherwise touch the
        durable bracket/session state -- resuming is purely re-arming the
        same flag `pause_capture` set."""
        self.capture_paused = False

    def results_status(self) -> dict[str, object]:
        """#153/#252: the kiosk relay's live-results seam -- wraps the
        session's results source into the ``{"work_orders": (...),
        "rows": [...]}`` shape ``KioskRelay._results_status()`` reads via a
        degrading getattr+callable call, mirroring ``capture_status``.
        ``work_orders`` is the sorted set of distinct work order ids (not the
        raw per-row duplication); ``rows`` is the raw per-slide verdict rows.

        NORMAL keeps reading ``list_results_ready_work_orders``' batch-atomic
        reveal. Open Retrieval reads ``list_retrieval_results`` and Hybrid
        modes read their compatibility ``list_hybrid_results`` view: every
        per-slide retrieval job across every work order this session, scored
        or not, so the results table -- and the
        router gate it feeds via ``results_ready_work_orders`` -- is
        reachable the moment Finish Slides closes the bracket, without
        waiting on scoring. The mode check reads ``self.session_mode``, the
        same durable, resume-carried value ``poll_drain`` already branches on
        for the Hybrid Candidate Pool freeze -- never inferred from whether
        rows/artifacts happen to exist.

        Both store calls are proxied over ``/rpc`` by ``RemoteProcessingStore``
        (#149 transport fix for the NORMAL/OPEN_RETRIEVAL path; the Hybrid
        path is proxied the same way), so this reads through to the PC store
        from the Pi. This method runs on the kiosk's poll path (``relay.state``
        -> ``results_status``), which reaches ``_camera_loop``'s bare
        ``except Exception`` if anything escapes -- so both branches degrade
        to empty rows rather than raise: the ``AttributeError`` degrade is a
        defensive fallback for any store handle that predates its proxy, and
        the broad ``except Exception`` around ``list_hybrid_results`` is the
        same blast-radius guard even though the store contract promises that
        method never raises on its own poll path."""
        from kiosk.results_table import evidence_paths_for_capture

        if self.session_mode == SessionMode.OPEN_RETRIEVAL:
            try:
                rows = list(self.store.list_retrieval_results(self.session.number))
            except Exception:
                log.warning(
                    "results_status: list_retrieval_results raised for session "
                    "%s; degrading to empty rows so the kiosk poll survives",
                    self.session.number,
                    exc_info=True,
                )
                return {"work_orders": (), "rows": []}
        elif self.session_mode in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
            try:
                rows = list(self.store.list_hybrid_results(self.session.number))
            except Exception:
                log.warning(
                    "results_status: list_hybrid_results raised for session "
                    "%s; degrading to empty rows so the kiosk poll survives",
                    self.session.number,
                    exc_info=True,
                )
                return {"work_orders": (), "rows": []}
        else:
            try:
                rows = list(self.list_results_ready_work_orders())
            except AttributeError:
                return {"work_orders": (), "rows": []}
        work_orders = tuple(sorted({row["work_order_id"] for row in rows}))
        claim_artifacts_dir = self.session.directory / "claim_artifacts"
        enriched: list[dict[str, object]] = []
        for row in rows:
            new_row = dict(row)
            new_row["evidence"] = evidence_paths_for_capture(
                claim_artifacts_dir,
                str(new_row["capture_id"]),
                new_row.get("verdict"),
            )
            enriched.append(new_row)
        return {"work_orders": work_orders, "rows": enriched}

    def hybrid_profile_status(self, *, now_ns: int) -> HybridProfileStatus:
        """#258: the ONE shared source both the kiosk relay and the console
        read for ``--profile`` Hybrid queue/timing display -- projects
        ``ProcessingStore.list_hybrid_profile_rows`` through
        ``session.profile_report.project_profile_rows`` into the display-
        ready ``ProfileRow`` tuple, alongside the queue count (rows still
        ``"PENDING"``).

        Mirrors ``results_status``'s own mode-gate + broad-except degrade
        shape exactly, including running on the same kiosk poll path
        (``relay.state()`` -> this method -> ``_camera_loop``'s bare
        ``except Exception``): NORMAL/OPEN_RETRIEVAL, or any unexpected
        store failure, both return the same empty-queue default rather than
        raise or propagate.

        This method has NO opinion on the ``--profile`` CLI/launch flag --
        callers (``KioskRelay``, the Pi console) each check their OWN
        ``.profile`` attribute before ever calling this, which is what keeps
        "no --profile means nothing rendered" true even though
        ``list_hybrid_profile_rows``'s ``profile_enabled`` row filter
        already makes it true independently by construction.
        """
        if self.session_mode not in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
            return {"queue_count": 0, "rows": ()}
        try:
            raw_rows = list(self.store.list_hybrid_profile_rows(self.session.number))
        except Exception:
            log.warning(
                "hybrid_profile_status: list_hybrid_profile_rows raised for "
                "session %s; degrading to an empty queue so the kiosk poll "
                "survives",
                self.session.number,
                exc_info=True,
            )
            return {"queue_count": 0, "rows": ()}
        rows = project_profile_rows(raw_rows, now_ns=now_ns)
        queue_count = sum(1 for row in rows if row.state == "PENDING")
        return {"queue_count": queue_count, "rows": rows}

    def results_evidence_bytes(self, path: str) -> bytes | None:
        """Read-only JPEG bytes for ``GET /results-evidence`` (#236)."""
        remote_reader = getattr(self.store, "results_evidence_bytes", None)
        if callable(remote_reader):
            return remote_reader(self.session.number, path)
        return self.inspection_sheet_bytes(path)

    def inspection_sheet_bytes(self, path: str) -> bytes | None:
        """#153/#151: read-only contact-sheet PNG bytes for
        ``KioskRelay.inspection_sheet_bytes``'s degrading getattr+callable
        seam. Mirrors the simplicity of ``review_still_jpeg`` -- a bare read,
        None on any OS-level failure (missing file, bad path) rather than
        raising through the kiosk HTTP layer."""
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def snapshot(self) -> WorkflowSnapshot:
        snapshot = self.store.snapshot(self.session)
        pending = (
            len(self.outbox.pending()) + len(self.outbox.pending_slides())
            + len(self.outbox.invalid_entries())
            + len(self.outbox.invalid_slide_entries())
        )
        return replace(
            snapshot,
            upload_state="idle" if pending == 0 else "active",
            pending_transfers=pending,
        )

    def summarize(self) -> SessionSummary:
        summary = self.store.summarize(self.session)
        return replace(
            summary,
            pending_uploads=tuple(
                capture.capture_id for capture in self.outbox.pending()
            ) + tuple(
                capture.capture_id for capture in self.outbox.pending_slides()
            ) + self.outbox.invalid_entries() + self.outbox.invalid_slide_entries(),
        )

    def end_session(self, *, confirm: bool) -> WorkflowSnapshot:
        """Explicit, deliberate finalization; a no-confirmation call changes nothing."""
        if not confirm:
            raise ValueError("End session requires confirmation")
        self.store.begin_finalization(self.session.number)
        return self.poll_finalization()

    def poll_finalization(self, *, replay: bool = True) -> WorkflowSnapshot:
        if replay:
            self.outbox.replay(self.session.number, self.transport)
            self.outbox.replay_slides(self.session.number, self.store)
        snapshot = self.snapshot()
        if snapshot.phase == "finalizing" and snapshot.pending_transfers:
            return snapshot
        if snapshot.phase == "finalizing":
            self.store.prepare_finalization(self.session.number)
            snapshot = self.snapshot()
        if snapshot.phase == "cleanup_pending":
            try:
                self.outbox.delete_acknowledged()
                self.store.complete_finalization(self.session.number)
            except Exception as exc:
                self.store.record_finalization_error(
                    self.session.number, f"Finalization cleanup failed: {exc}"
                )
                raise
        elif snapshot.phase == "finalized":
            self.store.complete_finalization(self.session.number)
        return self.snapshot()


from session.workflow_public import PUBLIC_SYMBOLS  # noqa: E402

__all__ = list(PUBLIC_SYMBOLS)
