"""Run the SessionWorkflow on the Raspberry Pi against a remote store.

Per ADR 0002, `SessionWorkflow` runs on the Pi (it is camera-coupled); the
processing computer's `ProcessingStore` is reached as a `RemoteProcessingStore`
proxy over the dedicated Ethernet link. This is the Pi-side counterpart to
`tools/run_receiver.py`, which runs on the processing computer and prints the
session number + URL this script connects to.

Boundary rule this script exists to enforce (ADR 0002 #1): framing
calibration is Pi-local and `store.root` never crosses the wire. This script
always builds its own Pi-local `FramingCalibrationStore` and passes it in
explicitly -- `SessionWorkflow` raises a clear `ValueError` if you forget,
rather than reaching for a `RemoteProcessingStore.root` that does not exist.

Usage (Pi, after `tools/run_receiver.py` has printed a session number and URL
on the processing computer):
    ./venv/bin/python tools/run_pi_session.py \\
        --receiver-url http://192.168.50.1:8077 --session 1

    # laptop-only shakeout, receiver on loopback:
    .\\venv\\Scripts\\python.exe tools\\run_pi_session.py ^
        --receiver-url http://127.0.0.1:8077 --session 1

At the prompt, type a command name (see the printed cheat-sheet); `help` or
`?` reprints it. Exit with Ctrl+C, Ctrl+D / Ctrl+Z+Enter (EOF), or by
closing the SSH window; all of those release the camera via ``runtime.close()``.

Pi-only camera imports are delayed until `main`, so this module remains
importable in off-hardware tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import argparse  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
import signal  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from threading import Event, RLock, Thread  # noqa: E402
import time  # noqa: E402
from typing import Mapping  # noqa: E402
from uuid import uuid4  # noqa: E402

from action_logger import ActionLogger  # noqa: E402
from capture_runtime import (  # noqa: E402
    CAPTURE_STAGE_TIMING_KEYS,
    SETTLING_STAGE_KEYS,
    CaptureConfiguration,
    CaptureController,
)
from capture_settings import (  # noqa: E402
    CAMERA,
    DETECTION,
    LIGHTS,
    CaptureSettingError,
    CaptureSettings,
)
from capture_telemetry import MotionCurveWriter, SettlingObserver  # noqa: E402
from constants import (  # noqa: E402
    BACKLIGHT_GPIO_PIN,
    CAPTURE_DIMENSIONS,
    ILLUMINATION_GPIO_PIN,
    MOTION_CURVE_FLUSH_INTERVAL_S,
    NEUTRAL_OBLIQUE_GPIO_PIN,
    PROFILE_SLOW_CAPTURE_MS,
    MOTION_SAMPLE_WINDOW_S,
    SETTLING_CONFIRMATION_FRAMES,
    TRIPLE_CAPTURE_EXPOSURES_US,
    TRIPLE_CAPTURE_VARIANTS,
)
from camera_profiles import apply_camera_profile  # noqa: E402
from capture_session import (  # noqa: E402
    CaptureMode,
    CaptureSession,
    CaptureState,
    SessionConfig,
    SessionEvent,
    summarize_motion_samples,
)
from capture_storage import CaptureStore  # noqa: E402
from focus import center_roi_sharpness  # noqa: E402
from deskew import DeskewConfigError, deskew_png, load_deskew_config  # noqa: E402
from camera_calibration import CameraCalibrationError  # noqa: E402
from illumination import Illumination  # noqa: E402

from kiosk import KioskRelay  # noqa: E402
from kiosk.server import KioskServer  # noqa: E402
from transport.remote import RemoteProcessingStore, StoreError, TransportError  # noqa: E402
from session.console import (  # noqa: E402
    COMMAND_HELP,
    command_cheat_sheet,
    dispatch,
    render_events,
    render_motion_sample,
    render_profile_capture_block,
    render_results,
    render_settings,
    render_summary,
)
from session.atomic_io import atomic_json as _atomic_json  # noqa: E402
from retrieval.hybrid_configuration import (  # noqa: E402
    HybridConfigurationError,
    load_hybrid_configuration,
)
from session.profile_report import format_profile_console  # noqa: E402
from session.session_mode import (  # noqa: E402
    HYBRID_FLAG,
    HYBRID_SHADOW_FLAG,
    OPEN_RETRIEVAL_FLAG,
    SessionMode,
    SessionModeConflictError,
    resolve_session_mode,
)
from session.workflow import (  # noqa: E402
    FramingCalibrationStore,
    HttpCaptureClient,
    PiOutbox,
    SessionSummary,
    SessionWorkflow,
    WorkflowEvent,
)
from slide.qr import scanner_identity  # noqa: E402

_DEFAULT_PI_LOCAL_DIR = Path.home() / "lji_pi_session"
# Pi-LOCAL path for the versioned Hybrid integration handoff, same
# "Pi-local durability" family as --calibration-path/--outbox-root. Only read
# (and only required to exist/validate) when the resolved SessionMode is
# HYBRID or HYBRID_SHADOW; NORMAL/OPEN_RETRIEVAL never touch this path.
_DEFAULT_HYBRID_CONFIG_PATH = _DEFAULT_PI_LOCAL_DIR / "hybrid_integration_handoff.json"
# Pi-local grid-deskew fit (homography JSON) used by the `deskew` verb.
_DEFAULT_DESKEW_CONFIG_PATH = _DEFAULT_PI_LOCAL_DIR / "deskew_grid.json"
# Pi-local operator overrides for camera, detection and light settings.
_DEFAULT_CAPTURE_SETTINGS_PATH = _DEFAULT_PI_LOCAL_DIR / "capture_settings.json"
# States where a camera change may run (or wait for) a fresh calibration.
_CAMERA_CHANGE_STATES = (
    CaptureState.WAITING_FOR_SCAN,
    CaptureState.AWAITING_BASELINE_CONFIRMATION,
    CaptureState.CALIBRATION_FAILED,
)
_DRAIN_POLL_INTERVAL_SECONDS = 1.0
# ~8 Hz live sharpness readout for the `focus` command.
FOCUS_SAMPLE_INTERVAL_S = 0.125
# Startup console label per resolved SessionMode; SessionMode.NORMAL
# is intentionally absent: the .get(..., "normal verification") default
# below is the single source for that label.
_MODE_LABELS: Mapping[SessionMode, str] = {
    SessionMode.OPEN_RETRIEVAL: "Open Retrieval",
    SessionMode.HYBRID: "Hybrid",
    SessionMode.HYBRID_SHADOW: "Hybrid Shadow",
}


def _build_illumination(
    *, include_neutral_oblique: bool = False, led_only_blocks: bool = False
) -> Illumination:
    """Build the configured GPIO adapter, importing it only at runtime."""
    from illumination import (
        GpiozeroDigitalOutput,
        GpiozeroPwmOutput,
        GpiozeroPwmBacklight,
        IlluminationController,
    )

    try:
        output = GpiozeroDigitalOutput()
        backlight = GpiozeroPwmBacklight()
        neutral_oblique = None
        if include_neutral_oblique:
            neutral_oblique = GpiozeroPwmOutput(NEUTRAL_OBLIQUE_GPIO_PIN)
        return IlluminationController(
            output,
            backlight=backlight,
            neutral_oblique=neutral_oblique,
            **({"backlight_brightness": 0.0} if led_only_blocks else {}),
        )
    except Exception as exc:
        for resource in ("neutral_oblique", "backlight", "output"):
            device = locals().get(resource)
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
        raise RuntimeError(
            f"unable to initialize BCM GPIO {ILLUMINATION_GPIO_PIN}/"
            f"{BACKLIGHT_GPIO_PIN}"
            + (f"/{NEUTRAL_OBLIQUE_GPIO_PIN}" if include_neutral_oblique else "")
            + f": {exc}"
        ) from exc


def _close_quietly(resource: object | None) -> None:
    """Best-effort startup cleanup that never replaces the original error."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        pass


def _stop_quietly(resource: object | None) -> None:
    """Best-effort kiosk cleanup for a partially started server."""
    if resource is None:
        return
    stop = getattr(resource, "stop", None)
    if stop is None:
        return
    try:
        stop()
    except Exception:
        pass


def _close_runtime_and_kiosk(
    runtime: object, kiosk_server: object | None
) -> Exception | None:
    """Attempt both shutdown operations and return, rather than mask, errors."""
    errors: list[Exception] = []
    stop = getattr(kiosk_server, "stop", None)
    if stop is not None:
        try:
            stop()
        except Exception as exc:
            errors.append(exc)
    close = getattr(runtime, "close", None)
    if close is not None:
        try:
            close()
        except Exception as exc:
            errors.append(exc)
    return errors[0] if errors else None


@dataclass(frozen=True)
class SlideConsumption:
    """Wraps one slide-mode `capture_slide` result with its per-stage
    decode/outbox/send durations (`SessionWorkflow.last_slide_stage_timings`)
    so `CaptureController` can forward the split without disturbing existing
    `.success`/`.block_id`/`.engine` readers -- unknown attributes fall
    through to the wrapped `SlideQRResult` via `__getattr__`."""

    result: object
    decode_ms: float | None
    outbox_ms: float | None
    send_ms: float | None
    capture_id: str | None

    def __getattr__(self, name):
        return getattr(self.result, name)


class PiCaptureRuntime:
    """Bridge the real capture controller to the durable remote workflow."""

    led_only_blocks = False

    def __init__(
        self,
        workflow: SessionWorkflow,
        camera: object,
        *,
        capture_root: str | Path,
        session_config: SessionConfig | None = None,
        action_logger: ActionLogger | None = None,
        review_captures: bool = False,
        allow_occupied_baseline: bool = False,
        open_retrieval: bool = False,
        hybrid: bool = False,
        hybrid_shadow: bool = False,
        profile: bool = False,
        illumination: Illumination | None = None,
        triple_capture: bool = False,
        led_only_blocks: bool = False,
        skip_removal_wait: bool = False,
        deskew_config_path: str | Path | None = None,
        capture_settings_path: str | Path | None = None,
    ):
        self.workflow = workflow
        self.deskew_config_path = Path(
            deskew_config_path
            if deskew_config_path is not None
            else _DEFAULT_DESKEW_CONFIG_PATH
        )
        self.camera = camera
        self.capture_root = Path(capture_root)
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.session_config = session_config or SessionConfig()
        # None keeps settings in memory only (tests); main() passes the Pi path.
        self.capture_settings = CaptureSettings(
            capture_settings_path, session_defaults=self.session_config
        )
        self.session_config = self.capture_settings.session_config(self.session_config)
        if led_only_blocks and (
            triple_capture or skip_removal_wait or allow_occupied_baseline
        ):
            raise ValueError(
                "LED-only blocks cannot use triple capture, skip removal, "
                "or an occupied baseline"
            )
        if led_only_blocks and illumination is None:
            raise ValueError("LED-only blocks require an illumination controller")
        self.led_only_blocks = led_only_blocks
        self._led_flush_frames = 0
        if led_only_blocks:
            self.camera.led_only_blocks = True
        self.review_captures = review_captures
        self.allow_occupied_baseline = allow_occupied_baseline
        # open_retrieval/hybrid/hybrid_shadow are independent launch
        # booleans (one per CLI flag), resolved ONCE here into a single
        # explicit `self.session_mode`. Not to be confused with `self.mode`,
        # the CaptureMode block/slide axis set by `_build_controller`.
        # `self.open_retrieval` is a derived projection kept for gating call
        # sites and the kiosk relay's getattr read. Hybrid/Hybrid Shadow use
        # the same capture flow as normal verification; their one
        # Finish-Blocks difference is that `poll_drain` freezes the Hybrid
        # Candidate Pool instead of calling `try_enter_slides`.
        self.session_mode = resolve_session_mode(
            open_retrieval=open_retrieval,
            hybrid=hybrid,
            hybrid_shadow=hybrid_shadow,
        )
        self.open_retrieval = self.session_mode is SessionMode.OPEN_RETRIEVAL
        self.profile = profile
        self.illumination = illumination
        self.triple_capture = triple_capture
        self.skip_removal_wait = skip_removal_wait
        # Pi-local-first durability: the motion curve and settling
        # observer only exist under --profile, and live under the same
        # Pi-local capture_root as PiOutbox/FramingCalibrationStore -- never
        # synced until end_session explicitly uploads it.
        self.motion_curve_path: Path | None = None
        self.profile_config_path: Path | None = None
        self.motion_curve_writer: MotionCurveWriter | None = None
        self._settling_observer: SettlingObserver | None = None
        self._last_frame_now: float = 0.0
        if self.profile:
            self.motion_curve_path = self.capture_root / "motion_curve.csv"
            self.profile_config_path = self.capture_root / "profile_config.json"
            self.profile_config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stable_duration": self.session_config.stable_duration,
                        "removal_duration": self.session_config.removal_duration,
                        "motion_threshold": self.session_config.motion_threshold,
                        "presence_threshold": self.session_config.presence_threshold,
                        "SETTLING_CONFIRMATION_FRAMES": (
                            SETTLING_CONFIRMATION_FRAMES
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self.motion_curve_writer = MotionCurveWriter(
                self.motion_curve_path, interval=MOTION_CURVE_FLUSH_INTERVAL_S
            ).__enter__()
            self._settling_observer = SettlingObserver(
                self.session_config.motion_threshold
            )
        self.work_order_open = self._seed_work_order_open()
        self.has_work_orders = self._seed_has_work_orders()
        self._action_logger = action_logger
        self._lock = RLock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._latest_preview_frame = None
        self.last_error: Exception | None = None
        # Handheld-scanner identity stashed while in slide mode; consumed by
        # the next settled slide capture, then cleared. See `scan_qr`.
        self._pending_slide_payload: str | None = None
        # Display id resolved from the stashed payload for the positive scan
        # indicator on screen 12; falls back to the raw payload if unparseable.
        self._pending_slide_id: str | None = None
        # One normal-mode benchmark follows the operator's specimen through
        # retakes.  It deliberately lives in the Pi runtime: only this
        # process can measure scanner/presence -> final receiver response on
        # one monotonic clock.  The final row is still keyed by the successful
        # slide capture ID, which is supplied by the workflow later.
        self._slide_benchmark_started_at: float | None = None
        self._slide_benchmark_trigger_to_presence_ms: int | None = None
        self._slide_benchmark_retake_count = 0
        self._pending_slide_profile_stage_fields: dict[str, object] | None = None
        phase = self.workflow.snapshot().phase
        self._draining_blocks = phase == "draining_blocks"
        self.mode = CaptureMode.SLIDE if phase == "slides" else CaptureMode.BLOCK
        self._build_controller(self.mode)
        self._apply_saved_light_settings()

    def _seed_work_order_open(self) -> bool:
        """Read the open bracket, blocking startup if the receiver lacks it.

        Gated on session_mode, not `self.open_retrieval`: Hybrid and Hybrid
        Shadow reuse the same work-order bracket, so a restart mid-Hybrid
        session with a bracket open must resume into it rather than show
        first_work_order."""
        if self.session_mode is SessionMode.NORMAL:
            return False
        try:
            return self.workflow.open_work_order_id() is not None
        except StoreError as exc:
            if str(exc) != "unknown method: open_work_order_id":
                raise
            raise RuntimeError(
                "Open Retrieval startup state is unavailable; restart the "
                "processing receiver and retry"
            ) from exc

    def _seed_has_work_orders(self) -> bool:
        """Read history, inferring true only from a positively observed bracket.

        Syncing the Pi cannot restart an already-running processing receiver.
        A receiver started before ``has_work_orders`` was added rejects only
        that new RPC. An already-seeded open bracket proves history exists;
        without that proof, startup blocks until the receiver restarts. Other
        store errors remain fatal.

        Gated on session_mode (like `_seed_work_order_open` and the
        `start_work_order`/`finish_work_order` gates), not
        `self.open_retrieval`: Hybrid/Hybrid Shadow share the same durable
        work-order history as Open Retrieval.
        """
        if self.session_mode is SessionMode.NORMAL:
            return False
        try:
            return self.workflow.has_work_orders()
        except StoreError as exc:
            if str(exc) != "unknown method: has_work_orders":
                raise
            if self.work_order_open:
                return True
            raise RuntimeError(
                "Open Retrieval work-order history is unavailable; restart "
                "the processing receiver and retry"
            ) from exc

    def _log_verb(self, verb: str) -> None:
        if self._action_logger is not None:
            self._action_logger.log("", verb=verb)

    def _log_event(self, event: SessionEvent) -> None:
        if self._action_logger is None:
            return
        if event.kind in ("capture_saved", "capture_error"):
            return
        self._action_logger.log(event.kind, state=event.state.name)

    def _build_controller(self, mode: CaptureMode) -> None:
        if getattr(self, "led_only_blocks", False) and self.illumination:
            if mode is CaptureMode.BLOCK and self.illumination.backlight_available:
                self.illumination.backlight_off()
            self.illumination.led_off()
        self.mode = mode
        self.capture_session = CaptureSession(
            self.session_config,
            mode=mode,
            review_captures=self.review_captures,
            match_baseline_as_present=self.allow_occupied_baseline,
            skip_removal_wait=self.skip_removal_wait,
            led_only=self.led_only_blocks and mode is CaptureMode.BLOCK,
        )
        install_calibrated_baseline = not (
            self.led_only_blocks and mode is CaptureMode.BLOCK
        )
        if mode is CaptureMode.BLOCK:
            pending_blocks = self.workflow.awaiting_capture_blocks()
            if len(pending_blocks) == 1:
                self.capture_session.restore_pending_block(pending_blocks[0])
            elif len(pending_blocks) > 1:
                install_calibrated_baseline = False
                print(
                    "Multiple blocks are awaiting capture; recovery requires "
                    f"operator review: {', '.join(pending_blocks)}"
                )
        elif mode is CaptureMode.SLIDE:
            # Symmetric to the block branch: re-arm durable unreadable-slide
            # recovery (fail-closed reposition / waiting-for-removal) and attach
            # the capture->workflow event bridge. On a fresh slide phase this
            # only wires the listener; after a restart it restores the persisted
            # recovery state so the operator resumes exactly where they left off.
            try:
                self.workflow.restore_slide_capture_session(self.capture_session)
            except RuntimeError:
                pass
        if install_calibrated_baseline:
            try:
                baseline = self.workflow.baseline_for(mode.value)
            except RuntimeError:
                baseline = None
            if baseline is not None:
                self.capture_session.install_locked_baseline(baseline)
        self._push_camera_settings()
        configuration = CaptureConfiguration(
            session=self.session_config,
            role_settings=self.capture_settings.role_settings(),
            # Read live role dims so --camera-profile imx219 retargets snap/stills.
            still_dimensions=CAPTURE_DIMENSIONS["block"],
        )
        self.controller = CaptureController(
            session=self.capture_session,
            camera=self.camera,
            store=CaptureStore(self.capture_root / "published"),
            working_dir=self.capture_root / ".pending",
            configuration=configuration,
            capture_consumer=self._consume_capture,
            illumination=self.illumination,
            triple_capture=self.triple_capture,
        )
        self.capture_session.add_event_listener(self._log_event)
        if self.led_only_blocks and mode is CaptureMode.BLOCK:
            self.capture_session.add_event_listener(self._led_block_event)
        self._attach_capture_logging()

    def _led_block_event(self, event: SessionEvent) -> None:
        if event.kind in (
            "removal_confirmed", "calibration_failed",
        ) or self.capture_session.state is CaptureState.WAITING_FOR_SCAN:
            self.illumination.led_off()

    def _start_led_block_light(self) -> None:
        if self.illumination.backlight_available:
            self.illumination.backlight_off()
        self.illumination.led_on()
        self._led_flush_frames = 4

    def _check_led_block_light(self) -> bool:
        if self.illumination.status().physical_output:
            return True
        self.capture_session.mark_calibration_failed()
        print("LED timed out or is off. Clear the tray and retry calibration.",
              file=sys.stderr)
        return False

    def _sync_preview_exposure(self, variant: str) -> None:
        """Make the live preview match the light currently shown."""
        self.controller.set_preview_variant(variant)

    def _sync_preview_exposure_to_lights(self) -> None:
        """Choose the preview exposure from the current illumination state."""
        if self.illumination is None:
            return
        status = self.illumination.status()
        if status.neutral_oblique_logical_output:
            variant = "neutral_oblique"
        elif status.logical_output:
            variant = "coax"
        elif status.backlight_brightness and status.backlight_brightness > 0:
            variant = "backlight"
        else:
            variant = "coax"
        self._sync_preview_exposure(variant)

    def _attach_capture_logging(self) -> None:
        if self._action_logger is None:
            return
        controller = self.controller
        original_capture = controller._capture

        def _capture_with_log(captured_at: datetime):
            self._action_logger.log(
                "capture_start",
                state=controller.session.state.name,
            )
            # Close the settling window (if any) before capture runs, so the
            # summary is ready in time to ride into this same capture's
            # stage_fields -- last_frame_result is already the
            # capture-requested frame; handle_frame set it just before
            # calling this wrapped _capture.
            settling_summary = None
            if self.profile:
                frame_result = controller.last_frame_result
                if frame_result is not None and frame_result.capture_requested:
                    settling_summary = self._settling_observer.observe(
                        now=self._last_frame_now,
                        state=controller.session.state,
                        frame=frame_result,
                    )
                    self.motion_curve_writer.record(
                        timestamp=captured_at.isoformat(),
                        state=controller.session.state.name,
                        presence_score=frame_result.presence_score,
                        motion_score=frame_result.motion_score,
                        stable_elapsed=frame_result.stable_elapsed,
                        event="capture_requested",
                    )
            start = time.perf_counter()
            result = original_capture(captured_at)
            # Outer elapsed_ms is the wrap total; stage keys come from controller
            # metadata.
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            if result is not None:
                stage_fields = {
                    key: result.metadata[key]
                    for key in CAPTURE_STAGE_TIMING_KEYS
                    if key in result.metadata
                }
                if settling_summary is not None:
                    duration_key, resets_key, max_motion_key = SETTLING_STAGE_KEYS
                    stage_fields[duration_key] = settling_summary.duration_ms
                    stage_fields[resets_key] = settling_summary.resets
                    stage_fields[max_motion_key] = settling_summary.max_motion
                self._action_logger.log(
                    "capture_saved", elapsed_ms=elapsed_ms, **stage_fields,
                    **self._capture_settings_field(result),
                )
                if self.profile:
                    print(
                        render_profile_capture_block(
                            stage_fields, slow_threshold_ms=PROFILE_SLOW_CAPTURE_MS
                        )
                    )
                    self.workflow.store.record_profile_capture(
                        self.workflow.session.number,
                        result.path.stem,
                        stage_fields,
                    )
                    slide_capture_id = result.metadata.get("slide_capture_id")
                    if result.role == "slide" and slide_capture_id:
                        self._record_slide_benchmark(
                            str(slide_capture_id), stage_fields
                        )
                    elif (
                        result.role == "slide"
                        and self.review_captures
                    ):
                        # ACCEPT runs the workflow later. Retain this final
                        # attempted still's Pi stages until that decision
                        # provides its final slide ID.
                        self._pending_slide_profile_stage_fields = dict(stage_fields)
            else:
                failure_meta = controller.last_failure_timings or {}
                stage_fields = {
                    key: failure_meta[key]
                    for key in CAPTURE_STAGE_TIMING_KEYS
                    if key in failure_meta
                }
                self._action_logger.log(
                    "capture_failed", elapsed_ms=elapsed_ms, **stage_fields
                )
            return result

        controller._capture = _capture_with_log

    def _capture_settings_field(self, result) -> dict[str, str]:
        """Compact JSON of the settings this capture's role ran under."""
        # __dict__ lookup: __getattr__ would delegate to the workflow.
        store = self.__dict__.get("capture_settings")
        role = getattr(result, "role", None)
        if store is None or role is None:
            return {}
        values = store.values_for_role(CaptureMode(role).value)
        return {"capture_settings": json.dumps(values, separators=(",", ":"))}

    def _begin_slide_benchmark(self) -> None:
        """Open one normal-mode benchmark at its operator-visible trigger."""
        if not self.profile or self.open_retrieval:
            return
        self._slide_benchmark_started_at = time.monotonic()
        self._slide_benchmark_trigger_to_presence_ms = None
        self._slide_benchmark_retake_count = 0

    def _record_confirmed_slide_presence(self) -> None:
        """Resolve the scanner->presence span, or start a camera-QR row."""
        if not self.profile or self.open_retrieval:
            return
        if self._slide_benchmark_started_at is None:
            # No handheld scanner: the confirmed (two-frame) presence
            # transition itself is the benchmark start.
            self._begin_slide_benchmark()
            self._slide_benchmark_trigger_to_presence_ms = 0
            return
        if self._slide_benchmark_trigger_to_presence_ms is None:
            self._slide_benchmark_trigger_to_presence_ms = max(
                0,
                int(round((time.monotonic() - self._slide_benchmark_started_at) * 1000)),
            )

    def _record_slide_benchmark(
        self, slide_capture_id: str, stage_fields: Mapping[str, object]
    ) -> None:
        """Upsert the complete normal-mode row after its final PC decision.

        Failed/retaken attempts intentionally do not create throwaway rows.
        Their elapsed time remains inside this timer until the eventual final
        capture ID is known, then one durable row is written.
        """
        if not self.profile or self.open_retrieval:
            return
        started_at = self._slide_benchmark_started_at
        full_total_ms = None
        if started_at is not None:
            full_total_ms = max(
                0, int(round((time.monotonic() - started_at) * 1000))
            )
        self.workflow.store.record_slide_benchmark(
            self.workflow.session.number,
            slide_capture_id,
            {
                "trigger_to_presence_ms": self._slide_benchmark_trigger_to_presence_ms,
                "settling_ms": stage_fields.get("settling_duration_ms"),
                "camera_capture_ms": stage_fields.get("camera_capture_ms"),
                "publish_ms": stage_fields.get("publish_ms"),
                "qr_decode_ms": stage_fields.get("consumer_decode_ms"),
                "outbox_write_ms": stage_fields.get("consumer_outbox_ms"),
                "transfer_wait_ms": stage_fields.get("consumer_send_ms"),
                "retake_count": self._slide_benchmark_retake_count,
                "full_total_ms": full_total_ms,
            },
        )
        self._slide_benchmark_started_at = None
        self._slide_benchmark_trigger_to_presence_ms = None
        self._slide_benchmark_retake_count = 0
        self._pending_slide_profile_stage_fields = None

    def _consume_capture(self, record):
        if record.role == CaptureMode.BLOCK.value:
            result = self.workflow.publish_scanned_block(
                record.block_id, record.path, captured_at=record.captured_at,
                profile=self.profile,
            )
            if self.triple_capture:
                self._upload_triple_variants(record)
            return result
        start = time.perf_counter()
        # A handheld-scanner identity, if one was stashed in slide mode, rides
        # into this capture and replaces camera QR decoding entirely; consume it
        # once so it never leaks onto the following slide.
        result = self.workflow.capture_slide(
            record.path,
            captured_at=record.captured_at,
            scanned_payload=self._pending_slide_payload,
            profile=self.profile,
        )
        self._pending_slide_payload = None
        self._pending_slide_id = None
        if self._action_logger is not None and not result.success:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            # reason includes label-rectangle vs QR-readable debug note
            # from decode_slide_identity (see slide_qr._with_label_debug).
            self._action_logger.log(
                "decode_failed",
                elapsed_ms=elapsed_ms,
                reason=result.reason,
            )
        # Carry decode/outbox/send sub-durations alongside the
        # SlideQRResult so CaptureController can forward them into
        # consumer_decode_ms/consumer_outbox_ms/consumer_send_ms without
        # disturbing existing .success/.block_id/.engine readers.
        timings = self.workflow.last_slide_stage_timings
        self._print_hybrid_profile()
        return SlideConsumption(
            result=result,
            decode_ms=timings.decode_ms if timings is not None else None,
            outbox_ms=timings.outbox_ms if timings is not None else None,
            send_ms=timings.send_ms if timings is not None else None,
            capture_id=timings.capture_id if timings is not None else None,
        )

    def _upload_triple_variants(self, record) -> None:
        """Best-effort upload of non-canonical training sidecars.

        The coax image has already followed the normal block outbox path. The
        two additional variants use the existing debug-snap transport so they
        do not create a second session capture or collide with the one-block
        receiver schema. The Pi manifest and PNGs remain the durable source if
        either sidecar upload is interrupted.
        """
        raw_manifest_path = record.metadata.get("triple_capture_manifest")
        if not isinstance(raw_manifest_path, str):
            print("WARNING: triple capture manifest is missing; sidecars kept")
            return
        manifest_path = Path(raw_manifest_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"WARNING: could not read triple capture manifest "
                f"{manifest_path}: {exc}; sidecars kept"
            )
            return
        variants = payload.get("variants", {})
        if not isinstance(variants, Mapping):
            print(
                f"WARNING: triple capture manifest has no variants: "
                f"{manifest_path}; sidecars kept"
            )
            return
        debug_snap = getattr(self.workflow.transport, "debug_snap", None)
        if not callable(debug_snap):
            print("WARNING: debug-snap transport is unavailable; sidecars kept")
            return

        for variant_name in TRIPLE_CAPTURE_VARIANTS[1:]:
            variant = variants.get(variant_name)
            if not isinstance(variant, Mapping):
                print(
                    f"WARNING: triple capture variant {variant_name} is missing; "
                    "sidecars kept"
                )
                continue
            if variant.get("state") == "uploaded":
                continue
            raw_path = variant.get("path")
            if not isinstance(raw_path, str):
                print(
                    f"WARNING: triple capture path for {variant_name} is missing; "
                    "sidecars kept"
                )
                continue
            path = Path(raw_path)
            label = f"block_{record.block_id}_{variant_name}"
            try:
                saved_path = debug_snap(path, label=label, open_image=False)
            except Exception as exc:
                print(
                    f"WARNING: triple capture upload failed for {variant_name}: "
                    f"{exc}; sidecars kept at {path}"
                )
                continue
            variant["state"] = "uploaded"
            variant["remote_path"] = str(saved_path)
            try:
                _atomic_json(manifest_path, payload)
            except OSError as exc:
                print(
                    f"WARNING: triple capture manifest update failed after "
                    f"uploading {variant_name}: {exc}; sidecars kept at {path}"
                )

    def _print_hybrid_profile(self) -> None:
        """Print the Hybrid backend queue/per-slide timings to the console.

        Same output the touchscreen renders (``session.profile_report.
        format_profile_console``), fetched from ``SessionWorkflow.
        hybrid_profile_status`` right after a slide capture reaches the
        outbox. Gated only on ``self.profile`` and Hybrid/Hybrid Shadow mode;
        NORMAL/OPEN_RETRIEVAL ``--profile`` output is unaffected.

        ``hybrid_profile_status`` already degrades internally on a raising
        store, but this runs on the same automatic-capture path
        ``_camera_loop`` owns, so an extra belt-and-suspenders guard keeps
        ANY unexpected failure here from reaching its bare
        ``except Exception`` and killing the camera loop.
        """
        if self.session_mode not in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
            return
        if not self.profile:
            return
        try:
            status = self.workflow.hybrid_profile_status(now_ns=time.time_ns())
            print(
                format_profile_console(
                    status["rows"], queue_count=status["queue_count"]
                )
            )
        except Exception:
            pass

    def start(self, *, background: bool = True) -> None:
        self.controller.start()
        if background:
            self._thread = Thread(target=self._camera_loop, daemon=True)
            self._thread.start()

    def _camera_loop(self) -> None:
        next_drain_poll = time.monotonic()
        try:
            while not self._stop.is_set():
                frame = self.camera.preview_frame()
                self.process_frame(
                    frame,
                    now=time.monotonic(),
                    captured_at=datetime.now(timezone.utc),
                )
                now = time.monotonic()
                if now >= next_drain_poll:
                    self._poll_drain_if_needed()
                    next_drain_poll = now + _DRAIN_POLL_INTERVAL_SECONDS
                time.sleep(1.0 / self.controller.configuration.preview_fps)
        except Exception as exc:
            self.last_error = exc
            if self._action_logger is not None:
                self._action_logger.log("camera_loop_error", error=str(exc))
            self._stop.set()
        finally:
            if self.led_only_blocks and self.illumination:
                self.illumination.led_off()

    def _poll_drain_if_needed(self) -> None:
        """Keep a completed block from stranding the live session in its drain."""
        if not self._draining_blocks:
            return
        try:
            self.poll_drain()
        except (TransportError, StoreError):
            # A dropped processing-computer link (TransportError) must not
            # stop camera capture -- the next background tick resumes the
            # durable drain check. StoreError is caught for the same reason:
            # any store-side rejection surfaces over `/rpc`
            # as HTTP 400 -> StoreError, and this method runs on
            # `_camera_loop`'s background tick, whose own `except Exception`
            # calls `self._stop.set()` -- killing the camera loop entirely
            # over what should be a retryable/ignorable drain-poll hiccup.
            return

    def process_frame(self, frame, *, now: float, captured_at: datetime):
        with self._lock:
            self._latest_preview_frame = frame
            self._last_frame_now = now
            state_before = self.capture_session.state
            if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
                active = state_before in (
                    CaptureState.PLACE_CALIBRATION_BLOCK,
                    CaptureState.REMOVE_CALIBRATION_BLOCK,
                    CaptureState.EMPTY, CaptureState.SETTLING,
                    CaptureState.WAITING_FOR_REMOVAL, CaptureState.CAPTURE_ERROR,
                )
                if active and not self._check_led_block_light():
                    return None
                if active and self._led_flush_frames:
                    self._led_flush_frames -= 1
                    return None
            # While Results is open, the operator must keep seeing live
            # preview (handled above) but no automatic shot may fire behind
            # the table. `self.workflow.capture_paused` is a plain bool set
            # in `SessionWorkflow.__init__` and flipped only by
            # pause_capture/resume_capture, so this read can never raise.
            if self.workflow.capture_paused:
                return None
            result = self.controller.handle_frame(
                frame, now=now, captured_at=captured_at
            )
            if (self.led_only_blocks and self.mode is CaptureMode.BLOCK
                    and self.capture_session.state is CaptureState.AWAITING_ACCEPT):
                self.illumination.led_off()
            if (
                self.mode is CaptureMode.SLIDE
                and state_before is CaptureState.EMPTY
                and self.capture_session.state in (
                    CaptureState.SETTLING,
                    CaptureState.CAPTURE_REQUESTED,
                )
            ):
                # `CaptureSession` reaches either state only after its
                # confirmed-presence transition (normally two present frames).
                self._record_confirmed_slide_presence()
            if self.profile:
                # The capture-requested frame is observed/recorded from
                # inside _capture_with_log instead (so its settling summary
                # is ready in time to ride into that same capture's
                # stage_fields); every other frame is observed here.
                frame_result = self.controller.last_frame_result
                if frame_result is not None and not frame_result.capture_requested:
                    self._settling_observer.observe(
                        now=now,
                        state=self.capture_session.state,
                        frame=frame_result,
                    )
                    self.motion_curve_writer.record(
                        timestamp=captured_at.isoformat(),
                        state=self.capture_session.state.name,
                        presence_score=frame_result.presence_score,
                        motion_score=frame_result.motion_score,
                        stable_elapsed=frame_result.stable_elapsed,
                        event="",
                    )
            return result

    def scan_block(self, block_id: str):
        self._log_verb("scan_block")
        with self._lock:
            outcome = self.workflow.scan_block(block_id)
            if outcome.accepted:
                capture_scan = self.capture_session.submit_scan(block_id)
                if not capture_scan.accepted:
                    self.workflow.unscan_block(block_id)
                    raise RuntimeError(capture_scan.message)
                if self.led_only_blocks:
                    try:
                        self._start_led_block_light()
                    except Exception:
                        self.capture_session.mark_calibration_failed()
                        raise
            return outcome

    def scan_qr(self, payload: str):
        """Single handheld-scanner front door; route by the active mode.

        Block mode registers the scanned 8-digit id immediately (the existing
        `scan_block` path). Slide mode stashes the payload so the next settled
        capture uses it as the identity and skips camera QR decoding; the
        operator scans, then the slide is captured on settle as usual.
        """
        if self.mode is CaptureMode.SLIDE:
            self._log_verb("scan_qr")
            # Scan-time duplicate guard (symmetric to block mode's): reject a
            # re-scan of an already-verdicted slide before stashing, so no
            # capture cycle is spent. The precheck emits duplicate_slide_scan
            # for the kiosk flash.
            if not self.workflow.precheck_slide_scan(payload):
                return None
            with self._lock:
                self._pending_slide_payload = payload
                identity = scanner_identity(payload)
                self._pending_slide_id = (
                    identity.block_id if identity.success else payload
                )
                # A valid scanner payload is the operator-visible start.
                # Presence later resolves `trigger_to_presence_ms` on this
                # same monotonic Pi clock.
                self._begin_slide_benchmark()
            return None
        return self.scan_block(payload)

    def _light_backlight_for_calibration(self) -> None:
        """Light the preview backlight before the baseline is built.

        Presence and motion detection only ever see the preview backlight, so a
        baseline built dark can detect nothing: the session never leaves Place
        Block and never clears Remove Block. Calibrating under the same light
        the detector will use is the whole point of the baseline.
        """
        if self.illumination is None:
            return
        if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
            self._start_led_block_light()
            return
        if not getattr(self.illumination, "backlight_available", False):
            return
        try:
            self.illumination.backlight_on()
        except Exception as exc:  # rig without a working backlight
            print(
                f"Could not light the backlight for calibration: {exc}",
                file=sys.stderr,
            )

    def confirm_empty(self) -> None:
        self._log_verb("confirm_empty")
        # Publish BUILDING_BASELINE under the status lock, then release so
        # /state can route to screen 05 while Camera Calibration runs.
        with self._lock:
            self.capture_session.begin_empty_backlight_setup()
        try:
            self._light_backlight_for_calibration()
            self.workflow.prepare_empty_backlight(self.mode.value)
        except CameraCalibrationError as exc:
            with self._lock:
                self.capture_session.mark_calibration_failed()
            if self._action_logger is not None:
                self._action_logger.log(
                    "calibration_rejected",
                    mode=exc.mode,
                    reason=exc.reason,
                    diagnostics=exc.diagnostics,
                )
            print(
                f"Camera calibration failed: {exc}\nClear the backlight and retry.",
                file=sys.stderr,
            )
            if exc.diagnostics:
                print(
                    f"Calibration diagnostics: {exc.diagnostics}",
                    file=sys.stderr,
                )
            return
        except Exception:
            with self._lock:
                self.capture_session.mark_calibration_failed()
            raise
        with self._lock:
            kwargs = {}
            if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
                kwargs["noise_floor"] = self.camera.block_presence_noise_floor
            self.capture_session.install_locked_baseline(
                self.workflow.baseline_for(self.mode.value), **kwargs
            )

    def retry_capture(self):
        self._log_verb("retry_capture")
        with self._lock:
            was_awaiting_accept = (
                self.capture_session.state is CaptureState.AWAITING_ACCEPT
            )
            if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
                if was_awaiting_accept:
                    self._start_led_block_light()
                elif not self._check_led_block_light():
                    return None
            result = self.controller.retry(captured_at=datetime.now(timezone.utc))
            if (self.led_only_blocks and self.mode is CaptureMode.BLOCK
                    and self.capture_session.state is CaptureState.AWAITING_ACCEPT):
                self.illumination.led_off()
            if (
                self.profile
                and not self.open_retrieval
                and self.mode is CaptureMode.SLIDE
                and self._slide_benchmark_started_at is not None
            ):
                self._slide_benchmark_retake_count += 1
            if was_awaiting_accept and self._action_logger is not None:
                self._action_logger.log("capture_retaken")
            return result

    def accept_capture(self, note: str | None = None):
        self._log_verb("accept_capture")
        with self._lock:
            # The review gate captures/publishes before ACCEPT, but defers the
            # workflow/PC decision until now. Keep the first capture's Pi
            # timings so the final benchmark row includes them.
            held = self.capture_session.last_capture
            pending = self.controller.pending_record
            result = self.controller.accept_capture()
            note = (note or "").strip()
            if note and pending is not None:
                # The capture is already accepted; a lost note must not undo that.
                try:
                    self.workflow.store.add_capture_note(
                        self.workflow.session.number,
                        pending.path.stem,
                        note,
                        pending.block_id,
                    )
                except Exception as exc:
                    print(f"WARNING: capture note not saved on laptop: {exc}")
                if self._action_logger is not None:
                    self._action_logger.log(
                        "capture_note",
                        capture_id=pending.path.stem,
                        note=repr(note),
                    )
            if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
                self._start_led_block_light()
            if (
                self.profile
                and not self.open_retrieval
                and result.role is CaptureMode.SLIDE
            ):
                timings = self.workflow.last_slide_stage_timings
                if timings is not None and timings.capture_id:
                    stage_fields = dict(
                        self._pending_slide_profile_stage_fields or {}
                    )
                    if held is not None:
                        stage_fields.update(held.metadata)
                    stage_fields.update(
                        consumer_decode_ms=timings.decode_ms,
                        consumer_outbox_ms=timings.outbox_ms,
                        consumer_send_ms=timings.send_ms,
                    )
                    self._record_slide_benchmark(timings.capture_id, stage_fields)
            if self._action_logger is not None:
                self._action_logger.log("capture_accepted")
            return result

    def skip_unreadable_slide(self) -> None:
        # 0-arg façade over the workflow method (which needs the capture
        # session); the kiosk `skip_slide` verb dispatches here. Overrides the
        # __getattr__ delegation so callers never pass the capture session in.
        self._log_verb("skip_unreadable_slide")
        with self._lock:
            self.workflow.skip_unreadable_slide(self.capture_session)

    def capture_now(self):
        """Operator override: shoot now instead of waiting for auto-capture.

        Detection stays advisory; the capture itself goes down the normal
        path, so the triple set, block ID, manifest and checksums are
        identical to an automatic capture.
        """
        self._log_verb("capture_now")
        with self._lock:
            if self.led_only_blocks and self.mode is CaptureMode.BLOCK:
                if not self._check_led_block_light():
                    return None
            result = self.controller.capture_now(
                captured_at=datetime.now(timezone.utc)
            )
            if (self.led_only_blocks and self.mode is CaptureMode.BLOCK
                    and self.capture_session.state is CaptureState.AWAITING_ACCEPT):
                self.illumination.led_off()
            return result

    def next_capture(self) -> None:
        """Override removal detection and arm the next capture."""
        self._log_verb("next_capture")
        with self._lock:
            self.capture_session.next_capture()

    def capture_status(self) -> dict[str, str]:
        # This is a lock-free snapshot on purpose.  `process_frame` holds the
        # runtime lock across the camera still and the remote consumer/upload;
        # waiting for that lock here prevented the kiosk from showing
        # CAPTURING until the expensive work had already finished.  Capture
        # state is an enum/reference assignment, and the local session object
        # is captured first so a phase rebuild cannot mix its mode with the
        # replacement session's state.
        capture_session = self.capture_session
        return {
            "capture_state": capture_session.state.name,
            "capture_mode": capture_session.mode.value,
            "led_only_blocks": getattr(self, "led_only_blocks", False),
            # the scanned-but-not-yet-captured block id (screens 07/18);
            # snapshot.latest_block_id only fills at capture time.
            "pending_block_id": capture_session.pending_block_id,
            # the scanned slide id for the screen 12 positive scan
            # indicator; None until a slide is scanned, cleared on capture.
            "pending_slide_id": self._pending_slide_id,
        }

    def pending_capture_status(self) -> dict[str, str] | None:
        """Read-only projection of the held still for the review gate."""
        # Keep the kiosk read path independent of the long-running capture
        # lock.  The controller reference is captured first for the same
        # phase-rebuild consistency as capture_status().
        controller = self.controller
        record = controller.pending_record
        if record is None:
            return None
        return {
            "capture_id": record.path.stem,
            "role": record.role,
            "block_id": record.block_id or "",
        }

    def review_still_jpeg(self) -> bytes | None:
        """Downscaled JPEG of the pending capture still, or None."""
        from kiosk.images import encode_still_jpeg

        controller = self.controller
        record = controller.pending_record
        if record is None:
            return None
        return encode_still_jpeg(record.path)

    def latest_preview_jpeg(self) -> bytes | None:
        """Downscaled JPEG of the latest preview frame, or None."""
        from kiosk.images import encode_preview_jpeg

        # `_latest_preview_frame` is replaced as one reference by the camera
        # loop. Do not wait behind a capture/upload that happens to hold the
        # runtime lock; the preview route is part of the kiosk's live surface.
        frame = self._latest_preview_frame
        if frame is None:
            return None
        return encode_preview_jpeg(frame)

    def _sync_phase(self) -> None:
        phase = self.workflow.snapshot().phase
        desired_mode = {
            "blocks": CaptureMode.BLOCK,
            "slides": CaptureMode.SLIDE,
        }.get(phase)
        if desired_mode is not None and desired_mode is not self.mode:
            with self._lock:
                self._build_controller(desired_mode)
                self.controller.start()

    def finish_blocks(self):
        self._log_verb("finish_blocks")
        with self._lock:
            result = self.workflow.finish_blocks()
            self._draining_blocks = result.phase == "draining_blocks"
            self._sync_phase()
            return result

    def poll_drain(self):
        with self._lock:
            result = self.workflow.poll_drain()
            self._draining_blocks = result.phase == "draining_blocks"
            self._sync_phase()
            return result

    def poll_status(self):
        with self._lock:
            result = self.workflow.poll_status()
            self._sync_phase()
            return result

    def snap(self, note: str | None = None, *, open_image: bool | None = None) -> str:
        """Capture one debug still without changing the light sequence.

        Leaves whatever lights are already on alone. If none are on, turns the
        coaxial LED on as the default lighting for the shot, then leaves it on
        so the operator can keep inspecting. The still never enters the session
        outbox or scoring path. Pi temp is deleted only after upload succeeds.
        """
        self._log_verb("snap")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pending = (
            self.capture_root
            / ".pending"
            / f"debug-snap-{stamp}-{uuid4().hex[:8]}.png"
        )
        self._capture_debug_still(pending)
        try:
            saved = self.workflow.transport.debug_snap(
                pending, label="still", note=note, open_image=open_image
            )
        except Exception as exc:
            kept = str(pending) if pending.exists() else "(missing)"
            raise RuntimeError(
                f"debug snap upload failed; Pi file kept at {kept}"
            ) from exc
        pending.unlink(missing_ok=True)
        if self._action_logger is not None:
            self._action_logger.log(
                "debug_snap_saved", still_path=saved, note=repr(note or "")
            )
        return f"Saved: {saved}"

    def _capture_debug_still(self, pending: Path) -> None:
        """One still under current lights, LED on if none are lit."""
        pending.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                if self.illumination is not None:
                    status = self.illumination.status()
                    # Preview backlight stays lit for motion detection, so it
                    # does not count as an intentional still light. Default to
                    # coax LED unless the LED or neutral light is already on.
                    if (
                        not status.logical_output
                        and not status.neutral_oblique_logical_output
                    ):
                        self.illumination.led_on()
                self.camera.capture_still(
                    pending,
                    settings=self.controller.settings,
                    size=self.controller.configuration.still_dimensions,
                )
            finally:
                self.camera.resume_preview()

    def deskew(
        self, note: str | None = None, *, open_image: bool | None = None
    ) -> str:
        """Debug still like `snap`, plus a copy warped by the saved grid fit.

        The config is loaded before capture so a missing fit costs no still.
        The raw still uploads first, so a resolution mismatch still keeps it.
        """
        self._log_verb("deskew")
        config = load_deskew_config(self.deskew_config_path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = self.capture_root / ".pending" / f"debug-snap-{stamp}-{uuid4().hex[:8]}"
        raw_pending = base.with_name(base.name + "-raw.png")
        warped_pending = base.with_name(base.name + "-deskewed.png")
        self._capture_debug_still(raw_pending)
        try:
            raw_saved = self.workflow.transport.debug_snap(
                raw_pending, label="raw", note=note, open_image=open_image
            )
        except Exception as exc:
            raise RuntimeError(
                f"debug snap upload failed; Pi file kept at {raw_pending}"
            ) from exc
        try:
            try:
                deskew_png(raw_pending, warped_pending, config)
            except DeskewConfigError as exc:
                raise RuntimeError(f"{exc} Raw still saved: {raw_saved}") from exc
            warped_saved = self.workflow.transport.debug_snap(
                warped_pending, label="deskewed", note=note, open_image=open_image
            )
        finally:
            raw_pending.unlink(missing_ok=True)
        warped_pending.unlink(missing_ok=True)
        if self._action_logger is not None:
            self._action_logger.log(
                "debug_deskew_saved", still_path=raw_saved, deskewed_path=warped_saved
            )
        return (
            f"Saved raw: {raw_saved}\n"
            f"Saved deskewed: {warped_saved} ({config.px_per_mm:.1f} px/mm)"
        )

    def snap_3(self, note: str | None = None) -> str:
        """Capture the multi-light debug still set (2 or 3 images).

        Normal sessions capture the transmitted-light and point-LED pair. When
        the optional neutral/oblique output is present and ``--triple-capture``
        is on, also uploads a third neutral/oblique still. These debug stills
        never enter the session outbox or scoring path. Pi temp files are
        deleted only after every requested upload succeeds.
        """
        self._log_verb("snap_3")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pair_id = uuid4().hex[:8]
        backlight_pending = (
            self.capture_root
            / ".pending"
            / f"debug-snap-{stamp}-{pair_id}-backlight.png"
        )
        led_pending = (
            self.capture_root
            / ".pending"
            / f"debug-snap-{stamp}-{pair_id}-point-led.png"
        )
        neutral_pending = (
            self.capture_root
            / ".pending"
            / f"debug-snap-{stamp}-{pair_id}-neutral-oblique.png"
        )
        backlight_pending.parent.mkdir(parents=True, exist_ok=True)

        def settings_for_variant(variant: str):
            """Select calibrated settings, including during rolling upgrades.

            A Pi process can briefly have a newer runner with an older
            ``CaptureController`` loaded from bytecode. Keep snap_3 usable
            during that transition by deriving the variant exposure locally.
            """
            selector = getattr(self.controller, "settings_for_variant", None)
            if callable(selector):
                return selector(variant)
            settings = self.controller.settings
            if self.capture_session.mode is not CaptureMode.BLOCK:
                return settings
            try:
                exposure_us = TRIPLE_CAPTURE_EXPOSURES_US[variant]
            except KeyError as exc:
                raise ValueError(
                    f"unknown triple-capture variant: {variant}"
                ) from exc
            return replace(settings, exposure_us=exposure_us)

        def capture(path: Path, *, variant: str) -> None:
            self.camera.capture_still(
                path,
                settings=settings_for_variant(variant),
                size=self.controller.configuration.still_dimensions,
            )

        pair = bool(
            self.illumination is not None
            and getattr(self.illumination, "backlight_available", False)
            and callable(getattr(self.illumination, "capture_pair", None))
        )
        triple = bool(
            self.triple_capture
            and pair
            and getattr(self.illumination, "neutral_oblique_available", False)
            and callable(getattr(self.illumination, "capture_triple", None))
        )
        with self._lock:
            try:
                if triple:
                    self.illumination.capture_triple(  # type: ignore[union-attr]
                        lambda: capture(led_pending, variant="coax"),
                        lambda: capture(backlight_pending, variant="backlight"),
                        lambda: capture(
                            neutral_pending, variant="neutral_oblique"
                        ),
                        restore_backlight=False,
                    )
                elif pair:
                    self.illumination.capture_pair(  # type: ignore[union-attr]
                        lambda: capture(backlight_pending, variant="backlight"),
                        lambda: capture(led_pending, variant="coax"),
                        restore_backlight=False,
                    )
                else:
                    capture(led_pending, variant="coax")
            finally:
                self.camera.resume_preview()
        try:
            if triple or pair:
                saved_backlight = self.workflow.transport.debug_snap(
                    backlight_pending, label="backlight", note=note
                )
            else:
                saved_backlight = None
            saved_led = self.workflow.transport.debug_snap(
                led_pending, label="point_led", note=note
            )
            if triple:
                saved_neutral = self.workflow.transport.debug_snap(
                    neutral_pending, label="neutral_oblique", note=note
                )
            else:
                saved_neutral = None
        except Exception as exc:
            kept = [
                str(path)
                for path in (backlight_pending, led_pending, neutral_pending)
                if path.exists()
            ]
            raise RuntimeError(
                "debug snap upload failed; Pi files kept at " + ", ".join(kept)
            ) from exc
        backlight_pending.unlink(missing_ok=True)
        led_pending.unlink(missing_ok=True)
        neutral_pending.unlink(missing_ok=True)
        if self._action_logger is not None:
            self._action_logger.log(
                "debug_snap_saved",
                backlight_path=saved_backlight,
                point_led_path=saved_led,
                neutral_oblique_path=saved_neutral,
                note=repr(note or ""),
            )
        if saved_backlight is None:
            return f"Saved point LED: {saved_led}"
        lines = [
            f"Saved backlight: {saved_backlight}",
            f"Saved point LED: {saved_led}",
        ]
        if saved_neutral is not None:
            lines.append(f"Saved neutral/oblique: {saved_neutral}")
        return "\n".join(lines)

    def sample_motion(self, *, clock=time.monotonic, sleep=time.sleep) -> str:
        """Poll `motion_score` from the running background camera loop for
        `MOTION_SAMPLE_WINDOW_S` seconds and report min/mean/max/crossings.

        Reads `self.controller.last_frame_result` -- already produced by the
        background `_camera_loop` / `process_frame` continuous state
        machine -- rather than driving a second parallel capture loop.
        Motion is only meaningful once a baseline is installed; before that
        `motion_score` is `0.0` per `CaptureSession.accept_frame`.
        """
        self._log_verb("motion")
        start = clock()
        scores: list[float] = []
        while clock() - start < MOTION_SAMPLE_WINDOW_S:
            with self._lock:
                result = self.controller.last_frame_result
                scores.append(0.0 if result is None else result.motion_score)
            sleep(0.1)
        sample = summarize_motion_samples(
            scores, threshold=self.session_config.motion_threshold
        )
        return render_motion_sample(
            sample,
            state=self.capture_session.state.name,
            threshold=self.session_config.motion_threshold,
        )

    def focus(
        self,
        seconds=30,
        roi_fraction=0.2,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
        write=None,
    ) -> str:
        """Live center-ROI sharpness under the coax LED for manual lens focus.

        Reads preview frames straight from the camera (stills are too slow for
        a live readout) and does not hold the runtime lock, so the camera loop
        and kiosk keep running.
        """
        self._log_verb("focus")
        seconds = float(seconds)
        if write is None:
            def write(text):
                sys.stdout.write(text)
                sys.stdout.flush()
        if self.illumination is not None:
            # Safety timer outlasts the window; the finally turns it off.
            self.led_on(seconds + 5)
        recent: list[float] = []
        best = 0.0
        previous = None
        try:
            start = clock()
            while clock() - start < seconds:
                frame = self.camera.preview_frame()
                recent = (recent + [center_roi_sharpness(frame, roi_fraction)])[-3:]
                score = sum(recent) / len(recent)
                best = max(best, score)
                if previous is None or abs(score - previous) <= 0.02 * previous:
                    trend = "= steady"
                elif score > previous:
                    trend = "▲ sharper"
                else:
                    trend = "▼ softer"
                previous = score
                write(f"\rsharpness {score:.1f} | best {best:.1f} | {trend}   ")
                sleep(FOCUS_SAMPLE_INTERVAL_S)
        except KeyboardInterrupt:
            pass
        finally:
            if self.illumination is not None:
                self.led_off()
            write("\n")
        return f"Focus done: best sharpness {best:.1f}"

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        errors: list[Exception] = []
        illumination = getattr(self, "illumination", None)
        if illumination is not None:
            try:
                illumination.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self.camera.close()
        except Exception as exc:
            errors.append(exc)
        if getattr(self, "profile", False) and getattr(
            self, "motion_curve_writer", None
        ) is not None:
            try:
                self.motion_curve_writer.__exit__(None, None, None)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def led_auto_on(self):
        if self.illumination is None:
            raise RuntimeError("LED control is unavailable")
        result = self.illumination.led_auto_on()
        self._sync_preview_exposure_to_lights()
        return result

    def led_auto_off(self):
        if self.illumination is None:
            raise RuntimeError("LED control is unavailable")
        result = self.illumination.led_auto_off()
        self._sync_preview_exposure_to_lights()
        return result

    def led_on(self, seconds=None):
        if self.illumination is None:
            raise RuntimeError("LED control is unavailable")
        result = self.illumination.led_on(seconds)
        self._sync_preview_exposure_to_lights()
        return result

    def led_off(self):
        if self.illumination is None:
            raise RuntimeError("LED control is unavailable")
        result = self.illumination.led_off()
        self._sync_preview_exposure_to_lights()
        return result

    def backlight_on(self):
        if self.illumination is None:
            raise RuntimeError("backlight control is unavailable")
        result = self.illumination.backlight_on()
        self._sync_preview_exposure_to_lights()
        return result

    def backlight_off(self):
        if self.illumination is None:
            raise RuntimeError("backlight control is unavailable")
        result = self.illumination.backlight_off()
        self._sync_preview_exposure_to_lights()
        return result

    def set_backlight_level(self, percent):
        if self.illumination is None:
            raise RuntimeError("backlight control is unavailable")
        result = self.illumination.set_backlight_level(percent)
        self._sync_preview_exposure_to_lights()
        return result

    def set_neutral_level(self, percent):
        if self.illumination is None:
            raise RuntimeError("neutral/oblique light control is unavailable")
        result = self.illumination.set_neutral_level(percent)
        self._sync_preview_exposure_to_lights()
        return result

    def neutral_on(self):
        if self.illumination is None:
            raise RuntimeError("neutral/oblique light control is unavailable")
        result = self.illumination.neutral_on()
        self._sync_preview_exposure_to_lights()
        return result

    def neutral_off(self):
        if self.illumination is None:
            raise RuntimeError("neutral/oblique light control is unavailable")
        result = self.illumination.neutral_off()
        self._sync_preview_exposure_to_lights()
        return result

    def led_status(self):
        if self.illumination is None:
            return "LED status: unavailable"
        return self.illumination.status_text()

    def settings_snapshot(self) -> list[dict[str, object]]:
        """Every editable capture setting as JSON-friendly rows."""
        return self.capture_settings.snapshot()

    def set_setting(self, name: str, value: object) -> dict[str, object]:
        """Validate, apply, and persist one capture setting.

        An active-role camera change recalibrates from Scan Block and is
        refused, unsaved, in any state outside `_CAMERA_CHANGE_STATES`.
        """
        store = self.capture_settings
        spec = store.spec(name)
        new_value = store.validate(name, value)
        with self._lock:
            previous = store.value(name)
            recalibrate = False
            if spec.applies == CAMERA:
                recalibrate = self._camera_change_recalibrates(spec.group)
            elif spec.applies == LIGHTS:
                refusal = self._light_refusal(name)
                if refusal is not None:
                    raise RuntimeError(refusal)
                self._apply_light_setting(name, new_value)
            store.set(name, new_value)
            self._apply_live_settings(spec.applies)
        if self._action_logger is not None:
            self._action_logger.log(
                "capture_setting_changed",
                name=name, value=new_value, previous=previous,
            )
        message = f"{name} = {new_value} (was {previous}, default {spec.default})"
        calibration_ok = None
        if recalibrate:
            calibration_ok, note = self._recalibrate_for_settings()
            message += f"; {note}"
        elif spec.applies == CAMERA:
            message += (
                f"; applies at the next {spec.group} calibration"
                if spec.group != self.mode.value
                else "; applies when the table is confirmed empty"
            )
        return {
            "setting": store.describe(name),
            "previous": previous,
            "recalibrated": recalibrate,
            "calibration_ok": calibration_ok,
            "message": message,
        }

    def reset_settings(self) -> dict[str, object]:
        """Restore every default under the same state rules as `set_setting`."""
        store = self.capture_settings
        warnings: list[str] = []
        with self._lock:
            specs = [store.spec(name) for name in store.overrides()]
            recalibrate = any([
                self._camera_change_recalibrates(spec.group)
                for spec in specs
                if spec.applies == CAMERA
            ])
            changed = store.reset()
            for applies in {spec.applies for spec in specs}:
                self._apply_live_settings(applies)
            for spec in specs:
                if spec.applies != LIGHTS or self._light_refusal(spec.name):
                    continue
                try:
                    self._apply_light_setting(spec.name, spec.default)
                except Exception as exc:
                    warnings.append(f"{spec.name} default not applied: {exc}")
        if self._action_logger is not None:
            self._action_logger.log(
                "capture_settings_reset", names=",".join(changed) or "none"
            )
        message = (
            f"Reset to defaults: {', '.join(changed)}" if changed
            else "All settings are already at their defaults"
        )
        message += "".join(f"\nWARNING: {warning}" for warning in warnings)
        calibration_ok = None
        if recalibrate:
            calibration_ok, note = self._recalibrate_for_settings()
            message += f"\n{note}"
        return {
            "reset": changed,
            "recalibrated": recalibrate,
            "calibration_ok": calibration_ok,
            "message": message,
        }

    def settings_status(self) -> dict[str, object]:
        """Settings rows plus whether each may change in the current state."""
        rows = self.capture_settings.snapshot()
        for row in rows:
            if row["applies"] == CAMERA:
                refusal = self._camera_refusal(str(row["group"]))
            elif row["applies"] == LIGHTS:
                refusal = self._light_refusal(str(row["name"]))
            else:
                refusal = None
            row["editable"] = refusal is None
            row["locked_reason"] = refusal
        return {"active_role": self.mode.value, "settings": rows}

    def _camera_refusal(self, role: str) -> str | None:
        """Why a camera change for `role` is unsafe now, if it is."""
        if role != self.mode.value:
            # Another role's settings only matter at that role's calibration.
            return None
        if self.capture_session.state in _CAMERA_CHANGE_STATES:
            return None
        return (
            "camera settings can only change on Scan Block with the table clear"
            if self.mode is CaptureMode.BLOCK
            else "slide camera settings can only change before slide calibration"
        )

    def _camera_change_recalibrates(self, role: str) -> bool:
        """Raise if a camera change for `role` is unsafe now; else say if it recalibrates."""
        refusal = self._camera_refusal(role)
        if refusal is not None:
            raise RuntimeError(refusal)
        return (
            role == self.mode.value
            and self.capture_session.state is CaptureState.WAITING_FOR_SCAN
        )

    def _recalibrate_for_settings(self) -> tuple[bool, str]:
        """Rebuild the baseline under the new camera settings via confirm_empty."""
        self.confirm_empty()
        state = self.capture_session.state
        if state is CaptureState.CALIBRATION_FAILED:
            return False, (
                "recalibrating... failed; change the setting back or clear "
                "the table and retry"
            )
        if state is CaptureState.PLACE_CALIBRATION_BLOCK:
            return True, "recalibrating... place the test block to finish LED setup"
        return True, "recalibrating... done"

    def _light_refusal(self, name: str) -> str | None:
        """Why a light setting makes no sense in this launch mode, if it does not."""
        if (
            name == "lights.backlight_level"
            and self.led_only_blocks
            and self.mode is CaptureMode.BLOCK
        ):
            return "backlight level cannot change in LED-only block mode; it stays off"
        if name == "lights.led_auto" and self.led_only_blocks:
            return "LED auto mode cannot change in LED-only mode; the runtime drives the LED"
        if name == "lights.led_auto" and self.triple_capture:
            return (
                "LED auto mode has no effect under triple capture; "
                "every block capture lights the LED"
            )
        return None

    def _apply_light_setting(self, name: str, value) -> None:
        if name == "lights.backlight_level":
            self.set_backlight_level(value)
        elif name == "lights.neutral_level":
            illumination = self.illumination
            if illumination is None or not getattr(
                illumination, "neutral_oblique_available", False
            ):
                raise RuntimeError("neutral/oblique light control is unavailable")
            if illumination.status().neutral_oblique_logical_output:
                self.set_neutral_level(value)
            else:
                # Remember the level without turning the light on.
                illumination.neutral_oblique_brightness = value / 100.0
        elif value:
            self.led_auto_on()
        else:
            self.led_auto_off()

    def _apply_saved_light_settings(self) -> None:
        """Re-apply saved light overrides at startup; failures only warn."""
        for name in self.capture_settings.overrides():
            if self.capture_settings.spec(name).applies != LIGHTS:
                continue
            try:
                refusal = self._light_refusal(name)
                if refusal is not None:
                    raise RuntimeError(refusal)
                self._apply_light_setting(name, self.capture_settings.value(name))
            except Exception as exc:
                print(f"WARNING: saved {name} not applied: {exc}", file=sys.stderr)

    def _apply_live_settings(self, applies: str) -> None:
        store = self.capture_settings
        if applies == DETECTION:
            self.session_config = store.session_config(self.session_config)
            self.capture_session.config = store.session_config(
                self.capture_session.config
            )
            if self._settling_observer is not None:
                self._settling_observer.threshold = self.session_config.motion_threshold
        elif applies == CAMERA:
            self._push_camera_settings()
            self.controller.settings = store.camera_settings(self.mode)
        else:
            return
        self.controller.configuration = replace(
            self.controller.configuration,
            session=self.session_config,
            role_settings=store.role_settings(),
        )

    def _push_camera_settings(self) -> None:
        # The Pi adapter reads these in activate_mode; fakes simply hold them.
        self.camera.role_settings = self.capture_settings.role_settings()

    def start_work_order(self):
        """Open a work order; rejected in SessionMode.NORMAL.

        Closed-set per-slide verification is the default. The work-order
        lifecycle only unlocks for OPEN_RETRIEVAL/HYBRID/HYBRID_SHADOW; in
        NORMAL this overrides the blanket ``__getattr__`` delegation with a
        hard reject so normal-mode capture is unchanged."""
        if self.session_mode is SessionMode.NORMAL:
            raise RuntimeError(
                "start_work_order requires --open-retrieval"
            )
        with self._lock:
            result = self.workflow.start_work_order()
            self.work_order_open = True
            self.has_work_orders = True
            self._sync_phase()
            return result

    def end_session(self, *, confirm: bool):
        """Flush + upload profile artifacts before finalizing.

        Mirrors `finish_blocks`/`start_work_order`'s explicit overrides of the
        blanket `__getattr__` proxy: profile-off sessions never created a
        motion curve, so they delegate straight through unchanged.
        """
        if not confirm:
            return self.workflow.end_session(confirm=False)
        self._shutdown_illumination()
        if self.profile:
            self.motion_curve_writer.flush()
            self.workflow.transport.upload_profile_curve(
                self.workflow.session.number, self.motion_curve_path
            )
            self.workflow.transport.upload_profile_config(
                self.workflow.session.number, self.profile_config_path
            )
        return self.workflow.end_session(confirm=confirm)

    def _shutdown_illumination(self) -> None:
        """Turn both lights off when the logical session is ending."""
        illumination = getattr(self, "illumination", None)
        if illumination is None:
            return
        shutdown = getattr(illumination, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return

        # Compatibility for older injected controllers during a rolling update.
        errors: list[Exception] = []
        for name in ("led_off", "backlight_off", "neutral_off"):
            operation = getattr(illumination, name, None)
            if not callable(operation):
                continue
            try:
                operation()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def finish_work_order(self):
        # Same NORMAL-only gate as start_work_order above.
        if self.session_mode is SessionMode.NORMAL:
            raise RuntimeError(
                "finish_work_order requires --open-retrieval"
            )
        with self._lock:
            if (
                self.mode is not CaptureMode.SLIDE
                or self.capture_session.state is not CaptureState.EMPTY
            ):
                raise RuntimeError(
                    "finish_work_order requires idle slide capture"
                )
            if self._pending_slide_payload is not None:
                raise RuntimeError(
                    "capture the scanned slide before finishing the work order"
                )
            result = self.workflow.finish_work_order()
            self.work_order_open = False
            return result

    def __getattr__(self, name: str):
        return getattr(self.workflow, name)


def _render_result(command: str, result: object) -> str:
    """Render one command's result the way an operator wants to read it."""
    if command == "summary" and isinstance(result, SessionSummary):
        return render_summary(result, debug=True)
    if command == "events" and isinstance(result, tuple):
        return render_events(tuple(event for event in result if isinstance(event, WorkflowEvent)))
    if command == "results" and isinstance(result, Mapping):
        return render_results(result)
    if command == "settings" and isinstance(result, list):
        return render_settings(result)
    if command in ("set", "reset_settings") and isinstance(result, Mapping):
        return str(result["message"])
    if isinstance(result, tuple):
        return "\n".join(str(item) for item in result) if result else "(none)"
    if result is None:
        return ""
    return str(result)


def run_one_command(workflow: SessionWorkflow, line: str) -> str:
    """Pure operator-loop core: parse one line, dispatch it, render the result.

    Never touches stdin/stdout and never raises `StoreError`/`TransportError`
    out to the caller -- both render as a clean one-line message instead, so
    the interactive loop below can stay a thin, untested wrapper around this.
    """
    line = line.strip()
    if not line:
        return ""
    command, *args = line.split()
    if command in ("help", "?") or command not in COMMAND_HELP:
        return command_cheat_sheet()
    if command == "scan_block" and len(args) != 1:
        return command_cheat_sheet()
    try:
        result = dispatch(workflow, command, *args)
    except StoreError as exc:
        return f"Rejected: {exc}"
    except TransportError as exc:
        return f"Connectivity problem, still running -- {exc}"
    except (RuntimeError, CaptureSettingError) as exc:
        return f"Rejected: {exc}"
    except ValueError:
        return command_cheat_sheet()
    return _render_result(command, result)


def _request_shutdown(_signum=None, _frame=None) -> None:
    """Raise SystemExit so ``main``'s ``finally`` can close the camera.

    Closing the SSH window delivers SIGHUP; the launcher sends SIGTERM. CPython's
    default for those signals kills the process without running ``finally``,
    which leaves Picamera2 holding the device ("Pipeline handler in use").
    """
    raise SystemExit(0)


def install_camera_release_signals() -> tuple[str, ...]:
    """Install SIGHUP/SIGTERM handlers that exit through Python cleanup.

    Returns the signal names that were installed (empty on platforms that lack
    them, e.g. Windows has no SIGHUP). Safe to call more than once.
    """
    installed: list[str] = []
    for name in ("SIGHUP", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        signal.signal(sig, _request_shutdown)
        installed.append(name)
    return tuple(installed)


def _interactive_loop(workflow: SessionWorkflow) -> None:
    """Thin wrapper: read a line, print what `run_one_command` renders."""
    print(command_cheat_sheet())
    while True:
        try:
            line = input("pi> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        rendered = run_one_command(workflow, line)
        if rendered:
            print(rendered)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--led-only", "--led-only-blocks", dest="led_only_blocks", action="store_true",
        help="Use the coax LED on a dark tray for block setup and placement; "
        "requires a test-block placement/removal check. Slides use backlight.",
    )
    parser.add_argument(
        "--receiver-url",
        required=True,
        help="Processing computer's base URL, printed by tools/run_receiver.py "
        "(e.g. http://192.168.50.1:8077).",
    )
    parser.add_argument(
        "--capture-root",
        default=str(_DEFAULT_PI_LOCAL_DIR / "captures"),
        help="Pi-LOCAL durable camera publication directory. Default: %(default)s",
    )
    parser.add_argument(
        "--session",
        type=int,
        required=True,
        metavar="N",
        help="Session number printed by tools/run_receiver.py to resume.",
    )
    parser.add_argument(
        "--calibration-path",
        default=str(_DEFAULT_PI_LOCAL_DIR / "framing_calibration.json"),
        help="Pi-LOCAL path for FramingCalibrationStore. Never a path on the "
        "processing computer -- this is exactly the state ADR 0002 keeps off "
        "the wire. Default: %(default)s",
    )
    parser.add_argument(
        "--deskew-config",
        default=str(_DEFAULT_DESKEW_CONFIG_PATH),
        help="Pi-LOCAL grid-deskew fit JSON used by the deskew verb. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--outbox-root",
        default=str(_DEFAULT_PI_LOCAL_DIR / "outbox"),
        help="Pi-LOCAL durable outbox directory for PiOutbox. Default: %(default)s",
    )
    parser.add_argument(
        "--kiosk",
        action="store_true",
        help="Also serve the Pi-local kiosk touchscreen UI (ADR 0004) beside "
        "the text console; point Chromium --kiosk at the printed URL. The "
        "console keeps working -- both renderers share one workflow.",
    )
    parser.add_argument(
        "--kiosk-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="Localhost port for the kiosk UI. Default: %(default)s",
    )
    parser.add_argument(
        "--review-captures",
        action="store_true",
        help="Pause on each capture for operator ACCEPT/RETAKE before upload "
        "(ADR 0006 debug review gate). Default: off.",
    )
    parser.add_argument(
        "--triple-capture",
        action="store_true",
        help="For block captures, save coax + backlight + neutral/oblique "
        "training images. Coax remains the canonical verifier image; "
        "requires the PWM BCM 22 MOSFET light. Default: off.",
    )
    parser.add_argument(
        OPEN_RETRIEVAL_FLAG,
        action="store_true",
        help="Enable open-retrieval mode (work-order-scoped n2 retrieval, "
        "issue #147-#151). Default: off (normal verification mode). "
        f"Mutually exclusive with {HYBRID_FLAG}/{HYBRID_SHADOW_FLAG}.",
    )
    parser.add_argument(
        HYBRID_FLAG,
        action="store_true",
        help="Enable Hybrid session scoring mode (#247 mode plumbing only "
        "-- boots into the existing capture flow and scores exactly as "
        "normal verification does; the queue/candidate-pool/reranking "
        "arrive in later slices). Default: off. Mutually exclusive with "
        f"{OPEN_RETRIEVAL_FLAG}/{HYBRID_SHADOW_FLAG}.",
    )
    parser.add_argument(
        HYBRID_SHADOW_FLAG,
        action="store_true",
        help="Enable Hybrid Shadow session scoring mode (#247 mode "
        "plumbing only; this flag's exact name may still be revised per "
        "issue #245). Default: off. Mutually exclusive with "
        f"{OPEN_RETRIEVAL_FLAG}/{HYBRID_FLAG}.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-stage capture timings and persist capture plus block "
        "readiness benchmarks in the session bundle. Default: off.",
    )
    parser.add_argument(
        "--allow-occupied-baseline",
        action="store_true",
        help="Debug: skip empty-tray occupancy reject and lock presence to "
        "whatever is on the tray so START SESSION and Remove Block work "
        "with a cassette already present. Default: off.",
    )
    parser.add_argument(
        "--skip-removal-wait",
        action="store_true",
        help="Blocks only: after a block capture go straight back to Scan "
        "Block instead of waiting for removal detection. The next scan is "
        "the gate between blocks. Slides are unaffected. Default: off.",
    )
    parser.add_argument(
        "--camera-profile",
        default="imx477",
        metavar="NAME",
        help="Still size profile: imx477 (HQ production default) or imx219 "
        "(Camera Module v2.1 smoke; aliases v2.1 / v21). Retargets capture "
        "dimensions only; exposures/gains/CV stay HQ-tuned. Default: %(default)s.",
    )
    parser.add_argument(
        "--hybrid-config",
        default=str(_DEFAULT_HYBRID_CONFIG_PATH),
        help="Pi-LOCAL path to the #249 versioned Hybrid integration handoff. "
        f"Validated before any store/camera side effect, only when {HYBRID_FLAG} "
        f"or {HYBRID_SHADOW_FLAG} is set -- normal verification and "
        f"{OPEN_RETRIEVAL_FLAG} never read this path. Default: %(default)s",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.led_only_blocks and (
        args.triple_capture or args.skip_removal_wait or args.allow_occupied_baseline
    ):
        parser.error(
            "--led-only cannot combine with --triple-capture, "
            "--skip-removal-wait, or --allow-occupied-baseline"
        )

    try:
        camera_profile = apply_camera_profile(args.camera_profile)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Reject any two-or-more scoring-mode flag combination loudly here,
    # before any store/camera/workflow side effect runs.
    try:
        session_mode = resolve_session_mode(
            open_retrieval=args.open_retrieval,
            hybrid=args.hybrid,
            hybrid_shadow=args.hybrid_shadow,
        )
    except SessionModeConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # A missing, malformed, stale, or version-incompatible Hybrid
    # Configuration must block --hybrid/--hybrid-shadow startup loudly,
    # before any store/camera/workflow side effect. NORMAL/OPEN_RETRIEVAL
    # never read the handoff file.
    # `ProcessingStore.freeze_hybrid_pool` builds fingerprints from the
    # descriptor recipe named here. Only the names cross the wire, so the
    # processing computer never needs its own copy of the Pi-local file.
    hybrid_descriptor_names: tuple[str, ...] = ()
    hybrid_candidate_configuration: dict[str, object] | None = None
    if session_mode in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
        try:
            hybrid_config = load_hybrid_configuration(args.hybrid_config)
        except HybridConfigurationError as exc:
            print(
                f"Hybrid configuration invalid; refusing to start "
                f"{_MODE_LABELS[session_mode]}: {exc}",
                file=sys.stderr,
            )
            return 2
        hybrid_descriptor_names = tuple(
            entry.name for entry in hybrid_config.descriptor_recipe
        )
        hybrid_candidate_configuration = {
            "architecture_kind": hybrid_config.architecture_kind,
            "architecture_name": hybrid_config.architecture_name,
            "architecture_methods": list(hybrid_config.architecture_methods),
            "candidate_band_thresholds": dict(hybrid_config.candidate_band_thresholds),
        }

    store = RemoteProcessingStore(args.receiver_url)
    transport = HttpCaptureClient(args.receiver_url)
    outbox = PiOutbox(args.outbox_root)
    framing = FramingCalibrationStore(args.calibration_path)

    try:
        session = store.resume_session(args.session)
    except (StoreError, TransportError) as exc:
        print(
            f"Could not resume session {args.session} at {args.receiver_url}: {exc}",
            file=sys.stderr,
        )
        return 2

    # The Pi's resolved SessionMode and the session's durable
    # `sessions.session_mode` column must never silently disagree, or a
    # Hybrid work order could fall through to the full N-by-N scoring path.
    # Hard-fail before the camera opens or `SessionWorkflow` is built. The
    # durable column is deliberately NOT auto-corrected: that would rewrite
    # the scoring mode of a session whose data may already be captured
    # under the other mode. `getattr(..., None)` skips the check for test
    # doubles that lack `session_mode`; a real `SessionIdentity` always
    # has it.
    durable_mode = getattr(session, "session_mode", None)
    if durable_mode is not None and durable_mode != session_mode.value:
        print(
            f"Refusing to start: session {args.session}'s durable "
            f"sessions.session_mode is {durable_mode!r}, but this Pi "
            f"resolved {session_mode.value!r} from its own launch flags. "
            "Starting anyway would let this session's scoring silently "
            "disagree with what its durable mode says (#269). Pass launch "
            "flags that match the session's durable mode, or omit them to "
            "resume in that mode as-is.",
            file=sys.stderr,
        )
        return 2

    # Delayed import keeps laptops/tests free of the Pi-only picamera2 package.
    from picamera2_adapter import Picamera2Adapter

    try:
        camera = Picamera2Adapter()
        camera.led_only_blocks = args.led_only_blocks
    except RuntimeError as exc:
        print(
            f"Could not open the camera: {exc}\n"
            "Is another run_pi_session.py already running? "
            "(ps aux | grep run_pi_session)",
            file=sys.stderr,
        )
        return 2

    try:
        if args.led_only_blocks:
            illumination = _build_illumination(
                include_neutral_oblique=True, led_only_blocks=True,
            )
        elif args.triple_capture:
            illumination = _build_illumination(include_neutral_oblique=True)
        else:
            illumination = _build_illumination()
    except RuntimeError as exc:
        _close_quietly(camera)
        print(f"Could not open the coaxial LED GPIO: {exc}", file=sys.stderr)
        return 2

    try:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=outbox,
            transport=transport,
            camera=camera,
            framing_calibration=framing,
            session_mode=session_mode,
            hybrid_descriptor_names=hybrid_descriptor_names,
            hybrid_candidate_configuration=hybrid_candidate_configuration,
        )
    except Exception as exc:
        _close_quietly(illumination)
        _close_quietly(camera)
        print(
            f"Could not start session workflow: {exc}",
            file=sys.stderr,
        )
        return 2

    action_log_path = _DEFAULT_PI_LOCAL_DIR / f"workflow_actions_session_{session.number}.log"
    try:
        action_logger = ActionLogger(action_log_path, session_number=session.number)
        runtime = PiCaptureRuntime(
            workflow,
            camera,
            capture_root=args.capture_root,
            action_logger=action_logger,
            review_captures=args.review_captures,
            allow_occupied_baseline=args.allow_occupied_baseline,
            open_retrieval=args.open_retrieval,
            hybrid=args.hybrid,
            hybrid_shadow=args.hybrid_shadow,
            profile=args.profile,
            illumination=illumination,
            **({"triple_capture": True} if args.triple_capture else {}),
            skip_removal_wait=args.skip_removal_wait,
            # Only forward an override, like led_only_blocks, so launch fakes stay valid.
            **(
                {"deskew_config_path": args.deskew_config}
                if args.deskew_config != str(_DEFAULT_DESKEW_CONFIG_PATH)
                else {}
            ),
            **({"led_only_blocks": True} if args.led_only_blocks else {}),
            capture_settings_path=_DEFAULT_CAPTURE_SETTINGS_PATH,
        )
    except Exception as exc:
        _close_quietly(illumination)
        _close_quietly(camera)
        print(f"Could not construct Pi capture runtime: {exc}", file=sys.stderr)
        return 2

    print(f"Connected to {args.receiver_url} -- session {session.number}")
    mode_label = _MODE_LABELS.get(session_mode, "normal verification")
    print(f"Mode: {mode_label}")
    print(
        f"Camera profile: {camera_profile.name} "
        f"(block {camera_profile.native_dimensions[0]}x"
        f"{camera_profile.native_dimensions[1]}, "
        f"slide {camera_profile.slide_dimensions[0]}x"
        f"{camera_profile.slide_dimensions[1]})"
    )
    if camera_profile.smoke_only:
        print(
            "WARNING: smoke camera profile; exposures/colour gains/CV remain "
            "HQ-tuned and are not qualified on this sensor.",
            file=sys.stderr,
        )
    if args.allow_occupied_baseline:
        print(
            "WARNING: --allow-occupied-baseline is on; occupancy is skipped "
            "and presence matches the locked occupied baseline.",
            file=sys.stderr,
        )
    if args.skip_removal_wait:
        print(
            "--skip-removal-wait is on; block captures return straight to "
            "Scan Block without waiting for removal detection."
        )
    if args.triple_capture:
        print(
            "Triple block capture: coax + backlight + neutral/oblique "
            f"(BCM {ILLUMINATION_GPIO_PIN}/{BACKLIGHT_GPIO_PIN}/"
            f"{NEUTRAL_OBLIQUE_GPIO_PIN}); coax remains canonical."
        )
    print(f"Workflow action log: {action_log_path}")
    try:
        runtime.start()
        # SIGHUP (SSH window closed) / SIGTERM (launcher kill) must run finally.
        install_camera_release_signals()
    except Exception as exc:
        _close_quietly(runtime)
        print(f"Could not start Pi capture runtime: {exc}", file=sys.stderr)
        return 2

    # Optional second renderer (ADR 0004). It shares the SAME runtime handle
    # -- constructor injection, no second source of truth -- and reads/writes
    # only through the same observe API + dispatch verbs the console uses, so
    # the interactive loop below is unaffected.
    kiosk_server: KioskServer | None = None
    try:
        if args.kiosk:
            kiosk_server = KioskServer(KioskRelay(runtime), port=args.kiosk_port)
            kiosk_server.start()
            print(f"Kiosk UI at {kiosk_server.url} -- point Chromium --kiosk here")
    except Exception as exc:
        _stop_quietly(kiosk_server)
        _close_quietly(runtime)
        print(f"Could not start kiosk UI: {exc}", file=sys.stderr)
        return 2

    try:
        _interactive_loop(runtime)
    except BaseException:
        # Preserve KeyboardInterrupt/SystemExit and any operator-loop failure;
        # cleanup is best effort and must not replace the original exception.
        _stop_quietly(kiosk_server)
        _close_quietly(runtime)
        raise
    cleanup_error = _close_runtime_and_kiosk(runtime, kiosk_server)
    if cleanup_error is not None:
        raise cleanup_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
