"""Off-hardware coverage for the Pi-side session runtime (issue #116).

Real camera + a physical two-machine cable-pull remain ready-for-human; this
module covers everything that lands and tests OFF hardware: the
framing-calibration/`store.root` seam fix in `SessionWorkflow.__init__`, the
operator command cheat-sheet in `session_console`, and the
`tools/run_pi_session.py` entry point's operator-loop core driven against a
REAL `LoopbackCaptureReceiver` + `ProcessingStore` over loopback with the
headless camera (mirrors `tests/test_remote_store.py`).

What this does NOT prove (left to hardware QA, per ADR 0002 / issue #116):
- A real camera feed (live preview, stillness detection, per-phase baselines).
- An actual physical cable pull / disconnect on the dedicated Ethernet link.
- Running `tools/run_pi_session.py` on real Pi hardware against
  `tools/run_receiver.py` on a separate physical processing computer.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from camera_calibration import (
    ActivatedCameraMode,
    CalibrationQuality,
    CameraCalibrationError,
    LockedCameraControls,
    PhaseCameraCalibration,
)
from capture_session import CaptureState
from session.session_mode import SessionMode

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    # Append (never insert at the front): conftest.py already put
    # `tools/manifest` etc. on sys.path as flat modules, and `tools` itself
    # contains a `manifest/` *package* that would otherwise shadow those.
    sys.path.append(str(_TOOLS_DIR))

from store.remote import RemoteProcessingStore
from session.console import COMMAND_HELP, _COMMANDS, command_cheat_sheet
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionIdentity,
    SessionWorkflow,
)
from slide.qr import DecodeCandidate, select_slide_identity

import constants
import run_pi_session
import session.workflow as workflow_module


STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


class _FakeRealCamera:
    def __init__(self):
        self.started = 0
        self.closed = 0

    def start_preview(self, **_kwargs):
        self.started += 1

    def preview_frame(self):
        return np.full((80, 120, 3), 180, dtype=np.uint8)

    def capture_still(self, path, **_kwargs):
        assert cv2.imwrite(
            str(path), np.full((3040, 4056), 100, dtype=np.uint8)
        )

    def resume_preview(self):
        pass

    def close(self):
        self.closed += 1


class _FakePhaseCamera:
    """Phase activation camera for SessionWorkflow (not the still/preview camera)."""

    def __init__(self):
        self.actions: list[tuple[str, str]] = []

    def activate_mode(self, mode: str) -> ActivatedCameraMode:
        self.actions.append(("activate", mode))
        calibration = PhaseCameraCalibration(
            mode=mode,
            controls=LockedCameraControls(1, 1.0, (1.0, 1.0)),
            quality=CalibrationQuality(
                True, 1, 0, 0.0, 0.0, 0.0, 0.0, 220.0, 0.0, 0.0
            ),
            metadata_samples=(),
        )
        baseline = np.full((480, 640, 3), 220, dtype=np.uint8)
        return ActivatedCameraMode(calibration, baseline)


class _FailThenOkPhaseCamera(_FakePhaseCamera):
    """Fails the first activate_mode with CameraCalibrationError, then succeeds."""

    def activate_mode(self, mode: str) -> ActivatedCameraMode:
        if not self.actions:
            self.actions.append(("activate", mode))
            raise CameraCalibrationError(
                mode,
                "capture area occupied",
                {"chromatic_fraction": 0.5},
            )
        return super().activate_mode(mode)


# --------------------------------------------------------------------------
# Change 1: framing calibration / store.root seam
# --------------------------------------------------------------------------


class _RemoteShapedStoreWithoutRoot:
    """Stands in for RemoteProcessingStore's shape: no `.root` attribute.

    The ctor path under test raises before touching any other store method
    when framing_calibration is omitted, so this fake need not implement
    anything else.
    """


def test_local_store_with_no_framing_calibration_still_uses_store_root(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:1"),
    )

    assert workflow.framing_calibration.path == store.root / "framing_calibration.json"


def test_rootless_store_with_no_framing_calibration_raises_clear_value_error(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with pytest.raises(ValueError, match="remote store requires an explicit"):
        SessionWorkflow(
            session=session,
            store=_RemoteShapedStoreWithoutRoot(),
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient("http://127.0.0.1:1"),
        )


def test_remote_shaped_store_with_injected_framing_calibration_constructs(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    framing = FramingCalibrationStore(tmp_path / "pi_local" / "framing_calibration.json")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        assert not hasattr(proxy, "root")

        workflow = SessionWorkflow(
            session=session,
            store=proxy,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=framing,
        )

    assert workflow.framing_calibration is framing


# --------------------------------------------------------------------------
# Change 2: operator command cheat-sheet
# --------------------------------------------------------------------------


def test_every_command_has_a_help_entry():
    assert set(COMMAND_HELP) == set(_COMMANDS)


def test_cheat_sheet_lists_every_command_name():
    sheet = command_cheat_sheet()

    for name in _COMMANDS:
        assert name in sheet


def test_cheat_sheet_includes_arg_hint_for_scan_block():
    sheet = command_cheat_sheet()

    assert "scan_block <id>" in sheet


def test_cheat_sheet_includes_arg_hint_for_scan_qr():
    sheet = command_cheat_sheet()

    assert "scan_qr <payload>" in sheet


# --------------------------------------------------------------------------
# Change 3/4: run_pi_session.py operator-loop core, over a real loopback wire
# --------------------------------------------------------------------------


def _build_workflow(tmp_path, store, session, receiver_url, *, camera=None):
    return SessionWorkflow(
        session=session,
        store=RemoteProcessingStore(receiver_url),
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient(receiver_url),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
        camera=camera,
    )


def test_run_one_command_summary_renders_a_session_summary(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        rendered = run_pi_session.run_one_command(workflow, "summary")

    assert f"Session {session.number}" in rendered
    assert "PASS" in rendered


def test_run_one_command_help_and_unknown_command_return_cheat_sheet(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        help_rendered = run_pi_session.run_one_command(workflow, "help")
        unknown_rendered = run_pi_session.run_one_command(workflow, "delete_everything")

    sheet = command_cheat_sheet()
    assert help_rendered == sheet
    assert unknown_rendered == sheet


def test_run_one_command_blank_line_is_a_no_op(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        assert run_pi_session.run_one_command(workflow, "   ") == ""


def test_run_one_command_bare_scan_block_shows_usage_without_calling_workflow():
    class TrackingWorkflow:
        called = False

        def scan_block(self, block_id):
            self.called = True

    workflow = TrackingWorkflow()

    rendered = run_pi_session.run_one_command(workflow, "scan_block")

    assert rendered == command_cheat_sheet()
    assert workflow.called is False


def test_run_one_command_extra_args_on_no_arg_command_show_usage():
    class TrackingWorkflow:
        called = False

        def summary(self):
            self.called = True

    workflow = TrackingWorkflow()
    assert run_pi_session.run_one_command(workflow, "summary typo") == command_cheat_sheet()
    assert workflow.called is False


def test_run_one_command_scan_block_round_trips_over_the_wire(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "51151378" in rendered
    assert "Accepted" in rendered


def test_run_one_command_store_error_renders_clean_message_without_raising(tmp_path):
    """A store-side protocol rejection (StoreError, HTTP 400 -- e.g. a session
    the server has never heard of) must render as a clean message and never
    raise out of the loop. Ordinary domain rejections (bad block id, phase
    closed) are NOT StoreErrors -- they come back as ordinary ScanOutcome
    values -- so this forces a genuine protocol-level rejection by pointing
    an already-constructed workflow's session at a number the server never
    started."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        workflow.session = SessionIdentity(999999, STARTED_AT, tmp_path)

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "unknown session" in rendered.lower()


def test_run_one_command_transport_error_renders_connectivity_note(tmp_path):
    """A transport-level failure (connection refused -- never reached any
    server) must render as a connectivity note and never raise out of the
    loop. Construct the workflow against a live receiver first (the ctor
    itself makes store calls), then swap in a dead endpoint so only the
    command under test fails at the transport level."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        workflow.store = RemoteProcessingStore(
            "http://127.0.0.1:1", max_attempts=1, backoff=0
        )

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "connectiv" in rendered.lower() or "connection" in rendered.lower()


def test_finalization_is_reachable_end_to_end_over_the_wire_for_an_empty_session(tmp_path):
    """Drive finish_blocks -> end_session -> poll_finalization through the
    operator-loop core against the real server. With zero blocks scanned,
    finalization has nothing to wait on and completes in one pass, proving
    finalization-over-the-wire without staging a full block/slide capture."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        run_pi_session.run_one_command(workflow, "finish_blocks")
        ended = run_pi_session.run_one_command(workflow, "end_session")
        polled = run_pi_session.run_one_command(workflow, "poll_finalization")

    assert "finalized" in ended.lower()
    assert "finalized" in polled.lower()
    assert store.snapshot(session).phase == "finalized"


def test_entry_point_help_smoke():
    with pytest.raises(SystemExit) as exc_info:
        run_pi_session.main(["--help"])
    assert exc_info.value.code == 0


def test_entry_point_requires_receiver_url_and_session():
    with pytest.raises(SystemExit):
        run_pi_session.main([])


def test_session_workflow_ctor_survives_failing_phase_camera(tmp_path):
    """After Task 3 deferral, ctor must not activate the camera; failure surfaces
    only when confirm_empty runs prepare_empty_backlight."""

    class _AlwaysFailCamera:
        def activate_mode(self, mode):
            raise CameraCalibrationError(
                mode, "capture area occupied", {"chromatic_fraction": 1.0}
            )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_AlwaysFailCamera()
        )

    assert workflow is not None


def test_confirm_empty_runs_full_empty_backlight_setup_fail_then_succeed(
    tmp_path, capsys,
):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    phase_camera = _FailThenOkPhaseCamera()
    logged = []

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=phase_camera
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
            action_logger=SimpleNamespace(
                log=lambda event, **fields: logged.append((event, fields))
            ),
        )

        assert (
            runtime.capture_status()["capture_state"]
            == CaptureState.AWAITING_BASELINE_CONFIRMATION.name
        )

        runtime.confirm_empty()
        assert (
            runtime.capture_status()["capture_state"]
            == CaptureState.CALIBRATION_FAILED.name
        )
        err = capsys.readouterr().err
        assert "Camera calibration failed" in err
        assert "capture area occupied" in err
        assert "chromatic_fraction" in err
        assert (
            "calibration_rejected",
            {
                "mode": "block",
                "reason": "capture area occupied",
                "diagnostics": {"chromatic_fraction": 0.5},
            },
        ) in logged

        runtime.confirm_empty()
        assert (
            runtime.capture_status()["capture_state"]
            == CaptureState.WAITING_FOR_SCAN.name
        )
        assert phase_camera.actions[-1] == ("activate", "block")


def test_confirm_empty_exposes_building_state_while_activate_runs(tmp_path):
    """Status lock must not hide BUILDING_BASELINE during Camera Calibration."""

    class _BlockingPhaseCamera(_FakePhaseCamera):
        def __init__(self):
            super().__init__()
            self.entered = Event()
            self.release = Event()

        def activate_mode(self, mode: str) -> ActivatedCameraMode:
            self.entered.set()
            assert self.release.wait(timeout=2.0)
            return super().activate_mode(mode)

    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    phase_camera = _BlockingPhaseCamera()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=phase_camera
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
        )

        errors: list[BaseException] = []

        def _run():
            try:
                runtime.confirm_empty()
            except BaseException as exc:  # noqa: BLE001 — collect for main thread
                errors.append(exc)

        worker = Thread(target=_run)
        worker.start()
        assert phase_camera.entered.wait(timeout=2.0)
        assert (
            runtime.capture_status()["capture_state"]
            == CaptureState.BUILDING_BASELINE.name
        )
        phase_camera.release.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert errors == []
        assert (
            runtime.capture_status()["capture_state"]
            == CaptureState.WAITING_FOR_SCAN.name
        )


def test_background_driver_advances_drain_after_block_preprocessing_completes(
    tmp_path, monkeypatch,
):
    """A block finishing after FINISH BLOCKS must enter slide mode unaided."""

    class _BlockingPreprocessor:
        def __init__(self):
            self.entered = Event()
            self.release = Event()

        def __call__(self, _path):
            self.entered.set()
            assert self.release.wait(timeout=2.0)
            return np.full((3040, 4056), 255, dtype=np.uint8), {"method": "test"}

    preprocessor = _BlockingPreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    monkeypatch.setattr(run_pi_session, "_DRAIN_POLL_INTERVAL_SECONDS", 0.0)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
        )
        drain_advanced_to_slides = Event()
        original_poll_drain = runtime.poll_drain

        def tracked_poll_drain():
            result = original_poll_drain()
            if preprocessor.release.is_set() and result.phase == "slides":
                drain_advanced_to_slides.set()
            return result

        runtime.poll_drain = tracked_poll_drain
        runtime.start()
        try:
            source = tmp_path / "block.png"
            assert cv2.imwrite(
                str(source), np.full((3040, 4056, 3), 100, dtype=np.uint8)
            )
            workflow.capture_block("51151378", source, captured_at=STARTED_AT)
            assert preprocessor.entered.wait(timeout=2.0)

            assert runtime.finish_blocks().phase == "draining_blocks"
            preprocessor.release.set()
            store.wait_for_jobs()

            assert drain_advanced_to_slides.wait(timeout=2.0), runtime.last_error
            assert workflow.snapshot().phase == "slides"
        finally:
            runtime.close()


def test_camera_loop_failure_is_written_to_the_operator_log():
    class _BrokenCamera:
        def preview_frame(self):
            raise RuntimeError("preview processing exploded")

    logged = []
    runtime = run_pi_session.PiCaptureRuntime.__new__(
        run_pi_session.PiCaptureRuntime
    )
    runtime.camera = _BrokenCamera()
    runtime.controller = SimpleNamespace(
        configuration=SimpleNamespace(preview_fps=10.0)
    )
    runtime._stop = Event()
    runtime.last_error = None
    runtime._action_logger = SimpleNamespace(
        log=lambda event, **fields: logged.append((event, fields))
    )

    runtime._camera_loop()

    assert isinstance(runtime.last_error, RuntimeError)
    assert logged == [
        ("camera_loop_error", {"error": "preview processing exploded"})
    ]


def _build_bare_runtime(*, capture_paused: bool):
    """A `PiCaptureRuntime` with just enough stand-ins for `process_frame`
    (#257): no real camera, capture store, or session -- only the attributes
    `process_frame` itself touches."""
    handled = []
    runtime = run_pi_session.PiCaptureRuntime.__new__(
        run_pi_session.PiCaptureRuntime
    )
    runtime._lock = RLock()
    runtime._latest_preview_frame = None
    runtime._last_frame_now = 0.0
    runtime.capture_session = SimpleNamespace(state=CaptureState.EMPTY)
    runtime.mode = run_pi_session.CaptureMode.BLOCK
    runtime.profile = False
    runtime.workflow = SimpleNamespace(capture_paused=capture_paused)
    runtime.controller = SimpleNamespace(
        handle_frame=lambda frame, *, now, captured_at: handled.append(
            (frame, now, captured_at)
        )
        or "handled"
    )
    return runtime, handled


def test_process_frame_paused_skips_capture_advance_but_updates_preview():
    runtime, handled = _build_bare_runtime(capture_paused=True)
    frame = np.full((80, 120, 3), 180, dtype=np.uint8)

    result = runtime.process_frame(frame, now=5.0, captured_at=STARTED_AT)

    assert result is None
    assert handled == []
    assert runtime._latest_preview_frame is frame
    assert runtime._last_frame_now == 5.0


def test_process_frame_not_paused_advances_capture_as_before():
    runtime, handled = _build_bare_runtime(capture_paused=False)
    frame = np.full((80, 120, 3), 180, dtype=np.uint8)

    result = runtime.process_frame(frame, now=5.0, captured_at=STARTED_AT)

    assert result == "handled"
    assert handled == [(frame, 5.0, STARTED_AT)]
    assert runtime._latest_preview_frame is frame
    assert runtime._last_frame_now == 5.0


def test_pi_capture_runtime_drives_controller_and_publishes_block(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    camera = _FakeRealCamera()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            camera,
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
        )
        runtime.start(background=False)
        assert runtime.capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
        runtime.confirm_empty()
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
        empty = np.full((80, 120, 3), 180, dtype=np.uint8)
        runtime.process_frame(empty, now=0.0, captured_at=STARTED_AT)
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
        assert runtime.scan_block("51151378").accepted
        specimen = empty.copy()
        cv2.rectangle(specimen, (30, 20), (90, 60), (40, 40, 40), -1)

        runtime.process_frame(specimen, now=1.0, captured_at=STARTED_AT)
        runtime.process_frame(specimen, now=2.0, captured_at=STARTED_AT)
        runtime.process_frame(specimen, now=3.0, captured_at=STARTED_AT)
        runtime.close()

    store.wait_for_jobs()
    row = store.get_set(session.number, "51151378")
    assert row["capture_id"] is not None
    assert Path(row["capture_path"]).is_file()
    assert camera.started == 1
    assert camera.closed == 1


def test_pi_capture_runtime_restores_one_accepted_block_without_rescan(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
        )
        assert runtime.capture_session.pending_block_id == "51151378"
        assert (
            runtime.capture_session.state
            is CaptureState.AWAITING_BASELINE_CONFIRMATION
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        assert runtime.capture_session.state is CaptureState.EMPTY
        empty = np.full((80, 120, 3), 180, dtype=np.uint8)
        specimen = empty.copy()
        cv2.rectangle(specimen, (30, 20), (90, 60), (40, 40, 40), -1)
        runtime.process_frame(empty, now=0.0, captured_at=STARTED_AT)
        runtime.process_frame(specimen, now=1.0, captured_at=STARTED_AT)
        runtime.process_frame(specimen, now=2.0, captured_at=STARTED_AT)
        runtime.process_frame(specimen, now=3.0, captured_at=STARTED_AT)
        runtime.close()

    assert store.get_set(session.number, "51151378")["capture_id"] is not None


def test_pi_capture_runtime_restores_fail_closed_slide_recovery_after_restart(tmp_path):
    """A Pi restart during the slides phase must re-arm fail-closed unreadable-
    slide recovery, mirroring the block-side restore. Regression for the missing
    SLIDE branch in ``_build_controller``: only ``restore_pending_block`` was
    wired, so ``restore_slide_capture_session`` was never called on hardware and
    a restart mid-slides came up armed instead of fail-closed reposition.
    """
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    # Enter the slides phase, then persist an unreadable slide so the durable
    # recovery state is "reposition" (the fail-closed state a restart must honor).
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()
    unreadable = tmp_path / "unreadable.png"
    assert cv2.imwrite(str(unreadable), np.full((80, 120, 3), 180, dtype=np.uint8))
    store.record_slide_capture(
        session.number,
        unreadable,
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=1500.0,
    )
    assert store.slide_recovery_state(session.number) == "reposition"

    # A fresh runtime is the post-restart process: it must come up re-armed.
    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )

    assert runtime.capture_session.mode is run_pi_session.CaptureMode.SLIDE
    assert runtime.capture_session.unreadable_slide_can_be_skipped


def _build_local_workflow(
    tmp_path, store, session, *, session_mode=None, hybrid_descriptor_names=(),
):
    """A ``SessionWorkflow`` over a LOCAL ``ProcessingStore`` (no
    ``LoopbackCaptureReceiver``/``RemoteProcessingStore`` hop) -- the
    ``start_work_order``/``finish_work_order`` RPC verbs are not yet wired
    across the Pi/processing-computer boundary (absent from ``_RPC_METHODS``
    in ``session_workflow.py``), so exercising the gating seam under test
    goes directly at the store the same way ``test_session_workflow.py``'s
    own work-order lifecycle tests do."""
    kwargs = {}
    if session_mode is not None:
        kwargs["session_mode"] = session_mode
        kwargs["hybrid_descriptor_names"] = hybrid_descriptor_names
    return SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:0"),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
        **kwargs,
    )


def test_pi_capture_runtime_rejects_start_work_order_without_open_retrieval_flag(
    tmp_path,
):
    """#153: closed-set per-slide verification is the default. Without
    ``--open-retrieval`` (``open_retrieval=False``), the work-order lifecycle
    verbs ``session_console._COMMANDS`` dispatches must be REJECTED -- not
    silently delegated to ``self.workflow`` via ``PiCaptureRuntime.
    __getattr__`` -- so normal-mode capture stays byte-for-byte unchanged."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
    )

    assert runtime.open_retrieval is False
    with pytest.raises(RuntimeError, match="open.retrieval"):
        runtime.start_work_order()
    with pytest.raises(RuntimeError, match="open.retrieval"):
        runtime.finish_work_order()


def test_pi_capture_runtime_start_and_finish_work_order_when_open_retrieval_enabled(
    tmp_path,
):
    """With ``--open-retrieval`` set, the work-order lifecycle verbs unlock
    and delegate through to the real ``SessionWorkflow``, exactly matching
    the zero-arg call shape ``session_console._COMMANDS`` dispatches
    (``lambda workflow, args: workflow.start_work_order()``)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        open_retrieval=True,
    )

    assert runtime.open_retrieval is True
    work_order_id = runtime.start_work_order()
    assert store.get_work_order(session.number, work_order_id)[
        "lifecycle_state"
    ] == "capturing"

    runtime.finish_blocks()
    runtime.confirm_empty()
    finished_id = runtime.finish_work_order()
    assert finished_id == work_order_id


def _fake_fingerprint_builder(mask):
    from verify.invariant_descriptors import DescriptorValue
    return {"fake_v1": DescriptorValue(vector=np.array([1.0]), construction_ns=1)}


def _fake_score_cache_builder(specimen):
    from verify.scorer import LockedScoreCache, _ComponentFeatures
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def _direct_capture_block(store, session_number, block_id, value):
    """Feed a block straight through ``store.scan_block``/``receive_capture``,
    bypassing ``PiOutbox``/``HttpCaptureClient`` entirely -- mirrors
    ``tests/test_hybrid_pool_freeze.py``'s helper of the same shape.
    ``_build_local_workflow``'s transport points at an unreachable loopback
    port (see its docstring), so driving captures through
    ``workflow.capture_block`` here would silently queue in the outbox
    forever instead of reaching the (local, same-process) store."""
    import hashlib
    assert store.scan_block(session_number, block_id).accepted
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    body = png.tobytes()
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


def _patch_lightweight_qc(monkeypatch):
    """The real QC panel renderer assumes the mask matches the capture's
    dimensions; these tests' small synthetic masks deliberately do not
    (mirrors tests/test_hybrid_pool_freeze.py's fixture of the same
    purpose)."""
    def write_qc(capture, mask, destination):
        assert cv2.imwrite(str(destination), np.zeros((4, 4, 3), dtype=np.uint8))

    def write_failure_qc(capture, reason, destination):
        assert cv2.imwrite(str(destination), np.zeros((4, 4, 3), dtype=np.uint8))

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


def test_pi_capture_runtime_start_and_finish_work_order_when_hybrid_enabled(
    tmp_path, monkeypatch,
):
    """#269: the work-order lifecycle verbs unlock for HYBRID too -- the
    whole point of this issue -- while ``runtime.open_retrieval`` stays
    False under Hybrid (pinned by ``tests/test_hybrid_launch.py``). The gate
    now keys on ``session_mode``, not the narrow ``open_retrieval`` flag."""
    _patch_lightweight_qc(monkeypatch)
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}
        ),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    workflow = _build_local_workflow(
        tmp_path, store, session, session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        hybrid=True,
    )

    assert runtime.session_mode is SessionMode.HYBRID
    assert runtime.open_retrieval is False
    work_order_id = runtime.start_work_order()
    assert store.get_work_order(session.number, work_order_id)[
        "lifecycle_state"
    ] == "capturing"

    _direct_capture_block(store, session.number, "11111111", 10)
    _direct_capture_block(store, session.number, "22222222", 20)
    store.wait_for_jobs()

    finished_snapshot = runtime.finish_blocks()
    assert finished_snapshot.phase == "slides"
    runtime.confirm_empty()
    finished_id = runtime.finish_work_order()
    assert finished_id == work_order_id
    # Hybrid never runs the N-by-N path: the bracket closes to 'finalized'
    # and stays there (#251/#252's queue/scoring is what would eventually
    # advance it), never 'scoring'.
    assert store.get_work_order(session.number, work_order_id)[
        "lifecycle_state"
    ] == "finalized"


# --------------------------------------------------------------------------
# #258: --profile Hybrid queue/timing instrumentation. Job 1 threads the
# existing --profile CLI flag through slide capture exactly like it already
# threads through block capture; the console print mirrors the existing
# per-capture "Profile:" block's `if self.profile:` shape.
# --------------------------------------------------------------------------


def _hybrid_profile_setup(tmp_path, monkeypatch, *, profile):
    """A Hybrid session with 2 usable blocks frozen into slide mode, then one
    slide capture -- #258 Job 1's proof rig. Mirrors ``test_pi_capture_
    runtime_start_and_finish_work_order_when_hybrid_enabled``'s block setup,
    with one profiled/unprofiled slide capture added at the end."""
    _patch_lightweight_qc(monkeypatch)
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}
        ),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    workflow = _build_local_workflow(
        tmp_path, store, session, session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        hybrid=True,
        profile=profile,
    )
    runtime.start_work_order()
    _direct_capture_block(store, session.number, "11111111", 10)
    _direct_capture_block(store, session.number, "22222222", 20)
    store.wait_for_jobs()
    runtime.finish_blocks()
    runtime.confirm_empty()
    assert runtime.mode is run_pi_session.CaptureMode.SLIDE

    # #258: the slide's claimed block id MUST be one of the two frozen
    # Hybrid Candidate Pool blocks above -- `record_slide_capture`'s
    # `profile` gate only ever collects timing for an accepted, in-pool
    # Hybrid claim (see its own docstring), so an unmatched block id would
    # decode fine but never pick up profiling at all.
    runtime.scan_qr("12080_11111111_01_HE")
    slide_png = tmp_path / "profiled_slide.png"
    assert cv2.imwrite(str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8))
    runtime._consume_capture(_FakeSlideRecord(slide_png))
    return store, session, runtime


@pytest.mark.parametrize("profile, expected_rows", [(True, 1), (False, 0)])
def test_hybrid_slide_capture_threads_the_profile_flag_to_the_store(
    tmp_path, monkeypatch, profile, expected_rows,
):
    """#258 Job 1's proof: --profile must thread through slide capture
    exactly like it already threads through block capture
    (``publish_scanned_block``'s own ``profile`` param) -- proven by whether
    ``ProcessingStore.list_hybrid_profile_rows`` (whose WHERE clause is
    ``profile_enabled=1``) picks up this slide at all, not by inspecting
    outbox/RPC internals directly."""
    store, session, _runtime = _hybrid_profile_setup(
        tmp_path, monkeypatch, profile=profile
    )

    rows = store.list_hybrid_profile_rows(session.number)

    assert len(rows) == expected_rows


def test_hybrid_profile_prints_to_the_console_with_no_debug_flag(
    tmp_path, monkeypatch, capsys,
):
    """#258 criteria 5/6: the console prints the SAME queue/timing numbers
    the touchscreen would render (``session.profile_report.format_profile_
    console``), driven only by ``--profile``. There is no debug flag
    anywhere on ``PiCaptureRuntime`` -- this test passing (rather than the
    print being silently swallowed by ``_print_hybrid_profile``'s own
    belt-and-suspenders ``except Exception``) is the non-vacuous proof that
    profiling never depended on one."""
    _hybrid_profile_setup(tmp_path, monkeypatch, profile=True)

    printed = capsys.readouterr().out

    assert "Hybrid profile: queue=" in printed
    assert "11111111" in printed


def test_hybrid_profile_prints_nothing_without_dash_dash_profile(
    tmp_path, monkeypatch, capsys,
):
    """Control for the test above (#258 criterion 3, console half): without
    --profile, the automatic capture path must print no queue/stage/timing
    text at all."""
    _hybrid_profile_setup(tmp_path, monkeypatch, profile=False)

    printed = capsys.readouterr().out

    assert "Hybrid profile" not in printed


def test_hybrid_profile_console_print_stays_off_for_normal_mode_even_with_profile(
    tmp_path, capsys,
):
    """Non-vacuous control (#258 criterion 9): must fail if the Hybrid/
    Hybrid Shadow mode gate in ``_print_hybrid_profile`` is dropped --
    ``--profile`` is on here precisely so "no Hybrid profile text" is proven
    by the mode gate, not merely by ``--profile`` being off. NORMAL/Open
    Retrieval's existing per-capture "Profile:" stage block is untouched by
    this new gate."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        profile=True,
    )
    assert runtime.session_mode is SessionMode.NORMAL

    runtime._print_hybrid_profile()

    assert "Hybrid profile" not in capsys.readouterr().out


def test_pi_capture_runtime_multiple_work_orders_in_one_hybrid_session_freeze_independent_pools(
    tmp_path, monkeypatch,
):
    """#269 acceptance criterion: one session can contain multiple sequential
    Hybrid work orders, each with its own open/close bracket driven through
    ``PiCaptureRuntime`` (the same start_work_order/finish_blocks/
    finish_work_order verbs Open Retrieval already uses), and each freezes
    its own independent Hybrid Candidate Pool."""
    _patch_lightweight_qc(monkeypatch)
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}
        ),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    workflow = _build_local_workflow(
        tmp_path, store, session, session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        hybrid=True,
    )

    work_order_a = runtime.start_work_order()
    _direct_capture_block(store, session.number, "11111111", 10)
    _direct_capture_block(store, session.number, "22222222", 20)
    store.wait_for_jobs()
    assert runtime.finish_blocks().phase == "slides"
    runtime.confirm_empty()
    runtime.finish_work_order()

    work_order_b = runtime.start_work_order()
    assert work_order_b != work_order_a
    _direct_capture_block(store, session.number, "33333333", 30)
    _direct_capture_block(store, session.number, "44444444", 40)
    store.wait_for_jobs()
    # No second confirm_empty(): calibration is per block/slide-phase
    # transition, not per work order -- work order A already calibrated the
    # slide-phase baseline this session, so B's blocks->slides transition
    # reuses it and lands straight in CaptureState.EMPTY (not
    # AWAITING_BASELINE_CONFIRMATION).
    assert runtime.finish_blocks().phase == "slides"
    runtime.finish_work_order()

    pool_a = store.hybrid_pool(work_order_a)
    pool_b = store.hybrid_pool(work_order_b)
    assert pool_a is not None and pool_b is not None
    assert set(pool_a.block_ids) == {"11111111", "22222222"}
    assert set(pool_b.block_ids) == {"33333333", "44444444"}
    assert set(pool_a.block_ids).isdisjoint(pool_b.block_ids)


# --------------------------------------------------------------------------
# `motion` console command (#169): PiCaptureRuntime.sample_motion
# --------------------------------------------------------------------------


def test_motion_sample_uses_MOTION_SAMPLE_WINDOW_S_constant():
    """The sample window is a single named constant defined once in the
    central `code/constants.py` module and imported (not re-literaled) by
    `tools/run_pi_session.py` -- mirrors `PROFILE_SLOW_CAPTURE_MS` (#168)."""
    assert hasattr(constants, "MOTION_SAMPLE_WINDOW_S")
    assert run_pi_session.MOTION_SAMPLE_WINDOW_S is constants.MOTION_SAMPLE_WINDOW_S
    assert constants.MOTION_SAMPLE_WINDOW_S > 0


def test_pi_capture_runtime_sample_motion_collects_for_the_configured_window(
    tmp_path, monkeypatch
):
    """`sample_motion` polls `self.controller.last_frame_result.motion_score`
    on an injectable clock/sleep until `MOTION_SAMPLE_WINDOW_S` elapses (no
    real threads/sleeping): `start = clock()`; each iteration samples once
    then calls `sleep(...)` before re-checking `clock() - start` against the
    window. A fake clock that only advances inside `sleep` makes the loop
    count deterministic without depending on any internal step size."""
    monkeypatch.setattr(run_pi_session, "MOTION_SAMPLE_WINDOW_S", 0.3)
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(
            baseline_frames=1, stable_duration=0.0
        ),
    )
    runtime.start(background=False)
    runtime.confirm_empty()
    assert runtime.scan_block("51151378").accepted
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    runtime.process_frame(empty, now=0.0, captured_at=STARTED_AT)

    clock_state = {"now": 0.0}
    sleep_calls = []

    def fake_clock():
        return clock_state["now"]

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        clock_state["now"] += 0.1

    rendered = runtime.sample_motion(clock=fake_clock, sleep=fake_sleep)

    assert isinstance(rendered, str)
    # window=0.3, step=0.1 -> exactly 3 sampled iterations before stopping.
    assert len(sleep_calls) == 3


# --------------------------------------------------------------------------
# #155: boot-seed / toggle of PiCaptureRuntime.work_order_open
# --------------------------------------------------------------------------


class _CountingTransport:
    """Wraps a real transport; counts POSTs and records each RPC method name
    (mirrors the fake in ``test_remote_store.py``) -- ``snapshot`` goes over
    GET, so this isolates the RPC round trips a boot-seed would add from the
    unavoidable ``snapshot()`` call every ``PiCaptureRuntime.__init__``
    already makes. Construction also issues unrelated RPCs (e.g.
    ``awaiting_capture_blocks``/``baseline_for`` via ``_build_controller``),
    so callers should assert on specific method names, not a raw count."""

    def __init__(self, inner):
        self.inner = inner
        self.post_calls = 0
        self.methods_called = []

    def post(self, url, payload):
        self.post_calls += 1
        try:
            method = json.loads(payload)["method"]
        except (ValueError, KeyError, TypeError):
            method = None
        self.methods_called.append(method)
        return self.inner.post(url, payload)

    def post_binary(self, url, payload):
        return self.inner.post_binary(url, payload)

    def get(self, url):
        return self.inner.get(url)


def test_pi_capture_runtime_seeds_work_order_open_true_after_restart_mid_bracket(
    tmp_path,
):
    """#155: a restart while a work order is mid-bracket (capturing) must come
    up showing FINISH, not START -- seeded once at construction time from
    ``workflow.open_work_order_id()``, gated on ``--open-retrieval``."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            open_retrieval=True,
        )

    assert runtime.work_order_open is True
    assert runtime.has_work_orders is True


def test_pi_capture_runtime_seeds_work_order_open_false_when_none_open(tmp_path):
    """A fresh session seeds no open bracket and no work-order history."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            open_retrieval=True,
        )

    assert runtime.work_order_open is False
    assert runtime.has_work_orders is False


@pytest.mark.parametrize("mode_kwarg", ["hybrid", "hybrid_shadow"])
def test_pi_capture_runtime_seeds_work_order_open_true_for_hybrid_after_restart(
    tmp_path, mode_kwarg,
):
    """#269: before this fix, ``_seed_work_order_open``/``_seed_has_work_orders``
    gated on ``self.open_retrieval`` alone, which is deliberately False for
    Hybrid/Hybrid Shadow (``tests/test_hybrid_launch.py`` pins that -- see
    ``test_pi_capture_runtime_hybrid_flag_resolves_hybrid_mode``). A restart
    mid-Hybrid-session with a bracket already open would then seed False and
    the kiosk would show first_work_order over an already-open bracket.
    Mirrors ``test_pi_capture_runtime_seeds_work_order_open_true_after_restart_mid_bracket``
    above, but for Hybrid/Hybrid Shadow instead of Open Retrieval."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT, session_mode=mode_kwarg)
    store.start_work_order(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            **{mode_kwarg: True},
        )

    assert runtime.work_order_open is True
    assert runtime.has_work_orders is True
    # #247 pin unchanged: open_retrieval stays narrowly OPEN_RETRIEVAL-only.
    assert runtime.open_retrieval is False


def test_pi_runtime_requires_receiver_restart_without_has_work_orders_rpc(
    tmp_path, monkeypatch,
):
    """A fresh session is not enough evidence to guess no durable history."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "has_work_orders")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        with pytest.raises(RuntimeError, match="restart the processing receiver"):
            run_pi_session.PiCaptureRuntime(
                workflow,
                _FakeRealCamera(),
                capture_root=tmp_path / "pi_captures",
                session_config=run_pi_session.SessionConfig(baseline_frames=1),
                open_retrieval=True,
            )


def test_pi_runtime_requires_receiver_restart_without_open_work_order_id_rpc(
    tmp_path, monkeypatch,
):
    """The client must not guess that an unqueryable bracket is closed."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "open_work_order_id")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        with pytest.raises(RuntimeError, match="restart the processing receiver"):
            run_pi_session.PiCaptureRuntime(
                workflow,
                _FakeRealCamera(),
                capture_root=tmp_path / "pi_captures",
                session_config=run_pi_session.SessionConfig(baseline_frames=1),
                open_retrieval=True,
            )


def test_pi_runtime_does_not_hide_open_bracket_when_receiver_lacks_seed_rpc(
    tmp_path, monkeypatch,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "open_work_order_id")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        with pytest.raises(RuntimeError, match="restart the processing receiver"):
            run_pi_session.PiCaptureRuntime(
                workflow,
                _FakeRealCamera(),
                capture_root=tmp_path / "pi_captures",
                session_config=run_pi_session.SessionConfig(baseline_frames=1),
                open_retrieval=True,
            )


def test_pi_runtime_does_not_hide_completed_history_when_receiver_lacks_history_rpc(
    tmp_path, monkeypatch,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    store.finish_work_order(session.number, start_job=False)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "has_work_orders")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        with pytest.raises(RuntimeError, match="restart the processing receiver"):
            run_pi_session.PiCaptureRuntime(
                workflow,
                _FakeRealCamera(),
                capture_root=tmp_path / "pi_captures",
                session_config=run_pi_session.SessionConfig(baseline_frames=1),
                open_retrieval=True,
            )


def test_pi_runtime_open_bracket_safely_implies_history_when_history_rpc_is_missing(
    tmp_path, monkeypatch,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "has_work_orders")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            open_retrieval=True,
        )

    assert runtime.work_order_open is True
    assert runtime.has_work_orders is True


def test_pi_capture_runtime_toggles_work_order_open_on_start_and_finish(tmp_path):
    """The open flag toggles while durable history stays true after start."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            open_retrieval=True,
        )
        assert runtime.work_order_open is False

        runtime.start_work_order()
        assert runtime.work_order_open is True
        assert runtime.has_work_orders is True

        runtime.finish_blocks()
        runtime.confirm_empty()
        runtime.finish_work_order()
        assert runtime.work_order_open is False
        assert runtime.has_work_orders is True


def test_starting_the_next_work_order_rebuilds_the_block_capture_controller(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        open_retrieval=True,
    )

    runtime.start_work_order()
    runtime.finish_blocks()
    assert runtime.mode is run_pi_session.CaptureMode.SLIDE
    runtime.confirm_empty()
    runtime.finish_work_order()

    runtime.start_work_order()

    assert workflow.snapshot().phase == "blocks"
    assert runtime.mode is run_pi_session.CaptureMode.BLOCK


def test_finish_work_order_rejects_a_scanned_slide_that_has_not_captured(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        open_retrieval=True,
    )
    work_order_id = runtime.start_work_order()
    runtime.finish_blocks()
    runtime.confirm_empty()
    runtime.scan_qr("12080_51137181_01_HE")

    with pytest.raises(RuntimeError, match="capture the scanned slide"):
        runtime.finish_work_order()

    assert runtime.work_order_open is True
    assert store.open_work_order_id(session.number) == work_order_id


def test_pi_capture_runtime_never_seeds_work_order_open_without_open_retrieval_flag(
    tmp_path,
):
    """Normal mode (no ``--open-retrieval``) must stay byte-for-byte
    unchanged: ``work_order_open`` stays False and construction never issues
    the seed RPC (no extra round trip beyond the existing ``snapshot()``
    call every runtime already makes)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        from session.workflow import HttpCaptureClient
        from store.remote import RemoteProcessingStore, UrlTransport

        transport = _CountingTransport(UrlTransport())
        proxy = RemoteProcessingStore(receiver.url, transport=transport)
        workflow = SessionWorkflow(
            session=session,
            store=proxy,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "pi_local" / "framing_calibration.json"
            ),
            camera=_FakePhaseCamera(),
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )

    assert runtime.open_retrieval is False
    assert runtime.work_order_open is False
    assert runtime.has_work_orders is False
    assert "open_work_order_id" not in transport.methods_called
    assert "has_work_orders" not in transport.methods_called


class _FakeSlideRecord:
    """Minimal capture record: `_consume_capture`'s slide branch reads only
    role/path/captured_at."""

    role = "slide"

    def __init__(self, path):
        self.path = path
        self.captured_at = STARTED_AT


def test_scan_qr_in_block_phase_registers_the_block_like_scan_block(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        empty = np.full((80, 120, 3), 180, dtype=np.uint8)
        runtime.process_frame(empty, now=0.0, captured_at=STARTED_AT)

        outcome = runtime.scan_qr("51151378")

    assert outcome.accepted
    assert runtime.capture_session.pending_block_id == "51151378"


def test_scan_qr_in_slide_phase_feeds_scanned_payload_to_next_capture(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    # An empty session drains immediately, entering the slides phase.
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        seen = {}

        def _spy_capture_slide(source, *, captured_at, scanned_payload=None, profile=False):
            seen["scanned_payload"] = scanned_payload
            return SimpleNamespace(success=True)

        runtime.workflow.capture_slide = _spy_capture_slide

        runtime.scan_qr("12080_51137181_01_HE")
        assert runtime._pending_slide_payload == "12080_51137181_01_HE"

        slide_png = tmp_path / "settled_slide.png"
        assert cv2.imwrite(str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8))
        runtime._consume_capture(_FakeSlideRecord(slide_png))

    # The stashed payload rode into the capture and was cleared for the next one.
    assert seen["scanned_payload"] == "12080_51137181_01_HE"
    assert runtime._pending_slide_payload is None


def test_scan_qr_in_slide_phase_sets_pending_slide_id_for_display(tmp_path):
    # Positive scan indicator: the scanned id is resolved and surfaced through
    # capture_status so screen 12 can show "SCANNED: <id> · place slide" before
    # the operator places the slide.
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        runtime.scan_qr("12080_51137181_01_HE")

        assert runtime._pending_slide_id == "51137181"
        assert runtime.capture_status()["pending_slide_id"] == "51137181"


def test_scanned_slide_capture_never_attempts_camera_qr_decode(tmp_path):
    # Goal B: once a slide is scanned, the scanned payload alone drives identity
    # and the camera QR decoder must never run. We prove it by wiring a decoder
    # that raises if called, then completing a real capture end-to-end.
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        def _decoder_must_not_run(_image):
            raise AssertionError(
                "camera QR decode ran after a slide was scanned"
            )

        runtime.workflow.slide_decoder = _decoder_must_not_run

        runtime.scan_qr("12080_51137181_01_HE")

        slide_png = tmp_path / "scanned_slide.png"
        assert cv2.imwrite(
            str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )
        result = runtime._consume_capture(_FakeSlideRecord(slide_png))

        assert result.success is True
        assert result.block_id == "51137181"
        assert result.engine == "scanner"
        # The pending scan state is consumed exactly once.
        assert runtime._pending_slide_payload is None
        assert runtime._pending_slide_id is None


# --------------------------------------------------------------------------
# #185 PR3: Pi-side scanner-skip -- decode the raster only when a QR Search
# is actually needed (camera path). The keyboard-scanner path resolves
# identity from the payload string alone and must never touch pixels.
# --------------------------------------------------------------------------


def test_scanner_path_skips_raster_decode_entirely(tmp_path, monkeypatch):
    """A scanned payload needs no pixels: `capture_slide` must call neither
    `cv2.imread` nor the injected camera-QR `slide_decoder` (the
    rotation/enhancement search), and must still resolve + publish the
    scanner identity exactly as before.

    Note: `record_slide_capture` -> `resolve_claim` (a downstream, main-side
    concern per ADR-0014, unaffected by this PR) legitimately decodes its OWN
    persisted store copy for claimed-pair scoring. That is a different path
    argument (the durable capture-store copy) than the one under test here
    (the exact `source` the Pi was handed), so the assertion is scoped to
    calls against `slide_png` specifically.
    """
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    imread_calls = []
    real_imread = workflow_module.cv2.imread

    def _spy_imread(*args, **kwargs):
        imread_calls.append(args)
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(workflow_module.cv2, "imread", _spy_imread)

    def _decoder_must_not_run(_image):
        raise AssertionError("camera QR search ran for a scanner payload")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.slide_decoder = _decoder_must_not_run
        workflow.prepare_empty_backlight("slide")

        slide_png = tmp_path / "scanned_slide.png"
        assert cv2.imwrite(
            str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )

        result = workflow.capture_slide(
            slide_png,
            captured_at=STARTED_AT,
            scanned_payload="12080_51137181_01_HE",
        )

        rows = store.slide_captures(session.number)

    source_decode_calls = [
        call for call in imread_calls if call and call[0] == str(slide_png)
    ]
    assert source_decode_calls == []
    assert result.success is True
    assert result.block_id == "51137181"
    assert result.engine == "scanner"
    assert len(rows) == 1
    assert Path(rows[0]["capture_path"]).is_file()


def test_scanner_path_with_unreadable_source_still_raises_but_one_layer_later(
    tmp_path,
):
    """Pins the VERIFIED actual behavior delta on #185 PR3 (corrects an
    earlier premise that this "does not raise" -- it still does, just not
    from the Pi's own guard).

    On origin/main, the unconditional `cv2.imread` doubled as an implicit
    "file readable" check, so an unreadable still COMBINED WITH a scanned
    payload raised `ValueError("slide capture could not be read")` on the Pi,
    before anything was queued. Skipping the decode removes that early Pi-
    side guard: `scanner_identity` resolves from the payload alone, and the
    still IS durably queued to this machine's outbox (`outbox.publish_slide`
    fires). But replaying that capture still raises -- from
    `record_slide_capture`'s own, independent readability guard (a pre-
    existing check, untouched by this PR, that decodes the persisted bytes
    and rejects a non-image) -- so the failure is not silently swallowed; it
    surfaces one layer later, on replay, as a different message than before.
    """
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")

        publish_calls = []
        real_publish = workflow.outbox.publish_slide

        def _spy_publish(source, *args, **kwargs):
            publish_calls.append(source)
            return real_publish(source, *args, **kwargs)

        workflow.outbox.publish_slide = _spy_publish

        bogus = tmp_path / "unreadable_scanned_slide.png"
        bogus.write_bytes(b"not a real png")

        with pytest.raises(ValueError, match="not a readable color image"):
            workflow.capture_slide(
                bogus,
                captured_at=STARTED_AT,
                scanned_payload="12080_51137181_01_HE",
            )

    # The durable Pi-local outbox copy was written before the replay/record
    # step raised -- the still is not lost even though the call still fails.
    assert publish_calls == [bogus]


def test_camera_path_decodes_raster_exactly_once(tmp_path, monkeypatch):
    """No scanned payload -> the camera-QR identity path runs, which means
    exactly one `cv2.imread` (of the exact `source` handed to `capture_slide`)
    and exactly one `slide_decoder` call.

    Note: `record_slide_capture` -> `resolve_claim` also decodes its own
    persisted store copy downstream (a separate, main-side concern per
    ADR-0014, out of scope here) -- the assertion is scoped to calls against
    `slide_png` specifically so that unrelated decode does not conflate with
    the Pi-side decode-once contract under test.
    """
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    imread_calls = []
    real_imread = workflow_module.cv2.imread

    def _spy_imread(*args, **kwargs):
        imread_calls.append(args)
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(workflow_module.cv2, "imread", _spy_imread)

    decoder_calls = []
    decoded = select_slide_identity((
        DecodeCandidate(
            "zxing", "DataMatrix", "label+raw", "12080_51137181_01_HE"
        ),
    ))

    def _spy_decoder(image):
        decoder_calls.append(image)
        return decoded

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.slide_decoder = _spy_decoder
        workflow.prepare_empty_backlight("slide")

        slide_png = tmp_path / "camera_slide.png"
        assert cv2.imwrite(
            str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )

        result = workflow.capture_slide(
            slide_png, captured_at=STARTED_AT, scanned_payload=None,
        )

        rows = store.slide_captures(session.number)

    source_decode_calls = [
        call for call in imread_calls if call and call[0] == str(slide_png)
    ]
    assert len(source_decode_calls) == 1
    assert len(decoder_calls) == 1
    assert result.success is True
    assert result.block_id == "51137181"
    assert result.engine != "scanner"
    assert len(rows) == 1
    assert Path(rows[0]["capture_path"]).is_file()


def test_unreadable_image_on_camera_path_still_raises_identically(tmp_path):
    """The `image is None` guard moved (it now only runs on the camera path)
    but must behave identically: an unreadable source still raises the same
    ValueError, and the (never-injected-to-fail here) decoder never runs."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")

        bogus = tmp_path / "not_an_image.png"
        bogus.write_bytes(b"not a real png")

        with pytest.raises(ValueError, match="slide capture could not be read"):
            workflow.capture_slide(
                bogus, captured_at=STARTED_AT, scanned_payload=None,
            )


def test_still_upload_uses_source_path_on_both_scanner_and_camera_paths(tmp_path):
    """The durable PNG republish (`outbox.publish_slide(source, ...)`) reads
    the still from disk by PATH, not from the in-memory decode -- so it must
    still fire on the scanner path (where no decode ever happens) exactly as
    it does on the camera path."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")

        publish_calls = []
        real_publish = workflow.outbox.publish_slide

        def _spy_publish(source, *args, **kwargs):
            publish_calls.append(source)
            return real_publish(source, *args, **kwargs)

        workflow.outbox.publish_slide = _spy_publish

        scanner_png = tmp_path / "scanner_slide.png"
        assert cv2.imwrite(
            str(scanner_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )
        workflow.capture_slide(
            scanner_png,
            captured_at=STARTED_AT,
            scanned_payload="12080_51137181_01_HE",
        )

        camera_png = tmp_path / "camera_slide.png"
        assert cv2.imwrite(
            str(camera_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )
        workflow.slide_decoder = lambda image: select_slide_identity((
            DecodeCandidate(
                "zxing", "DataMatrix", "label+raw", "12080_51137181_01_HE"
            ),
        ))
        workflow.capture_slide(
            camera_png,
            captured_at=STARTED_AT + timedelta(seconds=1),
            scanned_payload=None,
        )

    assert publish_calls == [scanner_png, camera_png]


def test_consume_capture_slide_result_exposes_slide_fields_and_stage_timings(tmp_path):
    """#171: `_consume_capture`'s slide branch must keep returning something
    duck-type-compatible as a `SlideQRResult` (.success/.block_id/.engine)
    while ALSO carrying decode_ms/outbox_ms/send_ms, so `CaptureController`
    can forward the split without breaking existing `.success`/`.block_id`
    readers (mirrors `test_scanned_slide_capture_never_attempts_camera_qr_decode`)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        workflow.prepare_empty_backlight("slide")
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        runtime.scan_qr("12080_51137181_01_HE")

        slide_png = tmp_path / "scanned_slide.png"
        assert cv2.imwrite(
            str(slide_png), np.full((80, 120, 3), 180, dtype=np.uint8)
        )
        result = runtime._consume_capture(_FakeSlideRecord(slide_png))

        assert result.success is True
        assert result.block_id == "51137181"
        assert result.engine == "scanner"
        assert result.decode_ms is not None
        assert result.outbox_ms is not None
        assert result.send_ms is not None


def test_scan_qr_in_slide_phase_skips_stash_when_precheck_rejects(tmp_path):
    # A re-scan of an already-verdicted slide is caught at scan time: the
    # runtime consults the workflow precheck and stashes nothing, so no wasted
    # capture cycle occurs (the duplicate flash fires off the emitted event).
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        runtime.workflow.precheck_slide_scan = lambda payload: False

        assert runtime.scan_qr("12080_51137181_01_HE") is None
        assert runtime._pending_slide_payload is None


def test_pi_capture_runtime_refuses_to_guess_between_multiple_pending_blocks(
    tmp_path, capsys,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "22222222")
    store.scan_block(session.number, "11111111")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow, _FakeRealCamera(), capture_root=tmp_path / "pi_captures",
        )

    assert runtime.capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    output = capsys.readouterr().out
    assert "22222222" in output and "11111111" in output


def test_rejected_second_runtime_scan_rolls_back_its_durable_row(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow, _FakeRealCamera(), capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        empty = np.full((80, 120, 3), 180, dtype=np.uint8)
        runtime.process_frame(empty, now=0.0, captured_at=STARTED_AT)
        assert runtime.scan_block("22222222").accepted
        with pytest.raises(RuntimeError, match="already pending"):
            runtime.scan_block("11111111")

    assert store.awaiting_capture_blocks(session.number) == ("22222222",)


def test_pi_capture_runtime_stores_latest_frame_for_preview_jpeg(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeRealCamera(),
            capture_root=tmp_path / "pi_captures",
        )

    assert runtime.latest_preview_jpeg() is None

    frame = np.full((80, 120, 3), 55, dtype=np.uint8)
    runtime.process_frame(frame, now=0.0, captured_at=STARTED_AT)

    jpeg = runtime.latest_preview_jpeg()
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


# ---------------------------------------------------------------------------
# --profile (#172): PiCaptureRuntime.end_session syncs the Pi-local motion
# curve into the main-computer bundle before delegating to
# workflow.end_session -- mirrors the existing finish_blocks/start_work_order
# overrides of the __getattr__ proxy.
# ---------------------------------------------------------------------------


def test_pi_capture_runtime_end_session_syncs_profile_artifacts_into_main_computer_bundle_when_profile_enabled(tmp_path):
    curve_path = tmp_path / "motion_curve.csv"
    curve_path.write_text("timestamp,state,motion_score\n", encoding="utf-8")
    config_path = tmp_path / "profile_config.json"
    config_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

    flushed: list[bool] = []
    uploads: list[tuple[str, int, Path]] = []

    class _Writer:
        def flush(self):
            flushed.append(True)

    class _Transport:
        def upload_profile_curve(self, session_number, path):
            uploads.append(("curve", session_number, Path(path)))
            return "uploaded"

        def upload_profile_config(self, session_number, path):
            uploads.append(("config", session_number, Path(path)))
            return "uploaded"

    class _Workflow:
        def __init__(self):
            self.session = SimpleNamespace(number=7)
            self.transport = _Transport()
            self.ended = False

        def end_session(self, *, confirm):
            assert confirm is True
            self.ended = True
            return "snapshot"

    workflow = _Workflow()
    runtime = run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)
    runtime.profile = True
    runtime.workflow = workflow
    runtime.motion_curve_writer = _Writer()
    runtime.motion_curve_path = curve_path
    runtime.profile_config_path = config_path

    result = runtime.end_session(confirm=True)

    # Flush-then-upload must happen before the workflow finalizes, so a
    # finalize-triggered cleanup never races the sync.
    assert flushed == [True]
    assert uploads == [("curve", 7, curve_path), ("config", 7, config_path)]
    assert workflow.ended is True
    assert result == "snapshot"


def test_pi_capture_runtime_end_session_skips_motion_curve_sync_when_profile_disabled():
    """Profile-off sessions never created a motion curve; end_session must
    delegate straight through without touching a writer that doesn't exist."""

    class _Workflow:
        def __init__(self):
            self.ended = False

        def end_session(self, *, confirm):
            assert confirm is True
            self.ended = True
            return "snapshot"

    workflow = _Workflow()
    runtime = run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)
    runtime.profile = False
    runtime.workflow = workflow

    result = runtime.end_session(confirm=True)

    assert workflow.ended is True
    assert result == "snapshot"
