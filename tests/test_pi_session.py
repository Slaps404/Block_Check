"""Off-hardware coverage for the Pi-side session runtime.

Drives `app.pi_session` (PiCaptureRuntime + the operator-loop core) against a
REAL `LoopbackCaptureReceiver` + `ProcessingStore` with headless cameras.
Real camera feeds and physical cable pulls stay with hardware QA (ADR 0002).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, RLock, Thread
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
from transport.remote import RemoteProcessingStore, UrlTransport
from session.console import COMMAND_ARITY, COMMAND_HELP, _COMMANDS, command_cheat_sheet
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionIdentity,
    SessionWorkflow,
)
from slide.qr import select_slide_identity

from app import pi_session as run_pi_session
import session.workflow as workflow_module
from store_harness import (  # noqa: F401
    fake_fingerprint_builder as _fake_fingerprint_builder,
    fake_score_cache_builder as _fake_score_cache_builder,
    lightweight_qc_artifacts,
    upload_block as _direct_capture_block,
)


STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
SLIDE_PAYLOAD = "12080_51137181_01_HE"


def _tiny_mask_preprocessor(_path):
    return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}


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


class _FakeSlideRecord:
    """Minimal capture record: `_consume_capture` reads role/path/captured_at."""

    role = "slide"

    def __init__(self, path):
        self.path = path
        self.captured_at = STARTED_AT


def _new_session(tmp_path, *, session_mode=None, **store_kwargs):
    store = ProcessingStore(tmp_path / "processing", **store_kwargs)
    mode_kwargs = {} if session_mode is None else {"session_mode": session_mode}
    return store, store.start_session(started_at=STARTED_AT, **mode_kwargs)


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


def _build_local_workflow(
    tmp_path, store, session, *, session_mode=None, hybrid_descriptor_names=(),
):
    """A workflow over the LOCAL store: work-order verbs are not RPC-wired."""
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


def _slide_workflow(tmp_path, store, session, receiver_url):
    """Remote workflow with the slide phase already calibrated."""
    workflow = _build_workflow(
        tmp_path, store, session, receiver_url, camera=_FakePhaseCamera()
    )
    workflow.prepare_empty_backlight("slide")
    return workflow


def _runtime(workflow, **kwargs):
    """PiCaptureRuntime with a fake camera, captures under <tmp>/pi_captures."""
    kwargs.setdefault("capture_root", workflow.outbox.directory.parent / "pi_captures")
    kwargs.setdefault(
        "session_config",
        run_pi_session.SessionConfig(baseline_frames=1, stable_duration=0.0),
    )
    camera = kwargs.pop("camera", None) or _FakeRealCamera()
    return run_pi_session.PiCaptureRuntime(workflow, camera, **kwargs)


def _bare_runtime():
    return run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)


def _empty_frame():
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _write_slide_png(path):
    assert cv2.imwrite(str(path), _empty_frame())
    return path


@pytest.fixture
def slide_phase_store(tmp_path):
    """(store, session) for an empty session already drained into slides."""
    store, session = _new_session(tmp_path)
    with LoopbackCaptureReceiver(store) as receiver:
        _build_workflow(tmp_path, store, session, receiver.url).finish_blocks()
    return store, session


@pytest.fixture
def hybrid_store(tmp_path, lightweight_qc_artifacts):
    """(store, session, local workflow) for a Hybrid session."""
    store, session = _new_session(
        tmp_path,
        session_mode="hybrid",
        preprocessor=lambda _path: (
            np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}
        ),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    workflow = _build_local_workflow(
        tmp_path, store, session, session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    return store, session, workflow


def test_capture_status_does_not_wait_for_long_running_capture_lock():
    """The kiosk must see CAPTURE_REQUESTED while the still/upload holds the lock."""
    runtime = _bare_runtime()
    runtime._lock = Lock()
    runtime.capture_session = SimpleNamespace(
        state=CaptureState.CAPTURE_REQUESTED,
        mode=run_pi_session.CaptureMode.SLIDE,
        pending_block_id=None,
    )
    runtime._pending_slide_id = "51137181"

    done = Event()
    result = []

    def read_status():
        result.append(runtime.capture_status())
        done.set()

    runtime._lock.acquire()
    reader = Thread(target=read_status)
    reader.start()
    try:
        assert done.wait(timeout=0.25), (
            "capture_status waited behind the long-running capture lock"
        )
    finally:
        runtime._lock.release()
        reader.join(timeout=1)

    assert result == [{
        "capture_state": "CAPTURE_REQUESTED",
        "capture_mode": "slide",
        "led_only_blocks": False,
        "pending_block_id": None,
        "pending_slide_id": "51137181",
    }]


# --------------------------------------------------------------------------
# Framing calibration / store.root seam
# --------------------------------------------------------------------------


class _RemoteShapedStoreWithoutRoot:
    """RemoteProcessingStore's shape: no `.root` (ctor raises before use)."""


def test_local_store_with_no_framing_calibration_still_uses_store_root(tmp_path):
    store, session = _new_session(tmp_path)

    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:1"),
    )

    assert workflow.framing_calibration.path == store.root / "framing_calibration.json"


def test_rootless_store_with_no_framing_calibration_raises_clear_value_error(tmp_path):
    _store, session = _new_session(tmp_path)

    with pytest.raises(ValueError, match="remote store requires an explicit"):
        SessionWorkflow(
            session=session,
            store=_RemoteShapedStoreWithoutRoot(),
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient("http://127.0.0.1:1"),
        )


# --------------------------------------------------------------------------
# Operator command table
# --------------------------------------------------------------------------


def test_every_command_has_a_help_entry():
    assert set(COMMAND_HELP) == set(_COMMANDS)
    # Debug/tooling verbs that other test files exercise only via the runtime.
    assert {"snap", "snap_3", "deskew", "focus"} <= set(_COMMANDS)
    assert COMMAND_ARITY["focus"] == (0, 1)


# --------------------------------------------------------------------------
# run_one_command operator-loop core, over a real loopback wire
# --------------------------------------------------------------------------


def test_run_one_command_summary_renders_a_session_summary(tmp_path):
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        rendered = run_pi_session.run_one_command(workflow, "summary")

    assert f"Session {session.number}" in rendered
    assert "PASS" in rendered


def test_run_one_command_help_unknown_and_blank_never_touch_the_workflow():
    workflow = object()
    sheet = command_cheat_sheet()

    assert run_pi_session.run_one_command(workflow, "help") == sheet
    assert run_pi_session.run_one_command(workflow, "delete_everything") == sheet
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
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "51151378" in rendered
    assert "Accepted" in rendered


def test_run_one_command_store_error_renders_clean_message_without_raising(tmp_path):
    """A protocol-level StoreError (unknown session) renders, never raises."""
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        workflow.session = SessionIdentity(999999, STARTED_AT, tmp_path)

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "unknown session" in rendered.lower()


def test_run_one_command_transport_error_renders_connectivity_note(
    tmp_path, dead_receiver_url
):
    """A transport failure renders a connectivity note, never raises."""
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        # Construct against a live receiver (the ctor makes store calls), then
        # swap in a dead endpoint so only the command under test fails.
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        workflow.store = RemoteProcessingStore(
            dead_receiver_url, max_attempts=1, backoff=0
        )

        rendered = run_pi_session.run_one_command(workflow, "scan_block 51151378")

    assert "connectiv" in rendered.lower() or "connection" in rendered.lower()


def test_finalization_is_reachable_end_to_end_over_the_wire_for_an_empty_session(tmp_path):
    """finish_blocks -> end_session -> poll_finalization over the real server."""
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)

        run_pi_session.run_one_command(workflow, "finish_blocks")
        ended = run_pi_session.run_one_command(workflow, "end_session")
        polled = run_pi_session.run_one_command(workflow, "poll_finalization")

    assert "finalized" in ended.lower()
    assert "finalized" in polled.lower()
    assert store.snapshot(session).phase == "finalized"


# --------------------------------------------------------------------------
# Camera calibration and the capture loop
# --------------------------------------------------------------------------


def test_confirm_empty_runs_full_empty_backlight_setup_fail_then_succeed(
    tmp_path, capsys,
):
    # Also proves the ctor defers camera activation to confirm_empty.
    store, session = _new_session(tmp_path, preprocessor=_tiny_mask_preprocessor)
    phase_camera = _FailThenOkPhaseCamera()
    logged = []

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=phase_camera
        )
        runtime = _runtime(
            workflow,
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

    store, session = _new_session(tmp_path, preprocessor=_tiny_mask_preprocessor)
    phase_camera = _BlockingPhaseCamera()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=phase_camera
        )
        runtime = _runtime(workflow)

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


@pytest.mark.native_resolution
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
    store, session = _new_session(tmp_path, preprocessor=preprocessor)
    monkeypatch.setattr(run_pi_session, "_DRAIN_POLL_INTERVAL_SECONDS", 0.02)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(tmp_path, store, session, receiver.url)
        runtime = _runtime(workflow)
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
    runtime = _bare_runtime()
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


@pytest.mark.parametrize(
    ("status", "expected_variant"),
    [
        (SimpleNamespace(neutral_oblique_logical_output=True, logical_output=True,
                         backlight_brightness=0.0), "neutral_oblique"),
        (SimpleNamespace(neutral_oblique_logical_output=False, logical_output=True,
                         backlight_brightness=0.5), "coax"),
        (SimpleNamespace(neutral_oblique_logical_output=False, logical_output=False,
                         backlight_brightness=0.5), "backlight"),
        (SimpleNamespace(neutral_oblique_logical_output=False, logical_output=False,
                         backlight_brightness=0.0), "coax"),
    ],
)
def test_preview_exposure_tracks_active_illumination(status, expected_variant):
    variants = []
    runtime = _bare_runtime()
    runtime.illumination = SimpleNamespace(status=lambda: status)
    runtime.controller = SimpleNamespace(
        set_preview_variant=variants.append
    )

    runtime._sync_preview_exposure_to_lights()

    assert variants == [expected_variant]


@pytest.mark.parametrize("capture_paused", [True, False])
def test_process_frame_updates_preview_and_advances_capture_unless_paused(
    capture_paused,
):
    """A paused workflow still refreshes the preview but never captures."""
    handled = []
    runtime = _bare_runtime()
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
    frame = _empty_frame()

    result = runtime.process_frame(frame, now=5.0, captured_at=STARTED_AT)

    if capture_paused:
        assert result is None
        assert handled == []
    else:
        assert result == "handled"
        assert handled == [(frame, 5.0, STARTED_AT)]
    assert runtime._latest_preview_frame is frame
    assert runtime._last_frame_now == 5.0


def _specimen_frame():
    specimen = _empty_frame()
    cv2.rectangle(specimen, (30, 20), (90, 60), (40, 40, 40), -1)
    return specimen


def _place_specimen(runtime):
    for now in (1.0, 2.0, 3.0):
        runtime.process_frame(_specimen_frame(), now=now, captured_at=STARTED_AT)


@pytest.mark.native_resolution
def test_pi_capture_runtime_drives_controller_and_publishes_block(tmp_path):
    store, session = _new_session(tmp_path, preprocessor=_tiny_mask_preprocessor)
    camera = _FakeRealCamera()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow, camera=camera)
        runtime.start(background=False)
        assert runtime.capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
        runtime.confirm_empty()
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
        runtime.process_frame(_empty_frame(), now=0.0, captured_at=STARTED_AT)
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
        assert runtime.scan_block("51151378").accepted

        _place_specimen(runtime)
        runtime.close()

    store.wait_for_jobs()
    row = store.get_set(session.number, "51151378")
    assert row["capture_id"] is not None
    assert Path(row["capture_path"]).is_file()
    assert camera.started == 1
    assert camera.closed == 1


@pytest.mark.native_resolution
def test_pi_capture_runtime_restores_one_accepted_block_without_rescan(tmp_path):
    store, session = _new_session(tmp_path, preprocessor=_tiny_mask_preprocessor)
    assert store.scan_block(session.number, "51151378").accepted

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow)
        assert runtime.capture_session.pending_block_id == "51151378"
        assert (
            runtime.capture_session.state
            is CaptureState.AWAITING_BASELINE_CONFIRMATION
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        assert runtime.capture_session.state is CaptureState.EMPTY
        runtime.process_frame(_empty_frame(), now=0.0, captured_at=STARTED_AT)
        _place_specimen(runtime)
        runtime.close()

    assert store.get_set(session.number, "51151378")["capture_id"] is not None


def test_pi_capture_runtime_restores_fail_closed_slide_recovery_after_restart(
    tmp_path, slide_phase_store,
):
    """A Pi restart mid-slides must re-arm fail-closed unreadable-slide recovery."""
    store, session = slide_phase_store
    store.record_slide_capture(
        session.number,
        _write_slide_png(tmp_path / "unreadable.png"),
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=1500.0,
    )
    assert store.slide_recovery_state(session.number) == "reposition"

    # A fresh runtime is the post-restart process: it must come up re-armed.
    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_slide_workflow(tmp_path, store, session, receiver.url))

    assert runtime.capture_session.mode is run_pi_session.CaptureMode.SLIDE
    assert runtime.capture_session.unreadable_slide_can_be_skipped


# --------------------------------------------------------------------------
# Work-order lifecycle gating (open retrieval, hybrid)
# --------------------------------------------------------------------------


def test_pi_capture_runtime_rejects_start_work_order_without_open_retrieval_flag(
    tmp_path,
):
    """Without --open-retrieval the work-order verbs are rejected."""
    store, session = _new_session(tmp_path)
    runtime = _runtime(_build_local_workflow(tmp_path, store, session))

    assert runtime.open_retrieval is False
    with pytest.raises(RuntimeError, match="open.retrieval"):
        runtime.start_work_order()
    with pytest.raises(RuntimeError, match="open.retrieval"):
        runtime.finish_work_order()


def test_pi_capture_runtime_start_and_finish_work_order_when_open_retrieval_enabled(
    tmp_path,
):
    """With --open-retrieval the verbs delegate to the real SessionWorkflow."""
    store, session = _new_session(tmp_path)
    runtime = _runtime(
        _build_local_workflow(tmp_path, store, session), open_retrieval=True
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


def test_pi_capture_runtime_start_and_finish_work_order_when_hybrid_enabled(
    hybrid_store,
):
    """The gate keys on session_mode, so the verbs unlock for Hybrid."""
    store, session, workflow = hybrid_store
    runtime = _runtime(workflow, hybrid=True)

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
    # Hybrid never runs the N-by-N path: it closes to 'finalized', not 'scoring'.
    assert store.get_work_order(session.number, work_order_id)[
        "lifecycle_state"
    ] == "finalized"


def test_pi_capture_runtime_multiple_work_orders_in_one_hybrid_session_freeze_independent_pools(
    hybrid_store,
):
    """Sequential Hybrid work orders each freeze their own candidate pool."""
    store, session, workflow = hybrid_store
    runtime = _runtime(workflow, hybrid=True)

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
    # No second confirm_empty(): the slide baseline from A is reused.
    assert runtime.finish_blocks().phase == "slides"
    runtime.finish_work_order()

    pool_a = store.hybrid_pool(work_order_a)
    pool_b = store.hybrid_pool(work_order_b)
    assert pool_a is not None and pool_b is not None
    assert set(pool_a.block_ids) == {"11111111", "22222222"}
    assert set(pool_b.block_ids) == {"33333333", "44444444"}
    assert set(pool_a.block_ids).isdisjoint(pool_b.block_ids)


# --------------------------------------------------------------------------
# Boot-seed / toggle of PiCaptureRuntime.work_order_open
# --------------------------------------------------------------------------


class _CountingTransport:
    """Wraps a real transport and records each RPC method name it POSTs."""

    def __init__(self, inner):
        self.inner = inner
        self.methods_called = []

    def post(self, url, payload):
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


@pytest.mark.parametrize("mode_kwarg", ["open_retrieval", "hybrid", "hybrid_shadow"])
def test_pi_capture_runtime_seeds_work_order_open_true_after_restart_mid_bracket(
    tmp_path, mode_kwarg,
):
    """A restart mid-bracket comes up showing FINISH, not START."""
    store, session = _new_session(
        tmp_path, session_mode=None if mode_kwarg == "open_retrieval" else mode_kwarg
    )
    store.start_work_order(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow, **{mode_kwarg: True})

    assert runtime.work_order_open is True
    assert runtime.has_work_orders is True
    # open_retrieval stays narrowly OPEN_RETRIEVAL-only.
    assert runtime.open_retrieval is (mode_kwarg == "open_retrieval")


@pytest.mark.parametrize(
    ("bracket", "missing_rpc"),
    [
        (None, "has_work_orders"),
        ("open", "open_work_order_id"),
    ],
    ids=["fresh-no-history-rpc", "open-no-open-rpc"],
)
def test_pi_runtime_requires_receiver_restart_when_a_seed_rpc_is_missing(
    tmp_path, monkeypatch, bracket, missing_rpc,
):
    """The client never guesses work-order state an old receiver cannot report."""
    store, session = _new_session(tmp_path)
    if bracket:
        store.start_work_order(session.number)
    monkeypatch.delitem(workflow_module._RPC_METHODS, missing_rpc)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        with pytest.raises(RuntimeError, match="restart the processing receiver"):
            _runtime(workflow, open_retrieval=True)


def test_pi_runtime_open_bracket_safely_implies_history_when_history_rpc_is_missing(
    tmp_path, monkeypatch,
):
    store, session = _new_session(tmp_path)
    store.start_work_order(session.number)
    monkeypatch.delitem(workflow_module._RPC_METHODS, "has_work_orders")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow, open_retrieval=True)

    assert runtime.work_order_open is True
    assert runtime.has_work_orders is True


def test_pi_capture_runtime_toggles_work_order_open_on_start_and_finish(tmp_path):
    """The open flag toggles while durable history stays true after start."""
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow, open_retrieval=True)
        # A fresh session seeds no open bracket and no history.
        assert runtime.work_order_open is False
        assert runtime.has_work_orders is False

        runtime.start_work_order()
        assert runtime.work_order_open is True
        assert runtime.has_work_orders is True

        runtime.finish_blocks()
        runtime.confirm_empty()
        runtime.finish_work_order()
        assert runtime.work_order_open is False
        assert runtime.has_work_orders is True


def test_starting_the_next_work_order_rebuilds_the_block_capture_controller(tmp_path):
    store, session = _new_session(tmp_path)
    workflow = _build_local_workflow(tmp_path, store, session)
    runtime = _runtime(workflow, open_retrieval=True)

    runtime.start_work_order()
    runtime.finish_blocks()
    assert runtime.mode is run_pi_session.CaptureMode.SLIDE
    runtime.confirm_empty()
    runtime.finish_work_order()

    runtime.start_work_order()

    assert workflow.snapshot().phase == "blocks"
    assert runtime.mode is run_pi_session.CaptureMode.BLOCK


def test_finish_work_order_rejects_a_scanned_slide_that_has_not_captured(tmp_path):
    store, session = _new_session(tmp_path)
    runtime = _runtime(
        _build_local_workflow(tmp_path, store, session), open_retrieval=True
    )
    work_order_id = runtime.start_work_order()
    runtime.finish_blocks()
    runtime.confirm_empty()
    runtime.scan_qr(SLIDE_PAYLOAD)

    with pytest.raises(RuntimeError, match="capture the scanned slide"):
        runtime.finish_work_order()

    assert runtime.work_order_open is True
    assert store.open_work_order_id(session.number) == work_order_id


def test_pi_capture_runtime_never_seeds_work_order_open_without_open_retrieval_flag(
    tmp_path,
):
    """Normal mode never issues the seed RPCs and seeds nothing open."""
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        transport = _CountingTransport(UrlTransport())
        workflow = SessionWorkflow(
            session=session,
            store=RemoteProcessingStore(receiver.url, transport=transport),
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "pi_local" / "framing_calibration.json"
            ),
            camera=_FakePhaseCamera(),
        )
        runtime = _runtime(workflow)

    assert runtime.open_retrieval is False
    assert runtime.work_order_open is False
    assert runtime.has_work_orders is False
    assert "open_work_order_id" not in transport.methods_called
    assert "has_work_orders" not in transport.methods_called


# --------------------------------------------------------------------------
# scan_qr and the slide capture path
# --------------------------------------------------------------------------


def test_scan_qr_in_block_phase_registers_the_block_like_scan_block(tmp_path):
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow)
        runtime.start(background=False)
        runtime.confirm_empty()
        runtime.process_frame(_empty_frame(), now=0.0, captured_at=STARTED_AT)

        outcome = runtime.scan_qr("51151378")

    assert outcome.accepted
    assert runtime.capture_session.pending_block_id == "51151378"


def test_scan_qr_in_slide_phase_sets_pending_slide_id_for_display(
    tmp_path, slide_phase_store,
):
    # Screen 12 shows "SCANNED: <id> · place slide" before the slide is placed.
    store, session = slide_phase_store

    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_build_workflow(tmp_path, store, session, receiver.url))
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        runtime.scan_qr(SLIDE_PAYLOAD)

        assert runtime._pending_slide_id == "51137181"
        assert runtime.capture_status()["pending_slide_id"] == "51137181"


def test_scan_qr_in_slide_phase_skips_stash_when_precheck_rejects(
    tmp_path, slide_phase_store,
):
    # A re-scan of an already-verdicted slide stashes nothing (no wasted cycle).
    store, session = slide_phase_store

    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_build_workflow(tmp_path, store, session, receiver.url))
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        runtime.workflow.precheck_slide_scan = lambda payload: False

        assert runtime.scan_qr(SLIDE_PAYLOAD) is None
        assert runtime._pending_slide_payload is None


def test_consume_capture_slide_result_exposes_slide_fields_and_stage_timings(
    tmp_path, slide_phase_store,
):
    """The scanned-slide result keeps SlideQRResult fields plus split
    timings, and the camera QR decoder never runs once a slide is scanned."""
    store, session = slide_phase_store

    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_slide_workflow(tmp_path, store, session, receiver.url))
        assert runtime.mode is run_pi_session.CaptureMode.SLIDE

        def _decoder_must_not_run(_image):
            raise AssertionError("camera QR decode ran after a slide was scanned")

        runtime.workflow.slide_decoder = _decoder_must_not_run

        runtime.scan_qr(SLIDE_PAYLOAD)
        result = runtime._consume_capture(
            _FakeSlideRecord(_write_slide_png(tmp_path / "scanned_slide.png"))
        )

        assert result.success is True
        assert result.block_id == "51137181"
        assert result.engine == "scanner"
        assert result.decode_ms is not None
        assert result.outbox_ms is not None
        assert result.send_ms is not None
        # The pending scan state is consumed exactly once.
        assert runtime._pending_slide_payload is None
        assert runtime._pending_slide_id is None


# --------------------------------------------------------------------------
# The scanner path never decodes the raster; unreadable stills still raise
# --------------------------------------------------------------------------


@pytest.fixture
def source_imreads(monkeypatch):
    """Records every cv2.imread call made inside session.workflow."""
    calls = []
    real_imread = workflow_module.cv2.imread

    def _spy_imread(*args, **kwargs):
        calls.append(args)
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(workflow_module.cv2, "imread", _spy_imread)
    return calls


def _decodes_of(imread_calls, path):
    # Scoped to `path`: record_slide_capture legitimately decodes its own copy.
    return [call for call in imread_calls if call and call[0] == str(path)]


def _spy_publish_slide(workflow):
    calls = []
    real_publish = workflow.outbox.publish_slide

    def _spy(source, *args, **kwargs):
        calls.append(source)
        return real_publish(source, *args, **kwargs)

    workflow.outbox.publish_slide = _spy
    return calls


def test_scanner_path_skips_raster_decode_entirely(
    tmp_path, slide_phase_store, source_imreads,
):
    """A scanned payload needs no pixels: no imread of the source, no decoder."""
    store, session = slide_phase_store

    def _decoder_must_not_run(_image):
        raise AssertionError("camera QR search ran for a scanner payload")

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _slide_workflow(tmp_path, store, session, receiver.url)
        workflow.slide_decoder = _decoder_must_not_run
        slide_png = _write_slide_png(tmp_path / "scanned_slide.png")

        result = workflow.capture_slide(
            slide_png, captured_at=STARTED_AT, scanned_payload=SLIDE_PAYLOAD,
        )

        rows = store.slide_captures(session.number)

    assert _decodes_of(source_imreads, slide_png) == []
    assert result.success is True
    assert result.block_id == "51137181"
    assert result.engine == "scanner"
    assert len(rows) == 1
    assert Path(rows[0]["capture_path"]).is_file()


def test_scanner_path_with_unreadable_source_still_raises_but_one_layer_later(
    tmp_path, slide_phase_store,
):
    """An unreadable scanned still is queued to the outbox, then fails on replay."""
    store, session = slide_phase_store

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _slide_workflow(tmp_path, store, session, receiver.url)
        publish_calls = _spy_publish_slide(workflow)
        bogus = tmp_path / "unreadable_scanned_slide.png"
        bogus.write_bytes(b"not a real png")

        with pytest.raises(ValueError, match="not a readable color image"):
            workflow.capture_slide(
                bogus, captured_at=STARTED_AT, scanned_payload=SLIDE_PAYLOAD,
            )

    # The durable Pi-local outbox copy was written before replay raised.
    assert publish_calls == [bogus]


def test_unreadable_image_on_camera_path_still_raises_identically(
    tmp_path, slide_phase_store,
):
    """The camera path still rejects an unreadable source with the same error."""
    store, session = slide_phase_store

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _slide_workflow(tmp_path, store, session, receiver.url)
        bogus = tmp_path / "not_an_image.png"
        bogus.write_bytes(b"not a real png")

        with pytest.raises(ValueError, match="slide capture could not be read"):
            workflow.capture_slide(
                bogus, captured_at=STARTED_AT, scanned_payload=None,
            )


# --------------------------------------------------------------------------
# Pending-block safety and preview
# --------------------------------------------------------------------------


def test_pi_capture_runtime_refuses_to_guess_between_multiple_pending_blocks(
    tmp_path, capsys,
):
    store, session = _new_session(tmp_path)
    store.scan_block(session.number, "22222222")
    store.scan_block(session.number, "11111111")

    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_build_workflow(tmp_path, store, session, receiver.url))

    assert runtime.capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    output = capsys.readouterr().out
    assert "22222222" in output and "11111111" in output


def test_rejected_second_runtime_scan_rolls_back_its_durable_row(tmp_path):
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = _build_workflow(
            tmp_path, store, session, receiver.url, camera=_FakePhaseCamera()
        )
        runtime = _runtime(workflow)
        runtime.start(background=False)
        runtime.confirm_empty()
        runtime.process_frame(_empty_frame(), now=0.0, captured_at=STARTED_AT)
        assert runtime.scan_block("22222222").accepted
        with pytest.raises(RuntimeError, match="already pending"):
            runtime.scan_block("11111111")

    assert store.awaiting_capture_blocks(session.number) == ("22222222",)


def test_pi_capture_runtime_stores_latest_frame_for_preview_jpeg(tmp_path):
    store, session = _new_session(tmp_path)

    with LoopbackCaptureReceiver(store) as receiver:
        runtime = _runtime(_build_workflow(tmp_path, store, session, receiver.url))

    assert runtime.latest_preview_jpeg() is None

    frame = np.full((80, 120, 3), 55, dtype=np.uint8)
    runtime.process_frame(frame, now=0.0, captured_at=STARTED_AT)

    jpeg = runtime.latest_preview_jpeg()
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


# ---------------------------------------------------------------------------
# end_session: lights off
# ---------------------------------------------------------------------------


def test_pi_capture_runtime_end_session_turns_both_lights_off():
    # Profile off: end_session delegates without touching any motion curve.
    class _Workflow:
        def end_session(self, *, confirm):
            assert confirm is True
            return "snapshot"

    class _Illumination:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    illumination = _Illumination()
    runtime = _bare_runtime()
    runtime.profile = False
    runtime.workflow = _Workflow()
    runtime.illumination = illumination

    assert runtime.end_session(confirm=True) == "snapshot"
    assert illumination.shutdown_calls == 1
