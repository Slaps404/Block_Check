"""Headless capture-review gate core (issue #136, ADR 0006)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from capture_runtime import CaptureController
from capture_session import (
    CaptureMode,
    CaptureResult,
    CaptureSession,
    CaptureState,
    SessionConfig,
)
from capture_storage import CaptureStore
from constants import CAPTURE_DIMENSIONS

NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def _empty() -> np.ndarray:
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _specimen() -> np.ndarray:
    frame = _empty()
    cv2.rectangle(frame, (30, 25), (90, 55), (45, 45, 45), -1)
    return frame


def _valid_png(tmp_path, *, role: str = "slide") -> "pytest.TempPathFactory":
    path = tmp_path / "capture.png"
    width, height = CAPTURE_DIMENSIONS[role]
    cv2.imwrite(str(path), np.zeros((height, width), dtype=np.uint8))
    return path


def _ready_slide_session(*, review_captures: bool = False) -> CaptureSession:
    session = CaptureSession(
        SessionConfig(baseline_frames=2, stable_duration=0.0),
        mode="slide",
        review_captures=review_captures,
    )
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    return session


class FakeCamera:
    def __init__(self):
        self.calls = []

    def start_preview(self, *, settings, size, fps):
        self.calls.append(("preview", settings, size, fps))

    def capture_still(self, path, *, settings, size):
        self.calls.append(("still", settings, size))
        cv2.imwrite(str(path), np.zeros((size[1], size[0]), dtype=np.uint8))

    def resume_preview(self):
        self.calls.append(("resume",))

    def close(self):
        self.calls.append(("close",))


def test_review_captures_defaults_false():
    session = CaptureSession()
    assert session.review_captures is False
    assert CaptureState.AWAITING_ACCEPT in CaptureState


def test_flag_on_successful_capture_lands_in_awaiting_accept(tmp_path):
    session = _ready_slide_session(review_captures=True)
    session.accept_frame(_specimen(), now=1.0)
    assert session.accept_frame(_specimen(), now=2.0).capture_requested

    path = _valid_png(tmp_path)
    session.accept_capture_result(
        CaptureResult.success(path, metadata={"role": "slide"})
    )

    assert session.state is CaptureState.AWAITING_ACCEPT
    assert session.last_capture is not None
    assert session.last_capture.path == path
    assert session.last_capture.role is CaptureMode.SLIDE


def test_flag_off_successful_capture_still_goes_to_waiting_for_removal(tmp_path):
    session = _ready_slide_session(review_captures=False)
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    path = _valid_png(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))

    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert CaptureState.AWAITING_ACCEPT not in (
        event.state for event in session.drain_events()
    )


def test_accept_capture_only_valid_from_awaiting_accept(tmp_path):
    session = _ready_slide_session(review_captures=True)
    with pytest.raises(RuntimeError, match="awaiting operator review"):
        session.accept_capture()

    path = _valid_png(tmp_path)
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    session.accept_capture_result(CaptureResult.success(path))
    assert session.state is CaptureState.AWAITING_ACCEPT

    held = session.accept_capture()
    assert held.path == path
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert any(e.kind == "capture_accepted" for e in session.drain_events())


def test_block_hold_preserves_pending_block_id(tmp_path):
    session = CaptureSession(
        SessionConfig(baseline_frames=2, stable_duration=0.0),
        mode="block",
        review_captures=True,
    )
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    session.submit_scan("51151378")
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    path = _valid_png(tmp_path, role="block")
    session.accept_capture_result(
        CaptureResult.success(
            path,
            metadata={"block_id": "51151378", "role": "block"},
        )
    )

    assert session.state is CaptureState.AWAITING_ACCEPT
    assert session.pending_block_id == "51151378"
    assert session.last_capture.block_id == "51151378"


def test_accept_capture_clears_pending_block_id(tmp_path):
    session = CaptureSession(
        SessionConfig(baseline_frames=2, stable_duration=0.0),
        mode="block",
        review_captures=True,
    )
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    session.submit_scan("51151378")
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    path = _valid_png(tmp_path, role="block")
    session.accept_capture_result(
        CaptureResult.success(
            path,
            metadata={"block_id": "51151378", "role": "block"},
        )
    )
    session.accept_capture()

    assert session.pending_block_id is None
    assert session.state is CaptureState.WAITING_FOR_REMOVAL


def test_retry_from_awaiting_accept_rearms_and_preserves_block_id(tmp_path):
    session = CaptureSession(
        SessionConfig(baseline_frames=2, stable_duration=0.0),
        mode="block",
        review_captures=True,
    )
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    session.submit_scan("51151378")
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    path = _valid_png(tmp_path, role="block")
    session.accept_capture_result(
        CaptureResult.success(
            path,
            metadata={"block_id": "51151378", "role": "block"},
        )
    )

    retry = session.retry_capture()
    assert retry.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED
    assert session.pending_block_id == "51151378"
    assert session.last_capture is None
    assert any(e.kind == "capture_retaken" for e in session.drain_events())


def test_accept_frame_does_not_request_capture_in_awaiting_accept(tmp_path):
    session = _ready_slide_session(review_captures=True)
    session.accept_frame(_specimen(), now=1.0)
    session.accept_frame(_specimen(), now=2.0)
    session.accept_capture_result(CaptureResult.success(_valid_png(tmp_path)))

    result = session.accept_frame(_specimen(), now=3.0)
    assert not result.capture_requested
    assert session.state is CaptureState.AWAITING_ACCEPT


def test_consumer_not_called_until_accept_when_review_on(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session(review_captures=True)
    consumer = MagicMock()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=consumer,
    )
    controller.start()
    controller.handle_frame(_specimen(), now=1.0, captured_at=NOW)
    controller.handle_frame(_specimen(), now=2.0, captured_at=NOW)

    consumer.assert_not_called()
    assert session.state is CaptureState.AWAITING_ACCEPT
    assert controller.pending_record is not None


def test_consumer_called_exactly_once_on_accept(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session(review_captures=True)
    consumer = MagicMock(return_value=SimpleNamespace(success=True))
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=consumer,
    )
    controller.handle_frame(_specimen(), now=1.0, captured_at=NOW)
    controller.handle_frame(_specimen(), now=2.0, captured_at=NOW)
    controller.accept_capture()

    consumer.assert_called_once()
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert controller.pending_record is None


def test_retry_from_awaiting_accept_via_controller(tmp_path):
    camera = FakeCamera()
    session = CaptureSession(
        SessionConfig(baseline_frames=2, stable_duration=0.0),
        mode="block",
        review_captures=True,
    )
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    session.submit_scan("51151378")
    consumer = MagicMock()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=consumer,
    )
    controller.handle_frame(_specimen(), now=1.0, captured_at=NOW)
    controller.handle_frame(_specimen(), now=2.0, captured_at=NOW)
    held_path = session.last_capture.path
    assert held_path.is_file()

    controller.retry(captured_at=NOW)

    consumer.assert_not_called()
    assert session.state is CaptureState.AWAITING_ACCEPT
    assert session.pending_block_id == "51151378"
    assert session.last_capture is not None
    assert session.last_capture.path != held_path
    assert not held_path.is_file()
    assert [call[0] for call in camera.calls].count("still") == 2


def test_flag_off_runtime_path_unchanged(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session(review_captures=False)
    consumer = MagicMock()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=consumer,
    )
    controller.handle_frame(_specimen(), now=1.0, captured_at=NOW)
    controller.handle_frame(_specimen(), now=2.0, captured_at=NOW)

    consumer.assert_called_once()
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert controller.pending_record is None


def test_pi_runtime_accept_capture_façade(tmp_path):
    import sys

    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    repo_tools = repo_root / "tools"
    if str(repo_tools) not in sys.path:
        sys.path.append(str(repo_tools))

    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        LoopbackCaptureReceiver,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    import run_pi_session

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=NOW)
    camera = FakeCamera()

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "framing_calibration.json"
            ),
        )
        publish = MagicMock(wraps=workflow.publish_scanned_block)
        workflow.publish_scanned_block = publish
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            camera,
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
            review_captures=True,
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        runtime.process_frame(_empty(), now=0.0, captured_at=NOW)
        assert runtime.scan_block("51151378").accepted
        runtime.process_frame(_specimen(), now=1.0, captured_at=NOW)
        runtime.process_frame(_specimen(), now=2.0, captured_at=NOW)

        assert runtime.capture_session.state is CaptureState.AWAITING_ACCEPT
        publish.assert_not_called()

        runtime.accept_capture()
        publish.assert_called_once()
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_REMOVAL
        runtime.close()


def test_pi_runtime_logs_capture_accepted_and_retaken(tmp_path, capsys):
    import sys

    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    repo_tools = repo_root / "tools"
    capture_dir = repo_root / "code" / "capture"
    for path in (repo_tools, capture_dir):
        if str(path) not in sys.path:
            sys.path.append(str(path))

    import run_pi_session
    from action_logger import ActionLogger
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        LoopbackCaptureReceiver,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=NOW)
    camera = FakeCamera()
    log_path = tmp_path / "actions.log"
    action_logger = ActionLogger(log_path, session_number=session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "framing_calibration.json"
            ),
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            camera,
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(
                baseline_frames=1, stable_duration=0.0
            ),
            review_captures=True,
            action_logger=action_logger,
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        runtime.process_frame(_empty(), now=0.0, captured_at=NOW)
        runtime.scan_block("51151378")
        runtime.process_frame(_specimen(), now=1.0, captured_at=NOW)
        runtime.process_frame(_specimen(), now=2.0, captured_at=NOW)
        runtime.accept_capture()
        empty = _empty()
        runtime.process_frame(empty, now=2.5, captured_at=NOW)
        runtime.process_frame(empty, now=3.0, captured_at=NOW)
        assert runtime.scan_block("87654321").accepted
        runtime.process_frame(_specimen(), now=3.5, captured_at=NOW)
        runtime.process_frame(_specimen(), now=4.0, captured_at=NOW)
        runtime.retry_capture()
        runtime.close()

    log_text = log_path.read_text(encoding="utf-8")
    assert "event=capture_accepted" in log_text
    assert "event=capture_retaken" in log_text
    saved_lines = [
        line for line in log_text.splitlines() if "event=capture_saved" in line
    ]
    assert saved_lines, "expected at least one capture_saved action log line"
    for line in saved_lines:
        assert "elapsed_ms=" in line
        assert "camera_capture_ms=" in line
        assert "publish_ms=" in line
        assert "session_accept_ms=" in line
        assert "total_capture_ms=" in line
        assert "final_file_size_bytes=" in line
        assert "capture_mode=" in line
