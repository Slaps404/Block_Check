"""Scanner-keyed block inventory behavior (issue #84)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from capture_session import (
    CaptureMode,
    CaptureResult,
    CaptureSession,
    CaptureState,
    SessionConfig,
)
from capture_storage import ValidatedStill
from constants import CAPTURE_DIMENSIONS


def _empty() -> np.ndarray:
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _block() -> np.ndarray:
    frame = _empty()
    cv2.rectangle(frame, (35, 25), (85, 55), (50, 50, 50), -1)
    return frame


@pytest.fixture(scope="module")
def valid_png(tmp_path_factory):
    path = tmp_path_factory.mktemp("capture") / "capture.png"
    assert cv2.imwrite(str(path), np.zeros((3040, 4056), dtype=np.uint8))
    return path


def _baseline(session: CaptureSession) -> None:
    session.confirm_empty()
    for index in range(session.config.baseline_frames):
        session.accept_frame(_empty(), now=index * 0.1)


def _request(session: CaptureSession) -> None:
    session.accept_frame(_block(), now=3.0)
    session.accept_frame(_block(), now=3.1)
    assert session.accept_frame(_block(), now=4.0).capture_requested


def test_block_mode_starts_awaiting_baseline_then_waits_for_scan():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.BLOCK
    )
    assert session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    assert session.baseline is None

    _baseline(session)

    assert session.state is CaptureState.WAITING_FOR_SCAN
    assert session.baseline is not None


def test_block_mode_waits_for_exactly_eight_numeric_digits():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.BLOCK
    )
    _baseline(session)
    assert session.state is CaptureState.WAITING_FOR_SCAN

    for value in ("", "1234567", "123456789", "12A45678", "１２３４５６７８"):
        result = session.submit_scan(value)
        assert not result.accepted
        assert "eight numeric digits" in result.message

    accepted = session.submit_scan("51151378")
    assert accepted.accepted
    assert session.pending_block_id == "51151378"
    assert session.state is CaptureState.EMPTY


def test_block_placement_is_ignored_until_scan_is_pending():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode="block"
    )
    _baseline(session)

    session.accept_frame(_block(), now=1.0)

    assert session.state is CaptureState.WAITING_FOR_SCAN


def test_one_pending_scan_cannot_be_replaced():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode="block"
    )
    _baseline(session)
    assert session.submit_scan("51151378").accepted

    replacement = session.submit_scan("87654321")

    assert not replacement.accepted
    assert session.pending_block_id == "51151378"


def test_capture_failure_and_retry_preserve_pending_block_id():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.BLOCK
    )
    _baseline(session)
    session.submit_scan("51151378")
    _request(session)

    session.accept_capture_result(CaptureResult.failure("camera failed"))
    assert session.state is CaptureState.CAPTURE_ERROR
    assert session.pending_block_id == "51151378"

    retry = session.retry_capture()
    assert retry.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED
    assert session.pending_block_id == "51151378"


def test_success_consumes_pending_id_but_retains_it_in_record(valid_png):
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.BLOCK
    )
    _baseline(session)
    session.submit_scan("51151378")
    _request(session)

    session.accept_capture_result(
        CaptureResult.success(
            valid_png,
            metadata={"counter": 4, "role": "block", "block_id": "51151378"},
        )
    )

    assert session.pending_block_id is None
    assert session.last_capture.block_id == "51151378"
    assert session.last_capture.metadata["counter"] == 4
    assert session.state is CaptureState.WAITING_FOR_REMOVAL


def test_next_block_requires_a_new_scan_after_removal(valid_png):
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode="block"
    )
    _baseline(session)
    session.submit_scan("51151378")
    _request(session)
    session.accept_capture_result(CaptureResult.success(valid_png))
    session.accept_frame(_empty(), now=5.0)
    session.accept_frame(_empty(), now=5.5)

    assert session.state is CaptureState.WAITING_FOR_SCAN
    assert session.pending_block_id is None

    session.submit_scan("87654321")
    assert session.state is CaptureState.EMPTY
    assert session.baseline is not None


def test_slide_mode_never_requires_scan():
    session = CaptureSession(SessionConfig(baseline_frames=2), mode="slide")

    assert session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    assert not session.submit_scan("51151378").accepted
    _baseline(session)
    assert session.state is CaptureState.EMPTY


def test_restored_pending_block_returns_to_empty_after_baseline():
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.BLOCK
    )
    session.restore_pending_block("51151378")
    assert session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION

    _baseline(session)

    assert session.pending_block_id == "51151378"
    assert session.state is CaptureState.EMPTY


def test_retry_capture_from_reposition_slide_requests_immediate_still(valid_png):
    session = CaptureSession(
        SessionConfig(baseline_frames=2), mode=CaptureMode.SLIDE
    )
    _baseline(session)
    _request(session)
    width, height = CAPTURE_DIMENSIONS["slide"]
    session.accept_capture_result(
        CaptureResult.success(
            valid_png,
            validated=ValidatedStill(width=width, height=height, format=".png"),
        )
    )
    session.mark_slide_unreadable()
    assert session.state is CaptureState.REPOSITION_SLIDE

    retry = session.retry_capture()

    assert retry.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED
    assert any(event.kind == "retry" for event in session.drain_events())


def test_retry_capture_raises_outside_error_or_reposition():
    session = CaptureSession(mode=CaptureMode.SLIDE)
    with pytest.raises(RuntimeError, match="Retry is only valid"):
        session.retry_capture()
