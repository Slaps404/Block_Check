"""Laptop-contract tests for the automatic capture session (issue #82)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from capture_session import (
    CaptureResult,
    CaptureSession,
    CaptureState,
    MotionSample,
    SessionConfig,
    summarize_motion_samples,
)
from capture_storage import ValidatedStill
from constants import CAPTURE_DIMENSIONS


def _empty(level: int = 180) -> np.ndarray:
    return np.full((100, 140, 3), level, dtype=np.uint8)


def _slide(*, x: int = 40, level: int = 180) -> np.ndarray:
    frame = _empty(level)
    body = round(level * 0.25)
    detail = round(level * 0.58)
    cv2.rectangle(frame, (x, 35), (x + 60, 65), (body,) * 3, -1)
    cv2.line(frame, (x + 8, 42), (x + 50, 58), (detail,) * 3, 3)
    return frame


def _write_configured_slide(tmp_path, name: str = "capture.png"):
    """Write a successful slide still using the active role contract."""
    width, height = CAPTURE_DIMENSIONS["slide"]
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))
    return path


def _ready_session(**overrides) -> CaptureSession:
    session = CaptureSession(SessionConfig(**overrides))
    session.confirm_empty()
    for index in range(session.config.baseline_frames):
        session.accept_frame(_empty(), now=index * 0.1)
    assert session.state is CaptureState.EMPTY
    return session


def test_session_config_production_defaults_are_explicit_policy():
    config = SessionConfig()

    assert config.baseline_frames == 20
    assert config.presence_threshold == 0.04
    assert config.motion_threshold == 0.02
    assert config.stable_duration == 0.5
    assert config.removal_duration == 0.5


def test_baseline_requires_confirmation_and_averages_startup_frames():
    session = CaptureSession(SessionConfig(baseline_frames=20))

    session.accept_frame(_empty(), now=0.0)
    assert session.baseline is None

    session.confirm_empty()
    for index in range(19):
        session.accept_frame(_empty(170 + index % 3), now=index * 0.1)
    assert session.state is CaptureState.BUILDING_BASELINE

    session.accept_frame(_empty(171), now=1.9)
    assert session.state is CaptureState.EMPTY
    assert session.baseline is not None


def test_locked_baseline_is_immediately_ready_without_second_collection():
    session = CaptureSession(SessionConfig(), mode="block")

    session.install_locked_baseline(_empty(220))
    session.confirm_empty()

    assert session.state is CaptureState.WAITING_FOR_SCAN
    assert session.baseline is not None


def test_global_brightness_change_is_not_presence_or_motion():
    session = _ready_session()

    result = session.accept_frame(_empty(225), now=3.0)

    assert result.presence_score < session.config.presence_threshold
    assert result.motion_score < session.config.motion_threshold
    assert session.state is CaptureState.EMPTY


def test_fast_placement_requires_two_present_frames_before_entering_settling():
    session = _ready_session()

    first = session.accept_frame(_slide(), now=3.0)
    result = session.accept_frame(_slide(), now=3.1)

    assert first.presence_score >= session.config.presence_threshold
    assert session.state is CaptureState.SETTLING
    # The stability clock begins on the first confirming frame, not the second.
    assert result.stable_elapsed == pytest.approx(0.1)
    assert session.state is CaptureState.SETTLING


def test_one_present_frame_is_ignored_before_settling(monkeypatch):
    session = _ready_session()
    # Each frame computes presence, then motion. The second frame is absent,
    # so a lone present reading must not move EMPTY into SETTLING.
    scores = iter((0.10, 0.0, 0.0, 0.0))
    monkeypatch.setattr(session, "_difference", lambda *_args: next(scores))

    session.accept_frame(_empty(), now=3.0)
    session.accept_frame(_empty(), now=3.1)

    assert session.state is CaptureState.EMPTY


def test_one_absent_frame_does_not_discard_accumulated_stability(monkeypatch):
    session = _ready_session()
    # Two present frames enter SETTLING. A lone absent frame is followed by a
    # quiet present frame after one full stable second, so it must still capture.
    scores = iter((0.10, 0.0, 0.10, 0.0, 0.0, 0.0, 0.10, 0.0))
    monkeypatch.setattr(session, "_difference", lambda *_args: next(scores))

    started_at = 3.0
    session.accept_frame(_empty(), now=started_at)
    session.accept_frame(_empty(), now=started_at + 0.1)
    absent = session.accept_frame(
        _empty(), now=started_at + session.config.stable_duration - 0.1
    )
    captured = session.accept_frame(
        _empty(), now=started_at + session.config.stable_duration
    )

    assert not absent.capture_requested
    assert captured.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED


def test_one_motion_frame_does_not_reset_accumulated_stability(monkeypatch):
    session = _ready_session()
    # The third frame is one over-threshold motion candidate. The next quiet
    # frame preserves the stable clock begun at the first present frame.
    scores = iter((0.10, 0.0, 0.10, 0.0, 0.10, 0.03, 0.10, 0.0))
    monkeypatch.setattr(session, "_difference", lambda *_args: next(scores))

    started_at = 3.0
    session.accept_frame(_empty(), now=started_at)
    session.accept_frame(_empty(), now=started_at + 0.1)
    candidate = session.accept_frame(
        _empty(), now=started_at + session.config.stable_duration - 0.1
    )
    captured = session.accept_frame(
        _empty(), now=started_at + session.config.stable_duration
    )

    assert not candidate.capture_requested
    assert captured.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED


def test_motion_resets_timer_and_requires_one_configured_stable_duration():
    session = _ready_session()
    session.accept_frame(_slide(x=35), now=3.0)
    session.accept_frame(_slide(x=35), now=3.1)
    session.accept_frame(_slide(x=45), now=3.4)
    session.accept_frame(_slide(x=55), now=3.5)

    quiet_started_at = 3.5
    before = session.accept_frame(
        _slide(x=55), now=quiet_started_at + session.config.stable_duration - 0.01
    )
    request = session.accept_frame(
        _slide(x=55), now=quiet_started_at + session.config.stable_duration
    )

    assert not before.capture_requested
    assert request.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED


def test_brightness_pulse_does_not_reset_stability():
    session = _ready_session()
    session.accept_frame(_slide(level=180), now=3.0)
    session.accept_frame(_slide(level=180), now=3.1)
    session.accept_frame(_slide(level=220), now=3.4)

    result = session.accept_frame(_slide(level=180), now=4.0)

    assert result.capture_requested


def test_success_disarms_until_configured_continuous_absence(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    requested = session.accept_frame(_slide(), now=4.0)
    assert requested.capture_requested

    path = _write_configured_slide(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))
    assert session.state is CaptureState.WAITING_FOR_REMOVAL

    assert not session.accept_frame(_slide(), now=4.2).capture_requested
    session.accept_frame(_empty(), now=4.3)
    session.accept_frame(_slide(), now=4.6)
    removal_started_at = 5.0
    session.accept_frame(_empty(), now=removal_started_at)
    session.accept_frame(
        _empty(), now=removal_started_at + session.config.removal_duration - 0.01
    )
    assert session.state is CaptureState.WAITING_FOR_REMOVAL

    session.accept_frame(
        _empty(), now=removal_started_at + session.config.removal_duration
    )
    assert session.state is CaptureState.EMPTY


def test_rearmed_session_can_request_exactly_one_new_capture(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    assert session.accept_frame(_slide(), now=4.0).capture_requested
    path = _write_configured_slide(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))
    session.accept_frame(_empty(), now=5.0)
    session.accept_frame(_empty(), now=5.5)

    session.accept_frame(_slide(), now=6.0)
    session.accept_frame(_slide(), now=6.1)
    first = session.accept_frame(_slide(), now=7.0)
    duplicate = session.accept_frame(_slide(), now=8.0)

    assert first.capture_requested
    assert not duplicate.capture_requested


def test_unreadable_slide_requires_movement_then_stability(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    assert session.accept_frame(_slide(), now=4.0).capture_requested
    path = _write_configured_slide(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))
    session.mark_slide_unreadable()

    still_present = session.accept_frame(_slide(), now=8.0)
    moved = session.accept_frame(_slide(x=50), now=9.0)
    motion_confirmed = session.accept_frame(_slide(x=55), now=9.1)
    quiet_started_at = 9.1
    before_stable = session.accept_frame(
        _slide(x=55), now=quiet_started_at + session.config.stable_duration - 0.01
    )
    stable = session.accept_frame(
        _slide(x=55), now=quiet_started_at + session.config.stable_duration
    )

    assert not still_present.capture_requested
    assert not moved.capture_requested
    assert not motion_confirmed.capture_requested
    assert not before_stable.capture_requested
    assert stable.capture_requested
    assert session.state is CaptureState.CAPTURE_REQUESTED


def test_reposition_ignores_one_motion_frame(monkeypatch):
    session = _ready_session()
    session.state = CaptureState.REPOSITION_SLIDE
    scores = iter((0.10, 0.03, 0.10, 0.0))
    monkeypatch.setattr(session, "_difference", lambda *_args: next(scores))

    session.accept_frame(_empty(), now=3.0)
    result = session.accept_frame(_empty(), now=4.0)

    assert not result.capture_requested
    assert session.state is CaptureState.REPOSITION_SLIDE


def test_unreadable_slide_removal_returns_to_waiting_for_slide(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    path = _write_configured_slide(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))
    session.mark_slide_unreadable()

    session.accept_frame(_empty(), now=5.0)
    session.accept_frame(_empty(), now=5.5)

    assert session.state is CaptureState.EMPTY
    assert session.drain_events()[-1].message == "Waiting for slide"


def test_skip_is_only_accepted_during_unreadable_slide_flow(tmp_path):
    session = _ready_session()
    try:
        session.skip_unreadable_slide()
    except RuntimeError as exc:
        assert "only valid" in str(exc)
    else:
        raise AssertionError("Skip must be rejected outside unreadable flow")

    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    path = _write_configured_slide(tmp_path)
    session.accept_capture_result(CaptureResult.success(path))
    session.mark_slide_unreadable()
    session.skip_unreadable_slide()

    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert session.drain_events()[-1].kind == "unreadable_slide_skipped"


def test_success_requires_reopenable_configured_size_png(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    wrong_size = tmp_path / "wrong.png"
    cv2.imwrite(str(wrong_size), np.zeros((30, 40, 3), dtype=np.uint8))

    session.accept_capture_result(CaptureResult.success(wrong_size))

    assert session.state is CaptureState.CAPTURE_ERROR


def test_typed_validated_still_accepts_without_redecode(tmp_path, monkeypatch):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    tiny = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny), np.zeros((4, 4, 3), dtype=np.uint8))
    width, height = CAPTURE_DIMENSIONS["slide"]
    validated = ValidatedStill(width=width, height=height, format=".png")

    decode_count = {"n": 0}
    original_imread = cv2.imread

    def counting_imread(path, flags):
        decode_count["n"] += 1
        return original_imread(path, flags)

    monkeypatch.setattr(cv2, "imread", counting_imread)
    session.accept_capture_result(CaptureResult.success(tiny, validated=validated))

    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert decode_count["n"] == 0


def test_spoofed_metadata_dimensions_do_not_bypass_validation(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    wrong_size = tmp_path / "wrong.png"
    cv2.imwrite(str(wrong_size), np.zeros((30, 40, 3), dtype=np.uint8))

    session.accept_capture_result(
        CaptureResult.success(
            wrong_size,
            metadata={
                "validated_width": CAPTURE_DIMENSIONS["slide"][0],
                "validated_height": CAPTURE_DIMENSIONS["slide"][1],
                "format": ".png",
            },
        )
    )

    assert session.state is CaptureState.CAPTURE_ERROR


# ---------------------------------------------------------------------------
# Role-aware capture dimension validation (#186, task B3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["block", "slide"])
def test_still_acceptable_dimension_check_is_role_aware(role, tmp_path):
    width, height = CAPTURE_DIMENSIONS[role]
    path = tmp_path / "capture.png"
    cv2.imwrite(str(path), np.zeros((10, 10, 3), dtype=np.uint8))

    matching = CaptureResult.success(
        path, validated=ValidatedStill(width=width, height=height, format=".png")
    )
    mismatched = CaptureResult.success(
        path, validated=ValidatedStill(width=width + 1, height=height, format=".png")
    )

    assert CaptureSession._still_acceptable(matching, role)
    assert not CaptureSession._still_acceptable(mismatched, role)


def test_slide_role_frame_at_capture_dimensions_validates(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    width, height = CAPTURE_DIMENSIONS["slide"]
    path = tmp_path / "capture.png"
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))

    session.accept_capture_result(CaptureResult.success(path))

    assert session.state is CaptureState.WAITING_FOR_REMOVAL


def test_slide_role_mismatched_dimensions_is_rejected(tmp_path):
    session = _ready_session()
    session.accept_frame(_slide(), now=3.0)
    session.accept_frame(_slide(), now=3.1)
    session.accept_frame(_slide(), now=4.0)
    width, height = CAPTURE_DIMENSIONS["slide"]
    path = tmp_path / "capture.png"
    cv2.imwrite(str(path), np.zeros((height + 10, width, 3), dtype=np.uint8))

    session.accept_capture_result(CaptureResult.success(path))

    assert session.state is CaptureState.CAPTURE_ERROR


def test_malformed_preview_frame_is_rejected():
    session = _ready_session()
    four_channel = np.zeros((100, 140, 4), dtype=np.uint8)
    try:
        session.accept_frame(four_channel, now=3.0)
    except ValueError as exc:
        assert "3-channel" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a 4-channel frame")


def test_begin_empty_backlight_setup_moves_awaiting_to_building():
    session = CaptureSession(mode="block")
    assert session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    session.begin_empty_backlight_setup()
    assert session.state is CaptureState.BUILDING_BASELINE


def test_mark_calibration_failed_from_building():
    session = CaptureSession(mode="block")
    session.begin_empty_backlight_setup()
    session.mark_calibration_failed()

    result = session.accept_frame(_empty(), now=1.0)

    assert session.state is CaptureState.CALIBRATION_FAILED
    assert not result.capture_requested


def test_begin_empty_backlight_setup_from_calibration_failed():
    session = CaptureSession(mode="block")
    session.begin_empty_backlight_setup()
    session.mark_calibration_failed()
    session.begin_empty_backlight_setup()
    assert session.state is CaptureState.BUILDING_BASELINE


def test_begin_empty_backlight_setup_rejects_idle_capture_state():
    session = CaptureSession(mode="block")
    session.begin_empty_backlight_setup()
    session.install_locked_baseline(np.full((480, 640, 3), 220, dtype=np.uint8))
    assert session.state is CaptureState.WAITING_FOR_SCAN
    try:
        session.begin_empty_backlight_setup()
    except RuntimeError as exc:
        assert "empty-backlight setup" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from idle capture state")


# ---------------------------------------------------------------------------
# `motion` console command (#169): pure motion-sample summarization
# ---------------------------------------------------------------------------


def test_summarize_motion_samples_reports_exact_min_mean_max_and_crossings():
    """Synthetic scripted motion_score sequence; a `moving` crossing counts a
    score as `>= threshold`, mirroring `CaptureSession.accept_frame`'s own
    `moving = motion_score >= self.config.motion_threshold` convention."""
    scores = (0.0, 0.01, 0.03, 0.05, 0.02, 0.10)
    threshold = 0.02

    sample = summarize_motion_samples(scores, threshold=threshold)

    assert sample.min_score == 0.0
    assert sample.max_score == 0.10
    assert sample.mean_score == pytest.approx(0.035)
    assert sample.sample_count == 6
    # >= threshold: 0.03, 0.05, 0.02, 0.10
    assert sample.threshold_crossings == 4
    assert isinstance(sample, MotionSample)
