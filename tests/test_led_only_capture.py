"""LED-only placement must work against a dark, fixed background."""
import numpy as np

from capture_session import CaptureMode, CaptureSession, CaptureState


def frame(value=2, block=False):
    image = np.full((80, 100, 3), value, dtype=np.uint8)
    if block:
        image[25:55, 35:65] = 90
    return image


def session():
    capture = CaptureSession(mode=CaptureMode.BLOCK, led_only=True)
    capture.install_locked_baseline(frame(), noise_floor=8)
    return capture


def qualify(capture):
    for now in (0, 0.1, 0.7):
        result = capture.accept_frame(frame(block=True), now=now)
        assert not result.capture_requested
    assert capture.state is CaptureState.REMOVE_CALIBRATION_BLOCK
    for now in (0.8, 1.4):
        capture.accept_frame(frame(), now=now)
    assert capture.state is CaptureState.WAITING_FOR_SCAN


def test_dark_background_requires_place_and_remove_qualification():
    capture = session()
    assert capture.state is CaptureState.PLACE_CALIBRATION_BLOCK
    for now in (0, 1, 2):
        assert capture.accept_frame(frame(), now=now).presence_score == 0
    qualify(capture)
    assert capture.baseline.mean() == 2


def test_noise_does_not_trigger_but_stationary_block_does():
    capture = session()
    qualify(capture)
    assert capture.submit_scan("12345678").accepted
    rng = np.random.default_rng(42)
    for now in (2, 2.1, 2.8):
        noisy = rng.integers(0, 6, size=(80, 100, 3), dtype=np.uint8)
        result = capture.accept_frame(noisy, now=now)
        assert result.presence_score == 0
        assert not result.capture_requested
    for now in (3, 3.1, 3.7):
        result = capture.accept_frame(frame(block=True), now=now)
    assert result.capture_requested
    assert capture.baseline.mean() == 2


def test_dark_baseline_never_uses_brightness_normalization():
    capture = session()
    capture._normalize = lambda _: (_ for _ in ()).throw(AssertionError())
    qualify(capture)


def test_led_mode_requires_normal_block_removal():
    import pytest
    with pytest.raises(ValueError):
        CaptureSession(mode="slide", led_only=True)
    with pytest.raises(ValueError):
        CaptureSession(mode="block", led_only=True, skip_removal_wait=True)
    with pytest.raises(ValueError):
        CaptureSession(mode="block", led_only=True, match_baseline_as_present=True)


def test_led_only_cannot_bypass_locked_baseline_qualification():
    import pytest
    from capture_session import SessionConfig
    capture = CaptureSession(SessionConfig(baseline_frames=1), mode="block", led_only=True)
    capture.confirm_empty()
    with pytest.raises(RuntimeError, match="locked camera"):
        capture.accept_frame(frame(), now=0)
