"""Focus aid: center-ROI sharpness score + PiCaptureRuntime.focus live loop."""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from focus import center_roi_sharpness
from illumination import IlluminationController
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
)

from app import pi_session as run_pi_session
from capture_fakes import _FakeBacklight, _FakeOutput


STARTED_AT = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)


def _checkerboard(height=240, width=320, square=8):
    rows = (np.arange(height) // square)[:, None]
    cols = (np.arange(width) // square)[None, :]
    board = ((rows + cols) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _blurred(sigma):
    board = _checkerboard()
    if sigma == 0:
        return board
    return cv2.GaussianBlur(board, (0, 0), sigma)


def test_sharp_checkerboard_scores_higher_than_blurred():
    assert center_roi_sharpness(_blurred(0)) > center_roi_sharpness(_blurred(3))


def test_only_center_roi_affects_score():
    frame = _checkerboard()
    changed = frame.copy()
    changed[:40, :] = 0
    changed[:, :60] = 255
    changed[-40:, :] = 77
    assert center_roi_sharpness(changed) == center_roi_sharpness(frame)


def test_grayscale_and_color_frames_score_alike():
    color = _checkerboard()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert center_roi_sharpness(gray) == pytest.approx(center_roi_sharpness(color))


def test_roi_fraction_must_be_in_range():
    with pytest.raises(ValueError):
        center_roi_sharpness(_checkerboard(), roi_fraction=0)


class _SequenceCamera:
    """Preview frames from a list; holds the last frame once exhausted."""

    def __init__(self, frames, output):
        self.frames = list(frames)
        self.output = output
        self.calls = 0
        self.led_states: list[bool] = []

    def start_preview(self, **_kwargs):
        pass

    def preview_frame(self):
        self.led_states.append(self.output.is_on)
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame

    def set_preview_exposure(self, _exposure_us):
        pass

    def resume_preview(self):
        pass

    def close(self):
        pass


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _run_focus(tmp_path, frames, *, sleep=None, seconds=2.0):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    output = _FakeOutput()
    camera = _SequenceCamera(frames, output)
    clock = _FakeClock()
    written: list[str] = []
    with LoopbackCaptureReceiver(store, open_debug_snaps=False) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "framing_calibration.json"
            ),
        )
        illumination = IlluminationController(
            output,
            backlight=_FakeBacklight(),
            sleep=lambda _seconds: None,
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow,
            camera,
            capture_root=tmp_path / "captures",
            illumination=illumination,
        )
        runtime.start(background=False)
        message = runtime.focus(
            seconds,
            clock=clock,
            sleep=sleep or clock.sleep,
            write=written.append,
        )
        led_after = output.is_on
        runtime.close()
    return message, "".join(written), camera, led_after


def test_runtime_focus_tracks_best_trend_and_turns_led_off(tmp_path):
    sigmas = [8, 6, 4, 2, 0, 2, 4, 6, 8]
    frames = [_blurred(sigma) for sigma in sigmas]

    message, written, camera, led_after = _run_focus(tmp_path, frames)

    raw = [center_roi_sharpness(frames[min(i, len(frames) - 1)])
           for i in range(camera.calls)]
    smoothed = [np.mean(raw[max(0, i - 2): i + 1]) for i in range(len(raw))]
    best = max(smoothed)
    assert camera.calls >= 10
    assert all(camera.led_states)
    assert led_after is False
    assert "\r" in written
    assert "▲ sharper" in written
    assert "▼ softer" in written
    assert "= steady" in written
    assert f"best {best:.1f}" in written
    assert message == f"Focus done: best sharpness {best:.1f}"


def test_runtime_focus_ctrl_c_still_turns_led_off(tmp_path):
    calls = {"n": 0}

    def interrupting_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt

    message, _written, camera, led_after = _run_focus(
        tmp_path, [_blurred(0)], sleep=interrupting_sleep, seconds=30.0
    )

    assert camera.calls == 3
    assert led_after is False
    assert message.startswith("Focus done: best sharpness ")

