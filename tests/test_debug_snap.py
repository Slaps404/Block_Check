"""Debug snap: POST /debug/snap + HttpCaptureClient + PiCaptureRuntime.snap."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
)
from illumination import IlluminationController

from app import pi_session as run_pi_session
from capture_fakes import _FakeBacklight, _FakeOutput


STARTED_AT = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


class _FakeCamera:
    def __init__(self):
        self.still_calls = 0
        self.still_settings = []
        self.resume_calls = 0

    def start_preview(self, **_kwargs):
        pass

    def preview_frame(self):
        return np.full((80, 120, 3), 180, dtype=np.uint8)

    def capture_still(self, path, **kwargs):
        self.still_calls += 1
        self.still_settings.append(kwargs["settings"])
        assert cv2.imwrite(str(path), np.full((100, 120), 90, dtype=np.uint8))

    def resume_preview(self):
        self.resume_calls += 1

    def close(self):
        pass


def _illumination(**kwargs):
    kwargs.setdefault("backlight", _FakeBacklight())
    return IlluminationController(
        kwargs.pop("output", None) or _FakeOutput(),
        sleep=lambda _seconds: None,
        **kwargs,
    )


@contextmanager
def _runtime(
    tmp_path, camera, *, illumination=None, open_debug_snaps=False, **runtime_kwargs
):
    """Started PiCaptureRuntime posting snaps to <tmp>/pi_captures."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(
        store,
        debug_snap_dir=tmp_path / "pi_captures",
        open_debug_snaps=open_debug_snaps,
    ) as receiver:
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
            capture_root=tmp_path / "captures",
            illumination=illumination or _illumination(),
            **runtime_kwargs,
        )
        runtime.start(background=False)
        try:
            yield runtime
        finally:
            runtime.close()


def _leftover_temp_snaps(tmp_path):
    return list((tmp_path / "captures").rglob("debug-snap-*.png"))


@pytest.mark.parametrize("led_already_on", [False, True])
def test_runtime_snap_takes_one_still_with_led_on(tmp_path, led_already_on):
    camera = _FakeCamera()
    output = _FakeOutput()
    illumination = _illumination(output=output)
    if led_already_on:
        illumination.led_on(30)

    with _runtime(tmp_path, camera, illumination=illumination) as runtime:
        message = runtime.snap()
        led_on_after_snap = output.is_on

    assert message.startswith("Saved:")
    saved = Path(message.split(":", 1)[1].strip())
    assert saved.exists()
    assert "_still_" in saved.name
    assert camera.still_calls == 1
    # Defaults the LED on, and leaves an already-on LED alone.
    assert led_on_after_snap is True
    assert _leftover_temp_snaps(tmp_path) == []


def test_kiosk_snap_quiet_saves_without_opening_but_console_snap_opens(
    tmp_path, monkeypatch
):
    from session import outbox_transport
    from session.console import dispatch

    opened = []
    monkeypatch.setattr(outbox_transport, "open_saved_image", opened.append)

    with _runtime(tmp_path, _FakeCamera(), open_debug_snaps=True) as runtime:
        quiet = dispatch(runtime, "snap_quiet", "block 123 | note")
        assert opened == []
        dispatch(runtime, "snap")

    saved = Path(quiet.split(":", 1)[1].strip())
    assert saved.with_suffix(".txt").read_text(encoding="utf-8") == "block 123 | note\n"
    assert len(opened) == 1


def _neutral():
    neutral = _FakeBacklight()
    neutral.pin = 22
    return neutral


def test_runtime_snap_3_uploads_deletes_pi_temp_and_resumes(tmp_path):
    dest = tmp_path / "pi_captures"
    camera = _FakeCamera()
    backlight = _FakeBacklight()
    neutral = _neutral()
    illumination = _illumination(backlight=backlight, neutral_oblique=neutral)

    with _runtime(tmp_path, camera, illumination=illumination) as runtime:
        message = runtime.snap_3()
        snap_backlight_value = backlight.value

    assert message.startswith("Saved backlight:")
    saved_lines = message.splitlines()
    backlight_path = Path(saved_lines[0].split(":", 1)[1].strip())
    point_led_path = Path(saved_lines[1].split(":", 1)[1].strip())
    assert backlight_path.exists()
    assert point_led_path.exists()
    assert backlight_path.parent == dest
    assert point_led_path.parent == dest
    assert camera.still_calls == 2
    assert camera.resume_calls >= 1
    # The preview backlight is the only light detection sees; a snap must hand
    # it back lit or automatic capture stays blind afterwards.
    assert snap_backlight_value == 0.5
    assert neutral.value == 0.0
    assert len(list(dest.glob("snap_*_backlight_*.png"))) == 1
    assert len(list(dest.glob("snap_*_point_led_*.png"))) == 1
    assert _leftover_temp_snaps(tmp_path) == []


def _write_deskew_config(path, source_size=(120, 100)):
    path.write_text(json.dumps({
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "output_size": [60, 40],
        "source_size": list(source_size),
        "px_per_mm": 50.0,
        "grid": "20x20 @ 0.8mm",
        "matched": 400,
        "expected": 400,
        "residual_px": {"median": 0.5, "p95": 1.2, "max": 2.0},
        "source_image": "board.png",
        "fitted_at": "2026-10-13T00:00:00Z",
    }))
    return path


def test_runtime_deskew_without_config_does_not_capture(tmp_path):
    camera = _FakeCamera()

    with _runtime(
        tmp_path, camera, deskew_config_path=tmp_path / "missing.json"
    ) as runtime:
        with pytest.raises(Exception, match="grid-deskew"):
            runtime.deskew()

    assert camera.still_calls == 0


def test_runtime_deskew_refuses_wrong_resolution_but_keeps_raw(tmp_path):
    camera = _FakeCamera()
    config_path = _write_deskew_config(
        tmp_path / "deskew_grid.json", source_size=(4056, 3040)
    )

    with _runtime(tmp_path, camera, deskew_config_path=config_path) as runtime:
        with pytest.raises(Exception, match="4056x3040"):
            runtime.deskew()

    assert list((tmp_path / "pi_captures").glob("snap_*_deskewed_*.png")) == []


def test_kiosk_deskew_quiet_saves_both_with_note_without_opening(
    tmp_path, monkeypatch
):
    from session import outbox_transport
    from session.console import dispatch

    opened = []
    monkeypatch.setattr(outbox_transport, "open_saved_image", opened.append)
    config_path = _write_deskew_config(tmp_path / "deskew_grid.json")

    with _runtime(
        tmp_path, _FakeCamera(), open_debug_snaps=True, deskew_config_path=config_path
    ) as runtime:
        dispatch(runtime, "deskew_quiet", "block 123")

    captures = tmp_path / "pi_captures"
    raw = list(captures.glob("snap_*_raw_*.png"))
    warped = list(captures.glob("snap_*_deskewed_*.png"))
    assert len(raw) == 1 and len(warped) == 1
    for png in raw + warped:
        assert png.with_suffix(".txt").read_text(encoding="utf-8") == "block 123\n"
    assert opened == []
