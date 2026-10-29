"""PiCaptureRuntime illumination seams: shutdown, calibration light, GPIO, main()."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from app import pi_session as run_pi_session
from capture_session import CaptureState
import illumination
from illumination import IlluminationController
from session.console import dispatch


class FakeOutput:
    def __init__(self):
        self.calls: list[str] = []
        self.is_on = False

    def on(self):
        self.calls.append("on")
        self.is_on = True

    def off(self):
        self.calls.append("off")
        self.is_on = False

    def close(self):
        self.calls.append("close")


class FakeBacklight:
    pin = 27

    def __init__(self):
        self.value = 0.0

    def set_brightness(self, value):
        self.value = value

    def off(self):
        self.value = 0.0

    def close(self):
        self.value = 0.0


class FakeTimer:
    def __init__(self, _delay, _callback):
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def test_runtime_close_cancels_manual_timer_and_forces_led_low():
    output = FakeOutput()
    timers: list[FakeTimer] = []

    def timer_factory(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    light = IlluminationController(output, timer_factory=timer_factory)
    output.calls.clear()
    light.led_on()

    runtime = run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)
    runtime._stop = Event()
    runtime._thread = None
    runtime.camera = SimpleNamespace(close=lambda: None)
    runtime.illumination = light
    runtime.profile = False
    runtime.motion_curve_writer = None

    runtime.close()

    assert timers[0].cancelled is True
    assert output.calls == ["on", "off", "close"]
    assert output.is_on is False


class _CalibrationWorkflow:
    """Minimal workflow seam that records the light state during calibration."""

    def __init__(self, backlight):
        self._backlight = backlight
        self.backlight_during_calibration = None
        self._calibrated = False

    def snapshot(self):
        return SimpleNamespace(phase="blocks")

    def awaiting_capture_blocks(self):
        return ()

    def prepare_empty_backlight(self, _mode):
        self.backlight_during_calibration = self._backlight.value
        self._calibrated = True

    def baseline_for(self, _mode):
        # Before calibration there is no stored baseline; that is the state the
        # operator confirms from.
        if not self._calibrated:
            raise RuntimeError("no baseline")
        return np.full((20, 20, 3), 200, dtype=np.uint8)


@pytest.mark.parametrize("operator_turned_backlight_off", [False, True])
def test_confirm_empty_lights_the_backlight_before_calibrating(
    tmp_path, operator_turned_backlight_off
):
    # The baseline must be built under the same light the detector will see;
    # calibrating dark locks a baseline that can detect nothing.
    backlight = FakeBacklight()
    light = IlluminationController(FakeOutput(), backlight=backlight)
    if operator_turned_backlight_off:
        light.backlight_off()
    workflow = _CalibrationWorkflow(backlight)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        SimpleNamespace(close=lambda: None),
        capture_root=tmp_path,
        illumination=light,
    )

    runtime.confirm_empty()

    assert workflow.backlight_during_calibration == pytest.approx(0.5)
    # And it stays lit afterwards, so Place Block can see the specimen arrive.
    assert backlight.value == pytest.approx(0.5)
    runtime.close()


def _led_runtime(tmp_path):
    backlight = FakeBacklight()
    output = FakeOutput()
    light = IlluminationController(
        output, backlight=backlight, backlight_brightness=0,
        timer_factory=FakeTimer,
    )

    class Workflow(_CalibrationWorkflow):
        capture_paused = False

        def baseline_for(self, mode):
            super().baseline_for(mode)
            return np.full((80, 100, 3), 2, dtype=np.uint8)

        def scan_block(self, block_id):
            return SimpleNamespace(accepted=True)

    runtime = run_pi_session.PiCaptureRuntime(
        Workflow(backlight), SimpleNamespace(block_presence_noise_floor=8, close=lambda: None),
        capture_root=tmp_path, illumination=light, led_only_blocks=True,
    )
    runtime.confirm_empty()
    assert output.is_on
    assert backlight.value == 0
    assert runtime.capture_session.state is CaptureState.PLACE_CALIBRATION_BLOCK
    runtime._led_flush_frames = 0
    empty = np.full((80, 100, 3), 2, dtype=np.uint8)
    block = empty.copy()
    block[25:55, 35:65] = 90
    for now, image in ((0, block), (0.1, block), (0.7, block),
                       (0.8, empty), (1.4, empty)):
        runtime.process_frame(image, now=now, captured_at=datetime.now(timezone.utc))
    assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
    assert not output.is_on
    return runtime, output, backlight, empty, block


def test_led_block_cycle_turns_off_after_removal_and_on_after_scan(tmp_path):
    runtime, output, backlight, empty, block = _led_runtime(tmp_path)
    try:
        runtime.scan_block("12345678")
        assert output.is_on
        assert backlight.value == 0
        assert runtime._led_flush_frames == 4
        # No dark/off frame from before the scan enters placement detection.
        for now in (2, 2.1, 2.2, 2.3):
            runtime.process_frame(empty, now=now, captured_at=datetime.now(timezone.utc))
        assert runtime.capture_session.state is CaptureState.EMPTY
        runtime.capture_session.state = CaptureState.WAITING_FOR_REMOVAL
        for now, image in ((3, block), (4, empty), (4.6, empty)):
            runtime.process_frame(image, now=now, captured_at=datetime.now(timezone.utc))
        assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN
        assert not output.is_on
        assert backlight.value == 0
    finally:
        runtime.close()


def test_led_timeout_invalidates_detection_and_retry_requalifies(tmp_path):
    runtime, output, backlight, empty, _ = _led_runtime(tmp_path)
    try:
        runtime.scan_block("12345678")
        output.off()  # The hardware timeout's observable effect.
        runtime.process_frame(empty, now=2, captured_at=datetime.now(timezone.utc))
        assert runtime.capture_session.state is CaptureState.CALIBRATION_FAILED
        assert runtime.capture_session.baseline is None
        runtime.confirm_empty()
        assert output.is_on
        assert backlight.value == 0
        assert runtime.capture_session.state is CaptureState.PLACE_CALIBRATION_BLOCK
        assert runtime.capture_session.pending_block_id == "12345678"
    finally:
        runtime.close()


def test_led_calibration_exception_forces_light_off(tmp_path):
    runtime, output, _, _, _ = _led_runtime(tmp_path)
    try:
        runtime.capture_session.mark_calibration_failed()

        def broken(_mode):
            raise RuntimeError("camera disconnected")

        runtime.workflow.prepare_empty_backlight = broken
        with pytest.raises(RuntimeError, match="disconnected"):
            runtime.confirm_empty()
        assert not output.is_on
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("include_neutral_oblique", "expected_pwm_pins"), [(False, []), (True, [22])],
    ids=["default", "triple-capture"],
)
def test_pi_illumination_factory_adds_gpio22_only_for_triple_capture(
    monkeypatch, include_neutral_oblique, expected_pwm_pins
):
    pins: list[int] = []
    pwm_pins: list[int] = []

    class Adapter(FakeOutput):
        def __init__(self, pin=17):
            super().__init__()
            self.pin = pin
            pins.append(pin)

    class PwmAdapter(FakeBacklight):
        def __init__(self, pin=27):
            super().__init__()
            self.pin = pin
            pwm_pins.append(pin)

    monkeypatch.setattr(illumination, "GpiozeroDigitalOutput", Adapter)
    monkeypatch.setattr(illumination, "GpiozeroPwmBacklight", FakeBacklight)
    monkeypatch.setattr(illumination, "GpiozeroPwmOutput", PwmAdapter)

    light = run_pi_session._build_illumination(
        include_neutral_oblique=include_neutral_oblique
    )

    assert light.auto_mode is True
    assert light.pin == 17
    assert pins == [17]
    assert pwm_pins == expected_pwm_pins
    assert light.neutral_oblique_available is include_neutral_oblique
    if include_neutral_oblique:
        assert light.neutral_oblique_pin == 22
    light.close()


def test_runtime_uploads_triple_sidecars_and_marks_manifest(tmp_path):
    sidecar_paths = {
        name: tmp_path / f"{name}.png"
        for name in ("coax", "backlight", "neutral_oblique")
    }
    for path in sidecar_paths.values():
        path.write_bytes(b"png")
    manifest_path = tmp_path / "triple.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variants": {
                    name: {"path": str(path), "state": "captured"}
                    for name, path in sidecar_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    uploaded: list[tuple[Path, str]] = []

    def debug_snap(path, *, label, open_image):
        uploaded.append((path, label))
        assert open_image is False
        return tmp_path / f"remote-{label}.png"

    runtime = run_pi_session.PiCaptureRuntime.__new__(
        run_pi_session.PiCaptureRuntime
    )
    runtime.workflow = SimpleNamespace(
        transport=SimpleNamespace(debug_snap=debug_snap)
    )
    record = SimpleNamespace(
        block_id="51151378",
        metadata={"triple_capture_manifest": str(manifest_path)},
    )

    runtime._upload_triple_variants(record)

    assert uploaded == [
        (sidecar_paths["backlight"], "block_51151378_backlight"),
        (sidecar_paths["neutral_oblique"], "block_51151378_neutral_oblique"),
    ]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("backlight", "neutral_oblique"):
        assert payload["variants"][name]["state"] == "uploaded"
        assert payload["variants"][name]["remote_path"]


def test_main_builds_neutral_gpio_and_runtime_in_triple_mode(monkeypatch, startup):
    _camera, illumination = startup
    built: dict[str, object] = {}

    def build_illumination(**kwargs):
        built.update(kwargs)
        return illumination

    monkeypatch.setattr(run_pi_session, "_build_illumination", build_illumination)

    class Runtime:
        def __init__(self, _workflow, _camera, **kwargs):
            built["runtime"] = kwargs

        def start(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(run_pi_session, "PiCaptureRuntime", Runtime)
    monkeypatch.setattr(run_pi_session, "_interactive_loop", lambda _runtime: None)

    assert _run_main("--triple-capture") == 0
    assert built["include_neutral_oblique"] is True
    assert built["runtime"]["triple_capture"] is True


class _Recorder:
    """Runtime stand-in that records every method call dispatch makes."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args)) or name


@pytest.mark.parametrize(
    ("line", "expected_call"),
    [
        ("backlight_on", ("backlight_on", ())),
        ("backlight_off", ("backlight_off", ())),
        ("led_on 12", ("led_on", ("12",))),
        ("backlight_level 40", ("set_backlight_level", ("40",))),
        ("neutral_level 60", ("set_neutral_level", ("60",))),
    ],
)
def test_light_console_commands_dispatch_to_runtime_with_arguments(line, expected_call):
    runtime = _Recorder()

    command, *args = line.split()
    assert dispatch(runtime, command, *args) == expected_call[0]
    assert runtime.calls == [expected_call]


def test_unavailable_led_commands_are_reported_cleanly():
    runtime = run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)
    runtime.illumination = None

    assert run_pi_session.run_one_command(runtime, "led_status") == (
        "LED status: unavailable"
    )
    for command in ("led_auto_on", "led_auto_off", "led_on", "led_off"):
        assert run_pi_session.run_one_command(runtime, command) == (
            "Rejected: LED control is unavailable"
        )
    for command in ("neutral_on", "neutral_off"):
        assert run_pi_session.run_one_command(runtime, command) == (
            "Rejected: neutral/oblique light control is unavailable"
        )
    assert run_pi_session.run_one_command(runtime, "neutral_level 40") == (
        "Rejected: neutral/oblique light control is unavailable"
    )


def test_runtime_neutral_commands_delegate_to_illumination():
    neutral = FakeBacklight()
    neutral.pin = 22
    light = IlluminationController(
        FakeOutput(), neutral_oblique=neutral
    )
    runtime = run_pi_session.PiCaptureRuntime.__new__(
        run_pi_session.PiCaptureRuntime
    )
    runtime.illumination = light
    runtime.controller = SimpleNamespace(set_preview_variant=lambda _variant: None)

    assert runtime.set_neutral_level(40).neutral_oblique_brightness == 0.4
    assert runtime.neutral_on().neutral_oblique_logical_output is True
    assert runtime.neutral_off().neutral_oblique_logical_output is False


def test_pi_illumination_factory_surfaces_initialization_failure(monkeypatch):
    def failing_adapter():
        raise RuntimeError("pin factory unavailable")

    monkeypatch.setattr(illumination, "GpiozeroDigitalOutput", failing_adapter)
    monkeypatch.setattr(illumination, "GpiozeroPwmBacklight", FakeBacklight)

    with pytest.raises(RuntimeError, match="pin factory unavailable"):
        run_pi_session._build_illumination()


class _TrackedClose:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


@pytest.fixture
def startup(monkeypatch):
    """Patch main()'s startup seams; yields the (camera, illumination) fakes."""
    camera = _TrackedClose()
    illumination = _TrackedClose()

    class Store:
        def resume_session(self, number):
            return SimpleNamespace(number=number)

    monkeypatch.setattr(run_pi_session, "RemoteProcessingStore", lambda _url: Store())
    monkeypatch.setattr(run_pi_session, "HttpCaptureClient", lambda _url: object())
    monkeypatch.setattr(run_pi_session, "PiOutbox", lambda _root: object())
    monkeypatch.setattr(
        run_pi_session, "FramingCalibrationStore", lambda _path: object()
    )
    monkeypatch.setattr(
        run_pi_session, "SessionWorkflow", lambda **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(run_pi_session, "_build_illumination", lambda: illumination)
    monkeypatch.setattr(run_pi_session, "ActionLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_pi_session, "install_camera_release_signals", lambda: ())

    import picamera2_adapter

    monkeypatch.setattr(picamera2_adapter, "Picamera2Adapter", lambda: camera)
    return camera, illumination


def _run_main(*extra_args):
    return run_pi_session.main(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1", *extra_args]
    )


def test_main_runtime_construction_failure_closes_camera_and_illumination(
    monkeypatch, capsys, startup
):
    camera, illumination = startup

    class FailingRuntime:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("runtime construction failed")

    monkeypatch.setattr(run_pi_session, "PiCaptureRuntime", FailingRuntime)

    result = _run_main()

    assert result == 2
    assert camera.close_calls == 1
    assert illumination.close_calls == 1
    assert "runtime construction failed" in capsys.readouterr().err


def test_main_gpio_initialization_failure_closes_camera(monkeypatch, capsys, startup):
    camera, illumination = startup

    def failing_illumination():
        raise RuntimeError("gpiozero pin factory unavailable")

    monkeypatch.setattr(run_pi_session, "_build_illumination", failing_illumination)

    result = _run_main()

    assert result == 2
    assert camera.close_calls == 1
    assert illumination.close_calls == 0
    assert "gpiozero pin factory unavailable" in capsys.readouterr().err


def test_main_start_failure_cleanup_does_not_mask_original_error(monkeypatch, capsys, startup):
    camera, illumination = startup

    class FailingStartRuntime:
        def __init__(self, _workflow, camera, *, illumination, **_kwargs):
            self.camera = camera
            self.illumination = illumination

        def start(self):
            raise RuntimeError("runtime start failed")

        def close(self):
            self.illumination.close()
            self.camera.close()
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(run_pi_session, "PiCaptureRuntime", FailingStartRuntime)

    result = _run_main()

    assert result == 2
    assert camera.close_calls == 1
    assert illumination.close_calls == 1
    stderr = capsys.readouterr().err
    assert "runtime start failed" in stderr
    assert "cleanup failed" not in stderr


def test_main_kiosk_start_failure_closes_runtime_resources(monkeypatch, capsys, startup):
    camera, illumination = startup

    class Runtime:
        def __init__(self, _workflow, camera, *, illumination, **_kwargs):
            self.camera = camera
            self.illumination = illumination

        def start(self):
            return None

        def close(self):
            self.illumination.close()
            self.camera.close()

    class FailingKiosk:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("kiosk start failed")

    monkeypatch.setattr(run_pi_session, "PiCaptureRuntime", Runtime)
    monkeypatch.setattr(run_pi_session, "KioskServer", FailingKiosk)

    result = _run_main("--kiosk")

    assert result == 2
    assert camera.close_calls == 1
    assert illumination.close_calls == 1
    assert "kiosk start failed" in capsys.readouterr().err
