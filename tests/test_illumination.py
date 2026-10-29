"""Off-hardware tests for the coaxial LED illumination seam."""
from __future__ import annotations

import pytest

from illumination import (
    GpiozeroDigitalOutput,
    GpiozeroPwmOutput,
    IlluminationController,
    IlluminationError,
)


class FakeOutput:
    def __init__(self):
        self.calls: list[str] = []
        self.is_on = False

    def on(self) -> None:
        self.calls.append("on")
        self.is_on = True

    def off(self) -> None:
        self.calls.append("off")
        self.is_on = False

    def close(self) -> None:
        self.calls.append("close")


class FailingOffOutput(FakeOutput):
    """off() raises once `fail_off` is set."""

    fail_off = False

    def off(self) -> None:
        self.calls.append("off")
        if self.fail_off:
            raise OSError("GPIO off failed")
        self.is_on = False


class FakeBacklight:
    def __init__(self, pin: int = 27):
        self.pin = pin
        self.calls: list[str] = []
        self.value = 0.0

    def set_brightness(self, value: float) -> None:
        self.calls.append(f"set:{value}")
        self.value = value

    def off(self) -> None:
        self.calls.append("off")
        self.value = 0.0

    def close(self) -> None:
        self.calls.append("close")


class EventOutput(FakeOutput):
    """Digital output that logs `<label>_on` / `<label>_off` (or bare on/off)."""

    def __init__(self, events: list, label: str | None = None):
        super().__init__()
        self.events = events
        self.prefix = f"{label}_" if label else ""

    def on(self) -> None:
        super().on()
        self.events.append(f"{self.prefix}on")

    def off(self) -> None:
        super().off()
        self.events.append(f"{self.prefix}off")


class EventBacklight(FakeBacklight):
    """PWM backlight that logs `backlight_off` / `backlight_<value>`."""

    def __init__(self, events: list):
        super().__init__()
        self.events = events

    def set_brightness(self, value: float) -> None:
        super().set_brightness(value)
        self.events.append(f"backlight_{value}")

    def off(self) -> None:
        super().off()
        self.events.append("backlight_off")


class FakeTimer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


class TimerLog(list):
    def factory(self, delay, callback):
        timer = FakeTimer(delay, callback)
        self.append(timer)
        return timer


@pytest.fixture
def timers() -> TimerLog:
    return TimerLog()


def test_illumination_starts_in_auto_mode_with_output_low():
    output = FakeOutput()

    light = IlluminationController(output)

    assert light.auto_mode is True
    assert light.logical_output is False
    assert output.is_on is False
    assert output.calls == ["off"]


def test_manual_commands_take_ownership_and_safety_timer_can_turn_led_off(timers):
    light = IlluminationController(FakeOutput(), timer_factory=timers.factory)

    led_on = light.led_on()
    assert led_on.mode == "manual"
    assert led_on.auto_mode is False
    assert led_on.logical_output is True
    assert led_on.physical_output is True
    assert timers[0].started is True
    assert timers[0].delay == light.manual_timeout_s

    timers[0].callback()
    assert light.logical_output is False
    assert light.physical_output is False

    led_off = light.led_off()
    assert led_off.mode == "manual"
    assert led_off.logical_output is False
    assert timers[0].cancelled is True


def test_led_on_accepts_seconds_duration_for_manual_timer(timers):
    light = IlluminationController(FakeOutput(), timer_factory=timers.factory)

    status = light.led_on(5)
    assert status.logical_output is True
    assert timers[0].delay == 5.0

    status = light.led_on("2.5")
    assert status.logical_output is True
    assert timers[1].delay == 2.5
    assert timers[0].cancelled is True


def test_led_on_rejects_non_positive_duration():
    light = IlluminationController(FakeOutput())

    with pytest.raises(IlluminationError, match="positive"):
        light.led_on(0)
    with pytest.raises(IlluminationError, match="positive"):
        light.led_on(-1)
    with pytest.raises(IlluminationError, match="invalid LED duration"):
        light.led_on("nope")


def test_led_on_turns_preview_backlight_off_and_led_off_restores_it():
    output = FakeOutput()
    backlight = FakeBacklight()
    light = IlluminationController(
        output,
        backlight=backlight,
        backlight_brightness=0.25,
        sleep=lambda _seconds: None,
    )
    # Construction lights the preview backlight exactly once.
    assert backlight.calls == ["set:0.25"]

    light.led_on(5)
    assert output.is_on is True
    assert backlight.value == 0.0

    light.led_off()
    assert output.is_on is False
    assert backlight.value == 0.25


def test_stale_manual_timer_cannot_turn_off_new_manual_session(timers):
    light = IlluminationController(FakeOutput(), timer_factory=timers.factory)

    light.led_on()
    first_timer = timers[0]
    light.led_off()
    light.led_on()
    second_timer = timers[1]

    # A cancelled threading.Timer callback can still fire; it must not switch
    # off the newer manual session.
    first_timer.callback()

    assert light.logical_output is True
    assert light.physical_output is True
    assert second_timer.cancelled is False

    second_timer.callback()
    assert light.logical_output is False
    assert light.physical_output is False


def test_auto_mode_commands_force_low_before_enabling_or_disabling_auto():
    output = FakeOutput()
    light = IlluminationController(output)

    light.led_on()
    output.calls.clear()
    status = light.led_auto_on()
    assert status.mode == "auto"
    assert status.auto_mode is True
    assert status.logical_output is False
    assert output.calls == ["off"]

    output.calls.clear()
    status = light.led_auto_off()
    assert status.mode == "auto_off"
    assert status.auto_mode is False
    assert status.logical_output is False
    assert output.calls == ["off"]


def test_auto_off_failure_disables_automation_but_preserves_logical_state(timers):
    output = FailingOffOutput()
    light = IlluminationController(output, timer_factory=timers.factory)
    light.led_on()
    output.fail_off = True

    with pytest.raises(IlluminationError, match="GPIO off failed"):
        light.led_auto_off()

    assert light.auto_mode is False
    assert light.mode == "auto_off"
    assert light.logical_output is True
    assert light.physical_output is True


def test_auto_block_capture_brackets_still_and_records_duration():
    events: list[object] = []
    clock = iter((10.0, 10.123))
    output = EventOutput(events)
    light = IlluminationController(
        output,
        sleep=lambda seconds: events.append(("sleep", seconds)),
        clock=lambda: next(clock),
    )
    events.clear()

    result, metadata = light.capture_block(
        lambda: events.append("camera") or "saved"
    )

    assert result == "saved"
    assert events == ["on", ("sleep", 0.010), "camera", ("sleep", 0.005), "off"]
    assert metadata["illumination_auto_mode"] is True
    assert metadata["illumination_pin"] == 17
    assert metadata["illumination_lead_ms"] == 10
    assert metadata["illumination_tail_ms"] == 5
    assert metadata["illumination_led_on_duration_ms"] == 123
    assert output.is_on is False


def test_manual_or_auto_off_block_capture_does_not_switch_led():
    output = FakeOutput()
    light = IlluminationController(output)

    light.led_auto_off()
    output.calls.clear()
    result, metadata = light.capture_block(lambda: "saved")

    assert result == "saved"
    assert output.calls == []
    assert metadata["illumination_auto_mode"] is False
    assert metadata["illumination_led_on_duration_ms"] is None


def test_gpio_adapter_initializes_low_and_closes_low():
    class FakeDevice:
        def __init__(self, pin, *, initial_value):
            self.pin = pin
            self.value = initial_value
            self.calls: list[str] = []

        def on(self):
            self.calls.append("on")
            self.value = True

        def off(self):
            self.calls.append("off")
            self.value = False

        def close(self):
            self.calls.append("close")

    devices: list[FakeDevice] = []

    def factory(pin, *, initial_value):
        device = FakeDevice(pin, initial_value=initial_value)
        devices.append(device)
        return device

    output = GpiozeroDigitalOutput(device_factory=factory)

    assert output.pin == 17
    assert devices[0].value is False
    output.on()
    output.close()
    assert devices[0].calls == ["on", "off", "close"]
    assert output.is_on is False


def test_pwm_adapter_controls_duty_cycle_and_closes_low():
    class FakeDevice:
        def __init__(self, pin, *, initial_value):
            self.pin = pin
            self.value = initial_value
            self.closed = False

        def close(self):
            self.closed = True

    devices: list[FakeDevice] = []

    def factory(pin, *, initial_value):
        device = FakeDevice(pin, initial_value=initial_value)
        devices.append(device)
        return device

    output = GpiozeroPwmOutput(22, device_factory=factory)

    assert output.pin == 22
    assert output.brightness == 0.0
    assert output.is_on is False
    output.set_brightness(0.4)
    assert output.brightness == pytest.approx(0.4)
    assert output.is_on is True
    output.close()
    assert devices[0].value == 0.0
    assert devices[0].closed is True


def test_failed_block_still_always_attempts_led_off():
    output = FakeOutput()
    light = IlluminationController(output)
    output.calls.clear()

    with pytest.raises(IlluminationError, match="camera failed"):
        light.capture_block(lambda: (_ for _ in ()).throw(OSError("camera failed")))

    assert output.calls == ["on", "off"]
    assert output.is_on is False


def test_block_cleanup_failure_preserves_logical_on_state():
    output = FailingOffOutput()
    light = IlluminationController(output)
    output.calls.clear()
    output.fail_off = True

    with pytest.raises(IlluminationError, match="GPIO off failed"):
        light.capture_block(lambda: "saved")

    assert output.calls == ["on", "off"]
    assert light.logical_output is True
    assert light.physical_output is True


def test_manual_timer_off_failure_keeps_logical_state_and_surfaces_error(timers):
    output = FailingOffOutput()
    light = IlluminationController(output, timer_factory=timers.factory)
    light.led_on()
    output.fail_off = True

    with pytest.raises(IlluminationError, match="GPIO off failed"):
        timers[0].callback()

    assert light.logical_output is True
    assert light.physical_output is True
    assert "GPIO off failed" in light.status_text()


def test_driver_state_read_failure_surfaces_as_illumination_error():
    class FailingReadOutput(FakeOutput):
        @property
        def is_on(self):
            raise OSError("GPIO read failed")

        @is_on.setter
        def is_on(self, _value):
            pass

    light = IlluminationController(FailingReadOutput())

    with pytest.raises(IlluminationError, match="GPIO read failed"):
        light.status()


def test_initial_backlight_failure_aborts_controller_creation():
    backlight = FakeBacklight()

    def explode(_value: float) -> None:
        raise OSError("pwm unavailable")

    backlight.set_brightness = explode

    with pytest.raises(IlluminationError, match="pwm unavailable"):
        IlluminationController(FakeOutput(), backlight=backlight)


def test_block_capture_drops_backlight_before_led_and_restores_after():
    events: list[str] = []
    light = IlluminationController(
        EventOutput(events, "led"),
        backlight=EventBacklight(events),
        backlight_brightness=0.25,
        backlight_settle_s=0.03,
    )
    events.clear()
    slept: list[float] = []

    _, metadata = light.capture_block(
        lambda: events.append("shutter"), sleep=slept.append, clock=lambda: 0.0
    )

    assert events == [
        "backlight_off",
        "led_on",
        "shutter",
        "led_off",
        "backlight_0.25",
    ]
    assert slept[0] == 0.03
    assert metadata["backlight_pin"] == 27
    assert metadata["backlight_preview_brightness"] == 0.25
    assert metadata["backlight_settle_ms"] == 30


def test_backlight_off_failure_aborts_the_still_before_the_led_fires():
    output = FakeOutput()
    backlight = FakeBacklight()
    light = IlluminationController(output, backlight=backlight)

    def explode() -> None:
        raise OSError("pwm stuck")

    backlight.off = explode
    fired: list[str] = []

    with pytest.raises(IlluminationError):
        light.capture_block(lambda: fired.append("shutter"))

    assert fired == []
    assert output.is_on is False


def test_backlight_restore_failure_never_fails_the_capture():
    backlight = FakeBacklight()
    light = IlluminationController(FakeOutput(), backlight=backlight)

    def explode(_value: float) -> None:
        raise OSError("pwm stuck")

    backlight.set_brightness = explode

    light.capture_block(lambda: None)

    assert light.status().last_error is not None
    assert light.status().backlight_brightness == 0.0


def test_close_shuts_the_backlight_and_neutral_oblique_down():
    backlight = FakeBacklight()
    neutral = FakeOutput()
    light = IlluminationController(
        FakeOutput(), backlight=backlight, neutral_oblique=neutral
    )

    light.close()

    assert backlight.calls[-2:] == ["off", "close"]
    assert neutral.calls[-2:] == ["off", "close"]


def test_shutdown_turns_both_lights_off_without_closing_devices():
    output = FakeOutput()
    backlight = FakeBacklight()
    light = IlluminationController(output, backlight=backlight)
    light.led_on()
    light.shutdown()

    assert output.is_on is False
    assert backlight.value == 0.0
    assert "close" not in output.calls + backlight.calls


def _debug_capture(light, *, triple):
    if triple:
        light.capture_triple(
            lambda: "coax",
            lambda: "transmitted",
            lambda: "neutral",
            restore_backlight=False,
            sleep=lambda _seconds: None,
        )
    else:
        light.capture_pair(
            lambda: "transmitted",
            lambda: "point_led",
            restore_backlight=False,
            sleep=lambda _seconds: None,
        )


@pytest.mark.parametrize("triple", [False, True], ids=["pair", "triple"])
def test_debug_capture_restores_the_preview_backlight(triple):
    # restore_backlight=False means "do not force the preview on", not "go
    # dark". A dark preview disables automatic capture until restart.
    output = FakeOutput()
    backlight = FakeBacklight()
    neutral = FakeOutput()
    light = IlluminationController(
        output, backlight=backlight, neutral_oblique=neutral
    )

    _debug_capture(light, triple=triple)

    assert output.is_on is False
    assert neutral.is_on is False
    assert backlight.value == pytest.approx(0.5)


def test_debug_pair_honours_an_operator_backlight_off():
    output = FakeOutput()
    backlight = FakeBacklight()
    light = IlluminationController(output, backlight=backlight)
    light.backlight_off()

    _debug_capture(light, triple=False)

    assert output.is_on is False
    assert backlight.value == 0.0


def test_backlight_console_controls_set_the_level_captures_restore():
    backlight = FakeBacklight()
    light = IlluminationController(FakeOutput(), backlight=backlight)

    light.set_backlight_level(40)
    assert backlight.value == 0.4

    light.backlight_off()
    assert backlight.value == 0.0
    # The dark level must survive a capture, not spring back to preview.
    light.capture_block(lambda: None)
    assert backlight.value == 0.0

    status = light.backlight_on()
    assert status.backlight_pin == 27
    assert backlight.value == pytest.approx(0.4)


def test_backlight_level_zero_turns_off_without_erasing_selected_level():
    backlight = FakeBacklight()
    light = IlluminationController(FakeOutput(), backlight=backlight)

    light.set_backlight_level(60)
    light.set_backlight_level(0)

    assert backlight.value == 0.0
    assert light.backlight_on().backlight_brightness == pytest.approx(0.6)
    assert backlight.value == pytest.approx(0.6)


def test_backlight_level_rejects_out_of_range_and_missing_hardware():
    light = IlluminationController(FakeOutput(), backlight=FakeBacklight())

    with pytest.raises(IlluminationError):
        light.set_backlight_level(140)
    with pytest.raises(IlluminationError):
        light.set_backlight_level("bright")

    no_backlight = IlluminationController(FakeOutput())
    with pytest.raises(IlluminationError):
        no_backlight.set_backlight_level(25)


def test_neutral_commands_toggle_new_output_and_report_status():
    neutral = FakeOutput()
    light = IlluminationController(FakeOutput(), neutral_oblique=neutral)
    neutral.calls.clear()

    on_status = light.neutral_on()

    assert on_status.neutral_oblique_pin == 22
    assert on_status.neutral_oblique_logical_output is True
    assert on_status.neutral_oblique_physical_output is True
    assert neutral.calls == ["on"]

    off_status = light.neutral_off()

    assert off_status.neutral_oblique_logical_output is False
    assert off_status.neutral_oblique_physical_output is False
    assert neutral.calls == ["on", "off"]


def test_neutral_commands_reject_missing_hardware():
    light = IlluminationController(FakeOutput())

    with pytest.raises(IlluminationError, match="neutral/oblique"):
        light.neutral_on()
    with pytest.raises(IlluminationError, match="neutral/oblique"):
        light.neutral_off()


def test_neutral_level_controls_pwm_and_on_restores_selected_level():
    neutral = FakeBacklight(pin=22)
    light = IlluminationController(FakeOutput(), neutral_oblique=neutral)

    # With no level selected yet, neutral_on defaults to full brightness.
    light.neutral_on()
    assert neutral.value == pytest.approx(1.0)

    status = light.set_neutral_level(40)

    assert neutral.value == pytest.approx(0.4)
    assert status.neutral_oblique_brightness == pytest.approx(0.4)
    assert status.neutral_oblique_logical_output is True

    light.neutral_off()
    assert neutral.value == 0.0
    assert light.status().neutral_oblique_brightness == 0.0

    light.neutral_on()
    assert neutral.value == pytest.approx(0.4)

    light.set_neutral_level(0)
    assert neutral.value == 0.0
    assert light.status().neutral_oblique_logical_output is False

    light.neutral_on()
    assert neutral.value == pytest.approx(0.4)


def test_neutral_level_rejects_intermediate_level_on_digital_output():
    light = IlluminationController(FakeOutput(), neutral_oblique=FakeOutput())

    with pytest.raises(IlluminationError, match="brightness control"):
        light.set_neutral_level(40)


def test_debug_pair_captures_backlight_then_point_led_and_restores_preview():
    events: list[str] = []
    output = EventOutput(events, "led")
    light = IlluminationController(
        output,
        backlight=EventBacklight(events),
        backlight_brightness=0.25,
        sleep=lambda seconds: events.append(f"sleep_{seconds}"),
    )
    events.clear()

    result = light.capture_pair(
        lambda: events.append("backlight_capture") or "transmitted",
        lambda: events.append("led_capture") or "point_led",
        clock=lambda: 0.0,
    )

    assert result[0:2] == ("transmitted", "point_led")
    assert events == [
        "led_off",
        "backlight_capture",
        "backlight_off",
        "sleep_0.03",
        "led_on",
        "sleep_0.01",
        "led_capture",
        "sleep_0.005",
        "led_off",
        "backlight_0.25",
    ]
    assert output.is_on is False


def test_triple_capture_runs_mutually_exclusive_training_sequence():
    events: list[str] = []
    output = EventOutput(events, "coax")
    neutral = EventOutput(events, "neutral")
    backlight = EventBacklight(events)
    light = IlluminationController(
        output,
        backlight=backlight,
        neutral_oblique=neutral,
        sleep=lambda seconds: events.append(f"sleep_{seconds}"),
    )
    events.clear()

    def capture_coax():
        assert output.is_on is True
        assert neutral.is_on is False
        assert backlight.value == 0.0
        return "coax"

    def capture_backlight():
        assert output.is_on is False
        assert neutral.is_on is False
        assert backlight.value == 0.5
        return "backlight"

    def capture_neutral():
        assert output.is_on is False
        assert neutral.is_on is True
        assert backlight.value == 0.0
        return "neutral"

    result = light.capture_triple(
        capture_coax,
        capture_backlight,
        capture_neutral,
        clock=lambda: 0.0,
    )

    assert result[:3] == ("coax", "backlight", "neutral")
    metadata = result[3]
    assert metadata["illumination_capture_mode"] == "triple"
    assert metadata["illumination_variant_sequence"] == (
        "coax,backlight,neutral_oblique"
    )
    assert metadata["neutral_oblique_pin"] == 22
    assert metadata["illumination_neutral_oblique_on_duration_ms"] == 0
    assert events == [
        "coax_off",
        "neutral_off",
        "backlight_off",
        "sleep_0.03",
        "coax_on",
        "sleep_0.01",
        "sleep_0.005",
        "coax_off",
        "backlight_0.5",
        "sleep_0.03",
        "backlight_off",
        "sleep_0.03",
        "neutral_on",
        "sleep_0.01",
        "sleep_0.005",
        "coax_off",
        "neutral_off",
        "backlight_0.5",
    ]
    assert output.is_on is False
    assert neutral.is_on is False
    assert backlight.value == 0.5


def test_triple_capture_uses_selected_neutral_pwm_level():
    neutral = FakeBacklight(pin=22)
    observed: list[float] = []
    light = IlluminationController(
        FakeOutput(),
        backlight=FakeBacklight(),
        neutral_oblique=neutral,
        neutral_oblique_brightness=0.4,
        sleep=lambda _seconds: None,
    )

    result = light.capture_triple(
        lambda: "coax",
        lambda: "backlight",
        lambda: observed.append(neutral.value) or "neutral",
        clock=lambda: 0.0,
    )

    assert result[:3] == ("coax", "backlight", "neutral")
    assert observed == [pytest.approx(0.4)]
    assert result[3]["neutral_oblique_brightness"] == pytest.approx(0.4)
    assert neutral.value == 0.0


def test_triple_capture_requires_neutral_oblique_output():
    light = IlluminationController(FakeOutput(), backlight=FakeBacklight())

    with pytest.raises(IlluminationError, match="neutral/oblique"):
        light.capture_triple(lambda: None, lambda: None, lambda: None)
