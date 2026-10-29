"""Deep module for the PT4115 DIM output and capture illumination policy.

The module deliberately has no Pi-only imports at module import time.  The
``GpiozeroDigitalOutput`` adapter imports ``gpiozero`` only when a production
process constructs it; laptop tests inject a small fake output instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterator, Protocol, TypeVar

from constants import (
    BACKLIGHT_GPIO_PIN,
    BACKLIGHT_PREVIEW_BRIGHTNESS,
    BACKLIGHT_SETTLE_DELAY_S,
    ILLUMINATION_GPIO_PIN,
    ILLUMINATION_LEAD_DELAY_S,
    ILLUMINATION_MANUAL_TIMEOUT_S,
    ILLUMINATION_TAIL_DELAY_S,
    NEUTRAL_OBLIQUE_BRIGHTNESS,
    NEUTRAL_OBLIQUE_GPIO_PIN,
    TRIPLE_CAPTURE_VARIANTS,
)


ILLUMINATION_METADATA_KEYS: tuple[str, ...] = (
    "illumination_auto_mode",
    "illumination_mode",
    "illumination_pin",
    "illumination_lead_ms",
    "illumination_tail_ms",
    "illumination_led_on_duration_ms",
    "backlight_pin",
    "backlight_preview_brightness",
    "backlight_settle_ms",
    "neutral_oblique_pin",
    "neutral_oblique_brightness",
    "illumination_capture_mode",
    "illumination_variant_sequence",
    "illumination_neutral_oblique_on_duration_ms",
)


class DigitalOutput(Protocol):
    """The tiny physical-output seam used by the illumination controller."""

    def on(self) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


class Backlight(Protocol):
    """Dimmable preview backlight seam (PWM on its own GPIO pin)."""

    pin: int

    def set_brightness(self, value: float) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


class PwmOutput(Protocol):
    """Generic PWM light seam used by the neutral/oblique MOSFET output."""

    pin: int

    def set_brightness(self, value: float) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


class Illumination(Protocol):
    """Public controller seam consumed by the capture runtime and console."""

    auto_mode: bool

    def capture_block(
        self,
        capture: Callable[[], T],
        *,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, dict[str, object]]: ...

    def capture_pair(
        self,
        backlight_capture: Callable[[], T],
        led_capture: Callable[[], U],
        *,
        restore_backlight: bool = True,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, U, dict[str, object]]: ...

    def capture_triple(
        self,
        coax_capture: Callable[[], T],
        backlight_capture: Callable[[], U],
        neutral_oblique_capture: Callable[[], V],
        *,
        restore_backlight: bool = True,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, U, V, dict[str, object]]: ...

    def capture_metadata(self, *, led_on_duration_ms: int | None = None) -> dict[str, object]: ...

    def status(self) -> IlluminationStatus: ...

    def prepare_slide_capture(self) -> None: ...

    def led_auto_on(self) -> IlluminationStatus: ...

    def led_auto_off(self) -> IlluminationStatus: ...

    def led_on(self, seconds: float | None = None) -> IlluminationStatus: ...

    def led_off(self) -> IlluminationStatus: ...

    def backlight_on(self) -> IlluminationStatus: ...

    def backlight_off(self) -> IlluminationStatus: ...

    def set_backlight_level(self, percent: float) -> IlluminationStatus: ...

    def set_neutral_level(self, percent: float) -> IlluminationStatus: ...

    def neutral_on(self) -> IlluminationStatus: ...

    def neutral_off(self) -> IlluminationStatus: ...

    def shutdown(self) -> None: ...

    def status_text(self) -> str: ...

    def close(self) -> None: ...


class IlluminationError(RuntimeError):
    """A light-control operation failed while protecting the LED output."""


@dataclass(frozen=True)
class IlluminationStatus:
    mode: str
    auto_mode: bool
    logical_output: bool
    physical_output: bool
    pin: int
    lead_delay_ms: int
    tail_delay_ms: int
    manual_timeout_s: float
    backlight_pin: int | None = None
    backlight_brightness: float | None = None
    neutral_oblique_brightness: float | None = None
    neutral_oblique_pin: int | None = None
    neutral_oblique_logical_output: bool = False
    neutral_oblique_physical_output: bool = False
    last_error: str | None = None

    def __str__(self) -> str:
        lines = [
            "LED status:",
            f"  mode: {self.mode}",
            f"  auto mode: {'on' if self.auto_mode else 'off'}",
            f"  logical output: {'on' if self.logical_output else 'off'}",
            f"  driver-reported output: {'on' if self.physical_output else 'off'}",
            f"  pin: BCM {self.pin}",
            f"  auto lead: {self.lead_delay_ms}ms",
            f"  auto tail: {self.tail_delay_ms}ms",
            f"  manual timeout: {self.manual_timeout_s:g}s",
        ]
        if self.backlight_pin is None:
            lines.append("  backlight: unavailable")
        else:
            lines.append(
                f"  backlight: BCM {self.backlight_pin} at "
                f"{(self.backlight_brightness or 0.0) * 100:g}%"
            )
        if self.neutral_oblique_pin is None:
            lines.append("  neutral/oblique: unavailable")
        else:
            lines.append(
                f"  neutral/oblique: BCM {self.neutral_oblique_pin} at "
                f"{(self.neutral_oblique_brightness or 0.0) * 100:g}%, "
                f"{'on' if self.neutral_oblique_physical_output else 'off'}"
            )
        if self.last_error:
            lines.append(f"  error: {self.last_error}")
        return "\n".join(lines)


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class IlluminationController:
    """Own LED mode, safety timer, and software-bracketed block captures."""

    def __init__(
        self,
        output: DigitalOutput,
        *,
        backlight: Backlight | None = None,
        neutral_oblique: DigitalOutput | PwmOutput | None = None,
        backlight_brightness: float = BACKLIGHT_PREVIEW_BRIGHTNESS,
        neutral_oblique_brightness: float = NEUTRAL_OBLIQUE_BRIGHTNESS,
        backlight_settle_s: float = BACKLIGHT_SETTLE_DELAY_S,
        pin: int | None = None,
        lead_delay_s: float = ILLUMINATION_LEAD_DELAY_S,
        tail_delay_s: float = ILLUMINATION_TAIL_DELAY_S,
        manual_timeout_s: float = ILLUMINATION_MANUAL_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
    ):
        self.output = output
        self.pin = (
            getattr(output, "pin", ILLUMINATION_GPIO_PIN) if pin is None else pin
        )
        if lead_delay_s < 0 or tail_delay_s < 0:
            raise ValueError("illumination lead/tail delays cannot be negative")
        if manual_timeout_s <= 0:
            raise ValueError("manual LED timeout must be positive")
        if not 0.0 <= backlight_brightness <= 1.0:
            raise ValueError("backlight brightness must be within 0..1")
        if not 0.0 <= neutral_oblique_brightness <= 1.0:
            raise ValueError("neutral/oblique brightness must be within 0..1")
        if backlight_settle_s < 0:
            raise ValueError("backlight settle delay cannot be negative")
        self.backlight = backlight
        self.neutral_oblique = neutral_oblique
        self.neutral_oblique_pin = (
            None
            if neutral_oblique is None
            else getattr(neutral_oblique, "pin", NEUTRAL_OBLIQUE_GPIO_PIN)
        )
        self.backlight_brightness = backlight_brightness
        self.neutral_oblique_brightness = neutral_oblique_brightness
        self.backlight_settle_s = backlight_settle_s
        self._backlight_on = False
        # Keep the selected PWM level separate from whether the operator has
        # requested the backlight to stay off. This lets off/on restore the
        # last level instead of silently replacing it with the default.
        self._backlight_restore_enabled = backlight_brightness > 0.0
        self._neutral_oblique_logical_output = False
        self.lead_delay_s = lead_delay_s
        self.tail_delay_s = tail_delay_s
        self.manual_timeout_s = manual_timeout_s
        self._sleep = sleep
        self._clock = clock
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._manual_timer = None
        self._manual_generation = 0
        self._mode = "auto"
        self._logical_output = False
        self._last_error: str | None = None
        self._off()
        self._neutral_off()
        initial_backlight_error = self._backlight_preview(force=True)
        if initial_backlight_error is not None:
            raise IlluminationError(initial_backlight_error)

    @property
    def auto_mode(self) -> bool:
        return self._mode == "auto"

    @property
    def backlight_available(self) -> bool:
        return self.backlight is not None

    @property
    def neutral_oblique_available(self) -> bool:
        return self.neutral_oblique is not None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def logical_output(self) -> bool:
        return self._logical_output

    @property
    def physical_output(self) -> bool:
        return self._physical_state(
            self.output, self._logical_output, "LED"
        )

    @property
    def neutral_oblique_physical_output(self) -> bool:
        return self._physical_state(
            self.neutral_oblique,
            self._neutral_oblique_logical_output,
            "neutral/oblique light",
        )

    def _physical_state(
        self,
        output: DigitalOutput | PwmOutput | None,
        fallback: bool,
        label: str,
    ) -> bool:
        if output is None:
            return False
        try:
            value = getattr(output, "is_on", None)
            if value is not None:
                return bool(value() if callable(value) else value)
            value = getattr(output, "value", None)
            if value is not None:
                return bool(value() if callable(value) else value)
            return fallback
        except Exception as exc:
            self._last_error = f"{label} state read failed: {exc}"
            raise IlluminationError(self._last_error) from exc

    def status(self) -> IlluminationStatus:
        with self._lock:
            return IlluminationStatus(
                mode=self._mode,
                auto_mode=self.auto_mode,
                logical_output=self._logical_output,
                physical_output=self.physical_output,
                pin=self.pin,
                lead_delay_ms=round(self.lead_delay_s * 1000),
                tail_delay_ms=round(self.tail_delay_s * 1000),
                manual_timeout_s=self.manual_timeout_s,
                backlight_pin=(
                    None
                    if self.backlight is None
                    else getattr(self.backlight, "pin", BACKLIGHT_GPIO_PIN)
                ),
                backlight_brightness=(
                    None
                    if self.backlight is None
                    else (self.backlight_brightness if self._backlight_on else 0.0)
                ),
                neutral_oblique_brightness=(
                    None
                    if self.neutral_oblique is None
                    else (
                        self.neutral_oblique_brightness
                        if self._neutral_oblique_logical_output
                        else 0.0
                    )
                ),
                neutral_oblique_pin=self.neutral_oblique_pin,
                neutral_oblique_logical_output=(
                    self._neutral_oblique_logical_output
                ),
                neutral_oblique_physical_output=(
                    self.neutral_oblique_physical_output
                ),
                last_error=self._last_error,
            )

    def status_text(self) -> str:
        return str(self.status())

    def led_auto_on(self) -> IlluminationStatus:
        with self._lock:
            self._cancel_manual_timer()
            try:
                self._off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._mode = "auto"
            self._logical_output = False
            self._last_error = None
            return self.status()

    def led_auto_off(self) -> IlluminationStatus:
        with self._lock:
            self._cancel_manual_timer()
            previous_logical_output = self._logical_output
            self._mode = "auto_off"
            try:
                self._off()
            except IlluminationError as exc:
                self._logical_output = previous_logical_output
                self._last_error = str(exc)
                raise
            self._logical_output = False
            self._last_error = None
            return self.status()

    def led_on(self, seconds: float | None = None) -> IlluminationStatus:
        if seconds is None:
            timeout_s = self.manual_timeout_s
        else:
            try:
                timeout_s = float(seconds)
            except (TypeError, ValueError) as exc:
                raise IlluminationError(
                    f"invalid LED duration: {seconds!r}"
                ) from exc
            if timeout_s <= 0:
                raise IlluminationError("LED duration must be positive")
        with self._lock:
            self._cancel_manual_timer()
            self._mode = "manual"
            try:
                # Manual coax inspection must be exclusive: the dim preview
                # backlight (and any neutral light) would otherwise stay on
                # from session startup and wash out the LED-only view.
                if self._backlight_off():
                    self._sleep(self.backlight_settle_s)
                self._neutral_off()
                self._on()
                self._logical_output = True
                generation = self._manual_generation
                timer = self._timer_factory(
                    timeout_s,
                    lambda: self._manual_timeout(generation),
                )
                timer.daemon = True
                timer.start()
            except Exception as exc:
                try:
                    self._off()
                except Exception as off_error:
                    self._last_error = f"{exc}; LED cleanup failed: {off_error}"
                else:
                    self._logical_output = False
                    self._last_error = str(exc)
                raise IlluminationError(f"manual LED on failed: {exc}") from exc
            self._manual_timer = timer
            self._last_error = None
            return self.status()

    def led_off(self) -> IlluminationStatus:
        with self._lock:
            self._cancel_manual_timer()
            self._mode = "manual"
            try:
                self._off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._logical_output = False
            restore_error = self._backlight_preview()
            self._last_error = restore_error
            return self.status()

    def backlight_on(self) -> IlluminationStatus:
        """Light the preview backlight at its remembered level."""
        with self._lock:
            if self.backlight is None:
                raise IlluminationError("backlight control is unavailable")
            if self.backlight_brightness <= 0.0:
                self.backlight_brightness = BACKLIGHT_PREVIEW_BRIGHTNESS
            self._backlight_restore_enabled = True
            error = self._backlight_preview(force=True)
            if error is not None:
                raise IlluminationError(error)
            self._last_error = None
            return self.status()

    def backlight_off(self) -> IlluminationStatus:
        """Hold the backlight dark, including after the next capture."""
        with self._lock:
            if self.backlight is None:
                raise IlluminationError("backlight control is unavailable")
            self._backlight_restore_enabled = False
            try:
                self._backlight_off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._last_error = None
            return self.status()

    def set_backlight_level(self, percent: float) -> IlluminationStatus:
        """Set the preview level (0-100%) that captures restore afterwards."""
        if self.backlight is None:
            raise IlluminationError("backlight control is unavailable")
        try:
            value = float(percent) / 100.0
        except (TypeError, ValueError) as exc:
            raise IlluminationError(f"invalid backlight level: {percent!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise IlluminationError("backlight level must be within 0..100")
        with self._lock:
            if value == 0.0:
                # Zero is an off request, not a replacement for the level
                # selected by the operator. Keep that level for backlight_on.
                self._backlight_restore_enabled = False
                try:
                    self._backlight_off()
                except IlluminationError as exc:
                    self._last_error = str(exc)
                    raise
            else:
                self.backlight_brightness = value
                self._backlight_restore_enabled = True
                error = self._backlight_preview(force=True)
                if error is not None:
                    raise IlluminationError(error)
            self._last_error = None
            return self.status()

    def set_neutral_level(self, percent: float) -> IlluminationStatus:
        """Set the neutral/oblique PWM duty cycle (0-100%)."""
        if self.neutral_oblique is None:
            raise IlluminationError(
                "neutral/oblique light control is unavailable"
            )
        try:
            value = float(percent) / 100.0
        except (TypeError, ValueError) as exc:
            raise IlluminationError(
                f"invalid neutral/oblique level: {percent!r}"
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise IlluminationError(
                "neutral/oblique level must be within 0..100"
            )

        with self._lock:
            setter = getattr(self.neutral_oblique, "set_brightness", None)
            if not callable(setter):
                if value not in (0.0, 1.0):
                    raise IlluminationError(
                        "neutral/oblique brightness control is unavailable"
                    )
                if value > 0.0:
                    self.neutral_oblique_brightness = value
                try:
                    if value == 0.0:
                        self._neutral_off()
                    else:
                        self._neutral_on()
                        self._neutral_oblique_logical_output = True
                except IlluminationError as exc:
                    self._last_error = str(exc)
                    raise
            else:
                if value > 0.0:
                    self.neutral_oblique_brightness = value
                try:
                    if value == 0.0:
                        self._neutral_off()
                    else:
                        setter(value)
                        self._neutral_oblique_logical_output = True
                except Exception as exc:
                    self._last_error = (
                        f"neutral/oblique brightness set failed: {exc}"
                    )
                    raise IlluminationError(self._last_error) from exc
            self._last_error = None
            return self.status()

    def neutral_on(self) -> IlluminationStatus:
        """Turn on the manually controlled neutral/oblique source."""
        with self._lock:
            if self.neutral_oblique is None:
                raise IlluminationError(
                    "neutral/oblique light control is unavailable"
                )
            try:
                self._neutral_on()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._neutral_oblique_logical_output = True
            self._last_error = None
            return self.status()

    def neutral_off(self) -> IlluminationStatus:
        """Turn off the manually controlled neutral/oblique source."""
        with self._lock:
            if self.neutral_oblique is None:
                raise IlluminationError(
                    "neutral/oblique light control is unavailable"
                )
            try:
                self._neutral_off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._last_error = None
            return self.status()

    def prepare_slide_capture(self) -> None:
        """Clear the point LED and restore the backlight for slide capture.

        The slide camera uses continuous transmitted/backlight illumination,
        so the backlight stays on across the slide phase. The point LED is
        forced low and the current manual/automatic mode is preserved. A
        physical ``off`` failure is surfaced to the capture error path and the
        logical state remains on because low was not confirmed.
        """
        with self._lock:
            self._cancel_manual_timer()
            try:
                self._off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            self._logical_output = False
            if self.backlight is not None:
                if self.backlight_brightness <= 0.0:
                    self.backlight_brightness = BACKLIGHT_PREVIEW_BRIGHTNESS
                self._backlight_restore_enabled = True
                restore_error = self._backlight_preview(force=True)
                if restore_error is not None:
                    raise IlluminationError(restore_error)
            self._last_error = None

    def capture_block(
        self,
        capture: Callable[[], T],
        *,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, dict[str, object]]:
        """Run one still, bracketing it only while automatic mode is on."""
        with self._lock:
            sleep = sleep or self._sleep
            clock = clock or self._clock
            if not self.auto_mode:
                result = capture()
                return result, self.capture_metadata()

            started_at: float | None = None
            result: T | None = None
            operation_error: Exception | None = None
            off_error: Exception | None = None
            try:
                if self._backlight_off():
                    sleep(self.backlight_settle_s)
                self._on()
                self._logical_output = True
                started_at = clock()
                sleep(self.lead_delay_s)
                result = capture()
                sleep(self.tail_delay_s)
            except Exception as exc:
                operation_error = exc
            finally:
                try:
                    self._off()
                except Exception as exc:
                    off_error = exc
                else:
                    self._logical_output = False
                restore_error = self._backlight_preview()

            duration_ms = (
                None
                if started_at is None
                else round((clock() - started_at) * 1000)
            )
            if operation_error is not None:
                if off_error is not None:
                    operation_error = IlluminationError(
                        f"{operation_error}; LED cleanup failed: {off_error}"
                    )
                    self._last_error = str(operation_error)
                else:
                    self._last_error = str(operation_error)
                raise IlluminationError(str(operation_error)) from operation_error
            if off_error is not None:
                self._last_error = str(off_error)
                raise IlluminationError(f"LED cleanup failed: {off_error}") from off_error
            # A failed preview restore only darkens preview, so it is reported
            # rather than raised.
            self._last_error = restore_error
            return result, self.capture_metadata(led_on_duration_ms=duration_ms)

    @contextmanager
    def _capture_cleanup(self, *, restore_backlight: bool) -> Iterator[None]:
        operation_error: Exception | None = None
        cleanup_errors: list[Exception] = []
        try:
            yield
        except Exception as exc:
            operation_error = exc
        finally:
            try:
                self._off()
            except Exception as exc:
                cleanup_errors.append(exc)
            else:
                self._logical_output = False
            try:
                self._neutral_off()
            except Exception as exc:
                cleanup_errors.append(exc)
            if restore_backlight:
                restore_error = self._backlight_preview(force=True)
            else:
                # Not "leave it dark": fall back to the operator's last preview
                # level. Forcing the backlight off here blinds presence/motion
                # detection after every debug snap, disabling automatic
                # capture until the process restarts.
                restore_error = self._backlight_preview()

        if operation_error is not None:
            if cleanup_errors:
                operation_error = IlluminationError(
                    f"{operation_error}; light cleanup failed: "
                    f"{cleanup_errors[0]}"
                )
            self._last_error = str(operation_error)
            raise IlluminationError(str(operation_error)) from operation_error
        if cleanup_errors:
            self._last_error = str(cleanup_errors[0])
            raise IlluminationError(
                f"light cleanup failed: {cleanup_errors[0]}"
            ) from cleanup_errors[0]
        self._last_error = restore_error

    def capture_pair(
        self,
        backlight_capture: Callable[[], T],
        led_capture: Callable[[], U],
        *,
        restore_backlight: bool = True,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, U, dict[str, object]]:
        """Capture one transmitted-light still and one point-LED still.

        This is a debug seam, not part of normal block or slide capture. The
        backlight image is taken first with the point LED forced low. The
        second image is taken after switching to the point LED. The preview
        backlight is restored before returning by default. Debug callers can
        disable that restore when the session should remain dark after snap.
        """
        with self._lock:
            if self.backlight is None:
                raise IlluminationError("backlight control is unavailable")
            sleep = sleep or self._sleep
            clock = clock or self._clock
            if self.backlight_brightness <= 0.0:
                self.backlight_brightness = BACKLIGHT_PREVIEW_BRIGHTNESS

            backlight_result: T | None = None
            led_result: U | None = None
            started_at: float | None = None
            with self._capture_cleanup(restore_backlight=restore_backlight):
                self._cancel_manual_timer()
                self._off()
                self._logical_output = False
                self._neutral_off()

                if not self._backlight_on:
                    restore_error = self._backlight_preview(force=True)
                    if restore_error is not None:
                        raise IlluminationError(restore_error)
                    sleep(self.backlight_settle_s)
                backlight_result = backlight_capture()

                if self._backlight_off():
                    sleep(self.backlight_settle_s)
                self._on()
                self._logical_output = True
                started_at = clock()
                sleep(self.lead_delay_s)
                led_result = led_capture()
                sleep(self.tail_delay_s)

            duration_ms = (
                None
                if started_at is None
                else round((clock() - started_at) * 1000)
            )
            return (
                backlight_result,
                led_result,
                self.capture_metadata(led_on_duration_ms=duration_ms),
            )

    def capture_triple(
        self,
        coax_capture: Callable[[], T],
        backlight_capture: Callable[[], U],
        neutral_oblique_capture: Callable[[], V],
        *,
        restore_backlight: bool = True,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[T, U, V, dict[str, object]]:
        """Capture coax, transmitted backlight, and neutral/oblique stills."""
        with self._lock:
            if self.backlight is None:
                raise IlluminationError("backlight control is unavailable")
            if self.neutral_oblique is None:
                raise IlluminationError(
                    "neutral/oblique light control is unavailable"
                )
            sleep = sleep or self._sleep
            clock = clock or self._clock
            if self.backlight_brightness <= 0.0:
                self.backlight_brightness = BACKLIGHT_PREVIEW_BRIGHTNESS

            coax_result: T | None = None
            backlight_result: U | None = None
            neutral_result: V | None = None
            coax_duration_ms: int | None = None
            neutral_duration_ms: int | None = None
            with self._capture_cleanup(restore_backlight=restore_backlight):
                self._cancel_manual_timer()
                self._off()
                self._logical_output = False
                self._neutral_off()

                if self._backlight_off():
                    sleep(self.backlight_settle_s)
                self._on()
                self._logical_output = True
                coax_started_at = clock()
                sleep(self.lead_delay_s)
                coax_result = coax_capture()
                sleep(self.tail_delay_s)
                coax_duration_ms = round((clock() - coax_started_at) * 1000)
                self._off()
                self._logical_output = False

                restore_error = self._backlight_preview(force=True)
                if restore_error is not None:
                    raise IlluminationError(restore_error)
                sleep(self.backlight_settle_s)
                backlight_result = backlight_capture()

                if self._backlight_off():
                    sleep(self.backlight_settle_s)
                self._neutral_on()
                self._neutral_oblique_logical_output = True
                neutral_started_at = clock()
                sleep(self.lead_delay_s)
                neutral_result = neutral_oblique_capture()
                sleep(self.tail_delay_s)
                neutral_duration_ms = round(
                    (clock() - neutral_started_at) * 1000
                )

            metadata = self.capture_metadata(led_on_duration_ms=coax_duration_ms)
            metadata.update(
                {
                    "illumination_capture_mode": "triple",
                    "illumination_variant_sequence": ",".join(
                        TRIPLE_CAPTURE_VARIANTS
                    ),
                    "illumination_neutral_oblique_on_duration_ms": (
                        neutral_duration_ms
                    ),
                }
            )
            return coax_result, backlight_result, neutral_result, metadata

    def capture_metadata(self, *, led_on_duration_ms: int | None = None) -> dict[str, object]:
        metadata = {
            "illumination_auto_mode": self.auto_mode,
            "illumination_mode": self.mode,
            "illumination_pin": self.pin,
            "illumination_lead_ms": round(self.lead_delay_s * 1000),
            "illumination_tail_ms": round(self.tail_delay_s * 1000),
            "illumination_led_on_duration_ms": led_on_duration_ms,
            "backlight_pin": (
                None
                if self.backlight is None
                else getattr(self.backlight, "pin", BACKLIGHT_GPIO_PIN)
            ),
            "backlight_preview_brightness": (
                None if self.backlight is None else self.backlight_brightness
            ),
            "backlight_settle_ms": (
                None if self.backlight is None else round(self.backlight_settle_s * 1000)
            ),
            "neutral_oblique_pin": self.neutral_oblique_pin,
            "neutral_oblique_brightness": (
                None
                if self.neutral_oblique is None
                else self.neutral_oblique_brightness
            ),
            "illumination_capture_mode": None,
            "illumination_variant_sequence": None,
            "illumination_neutral_oblique_on_duration_ms": None,
        }
        return {key: metadata[key] for key in ILLUMINATION_METADATA_KEYS}

    def shutdown(self) -> None:
        """Force all lights off while keeping GPIO devices open."""
        with self._lock:
            self._cancel_manual_timer()
            errors: list[Exception] = []
            try:
                self._off()
            except Exception as exc:
                errors.append(exc)
            else:
                self._logical_output = False

            if self.neutral_oblique is not None:
                try:
                    self._neutral_off()
                except Exception as exc:
                    errors.append(exc)

            if self.backlight is not None:
                try:
                    self.backlight.off()
                except Exception as exc:
                    errors.append(IlluminationError(f"backlight off failed: {exc}"))
                finally:
                    self._backlight_on = False

            if errors:
                message = "; ".join(str(error) for error in errors)
                self._last_error = message
                raise IlluminationError(
                    f"illumination shutdown failed: {message}"
                ) from errors[0]
            self._last_error = None

    def close(self) -> None:
        with self._lock:
            shutdown_error: Exception | None = None
            try:
                self.shutdown()
            except Exception as exc:
                shutdown_error = exc
            close_error: Exception | None = None
            try:
                self.output.close()
            except Exception as exc:
                close_error = exc
            if self.backlight is not None:
                try:
                    self.backlight.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
            if self.neutral_oblique is not None:
                try:
                    self.neutral_oblique.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
            error = shutdown_error or close_error
            if error is not None:
                self._last_error = str(error)
                raise IlluminationError(f"LED shutdown failed: {error}") from error
            self._last_error = None

    def _on(self) -> None:
        try:
            self.output.on()
        except Exception as exc:
            raise IlluminationError(f"LED on failed: {exc}") from exc

    def _off(self) -> None:
        try:
            self.output.off()
        except Exception as exc:
            raise IlluminationError(f"LED off failed: {exc}") from exc

    def _neutral_on(self) -> None:
        if self.neutral_oblique is None:
            raise IlluminationError("neutral/oblique light control is unavailable")
        try:
            setter = getattr(self.neutral_oblique, "set_brightness", None)
            if callable(setter):
                if self.neutral_oblique_brightness <= 0.0:
                    self.neutral_oblique_brightness = (
                        NEUTRAL_OBLIQUE_BRIGHTNESS
                    )
                setter(self.neutral_oblique_brightness)
            else:
                self.neutral_oblique.on()
        except Exception as exc:
            raise IlluminationError(f"neutral/oblique light on failed: {exc}") from exc

    def _neutral_off(self) -> None:
        if self.neutral_oblique is None:
            self._neutral_oblique_logical_output = False
            return
        try:
            self.neutral_oblique.off()
        except Exception as exc:
            raise IlluminationError(
                f"neutral/oblique light off failed: {exc}"
            ) from exc
        self._neutral_oblique_logical_output = False

    def _backlight_off(self) -> bool:
        """Drive the preview backlight low. Returns True if it was lit."""
        if self.backlight is None:
            return False
        was_on = self._backlight_on
        try:
            self.backlight.off()
        except Exception as exc:
            raise IlluminationError(f"backlight off failed: {exc}") from exc
        self._backlight_on = False
        return was_on

    def _backlight_preview(self, *, force: bool = False) -> str | None:
        """Restore the selected preview level, unless the operator chose off."""
        if self.backlight is None:
            return None
        if not force and not self._backlight_restore_enabled:
            try:
                self.backlight.off()
            except Exception as exc:
                self._backlight_on = False
                self._last_error = f"backlight preview off failed: {exc}"
                return self._last_error
            self._backlight_on = False
            return None
        try:
            self.backlight.set_brightness(self.backlight_brightness)
        except Exception as exc:
            self._backlight_on = False
            self._last_error = f"backlight preview restore failed: {exc}"
            return self._last_error
        self._backlight_on = self.backlight_brightness > 0
        return None

    def _cancel_manual_timer(self) -> None:
        self._manual_generation += 1
        timer = self._manual_timer
        self._manual_timer = None
        if timer is not None:
            timer.cancel()

    def _manual_timeout(self, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._manual_generation:
                return
            if self._mode != "manual" or not self._logical_output:
                return
            try:
                self._off()
            except IlluminationError as exc:
                self._last_error = str(exc)
                raise
            else:
                self._logical_output = False
                # Hand motion/presence detection back its preview backlight.
                self._last_error = self._backlight_preview()


class _GpiozeroCloseMixin:
    """Shared ``close()`` for the gpiozero output adapters below: turn the
    output off, then close the underlying device even if ``off()`` raised,
    then re-raise that error. Both adapters need the device closed no matter
    what, so the close is unconditional, but a caller that expected off() to
    succeed still sees its exception."""

    def off(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        off_error: Exception | None = None
        try:
            self.off()
        except Exception as exc:
            off_error = exc
        finally:
            self._device.close()
        if off_error is not None:
            raise off_error


class GpiozeroDigitalOutput(_GpiozeroCloseMixin):
    """Pi adapter using gpiozero's digital output, with imports delayed."""

    def __init__(self, pin: int = ILLUMINATION_GPIO_PIN, *, device_factory=None):
        if device_factory is None:
            try:
                from gpiozero import DigitalOutputDevice
            except ImportError as exc:
                raise RuntimeError(
                    "gpiozero is required on the Pi for coaxial LED control"
                ) from exc
            device_factory = DigitalOutputDevice
        self.pin = pin
        self._device = device_factory(pin, initial_value=False)

    @property
    def is_on(self) -> bool:
        return bool(self._device.value)

    def on(self) -> None:
        self._device.on()

    def off(self) -> None:
        self._device.off()


class GpiozeroPwmOutput(_GpiozeroCloseMixin):
    """Pi adapter for a PWM light output, with imports delayed."""

    description = "PWM light control"

    def __init__(self, pin: int, *, device_factory=None):
        if device_factory is None:
            try:
                from gpiozero import PWMOutputDevice
            except ImportError as exc:
                raise RuntimeError(
                    f"gpiozero is required on the Pi for {self.description}"
                ) from exc
            device_factory = PWMOutputDevice
        self.pin = pin
        self._device = device_factory(pin, initial_value=0.0)

    @property
    def brightness(self) -> float:
        return float(self._device.value)

    @property
    def is_on(self) -> bool:
        return self.brightness > 0.0

    def on(self) -> None:
        self.set_brightness(
            self.brightness if self.brightness > 0.0 else 1.0
        )

    def set_brightness(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("PWM brightness must be within 0..1")
        self._device.value = value

    def off(self) -> None:
        self._device.value = 0.0


class GpiozeroPwmBacklight(GpiozeroPwmOutput):
    """Pi adapter for the PWM-dimmed preview backlight on BCM 27."""

    description = "backlight control"

    def __init__(self, pin: int = BACKLIGHT_GPIO_PIN, *, device_factory=None):
        super().__init__(pin, device_factory=device_factory)
