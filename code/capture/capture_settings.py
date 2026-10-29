"""Live-editable capture settings: registry, validation, Pi-local persistence."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Union

from capture_runtime import ROLE_CAMERA_SETTINGS, CameraSettings
from capture_session import CaptureMode, SessionConfig
from constants import BACKLIGHT_PREVIEW_BRIGHTNESS, NEUTRAL_OBLIQUE_BRIGHTNESS
from session.atomic_io import atomic_json


SCHEMA_VERSION = 1
CAMERA = "camera"
DETECTION = "detection"
LIGHTS = "lights"

Value = Union[int, float, bool]

# (field, type, min, max, step); ranges sit inside what libcamera/IMX477 accept.
_CAMERA_FIELDS = (
    ("exposure_us", int, 100, 200000, 100),
    ("analogue_gain", float, 1.0, 16.0, 0.1),
    ("colour_gain_red", float, 0.5, 8.0, 0.01),
    ("colour_gain_blue", float, 0.5, 8.0, 0.01),
    ("contrast", float, 0.0, 4.0, 0.1),
    ("sharpness", float, 0.0, 16.0, 0.1),
)
_DETECTION_FIELDS = (
    ("presence_threshold", 0.001, 1.0, 0.005),
    ("motion_threshold", 0.001, 1.0, 0.001),
    ("stable_duration", 0.0, 10.0, 0.1),
    ("removal_duration", 0.0, 10.0, 0.1),
)
_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


class CaptureSettingError(ValueError):
    """A setting name or value was rejected."""


@dataclass(frozen=True)
class SettingSpec:
    name: str
    group: str
    applies: str
    kind: type
    default: Value
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def coerce(self, raw: object) -> Value:
        if self.kind is bool:
            return self._coerce_bool(raw)
        if isinstance(raw, bool):
            raise CaptureSettingError(self._type_error())
        try:
            number = float(raw) if isinstance(raw, str) else raw
            if not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError
        except (TypeError, ValueError):
            raise CaptureSettingError(self._type_error()) from None
        if self.kind is int:
            if number != int(number):
                raise CaptureSettingError(self._type_error())
            value: Value = int(number)
        else:
            value = float(number)
        if not self.minimum <= value <= self.maximum:
            raise CaptureSettingError(
                f"{self.name} must be between {self.minimum:g} and {self.maximum:g}"
            )
        return value

    def _coerce_bool(self, raw: object) -> bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise CaptureSettingError(f"{self.name} must be on or off")

    def _type_error(self) -> str:
        noun = "a whole number" if self.kind is int else "a number"
        return f"{self.name} must be {noun}"


def _camera_value(settings: CameraSettings, field: str) -> Value:
    if field == "colour_gain_red":
        return settings.colour_gains[0]
    if field == "colour_gain_blue":
        return settings.colour_gains[1]
    return getattr(settings, field)


def _build_specs(session_defaults: SessionConfig) -> tuple[SettingSpec, ...]:
    specs = []
    for mode in (CaptureMode.BLOCK, CaptureMode.SLIDE):
        base = ROLE_CAMERA_SETTINGS[mode]
        for field, kind, low, high, step in _CAMERA_FIELDS:
            specs.append(SettingSpec(
                f"{mode.value}.{field}", mode.value, CAMERA, kind,
                kind(_camera_value(base, field)), low, high, step,
            ))
    for field, low, high, step in _DETECTION_FIELDS:
        specs.append(SettingSpec(
            f"detection.{field}", DETECTION, DETECTION, float,
            float(getattr(session_defaults, field)), low, high, step,
        ))
    specs += [
        SettingSpec("lights.backlight_level", LIGHTS, LIGHTS, int,
                    round(BACKLIGHT_PREVIEW_BRIGHTNESS * 100), 0, 100, 1),
        SettingSpec("lights.neutral_level", LIGHTS, LIGHTS, int,
                    round(NEUTRAL_OBLIQUE_BRIGHTNESS * 100), 1, 100, 1),
        SettingSpec("lights.led_auto", LIGHTS, LIGHTS, bool, True),
    ]
    return tuple(specs)


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


class CaptureSettings:
    """Every editable capture setting; persists only values that differ from default."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        session_defaults: SessionConfig | None = None,
        warn: Callable[[str], object] = _warn,
    ):
        self.path = None if path is None else Path(path)
        self._warn = warn
        self._specs = {
            spec.name: spec for spec in _build_specs(session_defaults or SessionConfig())
        }
        self._overrides: dict[str, Value] = {}
        if self.path is not None:
            self._load()

    def spec(self, name: str) -> SettingSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise CaptureSettingError(f"unknown setting: {name}") from None

    def validate(self, name: str, raw: object) -> Value:
        return self.spec(name).coerce(raw)

    def value(self, name: str) -> Value:
        return self._overrides.get(name, self.spec(name).default)

    def overrides(self) -> dict[str, Value]:
        return dict(self._overrides)

    def set(self, name: str, raw: object) -> Value:
        """Validate, store, and persist one value."""
        value = self.validate(name, raw)
        if value == self.spec(name).default:
            self._overrides.pop(name, None)
        else:
            self._overrides[name] = value
        self.save()
        return value

    def reset(self) -> list[str]:
        """Restore every default; return the names that changed."""
        changed = sorted(self._overrides)
        self._overrides.clear()
        self.save()
        return changed

    def save(self) -> None:
        if self.path is None:
            return
        if not self._overrides:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.path,
            {"schema_version": SCHEMA_VERSION, "overrides": self._overrides},
        )

    def describe(self, name: str) -> dict[str, object]:
        spec = self.spec(name)
        value = self.value(name)
        return {
            "name": spec.name,
            "group": spec.group,
            "applies": spec.applies,
            "type": spec.kind.__name__,
            "value": value,
            "default": spec.default,
            "min": spec.minimum,
            "max": spec.maximum,
            "step": spec.step,
            "changed": value != spec.default,
        }

    def snapshot(self) -> list[dict[str, object]]:
        return [self.describe(name) for name in self._specs]

    def values_for_role(self, role: str) -> dict[str, Value]:
        """Effective values for one camera role plus detection and lights."""
        return {
            name: self.value(name)
            for name, spec in self._specs.items()
            if spec.group in (role, DETECTION, LIGHTS)
        }

    def camera_settings(self, mode: CaptureMode | str) -> CameraSettings:
        mode = CaptureMode(mode)
        prefix = f"{mode.value}."
        return replace(
            ROLE_CAMERA_SETTINGS[mode],
            exposure_us=self.value(prefix + "exposure_us"),
            analogue_gain=self.value(prefix + "analogue_gain"),
            colour_gains=(
                self.value(prefix + "colour_gain_red"),
                self.value(prefix + "colour_gain_blue"),
            ),
            contrast=self.value(prefix + "contrast"),
            sharpness=self.value(prefix + "sharpness"),
        )

    def role_settings(self) -> dict[str, CameraSettings]:
        return {mode.value: self.camera_settings(mode) for mode in CaptureMode}

    def session_config(self, base: SessionConfig) -> SessionConfig:
        return replace(base, **{
            field: self.value(f"detection.{field}") for field, *_ in _DETECTION_FIELDS
        })

    def _load(self) -> None:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            self._warn(f"WARNING: cannot read {self.path}: {exc}; using defaults")
            return
        try:
            overrides = json.loads(text)["overrides"]
            if not isinstance(overrides, dict):
                raise TypeError("overrides is not an object")
        except (ValueError, KeyError, TypeError) as exc:
            self._warn(f"WARNING: ignoring unreadable {self.path}: {exc}; using defaults")
            return
        for name, raw in overrides.items():
            try:
                value = self.validate(name, raw)
            except CaptureSettingError as exc:
                self._warn(f"WARNING: ignoring saved capture setting: {exc}")
                continue
            if value != self.spec(name).default:
                self._overrides[name] = value
