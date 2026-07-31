"""Pure camera-calibration types shared by workflow and camera adapters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


@dataclass(frozen=True)
class LockedCameraControls:
    exposure_time_us: int
    analogue_gain: float
    colour_gains: tuple[float, float]
    frame_duration_us: int | None = None


@dataclass(frozen=True)
class CalibrationQuality:
    stable: bool
    sample_count: int
    settling_frames: int
    exposure_cv: float
    gain_cv: float
    red_gain_cv: float
    blue_gain_cv: float
    background_luma_median: float
    clipped_high_fraction: float
    clipped_low_fraction: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class PhaseCameraCalibration:
    mode: str
    controls: LockedCameraControls
    quality: CalibrationQuality
    metadata_samples: tuple[Mapping[str, object], ...]
    calibrated_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
    calibration_id: str | None = None


@dataclass(frozen=True)
class ActivatedCameraMode:
    calibration: PhaseCameraCalibration
    baseline: object


class CameraCalibrationError(RuntimeError):
    """A phase cannot become capture-ready under unknown camera conditions."""

    def __init__(self, mode: str, reason: str, diagnostics: Mapping[str, object]):
        self.mode = mode
        self.reason = reason
        self.diagnostics = dict(diagnostics)
        super().__init__(f"{mode} camera calibration failed: {reason}")
