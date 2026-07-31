"""Full-frame chromatic-occupancy detection for empty-backlight calibration.

Pure numpy: an RGB frame in, a small result out. No hardware, no I/O. The
calibration gate uses this to fail-closed when the capture area holds a colored
specimen/cassette, while staying blind to neutral things (the black jig, neutral
dust, lens vignetting) so those never false-fail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCCUPANCY_SAT_MIN = 40.0        # HSV-style saturation (0-255); "chromatic" above this
OCCUPANCY_AREA_FRAC_MAX = 0.05  # occupied if chromatic pixels exceed this frame fraction


@dataclass(frozen=True)
class OccupancyResult:
    occupied: bool
    chromatic_fraction: float
    sat_min: float
    area_frac_max: float


def detect_occupancy(
    frame_rgb: np.ndarray,
    *,
    sat_min: float = OCCUPANCY_SAT_MIN,
    area_frac_max: float = OCCUPANCY_AREA_FRAC_MAX,
) -> OccupancyResult:
    """Report whether an RGB frame's capture area is occupied by a chromatic object.

    Saturation is the HSV S channel computed directly per pixel:
    (max - min) / max * 255, and 0 where max == 0. Neutral pixels (backlight,
    black jig, gray dust) have S ~ 0 and are ignored; colored pixels (purple
    cassette, stained tissue) have high S.
    """
    rgb = np.asarray(frame_rgb, dtype=np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / mx * 255.0, 0.0)
    chromatic_fraction = float(np.mean(sat > sat_min))
    return OccupancyResult(
        occupied=chromatic_fraction > area_frac_max,
        chromatic_fraction=chromatic_fraction,
        sat_min=sat_min,
        area_frac_max=area_frac_max,
    )
