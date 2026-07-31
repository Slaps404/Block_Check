"""Resolution-relative scaling for segmentation spatial controls.

The slide segmentation constants are tuned at ``SEGMENTATION_REFERENCE_WIDTH``.
When a slide is stored at a lower resolution, pixel lengths and areas must
shrink with the frame so the mask stays geometrically equivalent. Intensity and
colour gates and the scorer's fixed 256 grid are NOT scaled (ADR 0013).

Rounding matches the validated qualification experiment
(``experiments/active/capture_resolution_qualification/run.py``) so production
reproduces its IoU 0.987 evidence.
"""
from __future__ import annotations

import math

from constants import CAPTURE_DIMENSIONS, SEGMENTATION_REFERENCE_WIDTH

_SLIDE_MIN_SCALE = (
    CAPTURE_DIMENSIONS["slide"][0] / SEGMENTATION_REFERENCE_WIDTH
)  # 0.5, half-res floor
_BLOCK_MIN_SCALE = 0.25  # supports a four-times smaller block capture


def pixel_scale_for(frame_width: int) -> float:
    """Linear scale of ``frame_width`` against the tuning reference width.

    Clamped to [half-res floor, full-res ceiling] -- the only two real slide
    capture widths CAPTURE_DIMENSIONS defines. A width outside that domain
    (e.g. a small synthetic test fixture) is not a hardware capture and must
    not be allowed to collapse spatial controls (texture window, close
    kernel) to a degenerate size.
    """
    raw = frame_width / SEGMENTATION_REFERENCE_WIDTH
    return max(_SLIDE_MIN_SCALE, min(1.0, raw))


def block_pixel_scale_for(frame_width: int) -> float:
    """Scale block spatial controls, including a 4x downscaled capture.

    Block color thresholds are values in color space and intentionally stay
    fixed.  This factor is only for lengths, areas, and morphology.
    """
    raw = frame_width / SEGMENTATION_REFERENCE_WIDTH
    return max(_BLOCK_MIN_SCALE, min(1.0, raw))


def scale_odd_length(value: int, scale: float, *, minimum: int = 1) -> int:
    """Scale a kernel/window length; keep it odd and >= minimum."""
    result = max(minimum, int(round(value * scale)))
    return result if result % 2 else result + 1


def scale_reach(value: int, scale: float) -> int:
    """Scale a dilation radius (not forced odd), floored at 0."""
    return max(0, int(round(value * scale)))


def scale_area_min(value: int, scale: float) -> int:
    """Scale a minimum-area floor (px^2), floored at 1."""
    return max(1, int(math.ceil(value * scale * scale)))


def scale_area_max(value: int, scale: float) -> int:
    """Scale a maximum-area cap (px^2), floored at 0."""
    return max(0, int(math.floor(value * scale * scale)))
