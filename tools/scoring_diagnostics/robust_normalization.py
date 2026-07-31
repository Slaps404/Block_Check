"""Diagnostic-only alternatives to production max-radius normalization.

All modes preserve the full binary mask. Robust modes change only how the
area-weighted centroid's scale is estimated; production ``rms`` and historical
``max`` modes delegate to :mod:`locked_alignment`.
"""

from __future__ import annotations

import cv2
import numpy as np

from verify.locked_alignment import ALIGN_SIZE, radial_normalize_mask


NORMALIZATION_MODES = ("max", "rms", "percentile_98", "power_4")

# Robust statistics are numerically smaller than a maximum radius. Mapping the
# chosen statistic to 13% of the grid keeps remote residuals visible without
# allowing one of them to set the scale. The safety cap prevents clipping on
# masks whose outer radius is unusually large relative to the robust radius.
_ROBUST_TARGET_RADIUS = ALIGN_SIZE * 0.13
_OUTER_RADIUS_CAP = ALIGN_SIZE * 0.45


def _radius_statistic(distances: np.ndarray, mode: str) -> float:
    if mode == "percentile_98":
        return float(np.percentile(distances, 98.0))
    if mode == "power_4":
        return float(np.mean(np.power(distances, 4.0)) ** 0.25)
    raise ValueError(
        f"unknown normalization mode {mode!r}; expected one of {NORMALIZATION_MODES}"
    )


def normalize_mask(mask: np.ndarray, mode: str = "max") -> np.ndarray:
    """Normalize a binary mask with the selected diagnostic scale estimator."""
    if mode in {"max", "rms"}:
        return radial_normalize_mask(mask, mode=mode)
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"unknown normalization mode {mode!r}; expected one of {NORMALIZATION_MODES}"
        )
    if mask.ndim != 2:
        raise ValueError("normalization masks must be two-dimensional")

    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return np.zeros((ALIGN_SIZE, ALIGN_SIZE), dtype=np.uint8)

    cx, cy = float(xs.mean()), float(ys.mean())
    distances = np.hypot(xs - cx, ys - cy)
    robust_radius = _radius_statistic(distances, mode)
    outer_radius = float(distances.max())
    scale = min(
        _ROBUST_TARGET_RADIUS / max(robust_radius, 1.0),
        _OUTER_RADIUS_CAP / max(outer_radius, 1.0),
    )

    center = (ALIGN_SIZE - 1) / 2.0
    matrix = np.array([
        [scale, 0.0, center - cx * scale],
        [0.0, scale, center - cy * scale],
    ])
    return (cv2.warpAffine(
        binary,
        matrix,
        (ALIGN_SIZE, ALIGN_SIZE),
        flags=cv2.INTER_NEAREST,
    ) > 0).astype(np.uint8)
