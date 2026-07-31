"""Conservatively select the mounted tissue sheet nearest the slide label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slide.label_mask import LabelRect


@dataclass(frozen=True)
class SheetSelection:
    """The mask to score and whether a label-nearest selection was applied."""

    mask: np.ndarray
    applied: bool
    reason: str


def _two_means(values: np.ndarray) -> tuple[float, float]:
    """Return stable near/far one-dimensional foreground centres."""
    near, far = np.quantile(values, (0.25, 0.75)).astype(float)
    for _ in range(24):
        cut = (near + far) / 2.0
        near_values = values[values <= cut]
        far_values = values[values > cut]
        if not len(near_values) or not len(far_values):
            break
        next_near = float(near_values.mean())
        next_far = float(far_values.mean())
        if abs(next_near - near) + abs(next_far - far) < 0.01:
            break
        near, far = next_near, next_far
    return min(near, far), max(near, far)


def _label_distance(shape: tuple[int, int], side: str) -> np.ndarray:
    """Return each pixel's distance from the detected label edge."""
    height, width = shape
    if side == "left":
        return np.broadcast_to(np.arange(width), (height, width))
    if side == "right":
        return np.broadcast_to(np.arange(width - 1, -1, -1), (height, width))
    if side == "top":
        return np.broadcast_to(np.arange(height)[:, None], (height, width))
    if side == "bottom":
        return np.broadcast_to(np.arange(height - 1, -1, -1)[:, None], (height, width))
    raise ValueError(f"unknown label side: {side!r}")


def select_label_nearest_sheet(mask: np.ndarray, label_rect: LabelRect | None) -> SheetSelection:
    """Select a clean label-nearest lobe, otherwise return the original mask.

    A single tissue sheet can contain distant fragments. This function therefore
    selects only when the foreground forms two balanced, separated lobes and a
    three-pixel divider is completely free of tissue. The mounting convention
    establishes either lobe as comparable, so selection is deterministic rather
    than score-driven.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if label_rect is None or not label_rect.found:
        return SheetSelection(mask.copy(), False, "label_not_found")
    if label_rect.label_side not in {"left", "right", "top", "bottom"}:
        return SheetSelection(mask.copy(), False, "unsupported_label_side")

    foreground = mask > 0
    if not foreground.any():
        return SheetSelection(mask.copy(), False, "empty_mask")

    distance = _label_distance(mask.shape, label_rect.label_side)
    foreground_distance = distance[foreground]
    near_center, far_center = _two_means(foreground_distance)
    cut = int(round((near_center + far_center) / 2.0))
    extent = float(mask.shape[1] if label_rect.label_side in {"left", "right"} else mask.shape[0])
    separation_fraction = (far_center - near_center) / extent
    near = foreground & (distance <= cut)
    far = foreground & ~near
    total = float(foreground.sum())
    near_fraction = float(near.sum() / total)
    far_fraction = float(far.sum() / total)

    half_valley = max(1, int(round(extent * 0.015)))
    valley_fraction = float((foreground & (np.abs(distance - cut) <= half_valley)).sum() / total)
    divider_touches_tissue = bool((foreground & (np.abs(distance - cut) <= 1)).any())

    if divider_touches_tissue:
        return SheetSelection(mask.copy(), False, "divider_touches_tissue")
    if not (0.17 <= separation_fraction <= 0.30):
        return SheetSelection(mask.copy(), False, "separation_outside_calibrated_range")
    if min(near_fraction, far_fraction) < 0.20:
        return SheetSelection(mask.copy(), False, "unbalanced_lobes")
    if valley_fraction > 0.12:
        return SheetSelection(mask.copy(), False, "valley_contains_too_much_tissue")

    selected = np.where(near, mask, 0).astype(mask.dtype, copy=False)
    return SheetSelection(selected, True, f"label_nearest_{label_rect.label_side}")
