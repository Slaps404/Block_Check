"""Block ROI mask: crop to inset cassette window before brown/yellow gating.

Pipeline: Otsu dark blob → largest component → inset bbox. Fail-safe if no blob.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from constants import (
    BLOCK_CLIP_MIN_INSIDE_FRAC,
    BLOCK_CLOSE_KERNEL,
    BLOCK_INSET_FRAC,
    BLOCK_SEG_BUFFER_FRAC,
)
from verify.scale import scale_odd_length


def find_cassette_bbox(
    gray: np.ndarray, *, pixel_scale: float = 1.0
) -> Optional[tuple[int, int, int, int]]:
    """Return full cassette dark-blob bbox (x, y, w, h), or None.

    Unlike ``find_cassette_window``, this does **not** inset to the paraffin
    window — it keeps the whole cassette silhouette from Otsu + close.
    """
    _, dark = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_length = scale_odd_length(BLOCK_CLOSE_KERNEL, pixel_scale)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_length, close_length))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    if bw <= 0 or bh <= 0:
        return None
    return x, y, bw, bh


def find_cassette_window(
    gray: np.ndarray, *, pixel_scale: float = 1.0
) -> Optional[tuple[int, int, int, int]]:
    """Return inset paraffin-window bbox (x, y, w, h), or None."""
    bbox = find_cassette_bbox(gray, pixel_scale=pixel_scale)
    if bbox is None:
        return None
    x, y, bw, bh = bbox
    ix, iy = int(bw * BLOCK_INSET_FRAC), int(bh * BLOCK_INSET_FRAC)
    if bw - 2 * ix <= 0 or bh - 2 * iy <= 0:
        return None
    return x + ix, y + iy, bw - 2 * ix, bh - 2 * iy


def expand_window_for_segmentation(
    window: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    buffer_frac: float = BLOCK_SEG_BUFFER_FRAC,
) -> tuple[int, int, int, int]:
    """Expand cassette window for pre-seg crop; clamp to image bounds."""
    x, y, w, h = window
    pad_x = int(round(w * buffer_frac))
    pad_y = int(round(h * buffer_frac))
    h_img, w_img = image_shape
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def window_relative_to_crop(
    inner: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map an inner (x, y, w, h) bbox into crop-local coordinates."""
    ix, iy, iw, ih = inner
    cx, cy, _, _ = crop
    return ix - cx, iy - cy, iw, ih


def crop_bgr_at_bounds(
    bgr: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Axis-aligned crop; returns (crop, (x0, y0))."""
    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return bgr, (0, 0)
    return bgr[y:y + h, x:x + w].copy(), (x, y)


def clip_to_window(
    mask: np.ndarray,
    window: Optional[tuple[int, int, int, int]],
) -> np.ndarray:
    """Zero mask outside the given window bbox; passthrough if window is None."""
    if window is None:
        return mask
    x, y, w, h = window
    out = np.zeros_like(mask)
    out[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    return out


def keep_components_in_window(
    mask: np.ndarray,
    window: Optional[tuple[int, int, int, int]],
    min_inside_frac: float = BLOCK_CLIP_MIN_INSIDE_FRAC,
) -> np.ndarray:
    """Keep whole components that are mostly inside the window; drop the rest.

    Unlike clip_to_window's hard rectangle crop, a connected component is kept
    in its entirety (including any tail poking past the window edge) when at
    least `min_inside_frac` of its pixels fall inside the window. Components
    lying mostly or fully outside the window are dropped. This recovers tissue
    tips that rest against the paraffin-window wall while still removing the
    isolated specks the inset was protecting against. Passthrough if window
    is None.
    """
    if window is None:
        return mask
    x, y, w, h = window
    inside = np.zeros(mask.shape, dtype=bool)
    inside[y:y + h, x:x + w] = True
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros(mask.shape, dtype=bool)
    for i in range(1, num):
        comp = labels == i
        total = int(stats[i, cv2.CC_STAT_AREA])
        n_inside = int(np.count_nonzero(comp & inside))
        if total and (n_inside / total) >= min_inside_frac:
            keep |= comp
    out = np.zeros_like(mask)
    out[keep] = mask[keep]
    return out


def window_masked(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero mask outside cassette window; passthrough if window not found."""
    return clip_to_window(mask, find_cassette_window(gray))
