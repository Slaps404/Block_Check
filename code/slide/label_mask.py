"""Slide label masking for v2 claimed-pair verification.

Detect frosted/printed label rect and white-fill before segmentation so label
ink/QR never enters the stain mask. Rotation-aware; covers yellow MT stickers.
Fail-safe: no detection → image unchanged.

Code map
--------
LabelRect
    found flag + rect geometry; zeros when not found.
find_label_rect(bgr)
    minAreaRect + low-blue cue → label bbox.
apply_label_mask(bgr, rect)   ← preparation.py entry
    White-fill label region; passthrough if not found.
draw_label_overlay
    Debug visualization of detected rect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from constants import (
    AREA_MAX_FRAC,
    AREA_MIN_FRAC,
    BLACK_BG_THRESH,
    BLUE_LOW,
    CX_BAND,
    CY_BAND,
    LABEL_COMBINED_MAX_FRAC,
    LABEL_MORPH_FRAC,
    LABEL_YELLOW_G_MIN,
    LABEL_YELLOW_R_MIN,
    MARGIN_SCALE,
    RTY_MIN,
    SPAN_MAX,
    SPAN_MIN,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LabelRect:
    """Result of find_label_rect().

    When found is False all numeric fields are zero / empty and the image
    should pass through unmodified.
    """
    found: bool
    center: Tuple[float, float]   # (cx, cy) in pixels
    size: Tuple[float, float]     # (w, h) in pixels (long side first not guaranteed)
    angle: float                  # minAreaRect angle in degrees
    box_pts: np.ndarray           # (4, 2) float32 corner coordinates
    label_side: str               # "top", "bottom", "left", "right", or "none"


def _null_rect() -> LabelRect:
    return LabelRect(
        found=False, center=(0.0, 0.0), size=(0.0, 0.0), angle=0.0,
        box_pts=np.zeros((4, 2), dtype=np.float32), label_side="none",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _otsu_on_mask(inv_gray: np.ndarray, mask_bool: np.ndarray) -> int:
    """Otsu threshold using only pixels where mask_bool is True."""
    vals = inv_gray[mask_bool]
    if vals.size == 0:
        return 0
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist))
    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    threshold = 0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0.0:
            continue
        w_f = total - w_b
        if w_f == 0.0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            threshold = i
    return int(threshold)


def _label_side_from_center(cx_frac: float, cy_frac: float) -> str | None:
    """Return which image edge the label center is nearest, if in an edge band."""
    in_top = cy_frac < CY_BAND
    in_bottom = cy_frac > (1.0 - CY_BAND)
    in_left = cx_frac < CX_BAND
    in_right = cx_frac > (1.0 - CX_BAND)

    if not (in_top or in_bottom or in_left or in_right):
        return None

    # Pick the closest edge among bands the center qualifies for.
    edge_dist = {
        "top": cy_frac,
        "bottom": 1.0 - cy_frac,
        "left": cx_frac,
        "right": 1.0 - cx_frac,
    }
    candidates = [
        side for side, in_band in (
            ("top", in_top),
            ("bottom", in_bottom),
            ("left", in_left),
            ("right", in_right),
        )
        if in_band
    ]
    return min(candidates, key=lambda side: edge_dist[side])


def _border_touch_ok(
    x_bb: int,
    y_bb: int,
    bw: int,
    bh: int,
    W: int,
    H: int,
    label_side: str,
) -> bool:
    """Allow border contact on the edge where the label sits; reject other sides."""
    touches_left = x_bb <= 1
    touches_right = x_bb + bw >= W - 1
    touches_top = y_bb <= 1
    touches_bottom = y_bb + bh >= H - 1

    allowed = {
        "top": {"top"},
        "bottom": {"bottom"},
        "left": {"left"},
        "right": {"right"},
    }[label_side]

    bad = (
        (touches_left and "left" not in allowed)
        or (touches_right and "right" not in allowed)
        or (touches_top and "top" not in allowed)
        or (touches_bottom and "bottom" not in allowed)
    )
    return not bad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_label_rect(bgr_image: np.ndarray) -> LabelRect:
    """Detect frosted/yellow label as rotated rect; fail-safe passthrough."""
    H, W = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    nonblack = gray > BLACK_BG_THRESH
    inv = cv2.bitwise_not(gray)
    t = _otsu_on_mask(inv, nonblack)
    dark_mask = ((inv > t) & nonblack).astype(np.uint8) * 255

    b_channel = bgr_image[:, :, 0]
    g_channel = bgr_image[:, :, 1]
    r_channel = bgr_image[:, :, 2]
    yellow_mask = (
        (b_channel < BLUE_LOW)
        & (g_channel > LABEL_YELLOW_G_MIN)
        & (r_channel > LABEL_YELLOW_R_MIN)
    ).astype(np.uint8) * 255

    combined = cv2.bitwise_or(dark_mask, yellow_mask)

    nonblack_px = int(nonblack.sum())
    if nonblack_px > 0:
        combined_frac = float(combined.sum() / 255) / nonblack_px
        if combined_frac > LABEL_COMBINED_MAX_FRAC:
            return _null_rect()

    k = max(3, int(LABEL_MORPH_FRAC * W) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _null_rect()

    best: LabelRect | None = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / (W * H)

        rect = cv2.minAreaRect(cnt)
        (cx, cy_px), (rw, rh), angle = rect
        if rw * rh == 0:
            continue

        rectangularity = area / (rw * rh)
        span = max(rw, rh) / max(W, H)

        x_bb, y_bb, bw, bh = cv2.boundingRect(cnt)
        cy_frac = cy_px / H
        cx_frac = cx / W
        label_side = _label_side_from_center(cx_frac, cy_frac)
        if label_side is None:
            continue

        touches_border = not _border_touch_ok(
            x_bb, y_bb, bw, bh, W, H, label_side,
        )

        if not (AREA_MIN_FRAC <= area_frac <= AREA_MAX_FRAC):
            continue
        if rectangularity < RTY_MIN:
            continue
        if not (SPAN_MIN <= span <= SPAN_MAX):
            continue
        if touches_border:
            continue

        # Prefer tighter rectangles; tie-break toward nearest image edge.
        edge_dist = min(cx_frac, 1.0 - cx_frac, cy_frac, 1.0 - cy_frac)
        score = rectangularity + 0.01 * (1.0 - edge_dist)
        if score > best_score:
            best_score = score
            best = LabelRect(
                found=True,
                center=(cx, cy_px),
                size=(rw, rh),
                angle=angle,
                box_pts=cv2.boxPoints(rect).astype(np.float32),
                label_side=label_side,
            )

    return best if best is not None else _null_rect()


def translate_label_rect(
    rect: LabelRect,
    origin: tuple[int, int],
) -> LabelRect:
    """Shift label geometry from full-frame into crop-local coordinates."""
    if not rect.found:
        return rect
    ox, oy = origin
    box_pts = rect.box_pts.copy()
    box_pts[:, 0] -= ox
    box_pts[:, 1] -= oy
    cx, cy = rect.center
    return LabelRect(
        found=True,
        center=(cx - ox, cy - oy),
        size=rect.size,
        angle=rect.angle,
        box_pts=box_pts,
        label_side=rect.label_side,
    )


def apply_label_mask(bgr_image: np.ndarray, rect: LabelRect) -> np.ndarray:
    """White-fill expanded label rect; passthrough if not found."""
    out = bgr_image.copy()

    if not rect.found:
        return out

    H_img, W_img = bgr_image.shape[:2]
    cx, cy = rect.center
    w, h = rect.size
    angle = rect.angle

    # Expand the rect uniformly in both dimensions, keeping the same angle.
    min_margin_px = max(10, int(0.01 * min(W_img, H_img)))
    expanded_w = w * MARGIN_SCALE + min_margin_px
    expanded_h = h * MARGIN_SCALE + min_margin_px

    expanded_rect = ((cx, cy), (expanded_w, expanded_h), angle)
    box = cv2.boxPoints(expanded_rect)
    box = np.intp(box)  # integer pixel coords for fillPoly

    cv2.fillPoly(out, [box], (255, 255, 255))
    return out


def draw_label_overlay(
    bgr_image: np.ndarray,
    rect: LabelRect,
    out_path: Path,
) -> None:
    """Draw detection result onto a copy of the image and save as PNG.

    Green contour = detected label rectangle.
    Red contour = expanded fill rectangle (what apply_label_mask() fills).
    Text overlay shows found status, side, and angle.

    Leaf / diagnostic function - no other function calls this; TDD optional.

    Args:
        bgr_image: original BGR image
        rect: result of find_label_rect()
        out_path: destination path for the PNG (parent directory created if needed)
    """
    overlay = bgr_image.copy()

    if rect.found:
        # Green: the raw detected rect
        box = np.intp(rect.box_pts)
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), 3)

        # Red: the expanded fill rect
        H_img, W_img = bgr_image.shape[:2]
        min_margin_px = max(10, int(0.01 * min(W_img, H_img)))
        cx, cy = rect.center
        w, h = rect.size
        expanded_w = w * MARGIN_SCALE + min_margin_px
        expanded_h = h * MARGIN_SCALE + min_margin_px
        expanded_box = np.intp(
            cv2.boxPoints(((cx, cy), (expanded_w, expanded_h), rect.angle))
        )
        cv2.drawContours(overlay, [expanded_box], 0, (0, 0, 255), 3)

        label_text = (
            f"FOUND  side={rect.label_side}  angle={rect.angle:.1f}"
        )
        cv2.putText(overlay, label_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "NOT FOUND - passthrough", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
