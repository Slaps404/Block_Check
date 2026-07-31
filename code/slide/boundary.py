"""Physical slide-rectangle detection and slide-corner artifact masking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from constants import (
    SLIDE_CROP_FAR_INSET_FRAC,
    SLIDE_CROP_TAG_INSET_FRAC,
    SLIDE_CROP_WIDTH_INSET_FRAC,
)


@dataclass
class SlideCropRoi:
    """Axis-aligned pre-segmentation crop inside the detected slide bounds."""

    found: bool
    box_pts: np.ndarray
    crop_bounds: tuple[int, int, int, int]
    tag_end_inset_px: float
    far_end_inset_px: float
    width_inset_px: float
    tag_side: str

    def axis_aligned_bounds(self) -> tuple[int, int, int, int]:
        return self.crop_bounds


@dataclass
class SlideRect:
    found: bool
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float
    box_pts: np.ndarray

    def axis_aligned_bounds(self) -> tuple[int, int, int, int]:
        if not self.found:
            return (0, 0, 0, 0)
        x, y, w, h = cv2.boundingRect(np.intp(self.box_pts))
        return int(x), int(y), int(w), int(h)


def _null_rect() -> SlideRect:
    return SlideRect(
        found=False,
        center=(0.0, 0.0),
        size=(0.0, 0.0),
        angle=0.0,
        box_pts=np.zeros((4, 2), dtype=np.float32),
    )


def find_slide_rect(
    bgr_image: np.ndarray,
    *,
    label_rect: object | None = None,
) -> SlideRect:
    """Detect the physical glass slide rectangle in a bright Pi frame.

    Uses label geometry when glass contrast is weak. Fails safe when no
    slide-like rectangle is found.
    """
    if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
        return _null_rect()

    inferred = _infer_slide_rect_from_label(label_rect, bgr_image.shape[:2])
    if inferred.found:
        return inferred

    h_img, w_img = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    border = np.concatenate([
        blurred[:20, :].ravel(),
        blurred[-20:, :].ravel(),
        blurred[:, :20].ravel(),
        blurred[:, -20:].ravel(),
    ])
    bg_level = float(np.median(border))
    darker_than_pad = (blurred < max(0.0, bg_level - 6.0)).astype(np.uint8) * 255

    k = max(9, int(0.035 * min(h_img, w_img)) | 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(darker_than_pad, cv2.MORPH_CLOSE, close_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, close_k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _null_rect()

    best: SlideRect | None = None
    best_score = -1.0
    image_area = h_img * w_img

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        rect_area = rw * rh
        if rect_area <= 0:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / max(short_side, 1.0)
        area_frac = rect_area / image_area
        rectangularity = area / rect_area

        if not (0.04 <= area_frac <= 0.70):
            continue
        if not (2.0 <= aspect <= 5.5):
            continue
        if rectangularity < 0.55:
            continue

        score = area * rectangularity
        if score > best_score:
            best_score = score
            best = SlideRect(
                found=True,
                center=(float(cx), float(cy)),
                size=(float(rw), float(rh)),
                angle=float(angle),
                box_pts=cv2.boxPoints(rect).astype(np.float32),
            )

    return best if best is not None else _null_rect()


def compute_pre_seg_crop_roi(
    slide_rect: SlideRect,
    label_rect: object | None,
    image_shape: tuple[int, int],
) -> SlideCropRoi:
    """Axis-aligned inner crop from slide bounds and ``SLIDE_CROP_*`` fractions.

    Tag end uses ``label_rect.label_side`` from ``find_label_rect()``.
    Consumed by ``preparation.py`` for the slide pre-segmentation crop.
    """
    empty = SlideCropRoi(
        found=False,
        box_pts=np.zeros((4, 2), dtype=np.float32),
        crop_bounds=(0, 0, 0, 0),
        tag_end_inset_px=0.0,
        far_end_inset_px=0.0,
        width_inset_px=0.0,
        tag_side="none",
    )
    if not slide_rect.found:
        return empty
    if label_rect is None or not getattr(label_rect, "found", False):
        return empty
    tag_side = getattr(label_rect, "label_side", "none")
    if tag_side not in {"top", "bottom", "left", "right"}:
        return empty

    h_img, w_img = image_shape
    sx, sy, sw, sh = slide_rect.axis_aligned_bounds()
    if sw <= 0 or sh <= 0:
        return empty

    long_side = float(max(sw, sh))
    short_side = float(min(sw, sh))
    tag_inset = SLIDE_CROP_TAG_INSET_FRAC * long_side
    far_inset = SLIDE_CROP_FAR_INSET_FRAC * long_side
    width_inset = SLIDE_CROP_WIDTH_INSET_FRAC * short_side

    if tag_side in {"left", "right"}:
        crop_h = sh - 2 * width_inset
        crop_w = sw - tag_inset - far_inset
        if tag_side == "left":
            crop_x = sx + tag_inset
        else:
            crop_x = sx + far_inset
        crop_y = sy + width_inset
    else:
        crop_w = sw - 2 * width_inset
        crop_h = sh - tag_inset - far_inset
        crop_x = sx + width_inset
        if tag_side == "top":
            crop_y = sy + tag_inset
        else:
            crop_y = sy + far_inset

    crop_w = int(round(crop_w))
    crop_h = int(round(crop_h))
    crop_x = int(round(crop_x))
    crop_y = int(round(crop_y))
    if crop_w <= 0 or crop_h <= 0:
        return empty

    crop_x = max(0, min(crop_x, w_img - 1))
    crop_y = max(0, min(crop_y, h_img - 1))
    crop_w = min(crop_w, w_img - crop_x)
    crop_h = min(crop_h, h_img - crop_y)
    if crop_w <= 0 or crop_h <= 0:
        return empty

    box_pts = np.array([
        [crop_x, crop_y],
        [crop_x + crop_w, crop_y],
        [crop_x + crop_w, crop_y + crop_h],
        [crop_x, crop_y + crop_h],
    ], dtype=np.float32)

    return SlideCropRoi(
        found=True,
        box_pts=box_pts,
        crop_bounds=(crop_x, crop_y, crop_w, crop_h),
        tag_end_inset_px=float(tag_inset),
        far_end_inset_px=float(far_inset),
        width_inset_px=float(width_inset),
        tag_side=tag_side,
    )


def crop_bgr_for_segmentation(
    bgr: np.ndarray,
    crop_roi: SlideCropRoi,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Axis-aligned crop for pre-segmentation ROI; returns (crop, (x0, y0))."""
    if bgr is None or bgr.size == 0 or not crop_roi.found:
        return bgr, (0, 0)
    x, y, w, h = crop_roi.crop_bounds
    if w <= 0 or h <= 0:
        return bgr, (0, 0)
    return bgr[y:y + h, x:x + w].copy(), (x, y)


def clear_opposite_label_corner_zones(
    mask: np.ndarray,
    slide_rect: SlideRect,
    label_side: str,
    *,
    offset_frac: float = 0.13,
    radius_frac: float = 0.13,
) -> np.ndarray:
    """Clear small slide-corner zones on the end opposite the detected tag."""
    if (
        mask is None or mask.size == 0
        or not slide_rect.found
        or label_side not in {"top", "bottom"}
    ):
        return mask

    ordered = _order_box_points(slide_rect.box_pts)
    tl, tr, br, bl = ordered
    width = float(np.linalg.norm(tr - tl))
    height = float(np.linalg.norm(bl - tl))
    short_side = max(1.0, min(width, height))
    offset = offset_frac * short_side
    radius = max(3, int(round(radius_frac * short_side)))

    u = (tr - tl) / max(width, 1.0)
    v = (bl - tl) / max(height, 1.0)

    if label_side == "top":
        centers = [
            bl + u * offset - v * offset,
            br - u * offset - v * offset,
        ]
    else:
        centers = [
            tl + u * offset + v * offset,
            tr - u * offset + v * offset,
        ]

    out = mask.copy()
    corner_mask = np.zeros_like(mask)
    for center in centers:
        c = (int(round(float(center[0]))), int(round(float(center[1]))))
        cv2.circle(corner_mask, c, radius, 255, thickness=cv2.FILLED)
    out[corner_mask > 0] = 0
    return out


def _order_box_points(points: np.ndarray) -> np.ndarray:
    """Return box points ordered TL, TR, BR, BL for near-vertical slides."""
    pts = np.asarray(points, dtype=np.float32)
    by_y = sorted(pts, key=lambda p: (float(p[1]), float(p[0])))
    top = sorted(by_y[:2], key=lambda p: float(p[0]))
    bottom = sorted(by_y[2:], key=lambda p: float(p[0]))
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _infer_slide_rect_from_label(
    label_rect: object | None,
    image_shape: tuple[int, int],
    *,
    slide_aspect: float = 3.0,
) -> SlideRect:
    """Infer full slide geometry from a detected tag when glass contrast is weak."""
    if label_rect is None or not getattr(label_rect, "found", False):
        return _null_rect()
    label_side = getattr(label_rect, "label_side", "none")
    if label_side not in {"top", "bottom", "left", "right"}:
        return _null_rect()

    h_img, w_img = image_shape
    label_box = _order_box_points(
        np.asarray(getattr(label_rect, "box_pts"), dtype=np.float32)
    )
    tl, tr, br, bl = label_box
    along_edge = float(np.linalg.norm(tr - tl))
    into_slide = float(np.linalg.norm(bl - tl))
    if along_edge <= 0 or into_slide <= 0:
        return _null_rect()

    if label_side in {"top", "bottom"}:
        width = along_edge
        label_height = into_slide
        v = (bl - tl) / max(label_height, 1.0)
        slide_height = width * slide_aspect
        if label_side == "top":
            slide_box = np.array(
                [tl, tr, tr + v * slide_height, tl + v * slide_height],
                dtype=np.float32,
            )
        else:
            slide_box = np.array(
                [bl - v * slide_height, br - v * slide_height, br, bl],
                dtype=np.float32,
            )
    else:
        label_span = into_slide
        label_depth = along_edge
        slide_length = label_span * slide_aspect
        if label_side == "left":
            u = (tr - tl) / max(label_depth, 1.0)
            slide_box = np.array(
                [tl, bl, bl + u * slide_length, tl + u * slide_length],
                dtype=np.float32,
            )
        else:
            u = (tl - tr) / max(label_depth, 1.0)
            slide_box = np.array(
                [tr, br, br + u * slide_length, tr + u * slide_length],
                dtype=np.float32,
            )

    slide_box[:, 0] = np.clip(slide_box[:, 0], 0, w_img - 1)
    slide_box[:, 1] = np.clip(slide_box[:, 1], 0, h_img - 1)
    rect = cv2.minAreaRect(slide_box.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    return SlideRect(
        found=True,
        center=(float(cx), float(cy)),
        size=(float(rw), float(rh)),
        angle=float(angle),
        box_pts=cv2.boxPoints(rect).astype(np.float32),
    )
