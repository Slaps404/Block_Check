"""Image preparation for v2 claimed-pair verification.

Load image → label mask (slide) → ROI crop → segment → post-clip → ROI check.
Returns PreparedSpecimen or PreparationFailure.

Code map
--------
PreparedSpecimen / PreparationFailure
    Success vs failure dataclasses.
prepare_specimen(path, role)
    Load from file; thin wrapper around prepare_specimen_from_image.
prepare_specimen_from_image(img, role)   ← pipeline entry
    Full prep chain. Block: cassette window + buffered pre-seg crop, dark gate
    scoped to window, connectivity-aware clip and growth in crop space, embed
    once. Slide: label/slide ROI on full frame, pre-seg crop, then label mask
    and segmentation on the crop only.
_inset_border, _check_roi, _embed_crop_mask
    Slide border fringe zeroing; block ROI sanity; paste crop mask to frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import MutableMapping, Union

import cv2
import numpy as np

from block.growth import grow_block_mask
from block.roi_mask import (
    keep_components_in_window,
    crop_bgr_at_bounds,
    expand_window_for_segmentation,
    find_cassette_window,
    window_relative_to_crop,
)
from constants import (
    MAX_ASPECT_RATIO,
    MAX_MASK_COVERAGE,
    MIN_CONTOUR_AREA,
    SLIDE_BORDER_INSET_FRAC,
    SLIDE_OPPOSITE_TAG_ARTIFACT_MAX_AREA,
    SLIDE_OPPOSITE_TAG_BAND_PAD_FRAC,
    SLIDE_OPPOSITE_TAG_SIDE_START_FRAC,
)
from verify.scale import block_pixel_scale_for, pixel_scale_for, scale_area_max
from verify.segmentation import active_segmentation_backend, segment_tissue
from slide.boundary import (
    compute_pre_seg_crop_roi,
    crop_bgr_for_segmentation,
    find_slide_rect,
)
from slide.label_mask import LabelRect, apply_label_mask, find_label_rect, translate_label_rect
from slide.slot_selection import select_label_nearest_sheet


@dataclass
class PreparedSpecimen:
    role: str
    mask: np.ndarray
    roi_ok: bool
    roi_reason: str = ""
    segmentation_backend: str = "classical"


@dataclass
class PreparationFailure:
    role: str
    reason: str


PreparedResult = Union[PreparedSpecimen, PreparationFailure]


def prepare_specimen(
    path: str | Path,
    role: str,
    *,
    slide_close_ksize: int | None = None,
    stage_timings: MutableMapping[str, int] | None = None,
) -> PreparedResult:
    """Load image from path and prepare into a comparable mask.

    slide_close_ksize: optional morphology close kernel for slide segmentation.
    """
    img = cv2.imread(str(path))
    if img is None:
        return PreparationFailure(role=role, reason=f"could not read image: {path}")
    return prepare_specimen_from_image(
        img, role, slide_close_ksize=slide_close_ksize, stage_timings=stage_timings
    )


def prepare_specimen_from_image(
    img: np.ndarray,
    role: str,
    *,
    slide_close_ksize: int | None = None,
    stage_timings: MutableMapping[str, int] | None = None,
) -> PreparedResult:
    """Prepare an in-memory BGR image into a comparable mask.

    slide_close_ksize: optional morphology close kernel for slide segmentation.
    """
    if img is None or img.size == 0:
        return PreparationFailure(role=role, reason="empty or null image array")

    label_rect: LabelRect | None = None
    block_window: tuple[int, int, int, int] | None = None

    if role == "slide":
        pixel_scale = pixel_scale_for(img.shape[1])
        label_rect = find_label_rect(img)
        slide_rect = find_slide_rect(img, label_rect=label_rect)
        crop_roi = compute_pre_seg_crop_roi(slide_rect, label_rect, img.shape[:2])
        if crop_roi.found:
            seg_img, origin = crop_bgr_for_segmentation(img, crop_roi)
            work = apply_label_mask(
                seg_img,
                translate_label_rect(label_rect, origin),
            )
            mask_crop = _segment_with_timing(
                work,
                role,
                stage_timings=stage_timings,
                slide_close_ksize=slide_close_ksize,
                pixel_scale=pixel_scale,
            )
            mask = _embed_crop_mask(mask_crop, origin, img.shape[:2])
        else:
            work = apply_label_mask(img, label_rect)
            mask = _segment_with_timing(
                work,
                role,
                stage_timings=stage_timings,
                slide_close_ksize=slide_close_ksize,
                pixel_scale=pixel_scale,
            )
    elif role == "block":
        pixel_scale = block_pixel_scale_for(img.shape[1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_window = find_cassette_window(gray, pixel_scale=pixel_scale)
        if block_window is not None:
            seg_bounds = expand_window_for_segmentation(block_window, img.shape[:2])
            seg_img, origin = crop_bgr_at_bounds(img, seg_bounds)
            rel_window = window_relative_to_crop(block_window, seg_bounds)
            frame_area = int(img.shape[0] * img.shape[1])
            mask_crop = _segment_with_timing(
                seg_img,
                role,
                stage_timings=stage_timings,
                block_window=rel_window,
                block_area_reference=frame_area,
                pixel_scale=pixel_scale,
            )
            mask_crop = keep_components_in_window(mask_crop, rel_window)
            mask_crop = grow_block_mask(
                seg_img, mask_crop, rel_window, pixel_scale=pixel_scale
            )
            mask = _embed_crop_mask(mask_crop, origin, img.shape[:2])
        else:
            mask = _segment_with_timing(
                img,
                role,
                stage_timings=stage_timings,
                block_window=None,
                pixel_scale=pixel_scale,
            )
            mask = keep_components_in_window(mask, block_window)
            mask = grow_block_mask(img, mask, block_window, pixel_scale=pixel_scale)
    else:
        return PreparationFailure(role=role, reason=f"unknown role: {role!r}")

    if role == "slide":
        mask = _inset_border(mask, SLIDE_BORDER_INSET_FRAC)
        mask = _remove_opposite_tag_artifacts(mask, label_rect, pixel_scale=pixel_scale)
        mask = select_label_nearest_sheet(mask, label_rect).mask

    coverage = np.count_nonzero(mask) / mask.size
    if coverage == 0:
        return PreparationFailure(role=role, reason="no tissue found in mask")
    if coverage > MAX_MASK_COVERAGE:
        return PreparationFailure(
            role=role,
            reason=f"mask covers {coverage:.0%} of image (degenerate segmentation)",
        )

    roi_ok, roi_reason = _check_roi(mask) if role == "block" else (True, "ok")

    return PreparedSpecimen(
        role=role,
        mask=mask,
        roi_ok=roi_ok,
        roi_reason=roi_reason,
        segmentation_backend=active_segmentation_backend(role),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _segment_with_timing(
    image: np.ndarray,
    role: str,
    *,
    stage_timings: MutableMapping[str, int] | None,
    **kwargs: object,
) -> np.ndarray:
    """Run the canonical segmenter, optionally recording only its own time."""
    started_ns = perf_counter_ns() if stage_timings is not None else None
    mask = segment_tissue(image, role, **kwargs)
    if started_ns is not None:
        stage_timings["segmentation_ms"] = int(
            round((perf_counter_ns() - started_ns) / 1_000_000)
        )
    return mask


def _embed_crop_mask(
    mask_crop: np.ndarray,
    origin: tuple[int, int],
    full_shape: tuple[int, ...],
) -> np.ndarray:
    """Paste a crop-sized mask into a full-frame canvas at ``origin``."""
    x0, y0 = origin
    h_img, w_img = int(full_shape[0]), int(full_shape[1])
    out = np.zeros((h_img, w_img), dtype=mask_crop.dtype)
    ch, cw = mask_crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return out
    y1 = min(h_img, y0 + ch)
    x1 = min(w_img, x0 + cw)
    out[y0:y1, x0:x1] = mask_crop[: y1 - y0, : x1 - x0]
    return out


def _inset_border(mask: np.ndarray, frac: float) -> np.ndarray:
    """Return a copy of mask with the outer `frac` band on every edge zeroed."""
    if frac <= 0:
        return mask
    h, w = mask.shape
    by, bx = int(h * frac), int(w * frac)
    out = np.zeros_like(mask)
    out[by:h - by, bx:w - bx] = mask[by:h - by, bx:w - bx]
    return out


def _remove_opposite_tag_artifacts(
    mask: np.ndarray,
    rect: LabelRect | None,
    *,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Drop small components on the slide end opposite the detected tag."""
    if rect is None or not rect.found or rect.label_side == "none":
        return mask

    h, w = mask.shape
    tag_x0, tag_y0 = np.min(rect.box_pts, axis=0)
    tag_x1, tag_y1 = np.max(rect.box_pts, axis=0)
    pad_x = w * SLIDE_OPPOSITE_TAG_BAND_PAD_FRAC
    pad_y = h * SLIDE_OPPOSITE_TAG_BAND_PAD_FRAC

    if rect.label_side in ("left", "right"):
        band_min = max(0.0, tag_y0 - pad_y)
        band_max = min(float(h), tag_y1 + pad_y)
        if rect.label_side == "left":
            side_start = (
                tag_x1
                + (w - tag_x1) * SLIDE_OPPOSITE_TAG_SIDE_START_FRAC
            )

            def in_opposite_side(cx: float, cy: float) -> bool:
                return cx >= side_start and band_min <= cy <= band_max
        else:
            side_start = tag_x0 * (1.0 - SLIDE_OPPOSITE_TAG_SIDE_START_FRAC)

            def in_opposite_side(cx: float, cy: float) -> bool:
                return cx <= side_start and band_min <= cy <= band_max
    else:
        band_min = max(0.0, tag_x0 - pad_x)
        band_max = min(float(w), tag_x1 + pad_x)
        if rect.label_side == "top":
            side_start = (
                tag_y1
                + (h - tag_y1) * SLIDE_OPPOSITE_TAG_SIDE_START_FRAC
            )

            def in_opposite_side(cx: float, cy: float) -> bool:
                return cy >= side_start and band_min <= cx <= band_max
        else:
            side_start = tag_y0 * (1.0 - SLIDE_OPPOSITE_TAG_SIDE_START_FRAC)

            def in_opposite_side(cx: float, cy: float) -> bool:
                return cy <= side_start and band_min <= cx <= band_max

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if num <= 1:
        return mask

    max_area = scale_area_max(SLIDE_OPPOSITE_TAG_ARTIFACT_MAX_AREA, pixel_scale)
    out = mask.copy()
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        cx, cy = centroids[i]
        if in_opposite_side(float(cx), float(cy)):
            out[labels == i] = 0
    return out


def _check_roi(mask: np.ndarray) -> tuple[bool, str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "no contours found"
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return False, f"largest contour too small: {area:.0f} px"
    x, y, w, h = cv2.boundingRect(largest)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        return False, f"ROI aspect ratio degenerate: {aspect:.1f}"
    return True, f"ok (area={area:.0f}, aspect={aspect:.1f})"
