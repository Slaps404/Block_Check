"""Soft-IoU alignment locked once per mask pair.

Production and diagnostics share this pose search so their overlays and scores
cannot drift or let each metric choose a different flattering alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ALIGN_SIZE = 256
COARSE_STEP = 10
REFINE_STEP = 2
REFINE_HALF_WINDOW = 10
SOFT_DILATION_RADIUS = 3
RMS_TARGET_RADIUS_FRAC = 0.13
RMS_OUTER_RADIUS_CAP_FRAC = 0.45
_SOFT_IOU_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (SOFT_DILATION_RADIUS * 2 + 1, SOFT_DILATION_RADIUS * 2 + 1),
)


@dataclass(frozen=True)
class LockedAlignment:
    best_angle: float
    best_flip: bool
    align_soft_iou: float
    mask_iou: float
    block_mask: np.ndarray
    aligned_slide_mask: np.ndarray


def render_alignment_overlay(
    alignment: LockedAlignment, caption: str
) -> np.ndarray:
    """Draw block in green and aligned slide in magenta with a caption bar."""
    canvas = np.full((ALIGN_SIZE, ALIGN_SIZE, 3), 24, dtype=np.uint8)
    block_layer = np.zeros_like(canvas)
    block_layer[alignment.block_mask > 0] = (40, 210, 40)
    slide_layer = np.zeros_like(canvas)
    slide_layer[alignment.aligned_slide_mask > 0] = (210, 40, 210)
    cv2.addWeighted(block_layer, 0.65, canvas, 1.0, 0, canvas)
    cv2.addWeighted(slide_layer, 0.65, canvas, 1.0, 0, canvas)
    bar = np.full((34, ALIGN_SIZE, 3), 24, dtype=np.uint8)
    cv2.putText(
        bar, caption, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (230, 230, 230), 1, cv2.LINE_AA,
    )
    return np.vstack((bar, canvas))


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("alignment masks must be two-dimensional")
    return (mask > 0).astype(np.uint8)


def radial_normalize_mask(mask: np.ndarray, mode: str = "rms") -> np.ndarray:
    """Center by area-weighted centroid and scale by radial tissue mass.

    Production uses RMS radius, so every foreground pixel contributes and a
    remote residual component pulls in proportion to its area and squared
    distance. ``mode="max"`` preserves the previous farthest-pixel estimator
    for reproducible diagnostics.
    """
    if mode not in {"rms", "max"}:
        raise ValueError("normalization mode must be 'rms' or 'max'")
    mask = _binary(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((ALIGN_SIZE, ALIGN_SIZE), dtype=np.uint8)
    cx, cy = float(xs.mean()), float(ys.mean())
    distances = np.hypot(xs - cx, ys - cy)
    outer_radius = float(distances.max())
    if mode == "max":
        scale = (ALIGN_SIZE * 0.39) / max(outer_radius, 1.0)
    else:
        rms_radius = float(np.sqrt(np.mean(np.square(distances))))
        scale = min(
            (ALIGN_SIZE * RMS_TARGET_RADIUS_FRAC) / max(rms_radius, 1.0),
            (ALIGN_SIZE * RMS_OUTER_RADIUS_CAP_FRAC) / max(outer_radius, 1.0),
        )
    center = (ALIGN_SIZE - 1) / 2.0
    matrix = np.array([
        [scale, 0.0, center - cx * scale],
        [0.0, scale, center - cy * scale],
    ])
    return (cv2.warpAffine(
        mask, matrix, (ALIGN_SIZE, ALIGN_SIZE), flags=cv2.INTER_NEAREST
    ) > 0).astype(np.uint8)


def transform_mask(mask: np.ndarray, angle: float, flip: bool) -> np.ndarray:
    """Return ``mask`` flipped left-right (optionally), then rotated."""
    source = np.fliplr(_binary(mask)) if flip else _binary(mask)
    height, width = source.shape
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0
    )
    return (cv2.warpAffine(
        source, matrix, (width, height), flags=cv2.INTER_NEAREST
    ) > 0).astype(np.uint8)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero((left > 0) | (right > 0))
    if union == 0:
        return 0.0
    return float(np.count_nonzero((left > 0) & (right > 0)) / union)


def _soft_iou(left: np.ndarray, right: np.ndarray) -> float:
    return _iou(
        cv2.dilate(_binary(left), _SOFT_IOU_KERNEL),
        cv2.dilate(_binary(right), _SOFT_IOU_KERNEL),
    )


def align_masks(block_mask: np.ndarray, slide_mask: np.ndarray) -> LockedAlignment:
    """Search full rotation + flip and return one reusable locked pose."""
    return align_normalized_masks(
        radial_normalize_mask(block_mask),
        radial_normalize_mask(slide_mask),
    )


def align_normalized_masks(
    block: np.ndarray, slide: np.ndarray
) -> LockedAlignment:
    """Lock a pose for masks already normalized and cached per specimen."""
    block = _binary(block)
    slide = _binary(slide)
    if not np.any(block) or not np.any(slide):
        return LockedAlignment(0.0, False, 0.0, 0.0, block, slide)

    best_angle = 0.0
    best_flip = False
    best_score = -1.0
    for angle in range(0, 360, COARSE_STEP):
        for flip in (False, True):
            score = _soft_iou(block, transform_mask(slide, angle, flip))
            if score > best_score:
                best_angle, best_flip, best_score = float(angle), flip, score

    for offset in range(-REFINE_HALF_WINDOW, REFINE_HALF_WINDOW + 1, REFINE_STEP):
        angle = (best_angle + offset) % 360.0
        score = _soft_iou(block, transform_mask(slide, angle, best_flip))
        if score > best_score:
            best_angle, best_score = angle, score

    aligned = transform_mask(slide, best_angle, best_flip)
    return LockedAlignment(
        best_angle=best_angle,
        best_flip=best_flip,
        align_soft_iou=best_score,
        mask_iou=_iou(block, aligned),
        block_mask=block,
        aligned_slide_mask=aligned,
    )
