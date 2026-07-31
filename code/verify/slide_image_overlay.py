"""Composite slide tissue RGB onto the full-cassette block crop."""

from __future__ import annotations

import cv2
import numpy as np

from block.roi_mask import find_cassette_bbox
from verify.locked_alignment import (
    ALIGN_SIZE,
    RMS_OUTER_RADIUS_CAP_FRAC,
    RMS_TARGET_RADIUS_FRAC,
)

# QA-visible default; revisit after operator feedback.
DEFAULT_SLIDE_OVERLAY_OPACITY = 0.80


def _radial_affine_matrix(mask: np.ndarray) -> np.ndarray:
    """Return the same center/scale warp used by ``radial_normalize_mask``."""
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return np.array([
            [1.0, 0.0, (ALIGN_SIZE - 1) / 2.0],
            [0.0, 1.0, (ALIGN_SIZE - 1) / 2.0],
        ])
    cx, cy = float(xs.mean()), float(ys.mean())
    distances = np.hypot(xs - cx, ys - cy)
    outer_radius = float(distances.max())
    rms_radius = float(np.sqrt(np.mean(np.square(distances))))
    scale = min(
        (ALIGN_SIZE * RMS_TARGET_RADIUS_FRAC) / max(rms_radius, 1.0),
        (ALIGN_SIZE * RMS_OUTER_RADIUS_CAP_FRAC) / max(outer_radius, 1.0),
    )
    center = (ALIGN_SIZE - 1) / 2.0
    return np.array([
        [scale, 0.0, center - cx * scale],
        [0.0, scale, center - cy * scale],
    ])


def _to_3x3(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = matrix
    return out


def _slide_to_block_matrix(
    slide_mask: np.ndarray,
    block_mask: np.ndarray,
    best_angle: float,
    best_flip: bool,
) -> np.ndarray:
    """2x3 OpenCV src→dst matrix: slide pixels → block frame (default warpAffine).

    Matches the ALIGN_SIZE pose pipeline (radial normalize → optional fliplr →
    rotate → inverse onto block) but resamples the native slide once.
    """
    m_s = _to_3x3(_radial_affine_matrix(slide_mask))
    m_b = _to_3x3(_radial_affine_matrix(block_mask))
    center = ((ALIGN_SIZE - 1) / 2.0, (ALIGN_SIZE - 1) / 2.0)
    rotation = _to_3x3(cv2.getRotationMatrix2D(center, best_angle, 1.0))
    # Same order as transform_mask: fliplr, then rotate.
    if best_flip:
        flip = np.array([
            [-1.0, 0.0, float(ALIGN_SIZE - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        forward = np.linalg.inv(m_b) @ rotation @ flip @ m_s
    else:
        forward = np.linalg.inv(m_b) @ rotation @ m_s
    return forward[:2, :]


def _cassette_crop_bounds(
    block_bgr: np.ndarray,
    block_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Inclusive-exclusive (y0, y1, x0, x1) for the full cassette, else mask bbox."""
    gray = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2GRAY)
    bbox = find_cassette_bbox(gray)
    height, width = block_bgr.shape[:2]
    if bbox is not None:
        x, y, w, h = bbox
        return y, y + h, x, x + w
    ys, xs = np.nonzero(block_mask > 0)
    if xs.size == 0:
        return 0, height, 0, width
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def build_slide_image_overlay(
    block_bgr: np.ndarray,
    slide_bgr: np.ndarray,
    block_mask: np.ndarray,
    slide_mask: np.ndarray,
    best_angle: float,
    best_flip: bool,
    opacity: float = DEFAULT_SLIDE_OVERLAY_OPACITY,
) -> np.ndarray:
    """Blend masked slide tissue onto the full-cassette block crop.

    Uses the locked pose (angle/flip) from scoring, but warps native-resolution
    slide pixels in one step. Output canvas is the full cassette bbox (not the
    inset paraffin window, not a tight tissue-mask crop).
    """
    if not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in (0, 1]")
    if block_bgr.ndim != 3 or block_bgr.shape[2] != 3:
        raise ValueError("block image must be BGR with shape (H, W, 3)")
    if slide_bgr.ndim != 3 or slide_bgr.shape[2] != 3:
        raise ValueError("slide image must be BGR with shape (H, W, 3)")

    # Defensive: test doubles may use tiny masks; production masks match HxW.
    if block_mask.shape[:2] != block_bgr.shape[:2]:
        block_mask = cv2.resize(
            block_mask, (block_bgr.shape[1], block_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if slide_mask.shape[:2] != slide_bgr.shape[:2]:
        slide_mask = cv2.resize(
            slide_mask, (slide_bgr.shape[1], slide_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    height, width = block_bgr.shape[:2]
    forward = _slide_to_block_matrix(
        slide_mask, block_mask, best_angle, best_flip
    )
    slide_on_block = cv2.warpAffine(
        slide_bgr,
        forward,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha_on_block = cv2.warpAffine(
        (slide_mask > 0).astype(np.uint8),
        forward,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    y0, y1, x0, x1 = _cassette_crop_bounds(block_bgr, block_mask)
    block_crop = block_bgr[y0:y1, x0:x1].astype(np.float32)
    slide_crop = slide_on_block[y0:y1, x0:x1].astype(np.float32)
    alpha_crop = alpha_on_block[y0:y1, x0:x1]

    alpha = np.zeros(alpha_crop.shape, dtype=np.float32)
    alpha[alpha_crop > 0] = opacity
    alpha3 = alpha[:, :, np.newaxis]
    blended = block_crop * (1.0 - alpha3) + slide_crop * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)
