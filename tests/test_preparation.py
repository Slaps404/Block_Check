"""Tests for image preparation into comparable masks (issue #17).

All fixtures use synthetic in-memory images — no real image files needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from session.preparation import (
    PreparedSpecimen,
    PreparationFailure,
    _embed_crop_mask,
    _remove_opposite_tag_artifacts,
    prepare_specimen,
    prepare_specimen_from_image,
)
from slide.label_mask import LabelRect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Segmentation is color-based: block tissue is brown/tan, slide H&E tissue is
# pink/magenta. Synthetic fixtures must use those colors (a gray blob has no
# stain signal and is correctly rejected). Blobs are kept small: the block path
# rejects components covering >4% of the frame as cassette-sized.
_TISSUE_BROWN_BGR = (40, 90, 140)   # block: hue ~15, high saturation
_TISSUE_PINK_BGR = (200, 150, 230)  # slide H&E: hue ~161, magenta


def _block_tissue_image() -> np.ndarray:
    """800x600 BGR image: bright background, single brown tissue blob."""
    img = np.full((600, 800, 3), 230, dtype=np.uint8)
    cv2.circle(img, (400, 300), 60, _TISSUE_BROWN_BGR, -1)
    return img


def _blank_image() -> np.ndarray:
    """Uniform grey — no stain color, so no usable tissue."""
    return np.full((600, 800, 3), 128, dtype=np.uint8)


def _add_capture_texture(img: np.ndarray, *, std: float = 6.0) -> np.ndarray:
    """Add deterministic low-amplitude noise to model real-capture microtexture.

    The slide stain gate now includes a local-texture term (it rejects the
    perfectly smooth backlight vignette glow). A pristine solid-fill blob has
    zero interior texture, so it would be rejected like glow — which never
    happens for real tissue (sensor + specimen microstructure always carry
    texture). JPEG round-trip fixtures pass for this reason; in-memory pristine
    fixtures must add it explicitly. Fixed seed -> deterministic.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _slide_image_with_label() -> np.ndarray:
    """Slide image: gray label strip at the top, pink tissue blob lower half.

    The gray label has no stain color and must be excluded by the stain gate
    without any explicit label masking.
    """
    img = np.full((800, 600, 3), 230, dtype=np.uint8)
    # Label region in top quarter (gray — should NOT be segmented as tissue)
    cv2.rectangle(img, (0, 0), (600, 200), (80, 80, 80), -1)
    # Tissue blob in lower half
    cv2.circle(img, (300, 580), 100, _TISSUE_PINK_BGR, -1)
    return _add_capture_texture(img)


# ---------------------------------------------------------------------------
# Valid preparation
# ---------------------------------------------------------------------------

def test_valid_block_image_produces_specimen(tmp_path):
    img_path = tmp_path / "block.jpg"
    cv2.imwrite(str(img_path), _block_tissue_image())
    result = prepare_specimen(img_path, role="block")
    assert isinstance(result, PreparedSpecimen)
    assert result.role == "block"
    assert result.mask is not None
    assert result.mask.ndim == 2


def test_valid_slide_image_produces_specimen(tmp_path):
    img_path = tmp_path / "slide.jpg"
    cv2.imwrite(str(img_path), _slide_image_with_label())
    result = prepare_specimen(img_path, role="slide")
    assert isinstance(result, PreparedSpecimen)
    assert result.role == "slide"
    assert result.mask is not None


def test_block_roi_status_is_visible(tmp_path):
    img_path = tmp_path / "block.jpg"
    cv2.imwrite(str(img_path), _block_tissue_image())
    result = prepare_specimen(img_path, role="block")
    assert isinstance(result, PreparedSpecimen)
    assert isinstance(result.roi_ok, bool)
    assert isinstance(result.roi_reason, str)


def test_prepared_block_records_the_active_segmentation_backend(tmp_path):
    img_path = tmp_path / "block.jpg"
    cv2.imwrite(str(img_path), _block_tissue_image())

    result = prepare_specimen(img_path, role="block")

    assert isinstance(result, PreparedSpecimen)
    assert result.segmentation_backend == "classical"


# ---------------------------------------------------------------------------
# Preparation from in-memory image (seam used by tests and pipeline)
# ---------------------------------------------------------------------------

def test_prepare_specimen_from_image_block():
    result = prepare_specimen_from_image(_block_tissue_image(), role="block")
    assert isinstance(result, PreparedSpecimen)


def test_prepare_specimen_from_image_slide():
    result = prepare_specimen_from_image(_slide_image_with_label(), role="slide")
    assert isinstance(result, PreparedSpecimen)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_unreadable_file_returns_failure(tmp_path):
    bad_path = tmp_path / "nonexistent.jpg"
    result = prepare_specimen(bad_path, role="block")
    assert isinstance(result, PreparationFailure)
    assert result.role == "block"
    assert result.reason


def test_invalid_image_data_returns_failure(tmp_path):
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"not an image")
    result = prepare_specimen(bad_path, role="block")
    assert isinstance(result, PreparationFailure)
    assert result.reason


def test_blank_image_returns_failure():
    result = prepare_specimen_from_image(_blank_image(), role="block")
    assert isinstance(result, PreparationFailure)
    assert result.reason


def test_failure_reason_is_non_empty_string():
    result = prepare_specimen_from_image(np.zeros((10, 10, 3), dtype=np.uint8), role="slide")
    assert isinstance(result, PreparationFailure)
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Role awareness
# ---------------------------------------------------------------------------

def test_slide_preparation_does_not_crash_with_label_region():
    result = prepare_specimen_from_image(_slide_image_with_label(), role="slide")
    assert isinstance(result, PreparedSpecimen)


def test_block_and_slide_role_recorded_on_result():
    block = prepare_specimen_from_image(_block_tissue_image(), role="block")
    slide = prepare_specimen_from_image(_slide_image_with_label(), role="slide")
    assert block.role == "block"
    assert slide.role == "slide"


# ---------------------------------------------------------------------------
# Crop mask embed
# ---------------------------------------------------------------------------

def test_embed_crop_mask_places_crop_at_origin():
    full = np.zeros((100, 100), dtype=np.uint8)
    crop = np.full((20, 30), 255, dtype=np.uint8)

    out = _embed_crop_mask(crop, (10, 15), full.shape)

    assert out.shape == (100, 100)
    assert np.count_nonzero(out) == 20 * 30
    assert out[0, 0] == 0


# ---------------------------------------------------------------------------
# Slide border inset — backlight-edge fringe removal (see mvp_tuning_log.md)
# ---------------------------------------------------------------------------

def _slide_image_edge_fringe() -> np.ndarray:
    """Slide image: interior pink tissue blob + a pink strip hugging the right
    edge (simulates the colored backlight-boundary fringe). The right strip is
    vertically centered so the label detector does not treat it as a label."""
    img = np.full((800, 600, 3), 230, dtype=np.uint8)
    cv2.circle(img, (300, 400), 80, _TISSUE_PINK_BGR, -1)        # interior tissue
    cv2.rectangle(img, (585, 300), (600, 500), _TISSUE_PINK_BGR, -1)  # edge fringe
    return img




def test_opposite_tag_artifact_filter_handles_landscape_left_tag():
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.circle(mask, (720, 400), 80, 255, -1)      # real tissue
    cv2.circle(mask, (1080, 260), 12, 255, -1)     # far-end artifact dot
    cv2.circle(mask, (1080, 540), 12, 255, -1)     # far-end artifact dot

    rect = LabelRect(
        found=True,
        center=(130.0, 400.0),
        size=(180.0, 500.0),
        angle=0.0,
        box_pts=np.array([
            [40.0, 150.0],
            [220.0, 150.0],
            [220.0, 650.0],
            [40.0, 650.0],
        ], dtype=np.float32),
        label_side="left",
    )

    cleaned = _remove_opposite_tag_artifacts(mask, rect)

    assert cleaned[400, 720] > 0, "interior tissue was destroyed"
    assert cleaned[260, 1080] == 0, "upper far-end artifact survived"
    assert cleaned[540, 1080] == 0, "lower far-end artifact survived"
