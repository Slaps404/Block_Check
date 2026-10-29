"""Tests for image preparation into comparable masks.

All fixtures use synthetic in-memory images — no real image files needed.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

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
    assert isinstance(result.roi_ok, bool)
    assert isinstance(result.roi_reason, str)
    assert result.segmentation_backend == "classical"


def test_valid_slide_image_produces_specimen(tmp_path):
    """Gray label strip must not stop slide preparation."""
    img_path = tmp_path / "slide.jpg"
    cv2.imwrite(str(img_path), _slide_image_with_label())
    result = prepare_specimen(img_path, role="slide")
    assert isinstance(result, PreparedSpecimen)
    assert result.role == "slide"
    assert result.mask is not None


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("contents", [None, b"not an image"], ids=["missing", "corrupt"])
def test_unreadable_file_returns_failure(tmp_path, contents):
    bad_path = tmp_path / "bad.jpg"
    if contents is not None:
        bad_path.write_bytes(contents)
    result = prepare_specimen(bad_path, role="block")
    assert isinstance(result, PreparationFailure)
    assert result.role == "block"
    assert result.reason


@pytest.mark.parametrize("role, image", [
    ("block", _blank_image()),
    ("slide", np.zeros((10, 10, 3), dtype=np.uint8)),
])
def test_blank_image_returns_failure(role, image):
    result = prepare_specimen_from_image(image, role=role)
    assert isinstance(result, PreparationFailure)
    assert result.reason

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
