"""Tests for detecting the physical slide rectangle in Pi-style captures."""
from __future__ import annotations

import cv2
import numpy as np

from constants import SLIDE_CROP_FAR_INSET_FRAC, SLIDE_CROP_TAG_INSET_FRAC
from slide.boundary import (
    compute_pre_seg_crop_roi,
    crop_bgr_for_segmentation,
    find_slide_rect,
)
from slide.label_mask import find_label_rect


def _synthetic_slide_frame(
    *,
    height: int = 900,
    width: int = 700,
    slide_left: int = 260,
    slide_top: int = 80,
    slide_width: int = 210,
    slide_height: int = 760,
) -> np.ndarray:
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(
        img,
        (slide_left, slide_top),
        (slide_left + slide_width, slide_top + slide_height),
        (222, 222, 222),
        -1,
    )
    cv2.rectangle(
        img,
        (slide_left, slide_top),
        (slide_left + slide_width, slide_top + 150),
        (145, 145, 145),
        -1,
    )
    cv2.rectangle(
        img,
        (slide_left, slide_top),
        (slide_left + slide_width, slide_top + slide_height),
        (188, 188, 188),
        2,
    )
    return img


def test_find_slide_rect_detects_shifted_slide_not_image_frame():
    result = find_slide_rect(_synthetic_slide_frame())

    assert result.found
    x, y, w, h = result.axis_aligned_bounds()
    assert 240 <= x <= 280
    assert 60 <= y <= 100
    assert 190 <= w <= 240
    assert 720 <= h <= 790


def test_find_slide_rect_returns_not_found_for_uniform_frame():
    img = np.full((900, 700, 3), 248, dtype=np.uint8)

    result = find_slide_rect(img)

    assert not result.found


def test_find_slide_rect_can_use_detected_label_geometry_when_glass_is_faint():
    img = np.full((900, 700, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (260, 80), (470, 230), (145, 145, 145), -1)

    label_rect = find_label_rect(img)
    result = find_slide_rect(img, label_rect=label_rect)

    assert result.found
    x, y, w, h = result.axis_aligned_bounds()
    assert 240 <= x <= 280
    assert 60 <= y <= 100
    assert 190 <= w <= 240
    assert 610 <= h <= 670


def test_find_slide_rect_infers_from_left_label_on_landscape_frame():
    img = np.full((700, 900, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (80, 260), (230, 470), (145, 145, 145), -1)

    label_rect = find_label_rect(img)
    assert label_rect.found
    assert label_rect.label_side == "left"

    result = find_slide_rect(img, label_rect=label_rect)

    assert result.found
    x, y, w, h = result.axis_aligned_bounds()
    assert 60 <= x <= 100
    assert 240 <= y <= 280
    assert 610 <= w <= 670
    assert 190 <= h <= 240


def test_compute_pre_seg_crop_roi_uses_label_side_for_tag_end():
    img = _synthetic_slide_frame()
    label_rect = find_label_rect(img)
    slide_rect = find_slide_rect(img, label_rect=label_rect)

    crop_roi = compute_pre_seg_crop_roi(slide_rect, label_rect, img.shape[:2])

    assert crop_roi.found
    assert crop_roi.tag_side == label_rect.label_side == "top"


def test_crop_bgr_for_segmentation_returns_offset_and_smaller_image():
    img = _synthetic_slide_frame()
    label_rect = find_label_rect(img)
    slide_rect = find_slide_rect(img, label_rect=label_rect)
    crop_roi = compute_pre_seg_crop_roi(slide_rect, label_rect, img.shape[:2])

    cropped, origin = crop_bgr_for_segmentation(img, crop_roi)

    assert origin[0] == crop_roi.crop_bounds[0]
    assert origin[1] == crop_roi.crop_bounds[1]
    assert cropped.shape[0] == crop_roi.crop_bounds[3]
    assert cropped.shape[1] == crop_roi.crop_bounds[2]
    assert cropped.size < img.size


def test_crop_insets_are_fractions_of_slide_long_side():
    img = np.full((700, 900, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (80, 260), (230, 470), (145, 145, 145), -1)

    label_rect = find_label_rect(img)
    slide_rect = find_slide_rect(img, label_rect=label_rect)
    crop_roi = compute_pre_seg_crop_roi(slide_rect, label_rect, img.shape[:2])

    sx, sy, sw, sh = slide_rect.axis_aligned_bounds()
    long_side = float(max(sw, sh))

    assert crop_roi.found
    assert crop_roi.tag_side == "left"
    assert crop_roi.tag_end_inset_px == SLIDE_CROP_TAG_INSET_FRAC * long_side
    assert crop_roi.far_end_inset_px == SLIDE_CROP_FAR_INSET_FRAC * long_side


def test_compute_pre_seg_crop_roi_not_found_without_label():
    img = _synthetic_slide_frame()
    slide_rect = find_slide_rect(img)

    crop_roi = compute_pre_seg_crop_roi(slide_rect, None, img.shape[:2])

    assert not crop_roi.found
