"""Tests for slide_label_mask — portrait top + landscape left label detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from slide.label_mask import (  # noqa: E402
    LabelRect,
    apply_label_mask,
    find_label_rect,
)


def _make_portrait_top_label(
    h: int = 800,
    w: int = 600,
    label_top: int = 40,
    label_bottom: int = 110,
    label_left: int = 100,
    label_right: int = 420,
    label_gray: int = 170,
    bg_gray: int = 250,
) -> np.ndarray:
    """Portrait slide with frosted label strip at top (iPhone-style)."""
    img = np.full((h, w, 3), bg_gray, dtype=np.uint8)
    img[label_top:label_bottom, label_left:label_right] = label_gray
    img[label_top + 8:label_top + 18, label_left + 20:label_left + 100] = 20
    img[label_top + 24:label_top + 34, label_left + 20:label_left + 80] = 20
    cv2.circle(img, (w // 2, int(h * 0.7)), 80, (40, 40, 40), -1)
    return img


def _make_landscape_left_label(
    w: int = 1200,
    h: int = 800,
    label_left: int = 40,
    label_right: int = 220,
    label_top: int = 150,
    label_bottom: int = 650,
    label_gray: int = 170,
    bg_gray: int = 250,
) -> np.ndarray:
    """Landscape slide with frosted label strip on the left (Pi v3-style)."""
    img = np.full((h, w, 3), bg_gray, dtype=np.uint8)
    img[label_top:label_bottom, label_left:label_right] = label_gray
    img[label_top + 20:label_top + 120, label_left + 10:label_left + 90] = 20
    img[label_top + 140:label_top + 200, label_left + 10:label_left + 70] = 20
    cv2.circle(img, (int(w * 0.65), h // 2), 80, (40, 40, 40), -1)
    return img


class TestFindLabelRectPortrait:
    def test_detects_top_label(self):
        img = _make_portrait_top_label()
        result = find_label_rect(img)
        assert result.found
        assert result.label_side == "top"

    def test_center_tissue_not_selected(self):
        img = np.full((800, 600, 3), 240, dtype=np.uint8)
        cv2.circle(img, (300, 400), 150, (40, 40, 40), -1)
        result = find_label_rect(img)
        assert not result.found


class TestFindLabelRectLandscape:
    def test_detects_left_label(self):
        img = _make_landscape_left_label()
        result = find_label_rect(img)
        assert result.found
        assert result.label_side == "left"

    def test_left_center_not_on_tissue(self):
        img = _make_landscape_left_label()
        result = find_label_rect(img)
        assert result.found
        cx, _ = result.center
        assert cx < img.shape[1] * 0.35


class TestApplyLabelMask:
    def test_passthrough_when_not_found(self):
        img = _make_portrait_top_label()
        not_found = LabelRect(
            found=False, center=(0, 0), size=(0, 0), angle=0.0,
            box_pts=np.zeros((4, 2), dtype=np.float32), label_side="none",
        )
        result = apply_label_mask(img, not_found)
        np.testing.assert_array_equal(result, img)

    def test_fill_covers_landscape_label(self):
        img = _make_landscape_left_label()
        rect = find_label_rect(img)
        assert rect.found
        result = apply_label_mask(img, rect)
        cy = (150 + 650) // 2
        cx = (40 + 220) // 2
        assert all(ch >= 200 for ch in result[cy, cx])
