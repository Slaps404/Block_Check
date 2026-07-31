"""Regression tests for deterministic label-nearest slide-sheet selection."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from slide.label_mask import LabelRect  # noqa: E402
from slide.slot_selection import select_label_nearest_sheet  # noqa: E402


def _label(side: str, *, found: bool = True) -> LabelRect:
    return LabelRect(
        found=found,
        center=(0.0, 0.0),
        size=(0.0, 0.0),
        angle=0.0,
        box_pts=np.zeros((4, 2), dtype=np.float32),
        label_side=side,
    )


def _two_horizontal_lobes() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (60, 70), (90, 130), 255, -1)
    cv2.rectangle(mask, (110, 70), (140, 130), 255, -1)
    return mask


def test_selects_left_lobe_nearest_a_left_label():
    mask = _two_horizontal_lobes()

    result = select_label_nearest_sheet(mask, _label("left"))

    assert result.applied is True
    assert result.reason == "label_nearest_left"
    assert result.mask[100, 75] == 255
    assert result.mask[100, 125] == 0


def test_preserves_mask_when_divider_would_cut_a_tissue_strand():
    mask = _two_horizontal_lobes()
    cv2.line(mask, (90, 100), (110, 100), 255, 1)

    result = select_label_nearest_sheet(mask, _label("left"))

    assert result.applied is False
    assert result.reason == "divider_touches_tissue"
    assert np.array_equal(result.mask, mask)


def test_selects_top_lobe_nearest_a_top_label():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (70, 60), (130, 90), 255, -1)
    cv2.rectangle(mask, (70, 110), (130, 140), 255, -1)

    result = select_label_nearest_sheet(mask, _label("top"))

    assert result.applied is True
    assert result.reason == "label_nearest_top"
    assert result.mask[75, 100] == 255
    assert result.mask[125, 100] == 0


def test_selects_right_lobe_nearest_a_right_label():
    mask = _two_horizontal_lobes()

    result = select_label_nearest_sheet(mask, _label("right"))

    assert result.applied is True
    assert result.reason == "label_nearest_right"
    assert result.mask[100, 75] == 0
    assert result.mask[100, 125] == 255


def test_preserves_mask_when_label_is_not_detected():
    mask = _two_horizontal_lobes()

    result = select_label_nearest_sheet(mask, _label("none", found=False))

    assert result.applied is False
    assert result.reason == "label_not_found"
    assert np.array_equal(result.mask, mask)
