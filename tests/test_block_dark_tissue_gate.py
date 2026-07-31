"""Synthetic seam tests for the dark-absorbance tissue gate.

Covers:
  - Gate fires when window is given and blob is inside it.
  - Gate does NOT fire without a window (chroma-only fallback).
  - Gate is scoped: blob outside the window stays zero.
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

from verify import segmentation  # noqa: E402
from verify.segmentation import segment_tissue  # noqa: E402


@pytest.fixture(autouse=True)
def _use_classical_backend(monkeypatch):
    """This module verifies the classical color-segmentation branch only."""
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "classical")


# ---------------------------------------------------------------------------
# Shared image builder
# ---------------------------------------------------------------------------

_IMG_SIZE = 400
_BLOB_CENTER = (_IMG_SIZE // 2, _IMG_SIZE // 2)  # (cx, cy)
_BLOB_RADIUS = 30   # area ≈ 2827 px >> MIN_BLOCK_COMPONENT_AREA (60)

# Bright greenish wax background: high value, near-zero chroma -> passes NO
# chroma gate. BGR (120, 200, 120) -> HSV value ≈ 200 (well above dark gate).
_WAX_BGR = (120, 200, 120)

# Near-black dense tissue core: no chroma, HSV value ≈ 5 (well below 40).
_DARK_BGR = (5, 5, 5)


def _make_image() -> np.ndarray:
    """Return a 400x400 BGR image: greenish wax bg + near-black circle."""
    img = np.full((_IMG_SIZE, _IMG_SIZE, 3), _WAX_BGR, dtype=np.uint8)
    cv2.circle(img, _BLOB_CENTER, _BLOB_RADIUS, _DARK_BGR, thickness=-1)
    return img


def _blob_pixels(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the drawn dark blob."""
    blob = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(blob, _BLOB_CENTER, _BLOB_RADIUS, 255, thickness=-1)
    return blob > 0


# ---------------------------------------------------------------------------
# Test 1: dark gate fills core when window is given
# ---------------------------------------------------------------------------

def test_dark_tissue_core_filled_when_window_given():
    """Gate must fire: blob center and >= 90 % of blob pixels should be masked."""
    img = _make_image()

    # Window covers the whole image (includes the blob).
    win = (0, 0, _IMG_SIZE, _IMG_SIZE)

    result = segment_tissue(img, "block", block_window=win)

    cx, cy = _BLOB_CENTER
    assert result[cy, cx] == 255, (
        f"Blob center ({cx},{cy}) is 0 — dark gate did not fire inside window."
    )

    blob_mask = _blob_pixels(img)
    covered = (result[blob_mask] == 255).sum()
    frac = covered / blob_mask.sum()
    assert frac >= 0.90, (
        f"Only {frac:.1%} of blob pixels were masked (need >= 90%)."
    )


# ---------------------------------------------------------------------------
# Test 2: gate is inactive without a window
# ---------------------------------------------------------------------------

def test_dark_gate_inactive_without_window():
    """Without block_window, the chroma-only path must leave the near-black blob dark."""
    img = _make_image()

    result = segment_tissue(img, "block")  # no window -> dark gate disabled

    cx, cy = _BLOB_CENTER
    assert result[cy, cx] == 0, (
        f"Blob center ({cx},{cy}) is 255 without a window — "
        "dark gate fired without window (regression: old chroma-only behavior broken)."
    )


# ---------------------------------------------------------------------------
# Test 3: gate is scoped — blob outside window stays zero
# ---------------------------------------------------------------------------

def test_dark_gate_scoped_to_window():
    """Blob outside the given window must not be picked up by the dark gate."""
    img = _make_image()

    # Window placed in the top-left corner, well away from the blob at centre.
    win_x, win_y, win_w, win_h = 0, 0, 100, 100
    win = (win_x, win_y, win_w, win_h)

    result = segment_tissue(img, "block", block_window=win)

    cx, cy = _BLOB_CENTER
    assert result[cy, cx] == 0, (
        f"Blob center ({cx},{cy}) was masked even though it lies outside the "
        f"window (0,0,100,100) — dark gate is not properly scoped."
    )
