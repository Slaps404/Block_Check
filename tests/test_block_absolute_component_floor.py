"""Block absolute component-area floor (iteration 032).

The block component filter drops connected components below
MIN_BLOCK_COMPONENT_AREA (combined with the MIN_AREA_FRACTION term). Iteration
032 raises that floor from 60 to 500 px so the block_015 artifact speck (185 px)
is dropped, while staying ~2.7x below the smallest real tissue fragment in the
v3 set (block_003's ~1383 px bottom lobe), which must survive.

These are synthetic seam tests on clean_tissue_components — the same min_area
logic the production block path uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from verify.segmentation import clean_tissue_components  # noqa: E402


def _solid_block_mask() -> np.ndarray:
    """600x600 mask: a 300 px speck + a 6000 px real-tissue blob, well apart.

    On a 600x600 image the MIN_AREA_FRACTION term is 600*600*0.000008 = 2.9 px,
    so the effective floor is MIN_BLOCK_COMPONENT_AREA itself. Both blobs are
    solid squares (fill = 1.0, aspect ~1.0) so they clear the block fill/aspect
    gates and only the area floor decides their fate.
    """
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[50:70, 100:115] = 255      # 20 * 15 = 300 px speck
    mask[200:280, 200:275] = 255    # 80 * 75 = 6000 px real blob
    return mask


def test_block_floor_drops_subfloor_speck():
    """A 300 px component (below the 500 floor) is dropped; the 6000 px is kept."""
    out = clean_tissue_components(_solid_block_mask(), "block")

    assert out[240, 237] == 255, "6000 px real blob was wrongly dropped."
    assert out[60, 107] == 0, "300 px speck survived — absolute floor too low."

    num, _, _, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    assert num - 1 == 1, f"expected exactly 1 kept component, got {num - 1}."


def test_block_floor_scales_at_four_times_downsample():
    """A 32 px floor at 4x downscale keeps a 90 px tissue fragment."""
    mask = np.zeros((150, 150), dtype=np.uint8)
    mask[10:15, 10:16] = 255  # 30 px, below ceil(500 / 4^2) = 32
    mask[50:59, 50:60] = 255  # 90 px, scaled analogue of real tissue

    out = clean_tissue_components(mask, "block", pixel_scale=0.25)

    assert out[12, 12] == 0, "sub-floor speck survived at 4x downscale."
    assert out[54, 54] == 255, "real tissue was dropped at 4x downscale."


def test_block_floor_keeps_smallest_real_fragment():
    """A ~1380 px fragment (block_003-scale real tissue) must survive the floor."""
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[100:130, 100:146] = 255    # 30 * 46 = 1380 px, block_003-scale real lobe

    out = clean_tissue_components(mask, "block")

    assert out[115, 120] == 255, (
        "1380 px real-tissue fragment was dropped — floor encroaches on real tissue."
    )
