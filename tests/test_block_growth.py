"""Unit tests for the production topology-preserving block growth core."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from block.growth import (  # noqa: E402
    build_candidate_regions,
    grow_block_mask,
    grow_topology_preserving,
)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def _two_seed_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = np.zeros((25, 25), dtype=np.uint8)
    seed[9:14, 3:7] = 255
    seed[9:14, 10:14] = 255

    selective = np.zeros(seed.shape, dtype=bool)
    selective[9:14, 3:14] = True

    loose = selective.copy()
    loose[7:16, 2:15] = True
    return seed, selective, loose


def test_margin_zero_keeps_seed_components_separate_across_loose_bridge():
    seed, selective, loose = _two_seed_masks()

    result = grow_topology_preserving(
        seed,
        selective,
        loose,
        bridge_margin=0,
        clean_labels=False,
    )

    assert _component_count(result.mask) == 2
    assert result.unioned_seed_pairs == ()
    assert np.any(result.mask[:, 8] == 0), "contested boundary was painted as tissue"


def test_seed_pixels_are_preserved_after_growth():
    seed, selective, loose = _two_seed_masks()

    result = grow_topology_preserving(
        seed,
        selective,
        loose,
        bridge_margin=0,
        clean_labels=False,
    )

    assert np.all(result.mask[seed > 0] == 255)


def test_candidate_regions_are_window_scoped_and_use_fixed_predicates():
    hsv = np.zeros((5, 5, 3), dtype=np.uint8)
    hsv[:, :] = (70, 200, 200)  # green wax: neither selective nor loose
    hsv[2, 2] = (5, 150, 90)  # selective dark-brown
    hsv[2, 3] = (70, 200, 60)  # loose V<=60 candidate
    hsv[0, 0] = (5, 150, 90)  # selective color, but outside the window
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    selective, loose = build_candidate_regions(bgr, window=(1, 1, 3, 3))

    assert selective[2, 2]
    assert loose[2, 2]
    assert loose[2, 3]
    assert not selective[0, 0]
    assert not loose[0, 0]


def test_grow_block_mask_returns_uint8_binary_mask():
    bgr = np.full((25, 25, 3), 200, dtype=np.uint8)
    bgr[9:14, 3:14] = (30, 30, 90)  # dark-brown selective/loose band
    seed = np.zeros((25, 25), dtype=np.uint8)
    seed[9:14, 3:7] = 255
    seed[9:14, 10:14] = 255

    mask = grow_block_mask(bgr, seed, window=(0, 0, 25, 25))

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert np.all(mask[seed > 0] == 255)


def test_grow_block_mask_is_a_no_op_when_window_is_none():
    bgr = np.full((10, 10, 3), 30, dtype=np.uint8)
    seed = np.zeros((10, 10), dtype=np.uint8)
    seed[3:6, 3:6] = 255

    mask = grow_block_mask(bgr, seed, window=None)

    assert np.array_equal(mask, seed)
