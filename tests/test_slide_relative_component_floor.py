"""Slide mask cleanup uses a fraction of the largest fragment, not a raw px floor."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from verify.segmentation import _component_filter, _slide_relative_area_floor  # noqa: E402


def test_relative_floor_is_fraction_of_largest():
    assert _slide_relative_area_floor([10_000, 500, 200]) == 100
    assert _slide_relative_area_floor([]) == 0


def test_slide_filter_drops_specks_keeps_large_fragments():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (79, 79), 255, -1)
    cv2.rectangle(mask, (80, 80), (85, 85), 255, -1)
    cv2.rectangle(mask, (90, 10), (97, 17), 255, -1)

    out = _component_filter(mask, min_area=1, role="slide")

    num, _, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    areas = sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))
    assert len(areas) == 1
    assert areas[0] > 4000


def test_slide_filter_keeps_large_curved_low_fill_tissue():
    """Thin, curved tissue must not be mistaken for a sparse bounding box."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    curve = np.array(
        [[30, 320], [90, 260], [150, 210], [210, 160], [270, 100], [330, 40]],
        dtype=np.int32,
    )
    # Bbox fill is 0.0200. A real thin strand must be judged by its tissue
    # area and relative size, not by how much empty space its curve encloses.
    cv2.polylines(mask, [curve], False, 255, thickness=3)

    out = _component_filter(mask, min_area=1, role="slide")

    assert np.count_nonzero(out) == np.count_nonzero(mask)


def test_slide_filter_uses_absolute_min_with_relative_floor():
    # Absolute floor is MIN_SLIDE_COMPONENT_AREA (450, iteration 027); it is what
    # drops a fragment here, not the relative floor (3600 * 0.01 = 36 px).
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (69, 69), 255, -1)  # 3600 px largest
    cv2.rectangle(mask, (80, 80), (99, 99), 255, -1)  # 400 px spec, below 450 -> dropped
    cv2.rectangle(mask, (100, 10), (119, 34), 255, -1)  # 500 px, above 450 -> kept

    out = _component_filter(mask, min_area=1, role="slide")

    num, _, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    areas = sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))
    assert areas == [500, 3600]


def test_block_filter_still_uses_absolute_min_area():
    # Canvas must be large enough that a >=500 px blob stays under BLOCK_AREA_MAX_FRAC.
    mask = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (28, 28), 255, -1)   # 576 px, above floor
    cv2.rectangle(mask, (40, 40), (47, 47), 255, -1)  # 64 px, below floor

    from constants import MIN_BLOCK_COMPONENT_AREA  # noqa: E402

    out = _component_filter(mask, min_area=MIN_BLOCK_COMPONENT_AREA, role="block")
    assert np.count_nonzero(out) == 576
