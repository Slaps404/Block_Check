"""Block mask cleanup uses a decoupled relative-to-largest fragment floor.

History: issue #77 removed a redundant normalized-mask floor, exposing that the
block filter had no fraction-of-largest floor. #78/#80 mirrored the slide floor
(SLIDE_MIN_COMPONENT_REL_AREA = 0.01) onto blocks -- but that over-cleaned,
dropping real tissue the slide kept (block 003's bottom spec). Iteration 036 (a
near-miss A/B on metric_ablation.csv, 0.01 vs 0.003 vs 0.0) showed separation is
identical (17/22, no flips) across all three, so the floor is scoring-inert on
the frozen impostor set -- its only real effect is tissue fidelity. 0.003 (0.3%
of the largest block blob) is the chosen value: it KEEPS block 003's real spec
(0.36-0.48% of largest) while DROPPING block 010's spatially-outlying edge
artifact (0.24%) that a full 0.0 removal readmitted (costing 010 -0.091).

BLOCK_MIN_COMPONENT_REL_AREA is a SEPARATE knob from the slide floor because the
same physical spec is a smaller fraction of the largest blob on a block than on
its slide. The full-frame area-max guard remains; fill and aspect gates do not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import verify.segmentation as segmentation  # noqa: E402
from constants import MIN_BLOCK_COMPONENT_AREA  # noqa: E402
from verify.segmentation import _component_filter  # noqa: E402

# 700x700 = 490,000 px dominant blob on a 3600x3600 canvas (under BLOCK_AREA_MAX_FRAC
# = 4% = 518,400). Relative floor at 0.003 = 1,470 px, which dominates the 500 px
# absolute floor -- so it is the RELATIVE floor deciding these specks, not the
# absolute one (both specks exceed 500).
_CANVAS = 3600
_BLOB_AREA = 490_000
_ARTIFACT_AREA = 34 * 34    # 1,156 px = 0.236% of largest  (~block 010 edge artifact)
_REAL_SPEC_AREA = 40 * 40   # 1,600 px = 0.327% of largest  (~block 003 real spec)


def _build_mask() -> np.ndarray:
    mask = np.zeros((_CANVAS, _CANVAS), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (799, 799), 255, -1)      # 700x700 = 490,000 px
    cv2.rectangle(mask, (2000, 2000), (2033, 2033), 255, -1)  # 34x34 = 1,156 px (0.236%)
    cv2.rectangle(mask, (3000, 3000), (3039, 3039), 255, -1)  # 40x40 = 1,600 px (0.327%)
    return mask


def _kept_areas(out: np.ndarray) -> list[int]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    return sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))


def test_block_floor_keeps_real_spec_and_drops_edge_artifact():
    """At the production 0.003 floor: 0.327% spec kept, 0.236% artifact dropped.

    Both specks exceed the 500 px absolute floor, so the RELATIVE floor (0.3% of
    the 490k-px blob = 1,470 px) is what separates them -- exactly the block 003
    (keep) vs block 010 (drop) window from iteration 036.
    """
    assert segmentation.BLOCK_MIN_COMPONENT_REL_AREA == 0.003  # guards the shipped value
    assert _ARTIFACT_AREA > MIN_BLOCK_COMPONENT_AREA  # absolute floor alone would keep it
    assert _REAL_SPEC_AREA > MIN_BLOCK_COMPONENT_AREA

    out = _component_filter(_build_mask(), min_area=MIN_BLOCK_COMPONENT_AREA, role="block")
    areas = _kept_areas(out)

    assert len(areas) == 2, f"expected blob + real spec only, got {areas}"
    assert areas[0] == _REAL_SPEC_AREA, f"real spec (0.327%) must survive: {areas}"
    assert areas[1] > 400_000, f"dominant blob must survive: {areas}"
    assert _ARTIFACT_AREA not in areas, f"0.236% artifact must be dropped: {areas}"


def test_block_floor_zero_disables_relative_floor():
    """Setting the knob to 0.0 keeps every shape-valid component above the abs floor."""
    original = segmentation.BLOCK_MIN_COMPONENT_REL_AREA
    try:
        segmentation.BLOCK_MIN_COMPONENT_REL_AREA = 0.0
        out = _component_filter(_build_mask(), min_area=MIN_BLOCK_COMPONENT_AREA, role="block")
        areas = _kept_areas(out)
    finally:
        segmentation.BLOCK_MIN_COMPONENT_REL_AREA = original
    assert len(areas) == 3, f"0.0 should keep blob + both specks, got {areas}"


def test_block_cleanup_keeps_elongated_and_sparse_tissue_shapes():
    """Geometry alone must not reject a valid block tissue component."""
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (849, 51), 255, -1)  # 400:1 elongated strip
    cv2.line(mask, (100, 300), (899, 899), 255, thickness=3)  # sparse diagonal

    out = _component_filter(mask, min_area=MIN_BLOCK_COMPONENT_AREA, role="block")

    assert out[50, 400] == 255
    assert out[600, 500] == 255


def test_block_cleanup_retains_full_frame_maximum_area_guard():
    """Oversized components remain rejected even without geometry gates."""
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (399, 399), 255, -1)  # 9% of full frame

    out = _component_filter(mask, min_area=MIN_BLOCK_COMPONENT_AREA, role="block")

    assert not np.any(out)
