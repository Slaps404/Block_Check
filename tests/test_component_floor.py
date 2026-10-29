"""Component-area floors in mask cleanup (block and slide).

Block: an absolute floor (MIN_BLOCK_COMPONENT_AREA, 500 px: drops the
block_015 185 px speck, keeps block_003's ~1383 px lobe) plus a decoupled
relative-to-largest floor (BLOCK_MIN_COMPONENT_REL_AREA = 0.003: keeps
block_003's real spec at 0.33% of largest, drops block_010's 0.24% edge
artifact). It is a separate knob from the slide floor because the same spec is
a smaller fraction of the largest blob on a block than on its slide.

Slide: an absolute floor (MIN_SLIDE_COMPONENT_AREA, 450 px) plus a
fraction-of-largest floor (SLIDE_MIN_COMPONENT_REL_AREA).
"""
from __future__ import annotations

import cv2
import numpy as np

import verify.segmentation as segmentation
from constants import MIN_BLOCK_COMPONENT_AREA
from verify.segmentation import (
    _component_filter,
    _slide_relative_area_floor,
    clean_tissue_components,
)


def _kept_areas(out: np.ndarray) -> list[int]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    return sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))


# --- Block absolute floor ------------------------------------------------

def test_block_floor_drops_subfloor_speck():
    """A 300 px speck (below 500) is dropped; a 6000 px blob is kept.

    On 600x600 the MIN_AREA_FRACTION term is ~2.9 px, so the absolute floor decides.
    """
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[50:70, 100:115] = 255      # 300 px speck
    mask[200:280, 200:275] = 255    # 6000 px real blob

    out = clean_tissue_components(mask, "block")

    assert out[240, 237] == 255, "6000 px real blob was wrongly dropped."
    assert out[60, 107] == 0, "300 px speck survived; absolute floor too low."
    assert len(_kept_areas(out)) == 1


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
    mask[100:130, 100:146] = 255    # 1380 px, block_003-scale real lobe

    out = clean_tissue_components(mask, "block")

    assert out[115, 120] == 255, "1380 px real fragment dropped; floor encroaches on tissue."


# --- Block relative floor ------------------------------------------------

# 490,000 px dominant blob on a 3600x3600 canvas (under BLOCK_AREA_MAX_FRAC).
# Relative floor at 0.003 = 1,470 px dominates the 500 px absolute floor.
_ARTIFACT_AREA = 34 * 34    # 1,156 px = 0.236% of largest (~block 010 edge artifact)
_REAL_SPEC_AREA = 40 * 40   # 1,600 px = 0.327% of largest (~block 003 real spec)


def _relative_floor_mask() -> np.ndarray:
    mask = np.zeros((3600, 3600), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (799, 799), 255, -1)
    cv2.rectangle(mask, (2000, 2000), (2033, 2033), 255, -1)
    cv2.rectangle(mask, (3000, 3000), (3039, 3039), 255, -1)
    return mask


def test_block_floor_keeps_real_spec_and_drops_edge_artifact():
    """At 0.003: block 003's 0.327% spec kept, block 010's 0.236% artifact dropped."""
    assert segmentation.BLOCK_MIN_COMPONENT_REL_AREA == 0.003  # guards the shipped value
    assert _ARTIFACT_AREA > MIN_BLOCK_COMPONENT_AREA  # absolute floor alone would keep it
    assert _REAL_SPEC_AREA > MIN_BLOCK_COMPONENT_AREA

    areas = _kept_areas(
        _component_filter(_relative_floor_mask(), min_area=MIN_BLOCK_COMPONENT_AREA, role="block")
    )

    assert len(areas) == 2, f"expected blob + real spec only, got {areas}"
    assert areas[0] == _REAL_SPEC_AREA, f"real spec (0.327%) must survive: {areas}"
    assert areas[1] > 400_000, f"dominant blob must survive: {areas}"


def test_block_floor_zero_disables_relative_floor(monkeypatch):
    """Setting the knob to 0.0 keeps every component above the absolute floor."""
    monkeypatch.setattr(segmentation, "BLOCK_MIN_COMPONENT_REL_AREA", 0.0)
    out = _component_filter(_relative_floor_mask(), min_area=MIN_BLOCK_COMPONENT_AREA, role="block")
    assert len(_kept_areas(out)) == 3


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


# --- Slide floors ---------------------------------------------------------

def test_slide_relative_floor_is_fraction_of_largest():
    assert _slide_relative_area_floor([10_000, 500, 200]) == 100
    assert _slide_relative_area_floor([]) == 0


def test_slide_filter_uses_absolute_min_with_relative_floor():
    # The 450 px absolute floor drops the 400 px spec, not the relative floor (36 px).
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (69, 69), 255, -1)    # 3600 px largest
    cv2.rectangle(mask, (80, 80), (99, 99), 255, -1)    # 400 px -> dropped
    cv2.rectangle(mask, (100, 10), (119, 34), 255, -1)  # 500 px -> kept

    assert _kept_areas(_component_filter(mask, min_area=1, role="slide")) == [500, 3600]


def test_slide_relative_component_floor_drops_tiny_specks_near_large_fragment():
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(mask, (90, 80), (340, 330), 255, -1)
    cv2.rectangle(mask, (25, 430), (40, 445), 255, -1)

    cleaned = _component_filter(mask, min_area=35, role="slide")

    assert cleaned[160, 160] == 255, "large tissue fragment was removed"
    assert cleaned[435, 30] == 0, "tiny relative speck survived slide cleanup"


def test_slide_relative_component_floor_keeps_sparse_small_fragments():
    """Sparse all-small slides should retain compact fragments above old floor."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(mask, (60, 70), (78, 90), 255, -1)
    cv2.rectangle(mask, (125, 120), (137, 132), 255, -1)

    cleaned = _component_filter(mask, min_area=35, role="slide")

    assert cleaned[80, 70] == 255, "largest sparse fragment was removed"
    assert cleaned[126, 126] == 255, "secondary sparse fragment was removed"


def test_slide_filter_keeps_large_curved_low_fill_tissue():
    """Thin, curved tissue must not be mistaken for a sparse bounding box."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    curve = np.array(
        [[30, 320], [90, 260], [150, 210], [210, 160], [270, 100], [330, 40]],
        dtype=np.int32,
    )
    cv2.polylines(mask, [curve], False, 255, thickness=3)  # bbox fill ~0.02

    out = _component_filter(mask, min_area=1, role="slide")

    assert np.count_nonzero(out) == np.count_nonzero(mask)


def test_slide_close_kernel_is_ellipse(monkeypatch):
    """Slide _postprocess builds its close kernel with MORPH_ELLIPSE."""
    captured: list[tuple] = []
    original = cv2.getStructuringElement

    def recording(shape, ksize, *args, **kwargs):
        captured.append((shape, ksize))
        return original(shape, ksize, *args, **kwargs)

    monkeypatch.setattr(segmentation.cv2, "getStructuringElement", recording)
    segmentation._postprocess(np.zeros((20, 20), dtype=np.uint8), min_area=1, role="slide", close_ksize=9)

    assert (cv2.MORPH_ELLIPSE, (9, 9)) in captured
