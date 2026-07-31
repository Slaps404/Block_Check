"""Tests for the parametric slide close kernel (Hull Segmentation Experiment).

Steps 1+2: confirms the kernel parameter threads correctly through the call
chain and that default behavior is byte-identical to the prior production path.

All four tests use real images where available; tests 1, 3, 4 use synthetic
images so no file I/O is needed for the baseline suite.
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

from session.preparation import prepare_specimen_from_image, PreparedSpecimen  # noqa: E402
from verify.segmentation import segment_tissue  # noqa: E402

# ---------------------------------------------------------------------------
# Shared synthetic fixture helpers
# ---------------------------------------------------------------------------

_TISSUE_PINK_BGR = (200, 150, 230)   # H&E pink: passes slide stain gate
_TISSUE_BROWN_BGR = (40, 90, 140)    # block brown: passes block gate


def _add_capture_texture(img: np.ndarray, *, std: float = 6.0) -> np.ndarray:
    """Add deterministic low-amplitude noise to model real-capture microtexture.

    The slide stain gate's local-texture term rejects a perfectly smooth
    fill (zero interior variance), which real captures never have. See the
    identical helper/rationale in tests/test_preparation.py.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _slide_image() -> np.ndarray:
    """Synthetic 800x600 slide image with a pink tissue blob (passes stain gate)."""
    img = np.full((600, 800, 3), 230, dtype=np.uint8)
    # Large enough to survive MIN_SLIDE_COMPONENT_AREA after the close
    cv2.circle(img, (400, 300), 80, _TISSUE_PINK_BGR, -1)
    return _add_capture_texture(img)


def _block_image() -> np.ndarray:
    """Synthetic 600x800 block image with a brown tissue blob."""
    img = np.full((600, 800, 3), 230, dtype=np.uint8)
    cv2.circle(img, (400, 300), 60, _TISSUE_BROWN_BGR, -1)
    return img


# ---------------------------------------------------------------------------
# Test 1: default (no override) is byte-identical to explicit ksize=5
# ---------------------------------------------------------------------------

def test_slide_default_identical_to_ksize5():
    """With no slide_close_ksize, the output must be byte-identical to ksize=5.

    This asserts clean attribution: when we later sweep kernels, any mask
    change is *only* due to the parameter change, not an implementation drift.
    """
    img = _slide_image()

    mask_default = segment_tissue(img.copy(), "slide")
    mask_explicit5 = segment_tissue(img.copy(), "slide", slide_close_ksize=5)

    assert np.array_equal(mask_default, mask_explicit5), (
        "Default path (no ksize) produced a different mask than explicit ksize=5. "
        "SLIDE_CLOSE_KERNEL constant must equal 5."
    )


# ---------------------------------------------------------------------------
# Test 2: larger kernel reduces component count on set_09 slide
# ---------------------------------------------------------------------------

_SET09_PATH = (
    Path(__file__).resolve().parent.parent
    / "images"
    / "pi_images"
    / "set_09_slide_lungs_MT_wt5_WO7842.jpg"
)


@pytest.mark.skipif(
    not _SET09_PATH.exists(),
    reason="set_09 MT slide not found in images/pi_images/",
)
def test_larger_kernel_reduces_fragmentation_set09():
    """slide_close_ksize=35 must produce fewer components than ksize=5 on set_09.

    Set 09 is the known bad case: 26 components at ksize=5 vs its block's 4.
    A large close must merge those fragments.
    """
    img = cv2.imread(str(_SET09_PATH))
    assert img is not None, f"Could not read {_SET09_PATH}"

    mask5 = segment_tissue(img, "slide", slide_close_ksize=5)
    mask35 = segment_tissue(img, "slide", slide_close_ksize=35)

    def count_components(mask: np.ndarray, min_area: int = 20) -> int:
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return sum(
            1 for i in range(1, num)
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area
        )

    n5 = count_components(mask5)
    n35 = count_components(mask35)

    assert n35 < n5, (
        f"Expected ksize=35 to reduce component count vs ksize=5, "
        f"but got n5={n5}, n35={n35}"
    )


# ---------------------------------------------------------------------------
# Test 3: block masks are unaffected by slide_close_ksize
# ---------------------------------------------------------------------------

def test_block_unaffected_by_slide_close_ksize():
    """Block segmentation must be identical regardless of slide_close_ksize.

    The parameter only threads through the slide path; the block path ignores it.
    """
    img = _block_image()

    mask_no_param = segment_tissue(img.copy(), "block")
    mask_with_5 = segment_tissue(img.copy(), "block", slide_close_ksize=5)
    mask_with_35 = segment_tissue(img.copy(), "block", slide_close_ksize=35)

    assert np.array_equal(mask_no_param, mask_with_5), (
        "Block mask changed when slide_close_ksize=5 was passed — param leaked into block path."
    )
    assert np.array_equal(mask_no_param, mask_with_35), (
        "Block mask changed when slide_close_ksize=35 was passed — param leaked into block path."
    )


# ---------------------------------------------------------------------------
# Test 4: close structuring element is ELLIPSE (mock-based check)
# ---------------------------------------------------------------------------

def test_close_kernel_is_ellipse(monkeypatch):
    """_postprocess builds the close kernel with cv2.MORPH_ELLIPSE, not MORPH_RECT.

    Strategy: monkeypatch cv2.getStructuringElement inside segmentation to
    capture the shape argument passed when building the close kernel.  We run
    _postprocess with close_ksize=9 and verify that at least one call passed
    cv2.MORPH_ELLIPSE with size (9, 9).

    This is the cleanest assertion: it directly verifies the constant used,
    not fragile output equality on synthetic images where both kernels collapse
    to the same mask.
    """
    import verify.segmentation as seg_module

    captured_calls: list[tuple] = []
    original_gse = cv2.getStructuringElement

    def recording_gse(shape, ksize, *args, **kwargs):
        captured_calls.append((shape, ksize))
        return original_gse(shape, ksize, *args, **kwargs)

    monkeypatch.setattr(seg_module.cv2, "getStructuringElement", recording_gse)

    # Provide a minimal mask; actual pixel content doesn't matter for this test.
    mask = np.zeros((20, 20), dtype=np.uint8)

    seg_module._postprocess(mask, min_area=1, role="slide", close_ksize=9)

    # Collect all calls that built a (9, 9) kernel.
    ksize9_calls = [(shape, sz) for shape, sz in captured_calls if sz == (9, 9)]

    assert any(shape == cv2.MORPH_ELLIPSE for shape, _ in ksize9_calls), (
        f"Expected at least one getStructuringElement(MORPH_ELLIPSE, (9,9)) call. "
        f"Calls with ksize=(9,9): {ksize9_calls}"
    )


# ---------------------------------------------------------------------------
# Test 5: slide_close_ksize threads all the way through prepare_specimen_from_image
# ---------------------------------------------------------------------------

def test_prepare_specimen_from_image_passes_ksize():
    """prepare_specimen_from_image with ksize=5 equals no-ksize (identity chain).

    Built at full resolution (pixel_scale == 1.0): the identity only holds
    when SLIDE_CLOSE_KERNEL(=5) is unscaled, since an explicit slide_close_ksize
    is used raw while the default is scaled by pixel_scale_for(frame_width).
    """
    img = cv2.resize(_slide_image(), (4056, 3040), interpolation=cv2.INTER_NEAREST)

    result_default = prepare_specimen_from_image(img.copy(), "slide")
    result_ksize5 = prepare_specimen_from_image(img.copy(), "slide", slide_close_ksize=5)

    assert isinstance(result_default, PreparedSpecimen)
    assert isinstance(result_ksize5, PreparedSpecimen)
    assert np.array_equal(result_default.mask, result_ksize5.mask), (
        "prepare_specimen_from_image default != explicit ksize=5: chain is broken."
    )


def test_slide_relative_component_floor_drops_tiny_specks_near_large_fragment():
    """Slide cleanup should remove components tiny relative to real tissue."""
    import verify.segmentation as seg_module

    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(mask, (90, 80), (340, 330), 255, -1)
    cv2.rectangle(mask, (25, 430), (40, 445), 255, -1)

    cleaned = seg_module._component_filter(
        mask,
        min_area=35,
        role="slide",
    )

    assert cleaned[160, 160] == 255, "large tissue fragment was removed"
    assert cleaned[435, 30] == 0, "tiny relative speck survived slide cleanup"


def test_slide_relative_component_floor_keeps_sparse_small_fragments():
    """Sparse all-small slides should retain compact fragments above old floor."""
    import verify.segmentation as seg_module

    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(mask, (60, 70), (78, 90), 255, -1)
    cv2.rectangle(mask, (125, 120), (137, 132), 255, -1)

    cleaned = seg_module._component_filter(
        mask,
        min_area=35,
        role="slide",
    )

    assert cleaned[80, 70] == 255, "largest sparse fragment was removed"
    assert cleaned[126, 126] == 255, "secondary sparse fragment was removed"
