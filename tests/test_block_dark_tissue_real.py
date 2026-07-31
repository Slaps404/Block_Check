"""Real-image regression tests for the dark-absorbance tissue gate.

Verifies that:
  - Blocks 20 and 21 (dark-core spleen/liver & lung) now have >= 90 % of their
    dark-core pixels covered by the mask.
  - Block 22 (knee, low-coverage regression guard) still has very low total
    mask coverage (< 2.5 %).

Skips automatically when the image directory / files are missing (CI without
the real image dataset will not fail).
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
from block.roi_mask import find_cassette_window  # noqa: E402

# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------

_IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "pi_images_v3"

_FILES = {
    20: _IMG_DIR / "block_020_spleen-liver_SM4_01_HE.png",
    21: _IMG_DIR / "block_021_lung_SM12_01_HE.png",
    22: _IMG_DIR / "block_022_knee_SM13_01_HE.png",
}

# Skip the whole module if any required image is missing.
_ALL_PRESENT = all(p.exists() for p in _FILES.values())
pytestmark = pytest.mark.skipif(
    not _ALL_PRESENT,
    reason="One or more block 20-22 images not found in images/pi_images_v3/",
)

# ---------------------------------------------------------------------------
# Dark-core ground-truth helper
# ---------------------------------------------------------------------------

_DARK_CORE_MIN_AREA = 1500   # px — keep only large dark connected components
_DARK_THRESHOLD = 25          # HSV value < 25 defines "dark core"


def _dark_core_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of dark-core pixels inside the cassette window.

    Definition (matches the diagnosed bug):
      - Find cassette window via find_cassette_window.
      - Build boolean `inside` for the window bbox.
      - Mark pixels where HSV value < _DARK_THRESHOLD AND inside window.
      - Run connected components; keep components with area >= _DARK_CORE_MIN_AREA.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    window = find_cassette_window(gray)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    h_img, w_img = v.shape

    inside = np.zeros((h_img, w_img), dtype=bool)
    if window is not None:
        x, y, w, h_win = window
        inside[y:y + h_win, x:x + w] = True

    dark_raw = (inside & (v < _DARK_THRESHOLD)).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark_raw, connectivity=8)
    dark_core = np.zeros((h_img, w_img), dtype=bool)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= _DARK_CORE_MIN_AREA:
            dark_core[labels == i] = True

    return dark_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run(block_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Load image and run end-to-end pipeline; return (bgr, mask uint8)."""
    bgr = cv2.imread(str(_FILES[block_num]))
    assert bgr is not None, f"cv2.imread failed for block {block_num}"
    result = prepare_specimen_from_image(bgr, "block")
    assert isinstance(result, PreparedSpecimen), (
        f"Block {block_num} pipeline returned failure: {result}"
    )
    return bgr, result.mask


# ---------------------------------------------------------------------------
# Test: dark-core blocks (20, 21) must be >= 90 % covered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_num", [20, 21])
def test_dark_core_coverage_dark_blocks(block_num):
    """Dark-core blocks 20 and 21 must have >= 90 % of dark-core pixels masked."""
    bgr, mask = _load_run(block_num)
    dark_core = _dark_core_mask(bgr)

    total_dark = dark_core.sum()
    if total_dark == 0:
        pytest.skip(f"Block {block_num}: no dark-core pixels found — image may differ.")

    covered = (dark_core & (mask > 0)).sum()
    frac = covered / total_dark
    assert frac >= 0.90, (
        f"Block {block_num}: dark-core coverage is {frac:.1%} (need >= 90 %). "
        f"Dark-core pixels: {total_dark}, covered: {covered}."
    )


# ---------------------------------------------------------------------------
# Test: block 22 (knee, low-coverage guard) must not blow up in coverage
# ---------------------------------------------------------------------------

_LIGHT_COVERAGE_MAX = 0.025   # 2.5 % — baseline 22: 1.53 %


@pytest.mark.parametrize("block_num", [22])
def test_total_coverage_light_blocks(block_num):
    """Block 22 (knee) must stay below 2.5 % total coverage."""
    bgr, mask = _load_run(block_num)
    coverage = (mask > 0).mean()
    assert coverage < _LIGHT_COVERAGE_MAX, (
        f"Block {block_num}: coverage {coverage:.2%} exceeds regression guard "
        f"{_LIGHT_COVERAGE_MAX:.1%}. Dark gate may be firing outside its scope."
    )
