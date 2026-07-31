"""Real-image regression tests locking in promoted block-growth behavior.

Topology-preserving block growth (margin 0, `code/block/growth.py::grow_block_mask`)
is wired into `prepare_specimen_from_image(img, "block")` for block masks. These
tests exercise the actual liver targets that motivated the promotion (blocks
32/33/34) and assert on the FINAL production mask:

  - Large-dark-region coverage (cassette-window-scoped, 8-connected `V <= 100`
    components >= 1500 px) is >= 95 %.
  - Connected-component count on the final mask matches the spec-expected
    count for each block (7 / 7 / 6).

See `docs/mvp_tuning_log/037_topology_preserving_block_growth_diagnostic.md`
for the validated numbers (coverage ~99-100%, counts 7/7/6).

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
from block.growth import _inside_window  # noqa: E402
from block.roi_mask import find_cassette_window  # noqa: E402


def _large_value_region(
    bgr_image: np.ndarray,
    window: tuple[int, int, int, int] | None,
    value_max: int,
    *,
    min_area: int = 1500,
) -> np.ndarray:
    """Cassette-window large dark region used by liver coverage regression."""
    value = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)[:, :, 2]
    raw = (_inside_window(value.shape, window) & (value <= value_max)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    region = np.zeros(value.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            region[labels == label] = True
    return region

# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------

_IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "pi_images_v3"

_FILES = {
    32: _IMG_DIR / "block_032_liver_DGK-EX00_01_HE.png",
    33: _IMG_DIR / "block_033_liver_DGK-EX00_01_HE.png",
    34: _IMG_DIR / "block_034_liver_DGK-EX00_01_HE.png",
}

_EXPECTED_COMPONENT_COUNT = {32: 7, 33: 7, 34: 6}

_ALL_PRESENT = all(p.exists() for p in _FILES.values())
pytestmark = pytest.mark.skipif(
    not _ALL_PRESENT,
    reason="One or more block 32-34 liver images not found in images/pi_images_v3/",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LARGE_DARK_VALUE_MAX = 100
_LARGE_DARK_MIN_AREA = 1500
_MIN_LARGE_DARK_COVERAGE = 0.95


def _load_run(block_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Load image and run end-to-end pipeline; return (bgr, mask uint8)."""
    bgr = cv2.imread(str(_FILES[block_num]))
    assert bgr is not None, f"cv2.imread failed for block {block_num}"
    result = prepare_specimen_from_image(bgr, "block")
    assert isinstance(result, PreparedSpecimen), (
        f"Block {block_num} pipeline returned failure: {result}"
    )
    return bgr, result.mask


def _component_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    return sum(
        1 for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) > 0
    )


# ---------------------------------------------------------------------------
# Test: liver targets (32, 33, 34) must reach >= 95% large-dark coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_num", [32, 33, 34])
def test_large_dark_coverage_liver_targets(block_num):
    """Liver blocks 32/33/34 must have >= 95% of their large-dark region masked."""
    bgr, mask = _load_run(block_num)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    window = find_cassette_window(gray)

    region = _large_value_region(bgr, window, _LARGE_DARK_VALUE_MAX, min_area=_LARGE_DARK_MIN_AREA)
    total = int(np.count_nonzero(region))
    if total == 0:
        pytest.skip(f"Block {block_num}: no large-dark region found — image may differ.")

    covered = int(np.count_nonzero((mask > 0) & region))
    frac = covered / total
    assert frac >= _MIN_LARGE_DARK_COVERAGE, (
        f"Block {block_num}: large-dark coverage is {frac:.1%} "
        f"(need >= {_MIN_LARGE_DARK_COVERAGE:.0%}). "
        f"Region pixels: {total}, covered: {covered}."
    )


# ---------------------------------------------------------------------------
# Test: liver targets (32, 33, 34) must match spec-expected component counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_num", [32, 33, 34])
def test_component_count_liver_targets(block_num):
    """Liver blocks 32/33/34 must have the spec-expected final component count."""
    _, mask = _load_run(block_num)
    expected = _EXPECTED_COMPONENT_COUNT[block_num]
    actual = _component_count(mask)
    assert actual == expected, (
        f"Block {block_num}: final mask has {actual} connected components "
        f"(expected {expected})."
    )
