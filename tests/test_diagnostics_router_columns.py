"""TDD: diagnostic CSV must carry the exact production router signal.

The production router (scorer.score_pair_result_routed) picks mask_iou vs
point_layout based on min(block_frac, slide_frac) of each specimen's
normalized_mask (from build_locked_score_cache), compared against
SHAPE_ROUTER_SIZE_THRESHOLD. Downstream diagnostics previously reconstructed
routing from a different column (normalized_foreground_pixels, a bounding-box
fit) and got it wrong. This test locks the diagnostic row's router_size_signal,
selected_metric and score columns to the exact production computation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from constants import SHAPE_ROUTER_SIZE_THRESHOLD  # noqa: E402
from pair_diagnostics import run_selected_pairs_diagnostic  # noqa: E402
from session.preparation import prepare_specimen  # noqa: E402
from verify.scorer import build_locked_score_cache, score_pair_result_routed  # noqa: E402

_IMAGES = Path(__file__).resolve().parent.parent / "images" / "pi_images_v3"
_BLOCK_002 = _IMAGES / "block_002_lungs_NAIVE_01_HE.png"
_SLIDE_002 = _IMAGES / "slide_002_lungs_NAIVE_01_HE.png"


@pytest.mark.skipif(
    not (_BLOCK_002.exists() and _SLIDE_002.exists()),
    reason="pi_images_v3 set_002 images not found (gitignored, local-only fixtures)",
)
def test_router_columns_match_production_routed_scorer(tmp_path):
    out = tmp_path / "diag.csv"
    run_selected_pairs_diagnostic([(_BLOCK_002, _SLIDE_002)], out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]

    block_result = prepare_specimen(_BLOCK_002, role="block")
    slide_result = prepare_specimen(_SLIDE_002, role="slide")

    block_cache = build_locked_score_cache(block_result)
    slide_cache = build_locked_score_cache(slide_result)
    block_frac = block_cache.normalized_mask.sum() / block_cache.normalized_mask.size
    slide_frac = slide_cache.normalized_mask.sum() / slide_cache.normalized_mask.size
    expected_size_signal = min(block_frac, slide_frac)
    expected_method = (
        "mask_iou"
        if expected_size_signal >= SHAPE_ROUTER_SIZE_THRESHOLD
        else "point_layout"
    )
    expected_score = score_pair_result_routed(block_result, slide_result).score

    assert row["router_size_signal"] == f"{expected_size_signal:.4f}"
    assert row["selected_metric"] == expected_method
    assert row["score"] == f"{expected_score:.4f}"

    # set_002 lungs is the exact case the old reconstruction got wrong.
    assert expected_method == "mask_iou"
    assert abs(expected_size_signal - 0.0535) < 0.01
