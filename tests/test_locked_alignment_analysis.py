"""Tests for the single-metric near-miss margin analyzer."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "scoring_diagnostics"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_locked_alignment import analyze_near_miss_margin  # noqa: E402


def test_analyzer_reports_median_and_min_true_minus_best_near_miss(tmp_path):
    path = tmp_path / "scores.csv"
    rows = [
        ("A", "true_pair", "0.90"),
        ("A", "near_miss", "0.70"),
        ("A", "wrong_pair", "0.40"),
        ("B", "true_pair", "0.80"),
        ("B", "near_miss", "0.75"),
        ("B", "wrong_pair", "0.99"),
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_set", "diagnostic_label", "locked_mask_iou"])
        writer.writerows(rows)

    result = analyze_near_miss_margin(path, "locked_mask_iou")

    assert result.median_margin == pytest.approx(0.125)
    assert result.minimum_margin == pytest.approx(0.05)
    assert result.pair_count == 2
    assert result.verdict_line() == (
        "NEAR_MISS_MARGIN metric=locked_mask_iou median=0.1250 "
        "min=0.0500 n=2 verdict=SEPARATED"
    )
