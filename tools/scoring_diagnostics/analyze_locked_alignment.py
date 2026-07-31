"""Compute true-pair minus best same-tissue near-miss margins from raw CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class MarginSummary:
    metric: str
    median_margin: float
    minimum_margin: float
    pair_count: int

    def verdict_line(self) -> str:
        verdict = "SEPARATED" if self.minimum_margin > 0 else "OVERLAP"
        return (
            f"NEAR_MISS_MARGIN metric={self.metric} "
            f"median={self.median_margin:.4f} min={self.minimum_margin:.4f} "
            f"n={self.pair_count} verdict={verdict}"
        )


def analyze_near_miss_margin(csv_path: Path, metric: str) -> MarginSummary:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or metric not in rows[0]:
        raise ValueError(f"missing metric column: {metric}")

    margins: list[float] = []
    block_ids = {row["block_set"] for row in rows}
    for block_id in sorted(block_ids):
        block_rows = [row for row in rows if row["block_set"] == block_id]
        true_scores = [
            float(row[metric]) for row in block_rows
            if row["diagnostic_label"] == "true_pair" and row[metric]
        ]
        near_scores = [
            float(row[metric]) for row in block_rows
            if row["diagnostic_label"] == "near_miss" and row[metric]
        ]
        if true_scores and near_scores:
            best_near_miss = max(near_scores)
            margins.extend(score - best_near_miss for score in true_scores)
    if not margins:
        raise ValueError("no blocks have both true-pair and same-tissue near-miss scores")
    return MarginSummary(metric, median(margins), min(margins), len(margins))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--metric", default="locked_mask_iou")
    args = parser.parse_args()
    print(analyze_near_miss_margin(args.csv_path, args.metric).verdict_line())


if __name__ == "__main__":
    main()
