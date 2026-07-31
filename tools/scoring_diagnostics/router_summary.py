"""Produce a 41-row per-set router summary CSV.

Reads the clean + unmatchable no-decision diagnostic CSVs, reads the
production shape-router decision directly from `selected_metric` / `score`,
and writes one row per set showing:

  set_number, tissue_type, true_pair_score, highest_near_miss_score,
  selected_metric, confidence

The 6 unmatchable sets are flagged "low confidence - not reliable".
The 35 clean sets are flagged "reliable".
"""

from __future__ import annotations

import csv
from pathlib import Path

UNMATCHABLE_SETS = {
    "set_009", "set_024", "set_026",
    "set_027", "set_030", "set_036",
}

OUTPUT_COLUMNS = [
    "set_number",
    "tissue_type",
    "true_pair_score",
    "highest_near_miss_score",
    "selected_metric",
    "confidence",
]

DIAG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "diagnostics"
)


def _load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_summary(
    clean_csv: Path = DIAG_DIR / "v3_41pairs_no_decision_clean.csv",
    unmatchable_csv: Path = DIAG_DIR / "v3_41pairs_no_decision_unmatchable.csv",
    output_csv: Path = DIAG_DIR / "v3_41pairs_router_summary.csv",
) -> Path:
    rows = _load_rows(clean_csv) + _load_rows(unmatchable_csv)

    # Index by (block_set, diagnostic_label)
    true_pairs: dict[str, dict] = {}
    near_misses: dict[str, list[dict]] = {}

    for row in rows:
        label = row["diagnostic_label"]
        bset = row["block_set"]
        if label == "true_pair":
            true_pairs[bset] = row
        elif label == "near_miss":
            near_misses.setdefault(bset, []).append(row)

    summary_rows = []
    for bset in sorted(true_pairs):
        tp = true_pairs[bset]
        method = tp["selected_metric"]
        tp_score = float(tp["score"])

        best_nm = None
        for nm in near_misses.get(bset, []):
            nm_score = float(nm["score"])
            if best_nm is None or nm_score > best_nm:
                best_nm = nm_score

        set_num = bset.replace("set_", "")
        confidence = (
            "low confidence - not reliable"
            if bset in UNMATCHABLE_SETS
            else "reliable"
        )

        summary_rows.append({
            "set_number": set_num,
            "tissue_type": tp["tissue_bucket"],
            "true_pair_score": f"{tp_score:.4f}",
            "highest_near_miss_score": f"{best_nm:.4f}" if best_nm is not None else "",
            "selected_metric": method,
            "confidence": confidence,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows -> {output_csv}")
    return output_csv


if __name__ == "__main__":
    build_summary()
