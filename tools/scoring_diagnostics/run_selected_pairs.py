"""Re-score only the true-pair + same-tissue near-miss pairs from a prior run.

Reads an existing all-pairs diagnostic CSV, pulls out just the
``true_pair`` and ``near_miss`` rows, and re-scores ONLY those
explicit (block_path, slide_path) pairs -- skipping the full N×N cross product,
so it is much faster than a fresh all-pairs run. Output has no PASS/REVIEW
decision (diagnostic scores only).

Usage:
    python tools/scoring_diagnostics/run_selected_pairs.py \
        --source outputs/diagnostics/v3_41pairs_all_pairs.csv \
        --manifest outputs/manifests/pi_images_v3_manifest.csv \
        --output outputs/diagnostics/v3_41pairs_no_decision.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pair_diagnostics import run_selected_pairs_diagnostic  # noqa: E402
from run_diagnostics import load_manifest_paths  # noqa: E402

KEEP_LABELS = {"true_pair", "near_miss"}


def read_wanted_pairs(source_csv: Path) -> list[tuple[str, str]]:
    """Return (block_path, slide_path) for every true-pair / near-miss row."""
    with open(source_csv, newline="") as f:
        reader = csv.DictReader(f)
        pairs = [
            (row["block_path"], row["slide_path"])
            for row in reader
            if row["diagnostic_label"] in KEEP_LABELS
        ]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="outputs/diagnostics/v3_41pairs_all_pairs.csv",
        help="Existing all-pairs diagnostic CSV to pull the pair list from.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/pi_images_v3_manifest.csv",
        help="Manifest CSV for set/tissue metadata (matches the prior run's labels).",
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/v3_41pairs_no_decision.csv",
        help="Destination CSV for the re-scored subset (no PASS/REVIEW).",
    )
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=0,
        help="Number of true-pair alignment overlay PNGs to write (default 0).",
    )
    args = parser.parse_args()

    source_csv = ROOT / args.source
    if not source_csv.is_file():
        parser.error(f"source CSV not found: {source_csv}")
    manifest_csv = ROOT / args.manifest
    if not manifest_csv.is_file():
        parser.error(f"manifest file not found: {manifest_csv}")

    pairs = read_wanted_pairs(source_csv)
    if not pairs:
        parser.error(f"no true_pair / near_miss rows in {source_csv}")

    _, _, path_metadata = load_manifest_paths(manifest_csv, ROOT)

    n_blocks = len({b for b, _ in pairs})
    n_slides = len({s for _, s in pairs})
    print(f"pairs to score: {len(pairs)} "
          f"({n_blocks} unique blocks x {n_slides} unique slides prepared once each)")

    run_selected_pairs_diagnostic(
        pairs,
        ROOT / args.output,
        path_metadata=path_metadata,
    )
    print(f"wrote {ROOT / args.output}")


if __name__ == "__main__":
    main()
