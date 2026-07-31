"""Timed re-score of true-pair + near-miss pairs with per-step timing.

Wraps the same preparation/scoring pipeline as run_selected_pairs.py but
instruments each step (prepare_block, prepare_slide, build_cache, score) with
wall-clock timing. Prints per-pair timings and writes a JSON summary.

Usage:
    .\venv\Scripts\python.exe tools\scoring_diagnostics\run_selected_pairs_timed.py \
        --source outputs/diagnostics/v3_41pairs_all_pairs.csv \
        --manifest outputs/manifests/pi_images_v3_manifest.csv \
        --output outputs/diagnostics/v3_true_pairs_timed.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from session.preparation import prepare_specimen  # noqa: E402
from verify.scorer import build_locked_score_cache, score_routed_caches  # noqa: E402
from run_diagnostics import load_manifest_paths  # noqa: E402

KEEP_LABELS = {"true_pair"}


def read_wanted_pairs(source_csv: Path) -> list[tuple[str, str]]:
    with open(source_csv, newline="") as f:
        reader = csv.DictReader(f)
        return [
            (row["block_path"], row["slide_path"])
            for row in reader
            if row["diagnostic_label"] in KEEP_LABELS
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="outputs/diagnostics/v3_41pairs_all_pairs.csv",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/pi_images_v3_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/v3_true_pairs_timed.json",
    )
    args = parser.parse_args()

    source_csv = ROOT / args.source
    manifest_csv = ROOT / args.manifest
    if not source_csv.is_file():
        parser.error(f"source not found: {source_csv}")
    if not manifest_csv.is_file():
        parser.error(f"manifest not found: {manifest_csv}")

    pairs = read_wanted_pairs(source_csv)
    if not pairs:
        parser.error(f"no true_pair rows in {source_csv}")

    print(f"Scoring {len(pairs)} true pairs with timing...\n")

    pair_timings: list[dict] = []

    for i, (bpath, spath) in enumerate(pairs, 1):
        timing: dict = {"pair": i, "block": bpath, "slide": spath}

        t0 = time.perf_counter()
        block_prep = prepare_specimen(bpath, role="block")
        timing["prepare_block_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        slide_prep = prepare_specimen(spath, role="slide")
        timing["prepare_slide_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        block_cache = build_locked_score_cache(block_prep)
        timing["cache_block_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        slide_cache = build_locked_score_cache(slide_prep)
        timing["cache_slide_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = score_routed_caches(block_cache, slide_cache)
        timing["score_ms"] = (time.perf_counter() - t0) * 1000

        timing["total_ms"] = (
            timing["prepare_block_ms"] + timing["prepare_slide_ms"]
            + timing["cache_block_ms"] + timing["cache_slide_ms"]
            + timing["score_ms"]
        )
        timing["final_score"] = result.score
        timing["selected_metric"] = result.selected_metric

        pair_timings.append(timing)

        print(
            f"[{i}/{len(pairs)}] "
            f"prep_b={timing['prepare_block_ms']:.0f}ms  "
            f"prep_s={timing['prepare_slide_ms']:.0f}ms  "
            f"cache_b={timing['cache_block_ms']:.0f}ms  "
            f"cache_s={timing['cache_slide_ms']:.0f}ms  "
            f"score={timing['score_ms']:.0f}ms  "
            f"total={timing['total_ms']:.0f}ms  "
            f"→ {result.score:.4f}"
        )

    # Averages
    n = len(pair_timings)
    avg = {
        key: sum(t[key] for t in pair_timings) / n
        for key in [
            "prepare_block_ms", "prepare_slide_ms",
            "cache_block_ms", "cache_slide_ms",
            "score_ms", "total_ms",
        ]
    }

    print(f"\n{'='*60}")
    print(f"AVERAGES over {n} true pairs:")
    print(f"  prepare_block:  {avg['prepare_block_ms']:.1f} ms")
    print(f"  prepare_slide:  {avg['prepare_slide_ms']:.1f} ms")
    print(f"  cache_block:    {avg['cache_block_ms']:.1f} ms")
    print(f"  cache_slide:    {avg['cache_slide_ms']:.1f} ms")
    print(f"  score:          {avg['score_ms']:.1f} ms")
    print(f"  total:          {avg['total_ms']:.1f} ms")
    print(f"{'='*60}")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_pairs": n,
        "averages_ms": avg,
        "pairs": pair_timings,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote timing JSON: {output_path}")


if __name__ == "__main__":
    main()
