"""Render locked-alignment overlays through the production scorer's cache path."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from verify.locked_alignment import align_normalized_masks, render_alignment_overlay  # noqa: E402
from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402
from verify.scorer import (  # noqa: E402
    build_locked_score_cache,
    score_routed_caches,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "diagnostics" / "production_locked_overlays"
DEFAULT_PAIRS = ((6, 6), (6, 12), (19, 19), (19, 11))


def _image(dataset: Path, role: str, set_id: int) -> Path:
    matches = list(dataset.glob(f"{role}_{set_id:03d}_*.png"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {role} image for set {set_id:03d}, found {len(matches)}"
        )
    return matches[0]


def render_overlays(dataset: Path, output: Path) -> list[dict[str, str]]:
    """Prepare, score, and render the issue-69 proof pairs."""
    output.mkdir(parents=True, exist_ok=True)
    specimens: dict[tuple[str, int], PreparedSpecimen] = {}
    caches = {}
    for role, set_id in sorted({(role, n) for b, s in DEFAULT_PAIRS for role, n in (
        ("block", b), ("slide", s)
    )}):
        result = prepare_specimen(_image(dataset, role, set_id), role=role)
        if not isinstance(result, PreparedSpecimen):
            raise RuntimeError(f"{role} {set_id:03d} preparation failed: {result.reason}")
        specimens[role, set_id] = result
        caches[role, set_id] = build_locked_score_cache(result)

    rows = []
    for block_id, slide_id in DEFAULT_PAIRS:
        block_cache = caches["block", block_id]
        slide_cache = caches["slide", slide_id]
        score = score_routed_caches(block_cache, slide_cache)
        alignment = align_normalized_masks(
            block_cache.normalized_mask,
            slide_cache.normalized_mask,
        )
        caption = (
            f"{block_id:03d}/{slide_id:03d} s={score.score:.3f} "
            f"a={alignment.best_angle:.0f} f={int(alignment.best_flip)}"
        )
        filename = f"production_locked_{block_id:03d}_{slide_id:03d}.png"
        cv2.imwrite(str(output / filename), render_alignment_overlay(alignment, caption))
        rows.append({
            "block_set": f"{block_id:03d}",
            "slide_set": f"{slide_id:03d}",
            "score": f"{score.score:.6f}",
            "best_angle": f"{alignment.best_angle:.1f}",
            "best_flip": str(alignment.best_flip),
            "mask_iou": f"{score.mask_iou:.6f}",
            "overlay": filename,
        })

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "images" / "pi_images_v3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = render_overlays(args.dataset, args.output)
    for row in rows:
        print(
            f"{row['block_set']}->{row['slide_set']} "
            f"score={row['score']} overlay={row['overlay']}"
        )


if __name__ == "__main__":
    main()
