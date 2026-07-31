"""Scoring diagnostic contact sheets: mask overlays + metric breakdown per pair.

Produces one PNG per scored pair showing:
  Row 1: block image+mask overlay | slide image+mask overlay | scoring panel
  Row 2: summary bar (score, metric, angle, flip, component details)

Scoring panel varies by route:
  - IOU route: green/magenta overlap of aligned masks
  - Point-layout route: normalized-coordinate scatter of matched blob
    centroids (not a mask overlay) with size-ratio/shape-discrepancy labels
    and unmatched markers, mirroring the exact Hungarian assignment scored

Usage:
    .\\venv\\Scripts\\python.exe tools\\scoring_diagnostics\\scoring_contact_sheet.py \\
        --source outputs/diagnostics/v3_41pairs_all_pairs.csv \\
        --manifest outputs/manifests/pi_images_v3_manifest.csv \\
        --output outputs/diagnostics/scoring_sheets/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from verify.locked_alignment import LockedAlignment, align_normalized_masks  # noqa: E402
from constants import PASS_THRESHOLD  # noqa: E402
from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402
from verify.scorer import (  # noqa: E402
    _ComponentFeatures,
    _component_features,
    _point_layout_assignment,
    LockedScoreCache,
    score_routed_caches,
)
from robust_normalization import NORMALIZATION_MODES, normalize_mask  # noqa: E402

DEFAULT_LABELS = frozenset({"true_pair"})

_PANEL_SIZE = 300
_SUMMARY_H = 80
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM = 0.38
_FONT_MD = 0.45
_FONT_LG = 0.55


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _resize(img: np.ndarray, size: int = _PANEL_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _mask_overlay_panel(image_path: str | Path, mask: np.ndarray,
                        color: tuple[int, int, int], label: str) -> np.ndarray:
    """Original image with colored mask overlay and role label."""
    img = cv2.imread(str(image_path))
    if img is None:
        panel = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), 40, dtype=np.uint8)
        cv2.putText(panel, f"no image: {label}", (10, _PANEL_SIZE // 2),
                    _FONT, _FONT_SM, (180, 180, 180), 1)
        return panel

    img = _resize(img)
    mask_resized = cv2.resize(mask, (_PANEL_SIZE, _PANEL_SIZE),
                              interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask_resized > 0] = (
        (overlay[mask_resized > 0].astype(np.int16) + np.array(color, dtype=np.int16)) // 2
    ).clip(0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 1)

    cv2.putText(overlay, label, (6, 18), _FONT, _FONT_MD, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _iou_overlap_panel(alignment: LockedAlignment) -> np.ndarray:
    """Green/magenta overlap panel for IOU-routed pairs."""
    canvas = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), 24, dtype=np.uint8)
    bm = cv2.resize(alignment.block_mask, (_PANEL_SIZE, _PANEL_SIZE),
                    interpolation=cv2.INTER_NEAREST)
    sm = cv2.resize(alignment.aligned_slide_mask, (_PANEL_SIZE, _PANEL_SIZE),
                    interpolation=cv2.INTER_NEAREST)

    both = (bm > 0) & (sm > 0)
    block_only = (bm > 0) & ~both
    slide_only = (sm > 0) & ~both

    canvas[both] = (200, 220, 200)
    canvas[block_only] = (40, 200, 40)
    canvas[slide_only] = (200, 40, 200)

    iou = float(np.count_nonzero(both)) / max(np.count_nonzero((bm > 0) | (sm > 0)), 1)
    cv2.putText(canvas, f"IOU Overlap: {iou:.3f}", (6, 18),
                _FONT, _FONT_MD, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "green=block  magenta=slide  light=overlap", (6, _PANEL_SIZE - 10),
                _FONT, _FONT_SM, (160, 160, 160), 1, cv2.LINE_AA)
    return canvas


def _constellation_panel(
    block_features: _ComponentFeatures,
    slide_features: _ComponentFeatures,
) -> np.ndarray:
    """Draw the normalized points and assignment used by point layout."""
    canvas = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), 24, dtype=np.uint8)
    a, b = block_features.points, slide_features.points
    if len(a) == 0 and len(b) == 0:
        cv2.putText(canvas, "No components", (60, _PANEL_SIZE // 2),
                    _FONT, _FONT_MD, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    _, rows, cols = _point_layout_assignment(block_features, slide_features)
    margin = 28
    span = _PANEL_SIZE - 2 * margin

    def point_pixel(point: np.ndarray) -> tuple[int, int]:
        return (
            int(round(margin + float(point[0]) * span)),
            int(round(margin + float(point[1]) * span)),
        )

    for r, c in zip(rows, cols):
        if r >= len(a) or c >= len(b):
            continue
        block_pixel = point_pixel(a[r])
        slide_pixel = point_pixel(b[c])
        cv2.line(canvas, block_pixel, slide_pixel, (120, 220, 220), 1, cv2.LINE_AA)

        area_ratio = block_features.areas[r] / max(slide_features.areas[c], 1e-6)
        shape_dist = float(np.linalg.norm(
            block_features.shapes[r] - slide_features.shapes[c]))
        bx, by = block_pixel
        sx, sy = slide_pixel
        mid_x = (bx + sx) // 2
        mid_y = (by + sy) // 2

        label = f"{area_ratio:.1f}x {shape_dist:.2f}s"
        text_x = max(mid_x + 12, 4)
        text_y = mid_y
        cv2.putText(canvas, label, (text_x, text_y),
                    _FONT, 0.33, (255, 255, 255), 1, cv2.LINE_AA)

    unmatched_block = {
        int(r) for r, c in zip(rows, cols) if r < len(a) and c >= len(b)
    }
    unmatched_slide = {
        int(c) for r, c in zip(rows, cols) if r >= len(a) and c < len(b)
    }
    for index, point in enumerate(a):
        pixel = point_pixel(point)
        cv2.circle(canvas, pixel, 4, (40, 200, 40), -1, cv2.LINE_AA)
        if index in unmatched_block:
            cv2.drawMarker(canvas, pixel, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 13, 2)
    for index, point in enumerate(b):
        pixel = point_pixel(point)
        cv2.circle(canvas, pixel, 4, (200, 40, 200), -1, cv2.LINE_AA)
        if index in unmatched_slide:
            cv2.drawMarker(canvas, pixel, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 13, 2)

    cv2.putText(canvas, "Point Layout (normalized coordinates)", (6, 16),
                _FONT, _FONT_SM, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "green=block magenta=slide X=unmatched", (6, _PANEL_SIZE - 22),
                _FONT, _FONT_SM, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Nx=size ratio  Ns=shape dist", (6, _PANEL_SIZE - 8),
                _FONT, _FONT_SM, (160, 160, 160), 1, cv2.LINE_AA)
    return canvas


def _summary_bar(result, width: int, normalization_mode: str) -> np.ndarray:
    """Bottom summary bar with score, metric, alignment, and component details."""
    bar = np.full((_SUMMARY_H, width, 3), 30, dtype=np.uint8)

    verdict_color = (
        (80, 220, 80) if result.score >= PASS_THRESHOLD else (80, 80, 220)
    )
    line1 = (
        f"SCORE: {result.score:.4f}  |  metric: {result.selected_metric}  |  "
        f"normalization: {normalization_mode}  |  "
        f"angle: {result.best_angle:.0f}  flip: {result.best_flip}"
    )
    line2 = (
        f"block_frac: {result.block_occupied_fraction:.4f}  "
        f"slide_frac: {result.slide_occupied_fraction:.4f}  "
        f"router_signal: {result.router_size_signal:.4f}  "
        f"soft_iou: {result.align_soft_iou:.4f}  "
        f"mask_iou: {result.mask_iou:.4f}"
    )
    line3 = ""
    if result.point_layout is not None:
        line3 = f"point_layout: {result.point_layout:.4f}"

    cv2.putText(bar, line1, (8, 20), _FONT, _FONT_MD, verdict_color, 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (8, 42), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
    if line3:
        cv2.putText(bar, line3, (8, 62), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Sheet assembly
# ---------------------------------------------------------------------------

def _score_prepared_pair(
    block_prep: PreparedSpecimen,
    slide_prep: PreparedSpecimen,
    *,
    normalization_mode: str,
    force_iou: bool,
):
    """Build diagnostic caches and expose the locked alignment used by panels."""
    block_normalized = normalize_mask(block_prep.mask, normalization_mode)
    slide_normalized = normalize_mask(slide_prep.mask, normalization_mode)
    block_cache = LockedScoreCache(
        block_normalized,
        _component_features(block_normalized),
    )
    slide_cache = LockedScoreCache(
        slide_normalized,
        _component_features(slide_normalized),
    )
    result = score_routed_caches(block_cache, slide_cache)
    alignment = align_normalized_masks(block_normalized, slide_normalized)
    if force_iou:
        result = replace(
            result,
            score=alignment.mask_iou,
            selected_metric="mask_iou",
            point_layout=None,
        )
    return block_cache, slide_cache, result, alignment


def render_scoring_sheet(
    block_path: str | Path,
    slide_path: str | Path,
    *,
    normalization_mode: str = "rms",
    force_iou: bool = False,
) -> np.ndarray:
    """Produce one full diagnostic contact sheet PNG for a pair."""
    block_prep = prepare_specimen(block_path, role="block")
    slide_prep = prepare_specimen(slide_path, role="slide")

    if not isinstance(block_prep, PreparedSpecimen):
        raise RuntimeError(f"block prep failed: {block_prep.reason}")
    if not isinstance(slide_prep, PreparedSpecimen):
        raise RuntimeError(f"slide prep failed: {slide_prep.reason}")

    block_cache, _, result, alignment = _score_prepared_pair(
        block_prep,
        slide_prep,
        normalization_mode=normalization_mode,
        force_iou=force_iou,
    )

    block_panel = _mask_overlay_panel(block_path, block_prep.mask, (40, 200, 40), "BLOCK")
    slide_panel = _mask_overlay_panel(slide_path, slide_prep.mask, (200, 40, 200), "SLIDE")

    if result.selected_metric == "mask_iou":
        score_panel = _iou_overlap_panel(alignment)
    else:
        slide_features_aligned = _component_features(alignment.aligned_slide_mask)
        score_panel = _constellation_panel(
            block_cache.component_features,
            slide_features_aligned,
        )

    row1 = np.hstack([block_panel, slide_panel, score_panel])
    summary = _summary_bar(result, width=row1.shape[1], normalization_mode=normalization_mode)
    return np.vstack([row1, summary])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_wanted_pairs(
    source_csv: Path,
    labels: set[str] | frozenset[str] = DEFAULT_LABELS,
) -> list[tuple[str, str]]:
    with open(source_csv, newline="") as f:
        reader = csv.DictReader(f)
        return [
            (row["block_path"], row["slide_path"])
            for row in reader
            if row["diagnostic_label"] in labels
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="outputs/diagnostics/v3_41pairs_all_pairs.csv",
        help="Prior diagnostic CSV to pull pair list from.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/pi_images_v3_manifest.csv",
        help="Manifest CSV (used only for metadata display).",
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/scoring_sheets",
        help="Output directory for contact sheet PNGs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max pairs to render (0 = all).",
    )
    parser.add_argument(
        "--labels",
        default="true_pair",
        help="Comma-separated diagnostic labels to render (default: true_pair).",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="rms",
        help="Mask scale estimator used only for this diagnostic render.",
    )
    parser.add_argument(
        "--force-iou",
        action="store_true",
        help="Render and report IoU regardless of the production router threshold.",
    )
    args = parser.parse_args()

    source_csv = ROOT / args.source
    output_dir = ROOT / args.output
    if not source_csv.is_file():
        parser.error(f"source not found: {source_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    if not labels:
        parser.error("--labels must contain at least one diagnostic label")
    pairs = read_wanted_pairs(source_csv, labels)
    if not pairs:
        parser.error(f"no rows with labels {sorted(labels)} in {source_csv}")
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"Rendering {len(pairs)} scoring contact sheets...\n")

    for i, (bpath, spath) in enumerate(pairs, 1):
        bname = Path(bpath).stem
        sname = Path(spath).stem
        filename = f"score_{i:03d}_{bname}_vs_{sname}.png"
        try:
            sheet = render_scoring_sheet(
                bpath,
                spath,
                normalization_mode=args.normalization,
                force_iou=args.force_iou,
            )
            cv2.imwrite(str(output_dir / filename), sheet)
            print(f"  [{i}/{len(pairs)}] {filename}")
        except RuntimeError as e:
            print(f"  [{i}/{len(pairs)}] SKIP ({e})")

    print(f"\nDone. Sheets in: {output_dir}")


if __name__ == "__main__":
    main()
