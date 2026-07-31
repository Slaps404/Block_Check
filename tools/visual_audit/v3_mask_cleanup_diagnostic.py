"""Render v3 mask cleanup diagnostics for small-component threshold probes.

Diagnostic only. This intentionally monkeypatches the segmentation constants at
runtime so a baseline and candidate cleanup can be compared from one command.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import verify.segmentation as segmentation  # noqa: E402
from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402

DEFAULT_MANIFEST = ROOT / "outputs" / "manifests" / "pi_images_v3_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "v3_mask_cleanup"

PATCHED_CONSTANTS = (
    "DUST_COMPONENT_AREA",
    "MIN_SLIDE_COMPONENT_AREA",
    "MIN_AREA_FRACTION",
    "SLIDE_SMALL_FILL_MIN",
    "SLIDE_LAB_A_MIN",
    "SLIDE_SCORE_FLOOR",
    "SLIDE_SCORE_CEILING",
)

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "step1_strict_small_components": {
        "DUST_COMPONENT_AREA": 100,
        "MIN_SLIDE_COMPONENT_AREA": 410,
        "MIN_AREA_FRACTION": 0.00003325,
        "SLIDE_SMALL_FILL_MIN": 1.01,
    },
    "step2_strict_score10_12": {
        "DUST_COMPONENT_AREA": 100,
        "MIN_SLIDE_COMPONENT_AREA": 410,
        "MIN_AREA_FRACTION": 0.00003325,
        "SLIDE_SMALL_FILL_MIN": 1.01,
        "SLIDE_SCORE_FLOOR": 10,
        "SLIDE_SCORE_CEILING": 12,
    },
}

PANEL_W = 360
PANEL_H = 220
LABEL_H = 26
FONT = cv2.FONT_HERSHEY_SIMPLEX


def run(manifest: Path, output_dir: Path, set_ids: set[str] | None = None) -> None:
    rows = _read_manifest(manifest)
    if set_ids is not None:
        rows = [row for row in rows if row["set_id"] in set_ids]
    originals = {name: getattr(segmentation, name) for name in PATCHED_CONSTANTS}
    summaries: dict[str, list[dict[str, str]]] = {}

    for variant, overrides in VARIANTS.items():
        _apply_constants(originals | overrides)
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_rows: list[dict[str, str]] = []

        for row in rows:
            set_id = row["set_id"].replace("set_", "")
            block_path = ROOT / row["block_path"]
            slide_path = ROOT / row["slide_path"]
            block = prepare_specimen(block_path, "block")
            slide = prepare_specimen(slide_path, "slide")
            sheet = _pair_sheet(block_path, slide_path, block, slide, f"{row['set_id']} {variant}")
            cv2.imwrite(str(variant_dir / f"set_{set_id}_mask_overlay.png"), sheet)

            variant_rows.extend(_summary_rows(row["set_id"], block, slide))

        summaries[variant] = variant_rows
        _write_summary(variant_dir / "summary.csv", variant_rows)

    _apply_constants(originals)
    _write_comparison_sheets(rows, output_dir)
    _write_delta(output_dir / "component_delta.csv", summaries)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _apply_constants(values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(segmentation, name, value)


def _pair_sheet(
    block_path: Path,
    slide_path: Path,
    block: object,
    slide: object,
    title: str,
) -> np.ndarray:
    block_img = cv2.imread(str(block_path))
    slide_img = cv2.imread(str(slide_path))
    panels = [
        _labeled("block raw", _fit(block_img)),
        _labeled("block mask", _mask_panel(block, (0, 190, 0))),
        _labeled("slide raw", _fit(slide_img)),
        _labeled("slide overlay", _overlay_panel(slide_img, slide, (0, 0, 230))),
        _labeled("slide mask", _mask_panel(slide, (0, 0, 230))),
    ]
    body = np.hstack(panels)
    header = np.full((34, body.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (8, 23), FONT, 0.62, (20, 20, 20), 1)
    return np.vstack([header, body])


def _write_comparison_sheets(rows: list[dict[str, str]], output_dir: Path) -> None:
    baseline = output_dir / "baseline"
    strict = output_dir / "step1_strict_small_components"
    score_recovery = output_dir / "step2_strict_score10_12"
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for row in rows:
        set_id = row["set_id"].replace("set_", "")
        before = cv2.imread(str(baseline / f"set_{set_id}_mask_overlay.png"))
        after = cv2.imread(str(strict / f"set_{set_id}_mask_overlay.png"))
        if before is None or after is None:
            continue
        divider = np.full((before.shape[0], 18, 3), 255, dtype=np.uint8)
        cv2.putText(divider, ">", (2, before.shape[0] // 2), FONT, 0.4, (40, 40, 40), 1)
        comparison = np.hstack([before, divider, after])
        out = comparison_dir / f"set_{set_id}_baseline_vs_step1.png"
        cv2.imwrite(str(out), comparison)
        all_rows.append(_resize_width(comparison, 1800))

        recovered = cv2.imread(str(score_recovery / f"set_{set_id}_mask_overlay.png"))
        if recovered is not None:
            step2 = np.hstack([after, divider.copy(), recovered])
            cv2.imwrite(str(comparison_dir / f"set_{set_id}_step1_vs_step2.png"), step2)

    if all_rows:
        cv2.imwrite(str(output_dir / "all_sets_baseline_vs_step1.png"), np.vstack(all_rows))


def _summary_rows(set_id: str, block: object, slide: object) -> list[dict[str, str]]:
    return [
        _stats_row(set_id, "block", block),
        _stats_row(set_id, "slide", slide),
    ]


def _stats_row(set_id: str, role: str, result: object) -> dict[str, str]:
    if not isinstance(result, PreparedSpecimen):
        return {
            "set": set_id,
            "role": role,
            "foreground_pixels": "0",
            "coverage": "0.00000000",
            "components": "0",
            "components_lt_100": "0",
            "components_lt_410": "0",
            "status": getattr(result, "reason", "prep failed"),
        }

    mask = result.mask
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = np.array([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)])
    foreground = int(areas.sum()) if areas.size else 0
    return {
        "set": set_id,
        "role": role,
        "foreground_pixels": str(foreground),
        "coverage": f"{foreground / mask.size:.8f}",
        "components": str(int(areas.size)),
        "components_lt_100": str(int(np.count_nonzero(areas < 100))) if areas.size else "0",
        "components_lt_410": str(int(np.count_nonzero(areas < 410))) if areas.size else "0",
        "status": "ok",
    }


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "set",
        "role",
        "foreground_pixels",
        "coverage",
        "components",
        "components_lt_100",
        "components_lt_410",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_delta(path: Path, summaries: dict[str, list[dict[str, str]]]) -> None:
    baseline = {
        (row["set"], row["role"]): row for row in summaries.get("baseline", [])
    }
    strict = {
        (row["set"], row["role"]): row
        for row in summaries.get("step1_strict_small_components", [])
    }
    columns = [
        "set",
        "role",
        "baseline_components",
        "step1_components",
        "component_delta",
        "baseline_foreground_pixels",
        "step1_foreground_pixels",
        "foreground_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for key, before in baseline.items():
            after = strict.get(key)
            if after is None:
                continue
            writer.writerow({
                "set": key[0],
                "role": key[1],
                "baseline_components": before["components"],
                "step1_components": after["components"],
                "component_delta": str(int(after["components"]) - int(before["components"])),
                "baseline_foreground_pixels": before["foreground_pixels"],
                "step1_foreground_pixels": after["foreground_pixels"],
                "foreground_delta": (
                    str(int(after["foreground_pixels"]) - int(before["foreground_pixels"]))
                ),
            })


def _labeled(label: str, panel: np.ndarray) -> np.ndarray:
    header = np.full((LABEL_H, PANEL_W, 3), 245, dtype=np.uint8)
    cv2.putText(header, label, (7, 18), FONT, 0.48, (30, 30, 30), 1)
    return np.vstack([header, panel])


def _mask_panel(result: object, color: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(result, PreparedSpecimen):
        return _placeholder("prep failed")
    panel = np.zeros((*result.mask.shape, 3), dtype=np.uint8)
    panel[result.mask > 0] = color
    return _fit(panel, interpolation=cv2.INTER_NEAREST)


def _overlay_panel(
    image: np.ndarray | None,
    result: object,
    color: tuple[int, int, int],
) -> np.ndarray:
    if image is None or not isinstance(result, PreparedSpecimen):
        return _placeholder("prep failed")
    overlay = image.copy()
    tint = image.copy()
    tint[result.mask > 0] = color
    cv2.addWeighted(overlay, 0.68, tint, 0.32, 0, overlay)
    return _fit(overlay)


def _fit(image: np.ndarray | None, *, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    if image is None or image.size == 0:
        return _placeholder("missing")
    h, w = image.shape[:2]
    scale = min(PANEL_W / max(w, 1), PANEL_H / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    panel = np.full((PANEL_H, PANEL_W, 3), 255, dtype=np.uint8)
    y0 = (PANEL_H - new_h) // 2
    x0 = (PANEL_W - new_w) // 2
    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _placeholder(label: str) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), 235, dtype=np.uint8)
    cv2.putText(panel, label, (12, PANEL_H // 2), FONT, 0.55, (80, 80, 80), 1)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sets",
        default="",
        help="Comma-separated set IDs to render, e.g. set_003,set_008. Default: all.",
    )
    args = parser.parse_args()
    set_ids = {part.strip() for part in args.sets.split(",") if part.strip()} or None
    run(args.manifest, args.output_dir, set_ids=set_ids)
    print(f"Wrote diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
