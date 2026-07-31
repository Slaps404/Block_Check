"""Audit slide relative component floor across all v3 manifest sets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import constants  # noqa: E402
from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402

DEFAULT_MANIFEST = ROOT / "outputs" / "manifests" / "pi_images_v3_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "v3_relative_component_floor"

PANEL_W = 360
PANEL_H = 220
LABEL_H = 26
FONT = cv2.FONT_HERSHEY_SIMPLEX


def run(manifest: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str]] = []

    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        set_id = row["set_id"]
        set_num = set_id.replace("set_", "")
        block_path = ROOT / row["block_path"]
        slide_path = ROOT / row["slide_path"]
        tissue = row.get("slide_tissue", "")

        block = prepare_specimen(block_path, "block")
        slide = prepare_specimen(slide_path, "slide")

        rows_out.append(_stats_row(set_id, tissue, "block", block))
        rows_out.append(_stats_row(set_id, tissue, "slide", slide))

        sheet = _pair_sheet(
            block_path,
            slide_path,
            block,
            slide,
            (
                f"{set_id} rel_floor={constants.SLIDE_MIN_COMPONENT_REL_AREA:.3f} "
                f"| {tissue}"
            ),
        )
        cv2.imwrite(str(output_dir / f"set_{set_num}_mask_overlay.png"), sheet)

    _write_summary(output_dir / "summary.csv", rows_out)
    print(f"Wrote {len(rows) * 2} mask rows and {len(rows)} overlays to {output_dir}")


def _stats_row(
    set_id: str,
    tissue: str,
    role: str,
    result: object,
) -> dict[str, str]:
    if not isinstance(result, PreparedSpecimen):
        return {
            "set": set_id,
            "tissue": tissue,
            "role": role,
            "status": getattr(result, "reason", "prep failed"),
            "components": "0",
            "foreground_pixels": "0",
            "coverage": "0.00000000",
            "largest_area": "0",
            "smallest_area": "0",
            "smallest_rel_largest": "0.000000",
            "components_lt_500": "0",
        }

    mask = result.mask
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = np.array([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)])
    foreground = int(areas.sum()) if areas.size else 0
    largest = int(areas.max()) if areas.size else 0
    smallest = int(areas.min()) if areas.size else 0
    rel = (smallest / largest) if largest else 0.0
    return {
        "set": set_id,
        "tissue": tissue,
        "role": role,
        "status": "ok",
        "components": str(int(areas.size)),
        "foreground_pixels": str(foreground),
        "coverage": f"{foreground / mask.size:.8f}",
        "largest_area": str(largest),
        "smallest_area": str(smallest),
        "smallest_rel_largest": f"{rel:.6f}",
        "components_lt_500": (
            str(int(np.count_nonzero(areas < 500))) if areas.size else "0"
        ),
    }


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
    cv2.putText(header, title, (8, 23), FONT, 0.55, (20, 20, 20), 1)
    return np.vstack([header, body])


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


def _placeholder(label: str) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), 235, dtype=np.uint8)
    cv2.putText(panel, label, (12, PANEL_H // 2), FONT, 0.55, (80, 80, 80), 1)
    return panel


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "set",
        "tissue",
        "role",
        "status",
        "components",
        "foreground_pixels",
        "coverage",
        "largest_area",
        "smallest_area",
        "smallest_rel_largest",
        "components_lt_500",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
