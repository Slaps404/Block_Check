"""Render per-image segmentation overlays for a v3 PNG dataset."""

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

from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402

DEFAULT_DATASET = ROOT / "images" / "pi_images_v3"
DEFAULT_OUTPUT = ROOT / "outputs" / "v3_mask_overlays"
MAX_EDGE = 1600
COLORS = {
    "block": (0, 190, 0),
    "slide": (0, 0, 230),
}


def role_from_name(name: str) -> str | None:
    if name.startswith("block_"):
        return "block"
    if name.startswith("slide_"):
        return "slide"
    return None


def overlay_on_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = bgr.copy()
    tint = bgr.copy()
    tint[mask > 0] = color
    cv2.addWeighted(out, 0.68, tint, 0.32, 0, out)
    return out


def resize_max_edge(img: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def mask_stats(mask: np.ndarray) -> dict[str, str]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = np.array([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)])
    foreground = int(areas.sum()) if areas.size else 0
    largest = int(areas.max()) if areas.size else 0
    smallest = int(areas.min()) if areas.size else 0
    return {
        "components": str(int(areas.size)),
        "foreground_pixels": str(foreground),
        "coverage": f"{foreground / mask.size:.8f}",
        "largest_area": str(largest),
        "smallest_area": str(smallest),
    }


def run(dataset: Path, output_dir: Path) -> None:
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in sorted(dataset.glob("*.png")):
        role = role_from_name(path.name)
        if role is None:
            continue

        bgr = cv2.imread(str(path))
        if bgr is None:
            rows.append({
                "filename": path.name,
                "role": role,
                "status": "unreadable",
                "overlay_path": "",
            })
            continue

        result = prepare_specimen(path, role)
        overlay_path = overlay_dir / f"{path.stem}_overlay.png"

        if isinstance(result, PreparedSpecimen):
            overlay = overlay_on_bgr(bgr, result.mask, COLORS[role])
            cv2.imwrite(str(overlay_path), resize_max_edge(overlay, MAX_EDGE))
            row = {
                "filename": path.name,
                "role": role,
                "status": "ok",
                "overlay_path": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
                **mask_stats(result.mask),
            }
        else:
            fail = np.full_like(bgr, 235)
            cv2.putText(
                fail,
                f"prep failed: {result.reason[:70]}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 40, 200),
                2,
            )
            cv2.imwrite(str(overlay_path), resize_max_edge(fail, MAX_EDGE))
            row = {
                "filename": path.name,
                "role": role,
                "status": result.reason,
                "overlay_path": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
                "components": "0",
                "foreground_pixels": "0",
                "coverage": "0.00000000",
                "largest_area": "0",
                "smallest_area": "0",
            }
        rows.append(row)
        print(f"{path.name}: {row['status']}")

    summary_path = output_dir / "summary.csv"
    columns = [
        "filename",
        "role",
        "status",
        "overlay_path",
        "components",
        "foreground_pixels",
        "coverage",
        "largest_area",
        "smallest_area",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"Wrote {len(rows)} overlays ({ok} ok) to {overlay_dir}")
    print(f"Summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.dataset, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
