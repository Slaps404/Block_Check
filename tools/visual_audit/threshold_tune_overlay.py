"""Generate set_002 slide mask overlay for a threshold variant."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import verify.segmentation as segmentation  # noqa: E402
from session.preparation import PreparedSpecimen, prepare_specimen  # noqa: E402

PATCH_NAMES = (
    "DUST_COMPONENT_AREA",
    "MIN_SLIDE_COMPONENT_AREA",
    "MIN_AREA_FRACTION",
    "SLIDE_SMALL_FILL_MIN",
)

SLIDE_PATH = ROOT / "images" / "pi_images_v3" / "slide_002_lungs_NAIVE_01_HE.png"
OUT_DIR = ROOT / "experiments" / "active" / "threshold_tune_set_002"


def apply(overrides: dict) -> None:
    for name in PATCH_NAMES:
        if name in overrides:
            setattr(segmentation, name, overrides[name])


def restore(originals: dict) -> None:
    for name, value in originals.items():
        setattr(segmentation, name, value)


def overlay(slide_path: Path, result: PreparedSpecimen) -> np.ndarray:
    img = cv2.imread(str(slide_path))
    tint = img.copy()
    tint[result.mask > 0] = (0, 0, 230)
    out = img.copy()
    cv2.addWeighted(out, 0.68, tint, 0.32, 0, out)
    return out


def mask_panel(result: PreparedSpecimen) -> np.ndarray:
    panel = np.zeros((*result.mask.shape, 3), dtype=np.uint8)
    panel[result.mask > 0] = (0, 0, 230)
    return panel


def stats(result: PreparedSpecimen) -> str:
    num, _, st, _ = cv2.connectedComponentsWithStats(result.mask, connectivity=8)
    areas = [int(st[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    areas.sort(reverse=True)
    small = [a for a in areas if a < 500]
    return (
        f"components={len(areas)}  small(<500)={len(small)}  "
        f"areas={areas[:8]}"
    )


def run_variant(label: str, overrides: dict) -> Path:
    originals = {name: getattr(segmentation, name) for name in PATCH_NAMES}
    apply(overrides)
    result = prepare_specimen(SLIDE_PATH, "slide")
    restore(originals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not isinstance(result, PreparedSpecimen):
        raise RuntimeError(result.reason)

    sheet = np.hstack([overlay(SLIDE_PATH, result), mask_panel(result)])
    target_w = 1200
    scale = target_w / sheet.shape[1]
    resized = cv2.resize(sheet, (target_w, max(1, int(sheet.shape[0] * scale))))
    header = np.full((36, target_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        header, f"{label} | {stats(result)}", (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
    )
    out = np.vstack([header, resized])
    path = OUT_DIR / f"{label}.png"
    cv2.imwrite(str(path), out)
    print(f"{label}: {stats(result)}")
    print(f"  -> {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--dust", type=int)
    parser.add_argument("--min-slide", type=int)
    parser.add_argument("--min-frac", type=float)
    parser.add_argument("--small-fill", type=float)
    args = parser.parse_args()

    overrides: dict = {}
    if args.dust is not None:
        overrides["DUST_COMPONENT_AREA"] = args.dust
    if args.min_slide is not None:
        overrides["MIN_SLIDE_COMPONENT_AREA"] = args.min_slide
    if args.min_frac is not None:
        overrides["MIN_AREA_FRACTION"] = args.min_frac
    if args.small_fill is not None:
        overrides["SLIDE_SMALL_FILL_MIN"] = args.small_fill
    run_variant(args.label, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
