"""
Visual audit harness for slide_label_mask.py.

Runs detect + overlay on a chosen subset of slides and writes two PNGs per
image to phase1_outputs/label_mask_audit/:
  - <stem>_overlay.png : original image with green (detected) and red
    (expanded fill) rotated-rectangle contours drawn on top.
  - <stem>_masked.png  : what segment_tissue() will receive after masking.

Inspect the overlays to verify:
  - Green box sits on the tag.
  - Red (expanded) box fully covers the tag + all visible text.
  - No tissue is inside the red box.

Usage:
    python tools/visual_audit/audit_slide_label_mask.py
    python tools/visual_audit/audit_slide_label_mask.py --sets 001,002,003
    python tools/visual_audit/audit_slide_label_mask.py --sets all
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "code"))

from slide.label_mask import apply_label_mask, draw_label_overlay, find_label_rect  # noqa: E402

IMAGE_DIR = _REPO_ROOT / "images" / "pi_images_v3"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "diagnostics" / "label_mask_audit"


def _collect_files(set_ids: list[str]) -> list[Path]:
    """Glob slide images for the given set IDs (or all slides)."""
    if set_ids == ["all"]:
        patterns = [
            str(IMAGE_DIR / "slide_*.*"),
        ]
    else:
        patterns = []
        for sid in set_ids:
            patterns.append(str(IMAGE_DIR / f"slide_{sid}_*.*"))
    files = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    return files


def run_audit(set_ids: list[str]) -> None:
    """Detect, overlay, and mask all slide images for the given set IDs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = _collect_files(set_ids)

    if not files:
        print(f"No slide images found for sets {set_ids} in {IMAGE_DIR}")
        return

    found_count = 0
    miss_count = 0

    for path in files:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  SKIP (unreadable): {path.name}")
            continue

        rect = find_label_rect(bgr)

        overlay_path = OUTPUT_DIR / f"{path.stem}_overlay.png"
        draw_label_overlay(bgr, rect, overlay_path)

        masked = apply_label_mask(bgr, rect)
        masked_path = OUTPUT_DIR / f"{path.stem}_masked.png"
        cv2.imwrite(str(masked_path), masked)

        if rect.found:
            found_count += 1
            print(
                f"  HIT   {path.name}  "
                f"side={rect.label_side}  angle={rect.angle:.1f}°  "
                f"size={rect.size[0]:.0f}×{rect.size[1]:.0f}"
            )
        else:
            miss_count += 1
            print(f"  MISS  {path.name}  (passthrough — image unchanged)")

    total = found_count + miss_count
    print(f"\n{found_count}/{total} detected  |  {miss_count} passthrough (safe)")
    print(f"Overlay PNGs written to: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual audit of slide label masking"
    )
    parser.add_argument(
        "--sets",
        default="001,002",
        help=(
            "Comma-separated zero-padded set IDs (e.g. 001,002,003) "
            "or 'all' for the full library.  Default: 001,002"
        ),
    )
    args = parser.parse_args()

    raw = args.sets.strip()
    if raw == "all":
        set_ids = ["all"]
    else:
        set_ids = [s.strip().zfill(3) for s in raw.split(",")]

    print(f"Auditing sets: {', '.join(set_ids)}")
    run_audit(set_ids)


if __name__ == "__main__":
    main()
