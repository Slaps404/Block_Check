"""
decode_slide_qrs.py — Batch-decode slide QR codes and write a CSV report.

Usage:
    python tools/identity/decode_slide_qrs.py
    python tools/identity/decode_slide_qrs.py --glob "images/pi_images/*slide*.jpg"
    python tools/identity/decode_slide_qrs.py --out outputs/diagnostics/my_report.csv

Globs images/pi_images/*slide*.jpg by default, runs decode_slide_qr on each image,
writes one row per slide to a CSV, and prints 'decoded N/M'.

Output CSV columns (per spec §4):
    filename, success, reason, engine, preprocessing, raw_payload, format,
    block_id, slide_num, stain, work_order, email, genotype
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

# Ensure code/ is on sys.path when run as a script from any CWD
_REPO = Path(__file__).resolve().parent.parent.parent
_CODE = _REPO / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from slide.qr import decode_slide_qr  # noqa: E402

# ---------------------------------------------------------------------------
# CSV columns (order matches spec §4)
# ---------------------------------------------------------------------------

_COLUMNS = [
    "filename",
    "success",
    "reason",
    "engine",
    "preprocessing",
    "raw_payload",
    "format",
    "block_id",
    "slide_num",
    "stain",
    "work_order",
    "email",
    "genotype",
]

_DEFAULT_GLOB = "images/pi_images/*slide*.jpg"
_DEFAULT_OUT = "outputs/diagnostics/slide_qr_report.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode slide QR codes and write a CSV report."
    )
    parser.add_argument(
        "--glob",
        default=_DEFAULT_GLOB,
        help=f"Glob pattern for slide images (default: {_DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help=f"Output CSV path (default: {_DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve glob relative to repo root so the tool works from any CWD
    image_paths = sorted(_REPO.glob(args.glob))
    if not image_paths:
        print(f"No images matched: {args.glob}")
        sys.exit(1)

    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoded = 0
    total = len(image_paths)
    rows = []

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            row = {
                "filename": img_path.name,
                "success": False,
                "reason": "cv2.imread returned None",
                "engine": None,
                "preprocessing": None,
                "raw_payload": None,
                "format": None,
                "block_id": None,
                "slide_num": None,
                "stain": None,
                "work_order": None,
                "email": None,
                "genotype": None,
            }
        else:
            result = decode_slide_qr(bgr)
            if result.success:
                decoded += 1
            row = {
                "filename": img_path.name,
                "success": result.success,
                "reason": result.reason,
                "engine": result.engine,
                "preprocessing": result.preprocessing,
                "raw_payload": result.raw_payload,
                "format": result.format,
                "block_id": result.block_id,
                "slide_num": result.slide_num,
                "stain": result.stain,
                "work_order": result.work_order,
                "email": result.email,
                "genotype": result.genotype,
            }
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"decoded {decoded}/{total}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
