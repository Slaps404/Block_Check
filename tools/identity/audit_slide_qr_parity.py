"""Compare production and legacy slide-QR decoders on one image corpus.

This is evidence collection for #203, not a decoder-selection policy.  It runs
both public entry points on every image matched by one glob and writes a CSV
that preserves payloads, failure reasons, and timings for review.

Example:
    C:\\Users\\esears\\projects\\ljiblockcheck\\venv\\Scripts\\python.exe \
        tools\\identity\\audit_slide_qr_parity.py \
        --glob "images/pi_images/*slide*.jpg" \
        --out outputs/diagnostics/slide_qr_parity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_CODE = _REPO / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from slide.qr import SlideQRResult, decode_slide_identity, decode_slide_qr  # noqa: E402

_DEFAULT_GLOB = "images/pi_images/*slide*.jpg"
_DEFAULT_OUT = "outputs/diagnostics/slide_qr_parity.csv"

_COLUMNS = [
    "filename",
    "read_error",
    "production_success",
    "production_raw_payload",
    "production_reason",
    "production_duration_ms",
    "legacy_success",
    "legacy_raw_payload",
    "legacy_reason",
    "legacy_duration_ms",
    "disagreement",
    "disagreement_reasons",
]


def classify_disagreement(
    production: SlideQRResult, legacy: SlideQRResult,
) -> tuple[bool, str]:
    """Return all observable public-result differences, in stable order."""
    differences = []
    if production.success != legacy.success:
        differences.append("success")
    if production.raw_payload != legacy.raw_payload:
        differences.append("raw_payload")
    if not production.success and not legacy.success:
        if production.reason != legacy.reason:
            differences.append("failure_reason")
    return bool(differences), ";".join(differences)


def _timed_decode(
    decoder: Callable[[np.ndarray], SlideQRResult], image: np.ndarray,
    clock: Callable[[], float],
) -> tuple[SlideQRResult, float]:
    started = clock()
    result = decoder(image)
    return result, (clock() - started) * 1000.0


def audit_paths(
    image_paths: Iterable[Path],
    *,
    production_decoder: Callable[[np.ndarray], SlideQRResult] = decode_slide_identity,
    legacy_decoder: Callable[[np.ndarray], SlideQRResult] = decode_slide_qr,
    image_reader: Callable[[str], np.ndarray | None] = cv2.imread,
    clock: Callable[[], float] = perf_counter,
) -> list[dict[str, object]]:
    """Audit exactly the supplied corpus and return one report row per path."""
    rows = []
    for path in image_paths:
        image = image_reader(str(path))
        if image is None:
            rows.append({
                "filename": path.name,
                "read_error": "cv2.imread returned None",
                "production_success": None,
                "production_raw_payload": None,
                "production_reason": None,
                "production_duration_ms": None,
                "legacy_success": None,
                "legacy_raw_payload": None,
                "legacy_reason": None,
                "legacy_duration_ms": None,
                "disagreement": False,
                "disagreement_reasons": "read_error",
            })
            continue

        production, production_duration = _timed_decode(
            production_decoder, image, clock
        )
        legacy, legacy_duration = _timed_decode(legacy_decoder, image, clock)
        disagreement, reasons = classify_disagreement(production, legacy)
        rows.append({
            "filename": path.name,
            "read_error": None,
            "production_success": production.success,
            "production_raw_payload": production.raw_payload,
            "production_reason": production.reason,
            "production_duration_ms": f"{production_duration:.3f}",
            "legacy_success": legacy.success,
            "legacy_raw_payload": legacy.raw_payload,
            "legacy_reason": legacy.reason,
            "legacy_duration_ms": f"{legacy_duration:.3f}",
            "disagreement": disagreement,
            "disagreement_reasons": reasons,
        })
    return rows


def write_report(rows: Iterable[dict[str, object]], out_path: Path) -> None:
    """Write the complete audit, including matching rows, as a CSV artifact."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit production/legacy slide-QR decoder parity on one corpus."
    )
    parser.add_argument(
        "--glob", default=_DEFAULT_GLOB,
        help=f"Image glob relative to repo root (default: {_DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help=f"CSV report path relative to repo root (default: {_DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = sorted(_REPO.glob(args.glob))
    if not paths:
        print(f"No images matched: {args.glob}")
        raise SystemExit(1)

    rows = audit_paths(paths)
    out_path = _REPO / args.out
    write_report(rows, out_path)

    disagreements = [row for row in rows if row["disagreement"]]
    counts = Counter(
        reason
        for row in disagreements
        for reason in str(row["disagreement_reasons"]).split(";")
        if reason
    )
    print(f"audited {len(rows)} images, disagreements {len(disagreements)}")
    if counts:
        print("disagreement types: " + ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        ))
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
