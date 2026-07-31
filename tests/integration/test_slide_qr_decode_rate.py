"""
Integration test: decode rate floor for images/pi_images/*slide*.jpg.

Asserts that at least 44 of the 47 real slide images decode successfully
(the proven baseline; the 3 holdouts are blur-limited recapture cases).
Guards against pipeline regressions.

Run explicitly (excluded from default pytest run):
    venv/Scripts/python.exe -m pytest tests/integration/test_slide_qr_decode_rate.py -v
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import cv2
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_CODE = _REPO / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from slide.qr import decode_slide_identity, decode_slide_qr  # noqa: E402

RATE_FLOOR = 44
EXPECTED_TOTAL = 47
IMAGE_GLOB = "images/pi_images/*slide*.jpg"
V3_GROUND_TRUTH = _REPO / "images/pi_images_v3/slide_code_ground_truth.csv"
V3_RATE_FLOOR = 38
V3_EXPECTED_TOTAL = 41


def _slide_paths():
    paths = sorted(_REPO.glob(IMAGE_GLOB))
    return paths


@pytest.fixture(scope="module")
def decode_results():
    """Run decode_slide_qr on all images/pi_images/*slide*.jpg once per session."""
    paths = _slide_paths()
    if not paths:
        pytest.skip(f"No images matched {IMAGE_GLOB} — is images/pi_images/ present?")

    results = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            results.append((p.name, None))
            continue
        result = decode_slide_qr(bgr)
        results.append((p.name, result))
    return results


def test_decode_rate_floor(decode_results):
    """Decoded count must be >= RATE_FLOOR."""
    total = len(decode_results)
    decoded = sum(
        1 for _, r in decode_results if r is not None and r.success
    )
    failures = [
        name for name, r in decode_results
        if r is None or not r.success
    ]

    print(f"\ndecoded {decoded}/{total}")
    if failures:
        print(f"  failures ({len(failures)}): {failures}")

    assert decoded >= RATE_FLOOR, (
        f"Decode rate too low: {decoded}/{total} < floor {RATE_FLOOR}. "
        f"Failures: {failures}"
    )


def test_total_slide_count(decode_results):
    """Sanity check: the full slide set must be present (guards missing data)."""
    total = len(decode_results)
    assert total == EXPECTED_TOTAL, (
        f"Found {total} slide images — expected exactly {EXPECTED_TOTAL}. "
        f"Missing/extra data would silently skew the decode rate."
    )


def test_v3_identity_decode_count_and_per_slide_latency():
    with V3_GROUND_TRUTH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    measurements = []
    for row in rows:
        number = int(row["Slide Number"])
        path = next((_REPO / "images/pi_images_v3").glob(
            f"slide_{number:03d}_*.png"
        ))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, f"could not read {path.name}"
        started = perf_counter()
        result = decode_slide_identity(image)
        duration_ms = (perf_counter() - started) * 1000.0
        measurements.append((path.name, row["ID"], result, duration_ms))

    decoded = sum(result.raw_payload is not None for _, _, result, _ in measurements)
    print(f"\nv3 decoded {decoded}/{len(measurements)}")
    for name, _, result, duration_ms in measurements:
        print(f"  {name}: {duration_ms:.1f} ms ({result.reason})")

    assert len(measurements) == V3_EXPECTED_TOTAL
    assert decoded >= V3_RATE_FLOOR
