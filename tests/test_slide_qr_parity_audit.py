"""Tests for the reproducible production-vs-legacy QR parity audit (#203)."""

from __future__ import annotations

import csv

import numpy as np

from slide.qr import SlideQRResult
from tools.identity.audit_slide_qr_parity import (
    audit_paths,
    classify_disagreement,
    write_report,
)


def _result(
    *, success: bool, payload: str | None = None, reason: str = "ok",
) -> SlideQRResult:
    return SlideQRResult(
        success, reason, payload, None, None, None, None, None, None, None,
        None, None,
    )


def test_classify_disagreement_records_all_public_differences():
    production = _result(success=False, payload=None, reason="no production code")
    legacy = _result(success=True, payload="old_payload")

    assert classify_disagreement(production, legacy) == (
        True, "success;raw_payload",
    )


def test_audit_paths_records_payload_reason_duration_and_read_error(tmp_path):
    readable = tmp_path / "readable.png"
    unreadable = tmp_path / "unreadable.png"
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    clocks = iter((1.0, 1.002, 2.0, 2.005))

    def reader(path: str):
        return None if path.endswith("unreadable.png") else image

    rows = audit_paths(
        [readable, unreadable],
        production_decoder=lambda _: _result(success=False, reason="production miss"),
        legacy_decoder=lambda _: _result(success=False, reason="legacy miss"),
        image_reader=reader,
        clock=lambda: next(clocks),
    )

    assert rows[0] == {
        "filename": "readable.png",
        "read_error": None,
        "production_success": False,
        "production_raw_payload": None,
        "production_reason": "production miss",
        "production_duration_ms": "2.000",
        "legacy_success": False,
        "legacy_raw_payload": None,
        "legacy_reason": "legacy miss",
        "legacy_duration_ms": "5.000",
        "disagreement": True,
        "disagreement_reasons": "failure_reason",
    }
    assert rows[1]["read_error"] == "cv2.imread returned None"
    assert rows[1]["disagreement"] is False
    assert rows[1]["disagreement_reasons"] == "read_error"


def test_write_report_includes_matching_and_disagreement_rows(tmp_path):
    out_path = tmp_path / "audit.csv"
    rows = [
        {"filename": "same.png", "disagreement": False},
        {"filename": "different.png", "disagreement": True},
    ]

    write_report(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as handle:
        report = list(csv.DictReader(handle))
    assert [row["filename"] for row in report] == ["same.png", "different.png"]
    assert report[1]["disagreement"] == "True"
