"""Block-readiness evidence written by opt-in live ``--profile`` sessions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from session.processing_store import ProcessingStore  # noqa: E402
from session.outbox_transport import HttpCaptureClient  # noqa: E402
from session.workflow_types import OutboxCapture  # noqa: E402
from session.workflow import LoopbackCaptureReceiver  # noqa: E402
from session import preparation  # noqa: E402
from session.processing_store import preprocess_block  # noqa: E402


STARTED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _capture(path: Path) -> Path:
    assert cv2.imwrite(str(path), np.full((16, 16, 3), 80, dtype=np.uint8))
    return path


def _preprocessor(_path: Path):
    return np.full((16, 16), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


def test_profiled_block_capture_writes_ready_breakdown(tmp_path):
    ticks = iter((0, 1_000_000, 6_000_000, 9_000_000))
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=_preprocessor,
        profile_clock_ns=lambda: next(ticks),
    )
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted

    payload = _capture(tmp_path / "block.png").read_bytes()
    store.receive_capture(
        session.number,
        capture_id="capture_000001_block_51151378_20260727T120000Z",
        block_id="51151378",
        checksum=hashlib.sha256(payload).hexdigest(),
        body=payload,
        profile=True,
    )
    store.wait_for_jobs()

    report = next((tmp_path / "processing").glob("session_*/block_benchmark.csv"))
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{
        "capture_id": "capture_000001_block_51151378_20260727T120000Z",
        "queue_wait_ms": "1",
        "block_preparation_ms": "5",
        "segmentation_ms": "",
        "artifact_write_ms": "3",
        "ready_after_receive_ms": "9",
        "status": "complete",
    }]


def test_unprofiled_block_capture_writes_no_benchmark(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=_preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    payload = _capture(tmp_path / "block.png").read_bytes()

    store.receive_capture(
        session.number,
        capture_id="capture_000001_block_51151378_20260727T120000Z",
        block_id="51151378",
        checksum=hashlib.sha256(payload).hexdigest(),
        body=payload,
    )
    store.wait_for_jobs()

    assert not list((tmp_path / "processing").glob("session_*/block_benchmark.csv"))


def test_profiled_upload_marks_the_pc_block_job_before_it_starts(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=_preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    path = _capture(tmp_path / "block.png")
    capture = OutboxCapture(
        "capture_000001_block_51151378_20260727T120000Z",
        path,
        "51151378",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        STARTED_AT,
        profile=True,
    )

    with LoopbackCaptureReceiver(store) as receiver:
        receipt = HttpCaptureClient(receiver.url).upload(session.number, capture)
    store.wait_for_jobs()

    assert receipt.acknowledged is True
    assert list((tmp_path / "processing").glob("session_*/block_benchmark.csv"))


def test_profiled_default_preparation_reports_segmentation_time(tmp_path, monkeypatch):
    ticks = iter((10_000_000, 15_000_000))

    def _single_pixel_mask(image, _role, **_kwargs):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[image.shape[0] // 2, image.shape[1] // 2] = 255
        return mask

    monkeypatch.setattr(preparation, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(preparation, "segment_tissue", _single_pixel_mask)

    _, metadata = preprocess_block(_capture(tmp_path / "block.png"), profile=True)

    assert metadata["segmentation_ms"] == 5


def test_profiled_failed_block_writes_a_failed_benchmark_row(tmp_path):
    def _fail(_path):
        raise ValueError("synthetic segmentation failure")

    store = ProcessingStore(tmp_path / "processing", preprocessor=_fail)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    payload = _capture(tmp_path / "block.png").read_bytes()
    store.receive_capture(
        session.number,
        capture_id="capture_000001_block_51151378_20260727T120000Z",
        block_id="51151378",
        checksum=hashlib.sha256(payload).hexdigest(),
        body=payload,
        profile=True,
    )
    store.wait_for_jobs()

    report = next((tmp_path / "processing").glob("session_*/block_benchmark.csv"))
    with report.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "failed"
