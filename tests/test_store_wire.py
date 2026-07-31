"""Round-trip fixtures pinning the store_wire contract (issue #114 slice 1).

store_wire.py is a pure-stdlib leaf and must never import session_workflow,
slide_qr, or pipeline itself. This test file is the one place allowed to pull
those (and therefore cv2/numpy transitively) in order to register the real
wire dataclasses and exercise the codec against them.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import store.wire as store_wire
from session.pipeline import ClaimDecision
from session.workflow import (
    BlockReadiness,
    ClaimOutcome,
    FailedBlockWarning,
    ScanOutcome,
    SessionIdentity,
    SessionSummary,
    UploadReceipt,
    WorkflowEvent,
    WorkflowSnapshot,
)
from slide.qr import DecodeAttempt, SlideQRResult

for _cls in (
    SessionIdentity,
    ScanOutcome,
    UploadReceipt,
    ClaimOutcome,
    WorkflowSnapshot,
    SessionSummary,
    FailedBlockWarning,
    BlockReadiness,
    WorkflowEvent,
    SlideQRResult,
    DecodeAttempt,
    ClaimDecision,
):
    store_wire.register(_cls)

NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# ScanOutcome
# ---------------------------------------------------------------------------

def test_scan_outcome_round_trip():
    for accepted, message in ((True, "accepted"), (False, "rejected: duplicate")):
        obj = ScanOutcome(accepted=accepted, message=message)
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj


# ---------------------------------------------------------------------------
# UploadReceipt
# ---------------------------------------------------------------------------

def test_upload_receipt_round_trip():
    obj = UploadReceipt(capture_id="cap-1", acknowledged=True, checksum="deadbeef")
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj


def test_upload_receipt_decodes_positional_payload():
    # HttpCaptureClient rebuilds UploadReceipt positionally from these three
    # keys today; the codec must decode that shape without an envelope.
    positional = {"capture_id": "cap-2", "acknowledged": False, "checksum": "cafebabe"}
    decoded = store_wire.decode(UploadReceipt, positional)
    assert decoded == UploadReceipt(**positional)


# ---------------------------------------------------------------------------
# WorkflowSnapshot
# ---------------------------------------------------------------------------

def test_workflow_snapshot_round_trip_variants():
    a = WorkflowSnapshot(
        session_number=1,
        started_at=NOW,
        phase="blocks",
        upload_state="idle",
        preprocessing_pending=0,
        latest_block_id=None,
        latest_block_status=None,
    )
    b = WorkflowSnapshot(
        session_number=2,
        started_at=NOW,
        phase="finalizing",
        upload_state="uploading",
        preprocessing_pending=3,
        latest_block_id="B1",
        latest_block_status="pass",
        pending_transfers=2,
        unresolved_blocks=1,
    )
    for obj in (a, b):
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj


def test_workflow_snapshot_normalizes_non_utc_timezone():
    tz = timezone(timedelta(hours=-5))
    started = datetime(2026, 7, 6, 7, 0, 0, tzinfo=tz)
    obj = WorkflowSnapshot(
        session_number=5,
        started_at=started,
        phase="blocks",
        upload_state="idle",
        preprocessing_pending=0,
        latest_block_id=None,
        latest_block_status=None,
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored.started_at.tzinfo is not None
    assert restored.started_at.utcoffset() == timedelta(0)
    assert restored.started_at == started  # same instant


# ---------------------------------------------------------------------------
# SessionSummary (+ nested FailedBlockWarning)
# ---------------------------------------------------------------------------

def test_session_summary_empty_tuples_stay_tuples():
    obj = SessionSummary(
        session_number=1,
        started_at=NOW,
        sets_processed=0,
        pass_count=0,
        review_count=0,
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj
    assert isinstance(restored.missing_slides, tuple)
    assert isinstance(restored.block_failures, tuple)
    assert isinstance(restored.skipped_decodes, tuple)
    assert isinstance(restored.pending_blocks, tuple)
    assert isinstance(restored.pending_uploads, tuple)
    assert restored.finalization_error is None


def test_session_summary_populated_with_nested_failures():
    failures = (
        FailedBlockWarning(block_id="B1", reason="blurry", qc_path=Path("C:/qc/b1.png")),
        FailedBlockWarning(
            block_id="B2",
            reason="dark",
            qc_path=Path("/mnt/qc/b2.png"),
            can_recapture=False,
            can_dismiss=False,
        ),
    )
    obj = SessionSummary(
        session_number=9,
        started_at=NOW,
        sets_processed=4,
        pass_count=2,
        review_count=2,
        missing_slides=("S1", "S2"),
        block_failures=failures,
        pending_blocks=("B3",),
        pending_uploads=("cap-3",),
        finalization_error="disk full",
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj
    assert isinstance(restored.block_failures, tuple)
    assert all(isinstance(item, FailedBlockWarning) for item in restored.block_failures)


# ---------------------------------------------------------------------------
# FailedBlockWarning
# ---------------------------------------------------------------------------

def test_failed_block_warning_paths_and_defaults():
    windows_style = FailedBlockWarning(
        block_id="B1", reason="r", qc_path=Path("C:\\qc\\out.png")
    )
    posix_style = FailedBlockWarning(
        block_id="B2",
        reason="r2",
        qc_path=Path("/var/qc/out.png"),
        can_recapture=False,
        can_dismiss=False,
    )
    for obj in (windows_style, posix_style):
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj
        assert isinstance(restored.qc_path, Path)


# ---------------------------------------------------------------------------
# BlockReadiness
# ---------------------------------------------------------------------------

def test_block_readiness_round_trip():
    for obj in (
        BlockReadiness(evaluable=True, review_reason=None),
        BlockReadiness(evaluable=False, review_reason="blurry"),
    ):
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj


# ---------------------------------------------------------------------------
# WorkflowEvent (+ list envelope)
# ---------------------------------------------------------------------------

def test_workflow_event_round_trip_with_and_without_ids():
    with_ids = WorkflowEvent(
        kind="block_captured",
        session_number=1,
        phase="blocks",
        message="ok",
        block_id="B1",
        capture_id="cap-1",
    )
    without_ids = WorkflowEvent(
        kind="phase_changed", session_number=1, phase="finalizing", message="done"
    )
    for obj in (with_ids, without_ids):
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj


def test_workflow_event_list_envelope_round_trips_tuple():
    with_ids = WorkflowEvent(
        kind="block_captured",
        session_number=1,
        phase="blocks",
        message="ok",
        block_id="B1",
        capture_id="cap-1",
    )
    without_ids = WorkflowEvent(
        kind="phase_changed", session_number=1, phase="finalizing", message="done"
    )

    empty_restored = store_wire.loads_list(WorkflowEvent, store_wire.dumps_list(()))
    assert empty_restored == ()
    assert isinstance(empty_restored, tuple)

    three = (with_ids, without_ids, with_ids)
    three_restored = store_wire.loads_list(WorkflowEvent, store_wire.dumps_list(three))
    assert three_restored == three
    assert isinstance(three_restored, tuple)


# ---------------------------------------------------------------------------
# ClaimOutcome
# ---------------------------------------------------------------------------

def test_claim_outcome_pass_and_review_none_floats_stay_none():
    passed = ClaimOutcome(
        accepted=True,
        message="ok",
        verdict="PASS",
        score=0.42,
        stage="score",
        reason="above threshold",
    )
    review = ClaimOutcome(
        accepted=False,
        message="needs review",
        verdict="REVIEW",
        score=None,
        stage=None,
        reason=None,
    )
    for obj in (passed, review):
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj

    restored_review = store_wire.loads(store_wire.dumps(review))
    assert restored_review.score is None
    assert restored_review.stage is None
    assert restored_review.reason is None


# ---------------------------------------------------------------------------
# SessionIdentity
# ---------------------------------------------------------------------------

def test_session_identity_round_trip():
    obj = SessionIdentity(number=3, started_at=NOW, directory=Path("C:/sessions/3"))
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj


def test_encode_rejects_naive_datetime():
    naive = SessionIdentity(
        number=1, started_at=datetime(2026, 7, 6, 12, 0, 0), directory=Path(".")
    )
    with pytest.raises(ValueError):
        store_wire.encode(naive)


# ---------------------------------------------------------------------------
# SlideQRResult (+ nested DecodeAttempt)
# ---------------------------------------------------------------------------

def test_slide_qr_result_success_round_trip():
    attempts = (
        DecodeAttempt(
            engine="zxing",
            symbology="QRCode",
            preprocessing="raw",
            payload="WO_B1_1_H&E",
            accepted=False,
            reason="bad format",
        ),
        DecodeAttempt(
            engine="cv2",
            symbology="QRCode",
            preprocessing="gray",
            payload="WO_B1_1_H&E",
            accepted=True,
            reason="ok",
        ),
    )
    obj = SlideQRResult(
        success=True,
        reason="ok",
        raw_payload="WO_B1_1_H&E",
        format="current",
        block_id="B1",
        slide_num="1",
        stain="H&E",
        work_order="WO",
        email=None,
        genotype=None,
        engine="cv2",
        preprocessing="gray",
        symbology="QRCode",
        attempts=attempts,
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj
    assert isinstance(restored.attempts, tuple)
    assert all(isinstance(item, DecodeAttempt) for item in restored.attempts)


def test_slide_qr_result_failure_round_trip_all_optionals_none():
    obj = SlideQRResult(
        success=False,
        reason="no_decode",
        raw_payload=None,
        format=None,
        block_id=None,
        slide_num=None,
        stain=None,
        work_order=None,
        email=None,
        genotype=None,
        engine=None,
        preprocessing=None,
        symbology=None,
        attempts=(),
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored == obj
    assert restored.attempts == ()
    assert isinstance(restored.attempts, tuple)


# ---------------------------------------------------------------------------
# DecodeAttempt (standalone)
# ---------------------------------------------------------------------------

def test_decode_attempt_round_trip():
    for accepted in (True, False):
        obj = DecodeAttempt(
            engine="zxing",
            symbology="DataMatrix",
            preprocessing="raw",
            payload="x",
            accepted=accepted,
            reason="r",
        )
        restored = store_wire.loads(store_wire.dumps(obj))
        assert restored == obj


# ---------------------------------------------------------------------------
# ClaimDecision (not frozen -> field-by-field asserts, not ==)
# ---------------------------------------------------------------------------

def test_claim_decision_full_round_trip_field_by_field():
    obj = ClaimDecision(
        claim_id="C1",
        block_path="b.png",
        slide_path="s.png",
        verdict="PASS",
        stage="score",
        reason="ok",
        score=0.91,
        selected_metric="mask_iou",
        router_size_signal=0.5,
        block_occupied_fraction=0.3,
        slide_occupied_fraction=0.4,
        best_angle=15.0,
        best_flip=True,
        align_soft_iou=0.8,
        mask_iou=0.75,
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored.claim_id == obj.claim_id
    assert restored.block_path == obj.block_path
    assert restored.slide_path == obj.slide_path
    assert restored.verdict == obj.verdict
    assert restored.stage == obj.stage
    assert restored.reason == obj.reason
    assert restored.score == obj.score
    assert restored.selected_metric == obj.selected_metric
    assert restored.router_size_signal == obj.router_size_signal
    assert restored.block_occupied_fraction == obj.block_occupied_fraction
    assert restored.slide_occupied_fraction == obj.slide_occupied_fraction
    assert restored.best_angle == obj.best_angle
    assert restored.best_flip == obj.best_flip
    assert restored.align_soft_iou == obj.align_soft_iou
    assert restored.mask_iou == obj.mask_iou


def test_claim_decision_minimal_round_trip_none_floats_stay_none():
    obj = ClaimDecision(
        claim_id="C2",
        block_path="b2.png",
        slide_path="s2.png",
        verdict="REVIEW",
        stage="gate",
        reason="occluded",
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored.claim_id == obj.claim_id
    assert restored.block_path == obj.block_path
    assert restored.slide_path == obj.slide_path
    assert restored.verdict == obj.verdict
    assert restored.stage == obj.stage
    assert restored.reason == obj.reason
    assert restored.score is None
    assert restored.selected_metric == ""
    assert restored.router_size_signal is None
    assert restored.block_occupied_fraction is None
    assert restored.slide_occupied_fraction is None
    assert restored.best_angle is None
    assert restored.best_flip is None
    assert restored.align_soft_iou is None
    assert restored.mask_iou is None


# ---------------------------------------------------------------------------
# Envelope dispatch
# ---------------------------------------------------------------------------

def test_dumps_loads_dispatch_via_registry():
    obj = WorkflowSnapshot(
        session_number=1,
        started_at=NOW,
        phase="blocks",
        upload_state="idle",
        preprocessing_pending=0,
        latest_block_id=None,
        latest_block_status=None,
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert isinstance(restored, WorkflowSnapshot)
    assert restored == obj


def test_loads_unknown_type_raises_value_error():
    text = json.dumps({"type": "Bogus", "data": {}})
    with pytest.raises(ValueError):
        store_wire.loads(text)


# ---------------------------------------------------------------------------
# Datetime edge cases
# ---------------------------------------------------------------------------

def test_datetime_fractional_seconds_round_trip():
    obj = SessionIdentity(
        number=1,
        started_at=datetime(2026, 7, 6, 12, 0, 0, 123456, tzinfo=timezone.utc),
        directory=Path("."),
    )
    restored = store_wire.loads(store_wire.dumps(obj))
    assert restored.started_at == obj.started_at


def test_datetime_z_suffix_and_offset_decode_to_same_instant():
    payload_z = {"number": 1, "started_at": "2026-07-06T12:00:00Z", "directory": "."}
    payload_offset = {"number": 1, "started_at": "2026-07-06T12:00:00+00:00", "directory": "."}
    decoded_z = store_wire.decode(SessionIdentity, payload_z)
    decoded_offset = store_wire.decode(SessionIdentity, payload_offset)
    assert decoded_z.started_at == decoded_offset.started_at
    assert decoded_z.started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Passthrough (untyped store returns)
# ---------------------------------------------------------------------------

def test_passthrough_dict_round_trip():
    d = {"a": 1, "b": None, "c": 2.5, "d": "text"}
    restored = store_wire.passthrough_dict(d)
    assert restored == d


def test_passthrough_dict_tuple_with_embedded_attempts_list():
    captures = (
        {
            "capture_id": "cap-1",
            "block_id": "B1",
            "state": "uploaded",
            "checksum": None,
            "attempts": [
                {"engine": "zxing", "accepted": True, "payload": "x", "reason": None},
                {"engine": "cv2", "accepted": False, "payload": None, "reason": "no decode"},
            ],
        },
        {
            "capture_id": "cap-2",
            "block_id": None,
            "state": "pending",
            "checksum": "abc123",
            "attempts": [],
        },
    )
    restored = store_wire.passthrough_dict_tuple(captures)
    assert restored == captures
    assert isinstance(restored, tuple)
