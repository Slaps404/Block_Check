"""Round-trip fixtures pinning the store_wire contract.

store_wire.py is a pure-stdlib leaf and must never import session_workflow,
slide_qr, or pipeline itself. This test file is the one place allowed to pull
those (and therefore cv2/numpy transitively) in order to exercise the codec
against the real wire dataclasses (registered by ``session.workflow_types``).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import transport.wire as store_wire
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

NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

_WITH_IDS = WorkflowEvent(
    kind="block_captured", session_number=1, phase="blocks", message="ok",
    block_id="B1", capture_id="cap-1",
)
_WITHOUT_IDS = WorkflowEvent(
    kind="phase_changed", session_number=1, phase="finalizing", message="done"
)
_ATTEMPTS = (
    DecodeAttempt(engine="zxing", symbology="QRCode", preprocessing="raw",
                  payload="WO_B1_1_H&E", accepted=False, reason="bad format"),
    DecodeAttempt(engine="cv2", symbology="QRCode", preprocessing="gray",
                  payload="WO_B1_1_H&E", accepted=True, reason="ok"),
)

_ROUND_TRIP = {
    "scan_outcome_accepted": ScanOutcome(accepted=True, message="accepted"),
    "scan_outcome_rejected": ScanOutcome(accepted=False, message="rejected: duplicate"),
    "upload_receipt": UploadReceipt(capture_id="cap-1", acknowledged=True, checksum="deadbeef"),
    "snapshot_minimal": WorkflowSnapshot(
        session_number=1, started_at=NOW, phase="blocks", upload_state="idle",
        preprocessing_pending=0, latest_block_id=None, latest_block_status=None,
    ),
    "snapshot_populated": WorkflowSnapshot(
        session_number=2, started_at=NOW, phase="finalizing", upload_state="uploading",
        preprocessing_pending=3, latest_block_id="B1", latest_block_status="pass",
        pending_transfers=2, unresolved_blocks=1,
    ),
    # Empty tuple fields must stay tuples; finalization_error stays None.
    "summary_empty": SessionSummary(
        session_number=1, started_at=NOW, sets_processed=0, pass_count=0, review_count=0,
    ),
    "summary_nested_failures": SessionSummary(
        session_number=9, started_at=NOW, sets_processed=4, pass_count=2, review_count=2,
        missing_slides=("S1", "S2"),
        block_failures=(
            FailedBlockWarning(block_id="B1", reason="blurry", qc_path=Path("C:/qc/b1.png")),
            FailedBlockWarning(block_id="B2", reason="dark", qc_path=Path("/mnt/qc/b2.png"),
                               can_recapture=False, can_dismiss=False),
        ),
        pending_blocks=("B3",), pending_uploads=("cap-3",), finalization_error="disk full",
    ),
    "failed_block_windows_path": FailedBlockWarning(
        block_id="B1", reason="r", qc_path=Path("C:\\qc\\out.png"),
    ),
    "failed_block_posix_path": FailedBlockWarning(
        block_id="B2", reason="r2", qc_path=Path("/var/qc/out.png"),
        can_recapture=False, can_dismiss=False,
    ),
    "readiness_evaluable": BlockReadiness(evaluable=True, review_reason=None),
    "readiness_review": BlockReadiness(evaluable=False, review_reason="blurry"),
    "event_with_ids": _WITH_IDS,
    "event_without_ids": _WITHOUT_IDS,
    "claim_outcome_pass": ClaimOutcome(
        accepted=True, message="ok", verdict="PASS", score=0.42, stage="score",
        reason="above threshold",
    ),
    "claim_outcome_review_none_floats": ClaimOutcome(
        accepted=False, message="needs review", verdict="REVIEW", score=None,
        stage=None, reason=None,
    ),
    "session_identity": SessionIdentity(
        number=3, started_at=NOW, directory=Path("C:/sessions/3"),
    ),
    "session_identity_fractional_seconds": SessionIdentity(
        number=1, started_at=datetime(2026, 7, 6, 12, 0, 0, 123456, tzinfo=timezone.utc),
        directory=Path("."),
    ),
    "slide_qr_success": SlideQRResult(
        success=True, reason="ok", raw_payload="WO_B1_1_H&E", format="current",
        block_id="B1", slide_num="1", stain="H&E", work_order="WO", email=None,
        genotype=None, engine="cv2", preprocessing="gray", symbology="QRCode",
        attempts=_ATTEMPTS,
    ),
    "slide_qr_failure_all_none": SlideQRResult(
        success=False, reason="no_decode", raw_payload=None, format=None, block_id=None,
        slide_num=None, stain=None, work_order=None, email=None, genotype=None,
        engine=None, preprocessing=None, symbology=None, attempts=(),
    ),
    "decode_attempt_accepted": DecodeAttempt(
        engine="zxing", symbology="DataMatrix", preprocessing="raw", payload="x",
        accepted=True, reason="r",
    ),
    "decode_attempt_rejected": DecodeAttempt(
        engine="zxing", symbology="DataMatrix", preprocessing="raw", payload="x",
        accepted=False, reason="r",
    ),
    "claim_decision_full": ClaimDecision(
        claim_id="C1", block_path="b.png", slide_path="s.png", verdict="PASS",
        stage="score", reason="ok", score=0.91, selected_metric="mask_iou",
        router_size_signal=0.5, block_occupied_fraction=0.3,
        slide_occupied_fraction=0.4, best_angle=15.0, best_flip=True,
        align_soft_iou=0.8, mask_iou=0.75,
    ),
    # Omitted floats stay None and selected_metric keeps its "" default.
    "claim_decision_minimal": ClaimDecision(
        claim_id="C2", block_path="b2.png", slide_path="s2.png", verdict="REVIEW",
        stage="gate", reason="occluded",
    ),
}


def _assert_same_types(restored, original):
    """Type preservation all the way down: tuples stay tuples, Paths stay
    Paths, nested dataclasses come back as their own class."""
    assert type(restored) is type(original)
    if dataclasses.is_dataclass(original):
        for field in dataclasses.fields(original):
            _assert_same_types(getattr(restored, field.name), getattr(original, field.name))
    elif isinstance(original, tuple):
        for restored_item, original_item in zip(restored, original):
            _assert_same_types(restored_item, original_item)


@pytest.mark.parametrize("obj", list(_ROUND_TRIP.values()), ids=list(_ROUND_TRIP))
def test_wire_round_trip_preserves_value_and_type(obj):
    restored = store_wire.loads(store_wire.dumps(obj))

    assert restored == obj
    _assert_same_types(restored, obj)


def test_workflow_event_list_envelope_round_trips_tuple():
    empty_restored = store_wire.loads_list(WorkflowEvent, store_wire.dumps_list(()))
    assert empty_restored == ()
    assert isinstance(empty_restored, tuple)

    three = (_WITH_IDS, _WITHOUT_IDS, _WITH_IDS)
    three_restored = store_wire.loads_list(WorkflowEvent, store_wire.dumps_list(three))
    assert three_restored == three
    assert isinstance(three_restored, tuple)


def test_upload_receipt_decodes_positional_payload():
    # HttpCaptureClient rebuilds UploadReceipt positionally from these three
    # keys today; the codec must decode that shape without an envelope.
    positional = {"capture_id": "cap-2", "acknowledged": False, "checksum": "cafebabe"}
    decoded = store_wire.decode(UploadReceipt, positional)
    assert decoded == UploadReceipt(**positional)


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


def test_encode_rejects_naive_datetime():
    naive = SessionIdentity(
        number=1, started_at=datetime(2026, 7, 6, 12, 0, 0), directory=Path(".")
    )
    with pytest.raises(ValueError):
        store_wire.encode(naive)


def test_loads_unknown_type_raises_value_error():
    text = json.dumps({"type": "Bogus", "data": {}})
    with pytest.raises(ValueError):
        store_wire.loads(text)


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
