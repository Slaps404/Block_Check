"""TDD coverage for the request_id idempotency ledger (issue #114 final slice).

Covers the four mutating methods the ledger activates for: `scan_block`,
`record_slide_capture`, `resolve_claim`, and `record_event`. The load-bearing
rule under test throughout: passing `request_id=None` must reproduce today's
exact behavior (no ledger, no fingerprint) so the 63-test in-process seam in
`tests/test_session_workflow.py` stays untouched.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

import store.wire as store_wire
from session.preparation import PreparedSpecimen
from slide.qr import DecodeCandidate, select_slide_identity
from session.workflow import (
    ClaimOutcome,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    ScanOutcome,
    _RPC_METHODS,
)

STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

_CAPTURE_PNGS: dict[int, bytes] = {}


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """Mirror test_session_workflow.py: keep tests off full-resolution QC
    rendering, which requires the mask/capture dimensions to match."""

    def write_qc(capture, mask, destination):
        assert capture.is_file()
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        assert capture.is_file()
        assert reason
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


def _capture(path: Path, value: int) -> Path:
    encoded = _CAPTURE_PNGS.get(value)
    if encoded is None:
        image = np.full((3040, 4056, 3), value, dtype=np.uint8)
        success, png = cv2.imencode(".png", image)
        assert success
        encoded = png.tobytes()
        _CAPTURE_PNGS[value] = encoded
    path.write_bytes(encoded)
    return path


class FastPreprocessor:
    def __call__(self, capture_path):
        assert capture_path.is_file()
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}


def _counting_slide_preprocessor(calls: dict):
    def _preprocess(_img):
        calls["count"] = calls.get("count", 0) + 1
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        )

    return _preprocess


def _valid_slide_result(block_id: str):
    return select_slide_identity((
        DecodeCandidate("zxing", "QRCode", "raw", f"12080_{block_id}_01_HE"),
    ))


def _evaluable_block(store, session, tmp_path, block_id="51151378"):
    assert store.scan_block(session.number, block_id).accepted
    capture = PiOutbox(tmp_path / "outbox_for_test").publish_block(
        _capture(tmp_path / f"{block_id}_block.png", 80), block_id, STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    return block_id


# --------------------------------------------------------------------------
# scan_block
# --------------------------------------------------------------------------


def test_scan_block_replays_original_outcome_on_repeated_request_id(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    first = store.scan_block(session.number, "51151378", request_id="A")
    second = store.scan_block(session.number, "51151378", request_id="A")

    assert first == ScanOutcome(True, "Accepted block 51151378")
    assert second == ScanOutcome(True, "Accepted block 51151378")
    assert second.accepted is True

    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("block_scanned") == 1


def test_scan_block_different_request_id_still_rejects_genuine_duplicate(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    first = store.scan_block(session.number, "51151378", request_id="A")
    second = store.scan_block(session.number, "51151378", request_id="B")

    assert first.accepted is True
    assert second == ScanOutcome(False, "Block already scanned")


def test_scan_block_request_id_reuse_with_different_args_raises_value_error(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.scan_block(session.number, "51151378", request_id="A")

    with pytest.raises(ValueError):
        store.scan_block(session.number, "51151300", request_id="A")


def test_same_request_id_is_independent_between_sessions(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    first_session = store.start_session(started_at=STARTED_AT)
    second_session = store.start_session(started_at=STARTED_AT + timedelta(seconds=1))

    first = store.scan_block(first_session.number, "51151378", request_id="A")
    second = store.scan_block(second_session.number, "51151378", request_id="A")

    assert first.accepted is True
    assert second.accepted is True
    assert store.get_set(first_session.number, "51151378")["block_id"] == "51151378"
    assert store.get_set(second_session.number, "51151378")["block_id"] == "51151378"


def test_same_request_id_is_independent_between_methods(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378", request_id="A")
    slide_path = _capture(tmp_path / "slide.png", 120)

    outcome = store.resolve_claim(
        session.number, "51151300", "slide_capture_1", slide_path,
        request_id="A",
    )

    assert outcome.accepted is True
    assert outcome.verdict == "REVIEW"


def test_scan_block_without_request_id_preserves_todays_behavior(tmp_path):
    """The None-request_id path must stay byte-identical: a repeated scan
    without a request_id is (and must remain) a genuine rejection."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    first = store.scan_block(session.number, "51151378")
    second = store.scan_block(session.number, "51151378")

    assert first.accepted is True
    assert second == ScanOutcome(False, "Block already scanned")


# --------------------------------------------------------------------------
# record_event
# --------------------------------------------------------------------------


def test_record_event_same_request_id_twice_appends_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    baseline = len(store.events(session.number))

    store.record_event(session.number, "operator_note", "hello", request_id="E")
    store.record_event(session.number, "operator_note", "hello", request_id="E")

    events = store.events(session.number)
    assert len(events) == baseline + 1
    assert events[-1].kind == "operator_note"


def test_record_event_without_request_id_appends_every_time(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    baseline = len(store.events(session.number))

    store.record_event(session.number, "operator_note", "hello")
    store.record_event(session.number, "operator_note", "hello")

    events = store.events(session.number)
    assert len(events) == baseline + 2


# --------------------------------------------------------------------------
# record_slide_capture (+ resolve_claim cascade)
# --------------------------------------------------------------------------


def test_record_slide_capture_same_request_id_replays_capture_id_and_fires_cascade_once(
    tmp_path,
):
    calls: dict = {}
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_counting_slide_preprocessor(calls),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    slide_path = _capture(tmp_path / "slide.png", 120)
    result = _valid_slide_result(block_id)

    first_id = store.record_slide_capture(
        session.number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=10.0, request_id="S1",
    )
    second_id = store.record_slide_capture(
        session.number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=10.0, request_id="S1",
    )

    assert first_id == second_id
    with_prefix = store.slide_captures(session.number)
    assert len(with_prefix) == 1
    assert calls.get("count", 0) == 1

    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"
    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("claim_pass") == 1
    assert kinds.count("slide_identity_validated") == 1


def test_record_slide_capture_retry_resumes_failed_claim_cascade(tmp_path, monkeypatch):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=lambda _img: PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    slide_path = _capture(tmp_path / "slide.png", 120)
    result = _valid_slide_result(block_id)
    original_resolve_claim = store.resolve_claim
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("forced cascade failure")
        return original_resolve_claim(*args, **kwargs)

    monkeypatch.setattr(store, "resolve_claim", fail_once)

    with pytest.raises(RuntimeError, match="forced cascade failure"):
        store.record_slide_capture(
            session.number, slide_path, captured_at=STARTED_AT, result=result,
            duration_ms=10.0, request_id="S1",
        )

    assert store.get_set(session.number, block_id)["verdict"] is None

    capture_id = store.record_slide_capture(
        session.number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=10.0, request_id="S1",
    )

    assert calls == 2
    assert len(store.slide_captures(session.number)) == 1
    row = store.get_set(session.number, block_id)
    assert row["slide_capture_id"] == capture_id
    assert row["verdict"] == "PASS"


def test_record_slide_capture_without_request_id_preserves_todays_behavior(tmp_path):
    """None-request_id path: calling twice with a fresh capture_id each time
    (today's real behavior, since capture_id is time+checksum derived) must
    be unaffected by the ledger -- both calls run to completion."""
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=lambda _img: PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    slide_path = _capture(tmp_path / "slide.png", 120)
    result = _valid_slide_result(block_id)

    capture_id = store.record_slide_capture(
        session.number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=10.0,
    )
    assert capture_id is not None
    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"


# --------------------------------------------------------------------------
# resolve_claim (direct)
# --------------------------------------------------------------------------


def test_resolve_claim_replay_returns_identical_outcome(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=lambda _img: PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)

    first = store.resolve_claim(
        session.number, block_id, "slide_capture_1", slide_path, request_id="R1",
    )
    second = store.resolve_claim(
        session.number, block_id, "slide_capture_1", slide_path, request_id="R1",
    )

    assert first == second
    assert first.accepted is True
    assert first.verdict == "PASS"


def test_resolve_claim_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=lambda _img: PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)

    first = store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)
    second = store.resolve_claim(session.number, block_id, "slide_capture_2", slide_path)

    assert first.accepted is True
    assert second == ClaimOutcome(False, "Slide already processed")


# --------------------------------------------------------------------------
# finish_work_order
# --------------------------------------------------------------------------


def test_finish_work_order_delayed_retry_cannot_mutate_a_newer_work_order(tmp_path):
    """The acceptance test: a retry that arrives after a NEW bracket has
    already opened must replay the ORIGINAL id and leave the newer, still-
    open work order completely untouched."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.start_work_order(session.number)
    a_id = store.finish_work_order(session.number, start_job=False, request_id="F1")

    b_id = store.start_work_order(session.number)
    again = store.finish_work_order(session.number, start_job=False, request_id="F1")

    assert again == a_id
    assert store.open_work_order_id(session.number) == b_id


def test_finish_work_order_same_request_id_twice_submits_job_once(tmp_path, monkeypatch):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)

    submit_calls = {"count": 0}
    original_submit = store._executor.submit

    def counting_submit(fn, *args, **kwargs):
        submit_calls["count"] += 1
        return original_submit(fn, *args, **kwargs)

    monkeypatch.setattr(store._executor, "submit", counting_submit)

    first = store.finish_work_order(session.number, request_id="F2")
    second = store.finish_work_order(session.number, request_id="F2")
    store.wait_for_jobs()

    assert first == second
    assert submit_calls["count"] == 1


def test_finish_work_order_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.start_work_order(session.number)
    first_id = store.finish_work_order(session.number, start_job=False)

    store.start_work_order(session.number)
    second_id = store.finish_work_order(session.number, start_job=False)

    assert first_id != second_id
    assert store.get_work_order(session.number, first_id)["lifecycle_state"] == "scoring"
    assert store.get_work_order(session.number, second_id)["lifecycle_state"] == "scoring"


# --------------------------------------------------------------------------
# dismiss_block
# --------------------------------------------------------------------------


class _FailingPreprocessor:
    def __call__(self, capture_path):
        raise ValueError("cassette window is not evaluable")


def _failed_block(store, session, tmp_path, block_id="51151378"):
    assert store.scan_block(session.number, block_id).accepted
    capture = PiOutbox(tmp_path / "outbox_for_dismiss").publish_block(
        _capture(tmp_path / f"{block_id}_failed.png", 80), block_id, STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    return block_id


def test_dismiss_block_same_request_id_twice_fires_event_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=_FailingPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    block_id = _failed_block(store, session, tmp_path)

    store.dismiss_block(session.number, block_id, reason="operator confirmed", request_id="D1")
    store.dismiss_block(session.number, block_id, reason="operator confirmed", request_id="D1")

    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("block_dismissed") == 1
    row = store.get_set(session.number, block_id)
    assert row["preprocessing_status"] == "unusable"


def test_dismiss_block_same_request_id_different_reason_raises_value_error(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=_FailingPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    block_id = _failed_block(store, session, tmp_path)

    store.dismiss_block(session.number, block_id, reason="reason one", request_id="D2")

    with pytest.raises(ValueError):
        store.dismiss_block(session.number, block_id, reason="reason two", request_id="D2")


def test_dismiss_block_genuine_second_dismiss_still_raises(tmp_path):
    """A different request_id (or none) dismissing an already-unusable block
    is still today's genuine rejection -- the ledger must not swallow it."""
    store = ProcessingStore(tmp_path / "processing", preprocessor=_FailingPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    block_id = _failed_block(store, session, tmp_path)

    store.dismiss_block(session.number, block_id, reason="first", request_id="D3")

    with pytest.raises(ValueError, match="only a failed block can be dismissed"):
        store.dismiss_block(session.number, block_id, reason="second", request_id="D4")


def test_dismiss_block_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=_FailingPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    block_id = _failed_block(store, session, tmp_path)

    store.dismiss_block(session.number, block_id, reason="operator confirmed")

    with pytest.raises(ValueError, match="only a failed block can be dismissed"):
        store.dismiss_block(session.number, block_id, reason="operator confirmed")


# --------------------------------------------------------------------------
# start_session
# --------------------------------------------------------------------------


def test_start_session_same_request_id_twice_returns_identical_identity_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing")

    first = store.start_session(started_at=STARTED_AT, request_id="S1")
    second = store.start_session(started_at=STARTED_AT, request_id="S1")

    assert first == second
    with store._connect() as db:
        count = db.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    assert count == 1
    assert sum(1 for _ in store.root.glob("session_*")) == 1


def test_start_session_without_request_id_creates_two_sessions(tmp_path):
    store = ProcessingStore(tmp_path / "processing")

    first = store.start_session(started_at=STARTED_AT)
    second = store.start_session(started_at=STARTED_AT + timedelta(seconds=1))

    assert first.number != second.number


def test_start_session_same_request_id_different_started_at_raises_value_error(tmp_path):
    store = ProcessingStore(tmp_path / "processing")

    store.start_session(started_at=STARTED_AT, request_id="S2")

    with pytest.raises(ValueError):
        store.start_session(started_at=STARTED_AT + timedelta(seconds=1), request_id="S2")


def test_start_session_persists_session_mode_and_ledger_replay_with_a_different_mode_raises(
    tmp_path,
):
    """#269: session_mode is hashed into start_session's ledger fingerprint
    alongside started_at (mirrors the started_at-mismatch-raises contract
    directly above) -- a replayed request_id with a DIFFERENT mode is a
    genuine conflicting-request bug, not a legitimate replay, and must raise
    rather than silently returning the first session's cached identity."""
    store = ProcessingStore(tmp_path / "processing")

    session = store.start_session(
        started_at=STARTED_AT, session_mode="hybrid", request_id="S3"
    )
    assert store._session_mode(session.number) == "hybrid"

    same = store.start_session(
        started_at=STARTED_AT, session_mode="hybrid", request_id="S3"
    )
    assert same == session

    with pytest.raises(ValueError):
        store.start_session(
            started_at=STARTED_AT, session_mode="hybrid_shadow", request_id="S3"
        )


def test_start_session_rejects_an_unknown_session_mode(tmp_path):
    store = ProcessingStore(tmp_path / "processing")

    with pytest.raises(ValueError, match="session_mode"):
        store.start_session(started_at=STARTED_AT, session_mode="not-a-real-mode")


# --------------------------------------------------------------------------
# unscan_block
# --------------------------------------------------------------------------


def test_unscan_block_same_request_id_replays_true_after_row_already_gone(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    first = store.unscan_block(session.number, "51151378", request_id="U1")
    second = store.unscan_block(session.number, "51151378", request_id="U1")

    assert first is True
    assert second is True


def test_unscan_block_different_request_id_on_already_removed_block_returns_false(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    first = store.unscan_block(session.number, "51151378", request_id="U2")
    second = store.unscan_block(session.number, "51151378", request_id="U3")

    assert first is True
    assert second is False


def test_unscan_block_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    first = store.unscan_block(session.number, "51151378")
    second = store.unscan_block(session.number, "51151378")

    assert first is True
    assert second is False


# --------------------------------------------------------------------------
# HTTP-level: the issue's explicit test
# --------------------------------------------------------------------------


def _rpc(url: str, session_number: int, method: str, args=None, request_id=None):
    body = {"method": method, "args": args or []}
    if request_id is not None:
        body["request_id"] = request_id
    request = Request(
        f"{url}/sessions/{session_number}/rpc",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urlopen(request, timeout=2)
        return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _profile_summary_csv(root: Path, session_number: int) -> Path:
    matches = list(root.glob(f"session_{session_number:06d}_*/profile_summary.csv"))
    assert matches, f"no profile_summary.csv under {root} for session {session_number}"
    return matches[0]


_PROFILE_FIELDS = {
    "camera_capture_ms": 100,
    "publish_ms": 20,
    "consumer_ms": 30,
    "session_accept_ms": 5,
    "total_capture_ms": 155,
    "final_file_size_bytes": 123456,
    "capture_mode": "block",
}


# --------------------------------------------------------------------------
# skip_unreadable_slide
# --------------------------------------------------------------------------


def _unreadable_slide_session(store, tmp_path):
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)
    store.record_slide_capture(
        session.number,
        _capture(tmp_path / "unreadable.png", 120),
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=1500.0,
    )
    assert store.slide_recovery_state(session.number) == "reposition"
    return session


def test_skip_unreadable_slide_same_request_id_twice_fires_event_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = _unreadable_slide_session(store, tmp_path)

    first = store.skip_unreadable_slide(session.number, request_id="K1")
    second = store.skip_unreadable_slide(session.number, request_id="K1")

    assert first is None
    assert second is None
    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("unreadable_slide_skipped") == 1
    assert store.slide_recovery_state(session.number) == "waiting_for_removal"


def test_skip_unreadable_slide_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = _unreadable_slide_session(store, tmp_path)

    store.skip_unreadable_slide(session.number)
    with pytest.raises(ValueError, match="Skip is only valid for an unreadable slide"):
        store.skip_unreadable_slide(session.number)


# --------------------------------------------------------------------------
# mark_waiting_for_slide
# --------------------------------------------------------------------------


def test_mark_waiting_for_slide_same_request_id_twice_fires_event_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.mark_waiting_for_slide(session.number, request_id="W1")
    store.mark_waiting_for_slide(session.number, request_id="W1")

    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("waiting_for_slide") == 1


def test_mark_waiting_for_slide_replay_does_not_clobber_a_newer_state(tmp_path):
    """The acceptance test: a delayed retry replaying the SAME request_id
    after the recovery state has genuinely moved on must not stomp it back
    to 'waiting'."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.mark_waiting_for_slide(session.number, request_id="W2")
    with store._connect() as db:
        db.execute(
            "UPDATE sessions SET slide_recovery_state='reposition' "
            "WHERE session_number=?",
            (session.number,),
        )

    store.mark_waiting_for_slide(session.number, request_id="W2")

    assert store.slide_recovery_state(session.number) == "reposition"


def test_mark_waiting_for_slide_without_request_id_preserves_todays_behavior(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    with store._connect() as db:
        db.execute(
            "UPDATE sessions SET slide_recovery_state='reposition' "
            "WHERE session_number=?",
            (session.number,),
        )

    store.mark_waiting_for_slide(session.number)

    assert store.slide_recovery_state(session.number) == "waiting"


# --------------------------------------------------------------------------
# record_profile_capture
# --------------------------------------------------------------------------


def test_record_profile_capture_same_request_id_twice_appends_one_row(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)

    store.record_profile_capture(
        session.number, "capture_000001", _PROFILE_FIELDS, request_id="P1"
    )
    store.record_profile_capture(
        session.number, "capture_000001", _PROFILE_FIELDS, request_id="P1"
    )

    csv_path = _profile_summary_csv(root, session.number)
    data_rows = csv_path.read_text(encoding="utf-8").splitlines()[1:]
    assert len(data_rows) == 1


def test_record_profile_capture_same_request_id_different_fields_raises_value_error(
    tmp_path,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    store.record_profile_capture(
        session.number, "capture_000001", _PROFILE_FIELDS, request_id="P2"
    )

    with pytest.raises(ValueError):
        store.record_profile_capture(
            session.number, "capture_000002", _PROFILE_FIELDS, request_id="P2"
        )


def test_record_profile_capture_without_request_id_preserves_todays_behavior(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)

    store.record_profile_capture(session.number, "capture_000001", _PROFILE_FIELDS)
    store.record_profile_capture(session.number, "capture_000001", _PROFILE_FIELDS)

    csv_path = _profile_summary_csv(root, session.number)
    data_rows = csv_path.read_text(encoding="utf-8").splitlines()[1:]
    assert len(data_rows) == 2


# --------------------------------------------------------------------------
# classification guard: every RPC method must be a deliberate ledger decision
# --------------------------------------------------------------------------


def test_every_rpc_method_is_classified_into_exactly_one_ledger_bucket():
    """Acceptance test for issue #196: any FUTURE new RPC method must force a
    classification decision here or this test fails.

    Three buckets, pairwise disjoint, whose union is the full `_RPC_METHODS`
    whitelist:
      - `_LEDGERED`: request_id-guarded via the request_ledger table (this
        slice's methods, plus the earlier scan_block/finish_work_order/etc
        slices).
      - `_REPLAY_SAFE_MUTATION`: mutating, but already idempotent on retry
        via SELECT-before-write guards or idempotent projections, so no
        ledger is needed.
      - `_READ_ONLY`: never mutates, so replay safety is not a question.
    """
    _LEDGERED = {
        "scan_block", "record_slide_capture", "resolve_claim", "record_event",
        "finish_work_order", "dismiss_block", "start_session", "unscan_block",
        "skip_unreadable_slide", "mark_waiting_for_slide", "record_profile_capture",
        "freeze_hybrid_pool", "retry_hybrid_slide", "recapture_hybrid_slide",
    }
    _REPLAY_SAFE_MUTATION = {
        "start_work_order", "begin_block_drain", "try_enter_slides",
        "begin_finalization", "prepare_finalization", "complete_finalization",
        "record_finalization_error", "reconcile_session_metadata", "resume_session",
        "record_slide_benchmark",
    }
    _READ_ONLY = {
        "awaiting_capture_blocks", "wait_for_jobs", "get_set", "precheck_slide_scan",
        "slide_captures", "slide_recovery_state", "active_warnings", "summarize",
        "block_readiness", "events", "open_work_order_id", "has_work_orders",
            "list_results_ready_work_orders", "list_hybrid_results",
            "list_retrieval_results",
        "list_hybrid_profile_rows",
    }

    assert _LEDGERED & _REPLAY_SAFE_MUTATION == set()
    assert _LEDGERED & _READ_ONLY == set()
    assert _REPLAY_SAFE_MUTATION & _READ_ONLY == set()
    assert _LEDGERED | _REPLAY_SAFE_MUTATION | _READ_ONLY == set(_RPC_METHODS)


def test_scan_block_via_rpc_replays_original_response_on_repeated_request_id(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status1, body1 = _rpc(
            receiver.url, session.number, "scan_block", args=["51151378"],
            request_id="wire-1",
        )
        status2, body2 = _rpc(
            receiver.url, session.number, "scan_block", args=["51151378"],
            request_id="wire-1",
        )

    assert status1 == 200
    assert status2 == 200
    outcome1 = store_wire.loads_as(ScanOutcome, body1.decode("utf-8"))
    outcome2 = store_wire.loads_as(ScanOutcome, body2.decode("utf-8"))
    assert outcome1 == ScanOutcome(True, "Accepted block 51151378")
    assert outcome2 == ScanOutcome(True, "Accepted block 51151378")
