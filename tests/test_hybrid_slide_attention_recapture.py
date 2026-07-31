"""TDD coverage for #256: Hybrid ERROR retry, recapture supersession, and
scheduling/idempotency for both.

Reuses tests/test_hybrid_slide_queue.py's synthetic harness (fixed-mask
block/slide preprocessors, an identical-fingerprint builder, a trivial
score_cache_builder) -- no camera, no network, no real timing (Event-based
blocking stands in for "the job is genuinely still running", exactly as the
issue's own testing decision prescribes).

A prior ERROR row is simulated the same way tests/test_hybrid_restart_
recovery.py already does -- `store._set_slide_job_state(capture_id, "error")`
-- rather than requiring a real, unreproducible scoring failure, except
where a genuine failure is the point of the test (the retry-fails-again
case below uses a real, deterministic preprocessing failure).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from session.preparation import PreparedSpecimen
from session.processing_store import ProcessingStore
from tests.test_hybrid_slide_queue import (
    _freeze_hybrid_session,
    _make_store,
    _score_cache_builder,
    _valid_slide_result,
    _write_slide_png,
)

STARTED_AT = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """Mirrors tests/test_hybrid_slide_queue.py's fixture of the same name:
    the real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do
    not. Autouse fixtures do not cross module boundaries, so this file
    needs its own copy."""

    def write_qc(capture, mask, destination):
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


# ---------------------------------------------------------------------------
# 1. Retry re-runs from the durable capture; no new capture row.
# ---------------------------------------------------------------------------


def test_retry_reruns_from_durable_capture_without_a_new_capture_row(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    # `start_job=False` + a direct job_state write (mirrors test_hybrid_
    # restart_recovery.py's own precedent): a genuine Hybrid Processing
    # Error never reaches `_finalize_claim`, so the block's `sets.verdict`
    # is durably NULL when it lands as ERROR -- forcing job_state='error'
    # AFTER a real completed run would leave a verdict already written,
    # which is not the invariant a real ERROR row ever has.
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(capture_id, "error")
    before_count = len(store.slide_captures(session_number))

    accepted = store.retry_hybrid_slide(session_number, capture_id)
    assert accepted is True
    store.wait_for_jobs()

    assert len(store.slide_captures(session_number)) == before_count
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["capture_path"] is not None and Path(str(row["capture_path"])).is_file()
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


def test_retry_is_a_no_op_when_the_row_is_not_in_error(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "complete"

    accepted = store.retry_hybrid_slide(session_number, capture_id)

    assert accepted is False
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "complete"


# ---------------------------------------------------------------------------
# 2. Retry that fails again records ERROR and does not stop the worker.
# ---------------------------------------------------------------------------


class _SelectivelyFailingSlidePreprocessor:
    """Deterministically fails ONLY for the fixed pixel `value`s named at
    construction time (mirrors `_FixedSlidePreprocessor`'s fully-filled-mask
    shape for every other value), so a test can force one specific slide
    capture into a REAL, reproducible preparation failure while a later,
    different slide still prepares normally."""

    def __init__(self, failing_values):
        self._failing_values = set(failing_values)

    def __call__(self, image: np.ndarray) -> PreparedSpecimen:
        value = int(image[0, 0, 0])
        if value in self._failing_values:
            raise RuntimeError("simulated persistent slide preparation failure")
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )


def test_retry_that_fails_again_records_error_and_a_later_job_still_completes(tmp_path):
    store = _make_store(
        tmp_path,
        slide_preprocessor=_SelectivelyFailingSlidePreprocessor(failing_values=(50,)),
    )
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "error"

    accepted = store.retry_hybrid_slide(session_number, capture_id)
    assert accepted is True
    store.wait_for_jobs()

    # Fails again -- still a durable ERROR, never an exception surfacing
    # here, and never stuck mid-flight.
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "error"

    # The single-worker executor is still alive: a SUBSEQUENT, genuinely
    # different slide (a pixel value the preprocessor does not fail on)
    # still reaches a real verdict.
    second_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide2.png", value=180),
        captured_at=STARTED_AT + timedelta(seconds=1),
        result=_valid_slide_result("22222222"), duration_ms=5.0,
    )
    store.wait_for_jobs()

    assert (
        store.get_slide_capture(session_number, second_capture_id)["job_state"]
        == "complete"
    )
    assert store.get_set(session_number, "22222222")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 3. Matching-identity recapture supersedes and creates a new job.
# ---------------------------------------------------------------------------


def test_recapture_with_matching_identity_supersedes_and_creates_new_job(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(old_capture_id, "error")
    old_artifact_path = Path(
        str(store.get_slide_capture(session_number, old_capture_id)["capture_path"])
    )
    assert old_artifact_path.is_file()

    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=_valid_slide_result("11111111"),  # SAME block claim
        duration_ms=5.0,
    )
    assert outcome.accepted is True
    assert outcome.new_capture_id is not None
    store.wait_for_jobs()

    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "superseded"
    assert old_artifact_path.is_file()  # artifacts stay on disk, never deleted

    new_row = store.get_slide_capture(session_number, outcome.new_capture_id)
    assert new_row["job_state"] == "complete"
    assert new_row["work_order_id"] == work_order_id
    assert new_row["priority"] == 0
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 4. Mismatching-identity recapture: no supersession, no replacement.
#    Non-vacuous: fails if the identity check is dropped.
# ---------------------------------------------------------------------------


def test_recapture_with_mismatching_identity_does_not_supersede(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(old_capture_id, "error")
    before_rows = len(store.slide_captures(session_number))

    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=_valid_slide_result("22222222"),  # DIFFERENT block -> mismatch
        duration_ms=5.0,
    )

    assert outcome.accepted is False
    assert outcome.new_capture_id is None
    assert len(store.slide_captures(session_number)) == before_rows

    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "error"  # untouched, never superseded
    # The original Results row for the claimed block is never replaced.
    assert store.get_set(session_number, "11111111")["verdict"] is None


def test_recapture_rejects_a_failed_decode_without_a_block_id(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(old_capture_id, "error")

    from slide.qr import SlideQRResult

    unreadable = SlideQRResult(
        success=False, reason="no code detected",
        raw_payload=None, format=None, block_id=None, slide_num=None, stain=None,
        work_order=None, email=None, genotype=None, engine=None, preprocessing=None,
    )

    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=unreadable, duration_ms=5.0,
    )

    assert outcome.accepted is False
    assert store.get_slide_capture(session_number, old_capture_id)["job_state"] == "error"


# ---------------------------------------------------------------------------
# 5. Recapture never interrupts a currently active job.
# ---------------------------------------------------------------------------


def test_recapture_does_not_interrupt_the_currently_active_job(tmp_path):
    entered = Event()
    release = Event()

    def blocking_score_cache_builder(specimen: PreparedSpecimen):
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the active job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=blocking_score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(old_capture_id, "error")

    active_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_active.png", value=100),
        captured_at=STARTED_AT + timedelta(seconds=1),
        result=_valid_slide_result("22222222"), duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "active job never started"

    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=150),
        captured_at=STARTED_AT + timedelta(seconds=2),
        result=_valid_slide_result("11111111"), duration_ms=5.0,
    )
    assert outcome.accepted is True
    assert outcome.new_capture_id is not None

    # The active job is untouched: a single-worker executor has no
    # preemption, and recapture never attempts one -- its row has not moved
    # past 'scoring' just because a recapture was accepted concurrently.
    active_row = store.get_slide_capture(session_number, active_capture_id)
    assert active_row["job_state"] == "scoring"

    release.set()
    store.wait_for_jobs()

    assert (
        store.get_slide_capture(session_number, active_capture_id)["job_state"]
        == "complete"
    )
    new_row = store.get_slide_capture(session_number, outcome.new_capture_id)
    assert new_row["job_state"] == "complete"
    # #255's durable scheduling column: a recapture always sorts ahead of
    # the ordinary FIFO queue.
    assert new_row["priority"] == 0


# ---------------------------------------------------------------------------
# 6. A late result from the superseded job must not overwrite the active
#    result (proves #255's CAS covers this new supersession path).
# ---------------------------------------------------------------------------


def test_late_result_from_a_recapture_superseded_job_does_not_overwrite_active_result(
    tmp_path,
):
    entered = Event()
    release = Event()

    def blocking_score_cache_builder(specimen: PreparedSpecimen):
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the retried job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=blocking_score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(capture_id, "error")

    # A retry brings it back to life -- genuinely mid-flight ('scoring',
    # blocked inside the scorer) when the recapture below is accepted.
    assert store.retry_hybrid_slide(session_number, capture_id) is True
    assert entered.wait(timeout=5), "retried job never started"
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "scoring"

    outcome = store.recapture_hybrid_slide(
        session_number, capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=_valid_slide_result("11111111"), duration_ms=5.0,
    )
    assert outcome.accepted is True
    assert outcome.new_capture_id is not None
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "superseded"

    # Let the STALE, still-scoring retry finish; its late write must not
    # clobber the row it no longer owns, and must not raise.
    release.set()
    store.wait_for_jobs()

    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "superseded"
    new_row = store.get_slide_capture(session_number, outcome.new_capture_id)
    assert new_row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 7. BLOCKER regression: a recapture landing while the superseded job is
#    still inside `_prepare_slide_for_artifacts` (the 'preparing' window,
#    BEFORE the 'scoring' write) must not let that stale job resurrect
#    itself and win `sets.verdict`; the recapture's own job (B) must still
#    reach a real verdict and `job_state='complete'`. The existing test
#    above (#6) blocks inside `score_cache_builder`, which runs AFTER the
#    'scoring' write -- structurally unable to reach the 'preparing' window
#    the entry-transition CAS bug lives in. This test blocks one step
#    earlier, inside the slide preprocessor `_prepare_slide_for_artifacts`
#    itself calls, so the recapture lands while the row is still
#    'preparing'.
# ---------------------------------------------------------------------------


def test_recapture_during_preparing_window_does_not_resurrect_stale_job(tmp_path):
    entered = Event()
    release = Event()

    def blocking_slide_preprocessor(image):
        entered.set()
        assert release.wait(timeout=5), "test did not release the blocked job"
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )

    store = _make_store(tmp_path, slide_preprocessor=blocking_slide_preprocessor)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "job A never reached the preparing window"
    # Confirm this test really lands in the 'preparing' window the bug
    # lives in, not 'scoring' (which #6 above already covers).
    assert store.get_slide_capture(session_number, old_capture_id)["job_state"] == "preparing"

    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=_valid_slide_result("11111111"),  # SAME block claim
        duration_ms=5.0,
    )
    assert outcome.accepted is True
    assert outcome.new_capture_id is not None
    assert (
        store.get_slide_capture(session_number, old_capture_id)["job_state"]
        == "superseded"
    )

    # Release the stale job A: with the bug, it resurrects its own row from
    # 'superseded' back to 'scoring' and goes on to win `sets.verdict`. With
    # the fix, its entry CAS into 'scoring' fails against the now-superseded
    # row, and it abandons without writing anything.
    release.set()
    store.wait_for_jobs()

    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "superseded"  # never resurrected

    new_row = store.get_slide_capture(session_number, outcome.new_capture_id)
    assert new_row["job_state"] == "complete"  # job B is not stranded

    set_row = store.get_set(session_number, "11111111")
    assert set_row["verdict"] in ("PASS", "REVIEW")
    # The strongest proof the stale job did NOT win: the durable verdict's
    # own `slide_capture_id` points at B (the recapture), never at A (the
    # superseded job).
    assert set_row["slide_capture_id"] == outcome.new_capture_id


# ---------------------------------------------------------------------------
# 8. HIGH fix: a superseded row must never leak into `list_hybrid_results`,
#    and the string "superseded" must never appear in any returned value.
# ---------------------------------------------------------------------------


def test_superseded_row_does_not_leak_into_list_hybrid_results(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()
    assert store.get_slide_capture(session_number, old_capture_id)["job_state"] == "complete"

    # `recapture_hybrid_slide` deliberately permits superseding an
    # already-`complete` row (see its own docstring) -- exactly the
    # scenario that used to leave two rows for the same block.
    outcome = store.recapture_hybrid_slide(
        session_number, old_capture_id,
        _write_slide_png(tmp_path / "slide_new.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert outcome.accepted is True
    store.wait_for_jobs()
    assert (
        store.get_slide_capture(session_number, old_capture_id)["job_state"]
        == "superseded"
    )

    results = store.list_hybrid_results(session_number)

    capture_ids = {row["capture_id"] for row in results}
    assert old_capture_id not in capture_ids
    assert outcome.new_capture_id in capture_ids
    # No returned value anywhere carries the internal lifecycle string --
    # `code/kiosk/results_table.py`'s documented invariant.
    for row in results:
        for value in row.values():
            assert "superseded" not in str(value)


# ---------------------------------------------------------------------------
# Control A: the CAS fix must not disturb the ordinary, un-superseded path --
# a normal job still transitions 'preparing' -> 'scoring' -> 'complete' and
# writes its verdict.
# ---------------------------------------------------------------------------


def test_ordinary_job_still_transitions_preparing_scoring_complete(tmp_path):
    entered = Event()
    release = Event()

    def blocking_slide_preprocessor(image):
        entered.set()
        assert release.wait(timeout=5), "test did not release the job"
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )

    store = _make_store(tmp_path, slide_preprocessor=blocking_slide_preprocessor)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "job never reached the preparing window"
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "preparing"

    release.set()
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# Control B: retry still works after the CAS change -- retry re-enters the
# worker via the same 'queued' state a fresh enqueue leaves behind; if the
# CAS's expected-state set were too narrow it would silently abandon every
# retry instead of completing it.
# ---------------------------------------------------------------------------


def test_retry_still_completes_after_entry_transition_cas_fix(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(capture_id, "error")

    accepted = store.retry_hybrid_slide(session_number, capture_id)
    assert accepted is True
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 9. Idempotency: a replayed retry / replayed recapture each produce exactly
#    one job.
# ---------------------------------------------------------------------------


def test_replayed_retry_produces_exactly_one_job(tmp_path):
    calls = {"n": 0}

    def counting_score_cache_builder(specimen: PreparedSpecimen):
        if specimen.role == "slide":
            calls["n"] += 1
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=counting_score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(capture_id, "error")
    assert calls["n"] == 0

    first = store.retry_hybrid_slide(session_number, capture_id, request_id="RETRY-1")
    store.wait_for_jobs()
    assert calls["n"] == 1
    second = store.retry_hybrid_slide(session_number, capture_id, request_id="RETRY-1")
    store.wait_for_jobs()

    assert first is True and second is True
    assert calls["n"] == 1  # the replay did not resubmit a second job
    assert len(store.slide_captures(session_number)) == 1


def test_replayed_recapture_produces_exactly_one_new_job(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_old.png", value=50),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(old_capture_id, "error")
    new_slide_path = _write_slide_png(tmp_path / "slide_new.png", value=90)
    result = _valid_slide_result("11111111")

    first = store.recapture_hybrid_slide(
        session_number, old_capture_id, new_slide_path,
        captured_at=STARTED_AT + timedelta(seconds=5), result=result,
        duration_ms=5.0, request_id="RECAP-1",
    )
    store.wait_for_jobs()
    second = store.recapture_hybrid_slide(
        session_number, old_capture_id, new_slide_path,
        captured_at=STARTED_AT + timedelta(seconds=5), result=result,
        duration_ms=5.0, request_id="RECAP-1",
    )
    store.wait_for_jobs()

    assert first.new_capture_id == second.new_capture_id
    assert first.accepted is True and second.accepted is True
    # The original row plus exactly ONE new recapture row -- never two.
    assert len(store.slide_captures(session_number)) == 2


# ---------------------------------------------------------------------------
# 10. Non-vacuous control: normal-mode job recovery/scoring is unaffected --
#     retry/recapture are Hybrid-only store methods with no NORMAL-mode
#     analog, so calling them against a NORMAL slide is rejected outright
#     rather than silently doing something.
# ---------------------------------------------------------------------------


def test_retry_is_a_no_op_for_a_normal_mode_capture_with_no_job_state(tmp_path):
    from tests.test_hybrid_slide_queue import _capture_block, _drain

    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="normal")
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)
    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )

    # A NORMAL-mode capture never gets a job_state at all (it is scored
    # synchronously, not via the Hybrid queue) -- retry must be a harmless
    # no-op, never a crash or a silent Hybrid-shaped resubmission.
    assert store.get_slide_capture(session.number, capture_id)["job_state"] is None
    assert store.retry_hybrid_slide(session.number, capture_id) is False
