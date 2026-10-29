"""Hybrid queue restart recovery, stale-write
protection, and enqueue idempotency.

Reuses tests/test_hybrid_slide_queue.py's synthetic harness (fixed-mask
block/slide preprocessors, an identical-fingerprint builder, a trivial
score_cache_builder) -- no camera, no network, no real timing. A "restart"
is simulated by constructing a SECOND
``ProcessingStore`` over the SAME ``tmp_path`` directory, which re-runs
``_initialize``/``_recover_jobs`` exactly as a real process restart would.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from session.preparation import PreparedSpecimen
from session.processing_store import ProcessingStore
from store_harness import (  # noqa: F401
    lightweight_qc_artifacts,
    capture_block as _capture_block,
    drain as _drain,
    freeze_hybrid_session as _freeze_hybrid_session,
    _make_store,
    score_cache_builder as _score_cache_builder,
    valid_slide_result as _valid_slide_result,
    write_slide_png as _write_slide_png,
)

STARTED_AT = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into the shared harness fixture."""


# ---------------------------------------------------------------------------
# 1-2. Stop after job commit ('queued') or during processing ('preparing'/
#      'scoring'): requeued and runs to a verdict, specifically NOT 'error'.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interrupted_state", ["queued", "preparing", "scoring"])
def test_recovery_requeues_an_unfinished_job_and_it_is_never_error(
    tmp_path, interrupted_state,
):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    assert store.get_slide_capture(session_number, capture_id)["job_state"] == "queued"
    if interrupted_state != "queued":
        # Simulate "the process restarted while this worker's Future was
        # genuinely mid-flight": the durable column carries an in-progress
        # state, but no live Future exists for it.
        store._set_slide_job_state(capture_id, interrupted_state)

    restarted = _make_store(tmp_path)
    row = restarted.get_slide_capture(session_number, capture_id)
    # Nothing about the slide failed -- the PROCESS restarted -- so recovery
    # must land here as 'queued' (or already resubmitted into 'preparing'/
    # 'scoring'/'complete'), NEVER 'error'.
    assert row["job_state"] != "error"

    restarted.wait_for_jobs()

    row = restarted.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["work_order_id"] == work_order_id
    assert restarted.get_set(session_number, "11111111")["verdict"] in (
        "PASS", "REVIEW",
    )


# ---------------------------------------------------------------------------
# 3. Stop after completion ('complete'): verdict preserved byte-for-byte,
#    scorer NOT invoked again.
# ---------------------------------------------------------------------------


def test_recovery_never_rescores_a_completed_job(tmp_path):
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
        duration_ms=5.0,
    )
    store.wait_for_jobs()
    assert calls["n"] == 1
    before = store.get_slide_capture(session_number, capture_id)
    assert before["job_state"] == "complete"

    restarted = _make_store(
        tmp_path, score_cache_builder=counting_score_cache_builder,
    )
    restarted.wait_for_jobs()

    assert calls["n"] == 1  # the scorer spy never fired again
    after = restarted.get_slide_capture(session_number, capture_id)
    assert after == before  # byte-for-byte preserved


# ---------------------------------------------------------------------------
# 4. Stop after supersession: stays superseded, never resurrected.
# ---------------------------------------------------------------------------


def test_recovery_never_resurrects_a_superseded_job(tmp_path):
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
    # Simulate supersession directly; recovery only needs to preserve it
    # once written.
    store._set_slide_job_state(capture_id, "superseded")

    restarted = _make_store(
        tmp_path, score_cache_builder=counting_score_cache_builder,
    )
    restarted.wait_for_jobs()

    assert calls["n"] == 0  # never scored at all
    row = restarted.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "superseded"
    assert restarted.get_set(session_number, "11111111")["verdict"] is None


# ---------------------------------------------------------------------------
# 5. Ordering: recapture-priority job first, then ordinary jobs FIFO.
# ---------------------------------------------------------------------------


def test_recovery_submits_priority_jobs_first_then_ordinary_fifo(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    # Two ordinary jobs, captured in a known order, plus a THIRD job
    # captured LAST but marked priority=0 -- it must still recover first.
    first_ordinary = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_a.png", value=150),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    second_ordinary = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_b.png", value=160),
        captured_at=STARTED_AT + timedelta(seconds=1),
        result=_valid_slide_result("11111111"), duration_ms=5.0, start_job=False,
    )
    priority_capture = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_c.png", value=170),
        captured_at=STARTED_AT + timedelta(seconds=2),
        result=_valid_slide_result("11111111"), duration_ms=5.0, start_job=False,
        priority=0,
    )

    submitted_order: list[str] = []
    real_submit = ProcessingStore._submit_hybrid_scoring

    def recording_submit(
        self, session, work_order_id, block_id, capture_id, slide_path, **kwargs
    ):
        # Keyword-only profiling args are forwarded unexamined; this test is
        # only about submission ORDER.
        submitted_order.append(capture_id)
        return real_submit(
            self, session, work_order_id, block_id, capture_id, slide_path, **kwargs
        )

    monkeypatch.setattr(ProcessingStore, "_submit_hybrid_scoring", recording_submit)
    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    assert submitted_order == [priority_capture, first_ordinary, second_ordinary]


# ---------------------------------------------------------------------------
# 7. Idempotency: a replayed acceptance creates exactly one job row and one
#    verdict write.
# ---------------------------------------------------------------------------


def test_replayed_acceptance_creates_exactly_one_job_and_one_verdict(tmp_path):
    calls = {"n": 0}

    def counting_score_cache_builder(specimen: PreparedSpecimen):
        if specimen.role == "slide":
            calls["n"] += 1
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=counting_score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    slide_path = _write_slide_png(tmp_path / "slide.png")
    result = _valid_slide_result("11111111")

    first_id = store.record_slide_capture(
        session_number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=5.0, request_id="ACCEPT-1",
    )
    store.wait_for_jobs()
    second_id = store.record_slide_capture(
        session_number, slide_path, captured_at=STARTED_AT, result=result,
        duration_ms=5.0, request_id="ACCEPT-1",
    )
    store.wait_for_jobs()

    assert first_id == second_id
    assert calls["n"] == 1
    assert len(store.slide_captures(session_number)) == 1
    row = store.get_slide_capture(session_number, first_id)
    assert row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 9. Non-vacuous control: NORMAL block-job recovery and NORMAL/OPEN_RETRIEVAL
#    work-order batch-scoring recovery are unaffected by Hybrid recovery.
# ---------------------------------------------------------------------------


def test_normal_mode_block_and_work_order_recovery_are_unaffected(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="normal")
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)
    store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )

    # Simulate a restart interrupting BLOCK preprocessing: the durable
    # column says 'processing' (an already-complete block's mask/qc are
    # still on disk, so re-running preprocessing is a harmless no-op here --
    # this only proves `_recover_jobs`'s block sweep still fires).
    with store._connect() as db:
        db.execute(
            "UPDATE sets SET preprocessing_status='processing' "
            "WHERE session_number=? AND block_id=?",
            (session.number, "11111111"),
        )
    # Simulate a restart interrupting the finalized->scoring work-order
    # commit: `start_job=False` leaves the row
    # `lifecycle_state='scoring'` with no live Future.
    finished_id = store.finish_work_order(session.number, start_job=False)
    assert finished_id == work_order_id
    assert (
        store.get_work_order(session.number, work_order_id)["lifecycle_state"]
        == "scoring"
    )

    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    assert (
        restarted.get_set(session.number, "11111111")["preprocessing_status"]
        == "complete"
    )
    assert (
        restarted.get_work_order(session.number, work_order_id)["lifecycle_state"]
        == "results_ready"
    )
