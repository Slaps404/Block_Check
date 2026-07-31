"""TDD coverage for #255: Hybrid queue restart recovery, stale-write
protection, and enqueue idempotency.

Reuses tests/test_hybrid_slide_queue.py's synthetic harness (fixed-mask
block/slide preprocessors, an identical-fingerprint builder, a trivial
score_cache_builder) -- no camera, no network, no real timing. A "restart"
is simulated the way the issue prescribes: constructing a SECOND
``ProcessingStore`` over the SAME ``tmp_path`` directory, which re-runs
``_initialize``/``_recover_jobs`` exactly as a real process restart would.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from threading import Event

import cv2
import numpy as np
import pytest

from session.preparation import PreparedSpecimen
from session.processing_store import ProcessingStore
from tests.test_hybrid_slide_queue import (
    _capture_block,
    _drain,
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
    not. Autouse fixtures do not cross module boundaries, so this file needs
    its own copy."""

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
# 1. Stop after job commit ('queued'): recovered and runs to a verdict.
# ---------------------------------------------------------------------------


def test_recovery_resubmits_a_queued_job_to_a_verdict(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "queued"

    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    row = restarted.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["work_order_id"] == work_order_id
    assert restarted.get_set(session_number, "11111111")["verdict"] in (
        "PASS", "REVIEW",
    )


# ---------------------------------------------------------------------------
# 2. Stop during processing ('preparing'/'scoring'): requeued and completes,
#    specifically NOT 'error'.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interrupted_state", ["preparing", "scoring"])
def test_recovery_requeues_a_mid_flight_job_and_it_is_never_error(
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
    # Simulate "the process restarted while this worker's Future was
    # genuinely mid-flight": the durable column already carries an
    # in-progress state, but no live Future exists for it -- exactly what a
    # crash between `_set_slide_job_state(capture_id, interrupted_state)`
    # and completion leaves behind.
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
    # #256 is the eventual writer of 'superseded'; simulate its effect
    # directly since this slice only needs to preserve it once written.
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
        # #258 added keyword-only `profile`/`profile_queued_ns` to the real
        # method; forwarded here unexamined since this test is only about
        # submission ORDER, not profiling.
        submitted_order.append(capture_id)
        return real_submit(
            self, session, work_order_id, block_id, capture_id, slide_path, **kwargs
        )

    monkeypatch.setattr(ProcessingStore, "_submit_hybrid_scoring", recording_submit)
    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    assert submitted_order == [priority_capture, first_ordinary, second_ordinary]


# ---------------------------------------------------------------------------
# 6. Stale write: a superseded job's completion must not overwrite the
#    active result, and must not raise.
# ---------------------------------------------------------------------------


def test_superseded_jobs_late_completion_does_not_overwrite_active_result(tmp_path):
    entered = Event()
    release = Event()

    def blocking_score_cache_builder(specimen: PreparedSpecimen):
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the scoring job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=blocking_score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "background job never started"

    # The job is genuinely mid-flight (job_state='scoring', blocked inside
    # the scorer seam). Supersede it -- the concrete race the issue
    # describes: the worker already read its row and is scoring, and the
    # row is superseded before it writes.
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "scoring"
    store._set_slide_job_state(capture_id, "superseded")

    release.set()
    store.wait_for_jobs()  # must not raise

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "superseded"  # not clobbered back to 'complete'
    assert store.get_set(session_number, "11111111")["verdict"] is None


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
# 8. Legacy-database migration: recovery works against a real pre-#255
#    `slide_captures` table (missing `priority`), never a silent no-op.
# ---------------------------------------------------------------------------


def test_legacy_database_without_priority_column_still_recovers(tmp_path):
    root = tmp_path / "processing"
    root.mkdir()
    connection = sqlite3.connect(root / "sessions.sqlite3")
    try:
        connection.execute(
            """CREATE TABLE slide_captures (
                capture_id TEXT PRIMARY KEY,
                session_number INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                capture_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                success INTEGER NOT NULL,
                reason TEXT NOT NULL,
                raw_payload TEXT,
                payload_format TEXT,
                block_id TEXT,
                slide_num TEXT,
                stain TEXT,
                work_order TEXT,
                email TEXT,
                genotype TEXT,
                engine TEXT,
                symbology TEXT,
                preprocessing TEXT,
                duration_ms REAL NOT NULL,
                attempts_json TEXT NOT NULL,
                verdict TEXT,
                claim_score REAL,
                claim_stage TEXT,
                claim_reason TEXT,
                claim_qc_path TEXT,
                claim_decided_at TEXT,
                work_order_id INTEGER,
                top_block TEXT,
                near_miss_blocks TEXT,
                job_state TEXT,
                candidate_selection_json TEXT,
                shadow_comparison_json TEXT
            )"""
        )
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert "priority" not in pre_migration_columns

    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert "priority" in post_migration_columns

    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "queued"
    assert row["priority"] is None

    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    row = restarted.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert restarted.get_set(session_number, "11111111")["verdict"] in (
        "PASS", "REVIEW",
    )


# ---------------------------------------------------------------------------
# 9. Non-vacuous control: NORMAL block-job recovery and NORMAL/OPEN_RETRIEVAL
#    work-order batch-scoring recovery are unchanged by this slice.
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
    # this is purely proving `_recover_jobs`'s pre-existing block sweep still
    # fires, not exercising new behavior).
    with store._connect() as db:
        db.execute(
            "UPDATE sets SET preprocessing_status='processing' "
            "WHERE session_number=? AND block_id=?",
            (session.number, "11111111"),
        )
    # Simulate a restart interrupting the finalized->scoring work-order
    # commit, exactly like `test_finish_work_order_without_request_id_
    # preserves_todays_behavior` -- `start_job=False` leaves the row
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
