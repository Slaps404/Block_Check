"""`ProcessingStore.list_hybrid_results`.

`list_results_ready_work_orders` can never surface a Hybrid row --
`finish_work_order` deliberately skips `_score_work_order` (the only thing
that ever sets `lifecycle_state='results_ready'`) for HYBRID/HYBRID_SHADOW,
parking those work orders at `finalized` forever. This file drives a
separate read (`list_hybrid_results`) that surfaces in-flight and resolved
Hybrid rows across every work order in a Hybrid session, projecting the
row's internal `job_state` into a durable-safe `verdict` the caller sees:
`ERROR` / `PENDING` / the real `PASS`/`REVIEW`.

Store-only harness: synthetic captures, an injected slide preprocessor, and an injected
`score_cache_builder` -- no camera, no network, `Event`-based blocking
stands in for "the job is genuinely still running" wherever a test needs
that proof.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from session.session_mode import SessionMode
from session.workflow import HttpCaptureClient, LoopbackCaptureReceiver, PiOutbox, SessionWorkflow
from session.preparation import PreparedSpecimen
from slide.qr import select_slide_identity
from transport.remote import RemoteProcessingStore
from verify.scorer import LockedScoreCache
from store_harness import (  # noqa: F401
    lightweight_qc_artifacts,
    capture_block as _capture_block,
    freeze_hybrid_session as _freeze_hybrid_session,
    DESCRIPTOR_NAMES as _DESCRIPTOR_NAMES,
    STARTED_AT,
    capture_png as _capture_png,
    drain as _drain,
    score_cache_builder as _score_cache_builder,
    valid_slide_result as _valid_slide_result,
    write_slide_png as _write_slide_png,
    _make_store,
)


# Internal lifecycle strings that must NEVER leak past the projection.
_INTERNAL_LIFECYCLE_STRINGS = {"queued", "preparing", "scoring", "superseded"}


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into the shared harness fixture."""


class _StubWorkOrderScorer:
    """Deterministic, synchronous stand-in for the production N^2 scorer used
    by NORMAL/OPEN_RETRIEVAL's batch scoring path (mirrors
    tests/test_session_workflow.py's `StubWorkOrderScorer`)."""

    def __init__(self):
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}

    def __call__(self, block_results, slide_results):
        return {
            capture_id: dict(self.scores_by_slide.get(capture_id, {}))
            for capture_id in slide_results
        }


def _row_for(rows: tuple[dict, ...], capture_id: str) -> dict:
    matches = [row for row in rows if row["capture_id"] == capture_id]
    assert len(matches) == 1, f"expected exactly one row for {capture_id}, got {matches}"
    return matches[0]


def test_unresolved_hybrid_capture_is_not_a_results_row(tmp_path):
    """A failed QR decode is capture-recovery/audit data, not a claim result."""
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number,
        _write_slide_png(tmp_path / "unresolved-slide.png"),
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=5.0,
    )

    assert capture_id.startswith("slide_unresolved_")
    assert store.list_hybrid_results(session_number) == ()


# ---------------------------------------------------------------------------
# 1-2. A queued Hybrid slide projects PENDING, then resolves IN PLACE after
#      wait_for_jobs(): same row identity, new verdict.
# ---------------------------------------------------------------------------


def test_hybrid_slide_resolves_in_place_after_wait_for_jobs(tmp_path):
    entered = Event()
    release = Event()

    def score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the scoring job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=score_cache_builder)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "background job never started"

    before_rows = store.list_hybrid_results(session_number)
    before_row = _row_for(before_rows, capture_id)
    assert before_row["verdict"] == "PENDING"

    release.set()
    store.wait_for_jobs()

    after_rows = store.list_hybrid_results(session_number)
    after_row = _row_for(after_rows, capture_id)
    # Same row identity (capture_id), resolved value -- not a different row.
    assert after_row["capture_id"] == before_row["capture_id"]
    assert after_row["block_id"] == before_row["block_id"]
    assert after_row["verdict"] in ("PASS", "REVIEW")


def test_completed_hybrid_worker_writes_projected_jpeg_evidence(tmp_path):
    """Projected evidence paths must exist after the real worker completes."""
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"), duration_ms=5.0,
    )
    store.wait_for_jobs()

    session = store.resume_session(session_number)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:1"),
        session_mode=SessionMode(session.session_mode),
    )
    row = _row_for(tuple(workflow.results_status()["rows"]), capture_id)
    evidence = row["evidence"]
    for key in ("block_thumb", "block_display", "slide_thumb", "slide_display"):
        assert Path(evidence[key]).is_file(), key
    # Hybrid's scorer already computed the claimed pair's locked pose, so the
    # result evidence must include the matching overlay rather than asking the
    # shared writer to guess one.
    assert Path(evidence["overlay_display"]).is_file()


# ---------------------------------------------------------------------------
# 3. A failed job projects ERROR; the durable verdict column stays NULL.
# ---------------------------------------------------------------------------


def test_failed_hybrid_job_projects_error_with_null_durable_verdict(tmp_path):
    should_raise = {"active": True}

    def flaky_score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide" and should_raise["active"]:
            raise RuntimeError("synthetic scoring failure")
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=flaky_score_cache_builder)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    rows = store.list_hybrid_results(session_number)
    row = _row_for(rows, capture_id)
    assert row["verdict"] == "ERROR"
    # Both halves matter: the projection says ERROR, AND the durable column
    # backing it was never written -- an ERROR row must never masquerade as
    # a durable PASS/REVIEW.
    assert store.get_set(session_number, "11111111")["verdict"] is None


# ---------------------------------------------------------------------------
# 4. Multiple work orders in one session all project, including a row from a
#    work order that still has an outstanding job, and a still-`capturing`
#    work order (not gated on `finalized`/`results_ready`).
# ---------------------------------------------------------------------------


def test_rows_from_multiple_work_orders_all_project_including_outstanding_job(tmp_path):
    """Two work orders in one session: work order 1's slide job is still
    genuinely outstanding (stuck mid-scoring, on the single-worker
    executor); work order 2 is opened afterward and is still `capturing`
    -- never `finalized`, never `results_ready` -- yet its own row
    (resolved via the synchronous, executor-free Out-of-Pool Claim path,
    so it cannot get stuck queued behind work order 1's blocked job) still
    projects. Proves both halves of the acceptance criterion at once: an
    earlier work order's outstanding job does not hide a later work order's
    row, and a still-`capturing` work order is not silently excluded.
    """
    entered = Event()
    release = Event()

    def score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the scoring job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=score_cache_builder)
    session_number, work_order_1 = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_1 = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_a.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "background job never started"

    # `start_work_order` is idempotent while a `capturing` bracket is still
    # open (mirrors `finish_work_order`'s own SELECT-before-write guard), so
    # a genuinely SECOND work order requires finishing the first one. For
    # Hybrid this is a synchronous, immediate `capturing -> finalized` jump
    # -- `finish_work_order` deliberately skips `_score_work_order` (and
    # therefore never waits on job 1's still-blocked scoring) for
    # HYBRID/HYBRID_SHADOW.
    store.finish_work_order(session_number, start_job=True)
    assert store.get_work_order(session_number, work_order_1)["lifecycle_state"] == (
        "finalized"
    )

    # `start_work_order` resets the session phase back to `blocks` for a
    # fresh bracket -- re-enter
    # `slides` directly, WITHOUT `wait_for_jobs()` (job 1 is still stuck,
    # and there are zero blocks pending for work order 2 anyway, so nothing
    # here needs to wait on the executor).
    work_order_2 = store.start_work_order(session_number)
    assert work_order_2 != work_order_1
    store.begin_block_drain(session_number)
    assert store.try_enter_slides(session_number)

    # Work order 2's pool is deliberately never frozen -- claiming ANY
    # block_id is therefore an Out-of-Pool Claim, decided
    # synchronously inside `record_slide_capture`'s own transaction with
    # NO job submitted to the (currently occupied) single-worker executor.
    # Submitting a real job here would queue it FIFO behind work order 1's
    # still-blocked job and it would never run before this test's assertions.
    capture_2 = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_b.png", value=150),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"),
        duration_ms=5.0,
    )

    # Work order 2 was never finished -- still genuinely `capturing`.
    assert store.get_work_order(session_number, work_order_2)["lifecycle_state"] == (
        "capturing"
    )

    rows = store.list_hybrid_results(session_number)
    row_1 = _row_for(rows, capture_1)
    row_2 = _row_for(rows, capture_2)
    assert row_1["work_order_id"] == work_order_1
    assert row_1["verdict"] == "PENDING"
    assert row_2["work_order_id"] == work_order_2
    # The Out-of-Pool Claim resolves immediately (job_state stays NULL --
    # no job was ever dispatched), so this row is proof of "still-capturing,
    # not gated on finalized/results_ready", not a PENDING/queued artifact.
    assert row_2["verdict"] == "REVIEW"

    release.set()
    store.wait_for_jobs()


# ---------------------------------------------------------------------------
# 5. No internal lifecycle string ever appears anywhere in a returned dict.
# ---------------------------------------------------------------------------


def test_no_internal_lifecycle_string_appears_in_projected_rows(tmp_path):
    should_raise = {"active": True}

    def flaky_score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide" and should_raise["active"]:
            raise RuntimeError("synthetic scoring failure")
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=flaky_score_cache_builder)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    # One slide that will error (job_state='error')...
    store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_a.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    # ...and one slide that resolves normally (job_state='complete').
    should_raise["active"] = False
    store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_b.png", value=180),
        captured_at=STARTED_AT, result=_valid_slide_result("22222222"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    rows = store.list_hybrid_results(session_number)
    assert len(rows) == 2
    for row in rows:
        for value in row.values():
            if isinstance(value, str):
                assert value not in _INTERNAL_LIFECYCLE_STRINGS, (
                    f"internal lifecycle string leaked into a projected row: {row}"
                )


# ---------------------------------------------------------------------------
# 6. Non-vacuous controls: NORMAL and OPEN_RETRIEVAL sessions never surface
#    rows here, even after genuinely reaching `results_ready`.
# ---------------------------------------------------------------------------


def test_open_retrieval_queued_slide_projects_pending_before_scoring(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )

    rows = store.list_retrieval_results(session.number)

    assert len(rows) == 1
    assert rows[0]["capture_id"] == capture_id
    assert rows[0]["work_order_id"] == work_order_id
    assert rows[0]["verdict"] == "PENDING"


def test_open_retrieval_scores_complete_pool_and_resolves_live_row(tmp_path):
    entered = Event()
    release = Event()
    scored_block_ids = []

    def scorer(block_results, slide_results):
        scored_block_ids.append(tuple(block_results))
        entered.set()
        assert release.wait(timeout=5)
        return {
            capture_id: {"11111111": 0.95, "22222222": 0.20}
            for capture_id in slide_results
        }

    store = _make_store(tmp_path, work_order_scorer=scorer)
    session = store.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 1)
    _capture_block(store, session.number, "22222222", 2)
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )

    assert entered.wait(timeout=5)
    assert store.list_retrieval_results(session.number)[0]["verdict"] == "PENDING"
    release.set()
    store.wait_for_jobs()

    rows = store.list_retrieval_results(session.number)
    assert len(rows) == 1
    assert rows[0]["capture_id"] == capture_id
    assert rows[0]["verdict"] in ("PASS", "REVIEW")
    assert scored_block_ids == [("11111111", "22222222")]


def test_finish_open_retrieval_work_order_does_not_rescore_completed_slide(tmp_path):
    calls = 0

    def scorer(block_results, slide_results):
        nonlocal calls
        calls += 1
        return {
            capture_id: {block_id: 0.95 for block_id in block_results}
            for capture_id in slide_results
        }

    store = _make_store(tmp_path, work_order_scorer=scorer)
    session = store.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)
    store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()
    assert calls == 1

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    assert calls == 1
    work_order = store.get_work_order(session.number, work_order_id)
    assert work_order["lifecycle_state"] == "results_ready"
    assert Path(work_order["verdict_csv_path"]).is_file()


def test_open_retrieval_finish_waits_for_running_slide_with_parallel_workers(tmp_path):
    entered = Event()
    release = Event()
    finalizer_ran = Event()

    def scorer(block_results, slide_results):
        entered.set()
        assert release.wait(timeout=5)
        return {
            capture_id: {"11111111": 0.95}
            for capture_id in slide_results
        }

    store = _make_store(tmp_path, work_order_scorer=scorer, workers=2)
    session = store.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)
    store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5)

    real_finalizer = store._finalize_open_retrieval_work_order

    def tracked_finalizer(session_identity, finalized_work_order_id):
        try:
            return real_finalizer(session_identity, finalized_work_order_id)
        finally:
            finalizer_ran.set()

    store._finalize_open_retrieval_work_order = tracked_finalizer

    store.finish_work_order(session.number)
    assert finalizer_ran.wait(timeout=5)
    assert store.get_work_order(session.number, work_order_id)["lifecycle_state"] == (
        "finalized"
    )

    release.set()
    store.wait_for_jobs()

    work_order = store.get_work_order(session.number, work_order_id)
    assert work_order["lifecycle_state"] == "results_ready"
    assert Path(work_order["verdict_csv_path"]).is_file()


def test_open_retrieval_queued_job_resumes_after_store_restart(tmp_path):
    first = _make_store(tmp_path, recover_jobs=False)
    session = first.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    first.start_work_order(session.number)
    _capture_block(first, session.number, "11111111")
    _drain(first, session.number)
    assert first.try_enter_slides(session.number)
    capture_id = first.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )

    calls = 0

    def scorer(block_results, slide_results):
        nonlocal calls
        calls += 1
        return {
            item_capture_id: {"11111111": 0.95}
            for item_capture_id in slide_results
        }

    restarted = _make_store(tmp_path, work_order_scorer=scorer)
    restarted.wait_for_jobs()

    assert calls == 1
    assert restarted.get_slide_capture(session.number, capture_id)["job_state"] == (
        "complete"
    )


def test_open_retrieval_finalized_order_recovers_jobs_then_writes_csv(tmp_path):
    first = _make_store(tmp_path, recover_jobs=False)
    session = first.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    work_order_id = first.start_work_order(session.number)
    _capture_block(first, session.number, "11111111")
    _drain(first, session.number)
    assert first.try_enter_slides(session.number)
    first.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    first.finish_work_order(session.number, start_job=False)

    def scorer(block_results, slide_results):
        return {
            capture_id: {"11111111": 0.95}
            for capture_id in slide_results
        }

    restarted = _make_store(tmp_path, work_order_scorer=scorer)
    restarted.wait_for_jobs()

    work_order = restarted.get_work_order(session.number, work_order_id)
    assert work_order["lifecycle_state"] == "results_ready"
    assert Path(work_order["verdict_csv_path"]).is_file()


def test_normal_session_hybrid_projection_remains_empty_and_unaffected(tmp_path):
    session_mode = "normal"
    scorer = _StubWorkOrderScorer()
    store = _make_store(tmp_path, work_order_scorer=scorer)
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    scorer.scores_by_slide[capture_id] = {"11111111": 0.95}

    # list_hybrid_results is empty even before the work order finishes --
    # the mode gate, not an empty database, is what returns empty.
    assert store.list_hybrid_results(session.number) == ()

    finished_id = store.finish_work_order(session.number)
    assert finished_id == work_order_id
    store.wait_for_jobs()

    # Genuinely reached results_ready -- proves the empty list_hybrid_results
    # above was not just "nothing exists yet".
    assert store.get_work_order(session.number, work_order_id)["lifecycle_state"] == (
        "results_ready"
    )
    ready_rows = store.list_results_ready_work_orders(session.number)
    assert len(ready_rows) == 1
    assert ready_rows[0]["capture_id"] == capture_id
    assert ready_rows[0]["verdict"] in ("PASS", "REVIEW")

    # list_hybrid_results still returns empty for this NORMAL/OPEN_RETRIEVAL
    # session, now that it has real, results-ready data to (wrongly) return
    # if the mode gate were deleted.
    assert store.list_hybrid_results(session.number) == ()


# ---------------------------------------------------------------------------
# 7. RPC round-trip: the real proxy gets the same dicts back over /rpc.
# ---------------------------------------------------------------------------


def test_list_hybrid_results_round_trips_over_rpc(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        remote_rows = proxy.list_hybrid_results(session_number)

    local_rows = store.list_hybrid_results(session_number)
    assert remote_rows == local_rows
    assert len(local_rows) == 1
    assert local_rows[0]["verdict"] in ("PASS", "REVIEW")


def test_list_retrieval_results_round_trips_open_pending_row_over_rpc(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(
        started_at=STARTED_AT, session_mode=SessionMode.OPEN_RETRIEVAL.value
    )
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)
    store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        remote_rows = proxy.list_retrieval_results(session.number)

    assert remote_rows == store.list_retrieval_results(session.number)
    assert remote_rows[0]["verdict"] == "PENDING"
