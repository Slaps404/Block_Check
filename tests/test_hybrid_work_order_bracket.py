"""Hybrid's work-order bracket, and the
comparison-scope isolation invariant that is the actual product requirement
behind it -- "a slide captured under work order A is ranked/scored/verdicted
against ONLY work order A's blocks."

Three things this file exists to prove, each structurally, not just by
assertion:

1. THE ISOLATION TEST: two work orders in one Hybrid session, with
   deliberately confusable (identical-fingerprint) blocks in the second work
   order, never leak into the first work order's frozen pool (the
   scoring-input structure the per-slide queue builds on).
2. Two work orders in one session freeze two fully independent pools with
   zero shared state -- proven at the SQL level too (two `hybrid_pools`
   rows in the same session).
3. A Hybrid work order that fails to freeze (fewer than two usable blocks)
   never runs the full N-by-N scoring path, on the live `finish_work_order`
   path OR after a simulated processing-computer restart.

Drives everything through `ProcessingStore` with synthetic captures, no
camera, no real segmentation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from session.workflow import ProcessingStore
from store_harness import (  # noqa: F401
    lightweight_qc_artifacts,
    capture_block as _capture_block,
    DESCRIPTOR_NAMES as _DESCRIPTOR_NAMES,
    FixedMaskPreprocessor as _FixedMaskPreprocessor,
    IdenticalFingerprintBuilder as _IdenticalFingerprintBuilder,
    capture_png as _capture_png,
    drain as _drain,
    score_cache_builder as _score_cache_builder,
    valid_slide_result as _valid_slide_result,
    write_slide_png as _write_slide_png,
)


STARTED_AT = datetime(2026, 7, 27, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into the shared harness fixture."""


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    kwargs.setdefault("preprocessor", _FixedMaskPreprocessor())
    kwargs.setdefault("fingerprint_builder", _IdenticalFingerprintBuilder())
    kwargs.setdefault("score_cache_builder", _score_cache_builder)
    return ProcessingStore(tmp_path / "processing", **kwargs)


# ---------------------------------------------------------------------------
# THE ISOLATION TEST
# ---------------------------------------------------------------------------


def test_isolation_second_work_orders_confusable_blocks_never_appear_in_first_scoring(
    tmp_path,
):
    """Headline requirement. Work order A freezes a pool and captures a
    slide claiming one of its own blocks. Work order B, in the SAME session,
    then captures blocks with IDENTICAL fingerprint values to A's (distinct
    block_ids only) and freezes its own pool. Asserts, at every level a
    candidate set could leak through, that A's blocks/slide never appear in
    B's data and vice versa."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")

    work_order_a = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _capture_block(store, session.number, "22222222")
    _drain(store, session.number)
    result_a = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_a.frozen is True

    slide_capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide_a.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert (
        store.get_slide_capture(session.number, slide_capture_id)["work_order_id"]
        == work_order_a
    )

    # Close A; open B in the SAME session with deliberately confusable
    # (identical-fingerprint) blocks under different block_ids.
    store.finish_work_order(session.number, start_job=False)
    work_order_b = store.start_work_order(session.number)
    assert work_order_b != work_order_a
    _capture_block(store, session.number, "33333333")
    _capture_block(store, session.number, "44444444")
    _drain(store, session.number)
    result_b = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_b.frozen is True

    pool_a = store.hybrid_pool(work_order_a)
    pool_b = store.hybrid_pool(work_order_b)
    assert pool_a is not None and pool_b is not None

    # (1) Block-id level: no cross-contamination in either direction.
    assert set(pool_a.block_ids) == {"11111111", "22222222"}
    assert set(pool_b.block_ids) == {"33333333", "44444444"}
    assert set(pool_a.block_ids).isdisjoint(pool_b.block_ids)

    # (2) Fingerprint/score-cache level: even though the VALUES are
    # deliberately identical between the two work orders (maximally
    # confusable), the KEY SETS remain fully separate -- no B block_id is
    # ever a key in A's fingerprints/score_caches dicts, or vice versa.
    assert set(pool_a.fingerprints) == {"11111111", "22222222"}
    assert set(pool_b.fingerprints) == {"33333333", "44444444"}
    assert set(pool_a.score_caches) == {"11111111", "22222222"}
    assert set(pool_b.score_caches) == {"33333333", "44444444"}
    for block_id in pool_a.block_ids:
        assert np.array_equal(
            pool_a.fingerprints[block_id]["fake_descriptor_v1"],
            np.array([1.0, 2.0, 3.0]),
        )

    # (3) SQL level: hybrid_pools holds TWO rows in the SAME session, one
    # per work_order_id (a session-keyed schema would instead raise
    # sqlite3.IntegrityError or silently hand back work order A's pool).
    with store._connect() as db:
        rows = db.execute(
            "SELECT work_order_id FROM hybrid_pools WHERE session_number=? "
            "ORDER BY work_order_id", (session.number,),
        ).fetchall()
    assert [int(row["work_order_id"]) for row in rows] == sorted(
        [work_order_a, work_order_b]
    )

    # (4) The slide captured under A stays associated with A -- never
    # silently reassociated with the later work order B.
    assert (
        store.get_slide_capture(session.number, slide_capture_id)["work_order_id"]
        == work_order_a
    )


# ---------------------------------------------------------------------------
# Two work orders freeze two independent pools with zero shared state
# ---------------------------------------------------------------------------


def test_two_work_orders_in_one_session_freeze_independent_pools_no_cross_contamination(
    tmp_path,
):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")

    work_order_a = store.start_work_order(session.number)
    _capture_block(store, session.number, "10000001", 1)
    _capture_block(store, session.number, "10000002", 2)
    _drain(store, session.number)
    result_a = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_a.frozen is True
    store.finish_work_order(session.number, start_job=False)

    work_order_b = store.start_work_order(session.number)
    _capture_block(store, session.number, "20000001", 3)
    _capture_block(store, session.number, "20000002", 4)
    _capture_block(store, session.number, "20000003", 5)
    _drain(store, session.number)
    result_b = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_b.frozen is True

    pool_a = store.hybrid_pool(work_order_a)
    pool_b = store.hybrid_pool(work_order_b)
    assert set(pool_a.block_ids) == {"10000001", "10000002"}
    assert set(pool_b.block_ids) == {"20000001", "20000002", "20000003"}

    # On-disk artifacts are separate files, one pair per work order.
    manifest_a = (
        session.directory / "work_orders" / f"work_order_{work_order_a:06d}_hybrid_pool.json"
    )
    manifest_b = (
        session.directory / "work_orders" / f"work_order_{work_order_b:06d}_hybrid_pool.json"
    )
    assert manifest_a.is_file() and manifest_b.is_file()
    assert manifest_a != manifest_b

    # Both work orders show up in the session's history/results-lifecycle
    # bookkeeping distinguishably.
    wo_a_row = store.get_work_order(session.number, work_order_a)
    wo_b_row = store.get_work_order(session.number, work_order_b)
    assert wo_a_row["work_order_id"] != wo_b_row["work_order_id"]


# ---------------------------------------------------------------------------
# Hybrid never runs full N-by-N: live path AND simulated restart
# ---------------------------------------------------------------------------


def test_hybrid_work_order_that_failed_to_freeze_never_runs_nxn_on_finish_or_restart(
    tmp_path,
):
    """The case an "infer from hybrid_pools existence" approach gets
    wrong: fewer than 2 usable blocks means
    NO hybrid_pools row is ever written. finish_work_order must still never
    submit N-by-N scoring, and a fresh ProcessingStore pointed at the same
    root (simulated restart) must not resume/submit it either."""
    root = tmp_path / "processing"
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "99999999", 42)  # only 1 block
    _drain(store, session.number)

    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is False
    assert store.hybrid_pool(work_order_id) is None
    with store._connect() as db:
        pool_rows = db.execute(
            "SELECT COUNT(*) AS n FROM hybrid_pools WHERE work_order_id=?",
            (work_order_id,),
        ).fetchone()
    assert pool_rows["n"] == 0

    # Live finish path: closing the bracket must never run N-by-N.
    finished_id = store.finish_work_order(session.number, start_job=True)
    assert finished_id == work_order_id
    store.wait_for_jobs()
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "finalized"  # never 'scoring' or 'results_ready'
    assert wo["verdict_csv_path"] is None

    # Simulated restart: a brand-new ProcessingStore over the SAME durable
    # root must not resume/submit N-by-N scoring either.
    restarted = ProcessingStore(
        root, preprocessor=_FixedMaskPreprocessor(),
        fingerprint_builder=_IdenticalFingerprintBuilder(),
        score_cache_builder=_score_cache_builder,
    )
    restarted.wait_for_jobs()
    wo_after_restart = restarted.get_work_order(session.number, work_order_id)
    assert wo_after_restart["lifecycle_state"] == "finalized"
    assert wo_after_restart["verdict_csv_path"] is None


@pytest.mark.parametrize("session_mode", ["hybrid", "hybrid_shadow"])
def test_frozen_hybrid_work_order_never_resumes_scoring_on_finish_or_restart(
    tmp_path, session_mode,
):
    """Ordinary, successfully-frozen case: finish_work_order leaves it
    'finalized' (not 'scoring'), and a restart must not resume it into
    scoring either. The gate excludes BOTH hybrid modes, not just 'hybrid'."""
    root = tmp_path / "processing"
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "55555555", 5)
    _capture_block(store, session.number, "66666666", 6)
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True

    store.finish_work_order(session.number, start_job=True)
    store.wait_for_jobs()
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "finalized"

    restarted = ProcessingStore(
        root, preprocessor=_FixedMaskPreprocessor(),
        fingerprint_builder=_IdenticalFingerprintBuilder(),
        score_cache_builder=_score_cache_builder,
    )
    restarted.wait_for_jobs()
    wo_after_restart = restarted.get_work_order(session.number, work_order_id)
    assert wo_after_restart["lifecycle_state"] == "finalized"
