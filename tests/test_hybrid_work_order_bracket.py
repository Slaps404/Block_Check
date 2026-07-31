"""TDD coverage for #269: Hybrid gets a real work-order bracket, and the
comparison-scope isolation invariant that is the actual product requirement
behind it -- "a slide captured under work order A is ranked/scored/verdicted
against ONLY work order A's blocks."

Three things this file exists to prove, each structurally, not just by
assertion:

1. THE ISOLATION TEST: two work orders in one Hybrid session, with
   deliberately confusable (identical-fingerprint) blocks in the second work
   order, never leak into the first work order's frozen pool -- the only
   "candidate set"/scoring-input structure that exists at this slice
   (#251/#252 build the actual per-slide queue/reranking on top of it).
2. Two work orders in one session freeze two fully independent pools with
   zero shared state -- proven at the SQL level too (two `hybrid_pools`
   rows in the same session, impossible under the pre-#269 schema).
3. A Hybrid work order that fails to freeze (fewer than two usable blocks)
   never runs the full N-by-N scoring path, on the live `finish_work_order`
   path OR after a simulated processing-computer restart.

Drives everything through `ProcessingStore` with synthetic captures, no
camera, no real segmentation -- mirrors `tests/test_hybrid_pool_freeze.py`'s
testing decision.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from session.workflow import ProcessingStore
from slide.qr import DecodeCandidate, select_slide_identity
from verify.invariant_descriptors import DescriptorValue
from verify.scorer import LockedScoreCache, _ComponentFeatures


STARTED_AT = datetime(2026, 7, 27, 10, 0, 0, tzinfo=timezone.utc)
_DESCRIPTOR_NAMES = ("fake_descriptor_v1",)


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """The real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do not
    (mirrors tests/test_hybrid_pool_freeze.py's fixture of the same name)."""
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


class _FixedMaskPreprocessor:
    """Every block prepares to the SAME fully-filled mask regardless of
    pixel content."""

    def __call__(self, capture_path: Path):
        return np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


class _IdenticalFingerprintBuilder:
    """Every block gets the IDENTICAL fingerprint vector -- the maximally
    confusable case the isolation test asks for: if isolation only held
    because descriptor VALUES happened to differ between work orders, this
    would expose it. Only distinct block_ids ever separate them (`sets`
    uniquely keys (session_number, block_id), so two work orders in one
    session cannot literally share a block_id)."""

    def __call__(self, mask):
        return {
            "fake_descriptor_v1": DescriptorValue(
                vector=np.array([1.0, 2.0, 3.0]), construction_ns=1
            )
        }


def _score_cache_builder(specimen):
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    kwargs.setdefault("preprocessor", _FixedMaskPreprocessor())
    kwargs.setdefault("fingerprint_builder", _IdenticalFingerprintBuilder())
    kwargs.setdefault("score_cache_builder", _score_cache_builder)
    return ProcessingStore(tmp_path / "processing", **kwargs)


def _capture_png(value: int) -> bytes:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    return png.tobytes()


def _capture_block(
    store: ProcessingStore, session_number: int, block_id: str, value: int = 0
) -> None:
    assert store.scan_block(session_number, block_id).accepted
    body = _capture_png(value)
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


def _drain(store: ProcessingStore, session_number: int) -> None:
    store.wait_for_jobs()
    store.begin_block_drain(session_number)


def _valid_slide_result(block_id: str):
    return select_slide_identity((
        DecodeCandidate("zxing", "QRCode", "raw", f"12080_{block_id}_01_HE"),
    ))


def _write_slide_png(path: Path, value: int = 200) -> Path:
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


# ---------------------------------------------------------------------------
# THE ISOLATION TEST
# ---------------------------------------------------------------------------


def test_isolation_second_work_orders_confusable_blocks_never_appear_in_first_scoring(
    tmp_path,
):
    """#269 headline requirement. Work order A freezes a pool and captures a
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

    # (3) SQL level: hybrid_pools now holds TWO rows in the SAME session,
    # one per work_order_id -- structurally impossible under the pre-#269
    # session_number-PRIMARY-KEY schema (a second INSERT would have either
    # raised sqlite3.IntegrityError or been silently told "already frozen"
    # and handed back work order A's pool).
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
    """The exact case the "infer from hybrid_pools existence" approach gets
    wrong (see the issue's design-gap note): fewer than 2 usable blocks means
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


def test_recover_work_orders_does_not_resume_scoring_for_a_hybrid_work_order(tmp_path):
    """Ordinary, successfully-frozen Hybrid case: finish_work_order leaves it
    'finalized' (not 'scoring'), and a restart must not resume it into
    scoring either -- distinct from the "failed to freeze" case above, which
    also has no hybrid_pools row."""
    root = tmp_path / "processing"
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
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


def test_hybrid_shadow_work_order_also_never_runs_nxn_on_finish(tmp_path):
    """The gate excludes BOTH hybrid modes, not just 'hybrid'."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid_shadow")
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "77777777", 7)
    _capture_block(store, session.number, "88888888", 8)
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True

    store.finish_work_order(session.number, start_job=True)
    store.wait_for_jobs()
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "finalized"


# ---------------------------------------------------------------------------
# _resolve_block_drain is work-order-scoped when a work_order_id is given
# (#269 review MEDIUM finding 4, low-risk hardening)
# ---------------------------------------------------------------------------


def test_resolve_block_drain_scopes_to_work_order_when_given_one(tmp_path):
    """`_resolve_block_drain`'s failed-row auto-accept and unresolved-count
    queries were session-scoped only, unlike the already-#269-fixed
    candidate SELECT in `freeze_hybrid_pool`. Neither reviewer could
    construct an operator-reachable failure through the live capture flow
    (both freeze/try_enter_slides call sites always resolve every block of
    the CURRENTLY open work order before a later one can even be opened), so
    this exercises the private helper directly to prove the new optional
    `work_order_id` scoping itself is correct: a stray 'failed' row planted
    under a DIFFERENT work order is invisible to a call scoped to this one,
    and is left untouched rather than being auto-accepted as unusable."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    work_order_a = store.start_work_order(session.number)
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """INSERT INTO sets(
                   session_number, block_id, work_order_id,
                   preprocessing_status, failure_reason
               ) VALUES (?, ?, ?, 'failed', ?)""",
            (session.number, "10000009", work_order_a, "synthetic failure"),
        )
    other_work_order_id = work_order_a + 1

    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        auto_accepted, unresolved = store._resolve_block_drain(
            db, session.number, work_order_id=other_work_order_id,
        )

    assert auto_accepted == []
    assert unresolved == 0
    # Work order A's own failed row is untouched -- never auto-accepted,
    # because the call above was scoped away from it.
    row = store.get_set(session.number, "10000009")
    assert row["preprocessing_status"] == "failed"

    # Scoped to its OWN work order, the same row IS resolved, exactly like
    # the unscoped (`try_enter_slides`) behavior.
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        auto_accepted, unresolved = store._resolve_block_drain(
            db, session.number, work_order_id=work_order_a,
        )
    assert len(auto_accepted) == 1
    assert auto_accepted[0][0] == "10000009"
    assert unresolved == 0
