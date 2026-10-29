"""An accepted, in-pool Hybrid slide leaves the
operator's critical path immediately, gets scored in the background against
its ENTIRE frozen Hybrid Candidate Pool, and its slide_captures row resolves
in place -- `job_state` durably tracking `queued -> preparing -> scoring ->
complete|error`.

A successful, in-pool Hybrid claim (`stamped_work_order_id is not None`, not
an Out-of-Pool Claim) matches neither `record_slide_capture`'s unstamped
inline-resolve branch nor its out-of-pool REVIEW branch, so it needs its own
background job. This file drives the whole store-side slice through `ProcessingStore` with synthetic
captures, an injected slide preprocessor, and an injected `score_cache_builder`
-- no camera, no network, no real timing (`Event`-based blocking stands in
for "the job is genuinely still running" wherever a test needs that proof).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

import json

import session.processing_store as processing_store
from session.processing_store import ProcessingStore
from session.preparation import PreparationFailure, PreparedResult, PreparedSpecimen
from retrieval.candidate_band import DENSE_DESCRIPTOR_WEIGHTS
from retrieval.invariant_descriptors import build_descriptor_values
from verify.scorer import LockedScoreCache
from retrieval.work_order_evaluator import evaluate_work_order as _real_evaluate_work_order
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


# The real dense-shape descriptor names (weight > 0), so a pool built
# with this recipe never hits `select_candidate_band`'s missing-descriptor
# fallback -- unlike the harness `_DESCRIPTOR_NAMES`, which deliberately
# uses a name outside the real catalog.
_DENSE_CATALOG_DESCRIPTOR_NAMES = tuple(
    name for name, weight in DENSE_DESCRIPTOR_WEIGHTS.items() if weight > 0
)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into the shared harness fixture."""


# ---------------------------------------------------------------------------
# 1. Acceptance commits durably; the operator is not blocked on scoring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session_mode", ["hybrid", "hybrid_shadow"])
def test_accepted_in_pool_claim_does_not_block_capture_on_scoring(tmp_path, session_mode):
    entered = Event()
    release = Event()

    def score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the scoring job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, session_mode, ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )

    # The call above already returned -- proof the durable commit (bytes,
    # identity, pool association, job row) did not wait on the job the
    # executor is now running in the background, currently stuck inside
    # `score_cache_builder` until released below.
    assert entered.wait(timeout=5), "background job never started"
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] in ("queued", "preparing", "scoring")
    assert row["work_order_id"] == work_order_id
    # The claimed block is a real `sets` row (in-pool), so the durable
    # verdict home is `sets.verdict` (mirrors `_score_work_order`'s own
    # `_finalize_claim` usage) -- confirm no verdict has landed yet.
    assert store.get_set(session_number, "11111111")["verdict"] is None

    release.set()
    store.wait_for_jobs()


# ---------------------------------------------------------------------------
# 2/3. Happy path: after wait_for_jobs(), a verdict lands and job completes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session_mode", ["hybrid", "hybrid_shadow"])
def test_accepted_in_pool_claim_resolves_after_wait_for_jobs(tmp_path, session_mode):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, session_mode, ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["work_order_id"] == work_order_id
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 4. A worker exception is durable and never kills the single-worker executor.
# ---------------------------------------------------------------------------


def test_worker_exception_yields_error_state_and_does_not_stop_the_worker(tmp_path):
    should_raise = {"active": False}

    def flaky_score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide" and should_raise["active"]:
            raise RuntimeError("synthetic scoring failure")
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=flaky_score_cache_builder)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    should_raise["active"] = True
    first_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_a.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    first_row = store.get_slide_capture(session_number, first_capture_id)
    assert first_row["job_state"] == "error"
    assert first_row["verdict"] is None
    assert (
        store._session_identity(session_number).directory
        / "slide_artifacts"
        / f"{first_capture_id}_failed_qc.png"
    ).is_file()

    # The single-worker executor must still be alive: a SECOND job, on the
    # SAME store, submitted after the first one failed, still completes.
    should_raise["active"] = False
    second_capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide_b.png", value=180),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    second_row = store.get_slide_capture(session_number, second_capture_id)
    assert second_row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 5. Expected outcomes (gate failure / identity mismatch) are REVIEW, not
#    ERROR -- the whole point of the ERROR row state's "system failure only".
# ---------------------------------------------------------------------------


def test_quality_gate_failure_yields_review_not_error(tmp_path):
    def failing_slide_preprocessor(image: np.ndarray) -> PreparedResult:
        return PreparationFailure(role="slide", reason="no tissue found")

    store = _make_store(tmp_path, slide_preprocessor=failing_slide_preprocessor)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["top_score"] is None
    assert row["runner_up_score"] is None
    assert row["match_margin"] is None
    assert not [
        pair for pair in store.list_matching_pairs(session_number)
        if pair["slide_capture_id"] == capture_id
    ]
    assert store.get_set(session_number, "11111111")["decision_reason"] == "Preparation failed."
    assert store.get_set(session_number, "11111111")["verdict"] == "REVIEW"


class _ValueEncodingBlockPreprocessor:
    """Encodes each block capture's fill value into its mask's fill
    FRACTION (rather than `_FixedMaskPreprocessor`'s always-identical fully-
    filled mask), so two blocks captured with different pixel values end up
    with genuinely different, order-surviving mask content -- needed because
    `hybrid_pool()` decompresses fresh array objects on every read, so a
    discriminator keyed on object identity would not survive the archive
    round-trip between freeze and the worker's own `self.hybrid_pool()` call.
    """

    def __call__(self, capture_path: Path):
        image = cv2.imread(str(capture_path))
        assert image is not None
        fill_value = int(image[0, 0, 0])
        rows_filled = max(1, min(8, round(fill_value / 255 * 8)))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:rows_filled, :] = 255
        return mask, {"role": "block", "roi_ok": True}


def test_identity_mismatch_yields_review_not_error(tmp_path, monkeypatch):
    """Force the pool's OTHER block to always outscore the claimed block --
    a real "claim disagreement", not a gate failure -- and confirm it still
    resolves to REVIEW with `job_state='complete'`, never `error`."""
    store = _make_store(tmp_path, preprocessor=_ValueEncodingBlockPreprocessor())
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", value=30)   # few filled rows
    _capture_block(store, session.number, "22222222", value=220)  # many filled rows
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True

    # Fake the module-level scorer (imported directly into
    # `session.processing_store`) to discriminate purely on the pool block's OWN mask mean
    # -- never the slide -- so the higher-fill block deterministically
    # outscores the claimed (lower-fill) one.
    def fake_score_routed_caches(block, slide, **kwargs):
        return type(
            "FakeScoreResult", (), {
                "score": float(block.normalized_mask.mean()),
                "selected_metric": "synthetic_test_metric",
                "router_size_signal": 1.0,
                "block_occupied_fraction": 1.0,
                "slide_occupied_fraction": 1.0,
                "best_angle": 0.0,
                "best_flip": False,
                "align_soft_iou": 1.0,
                "mask_iou": 1.0,
            }
        )()

    monkeypatch.setattr(
        processing_store, "score_routed_caches", fake_score_routed_caches
    )

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session.number, capture_id)
    assert row["job_state"] == "complete"
    assert row["top_block"] == "22222222"
    assert store.get_set(session.number, "11111111")["verdict"] == "REVIEW"


# ---------------------------------------------------------------------------
# 6. Score audit: the entire frozen pool is scored and persisted when
#    selection falls back (this fixture's descriptor is outside the catalog).
# ---------------------------------------------------------------------------


def test_hybrid_score_audit_persists_each_scored_pair_and_margin_summary(tmp_path):
    """A completed Hybrid job must retain the actual candidate ranking.

    The fixed-mask fixture gives every pool candidate the same real score, so
    lexical block-id tie-breaking makes rank/order deterministic and the
    stored zero margin is directly checkable.
    """
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222", "33333333"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["top_block"] == "11111111"
    assert row["top_score"] == pytest.approx(1.0)
    assert row["runner_up_score"] == pytest.approx(1.0)
    assert row["match_margin"] == pytest.approx(0.0)

    pairs = {
        row["block_id"]: row
        for row in store.list_matching_pairs(session_number)
        if row["slide_capture_id"] == capture_id
    }
    assert set(pairs) == {"11111111", "22222222", "33333333"}
    assert [pairs[block_id]["rank_for_block"] for block_id in sorted(pairs)] == [1, 2, 3]
    assert {row["classical_score"] for row in pairs.values()} == {1.0}
    assert {row["metric"] for row in pairs.values()} == {"mask_iou"}
    assert pairs["11111111"]["pair_source"] == "true_pair"
    assert pairs["11111111"]["is_match"] == 1
    assert {pairs[block_id]["is_match"] for block_id in ("22222222", "33333333")} == {0}


# ---------------------------------------------------------------------------
# 7. Legacy-DB migration: every later slide_captures column is added
#    additively to the OLDEST real schema (pre-job_state), never a silent
#    no-op that crashes the first write with "no such column".
# ---------------------------------------------------------------------------

_OLDEST_SLIDE_CAPTURES_SCHEMA = """CREATE TABLE slide_captures (
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
    near_miss_blocks TEXT
)"""

_MIGRATED_SLIDE_CAPTURE_COLUMNS = {
    "job_state",
    "top_score", "runner_up_score", "match_margin",  # score audit
    "candidate_selection_json",
    "shadow_comparison_json",
    "priority",
    "profile_enabled", "profile_queued_ns", "profile_current_stage",
    "profile_stage_ms_json", "profile_total_ms", "profile_shadow",
}


def test_oldest_slide_captures_schema_is_migrated_and_recovers_a_job(tmp_path):
    root = tmp_path / "processing"
    root.mkdir()
    connection = sqlite3.connect(root / "sessions.sqlite3")
    try:
        connection.execute(_OLDEST_SLIDE_CAPTURES_SCHEMA)
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert not (_MIGRATED_SLIDE_CAPTURE_COLUMNS & pre_migration_columns)

    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert _MIGRATED_SLIDE_CAPTURE_COLUMNS <= post_migration_columns

    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid_shadow", ("11111111", "22222222"),
    )
    # `start_job=False` keeps this deterministic: with no Future submitted,
    # `job_state == 'queued'` proves the migrated columns hold what
    # `record_slide_capture` committed.
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False, profile=True,
    )
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "queued"
    assert row["priority"] is None

    # A restart's `_recover_jobs` (which orders by `priority`) carries the
    # queued job through to a real verdict on the migrated table.
    restarted = _make_store(tmp_path)
    restarted.wait_for_jobs()

    row = restarted.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["work_order_id"] == work_order_id
    assert restarted.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")
    # Every migrated column round-trips a real write.
    assert row["top_score"] is not None
    assert row["candidate_selection_json"] is not None
    assert json.loads(row["shadow_comparison_json"])
    assert row["profile_enabled"] == 1
    assert row["profile_total_ms"] is not None
    assert json.loads(str(row["profile_stage_ms_json"]))


# ---------------------------------------------------------------------------
# 8. Non-vacuous negative controls: NORMAL / OPEN_RETRIEVAL are unaffected.
# ---------------------------------------------------------------------------


def test_normal_mode_slide_leaves_job_state_null(tmp_path):
    """NORMAL mode, no work order bracket (`stamped_work_order_id` stays
    None) -- the immediate `resolve_claim` path: no job, no job_state."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="normal")
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"),
        duration_ms=5.0,
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["job_state"] is None
    assert row["verdict"] == "REVIEW"  # immediate identity-lookup REVIEW, unaffected
    # NORMAL never runs `_score_hybrid_slide`/`select_hybrid_candidates`
    # at all, so the audit column stays untouched here too.
    assert row["candidate_selection_json"] is None


def test_open_retrieval_mode_slide_with_open_work_order_is_durably_queued(tmp_path):
    """Open Retrieval shares the work-order slide queue lifecycle with Hybrid.

    ``start_job=False`` models the durable commit before disposable executor
    submission, so this test observes the public acceptance seam without
    depending on worker timing.
    """
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="open_retrieval")
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _drain(store, session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["job_state"] == "queued"
    assert row["verdict"] is None
    assert row["work_order_id"] == work_order_id
    # Open's strategy scores the complete pool and never records Hybrid's
    # heuristic-selection audit.
    assert row["candidate_selection_json"] is None


# ---------------------------------------------------------------------------
# 9. Finish Slides and a new work order never wait on outstanding jobs.
# ---------------------------------------------------------------------------


def test_finish_work_order_and_new_work_order_do_not_wait_on_outstanding_hybrid_job(
    tmp_path,
):
    entered = Event()
    release = Event()

    def score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
        if specimen.role == "slide":
            entered.set()
            assert release.wait(timeout=5), "test did not release the scoring job"
        return _score_cache_builder(specimen)

    store = _make_store(tmp_path, score_cache_builder=score_cache_builder)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    assert entered.wait(timeout=5), "background job never started"

    # Finish Slides returns immediately -- it does not wait for the job
    # that is, right now, genuinely still stuck inside score_cache_builder.
    finished_id = store.finish_work_order(session_number, start_job=True)
    assert finished_id == work_order_id
    work_order_row = store.get_work_order(session_number, work_order_id)
    assert work_order_row["lifecycle_state"] == "finalized"

    # A second work order can be opened in the same session while the first
    # order's job is still outstanding.
    second_work_order_id = store.start_work_order(session_number)
    assert second_work_order_id != work_order_id

    release.set()
    store.wait_for_jobs()


# ---------------------------------------------------------------------------
# 10. Heuristic Candidate Band wiring.
# ---------------------------------------------------------------------------


def _freeze_dense_pool_with_real_descriptors(
    store: ProcessingStore, session_number: int, block_fill_values: dict[str, int],
) -> int:
    """Freeze a pool using the REAL production descriptor catalog (dense
    weight set) rather than the fake `_DESCRIPTOR_NAMES` -- so
    `select_candidate_band` never hits its missing-descriptor fallback and
    tests can prove genuine pruning, not just the safety net.
    """
    work_order_id = store.start_work_order(session_number)
    for block_id, fill_value in block_fill_values.items():
        _capture_block(store, session_number, block_id, fill_value)
    _drain(store, session_number)
    freeze_result = store.freeze_hybrid_pool(
        session_number,
        descriptor_names=_DENSE_CATALOG_DESCRIPTOR_NAMES,
        candidate_configuration={
            "architecture_kind": "individual",
            "architecture_name": "individual",
            "architecture_methods": ("global_morphology_v1",),
            "candidate_band_thresholds": {"global_morphology_v1": 0.05},
        },
    )
    assert freeze_result.frozen is True
    return work_order_id


def test_only_the_banded_subset_is_accurately_scored_and_selection_is_persisted(
    tmp_path, monkeypatch,
):
    """Prove the scorer runs for
    strictly FEWER blocks than the full pool once the real (non-fake)
    descriptor catalog lets `select_candidate_band` actually prune, and that
    the selection is durably persisted so a pruned block is distinguishable
    from one never scanned. Uses the REAL selection math (no fakes on
    `select_candidate_band` itself) -- this is proof of the actual savings.
    """
    captured_selections: list = []
    real_select = ProcessingStore.select_hybrid_candidates

    def spy_select(self, pool, claim_id, slide_cache, **kwargs):
        selection = real_select(self, pool, claim_id, slide_cache, **kwargs)
        captured_selections.append(selection)
        return selection

    monkeypatch.setattr(ProcessingStore, "select_hybrid_candidates", spy_select)

    scored_call_count = {"n": 0}
    real_score_routed_caches = processing_store.score_routed_caches

    def counting_score_routed_caches(*args, **kwargs):
        scored_call_count["n"] += 1
        return real_score_routed_caches(*args, **kwargs)

    monkeypatch.setattr(
        processing_store, "score_routed_caches", counting_score_routed_caches
    )

    store = _make_store(
        tmp_path,
        preprocessor=_ValueEncodingBlockPreprocessor(),
        fingerprint_builder=build_descriptor_values,
    )
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    # One block (`runner_up`) closely matches the fully-filled slide; six
    # "far" blocks are nearly empty and very unlike it -- enough spread, at
    # this pool size, for the dense gap+floor to genuinely exclude some of
    # them (`floor_count = max(3, ceil(0.25 * 8)) = 3` non-claim survivors
    # out of 7 non-claim candidates).
    block_fill_values = {
        "10000001": 128,  # claim
        "10000002": 255,  # runner_up -- closely matches the slide
        "10000003": 10, "10000004": 10, "10000005": 10,
        "10000006": 10, "10000007": 10, "10000008": 10,
    }
    _freeze_dense_pool_with_real_descriptors(
        store, session.number, block_fill_values,
    )

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("10000001"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    assert len(captured_selections) == 1
    selection = captured_selections[0]
    assert selection.fallback_required is False
    # The whole point: genuine pruning happened, and the scorer ran for
    # exactly the selected band + claim -- fewer than the whole 8-block pool.
    assert selection.pruned_ids  # non-empty: at least one block was pruned
    assert scored_call_count["n"] == len(selection.accurate_scoring_ids)
    assert scored_call_count["n"] < len(block_fill_values)

    # Durable audit evidence: a pruned block is explicit, not a missing key
    # indistinguishable from "never scanned in this work order".
    with store._connect() as db:
        row = db.execute(
            "SELECT candidate_selection_json FROM slide_captures WHERE capture_id=?",
            (capture_id,),
        ).fetchone()
    record = json.loads(row["candidate_selection_json"])
    assert record["claim_id"] == "10000001"
    assert set(record["pruned_ids"])  # explicitly recorded, not merely absent
    assert not (set(record["pruned_ids"]) & set(record["candidate_ids"]))
    assert record["claim_id"] not in record["candidate_ids"]
    assert record["claim_id"] not in record["pruned_ids"]
    # Every pool block is accounted for exactly once between candidate/
    # pruned/claim -- the structural guarantee that makes "pruned" and
    # "never scanned" distinguishable from the outside.
    assert (
        set(record["candidate_ids"]) | set(record["pruned_ids"]) | {record["claim_id"]}
        == set(block_fill_values)
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["job_state"] == "complete"
    assert store.get_set(session.number, "10000001")["verdict"] in ("PASS", "REVIEW")


def test_claim_is_scored_even_when_it_would_rank_dead_last(tmp_path, monkeypatch):
    """`accurate_scoring_ids` structurally always includes the claim,
    regardless of its heuristic rank. Build the claim as the block LEAST
    like the slide (it would rank dead last among the pool if it were ever
    ranked) and confirm it is still scored -- not silently dropped."""
    seen_candidate_maps: list[dict] = []
    real_evaluate = _real_evaluate_work_order

    def spy_evaluate_work_order(candidate_scores, claimed_block, **kwargs):
        seen_candidate_maps.append(dict(candidate_scores))
        return real_evaluate(candidate_scores, claimed_block, **kwargs)

    monkeypatch.setattr(
        processing_store, "evaluate_work_order", spy_evaluate_work_order
    )

    store = _make_store(
        tmp_path,
        preprocessor=_ValueEncodingBlockPreprocessor(),
        fingerprint_builder=build_descriptor_values,
    )
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")
    # The claim ("10000001") is nearly empty -- the LEAST like the
    # fully-filled slide of the whole pool; every other block closely
    # matches the slide, so the claim would be the worst possible ranked
    # candidate if it were ever ranked at all.
    block_fill_values = {
        "10000001": 10,  # claim -- least like the slide
        "10000002": 255, "10000003": 255, "10000004": 255,
        "10000005": 255, "10000006": 255, "10000007": 255,
        "10000008": 255,
    }
    _freeze_dense_pool_with_real_descriptors(store, session.number, block_fill_values)

    store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("10000001"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    assert len(seen_candidate_maps) == 1
    candidate_scores = seen_candidate_maps[0]
    assert "10000001" in candidate_scores  # scored, never silently dropped
    assert candidate_scores["10000001"] is not None


def test_selection_failure_falls_back_to_complete_pool_scoring_not_error(
    tmp_path, monkeypatch,
):
    """Blast radius: candidate selection raising must degrade to the
    permanent whole-pool fallback and still produce a real verdict -- never
    this job's `job_state='error'` outcome, which is reserved for a genuine
    scoring/IO failure."""
    def raising_select_hybrid_candidates(self, pool, claim_id, slide_cache, **kwargs):
        raise RuntimeError("synthetic #253 selection failure")

    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates", raising_select_hybrid_candidates
    )

    seen_candidate_maps: list[dict] = []
    real_evaluate = _real_evaluate_work_order

    def spy_evaluate_work_order(candidate_scores, claimed_block, **kwargs):
        seen_candidate_maps.append(dict(candidate_scores))
        return real_evaluate(candidate_scores, claimed_block, **kwargs)

    monkeypatch.setattr(
        processing_store, "evaluate_work_order", spy_evaluate_work_order
    )

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222", "33333333"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"  # never 'error' -- selection != scoring
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")
    # Fell back to the WHOLE pool, exactly like a missing-descriptor fallback.
    assert len(seen_candidate_maps) == 1
    assert set(seen_candidate_maps[0]) == {"11111111", "22222222", "33333333"}
    # The audit record names the selection failure, distinct from a real
    # (possibly fallback_required) CandidateSelection.
    with store._connect() as db:
        audit_row = db.execute(
            "SELECT candidate_selection_json FROM slide_captures WHERE capture_id=?",
            (capture_id,),
        ).fetchone()
    audit_record = json.loads(audit_row["candidate_selection_json"])
    assert "selection_error" in audit_record
