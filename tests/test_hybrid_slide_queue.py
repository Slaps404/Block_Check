"""TDD coverage for #252: an accepted, in-pool Hybrid slide leaves the
operator's critical path immediately, gets scored in the background against
its ENTIRE frozen Hybrid Candidate Pool, and its slide_captures row resolves
in place -- `job_state` durably tracking `queued -> preparing -> scoring ->
complete|error`.

This is the "hole" `record_slide_capture`'s existing dispatch fork left open:
a successful, in-pool Hybrid claim (`stamped_work_order_id is not None`, not
an Out-of-Pool Claim -- #251) matched neither the unstamped inline-resolve
branch nor the out-of-pool REVIEW branch, so nothing happened. This file
drives the whole store-side slice through `ProcessingStore` with synthetic
captures, an injected slide preprocessor, and an injected `score_cache_builder`
-- no camera, no network, no real timing (`Event`-based blocking stands in
for "the job is genuinely still running" wherever a test needs that proof).
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

import json

import session.processing_store as processing_store
from session.processing_store import ProcessingStore
from session.preparation import PreparationFailure, PreparedResult, PreparedSpecimen
from slide.qr import DecodeCandidate, select_slide_identity
from verify.candidate_band import DENSE_DESCRIPTOR_WEIGHTS
from verify.invariant_descriptors import DescriptorValue, build_descriptor_values
from verify.scorer import LockedScoreCache, _ComponentFeatures
from verify.work_order_evaluator import evaluate_work_order as _real_evaluate_work_order


# #253: the real dense-shape descriptor names (weight > 0), so a pool built
# with this recipe never hits `select_candidate_band`'s missing-descriptor
# fallback -- unlike this file's `_DESCRIPTOR_NAMES`/`_IdenticalFingerprintBuilder`
# fixture above, which deliberately uses a name outside the real catalog.
_DENSE_CATALOG_DESCRIPTOR_NAMES = tuple(
    name for name, weight in DENSE_DESCRIPTOR_WEIGHTS.items() if weight > 0
)


STARTED_AT = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
_DESCRIPTOR_NAMES = ("fake_descriptor_v1",)


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """The real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do not
    (mirrors tests/test_hybrid_out_of_pool_claim.py's fixture of the same
    name)."""
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
    pixel content (mirrors tests/test_hybrid_pool_freeze.py's fixture)."""

    def __call__(self, capture_path: Path):
        return np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


class _FixedSlidePreprocessor:
    """Every slide prepares to the SAME fully-filled specimen -- avoids
    depending on the real tissue-detecting `preprocess_slide` against tiny
    synthetic solid-color PNGs, which is not what this file tests."""

    def __call__(self, image: np.ndarray) -> PreparedResult:
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )


class _IdenticalFingerprintBuilder:
    """Every block gets the IDENTICAL fingerprint vector -- only distinct
    block_ids separate them (mirrors the #269/#251 test fixtures)."""

    def __call__(self, mask):
        return {
            "fake_descriptor_v1": DescriptorValue(
                vector=np.array([1.0, 2.0, 3.0]), construction_ns=1
            )
        }


def _score_cache_builder(specimen: PreparedSpecimen) -> LockedScoreCache:
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    kwargs.setdefault("preprocessor", _FixedMaskPreprocessor())
    kwargs.setdefault("slide_preprocessor", _FixedSlidePreprocessor())
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


def _freeze_hybrid_session(
    store: ProcessingStore, session_mode: str, block_ids: tuple[str, ...],
) -> tuple[int, int]:
    """Start a Hybrid session, capture/freeze ``block_ids`` into one pool.

    Returns (session_number, work_order_id).
    """
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    for index, block_id in enumerate(block_ids):
        _capture_block(store, session.number, block_id, index + 1)
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True
    return session.number, work_order_id


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
    # `session.processing_store`, mirrors this file's other "spy/fake the
    # seam" tests) to discriminate purely on the pool block's OWN mask mean
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
# 6. The ENTIRE frozen pool is scored when candidate selection falls back.
#
# #253 landed since this test was written: this file's fixture
# (`_IdenticalFingerprintBuilder`) deliberately uses a descriptor name
# (`fake_descriptor_v1`) absent from the real Heuristic Descriptor Catalog,
# so `select_candidate_band` always reports `fallback_required` here (a
# missing-descriptor fallback, per `code/verify/candidate_band.py`) -- this
# test now proves the PERMANENT whole-pool fallback path stays correct, not
# "pruning does not exist yet". See section 10 below for real #253 pruning
# with the production descriptor catalog.
# ---------------------------------------------------------------------------


def test_entire_frozen_pool_is_scored_no_pruning(tmp_path, monkeypatch):
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

    store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    assert len(seen_candidate_maps) == 1
    assert set(seen_candidate_maps[0]) == {"11111111", "22222222", "33333333"}


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
# 7. Legacy-DB migration: `job_state` is added additively, never a silent
#    no-op against a real pre-existing database.
# ---------------------------------------------------------------------------


def test_legacy_slide_captures_table_without_job_state_is_migrated(tmp_path):
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
                near_miss_blocks TEXT
            )"""
        )
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert "job_state" not in pre_migration_columns

    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert "job_state" in post_migration_columns
    assert {"top_score", "runner_up_score", "match_margin"} <= post_migration_columns

    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    # `start_job=False` (mirrors `receive_capture`'s own seam) keeps this
    # deterministic: with no Future ever submitted, nothing can race the
    # read below, so `job_state == 'queued'` proves the migrated column
    # is writable and holds the value `record_slide_capture` committed --
    # not a guess about whichever state the background worker reached
    # first.
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, start_job=False,
    )
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "queued"

    # Submit the deferred job ourselves (same call `record_slide_capture`
    # would have made with `start_job=True`) to prove the migrated table
    # still carries a capture through to a real verdict.
    session = store._session_identity(session_number)
    slide_path = session.directory / "slide_captures" / f"{capture_id}.png"
    store._submit_hybrid_scoring(
        session, work_order_id, "11111111", capture_id, slide_path
    )
    store.wait_for_jobs()
    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert store.get_set(session_number, "11111111")["verdict"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# 8. Non-vacuous negative controls: NORMAL / OPEN_RETRIEVAL are unaffected.
# ---------------------------------------------------------------------------


def test_normal_mode_slide_leaves_job_state_null(tmp_path):
    """NORMAL mode, no work order bracket (`stamped_work_order_id` stays
    None) -- the pre-#251/#252 immediate `resolve_claim` path. Must remain
    exactly unaffected: no job, no job_state."""
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
    # #253: NORMAL never runs `_score_hybrid_slide`/`select_hybrid_candidates`
    # at all, so the new audit column stays untouched here too.
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
# 10. #253 Heuristic Candidate Band wiring.
# ---------------------------------------------------------------------------


def _freeze_dense_pool_with_real_descriptors(
    store: ProcessingStore, session_number: int, block_fill_values: dict[str, int],
) -> int:
    """Freeze a pool using the REAL production descriptor catalog (dense
    weight set) rather than this file's fake `_DESCRIPTOR_NAMES` -- so
    `select_candidate_band` never hits its missing-descriptor fallback and
    this file can prove genuine #253 pruning, not just the safety net.
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
    """#253: the whole point of the issue -- prove the scorer runs for
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
    """#253: `accurate_scoring_ids` structurally always includes the claim,
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
    """#253 BLAST RADIUS: candidate selection raising must degrade to the
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


def test_legacy_slide_captures_table_without_candidate_selection_json_is_migrated(
    tmp_path,
):
    """#253 migration: `candidate_selection_json` is added additively to a
    real pre-#253 `slide_captures` table (already carrying every earlier
    Hybrid column, including `job_state`) -- never a silent no-op that would
    crash the first `_persist_candidate_selection` write with `sqlite3
    .OperationalError: no such column`."""
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
                job_state TEXT
            )"""
        )
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert "candidate_selection_json" not in pre_migration_columns

    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert "candidate_selection_json" in post_migration_columns

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
    # The migrated column round-trips a real write from
    # `_persist_candidate_selection` (fallback here, since this fixture's
    # fake descriptor name is outside the real catalog -- irrelevant to what
    # this test proves, which is that the column exists and is writable).
    with store._connect() as db:
        audit_row = db.execute(
            "SELECT candidate_selection_json FROM slide_captures WHERE capture_id=?",
            (capture_id,),
        ).fetchone()
    assert audit_row["candidate_selection_json"] is not None


# ---------------------------------------------------------------------------
# 8. #258: profiled Hybrid worker timing -- controlled clock, exact integers,
#    selection excludes scoring, gated strictly on `profile`/session mode,
#    and a legacy-DB migration for the new columns.
# ---------------------------------------------------------------------------


def test_profiled_hybrid_slide_records_five_exact_stage_timings(tmp_path):
    """`self._profile_clock_ns` is called exactly 6 times for one profiled,
    gate-passing slide: once at enqueue (`record_slide_capture`), then once
    each at worker start, end of preparation, end of selection, end of
    scoring, and completion (`_score_hybrid_slide`) -- see those methods'
    own docstrings. Selection's own clock read happens BEFORE
    `score_routed_caches` runs, so `heuristic_selection_ms` can never
    include `accurate_scoring_ms`."""
    ticks = iter((0, 1_000_000, 4_000_000, 6_000_000, 10_000_000, 13_000_000))
    store = _make_store(tmp_path, profile_clock_ns=lambda: next(ticks))
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, profile=True,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["work_order_id"] == work_order_id
    assert row["profile_enabled"] == 1
    assert row["profile_queued_ns"] == 0
    assert row["profile_total_ms"] == 13
    assert row["profile_shadow"] == 0
    assert json.loads(str(row["profile_stage_ms_json"])) == {
        "queue_wait": 1,
        "preparation": 3,
        "heuristic_selection": 2,
        "accurate_scoring": 4,
        "artifact_write": 3,
    }

    profile_rows = store.list_hybrid_profile_rows(session_number)
    assert len(profile_rows) == 1
    assert profile_rows[0]["capture_id"] == capture_id
    assert profile_rows[0]["block_id"] == "11111111"
    assert profile_rows[0]["queued_ns"] == 0
    assert profile_rows[0]["total_ms"] == 13
    assert profile_rows[0]["shadow"] == 0
    assert json.loads(str(profile_rows[0]["stage_ms_json"])) == {
        "queue_wait": 1,
        "preparation": 3,
        "heuristic_selection": 2,
        "accurate_scoring": 4,
        "artifact_write": 3,
    }


def test_profiled_hybrid_shadow_slide_is_tagged_in_persisted_row(tmp_path):
    ticks = iter((0, 1_000_000, 4_000_000, 6_000_000, 10_000_000, 13_000_000))
    store = _make_store(tmp_path, profile_clock_ns=lambda: next(ticks))
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", ("11111111", "22222222"),
    )

    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, profile=True,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["profile_shadow"] == 1
    assert store.list_hybrid_profile_rows(session_number)[0]["shadow"] == 1


def test_unprofiled_hybrid_slide_persists_nothing_and_list_is_empty(tmp_path):
    store = _make_store(tmp_path)
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
    assert row["profile_enabled"] == 0
    assert row["profile_queued_ns"] is None
    assert row["profile_current_stage"] is None
    assert row["profile_stage_ms_json"] is None
    assert row["profile_total_ms"] is None
    assert row["profile_shadow"] is None
    assert store.list_hybrid_profile_rows(session_number) == ()


@pytest.mark.parametrize("session_mode", ["normal", "open_retrieval"])
def test_list_hybrid_profile_rows_is_empty_for_normal_and_open_retrieval(
    tmp_path, session_mode,
):
    """Non-vacuous: a profiled-looking row is inserted directly, so this
    fails if `list_hybrid_profile_rows`'s session-mode gate is ever deleted
    -- it is the gate, not merely the absence of any row, that this test
    protects."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    with store._connect() as db:
        db.execute(
            """INSERT INTO slide_captures (
               capture_id, session_number, captured_at, capture_path, checksum,
               success, reason, duration_ms, attempts_json, job_state,
               profile_enabled, profile_queued_ns
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "leaked-capture", session.number, STARTED_AT.isoformat(),
                "x", "y", 1, "ok", 5.0, "[]", "complete", 1, 0,
            ),
        )

    assert store.list_hybrid_profile_rows(session.number) == ()


def test_legacy_slide_captures_table_without_profile_columns_is_migrated(tmp_path):
    """#258 migration: every `profile_*` column is added additively to a
    real pre-#258 `slide_captures` table (already carrying every earlier
    Hybrid column) -- never a silent no-op that would crash the first
    `record_slide_capture(profile=True)` write with `sqlite3.OperationalError:
    no such column`."""
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
                shadow_comparison_json TEXT,
                priority INTEGER
            )"""
        )
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert "profile_enabled" not in pre_migration_columns

    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert {
        "profile_enabled", "profile_queued_ns", "profile_current_stage",
        "profile_stage_ms_json", "profile_total_ms", "profile_shadow",
    } <= post_migration_columns

    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    capture_id = store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0, profile=True,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    assert row["profile_enabled"] == 1
    assert row["profile_total_ms"] is not None
    assert json.loads(str(row["profile_stage_ms_json"]))
