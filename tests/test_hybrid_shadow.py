"""TDD coverage for #254: Hybrid Shadow.

A `hybrid_shadow` slide job selects candidates exactly as Hybrid would
(`ProcessingStore.select_hybrid_candidates`), then runs ONE complete accurate
score pass over the WHOLE frozen pool. The complete map/verdict is what gets
written durably (via the unchanged `_finalize_claim`) and is what the
operator sees. The proposed Hybrid verdict is derived afterward by FILTERING
that same complete map down to the claim-plus-candidate subset and calling
`evaluate_work_order` a second time (a second pure computation, never a
second score) -- see `ProcessingStore._persist_shadow_comparison`.

This file drives the whole store-side slice through `ProcessingStore` with
synthetic captures, an injected slide preprocessor, an injected
`score_cache_builder`, and a stubbed `select_hybrid_candidates`/
`score_routed_caches` seam -- no camera, no network, no real timing, no real
descriptor catalog. Candidate selection is hand-authored throughout (per the
issue's testing decision) rather than driven through the real Heuristic
Candidate Band math, which `tests/test_hybrid_slide_queue.py` (#253) and
`tests/test_candidate_band.py` already cover.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import session.processing_store as processing_store
from session.processing_store import ProcessingStore
from session.preparation import PreparedResult, PreparedSpecimen
from slide.qr import DecodeCandidate, select_slide_identity
from verify.candidate_band import CandidateSelection
from verify.invariant_descriptors import DescriptorValue
from verify.scorer import LockedScoreCache, _ComponentFeatures


STARTED_AT = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
_DESCRIPTOR_NAMES = ("fake_descriptor_v1",)

CLAIM = "10000001"
RUNNER_UP = "10000002"  # the true strongest non-claim competitor
OTHER = "10000003"

# Hand-authored accurate score map (issue's testing decision: drive the
# comparison from hand-authored score maps and candidate sets, not the real
# scorer/descriptor pipeline). MATCH_MARGIN is 0.05 (code/constants.py):
# CLAIM vs RUNNER_UP sit 0.015 apart (a genuine near-miss under the current
# 0.02 MATCH_MARGIN); CLAIM vs OTHER sit
# 0.80 apart (a clear win once RUNNER_UP is out of the picture).
_SCORE_BY_BLOCK = {CLAIM: 0.90, RUNNER_UP: 0.885, OTHER: 0.10}
# Fraction each block's fixed mask is filled to (see _ValueEncodingBlockPreprocessor
# below) -- distinct so the fake scorer can tell blocks apart by mask content
# alone, exactly like tests/test_hybrid_slide_queue.py's own fixture of the
# same name.
_FRACTION_BY_BLOCK = {CLAIM: 1.0, RUNNER_UP: 0.875, OTHER: 0.125}


@dataclass
class _FakeScoreResult:
    score: float
    selected_metric: str = "synthetic_test_metric"
    router_size_signal: float = 1.0
    block_occupied_fraction: float = 1.0
    slide_occupied_fraction: float = 1.0
    best_angle: float = 0.0
    best_flip: bool = False
    align_soft_iou: float = 1.0
    mask_iou: float = 1.0


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """The real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do not
    (mirrors tests/test_hybrid_slide_queue.py's fixture of the same name)."""
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


class _ValueEncodingBlockPreprocessor:
    """Encodes each block capture's fill value into its mask's fill FRACTION,
    so distinct blocks end up with genuinely distinguishable mask content --
    needed because `hybrid_pool()` decompresses fresh array objects on every
    read, so a fake scorer keyed on object identity would not survive the
    archive round-trip between freeze and the worker's own call (mirrors
    tests/test_hybrid_slide_queue.py's fixture of the same name).
    """

    def __call__(self, capture_path: Path):
        image = cv2.imread(str(capture_path))
        assert image is not None
        fill_value = int(image[0, 0, 0])
        rows_filled = max(1, min(8, round(fill_value / 255 * 8)))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:rows_filled, :] = 255
        return mask, {"role": "block", "roi_ok": True}


class _FixedSlidePreprocessor:
    def __call__(self, image: np.ndarray) -> PreparedResult:
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )


class _IdenticalFingerprintBuilder:
    """Every block/slide gets the IDENTICAL fingerprint vector, keyed under
    the fake descriptor name `_DESCRIPTOR_NAMES` freezes with -- selection is
    fully stubbed in this file (`_stub_selection`), so the real fingerprint
    content is never inspected; this only needs to exist so
    `freeze_hybrid_pool`'s `values[name].vector for name in descriptor_names`
    lookup does not KeyError on a descriptor name outside the real catalog
    (mirrors tests/test_hybrid_slide_queue.py's fixture of the same name)."""

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


def _hand_authored_score_routed_caches(block_cache, slide_cache, **kwargs):
    """Fake accurate scorer: score is purely a function of the BLOCK mask's
    fill fraction (`_FRACTION_BY_BLOCK`), never the slide -- deterministic and
    independent of which/how many blocks get scored, so counting invocations
    and asserting per-block scores is unambiguous."""
    fraction = round(float(block_cache.normalized_mask.mean()) / 255.0, 3)
    for block_id, block_fraction in _FRACTION_BY_BLOCK.items():
        if abs(fraction - block_fraction) < 1e-6:
            return _FakeScoreResult(score=_SCORE_BY_BLOCK[block_id])
    raise AssertionError(f"unrecognized block mask fraction {fraction}")


def _fill_value_for(block_id: str) -> int:
    return round(_FRACTION_BY_BLOCK[block_id] * 255)


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    kwargs.setdefault("preprocessor", _ValueEncodingBlockPreprocessor())
    kwargs.setdefault("slide_preprocessor", _FixedSlidePreprocessor())
    kwargs.setdefault("fingerprint_builder", _IdenticalFingerprintBuilder())
    kwargs.setdefault("score_cache_builder", _score_cache_builder)
    return ProcessingStore(tmp_path / "processing", **kwargs)


def _capture_png(value: int) -> bytes:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    return png.tobytes()


def _capture_block(store: ProcessingStore, session_number: int, block_id: str) -> None:
    assert store.scan_block(session_number, block_id).accepted
    body = _capture_png(_fill_value_for(block_id))
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
    """Start a session in ``session_mode``, capture/freeze ``block_ids`` into
    one pool. Returns (session_number, work_order_id)."""
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    for block_id in block_ids:
        _capture_block(store, session.number, block_id)
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True
    return session.number, work_order_id


def _stub_selection(candidate_ids: tuple[str, ...], pruned_ids: tuple[str, ...]):
    """A hand-authored, always-valid CandidateSelection: exactly what real
    Hybrid would have selected, with no dependency on the real descriptor
    catalog/fusion math (already covered by #253's own tests)."""
    selection = CandidateSelection(
        claim_id=CLAIM,
        candidate_ids=candidate_ids,
        pruned_ids=pruned_ids,
        shape_class=None,
        gap_threshold=0.0,
        floor_count=0,
        weight_set_name="hand_authored_test",
        fallback_required=False,
        fallback_reason=None,
    )

    def fake_select_hybrid_candidates(self, pool, claim_id, slide_cache, **kwargs):
        assert claim_id == CLAIM
        return selection

    return fake_select_hybrid_candidates


def _record_slide(store: ProcessingStore, session_number: int, tmp_path: Path) -> str:
    return store.record_slide_capture(
        session_number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result(CLAIM),
        duration_ms=5.0,
    )


# ---------------------------------------------------------------------------
# 1. Exactly one complete pass: scorer invocation count equals pool size,
#    never a double-score of any (block, slide) pair.
# ---------------------------------------------------------------------------


def test_shadow_scores_entire_pool_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(OTHER,), pruned_ids=(RUNNER_UP,)),
    )
    scored_cache_ids: list[int] = []
    real_scorer = _hand_authored_score_routed_caches

    def counting_scorer(block_cache, slide_cache, **kwargs):
        scored_cache_ids.append(id(block_cache))
        return real_scorer(block_cache, slide_cache, **kwargs)

    monkeypatch.setattr(processing_store, "score_routed_caches", counting_scorer)

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", (CLAIM, RUNNER_UP, OTHER),
    )
    _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    # Exactly one call per pool block -- pool size, no duplicates.
    assert len(scored_cache_ids) == 3
    assert len(set(scored_cache_ids)) == 3


# ---------------------------------------------------------------------------
# 2/3. Non-vacuous divergence: the pruned block is the true runner-up, so the
#    COMPLETE map yields REVIEW (near-miss) while the SUBSET yields PASS
#    (clear win). Displayed/durable verdict must be the complete REVIEW; the
#    comparison record must capture the proposed PASS and both margins.
# ---------------------------------------------------------------------------


def test_complete_verdict_is_durable_and_comparison_records_the_divergence(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(OTHER,), pruned_ids=(RUNNER_UP,)),
    )
    monkeypatch.setattr(
        processing_store, "score_routed_caches", _hand_authored_score_routed_caches
    )

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", (CLAIM, RUNNER_UP, OTHER),
    )
    capture_id = _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"

    # The displayed/durable verdict is the COMPLETE map's REVIEW (near-miss
    # against the pruned true runner-up) -- never the subset's PASS.
    assert store.get_set(session_number, CLAIM)["verdict"] == "REVIEW"

    comparison = json.loads(row["shadow_comparison_json"])
    assert comparison["claim_id"] == CLAIM
    assert comparison["candidate_ids"] == sorted([CLAIM, OTHER])
    assert comparison["pruned_ids"] == [RUNNER_UP]
    assert comparison["complete_verdict"] == "REVIEW"
    assert comparison["proposed_verdict"] == "PASS"
    assert comparison["verdict_differs"] is True
    assert comparison["complete_match_margin"] == pytest.approx(0.015)
    assert comparison["proposed_match_margin"] == pytest.approx(0.80)
    assert comparison["match_margin_delta"] == pytest.approx(0.015 - 0.80)


# ---------------------------------------------------------------------------
# 4. Agreement case: complete and subset produce the SAME verdict -- the
#    comparison record must say so, no false-positive difference.
# ---------------------------------------------------------------------------


def test_comparison_reports_no_difference_when_verdicts_agree(tmp_path, monkeypatch):
    # This time the TRUE runner-up (RUNNER_UP) survives selection, and OTHER
    # (harmless) is the one pruned -- both complete and subset see the same
    # near-miss competitor, so both verdicts agree.
    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(RUNNER_UP,), pruned_ids=(OTHER,)),
    )
    monkeypatch.setattr(
        processing_store, "score_routed_caches", _hand_authored_score_routed_caches
    )

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", (CLAIM, RUNNER_UP, OTHER),
    )
    capture_id = _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    comparison = json.loads(row["shadow_comparison_json"])
    assert comparison["complete_verdict"] == "REVIEW"
    assert comparison["proposed_verdict"] == "REVIEW"
    assert comparison["verdict_differs"] is False
    assert comparison["match_margin_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Plain hybrid control: unchanged -- scores only the candidate subset,
#    never the whole pool, and never persists a shadow comparison.
# ---------------------------------------------------------------------------


def test_plain_hybrid_still_scores_only_the_subset_not_the_whole_pool(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(RUNNER_UP,), pruned_ids=(OTHER,)),
    )
    scored_block_fractions: list[float] = []
    real_scorer = _hand_authored_score_routed_caches

    def recording_scorer(block_cache, slide_cache, **kwargs):
        scored_block_fractions.append(
            round(float(block_cache.normalized_mask.mean()) / 255.0, 3)
        )
        return real_scorer(block_cache, slide_cache, **kwargs)

    monkeypatch.setattr(processing_store, "score_routed_caches", recording_scorer)

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", (CLAIM, RUNNER_UP, OTHER),
    )
    capture_id = _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    # Only claim + selected candidate (2), never OTHER (pruned) -- strictly
    # fewer than the pool size of 3. Deleting the mode branch (always
    # scoring `pool.block_ids`) would make this 3.
    assert len(scored_block_fractions) == 2
    assert set(scored_block_fractions) == {
        _FRACTION_BY_BLOCK[CLAIM], _FRACTION_BY_BLOCK[RUNNER_UP],
    }

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    # Plain Hybrid never runs the #254 comparison path at all.
    assert row["shadow_comparison_json"] is None
    assert store.get_set(session_number, CLAIM)["verdict"] == "REVIEW"


# ---------------------------------------------------------------------------
# 6. A comparison-persistence failure must not lose the real verdict, and
#    must not become job_state='error' -- only a genuine scoring/IO failure
#    earns ERROR.
# ---------------------------------------------------------------------------


def test_shadow_comparison_persistence_failure_still_writes_real_verdict(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(OTHER,), pruned_ids=(RUNNER_UP,)),
    )
    monkeypatch.setattr(
        processing_store, "score_routed_caches", _hand_authored_score_routed_caches
    )

    def raising_persist_shadow_comparison(self, *args, **kwargs):
        raise RuntimeError("synthetic #254 comparison persistence failure")

    monkeypatch.setattr(
        ProcessingStore, "_persist_shadow_comparison", raising_persist_shadow_comparison
    )

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", (CLAIM, RUNNER_UP, OTHER),
    )
    capture_id = _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    # Never ERROR: the complete pass already produced an authoritative
    # verdict before the comparison was even attempted.
    assert row["job_state"] == "complete"
    assert row["shadow_comparison_json"] is None
    assert store.get_set(session_number, CLAIM)["verdict"] == "REVIEW"

    # The failure is degraded, not silent: it lands in the durable log.
    session = store._session_identity(session_number)
    log_text = (session.directory / "events.log").read_text(encoding="utf-8")
    assert "hybrid_shadow_comparison_failed" in log_text


# ---------------------------------------------------------------------------
# 7. Legacy-DB migration: `shadow_comparison_json` is added additively, never
#    a silent no-op against a real pre-#254 database.
# ---------------------------------------------------------------------------


def test_legacy_slide_captures_table_without_shadow_comparison_json_is_migrated(
    tmp_path, monkeypatch,
):
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
                top_score REAL,
                runner_up_score REAL,
                match_margin REAL,
                job_state TEXT,
                candidate_selection_json TEXT
            )"""
        )
        connection.commit()
        pre_migration_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(slide_captures)")
        }
    finally:
        connection.close()
    assert "shadow_comparison_json" not in pre_migration_columns

    monkeypatch.setattr(
        ProcessingStore, "select_hybrid_candidates",
        _stub_selection(candidate_ids=(OTHER,), pruned_ids=(RUNNER_UP,)),
    )
    monkeypatch.setattr(
        processing_store, "score_routed_caches", _hand_authored_score_routed_caches
    )
    store = _make_store(tmp_path)
    with store._connect() as db:
        post_migration_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(slide_captures)")
        }
    assert "shadow_comparison_json" in post_migration_columns

    session_number, _ = _freeze_hybrid_session(
        store, "hybrid_shadow", (CLAIM, RUNNER_UP, OTHER),
    )
    capture_id = _record_slide(store, session_number, tmp_path)
    store.wait_for_jobs()

    row = store.get_slide_capture(session_number, capture_id)
    assert row["job_state"] == "complete"
    # The migrated column round-trips a real write from
    # `_persist_shadow_comparison`.
    comparison = json.loads(row["shadow_comparison_json"])
    assert comparison["complete_verdict"] == "REVIEW"
    assert comparison["proposed_verdict"] == "PASS"
