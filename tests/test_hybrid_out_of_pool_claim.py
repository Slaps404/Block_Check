"""TDD coverage for #251: an Out-of-Pool Claim (CONTEXT.md glossary entry) --
a Hybrid slide whose decoded block id is absent from ITS OWN work order's
frozen Hybrid Candidate Pool.

Per the product decision superseding the issue's original body: the attempt
is REJECTED as an identity mismatch (never scored), gets an immediate REVIEW
verdict with the reason "block id not found in session inventory", and stays
durably recorded and attributable to the work order it was captured under --
visible on a per-work-order Results view once #252 wires the results-ready
gate. This file asserts DB state only, never kiosk/Results screen state
(`lifecycle_state` only reaches `results_ready` via the scoring pass #252
adds, which Hybrid never calls today).

The real point of this issue (see THE ISOLATION TEST below) is closing the
hole `resolve_claim`'s existing session-wide `get_set` lookup leaves open:
`sets` is keyed by `(session_number, block_id)` only, so a block captured
under work order A is still found by `get_set` when a slide in a DIFFERENT
work order B claims the same block id -- and would be wrongly scored against
it. The guard in `record_slide_capture` must decide membership against work
order B's own frozen `hybrid_pools` row, never the whole-session `sets`
table.

Drives everything through `ProcessingStore` with synthetic captures, no
camera, no real segmentation -- mirrors tests/test_hybrid_work_order_bracket.py
and tests/test_hybrid_pool_freeze.py's testing decision.
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


STARTED_AT = datetime(2026, 7, 28, 10, 0, 0, tzinfo=timezone.utc)
_DESCRIPTOR_NAMES = ("fake_descriptor_v1",)

# The exact, canonical operator-facing reason string (CONTEXT.md "Out-of-Pool
# Claim"; also `resolve_claim`'s existing KeyError-path wording, reused
# verbatim so the operator sees one consistent string regardless of WHY the
# block wasn't found).
_OUT_OF_POOL_REASON = "block id not found in session inventory"


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """The real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do not
    (mirrors tests/test_hybrid_work_order_bracket.py's fixture of the same
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
    pixel content."""

    def __call__(self, capture_path: Path):
        return np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


class _IdenticalFingerprintBuilder:
    """Every block gets the IDENTICAL fingerprint vector -- the maximally
    confusable case the isolation test asks for: if isolation only held
    because descriptor VALUES happened to differ between work orders, this
    would expose it. Only distinct block_ids ever separate them."""

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


def _job_count(store: ProcessingStore) -> int:
    """`store._jobs` is a lifetime-cumulative list (block preprocessing jobs
    are appended but never pruned once complete), so "no scoring job
    queued" must be a before/after DELTA around the call under test, never
    an absolute-zero check -- this test always captures at least one real
    block first."""
    with store._jobs_lock:
        return len(store._jobs)


# ---------------------------------------------------------------------------
# Primary red test: an out-of-pool claim in a single Hybrid work order.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session_mode", ["hybrid", "hybrid_shadow"])
def test_out_of_pool_claim_is_reviewed_pool_unmutated_and_capture_continues(
    tmp_path, session_mode,
):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _capture_block(store, session.number, "22222222")
    _drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True

    # The claimed block id was never captured at all in this session.
    jobs_before = _job_count(store)
    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide_missing.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"),
        duration_ms=5.0,
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["verdict"] == "REVIEW"
    assert row["claim_reason"] == _OUT_OF_POOL_REASON
    assert row["work_order_id"] == work_order_id

    # No scoring job was ever queued: the REVIEW came from identity mismatch
    # alone, never from scoring against anything.
    assert _job_count(store) == jobs_before

    # The frozen pool is completely unchanged by the rejected attempt.
    pool = store.hybrid_pool(work_order_id)
    assert set(pool.block_ids) == {"11111111", "22222222"}

    # A following IN-POOL slide capture still processes normally -- capture
    # continues past the rejection with no operator-visible block. Its
    # verdict legitimately stays PENDING (NULL): #251 only builds the
    # reject arm; #252 wires the accept arm's per-slide scoring.
    next_capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide_valid.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )
    next_row = store.get_slide_capture(session.number, next_capture_id)
    assert next_row["work_order_id"] == work_order_id
    assert next_row["verdict"] is None


# ---------------------------------------------------------------------------
# THE ISOLATION TEST -- the actual point of this issue.
# ---------------------------------------------------------------------------


def test_isolation_claim_for_a_different_work_orders_block_is_rejected_not_scored(
    tmp_path,
):
    """Work order A freezes a pool with blocks 11111111/22222222. Work order
    B, in the SAME session, freezes its own pool with DELIBERATELY
    confusable (identical-fingerprint) blocks 33333333/44444444. A slide
    captured under B claiming A's block "11111111" must be REVIEW-rejected
    -- never matched/scored against A's block, even though `sets` (keyed
    only by (session_number, block_id)) still holds a row for "11111111"
    that a whole-session lookup would find."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="hybrid")

    work_order_a = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111")
    _capture_block(store, session.number, "22222222")
    _drain(store, session.number)
    result_a = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_a.frozen is True
    store.finish_work_order(session.number, start_job=False)

    work_order_b = store.start_work_order(session.number)
    assert work_order_b != work_order_a
    _capture_block(store, session.number, "33333333")
    _capture_block(store, session.number, "44444444")
    _drain(store, session.number)
    result_b = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    assert result_b.frozen is True

    # B claims A's block id.
    jobs_before = _job_count(store)
    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide_cross.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("11111111"),
        duration_ms=5.0,
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["verdict"] == "REVIEW"
    assert row["claim_reason"] == _OUT_OF_POOL_REASON
    assert row["work_order_id"] == work_order_b

    assert _job_count(store) == jobs_before

    # A's own `sets` row for "11111111" was never touched by B's claim --
    # proof the wrong pairing was never scored.
    a_set_row = store.get_set(session.number, "11111111")
    assert a_set_row["verdict"] is None
    assert a_set_row["work_order_id"] == work_order_a

    # Both pools remain exactly as frozen -- no leakage in either direction.
    pool_a = store.hybrid_pool(work_order_a)
    pool_b = store.hybrid_pool(work_order_b)
    assert set(pool_a.block_ids) == {"11111111", "22222222"}
    assert set(pool_b.block_ids) == {"33333333", "44444444"}


# ---------------------------------------------------------------------------
# Negative controls: NORMAL / OPEN_RETRIEVAL are byte-identical to today.
# ---------------------------------------------------------------------------


def test_normal_mode_missing_block_claim_is_unaffected_by_the_new_guard(tmp_path):
    """NORMAL mode, no work order bracket at all (`stamped_work_order_id`
    stays None) -- the pre-#251 immediate `resolve_claim` inline path. This
    must still fire exactly as it always has: an immediate REVIEW the moment
    the slide is captured, not deferred."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="normal")
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    jobs_before = _job_count(store)
    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"),
        duration_ms=5.0,
    )

    row = store.get_slide_capture(session.number, capture_id)
    assert row["verdict"] == "REVIEW"
    assert row["claim_reason"] == _OUT_OF_POOL_REASON
    assert _job_count(store) == jobs_before


def test_open_retrieval_missing_block_claim_resolves_review_in_slide_job(
    tmp_path,
):
    """OPEN_RETRIEVAL, WITH an open work order bracket (`stamped_work_order_id`
    is NOT None, exactly the shape the new guard inspects) -- but the mode
    is not HYBRID/HYBRID_SHADOW, so the new guard must not fire. Today's
    claim remains an Open scoring outcome, not Hybrid's immediate Out-of-Pool
    rejection. With no candidate block, the per-slide job fails closed to
    REVIEW through the ordinary work-order evaluator."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT, session_mode="open_retrieval")
    store.start_work_order(session.number)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    capture_id = store.record_slide_capture(
        session.number, _write_slide_png(tmp_path / "slide.png"),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"),
        duration_ms=5.0,
    )
    store.wait_for_jobs()

    row = store.get_slide_capture(session.number, capture_id)
    assert row["job_state"] == "complete"
    assert row["verdict"] == "REVIEW"
