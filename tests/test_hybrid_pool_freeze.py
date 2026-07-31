"""TDD coverage for Finish Blocks freezing the Hybrid Candidate Pool (#250),
re-keyed per WORK ORDER instead of per session (#269).

Drives the freeze entirely through `ProcessingStore` with synthetic block
captures -- no camera, no real segmentation -- per the issue's testing
decision. `fingerprint_builder`/`score_cache_builder` are constructor-injected
fakes (mirroring the existing `preprocessor` seam) so "built exactly once" is
a call-counter assertion, never a timing assertion.

Every freeze here now opens an explicit work-order bracket first
(`store.start_work_order(session.number)`) -- #269 gives Hybrid the same real
`work_orders` bracket Open Retrieval already has, and `hybrid_pools` is keyed
by `work_order_id`, not `session_number`; `store.hybrid_pool(...)` is called
with the work order id accordingly.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from session.workflow import ProcessingStore
from session.workflow_types import HybridPoolFreezeResult
from verify.invariant_descriptors import DescriptorValue
from verify.scorer import LockedScoreCache, _ComponentFeatures


STARTED_AT = datetime(2026, 7, 27, 9, 0, 0, tzinfo=timezone.utc)
_DESCRIPTOR_NAMES = ("fake_descriptor_v1",)


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """Mirrors test_session_workflow.py's fixture of the same name: the real
    QC panel renderer assumes the mask matches the capture's dimensions,
    which these tests' small synthetic masks deliberately do not."""
    def write_qc(capture, mask, destination):
        assert capture.is_file()
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        assert capture.is_file()
        assert reason
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


class _RoiAwarePreprocessor:
    """Every capture prepares successfully with a small fully-filled mask;
    captures written with `bad_roi_value` additionally fail the block-only
    ROI quality check (roi_ok=False) -- a block that completes preparation
    but must still be excluded as not usable."""

    def __init__(self, bad_roi_value: int | None = None):
        self.bad_roi_value = bad_roi_value

    def __call__(self, capture_path: Path):
        image = cv2.imread(str(capture_path))
        value = int(image[0, 0, 0])
        roi_ok = self.bad_roi_value is None or value != self.bad_roi_value
        mask = np.full((8, 8), 255, dtype=np.uint8)
        metadata = {
            "role": "block", "roi_ok": roi_ok,
            "roi_reason": "" if roi_ok else "synthetic bad ROI",
        }
        return mask, metadata


class _CountingFingerprintBuilder:
    def __init__(self):
        self.calls = 0

    def __call__(self, mask: np.ndarray):
        self.calls += 1
        return {
            "fake_descriptor_v1": DescriptorValue(
                vector=np.array([1.0, 2.0]), construction_ns=1
            )
        }


class _CountingScoreCacheBuilder:
    def __init__(self):
        self.calls = 0

    def __call__(self, specimen):
        self.calls += 1
        return LockedScoreCache(
            normalized_mask=specimen.mask,
            component_features=_ComponentFeatures(
                points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
            ),
        )


def _forbidden_builder(*_args, **_kwargs):
    raise AssertionError(
        "must not be called -- the pool must be read from durable storage, "
        "never recomputed"
    )


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    kwargs.setdefault("preprocessor", _RoiAwarePreprocessor())
    kwargs.setdefault("fingerprint_builder", _CountingFingerprintBuilder())
    kwargs.setdefault("score_cache_builder", _CountingScoreCacheBuilder())
    return ProcessingStore(tmp_path / "processing", **kwargs)


def _capture_png(value: int) -> bytes:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    return png.tobytes()


def _capture_block(store: ProcessingStore, session_number: int, block_id: str, value: int) -> None:
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


# ---------------------------------------------------------------------------
# <2 usable blocks: no freeze, no abandon, back to blocks with a message
# ---------------------------------------------------------------------------


def test_zero_blocks_does_not_freeze_and_returns_to_blocks_with_message(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    store.begin_block_drain(session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result == HybridPoolFreezeResult(
        False, (), result.message
    )
    assert "at least 2" in result.message
    assert store.snapshot(session).phase == "blocks"
    assert store.hybrid_pool(work_order_id) is None
    # The work order remains usable: capture can resume immediately.
    assert store.scan_block(session.number, "11111111").accepted


def test_one_usable_block_does_not_freeze_and_names_the_short_pool(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _drain(store, session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is False
    assert result.usable_block_ids == ("11111111",)
    assert store.snapshot(session).phase == "blocks"
    assert store.hybrid_pool(work_order_id) is None


def test_one_usable_of_two_captured_blocks_falls_short(tmp_path):
    """A block that completes preparation but fails the block-only ROI gate
    is not usable, even though its preprocessing_status is 'complete'."""
    store = _make_store(tmp_path, preprocessor=_RoiAwarePreprocessor(bad_roi_value=99))
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 99)  # bad ROI
    _drain(store, session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is False
    assert result.usable_block_ids == ("11111111",)
    assert store.snapshot(session).phase == "blocks"
    row = store.get_set(session.number, "22222222")
    # The excluded block's own row is untouched -- #250 does not invent a
    # correction path; it is still 'complete' from preparation's point of
    # view, merely not counted toward the pool.
    assert row["preprocessing_status"] == "complete"


def test_unresolved_block_work_blocks_the_freeze(tmp_path):
    """A scanned block whose capture never arrives stays 'awaiting_capture'
    forever -- unresolved, so the freeze must wait, exactly like
    try_enter_slides waits today."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    # A second, fully-captured-and-prepared block exists too, so the ONLY
    # reason the freeze can't proceed is the still-uncaptured scan below.
    _capture_block(store, session.number, "22222222", 20)
    assert store.scan_block(session.number, "11111111").accepted
    store.begin_block_drain(session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result == HybridPoolFreezeResult(False, (), "block work is still resolving")
    assert store.snapshot(session).phase == "draining_blocks"
    assert store.hybrid_pool(work_order_id) is None


# ---------------------------------------------------------------------------
# No open work order: never raises, mirrors "unknown session" (#269)
# ---------------------------------------------------------------------------


def test_freeze_without_an_open_work_order_returns_not_frozen_without_raising(tmp_path):
    """#269: freeze_hybrid_pool now resolves the currently open (capturing)
    work order internally and keys everything on it. A direct/console caller
    that skips start_work_order -- reachable by DEFAULT, not merely a
    console edge case, since the kiosk enters 'blocks' phase before any
    work order is ever opened -- must still get a clean not-frozen result,
    never a raise -- this can run on the background poll-drain tick.

    BLOCKER regression check (review finding): before the fix, this branch
    returned without bouncing phase back to 'blocks', so `begin_block_drain`
    (which always runs before Finish Blocks) left the session stuck in
    'draining_blocks' forever -- `scan_block` refuses once phase leaves
    'blocks', and `SessionWorkflow.poll_drain` calls this every second with
    no way to ever exit. Phase must return to 'blocks', and capture must
    reopen, exactly like every other insufficient-freeze bounce."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is False
    assert "no open work order" in result.message
    assert store.snapshot(session).phase == "blocks"
    # The session must not be live-locked: capture can resume immediately,
    # and polling freeze_hybrid_pool again (mirroring poll_drain) must not
    # re-stick the phase in 'draining_blocks'.
    assert store.scan_block(session.number, "11111111").accepted


# ---------------------------------------------------------------------------
# >=2 usable blocks: freeze, durable, fingerprints/caches built exactly once
# ---------------------------------------------------------------------------


def _frozen_session(tmp_path, **store_kwargs):
    store = _make_store(tmp_path, **store_kwargs)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 20)
    _drain(store, session.number)
    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)
    return store, session, work_order_id, result


def test_two_usable_blocks_freezes_and_enters_slides(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)

    assert result.frozen is True
    assert set(result.usable_block_ids) == {"11111111", "22222222"}
    assert store.snapshot(session).phase == "slides"


def test_frozen_pool_is_readable_and_matches_what_was_frozen(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)

    pool = store.hybrid_pool(work_order_id)
    assert pool is not None
    assert pool.work_order_id == work_order_id
    assert pool.session_number == session.number
    assert set(pool.block_ids) == set(result.usable_block_ids)
    assert pool.descriptor_names == _DESCRIPTOR_NAMES
    for block_id in pool.block_ids:
        assert np.array_equal(
            pool.fingerprints[block_id]["fake_descriptor_v1"], np.array([1.0, 2.0])
        )
        assert pool.score_caches[block_id].normalized_mask.shape == (8, 8)


def test_frozen_pool_artifacts_live_under_work_orders_named_by_work_order_id(tmp_path):
    """#269: artifact paths move from <session_dir>/hybrid_pool.{json,npz}
    (one per session) to <session_dir>/work_orders/work_order_{id:06d}_
    hybrid_pool.{json,npz} -- the same convention as the existing
    work_order_{id:06d}_verdicts.csv/_sheets artifacts."""
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    work_orders_dir = session.directory / "work_orders"
    manifest_path = work_orders_dir / f"work_order_{work_order_id:06d}_hybrid_pool.json"
    archive_path = work_orders_dir / f"work_order_{work_order_id:06d}_hybrid_pool.npz"
    assert manifest_path.is_file()
    assert archive_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["work_order_id"] == work_order_id
    assert manifest["session_number"] == session.number


def test_fingerprints_and_score_caches_are_built_exactly_once_per_block(tmp_path):
    fingerprint_builder = _CountingFingerprintBuilder()
    score_cache_builder = _CountingScoreCacheBuilder()
    store, session, work_order_id, result = _frozen_session(
        tmp_path, fingerprint_builder=fingerprint_builder,
        score_cache_builder=score_cache_builder,
    )
    assert result.frozen is True
    assert fingerprint_builder.calls == 2  # once per usable block, never per pair
    assert score_cache_builder.calls == 2

    # Reading the pool again (in the same process) must not re-enter either
    # builder -- a spy/counter assertion, not a timing one.
    store.hybrid_pool(work_order_id)
    store.hybrid_pool(work_order_id)
    assert fingerprint_builder.calls == 2
    assert score_cache_builder.calls == 2


def test_repeated_freeze_is_idempotent_and_does_not_recompute(tmp_path):
    fingerprint_builder = _CountingFingerprintBuilder()
    score_cache_builder = _CountingScoreCacheBuilder()
    store, session, work_order_id, first = _frozen_session(
        tmp_path, fingerprint_builder=fingerprint_builder,
        score_cache_builder=score_cache_builder,
    )
    assert first.frozen is True

    second = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert second.frozen is True
    assert set(second.usable_block_ids) == set(first.usable_block_ids)
    assert fingerprint_builder.calls == 2
    assert score_cache_builder.calls == 2
    assert store.snapshot(session).phase == "slides"


def test_freeze_is_durable_across_a_simulated_restart(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    # Simulate a processing-computer restart: a brand new ProcessingStore
    # instance pointed at the SAME durable root, with builders that raise if
    # ever invoked -- proving the restart never recomputes anything.
    restarted = ProcessingStore(
        tmp_path / "processing",
        preprocessor=_RoiAwarePreprocessor(),
        fingerprint_builder=_forbidden_builder,
        score_cache_builder=_forbidden_builder,
    )

    pool = restarted.hybrid_pool(work_order_id)

    assert pool is not None
    assert set(pool.block_ids) == set(result.usable_block_ids)
    assert pool.descriptor_names == _DESCRIPTOR_NAMES
    assert restarted.snapshot(
        restarted.resume_session(session.number)
    ).phase == "slides"
    # A second freeze call after "restart" is still idempotent and still
    # never touches the forbidden builders.
    replay = restarted.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )
    assert replay.frozen is True


# ---------------------------------------------------------------------------
# One-way immutability: the frozen pool rejects mutation attempts
# ---------------------------------------------------------------------------


def test_frozen_pool_refuses_new_block_scans(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    outcome = store.scan_block(session.number, "33333333")

    assert outcome.accepted is False
    assert "closed" in outcome.message.lower()


def test_frozen_pool_blocks_cannot_be_unscanned(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    removed = store.unscan_block(session.number, "11111111")

    assert removed is False
    row = store.get_set(session.number, "11111111")
    assert row["preprocessing_status"] == "complete"


def test_frozen_pool_blocks_cannot_be_dismissed_or_corrected(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    with pytest.raises(ValueError, match="only a failed block can be dismissed"):
        store.dismiss_block(session.number, "11111111", reason="operator mistake")

    row = store.get_set(session.number, "11111111")
    assert row["preprocessing_status"] == "complete"
    assert row["dismissed_at"] is None


def test_frozen_pool_blocks_cannot_be_recaptured(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    body = _capture_png(77)
    checksum = hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="unique unfilled set"):
        store.receive_capture(
            session.number, capture_id="cap-11111111-retry", block_id="11111111",
            checksum=checksum, body=body, recapture=True,
        )


# ---------------------------------------------------------------------------
# Unknown session: never raises (mirrors try_enter_slides) -- #250 review F6
# ---------------------------------------------------------------------------


def test_freeze_on_unknown_session_returns_not_frozen_without_raising(tmp_path):
    """#250 review F6: this used to raise ValueError("unknown session"),
    which is exactly the danger `try_enter_slides` was already designed to
    avoid -- `poll_drain` calls this on a background camera tick, and a raise
    there propagates through the RPC layer as a `store.remote.StoreError`
    (a `ValueError` subclass) straight into `PiCaptureRuntime._camera_loop`'s
    `except Exception`, which kills the camera loop. `try_enter_slides`
    already returns `False` for an unknown session instead of raising;
    `freeze_hybrid_pool` must match that contract."""
    store = _make_store(tmp_path)

    result = store.freeze_hybrid_pool(999, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is False
    assert result.usable_block_ids == ()


# ---------------------------------------------------------------------------
# Empty descriptor_names: refuse to freeze a pool with zero fingerprints
# (#250 review F2) -- the loader-side rejection lives in
# tests/test_hybrid_configuration.py; this is the store-side backstop for
# any caller that bypasses the loader.
# ---------------------------------------------------------------------------


def test_empty_descriptor_names_refuses_to_freeze(tmp_path):
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 20)
    _drain(store, session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=())

    assert result.frozen is False
    assert "configuration defect" in result.message
    assert store.snapshot(session).phase == "blocks"
    # #269 FIX5c: this negative control (no pool was ever frozen) was
    # dropped during the #269 re-key rather than converted to the
    # work-order-keyed form -- restore the coverage.
    assert store.hybrid_pool(work_order_id) is None
    # The work order remains usable, exactly like the <2-usable-blocks bounce.
    assert store.scan_block(session.number, "33333333").accepted


# ---------------------------------------------------------------------------
# Already-frozen branch repairs a stuck 'draining_blocks' phase (#250 review
# F3), now on the work-order-keyed row.
# ---------------------------------------------------------------------------


def test_already_frozen_repairs_phase_stuck_in_draining_blocks(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    assert store.snapshot(session).phase == "slides"

    # Force phase back to 'draining_blocks' directly, bypassing the ordinary
    # phase-transition methods -- the same stuck state the review proved:
    # poll_drain calls freeze_hybrid_pool every second while draining, and
    # the already-frozen branch used to never touch phase at all.
    with store._connect() as db:
        db.execute(
            "UPDATE sessions SET phase='draining_blocks' WHERE session_number=?",
            (session.number,),
        )
    assert store.snapshot(session).phase == "draining_blocks"

    second = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES
    )

    assert second.frozen is True
    assert set(second.usable_block_ids) == set(result.usable_block_ids)
    assert store.snapshot(session).phase == "slides"


# ---------------------------------------------------------------------------
# Ledger replay stability of a bounce (#250 review F11)
# ---------------------------------------------------------------------------


def test_bounce_ledger_replay_returns_the_same_result(tmp_path):
    """`freeze_hybrid_pool` is in `_LEDGERED`; tests/test_idempotency_ledger.py
    documents that a replay must return the SAME response. Before the fix,
    a bounce never recorded a ledger row, so a replay took the
    `phase != 'draining_blocks'` branch and returned a DIFFERENT message
    ("block drain has not started") than the original bounce."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _drain(store, session.number)

    first = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES,
        request_id="req-bounce-replay",
    )
    assert first.frozen is False

    replay = store.freeze_hybrid_pool(
        session.number, descriptor_names=_DESCRIPTOR_NAMES,
        request_id="req-bounce-replay",
    )

    assert replay == first


# ---------------------------------------------------------------------------
# hybrid_pool() read failures are loud, never silent (#250 review F7)
# ---------------------------------------------------------------------------


def test_hybrid_pool_raises_clearly_on_missing_archive(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    archive_path = (
        session.directory / "work_orders"
        / f"work_order_{work_order_id:06d}_hybrid_pool.npz"
    )
    assert archive_path.is_file()
    archive_path.unlink()

    with pytest.raises(ValueError, match="archive"):
        store.hybrid_pool(work_order_id)


def test_hybrid_pool_raises_clearly_on_truncated_archive(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    archive_path = (
        session.directory / "work_orders"
        / f"work_order_{work_order_id:06d}_hybrid_pool.npz"
    )
    body = archive_path.read_bytes()
    archive_path.write_bytes(body[: len(body) // 2])

    with pytest.raises(ValueError, match="archive"):
        store.hybrid_pool(work_order_id)


def test_hybrid_pool_raises_clearly_on_manifest_schema_mismatch(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    manifest_path = (
        session.directory / "work_orders"
        / f"work_order_{work_order_id:06d}_hybrid_pool.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = manifest["schema_version"] + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        store.hybrid_pool(work_order_id)


# ---------------------------------------------------------------------------
# Concurrent duplicate request_id: must not corrupt the archive or crash
# (#250 review F4)
# ---------------------------------------------------------------------------


def test_concurrent_duplicate_freeze_requests_do_not_corrupt_or_crash(tmp_path):
    """A retried duplicate request (e.g. an RPC client timeout retry) can run
    concurrently with the original attempt -- both threads race into
    `freeze_hybrid_pool` with the SAME request_id. The per-session lock must
    serialize them: no exception either thread's caller ever sees, both
    return the IDENTICAL result, and the archive is fully, uncorrupted
    readable afterward (the old fixed '.hybrid_pool.npz.tmp' staging name let
    a second concurrent writer keep writing into an inode the first had
    already `os.replace`'d into place)."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 20)
    _drain(store, session.number)

    results: list[HybridPoolFreezeResult] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _call():
        try:
            barrier.wait(timeout=5)
            results.append(
                store.freeze_hybrid_pool(
                    session.number, descriptor_names=_DESCRIPTOR_NAMES,
                    request_id="req-concurrent-freeze",
                )
            )
        except BaseException as exc:  # pragma: no cover -- assert-checked below
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].frozen is True

    pool = store.hybrid_pool(work_order_id)
    assert pool is not None
    assert set(pool.block_ids) == set(results[0].usable_block_ids)


# ---------------------------------------------------------------------------
# Legacy (pre-#269) hybrid_pools schema: migrated, never left broken
# (#269 review BLOCKER finding 2)
# ---------------------------------------------------------------------------


def test_legacy_session_keyed_hybrid_pools_table_is_migrated_not_left_broken(tmp_path):
    """Commit df67237 (already committed on this branch) shipped
    `hybrid_pools` with `session_number INTEGER PRIMARY KEY` and no
    `work_order_id` column at all. `CREATE TABLE IF NOT EXISTS` is a no-op
    against a database file that already has that legacy shape, so a real
    pre-#269 database used to raise
    `sqlite3.OperationalError: no such column: work_order_id` out of BOTH
    `freeze_hybrid_pool` and `hybrid_pool` -- contradicting
    `freeze_hybrid_pool`'s own "Never raises" docstring contract. Build the
    legacy shape by hand (dev-only data, nothing worth preserving, per
    #269's own migration decision), then prove a fresh `ProcessingStore`
    pointed at that file migrates cleanly and both methods work."""
    root = tmp_path / "processing"
    root.mkdir()
    connection = sqlite3.connect(root / "sessions.sqlite3")
    try:
        connection.execute(
            """CREATE TABLE hybrid_pools (
                session_number INTEGER PRIMARY KEY,
                frozen_at TEXT NOT NULL,
                block_ids TEXT NOT NULL,
                descriptor_names TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                archive_path TEXT NOT NULL
            )"""
        )
        connection.commit()
    finally:
        connection.close()

    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 20)
    _drain(store, session.number)

    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is True
    pool = store.hybrid_pool(work_order_id)
    assert pool is not None
    assert set(pool.block_ids) == {"11111111", "22222222"}


# ---------------------------------------------------------------------------
# Blocks scanned/captured before start_work_order are adopted, not stranded
# (#269 review BLOCKER finding 3)
# ---------------------------------------------------------------------------


def test_blocks_captured_before_start_work_order_are_adopted_into_the_pool(tmp_path):
    """#269 review: `scan_block` stamps `sets.work_order_id` from whichever
    work order is open AT SCAN TIME -- NULL if none is. Before the fix,
    `start_work_order` never adopted those NULL rows, and the freeze's
    `AND work_order_id=?` candidate filter made them permanently invisible:
    two fully captured, quality-passing blocks existed and the operator was
    told "capture more blocks" with no way to recover (re-scanning fails
    with "Block already scanned"; `unscan_block` requires
    `capture_id IS NULL`, which a captured block never has)."""
    store = _make_store(tmp_path)
    session = store.start_session(started_at=STARTED_AT)
    _capture_block(store, session.number, "11111111", 10)
    _capture_block(store, session.number, "22222222", 20)

    work_order_id = store.start_work_order(session.number)
    _drain(store, session.number)
    result = store.freeze_hybrid_pool(session.number, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is True
    assert set(result.usable_block_ids) == {"11111111", "22222222"}
    pool = store.hybrid_pool(work_order_id)
    assert pool is not None
    assert set(pool.block_ids) == {"11111111", "22222222"}
