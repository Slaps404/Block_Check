"""Finish Blocks freezing the Hybrid Candidate Pool, keyed per WORK ORDER.

Drives the freeze entirely through `ProcessingStore` with synthetic block
captures -- no camera, no real segmentation. `fingerprint_builder`/`score_cache_builder` are constructor-injected
fakes (mirroring the existing `preprocessor` seam) so "built exactly once" is
a call-counter assertion, never a timing assertion.

Every freeze here opens an explicit work-order bracket first
(`store.start_work_order(session.number)`): Hybrid uses the same
`work_orders` bracket as Open Retrieval, and `hybrid_pools` is keyed by
`work_order_id`, not `session_number`.
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
from retrieval.invariant_descriptors import DescriptorValue
from verify.scorer import LockedScoreCache, _ComponentFeatures
from store_harness import (  # noqa: F401
    DESCRIPTOR_NAMES as _DESCRIPTOR_NAMES,
    capture_png as _capture_png,
    drain as _drain,
    lightweight_qc_artifacts,
)


STARTED_AT = datetime(2026, 7, 27, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into the shared harness fixture."""


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


def _capture_block(store: ProcessingStore, session_number: int, block_id: str, value: int) -> None:
    assert store.scan_block(session_number, block_id).accepted
    body = _capture_png(value)
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


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
    # The excluded block's own row is untouched -- there is no correction
    # path; it is still 'complete' from preparation's point of
    # view, merely not counted toward the pool.
    assert row["preprocessing_status"] == "complete"


def test_unresolved_block_work_blocks_the_freeze(tmp_path):
    """A scanned block whose capture never arrives stays 'awaiting_capture'
    forever -- unresolved, so the freeze must wait, exactly like
    try_enter_slides waits."""
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
# No open work order: never raises, mirrors "unknown session"
# ---------------------------------------------------------------------------


def test_freeze_without_an_open_work_order_returns_not_frozen_without_raising(tmp_path):
    """freeze_hybrid_pool resolves the currently open (capturing)
    work order internally and keys everything on it. A direct/console caller
    that skips start_work_order -- reachable by DEFAULT, not merely a
    console edge case, since the kiosk enters 'blocks' phase before any
    work order is ever opened -- must still get a clean not-frozen result,
    never a raise -- this can run on the background poll-drain tick.

    Regression: if this branch returned without bouncing phase back to
    'blocks', `begin_block_drain` (which always runs before Finish Blocks)
    would leave the session stuck in 'draining_blocks' forever --
    `scan_block` refuses once phase leaves 'blocks', and
    `SessionWorkflow.poll_drain` calls this every second with no way to ever
    exit. Phase must return to 'blocks', and capture must reopen, exactly
    like every other insufficient-freeze bounce."""
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


def test_two_usable_blocks_freeze_into_a_readable_work_order_pool(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)

    assert result.frozen is True
    assert set(result.usable_block_ids) == {"11111111", "22222222"}
    assert store.snapshot(session).phase == "slides"

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

    # Artifacts live at <session_dir>/work_orders/work_order_{id:06d}_
    # hybrid_pool.{json,npz}, the same convention as _verdicts.csv/_sheets.
    work_orders_dir = session.directory / "work_orders"
    manifest_path = work_orders_dir / f"work_order_{work_order_id:06d}_hybrid_pool.json"
    archive_path = work_orders_dir / f"work_order_{work_order_id:06d}_hybrid_pool.npz"
    assert manifest_path.is_file()
    assert archive_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["work_order_id"] == work_order_id
    assert manifest["session_number"] == session.number


def test_fingerprints_and_caches_are_built_once_and_refreeze_is_idempotent(tmp_path):
    fingerprint_builder = _CountingFingerprintBuilder()
    score_cache_builder = _CountingScoreCacheBuilder()
    store, session, work_order_id, first = _frozen_session(
        tmp_path, fingerprint_builder=fingerprint_builder,
        score_cache_builder=score_cache_builder,
    )
    assert first.frozen is True
    assert fingerprint_builder.calls == 2  # once per usable block, never per pair
    assert score_cache_builder.calls == 2

    # Reading the pool again must not re-enter either builder.
    store.hybrid_pool(work_order_id)
    store.hybrid_pool(work_order_id)
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


def test_frozen_pool_refuses_scan_unscan_dismiss_and_recapture(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True

    outcome = store.scan_block(session.number, "33333333")
    assert outcome.accepted is False
    assert "closed" in outcome.message.lower()

    assert store.unscan_block(session.number, "11111111") is False

    with pytest.raises(ValueError, match="only a failed block can be dismissed"):
        store.dismiss_block(session.number, "11111111", reason="operator mistake")

    body = _capture_png(77)
    checksum = hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="unique unfilled set"):
        store.receive_capture(
            session.number, capture_id="cap-11111111-retry", block_id="11111111",
            checksum=checksum, body=body, recapture=True,
        )

    row = store.get_set(session.number, "11111111")
    assert row["preprocessing_status"] == "complete"
    assert row["dismissed_at"] is None


# ---------------------------------------------------------------------------
# Unknown session: never raises (mirrors try_enter_slides)
# ---------------------------------------------------------------------------


def test_freeze_on_unknown_session_returns_not_frozen_without_raising(tmp_path):
    """Raising ValueError("unknown session") here is exactly the danger
    `try_enter_slides` was designed to avoid -- `poll_drain` calls this on a background camera tick, and a raise
    there propagates through the RPC layer as a `transport.remote.StoreError`
    (a `ValueError` subclass) straight into `PiCaptureRuntime._camera_loop`'s
    `except Exception`, which kills the camera loop. `try_enter_slides`
    already returns `False` for an unknown session instead of raising;
    `freeze_hybrid_pool` must match that contract."""
    store = _make_store(tmp_path)

    result = store.freeze_hybrid_pool(999, descriptor_names=_DESCRIPTOR_NAMES)

    assert result.frozen is False
    assert result.usable_block_ids == ()


# ---------------------------------------------------------------------------
# Empty descriptor_names: refuse to freeze a pool with zero fingerprints.
# The loader-side rejection lives in
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
    # Negative control: no pool was ever frozen.
    assert store.hybrid_pool(work_order_id) is None
    # The work order remains usable, exactly like the <2-usable-blocks bounce.
    assert store.scan_block(session.number, "33333333").accepted


# ---------------------------------------------------------------------------
# Already-frozen branch repairs a stuck 'draining_blocks' phase on the
# work-order-keyed row.
# ---------------------------------------------------------------------------


def test_already_frozen_repairs_phase_stuck_in_draining_blocks(tmp_path):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    assert store.snapshot(session).phase == "slides"

    # Force phase back to 'draining_blocks' directly, bypassing the ordinary
    # phase-transition methods: poll_drain calls freeze_hybrid_pool every
    # second while draining, so the already-frozen branch must repair phase.
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
# Ledger replay stability of a bounce
# ---------------------------------------------------------------------------


def test_bounce_ledger_replay_returns_the_same_result(tmp_path):
    """`freeze_hybrid_pool` is in `_LEDGERED`; tests/test_idempotency_ledger.py
    documents that a replay must return the SAME response. A bounce must
    record a ledger row; otherwise a replay takes the
    `phase != 'draining_blocks'` branch and returns a DIFFERENT message
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
# hybrid_pool() read failures are loud, never silent
# ---------------------------------------------------------------------------


def _delete_archive(manifest_path, archive_path):
    archive_path.unlink()


def _truncate_archive(manifest_path, archive_path):
    body = archive_path.read_bytes()
    archive_path.write_bytes(body[: len(body) // 2])


def _bump_manifest_schema(manifest_path, archive_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = manifest["schema_version"] + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (_delete_archive, "archive"),
        (_truncate_archive, "archive"),
        (_bump_manifest_schema, "schema version"),
    ],
    ids=["missing-archive", "truncated-archive", "manifest-schema-mismatch"],
)
def test_hybrid_pool_raises_clearly_on_damaged_artifacts(tmp_path, corrupt, match):
    store, session, work_order_id, result = _frozen_session(tmp_path)
    assert result.frozen is True
    stem = session.directory / "work_orders" / f"work_order_{work_order_id:06d}_hybrid_pool"
    archive_path = stem.with_suffix(".npz")
    assert archive_path.is_file()
    corrupt(stem.with_suffix(".json"), archive_path)

    with pytest.raises(ValueError, match=match):
        store.hybrid_pool(work_order_id)


# ---------------------------------------------------------------------------
# Concurrent duplicate request_id: must not corrupt the archive or crash
# ---------------------------------------------------------------------------


def test_concurrent_duplicate_freeze_requests_do_not_corrupt_or_crash(tmp_path):
    """A retried duplicate request (e.g. an RPC client timeout retry) can run
    concurrently with the original attempt -- both threads race into
    `freeze_hybrid_pool` with the SAME request_id. The per-session lock must
    serialize them: no exception either thread's caller ever sees, both
    return the IDENTICAL result, and the archive is fully, uncorrupted
    readable afterward (a fixed staging filename would let a second
    concurrent writer keep writing into an inode the first had already
    `os.replace`'d into place)."""
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
# Legacy session-keyed hybrid_pools schema: migrated, never left broken
# ---------------------------------------------------------------------------


def test_legacy_session_keyed_hybrid_pools_table_is_migrated_not_left_broken(tmp_path):
    """A legacy `hybrid_pools` table has `session_number INTEGER PRIMARY KEY`
    and no `work_order_id` column. `CREATE TABLE IF NOT EXISTS` is a no-op
    against that shape, so without migration both `freeze_hybrid_pool` and
    `hybrid_pool` raise `sqlite3.OperationalError: no such column:
    work_order_id`, breaking the "Never raises" contract. Build the legacy
    shape by hand (dev-only data, nothing worth preserving), then prove a
    fresh `ProcessingStore` pointed at that file migrates cleanly and both
    methods work."""
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
# ---------------------------------------------------------------------------


def test_blocks_captured_before_start_work_order_are_adopted_into_the_pool(tmp_path):
    """`scan_block` stamps `sets.work_order_id` from whichever work order is
    open AT SCAN TIME -- NULL if none is. If `start_work_order` did not adopt
    those NULL rows, the freeze's `AND work_order_id=?` candidate filter
    would make them permanently invisible: two fully captured,
    quality-passing blocks would exist and the operator would be told
    "capture more blocks" with no way to recover (re-scanning fails
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
