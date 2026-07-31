"""Canonical SQLite session state and durable processing artifacts (#201 slice 4)."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from time import perf_counter_ns
import traceback
from typing import Callable, Mapping, Sequence
import cv2
import numpy as np

from capture_runtime import CAPTURE_STAGE_TIMING_KEYS, SETTLING_STAGE_KEYS
from contact_sheet import ContactSheetRenderer, write_contact_sheet
from session.pipeline import ClaimDecision, VERDICT_PASS, VERDICT_REVIEW, decide_claim
from session.preparation import (
    PreparationFailure,
    PreparedResult,
    PreparedSpecimen,
    prepare_specimen,
    prepare_specimen_from_image,
)
from runtime_observer import RuntimeObserver, observed
from session.session_mode import SessionMode
from verify.candidate_band import (
    CandidateSelection,
    SpecimenFingerprint,
    select_candidate_band,
    select_configured_candidate_band,
    validate_selection,
)
from verify.gates import _check_mask_quality, check_block_quality
from verify.invariant_descriptors import DescriptorValue, build_descriptor_values
from verify.scorer import (
    ProductionScoreResult,
    LockedScoreCache,
    _ComponentFeatures,
    build_locked_score_cache,
    score_routed_caches,
)
from verify.slide_image_overlay import build_slide_image_overlay
from slide.qr import SlideQRResult
import store.wire as store_wire
from session.atomic_io import (
    atomic_bytes as _atomic_bytes,
    atomic_json as _atomic_json,
    sha256 as _sha256,
)
from session.outbox_transport import _slide_capture_id
from session.workflow_types import (
    BlockReadiness,
    ClaimOutcome,
    FailedBlockWarning,
    HybridPoolFreezeResult,
    RecaptureOutcome,
    ScanOutcome,
    SessionIdentity,
    SessionSummary,
    UploadReceipt,
    WorkOrderScoringResult,
    WorkflowEvent,
    WorkflowSnapshot,
    normalize_work_order_scoring_result as _normalize_work_order_scoring_result,
)
from verify.work_order_evaluator import WorkOrderVerdict, evaluate_work_order, flagged_pairs
from session.matching_corpus import (
    ScoredPair,
    TruePairRef,
    expand_same_work_order_candidates,
    make_pair_id,
    promote_near_misses,
)


# #269: durable Hybrid Candidate Pool artifact names, one pair per WORK
# ORDER (not per session -- #250 was a deliberate v1 simplification the
# CREATE TABLE comment below explains), living under
# ``<session_dir>/work_orders/`` alongside the existing
# ``work_order_{id:06d}_verdicts.csv``/``_sheets`` artifacts. Schema-versioned
# so a future format change can be detected on read rather than guessed;
# bumped to 2 for #269's manifest shape change (work_order_id/session_number
# now recorded in the manifest body, not just the DB row).
_HYBRID_POOL_SCHEMA_VERSION = 3
_HYBRID_POOL_MANIFEST_SUFFIX = "_hybrid_pool.json"
_HYBRID_POOL_ARCHIVE_SUFFIX = "_hybrid_pool.npz"

# #269: the durable string values sessions.session_mode is allowed to hold --
# exactly SessionMode's own values, so start_session rejects a typo/garbage
# mode loudly at session-creation time (a startup-shaped call, not a
# per-poll/per-tick path) rather than persisting a value nothing else
# recognizes.
_VALID_SESSION_MODES = frozenset(mode.value for mode in SessionMode)


@dataclass(frozen=True)
class HybridCandidatePool:
    """The immutable, durable Hybrid Candidate Pool for one frozen WORK ORDER
    (#250, re-keyed per-work-order by #269).

    Process-local only -- never crosses the Pi/``RemoteProcessingStore`` wire
    (fingerprints/score caches are ``numpy`` arrays, and ``store.wire``
    deliberately refuses to serialize those), so this is not a
    ``store.wire``-registered type. It is read by ``ProcessingStore.hybrid_pool``
    for processing-computer-local consumption (the future #251 per-slide job);
    nothing that reads it recomputes ``fingerprints``/``score_caches``.
    """

    work_order_id: int
    session_number: int
    frozen_at: str
    block_ids: tuple[str, ...]
    descriptor_names: tuple[str, ...]
    candidate_configuration: Mapping[str, object] | None
    fingerprints: Mapping[str, Mapping[str, np.ndarray]]
    score_caches: Mapping[str, LockedScoreCache]


def _save_hybrid_pool_archive(
    path: Path,
    *,
    block_ids: Sequence[str],
    descriptor_names: Sequence[str],
    fingerprints: Mapping[str, Mapping[str, np.ndarray]],
    score_caches: Mapping[str, LockedScoreCache],
) -> None:
    """Durably persist every frozen block's cache + fingerprint vectors once.

    One compressed archive for the whole pool (not one file per block): the
    pool is owned by the work order, not by any individual block or pair
    (#250 acceptance criterion), so one artifact mirrors that ownership.
    """
    arrays: dict[str, np.ndarray] = {}
    for block_id in block_ids:
        cache = score_caches[block_id]
        arrays[f"{block_id}::mask"] = cache.normalized_mask
        arrays[f"{block_id}::points"] = cache.component_features.points
        arrays[f"{block_id}::areas"] = cache.component_features.areas
        arrays[f"{block_id}::shapes"] = cache.component_features.shapes
        for name in descriptor_names:
            arrays[f"{block_id}::fp::{name}"] = fingerprints[block_id][name]
    buffer = io.BytesIO()
    # `allow_pickle=False` is explicit (numpy's own default already matches):
    # it is a real keyword-only parameter between `*args` and `**kwds` in
    # `savez_compressed`'s signature, so naming it here -- rather than
    # leaving it implicit -- means a future `arrays` key that ever collided
    # with the literal string "allow_pickle" would raise a loud
    # "multiple values for keyword argument" TypeError instead of silently
    # overriding pickle safety with a non-bool ndarray. Every value in
    # `arrays` is a plain numeric ndarray, so pickling is never needed.
    np.savez_compressed(buffer, allow_pickle=False, **arrays)
    _atomic_bytes(path, buffer.getvalue())


def _load_hybrid_pool_archive(
    path: Path, *, block_ids: Sequence[str], descriptor_names: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, LockedScoreCache]]:
    """Read back one pool archive with zero recomputation (#250 durability)."""
    fingerprints: dict[str, dict[str, np.ndarray]] = {}
    score_caches: dict[str, LockedScoreCache] = {}
    with np.load(path) as archive:
        for block_id in block_ids:
            score_caches[block_id] = LockedScoreCache(
                normalized_mask=archive[f"{block_id}::mask"],
                component_features=_ComponentFeatures(
                    points=archive[f"{block_id}::points"],
                    areas=archive[f"{block_id}::areas"],
                    shapes=archive[f"{block_id}::shapes"],
                ),
            )
            fingerprints[block_id] = {
                name: archive[f"{block_id}::fp::{name}"] for name in descriptor_names
            }
    return fingerprints, score_caches


def _pool_specimen_fingerprint(
    pool: "HybridCandidatePool", block_id: str,
) -> SpecimenFingerprint:
    """Build one frozen pool block's #253 retrieval fingerprint for FREE.

    ``pool.fingerprints``/``pool.score_caches`` already hold everything
    needed -- the descriptor vectors and normalized mask #250 built exactly
    once at freeze time and durably persisted (``hybrid_pools`` row +
    ``.npz`` archive). This is pure reshaping of already-computed,
    already-loaded ``numpy`` arrays into ``SpecimenFingerprint``'s shape: no
    cv2, no I/O, no descriptor recomputation. ``construction_ns=0`` is a
    placeholder (the real per-descriptor build duration was only meaningful
    once, at freeze time, and is not persisted); nothing #253 reads consults
    it (`compare_descriptor_values` only reads ``.vector``).
    """
    return SpecimenFingerprint(
        specimen_id=block_id,
        occupied_fraction=float(pool.score_caches[block_id].normalized_mask.mean()),
        descriptor_values={
            name: DescriptorValue(vector=vector, construction_ns=0)
            for name, vector in pool.fingerprints[block_id].items()
        },
    )


def _slide_specimen_fingerprint(
    fingerprint_builder: Callable[[np.ndarray], Mapping[str, DescriptorValue]],
    slide_cache: LockedScoreCache,
) -> SpecimenFingerprint:
    """Build one slide's #253 retrieval fingerprint.

    Unlike ``_pool_specimen_fingerprint``, this genuinely computes something
    new each job: a slide is never frozen ahead of time, so its descriptor
    vectors cannot be precomputed. Reuses the SAME injectable
    ``fingerprint_builder`` seam #250's ``freeze_hybrid_pool`` already calls
    for every pool block, on the same normalized mask shape, rather than a
    second code path.
    """
    return SpecimenFingerprint(
        specimen_id="slide",
        occupied_fraction=float(slide_cache.normalized_mask.mean()),
        descriptor_values=fingerprint_builder(slide_cache.normalized_mask),
    )


# One joined, operator-facing row per normal-mode slide when --profile is on.
SLIDE_BENCHMARK_COLUMNS = (
    "trigger_to_presence_ms", "settling_ms", "camera_capture_ms", "publish_ms",
    "qr_decode_ms", "outbox_write_ms", "transfer_wait_ms", "receive_persist_ms",
    "identity_lookup_ms", "slide_preparation_ms", "quality_gates_ms",
    "locked_nm_preparation_ms", "alignment_scoring_ms", "qc_render_ms",
    "verdict_commit_export_ms", "retake_count", "full_total_ms",
)

# One PC-local readiness row per block when the Pi marks that capture as
# profiled. Durations begin at receiver acceptance, so they never compare the
# Pi's clock with the processing computer's clock.
BLOCK_BENCHMARK_COLUMNS = (
    "queue_wait_ms", "block_preparation_ms", "segmentation_ms",
    "artifact_write_ms", "ready_after_receive_ms", "status",
)

_PROFILE_STAGE_NAMES = {
    "quality_gates": "quality_gates_ms",
    "locked_cache": "locked_nm_preparation_ms",
    "alignment_scoring": "alignment_scoring_ms",
}


class _SlideBenchmarkObserver:
    """Adapt existing scoring observations to one live slide benchmark row."""

    def __init__(self, store: "ProcessingStore", capture_id: str) -> None:
        self._store = store
        self._capture_id = capture_id

    def record(self, stage: str, elapsed_ns: int, item_id: str) -> None:
        del item_id
        column = _PROFILE_STAGE_NAMES.get(stage)
        if column is not None:
            with self._store._slide_profile_lock:
                self._store._slide_profile_stages.setdefault(self._capture_id, {})[
                    column
                ] = int(round(elapsed_ns / 1_000_000))


class ProcessingStore:
    """Canonical SQLite state and durable processing-computer artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        preprocessor: Callable[
            [Path], tuple[np.ndarray, Mapping[str, object]]
        ] | None = None,
        slide_preprocessor: Callable[[np.ndarray], PreparedResult] | None = None,
        work_order_scorer: Callable[
            [Mapping[str, PreparedResult], Mapping[str, PreparedResult]],
            WorkOrderScoringResult | Mapping[str, Mapping[str, float | None]],
        ] | None = None,
        contact_sheet_renderer: ContactSheetRenderer | None = None,
        fingerprint_builder: Callable[
            [np.ndarray], Mapping[str, DescriptorValue]
        ] | None = None,
        score_cache_builder: Callable[[PreparedSpecimen], LockedScoreCache] | None = None,
        workers: int = 1,
        recover_jobs: bool = True,
        runtime_observer: RuntimeObserver | None = None,
        profile_clock_ns: Callable[[], int] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "sessions.sqlite3"
        self.preprocessor = preprocessor or preprocess_block
        self._uses_default_block_preprocessor = preprocessor is None
        self.slide_preprocessor = slide_preprocessor or preprocess_slide
        self.work_order_scorer = work_order_scorer or default_work_order_scorer
        # #250: injectable seams for the Hybrid Candidate Pool freeze, mirroring
        # `preprocessor`/`slide_preprocessor` above -- tests inject a counting
        # wrapper to prove fingerprints/caches are built exactly once per
        # usable block, never re-entered on a later read.
        self.fingerprint_builder = fingerprint_builder or build_descriptor_values
        self.score_cache_builder = score_cache_builder or build_locked_score_cache
        self._contact_sheet_renderer = contact_sheet_renderer or write_contact_sheet
        self.runtime_observer = runtime_observer
        self._slide_profile_stages: dict[str, dict[str, int]] = {}
        self._slide_profile_lock = Lock()
        self._block_profile_lock = Lock()
        self._profile_clock_ns = profile_clock_ns or perf_counter_ns
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._jobs: list[Future] = []
        self._jobs_lock = Lock()
        self._events: dict[int, list[WorkflowEvent]] = {}
        self._events_lock = Lock()
        # #250 review F4: serializes the ENTIRE freeze_hybrid_pool body per
        # session. Freeze does multi-second file/CV work -- well past the
        # RPC client's 10s timeout -- so a retried duplicate request can
        # otherwise run concurrently with the original attempt, corrupting
        # the shared archive staging path and racing the ledger insert. One
        # lock per session (mirrors `_jobs_lock`'s shape); `_hybrid_freeze_
        # locks_guard` only protects the dict's own get-or-create.
        self._hybrid_freeze_locks: dict[int, Lock] = {}
        self._hybrid_freeze_locks_guard = Lock()
        # #250 review F1/F5: `_emit` alone is in-memory-only (lost on
        # restart); a handful of call sites additionally need a durable,
        # on-disk trace of "why" (a Finish-Blocks bounce reason, a swallowed
        # per-block exception) that survives a crash/restart.
        self._durable_log_lock = Lock()
        # Advisory, non-durable dedup for `record_event` retries: events are
        # already lost on restart, so an in-memory set (rather than a
        # `request_ledger` row) is sufficient to collapse "same request_id
        # twice" into one append.
        self._seen_event_request_ids: set[str] = set()
        self._initialize()
        if recover_jobs:
            self._recover_jobs()
            self._recover_work_orders()
        self._recover_claims()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            # #269 FIX2: `hybrid_pools` was ALREADY COMMITTED (df67237) with a
            # `session_number INTEGER PRIMARY KEY` shape and no
            # `work_order_id` column at all -- `CREATE TABLE IF NOT EXISTS`
            # below is a no-op against a database file that already has that
            # legacy table, so a real pre-#269 DB would raise
            # `sqlite3.OperationalError: no such column: work_order_id` out
            # of every `freeze_hybrid_pool`/`hybrid_pool` call forever,
            # contradicting `freeze_hybrid_pool`'s own "Never raises"
            # contract. #269 is dev-only data with nothing worth
            # preserving (the old rows are session-keyed v1 manifests that
            # cannot be losslessly converted to per-work-order), so this is a
            # straight drop-and-recreate rather than a copy-forward migration
            # like the `request_ledger` PK rebuild below. `PRAGMA
            # table_info` on a table that does not exist yet returns zero
            # rows (no error), so a fresh database skips this entirely.
            legacy_hybrid_pool_columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(hybrid_pools)").fetchall()
            }
            if (
                legacy_hybrid_pool_columns
                and "work_order_id" not in legacy_hybrid_pool_columns
            ):
                db.execute("DROP TABLE hybrid_pools")
            db.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    session_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL UNIQUE,
                    phase TEXT NOT NULL,
                    slide_recovery_state TEXT NOT NULL DEFAULT 'waiting',
                    finalized_at TEXT,
                    last_finalization_error TEXT
                );
                CREATE TABLE IF NOT EXISTS sets (
                    session_number INTEGER NOT NULL,
                    block_id TEXT NOT NULL,
                    capture_id TEXT,
                    capture_path TEXT,
                    checksum TEXT,
                    preprocessing_status TEXT NOT NULL DEFAULT 'awaiting_capture',
                    profile_enabled INTEGER NOT NULL DEFAULT 0,
                    profile_queued_ns INTEGER,
                    preprocessing_metadata TEXT,
                    mask_path TEXT,
                    qc_path TEXT,
                    failure_reason TEXT,
                    dismissed_at TEXT,
                    unusable_reason TEXT,
                    slide_capture_id TEXT,
                    verdict TEXT,
                    score REAL,
                    decision_stage TEXT,
                    decision_reason TEXT,
                    decided_at TEXT,
                    PRIMARY KEY (session_number, block_id),
                    UNIQUE (capture_id),
                    FOREIGN KEY (session_number) REFERENCES sessions(session_number)
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    capture_id TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    session_number INTEGER NOT NULL,
                    block_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS request_ledger (
                    request_id          TEXT NOT NULL,
                    method               TEXT NOT NULL,
                    session_number       INTEGER NOT NULL,
                    request_fingerprint  TEXT NOT NULL,
                    response_json        TEXT NOT NULL,
                    status               TEXT NOT NULL,
                    recorded_at          TEXT NOT NULL,
                    PRIMARY KEY (session_number, method, request_id)
                );
                CREATE TABLE IF NOT EXISTS slide_captures (
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
                    -- Legacy physical name for the decoded laboratory work
                    -- order. Keep it for durable-session compatibility; code
                    -- exposes it as lab_work_order where possible. It is
                    -- distinct from work_order_id, the local capture/scoring
                    -- bracket record.
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
                    FOREIGN KEY (session_number) REFERENCES sessions(session_number)
                );
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_number INTEGER NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    verdict_csv_path TEXT,
                    contact_sheet_dir TEXT,
                    failure_reason TEXT,
                    FOREIGN KEY (session_number) REFERENCES sessions(session_number)
                );
                -- #269: keyed by work_order_id (the durably unique bracket
                -- that froze it), NOT session_number -- #250 shipped a
                -- deliberate v1 simplification (session_number PRIMARY KEY)
                -- that was only safe while Hybrid could not open a second
                -- work_orders bracket. This is that re-key: work_order_id is
                -- globally unique (work_orders.work_order_id is an
                -- AUTOINCREMENT primary key, not scoped per session), so a
                -- pool is physically addressed by the work order that froze
                -- it -- this is the isolation invariant's structural
                -- enforcement point, not a convention a caller must
                -- remember. session_number remains an ordinary column for
                -- convenient joins/lookups only.
                CREATE TABLE IF NOT EXISTS hybrid_pools (
                    work_order_id INTEGER PRIMARY KEY,
                    session_number INTEGER NOT NULL,
                    frozen_at TEXT NOT NULL,
                    block_ids TEXT NOT NULL,
                    descriptor_names TEXT NOT NULL,
                    candidate_configuration TEXT,
                    manifest_path TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id),
                    FOREIGN KEY (session_number) REFERENCES sessions(session_number)
                );
                CREATE TABLE IF NOT EXISTS matching_pairs (
                    pair_id TEXT PRIMARY KEY,
                    session_number INTEGER NOT NULL,
                    work_order TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    slide_capture_id TEXT NOT NULL,
                    pair_source TEXT NOT NULL,
                    is_match INTEGER,
                    classical_score REAL,
                    rank_for_block INTEGER,
                    metric TEXT,
                    scored_at TEXT,
                    FOREIGN KEY (session_number) REFERENCES sessions(session_number)
                );
                CREATE INDEX IF NOT EXISTS matching_pairs_wo
                    ON matching_pairs(session_number, work_order);
                """
            )
            ledger_pk = [
                row["name"]
                for row in sorted(
                    db.execute("PRAGMA table_info(request_ledger)").fetchall(),
                    key=lambda row: row["pk"],
                )
                if row["pk"]
            ]
            if ledger_pk == ["request_id"]:
                db.executescript(
                    """
                    ALTER TABLE request_ledger RENAME TO request_ledger_legacy;
                    CREATE TABLE request_ledger (
                        request_id          TEXT NOT NULL,
                        method               TEXT NOT NULL,
                        session_number       INTEGER NOT NULL,
                        request_fingerprint  TEXT NOT NULL,
                        response_json        TEXT NOT NULL,
                        status               TEXT NOT NULL,
                        recorded_at          TEXT NOT NULL,
                        PRIMARY KEY (session_number, method, request_id)
                    );
                    INSERT INTO request_ledger
                    SELECT * FROM request_ledger_legacy;
                    DROP TABLE request_ledger_legacy;
                    """
                )
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(sets)").fetchall()
            }
            for name, decl in (
                ("failure_reason", "TEXT"),
                ("dismissed_at", "TEXT"),
                ("unusable_reason", "TEXT"),
                ("slide_capture_id", "TEXT"),
                ("verdict", "TEXT"),
                ("score", "REAL"),
                ("decision_stage", "TEXT"),
                ("decision_reason", "TEXT"),
                ("decided_at", "TEXT"),
                ("work_order_id", "INTEGER REFERENCES work_orders(work_order_id)"),
                ("profile_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("profile_queued_ns", "INTEGER"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE sets ADD COLUMN {name} {decl}")
            session_columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "slide_recovery_state" not in session_columns:
                db.execute(
                    "ALTER TABLE sessions ADD COLUMN slide_recovery_state "
                    "TEXT NOT NULL DEFAULT 'waiting'"
                )
            for name, decl in (
                ("finalized_at", "TEXT"),
                ("last_finalization_error", "TEXT"),
                # #269: durable session scoring mode, written once at
                # start_session so finish_work_order/_recover_work_orders can
                # read it directly instead of inferring it from whether a
                # hybrid_pools row happens to exist (see the design-gap note
                # on finish_work_order below). sessions is a long-standing
                # table with real committed history, so this follows the
                # additive ALTER TABLE convention already used for
                # slide_recovery_state/finalized_at/last_finalization_error --
                # unlike hybrid_pools (brand-new, CREATE TABLE rewritten
                # outright above). Every pre-#269 row defaults to 'normal',
                # which correctly answers "is this hybrid/hybrid_shadow?"
                # (the only question this column controls) regardless of
                # which pre-Hybrid mode a given historical row actually was.
                ("session_mode", "TEXT NOT NULL DEFAULT 'normal'"),
            ):
                if name not in session_columns:
                    db.execute(f"ALTER TABLE sessions ADD COLUMN {name} {decl}")
            hybrid_pool_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(hybrid_pools)").fetchall()
            }
            if "candidate_configuration" not in hybrid_pool_columns:
                db.execute("ALTER TABLE hybrid_pools ADD COLUMN candidate_configuration TEXT")
            slide_capture_columns = {
                row["name"]
                for row in db.execute(
                    "PRAGMA table_info(slide_captures)"
                ).fetchall()
            }
            for name, decl in (
                ("verdict", "TEXT"),
                ("claim_score", "REAL"),
                ("claim_stage", "TEXT"),
                ("claim_reason", "TEXT"),
                ("claim_qc_path", "TEXT"),
                ("claim_decided_at", "TEXT"),
                ("work_order_id", "INTEGER REFERENCES work_orders(work_order_id)"),
                ("top_block", "TEXT"),
                ("near_miss_blocks", "TEXT"),
                # Every Hybrid score must remain inspectable after the
                # worker finishes.  `matching_pairs` holds the individual
                # scored candidates; these columns retain the ranking summary
                # used by `evaluate_work_order` without asking a later reader
                # to reconstruct it from a mutable pool or a display string.
                # NULL means no ranked comparison was possible (for example a
                # gate-failed slide), never a synthetic zero score.
                ("top_score", "REAL"),
                ("runner_up_score", "REAL"),
                ("match_margin", "REAL"),
                # #252/ADR 0017: the durable retrieval per-slide job queue.
                # already existed in committed databases before this column, so
                # (mirroring every other additive ALTER above -- NOT the
                # `hybrid_pools` drop-and-recreate, which was only safe because
                # that table was dev-only data) this is an additive ALTER, not
                # a rewritten CREATE TABLE; a bare `CREATE TABLE IF NOT EXISTS`
                # would silently no-op against a real pre-existing DB and the
                # Pi would crash on the first queued retrieval capture.
                # slice only ever write 'queued'/'preparing'/'scoring'/
                # 'complete'/'error' (see `_score_hybrid_slide`); 'superseded'
                # is reserved for #256 and is never written here. NULL means
                # "not a retrieval job" -- every NORMAL row, Hybrid out-of-pool
                # row, and every pre-queue row -- and that must stay the read
                # signal for "no background job exists for this slide".
                ("job_state", "TEXT"),
                # #253: durable audit evidence for one slide's Heuristic
                # Candidate Band selection (`_persist_candidate_selection`).
                # Additive ALTER, same reasoning as `job_state` immediately
                # above: `slide_captures` already has real committed rows, so
                # a bare `CREATE TABLE IF NOT EXISTS` would silently no-op and
                # every write to this column would raise `OperationalError:
                # no such column` on a real pre-#253 database. NULL means "no
                # selection was ever recorded for this slide" -- every
                # NORMAL/OPEN_RETRIEVAL row, and every pre-#253 Hybrid row.
                ("candidate_selection_json", "TEXT"),
                # #254: durable Hybrid Shadow safety-comparison evidence
                # (`_persist_shadow_comparison`) -- the complete-pool verdict
                # vs. the proposed-Hybrid-subset verdict, both match margins,
                # and the candidate/pruned id sets, for one hybrid_shadow
                # slide job. Additive ALTER, same reasoning as `job_state`/
                # `candidate_selection_json` immediately above: `slide_captures`
                # already has real committed rows, so a bare `CREATE TABLE IF
                # NOT EXISTS` would silently no-op and the first
                # `_persist_shadow_comparison` write would raise
                # `OperationalError: no such column` on a real pre-#254
                # database. NULL means "not a hybrid_shadow slide, a
                # gate-failed slide (nothing was scored to compare), or a
                # comparison-persistence failure that was deliberately
                # degraded rather than turned into job_state='error'" --
                # every NORMAL/OPEN_RETRIEVAL/plain-Hybrid row, and every
                # pre-#254 row. Operator-hidden: nothing in `code/kiosk/` or
                # `_refresh_decisions_export`'s CSV reads this column -- it
                # is analysis evidence, not a Results field.
                ("shadow_comparison_json", "TEXT"),
                # #255: durable scheduling-order key for the Hybrid per-slide
                # job queue. NULL (every pre-#255 row, and every ordinary
                # queued job) means "ordinary FIFO" -- ordered only by
                # captured_at/capture_id, exactly today's behavior. A NON-
                # NULL value sorts ahead of every NULL row, smallest first;
                # #256 is the first real writer (an accepted recapture gets a
                # lower value than any ordinary job, e.g. 0), so recaptures
                # run ahead of the ordinary FIFO queue both live and after a
                # restart (`_recover_retrieval_jobs` orders resubmission by
                # this same column). Nothing in this slice ever writes a
                # non-NULL value; this is the durable column #256 needs, not
                # scaffolding to be removed later. Additive ALTER, same
                # reasoning as `job_state`/`candidate_selection_json`/
                # `shadow_comparison_json` immediately above.
                ("priority", "INTEGER"),
                # #258: durable per-slide Hybrid worker timing, mirroring the
                # `sets.profile_enabled`/`sets.profile_queued_ns` block
                # precedent exactly. Additive ALTER, same reasoning as every
                # other Hybrid column above: `slide_captures` already has real
                # committed rows, so a bare `CREATE TABLE IF NOT EXISTS` would
                # silently no-op and the first write would raise
                # `OperationalError: no such column` on a real pre-#258
                # database.
                #
                # `profile_enabled` is 0 for every row this slice does not
                # itself set to 1 -- every NORMAL/OPEN_RETRIEVAL/out-of-pool
                # row, and every pre-#258 Hybrid row -- so `profile_enabled=1`
                # is the one gate `list_hybrid_profile_rows` filters on: no
                # `--profile` means no row is ever collected or exposed.
                # `profile_queued_ns` is this row's enqueue timestamp (taken
                # via the injectable `self._profile_clock_ns`, never a raw
                # clock read) and is NULL whenever `profile_enabled=0`.
                # `profile_current_stage` is the one of `PROFILE_STAGE_ORDER`
                # (`code/session/profile_report.py`) the WORKER is currently
                # inside while this row is still pending; NULL once the row
                # is no longer pending or was never profiled.
                # `profile_stage_ms_json` is the completed row's five-stage
                # breakdown (a JSON object keyed by `PROFILE_STAGE_ORDER`
                # names; a stage that never ran, e.g. a gate-failed slide's
                # selection/scoring, is simply absent as a key, never a
                # fabricated zero) and `profile_total_ms` is that same row's
                # queue-to-complete total; both stay NULL until the worker
                # finishes, and forever NULL when `profile_enabled=0`.
                # `profile_shadow` is 1 for a `hybrid_shadow` row, 0 for plain
                # Hybrid, NULL when `profile_enabled=0` -- written once at
                # enqueue from the session's own durable `session_mode`
                # (which cannot change mid-session), so a shadow row's
                # complete-pool timing can never be mistaken for pruned
                # Hybrid timing in the PERSISTED data, not only on screen.
                ("profile_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("profile_queued_ns", "INTEGER"),
                ("profile_current_stage", "TEXT"),
                ("profile_stage_ms_json", "TEXT"),
                ("profile_total_ms", "INTEGER"),
                ("profile_shadow", "INTEGER"),
            ):
                if name not in slide_capture_columns:
                    db.execute(
                        f"ALTER TABLE slide_captures ADD COLUMN {name} {decl}"
                    )

    def start_session(
        self, *, started_at: datetime, session_mode: str = "normal",
        request_id: str | None = None,
    ) -> SessionIdentity:
        if started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if session_mode not in _VALID_SESSION_MODES:
            raise ValueError(
                f"session_mode must be one of {sorted(_VALID_SESSION_MODES)}, "
                f"got {session_mode!r}"
            )
        utc = started_at.astimezone(timezone.utc)
        # Sessions have no session_number of their own yet (this call MINTS
        # one), so the ledger is keyed on the placeholder session_number 0,
        # which real sessions (autoincrement from 1) never collide with.
        # #269: session_mode is hashed alongside started_at so a replayed
        # request_id with a DIFFERENT mode raises, mirroring the existing
        # started_at-mismatch-raises contract.
        fingerprint = (
            self._fingerprint(
                {"started_at": utc.isoformat(), "session_mode": session_mode}
            )
            if request_id is not None else None
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "start_session", 0, fingerprint
                )
                if hit:
                    return store_wire.loads_as(SessionIdentity, cached)
            cursor = db.execute(
                "INSERT INTO sessions(started_at, phase, session_mode) "
                "VALUES (?, 'blocks', ?)",
                (utc.isoformat(), session_mode),
            )
            number = int(cursor.lastrowid)
            directory = self.root / f"session_{number:06d}_{utc:%Y%m%dT%H%M%SZ}"
            identity = SessionIdentity(number, utc, directory, session_mode)
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "start_session", 0, fingerprint,
                    store_wire.dumps(identity),
                )
        # Fresh path only: a crash here (before mkdir) leaves the session row
        # durable but the directory/event missing -- no worse than today's
        # status quo (a dropped response before this ledger existed could
        # already double the session), and a genuine retry with the same
        # request_id always ran mkdir/emit before its response was lost, so
        # replay never re-runs them.
        directory.mkdir()
        _atomic_json(
            directory / "session.json",
            {"session_number": number, "started_at": utc.isoformat(), "phase": "blocks"},
        )
        self._emit(number, "session_started", "Session started")
        return identity

    def _session_mode(self, session_number: int) -> str:
        """Read the persisted ``sessions.session_mode`` for one session.

        #269: the durable fact ``finish_work_order``/``_recover_work_orders``
        gate the N-by-N scoring path on, instead of inferring Hybrid-ness from
        whether a ``hybrid_pools`` row happens to exist (see the design-gap
        note on ``finish_work_order``).

        #269 FIX5d: fails CLOSED (raises) for an unknown session number,
        rather than defaulting to ``'normal'``. Defaulting to ``'normal'``
        here is a safety-gate failure-OPEN bug: this is the one fact
        ``finish_work_order`` uses to decide whether a work order may run
        the full N-by-N scoring path, so silently guessing "not Hybrid" for
        a session this can't find is exactly backwards -- it would let a
        Hybrid work order whose session row is somehow missing fall through
        to full N-by-N scoring instead of being safely excluded from it. The
        sole caller, ``finish_work_order``, always resolves ``session_number``
        moments earlier from an already-open ``work_orders`` row, and
        ``work_orders.session_number`` is FK-constrained to ``sessions``, so
        this raise is unreachable on that live path today; it exists purely
        as a fail-closed backstop against a future caller or a corrupted
        database, not a case this module expects to hit.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT session_mode FROM sessions WHERE session_number=?",
                (session_number,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown session: {session_number}")
        return str(row["session_mode"])

    def resume_session(self, session_number: int | None = None) -> SessionIdentity:
        with self._connect() as db:
            if session_number is None:
                row = db.execute(
                    "SELECT session_number FROM sessions ORDER BY session_number DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    raise ValueError("no session is available to resume")
                session_number = int(row["session_number"])
        identity = self._session_identity(session_number)
        self.reconcile_session_metadata(session_number)
        return identity

    @staticmethod
    def _fingerprint(args: Mapping[str, object]) -> str:
        """sha256 over a stable JSON rendering of one method's logical args.

        Used to reject `request_id` reuse against DIFFERENT arguments (a
        client bug), while letting a byte-identical retry replay cleanly.
        """
        canonical = json.dumps(args, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _ledger_hit(
        db: sqlite3.Connection,
        request_id: str,
        method: str,
        session_number: int,
        fingerprint: str,
    ) -> tuple[bool, str | None]:
        """Look up `request_id` inside the caller's already-open transaction.

        Returns `(True, response_json)` on a replay, `(False, None)` on a
        genuinely new request. Raises `ValueError` if the SAME request_id was
        previously recorded against DIFFERENT arguments.
        """
        row = db.execute(
            "SELECT request_fingerprint, response_json, status FROM request_ledger "
            "WHERE session_number=? AND method=? AND request_id=?",
            (session_number, method, request_id),
        ).fetchone()
        if row is None:
            return False, None
        if row["request_fingerprint"] != fingerprint:
            raise ValueError(
                "request_id was already used with different request arguments"
            )
        return row["status"] == "ok", row["response_json"]

    @staticmethod
    def _ledger_record(
        db: sqlite3.Connection,
        request_id: str,
        method: str,
        session_number: int,
        fingerprint: str,
        response_json: str,
        *,
        status: str = "ok",
    ) -> None:
        """Durably record one original response inside the caller's transaction.

        Collision-tolerant (#250 review F4): a concurrent duplicate-
        ``request_id`` retry that raced past its own ``_ledger_hit`` check
        (e.g. a slow mutating call outliving the RPC client's timeout) can
        try to insert this exact ``(session_number, method, request_id)`` row
        a second time. Losing that race must never crash the RPC server --
        the winner's row is already the durable record, so a PK collision
        here is swallowed rather than propagated.
        """
        try:
            db.execute(
                """INSERT INTO request_ledger(
                   request_id, method, session_number, request_fingerprint,
                   response_json, status, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, method, session_number, fingerprint, response_json,
                    status, datetime.now(timezone.utc).isoformat(),
                ),
            )
        except sqlite3.IntegrityError:
            pass

    def scan_block(
        self, session_number: int, block_id: str, *, request_id: str | None = None
    ) -> ScanOutcome:
        if not (len(block_id) == 8 and block_id.isascii() and block_id.isdigit()):
            return ScanOutcome(False, "Block ID must contain eight numeric digits")
        fingerprint = (
            self._fingerprint({"block_id": block_id}) if request_id is not None else None
        )
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if request_id is not None:
                    hit, cached = self._ledger_hit(
                        db, request_id, "scan_block", session_number, fingerprint
                    )
                    if hit:
                        return store_wire.loads_as(ScanOutcome, cached)
                phase = db.execute(
                    "SELECT phase FROM sessions WHERE session_number=?",
                    (session_number,),
                ).fetchone()
                if phase is None:
                    raise ValueError("unknown session")
                if phase["phase"] != "blocks":
                    outcome = ScanOutcome(False, "Block scanning is closed")
                    if request_id is not None:
                        self._ledger_record(
                            db, request_id, "scan_block", session_number,
                            fingerprint, store_wire.dumps(outcome),
                        )
                    return outcome
                open_work_order = db.execute(
                    """SELECT work_order_id FROM work_orders
                       WHERE session_number=? AND lifecycle_state='capturing'
                       ORDER BY work_order_id DESC LIMIT 1""",
                    (session_number,),
                ).fetchone()
                work_order_id = (
                    int(open_work_order["work_order_id"])
                    if open_work_order is not None else None
                )
                db.execute(
                    "INSERT INTO sets(session_number, block_id, work_order_id) "
                    "VALUES (?, ?, ?)",
                    (session_number, block_id, work_order_id),
                )
                outcome = ScanOutcome(True, f"Accepted block {block_id}")
                if request_id is not None:
                    self._ledger_record(
                        db, request_id, "scan_block", session_number,
                        fingerprint, store_wire.dumps(outcome),
                    )
        except sqlite3.IntegrityError:
            self._emit(
                session_number, "duplicate_block_scan", "Block already scanned", block_id
            )
            return ScanOutcome(False, "Block already scanned")
        self._emit(session_number, "block_scanned", "Block scan accepted", block_id)
        return outcome

    def awaiting_capture_blocks(self, session_number: int) -> tuple[str, ...]:
        """Return accepted block scans that do not yet have a capture."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT block_id FROM sets
                   WHERE session_number=? AND preprocessing_status='awaiting_capture'
                   ORDER BY rowid""",
                (session_number,),
            ).fetchall()
        return tuple(str(row["block_id"]) for row in rows)

    def unscan_block(
        self, session_number: int, block_id: str, *, request_id: str | None = None
    ) -> bool:
        """Remove an accepted scan only while no capture is attached to it."""
        fingerprint = (
            self._fingerprint({"block_id": block_id}) if request_id is not None else None
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "unscan_block", session_number, fingerprint
                )
                if hit:
                    return bool(json.loads(cached))
            deleted = db.execute(
                """DELETE FROM sets
                   WHERE session_number=? AND block_id=? AND capture_id IS NULL""",
                (session_number, block_id),
            ).rowcount
            result = deleted == 1
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "unscan_block", session_number,
                    fingerprint, json.dumps(result),
                )
        return result

    def begin_block_drain(self, session_number: int) -> None:
        changed = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown session")
            if row["phase"] == "blocks":
                db.execute(
                    "UPDATE sessions SET phase='draining_blocks' WHERE session_number=?",
                    (session_number,),
                )
                changed = True
        if changed:
            self.reconcile_session_metadata(session_number)
            self._emit(
                session_number, "block_drain_started", "Block drain started"
            )

    def _resolve_block_drain(
        self, db: sqlite3.Connection, session_number: int,
        *, work_order_id: int | None = None,
    ) -> tuple[list[tuple[str, str]], int]:
        """#187 auto-accept-failed-as-unusable + unresolved-count, shared by
        `try_enter_slides` (NORMAL/OPEN_RETRIEVAL) and `freeze_hybrid_pool`
        (HYBRID/HYBRID_SHADOW) so the rule cannot drift between the two drain
        paths (#250 review F8: this used to be duplicated verbatim in both
        methods with nothing asserting the copies agreed). Extracted with the
        exact same SQL/order both callers already used, so neither caller's
        behavior changes.

        Must run inside the caller's own `BEGIN IMMEDIATE` transaction --
        `db` is the caller's connection, never a fresh one -- so the
        auto-accept mutation and the unresolved count it feeds are read and
        written atomically together.

        `work_order_id` is OPTIONAL and defaults to `None`, which preserves
        `try_enter_slides`'s existing session-only scoping byte-for-byte
        (NORMAL/OPEN_RETRIEVAL never pass it). #269 review: only
        `freeze_hybrid_pool` passes a real `work_order_id`, scoping BOTH the
        auto-accept mutation and the unresolved count to that work order's
        own rows -- closing, for these two queries, the same cross-work-
        order-contamination gap #269 already closed for the candidate
        `SELECT` in `freeze_hybrid_pool` itself.
        """
        auto_accepted: list[tuple[str, str]] = []
        work_order_clause = "" if work_order_id is None else " AND work_order_id=?"
        work_order_params = () if work_order_id is None else (work_order_id,)
        failed_rows = db.execute(
            f"""SELECT block_id, failure_reason FROM sets
               WHERE session_number=?{work_order_clause}
               AND preprocessing_status='failed'
               ORDER BY rowid""",
            (session_number, *work_order_params),
        ).fetchall()
        if failed_rows:
            dismissed_at = datetime.now(timezone.utc).isoformat()
            for failed in failed_rows:
                reason = (
                    "auto-accepted unusable after preprocessing failure: "
                    f"{failed['failure_reason']}"
                )
                db.execute(
                    """UPDATE sets SET preprocessing_status='unusable',
                       dismissed_at=?, unusable_reason=?, mask_path=NULL
                       WHERE session_number=? AND block_id=?
                       AND preprocessing_status='failed'""",
                    (dismissed_at, reason, session_number, failed["block_id"]),
                )
                auto_accepted.append((str(failed["block_id"]), reason))
        unresolved = db.execute(
            f"""SELECT COUNT(*) AS count FROM sets WHERE session_number=?
               {work_order_clause}
               AND preprocessing_status NOT IN ('complete', 'unusable')""",
            (session_number, *work_order_params),
        ).fetchone()["count"]
        return auto_accepted, int(unresolved)

    def try_enter_slides(self, session_number: int) -> bool:
        """Atomically enter slides only when every block has a terminal outcome.

        #187: failed preprocessing is auto-accepted as unusable during drain so
        one bad segment cannot stall the whole session on Processing… Recapture
        remains available while still in the blocks phase.
        """
        entered = False
        auto_accepted: list[tuple[str, str]] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
            if row is None or row["phase"] != "draining_blocks":
                return bool(row and row["phase"] == "slides")
            auto_accepted, unresolved = self._resolve_block_drain(db, session_number)
            if not unresolved:
                db.execute(
                    "UPDATE sessions SET phase='slides' WHERE session_number=?",
                    (session_number,),
                )
                entered = True
        for block_id, reason in auto_accepted:
            self._emit(session_number, "block_dismissed", reason, block_id)
        if entered or auto_accepted:
            self.reconcile_session_metadata(session_number)
        return entered

    def _hybrid_freeze_lock(self, session_number: int) -> Lock:
        """The per-session in-process lock serializing `freeze_hybrid_pool`
        (#250 review F4)."""
        with self._hybrid_freeze_locks_guard:
            return self._hybrid_freeze_locks.setdefault(session_number, Lock())

    def _log_durable(self, session_number: int, kind: str, message: str) -> None:
        """Append one durable line to `<session_dir>/events.log` (#250 review
        F1/F5). `_emit` alone appends only to `self._events`, an in-memory
        list that is lost on restart -- fine for the kiosk's live poll, not
        fine for "why did Finish Blocks bounce"/"why did every block fail"
        surviving a crash or restart. Best-effort: a logging failure here
        must never crash the drain/freeze path it is reporting on.
        """
        try:
            session = self._session_identity(session_number)
            line = json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "session_number": session_number,
                    "kind": kind,
                    "message": message,
                },
                sort_keys=True,
            )
            with self._durable_log_lock:
                with (session.directory / "events.log").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        except (OSError, ValueError):
            pass

    def _log_durable_exception(
        self, session_number: int, kind: str, message: str, exc: BaseException
    ) -> None:
        """`_log_durable`, with the swallowed exception's full traceback
        appended (#250 review F5): fail-closed demotion must stay silent to
        the *drain*, never to the durable record an operator/developer can
        read after the fact."""
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._log_durable(session_number, kind, f"{message}\n{rendered}")

    def _hybrid_pool_block_ids_for_out_of_pool_guard(
        self, db: sqlite3.Connection, session_number: int, work_order_id: int,
    ) -> frozenset[str]:
        """#251: the narrow membership read `record_slide_capture`'s
        Out-of-Pool Claim guard needs -- just this ONE work order's frozen
        `block_ids`, on the connection already open in the caller's
        transaction.

        Deliberately NOT `self.hybrid_pool(work_order_id)`: that
        decompresses the whole `.npz` archive (fingerprints and
        score-caches for every block) on EVERY slide capture, and it
        deliberately raises `ValueError` on a missing, corrupt, or
        version-mismatched manifest (see its own docstring). This method
        runs on the Pi drain/replay path, where the blast-radius rule is
        absolute: nothing on a per-poll/per-tick/per-row path may raise --
        a raise here would reach `_camera_loop`'s `except Exception` and
        kill the camera loop. Also deliberately NOT the session-wide
        `get_set`/`sets` lookup `resolve_claim` uses: that searches the
        WHOLE session, so a block belonging to a DIFFERENT work order in
        the same session would be found and wrongly treated as in-pool --
        exactly the leak this guard exists to close.

        Any failure (no frozen row yet, corrupt JSON) degrades to an empty
        set -- "nothing is in pool" -- which the caller correctly treats as
        an out-of-pool claim. An actual exception (corrupt JSON, not
        merely an absent row) is durably logged first so the failure mode
        stays diagnosable rather than silent.
        """
        try:
            row = db.execute(
                "SELECT block_ids FROM hybrid_pools WHERE work_order_id=?",
                (work_order_id,),
            ).fetchone()
            if row is None:
                return frozenset()
            return frozenset(json.loads(row["block_ids"]))
        except Exception as exc:  # fail closed: never let this crash the capture
            self._log_durable_exception(
                session_number, "hybrid_pool_membership_check_failed",
                f"Work order {work_order_id} pool membership check raised; "
                "treating pool as empty for the out-of-pool-claim guard",
                exc,
            )
            return frozenset()

    def _hybrid_block_usability(
        self, session_number: int, row: Mapping[str, object]
    ) -> tuple[bool, str, PreparedResult | None]:
        """CONTEXT.md "Usable Hybrid Block": unique identity (the ``sets``
        primary key already guarantees this), durable capture, successful
        canonical preparation, and passed block quality checks -- judged from
        the block alone (#250), before any slide exists to pair it with.

        Returns the loaded ``PreparedResult`` too when usable, so the caller
        never re-reads the mask PNG a second time to build the fingerprint/
        score cache (#250 review F9).

        Never raises: every failure mode (missing/corrupt files, a bad mask)
        demotes the block to "not usable" with a reason instead of
        propagating -- durably logged with a full traceback first (#250
        review F5), so a systemic defect (e.g. a misconfigured descriptor
        name) is diagnosable instead of merely "0 usable blocks" for every
        capture. This matters because it runs on the same drain path a
        background poll tick can trigger (``SessionWorkflow.poll_drain``),
        where an uncaught exception would stop capture entirely, not just
        this one freeze attempt.
        """
        if row["preprocessing_status"] != "complete" or not row["mask_path"]:
            return False, "block preprocessing did not complete", None
        try:
            capture_path = row["capture_path"]
            if not capture_path or not Path(str(capture_path)).is_file():
                return False, "durable block capture is missing from disk", None
            if _sha256(Path(str(capture_path))) != row["checksum"]:
                return False, "durable block capture failed checksum verification", None
            if not Path(str(row["mask_path"])).is_file():
                return False, "comparable block mask is missing from disk", None
            block_result = self._load_block_result(row)
            if isinstance(block_result, PreparationFailure):
                return False, block_result.reason, None
            gate = check_block_quality(block_result)
            if not gate.passed:
                return False, gate.reason, None
        except Exception as exc:  # fail closed: never let one block crash the drain
            reason = f"usability check failed: {exc}"
            self._log_durable_exception(
                session_number, "hybrid_block_usability_check_failed",
                f"Block {row.get('block_id', '?')} usability check raised: {reason}",
                exc,
            )
            return False, reason, None
        return True, "", block_result

    def _bounce_hybrid_pool_to_blocks(
        self,
        session_number: int,
        block_ids: Sequence[str],
        message: str,
        *,
        request_id: str | None = None,
        fingerprint: str | None = None,
    ) -> HybridPoolFreezeResult:
        """<2 usable blocks (or a configuration defect caught before ever
        reaching that count): do not freeze, do not abandon -- return to
        block capture with a clear message (#250 acceptance criterion).

        The caller composes ``message``; this helper only performs the
        mechanical bounce: phase repair, the operator-visible event, a
        durable log line (`_emit` alone is in-memory-only and would be lost
        on restart -- #250 review F1), and ledger-replay stability for a
        retried ``request_id`` (#250 review F11: without this, a replay took
        the ``phase != 'draining_blocks'`` branch above and returned a
        DIFFERENT message than the original call, violating the documented
        `_LEDGERED` "replay returns the same response" contract).
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE sessions SET phase='blocks' "
                "WHERE session_number=? AND phase='draining_blocks'",
                (session_number,),
            )
            result = HybridPoolFreezeResult(False, tuple(block_ids), message)
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "freeze_hybrid_pool", session_number,
                    fingerprint, store_wire.dumps(result),
                )
        self._emit(session_number, "hybrid_pool_freeze_insufficient_blocks", message)
        self._log_durable(session_number, "hybrid_pool_freeze_insufficient_blocks", message)
        self.reconcile_session_metadata(session_number)
        return result

    def freeze_hybrid_pool(
        self,
        session_number: int,
        *,
        descriptor_names: Sequence[str],
        candidate_configuration: Mapping[str, object] | None = None,
        request_id: str | None = None,
    ) -> HybridPoolFreezeResult:
        """One-way Finish-Blocks freeze of the Hybrid Candidate Pool (#250),
        re-keyed per WORK ORDER, not per session (#269).

        Called (via ``SessionWorkflow.poll_drain``) only while the session is
        in HYBRID/HYBRID_SHADOW mode and the drain has been started with
        ``begin_block_drain``. Mirrors ``try_enter_slides``'s "wait for every
        block to resolve" contract (sharing its ``_resolve_block_drain`` #187
        auto-accept-failed-as-unusable rule), then adds: resolving the
        session's currently open (``lifecycle_state='capturing'``) work
        order -- the SAME lookup ``start_work_order`` itself uses -- and
        keying every ``hybrid_pools`` lookup/insert and both artifact paths
        on that work_order_id, never session_number. This is the isolation
        invariant's structural enforcement point: a pool frozen for work
        order A is physically addressed by A's id, so a second work order B
        in the same session cannot read, overwrite, or be silently told
        "already frozen" against A's pool. On top of that: rejecting an
        empty ``descriptor_names`` (a configuration defect that must never
        freeze a pool with zero fingerprints -- #250 review F2); usable-block
        counting SCOPED to this work order's own candidate rows (never a
        different work order's already-'complete' blocks lingering in the
        same session's ``sets`` table); freezing an immutable pool plus its
        fingerprints/accurate-scoring caches -- built exactly once, ever, per
        work order -- on >=2 usable; or bouncing phase back to ``'blocks'``
        (never staying in ``'draining_blocks'``) on <2 so the operator can
        capture more and click Finish Blocks again.

        Serialized per-session (``_hybrid_freeze_lock``): freeze does real
        file I/O (archive write) plus multiple seconds of CV work, well past
        the RPC client's 10s timeout, so a retried duplicate request can
        otherwise run concurrently with the original attempt -- corrupting
        the shared archive staging path and racing the ledger insert (#250
        review F4). The lock makes this method's entire body run for one
        session at a time (only one work order can be ``capturing`` per
        session at once, so this remains correct under #269's re-key);
        ``_ledger_record`` is additionally collision-tolerant as defense in
        depth.

        Idempotent: once a ``hybrid_pools`` row exists for this work order,
        every later call returns that identical result without touching
        ``fingerprint_builder``/``score_cache_builder`` again -- the row's
        mere existence is the durable one-way marker (its ``work_order_id``
        primary key also makes a second INSERT impossible at the schema
        level, a defense-in-depth backstop under the application-level check
        below). Never raises: an unknown session, or a session with no open
        work order, returns a not-frozen result (mirroring
        ``try_enter_slides``'s own ``bool(row and ...)`` contract) rather
        than raising, because this can run on a background poll tick where a
        raise would propagate through the RPC layer as a
        ``store.remote.StoreError`` and kill the Pi camera loop (#250 review
        F6).
        """
        descriptor_names = tuple(descriptor_names)
        configuration = (
            dict(candidate_configuration) if candidate_configuration is not None else None
        )
        fingerprint = (
            self._fingerprint({
                "descriptor_names": list(descriptor_names),
                "candidate_configuration": configuration,
            })
            if request_id is not None else None
        )
        with self._hybrid_freeze_lock(session_number):
            already_frozen_result: HybridPoolFreezeResult | None = None
            empty_descriptor_recipe = False
            no_open_work_order = False
            auto_accepted: list[tuple[str, str]] = []
            unresolved = 0
            candidate_rows: list[dict[str, object]] = []
            work_order_id: int | None = None
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if request_id is not None:
                    hit, cached = self._ledger_hit(
                        db, request_id, "freeze_hybrid_pool", session_number, fingerprint
                    )
                    if hit:
                        return store_wire.loads_as(HybridPoolFreezeResult, cached)
                # #269: resolve the pool's real key FIRST -- mirrors
                # start_work_order's own lifecycle_state='capturing' lookup.
                open_work_order = db.execute(
                    """SELECT work_order_id FROM work_orders
                       WHERE session_number=? AND lifecycle_state='capturing'
                       ORDER BY work_order_id DESC LIMIT 1""",
                    (session_number,),
                ).fetchone()
                work_order_id = (
                    int(open_work_order["work_order_id"])
                    if open_work_order is not None else None
                )
                existing = (
                    db.execute(
                        "SELECT block_ids FROM hybrid_pools WHERE work_order_id=?",
                        (work_order_id,),
                    ).fetchone()
                    if work_order_id is not None else None
                )
                if existing is not None:
                    # #250 review F3: repair phase even on the already-frozen
                    # branch. Without this, a session whose phase somehow
                    # re-entered 'draining_blocks' after freezing (proven
                    # reachable; see tests/test_hybrid_pool_freeze.py) would
                    # stay stuck there forever -- poll_drain calls this every
                    # second and this branch never used to touch phase.
                    db.execute(
                        "UPDATE sessions SET phase='slides' "
                        "WHERE session_number=? AND phase='draining_blocks'",
                        (session_number,),
                    )
                    already_frozen_result = HybridPoolFreezeResult(
                        True, tuple(json.loads(existing["block_ids"])),
                        "Hybrid Candidate Pool already frozen for this work order",
                    )
                    if request_id is not None:
                        self._ledger_record(
                            db, request_id, "freeze_hybrid_pool", session_number,
                            fingerprint, store_wire.dumps(already_frozen_result),
                        )
                else:
                    row = db.execute(
                        "SELECT phase FROM sessions WHERE session_number=?",
                        (session_number,),
                    ).fetchone()
                    if row is None:
                        result = HybridPoolFreezeResult(False, (), "unknown session")
                        if request_id is not None:
                            self._ledger_record(
                                db, request_id, "freeze_hybrid_pool", session_number,
                                fingerprint, store_wire.dumps(result),
                            )
                        return result
                    if work_order_id is None:
                        # No open work-order bracket to freeze: reachable by
                        # DEFAULT, not merely by a direct/console caller --
                        # #250's `begin_block_drain` already flipped phase to
                        # 'draining_blocks' before this method is ever
                        # called, so returning here WITHOUT bouncing back to
                        # 'blocks' (as this branch used to) live-locks the
                        # session forever: `scan_block` refuses once phase
                        # leaves 'blocks', and `SessionWorkflow.poll_drain`
                        # calls this every second, so nothing ever moves the
                        # phase again. Set the flag and fall through to the
                        # SAME `_bounce_hybrid_pool_to_blocks` helper every
                        # other insufficient-freeze branch uses, rather than
                        # returning directly from inside this transaction --
                        # the bounce opens its own connection/transaction, so
                        # it must run only after this one has closed.
                        no_open_work_order = True
                    else:
                        phase = row["phase"]
                        if phase != "draining_blocks":
                            message = (
                                "block drain has not started" if phase != "slides" else
                                "session already entered slides without a Hybrid "
                                "Candidate Pool"
                            )
                            result = HybridPoolFreezeResult(False, (), message)
                            if request_id is not None:
                                self._ledger_record(
                                    db, request_id, "freeze_hybrid_pool",
                                    session_number, fingerprint,
                                    store_wire.dumps(result),
                                )
                            return result

                        if not descriptor_names:
                            empty_descriptor_recipe = True
                        else:
                            auto_accepted, unresolved = self._resolve_block_drain(
                                db, session_number, work_order_id=work_order_id
                            )
                            # #269 isolation fix: scoped to THIS work order's
                            # own rows. Without `AND work_order_id=?`, a
                            # second work order in the same session would
                            # also pick up every earlier (already-'complete')
                            # work order's blocks still sitting in `sets` --
                            # exactly the cross-work-order contamination this
                            # issue exists to prevent.
                            candidate_rows = (
                                [] if unresolved else [
                                    dict(candidate_row)
                                    for candidate_row in db.execute(
                                        """SELECT * FROM sets
                                           WHERE session_number=?
                                           AND work_order_id=?
                                           AND preprocessing_status='complete'
                                           ORDER BY rowid""",
                                        (session_number, work_order_id),
                                    ).fetchall()
                                ]
                            )
                            if unresolved and request_id is not None:
                                self._ledger_record(
                                    db, request_id, "freeze_hybrid_pool",
                                    session_number, fingerprint, store_wire.dumps(
                                        HybridPoolFreezeResult(
                                            False, (), "block work is still resolving"
                                        ),
                                    ),
                                )

            if already_frozen_result is not None:
                self.reconcile_session_metadata(session_number)
                return already_frozen_result

            if no_open_work_order:
                # #269 FIX1: bounce through the SAME helper every other
                # insufficient-freeze branch uses, so phase returns to
                # 'blocks' (never stays stuck in 'draining_blocks') with a
                # durable, emitted, operator-legible reason -- exactly like
                # <2 usable blocks or an empty descriptor recipe.
                return self._bounce_hybrid_pool_to_blocks(
                    session_number, (), "no open work order for this session",
                    request_id=request_id, fingerprint=fingerprint,
                )

            for block_id, reason in auto_accepted:
                self._emit(session_number, "block_dismissed", reason, block_id)
            if auto_accepted:
                self.reconcile_session_metadata(session_number)

            if empty_descriptor_recipe:
                message = (
                    "Hybrid Candidate Pool cannot be frozen: the Hybrid "
                    "configuration named zero descriptors (descriptor_names "
                    "is empty). This is a configuration defect, not a "
                    "block-count problem -- fix the Hybrid Configuration "
                    "handoff and restart before Finish Blocks can succeed."
                )
                return self._bounce_hybrid_pool_to_blocks(
                    session_number, (), message,
                    request_id=request_id, fingerprint=fingerprint,
                )
            if unresolved:
                return HybridPoolFreezeResult(False, (), "block work is still resolving")

            usable_block_ids: list[str] = []
            usable_block_results: dict[str, PreparedResult] = {}
            for candidate_row in candidate_rows:
                usable, reason, block_result = self._hybrid_block_usability(
                    session_number, candidate_row
                )
                block_id = str(candidate_row["block_id"])
                if usable:
                    usable_block_ids.append(block_id)
                    usable_block_results[block_id] = block_result
                else:
                    self._emit(
                        session_number, "hybrid_block_not_usable",
                        f"Block excluded from Hybrid Candidate Pool: {reason}",
                        block_id,
                    )
            if len(usable_block_ids) < 2:
                if len(candidate_rows) >= 2 and not usable_block_ids:
                    # #250 review F5: distinguish "every candidate block
                    # errored" from "not enough blocks were captured" -- both
                    # produced the same misleading "Only 0 usable block(s)"
                    # message before.
                    message = (
                        f"All {len(candidate_rows)} candidate block(s) "
                        "failed a usability check (identity/capture/"
                        "preparation/quality); 0 usable. See the session's "
                        "durable event log for the exact per-block reason, "
                        "then Capture more blocks, then Finish Blocks again."
                    )
                else:
                    message = (
                        f"Only {len(usable_block_ids)} usable block(s); "
                        "Hybrid requires at least 2. Capture more blocks, "
                        "then Finish Blocks again."
                    )
                return self._bounce_hybrid_pool_to_blocks(
                    session_number, usable_block_ids, message,
                    request_id=request_id, fingerprint=fingerprint,
                )

            # >= 2 usable: build each block's fingerprint + accurate-scoring
            # cache exactly once, reusing retrieval_evidence's own established
            # order -- normalize once (the cache build), then read
            # descriptors off that SAME normalized mask, never off the raw
            # block mask. Reuses the PreparedResult the usability check
            # already loaded (#250 review F9): the block mask PNG is decoded
            # exactly once per freeze, never twice.
            fingerprints: dict[str, dict[str, np.ndarray]] = {}
            score_caches: dict[str, LockedScoreCache] = {}
            frozen_block_ids: list[str] = []
            for block_id in usable_block_ids:
                try:
                    block_result = usable_block_results[block_id]
                    score_cache = self.score_cache_builder(block_result)
                    values = self.fingerprint_builder(score_cache.normalized_mask)
                    fingerprints[block_id] = {
                        name: values[name].vector for name in descriptor_names
                    }
                    score_caches[block_id] = score_cache
                except Exception as exc:  # never let one block's CV failure crash the drain
                    reason = f"fingerprint construction failed: {exc}"
                    self._emit(
                        session_number, "hybrid_block_not_usable",
                        f"Block excluded from Hybrid Candidate Pool: {reason}",
                        block_id,
                    )
                    self._log_durable_exception(
                        session_number, "hybrid_block_fingerprint_failed",
                        f"Block {block_id} excluded from Hybrid Candidate "
                        f"Pool: {reason}", exc,
                    )
                    continue
                frozen_block_ids.append(block_id)

            if len(frozen_block_ids) < 2:
                if usable_block_ids and not frozen_block_ids:
                    # #250 review F5: every usable block errored identically
                    # at fingerprint construction (proven reachable: a
                    # descriptor name absent from the catalog) -- name that,
                    # rather than the generic "Only 0 usable block(s)".
                    message = (
                        f"All {len(usable_block_ids)} usable block(s) failed "
                        "fingerprint or accurate-scoring-cache construction; "
                        "0 remain. This is a configuration or code defect, "
                        "not a block-count problem -- see the session's "
                        "durable event log for the exact per-block error."
                    )
                else:
                    message = (
                        f"Only {len(frozen_block_ids)} usable block(s) "
                        "survived fingerprint construction; Hybrid requires "
                        "at least 2. Capture more blocks, then Finish Blocks "
                        "again."
                    )
                return self._bounce_hybrid_pool_to_blocks(
                    session_number, frozen_block_ids, message,
                    request_id=request_id, fingerprint=fingerprint,
                )

            session = self._session_identity(session_number)
            frozen_at = datetime.now(timezone.utc).isoformat()
            # #269: artifacts live under <session_dir>/work_orders/, named by
            # work_order_id -- the same convention as the existing
            # work_order_{id:06d}_verdicts.csv/_sheets artifacts, and the
            # on-disk half of the isolation invariant (a second work order's
            # freeze can never collide with or overwrite the first's files).
            work_orders_dir = session.directory / "work_orders"
            work_orders_dir.mkdir(exist_ok=True)
            archive_path = (
                work_orders_dir
                / f"work_order_{work_order_id:06d}{_HYBRID_POOL_ARCHIVE_SUFFIX}"
            )
            manifest_path = (
                work_orders_dir
                / f"work_order_{work_order_id:06d}{_HYBRID_POOL_MANIFEST_SUFFIX}"
            )
            _save_hybrid_pool_archive(
                archive_path, block_ids=frozen_block_ids,
                descriptor_names=descriptor_names,
                fingerprints=fingerprints, score_caches=score_caches,
            )
            _atomic_json(manifest_path, {
                "schema_version": _HYBRID_POOL_SCHEMA_VERSION,
                "work_order_id": work_order_id,
                "session_number": session_number,
                "frozen_at": frozen_at,
                "block_ids": list(frozen_block_ids),
                "descriptor_names": list(descriptor_names),
                "candidate_configuration": configuration,
                "archive_path": str(archive_path),
            })

            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                already = db.execute(
                    "SELECT block_ids FROM hybrid_pools WHERE work_order_id=?",
                    (work_order_id,),
                ).fetchone()
                if already is not None:
                    # Lost a race with a concurrent freeze of the same work
                    # order; the just-written archive/manifest above are
                    # harmless orphan files -- the DB row stays the single
                    # source of truth for what is frozen. The per-session
                    # lock above makes this unreachable for a duplicate
                    # request_id retry; kept as defense in depth.
                    result = HybridPoolFreezeResult(
                        True, tuple(json.loads(already["block_ids"])),
                        "Hybrid Candidate Pool already frozen for this work order",
                    )
                else:
                    db.execute(
                        """INSERT INTO hybrid_pools(
                           work_order_id, session_number, frozen_at, block_ids,
                           descriptor_names, candidate_configuration, manifest_path, archive_path
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            work_order_id, session_number, frozen_at,
                            json.dumps(frozen_block_ids),
                            json.dumps(list(descriptor_names)), json.dumps(configuration),
                            str(manifest_path),
                            str(archive_path),
                        ),
                    )
                    db.execute(
                        "UPDATE sessions SET phase='slides' "
                        "WHERE session_number=? AND phase='draining_blocks'",
                        (session_number,),
                    )
                    result = HybridPoolFreezeResult(
                        True, tuple(frozen_block_ids),
                        f"Hybrid Candidate Pool frozen with {len(frozen_block_ids)} "
                        "usable block(s)",
                    )
                if request_id is not None:
                    self._ledger_record(
                        db, request_id, "freeze_hybrid_pool", session_number,
                        fingerprint, store_wire.dumps(result),
                    )
            self.reconcile_session_metadata(session_number)
            if "frozen with" in result.message:
                self._emit(session_number, "hybrid_pool_frozen", result.message)
            return result

    def hybrid_pool(self, work_order_id: int) -> HybridCandidatePool | None:
        """Read the frozen Hybrid Candidate Pool back from durable storage.

        #269: keyed by ``work_order_id``, not ``session_number`` -- see the
        ``hybrid_pools`` CREATE TABLE comment. Pure read: never calls
        ``fingerprint_builder``/``score_cache_builder``. This is what proves
        the freeze is durable across a restart -- a fresh ``ProcessingStore``
        pointed at the same ``root`` can call this and get the identical pool
        without recomputing anything (#250's "built once" TDD requirement).
        Process-local only (see ``HybridCandidatePool``); not part of the
        ``/rpc`` surface, so it has no ``RemoteProcessingStore`` proxy or
        whitelist entry.

        The SQLite ``hybrid_pools`` row is authoritative for ``block_ids``/
        ``descriptor_names``/``session_number`` (read straight from it above);
        the manifest JSON is a second, human/audit-readable copy of the same
        facts. #250 review F7: this method DOES validate the manifest's
        ``schema_version`` against ``_HYBRID_POOL_SCHEMA_VERSION`` on every
        read (making that module-level constant's stated purpose -- "a future
        format change can be detected on read rather than guessed" --
        actually true, rather than merely claimed) and raises a clear
        ``ValueError`` if the manifest or archive is missing, corrupt, or the
        wrong version, rather than silently reading only from the SQLite row
        and the archive.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM hybrid_pools WHERE work_order_id=?", (work_order_id,)
            ).fetchone()
        if row is None:
            return None
        session_number = int(row["session_number"])
        block_ids = tuple(json.loads(row["block_ids"]))
        descriptor_names = tuple(json.loads(row["descriptor_names"]))
        candidate_configuration = (
            json.loads(row["candidate_configuration"])
            if row["candidate_configuration"] is not None else None
        )
        manifest_path = Path(row["manifest_path"])
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Hybrid Candidate Pool manifest is missing or unreadable at "
                f"{manifest_path}: {exc}"
            ) from exc
        manifest_version = manifest.get("schema_version")
        if manifest_version != _HYBRID_POOL_SCHEMA_VERSION:
            raise ValueError(
                "Hybrid Candidate Pool manifest schema version mismatch: "
                f"expected {_HYBRID_POOL_SCHEMA_VERSION!r}, found "
                f"{manifest_version!r} at {manifest_path}"
            )
        try:
            fingerprints, score_caches = _load_hybrid_pool_archive(
                Path(row["archive_path"]), block_ids=block_ids,
                descriptor_names=descriptor_names,
            )
        except Exception as exc:  # wrap-and-raise: never swallowed, always re-raised
            raise ValueError(
                "Hybrid Candidate Pool archive is missing, truncated, or "
                f"unreadable at {row['archive_path']}: {exc}"
            ) from exc
        return HybridCandidatePool(
            work_order_id=work_order_id,
            session_number=session_number,
            frozen_at=str(row["frozen_at"]),
            block_ids=block_ids,
            descriptor_names=descriptor_names,
            candidate_configuration=candidate_configuration,
            fingerprints=fingerprints,
            score_caches=score_caches,
        )

    def select_hybrid_candidates(
        self,
        pool: HybridCandidatePool,
        claim_id: str,
        slide_cache: LockedScoreCache,
        *,
        mask_quality_fallback_reason: str | None = None,
    ) -> CandidateSelection:
        """#253: rank an already-loaded Hybrid Candidate Pool against one
        already-built slide ``LockedScoreCache`` and return the Heuristic
        Candidate Band selection -- WITHOUT scoring anything.

        Selection, separated out from ``_score_hybrid_slide``'s scoring, so
        it is independently callable: #254 (Hybrid Shadow) calls this exact
        method (``store.select_hybrid_candidates(pool, claim_id,
        slide_cache)``, with its own ``pool = store.hybrid_pool(work_order_id)``
        already in hand) to observe what the band would have selected without
        running ``score_routed_caches`` at all. ``_score_hybrid_slide`` also
        calls it, with the SAME already-loaded ``pool`` it needed anyway,
        then goes on to score. Taking ``pool`` rather than ``work_order_id``
        is deliberate: ``self.hybrid_pool`` re-reads the SQLite row + ``.npz``
        archive from disk on every call (#250's "pure read" contract, no
        in-memory cache across calls) -- a second internal lookup here would
        silently double that I/O on every scored slide.

        Every pool block's fingerprint is built via ``_pool_specimen_fingerprint``
        from data #250's ``freeze_hybrid_pool`` already computed once and
        durably persisted (``pool.fingerprints``/``.score_caches``) -- no
        cv2, no recomputation. Only the slide's fingerprint is genuinely
        built here, per job, via the same injectable ``self.fingerprint_builder``
        seam #250 already uses for pool blocks (``_slide_specimen_fingerprint``).
        """
        pool_fingerprints = {
            block_id: _pool_specimen_fingerprint(pool, block_id)
            for block_id in pool.block_ids
        }
        slide_fingerprint = _slide_specimen_fingerprint(
            self.fingerprint_builder, slide_cache
        )
        if mask_quality_fallback_reason is not None:
            return select_candidate_band(
                pool_fingerprints, slide_fingerprint, claim_id,
                mask_quality_fallback_reason=mask_quality_fallback_reason,
            )
        configuration = pool.candidate_configuration
        if configuration is None:
            return select_candidate_band(
                pool_fingerprints, slide_fingerprint, claim_id,
                mask_quality_fallback_reason=(
                    "Hybrid Candidate Pool has no versioned handoff configuration; "
                    "scoring complete pool"
                ),
            )
        return select_configured_candidate_band(
            pool_fingerprints, slide_fingerprint, claim_id,
            architecture_kind=str(configuration.get("architecture_kind", "")),
            architecture_name=str(configuration.get("architecture_name", "")),
            architecture_methods=tuple(configuration.get("architecture_methods", ())),
            candidate_band_thresholds=configuration.get("candidate_band_thresholds", {}),
        )

    def reconcile_session_metadata(self, session_number: int) -> None:
        """Regenerate human-readable metadata from canonical SQLite state."""
        session = self._session_identity(session_number)
        metadata_path = session.directory / "session.json"
        with self._connect() as db:
            row = db.execute(
                """SELECT phase, finalized_at, last_finalization_error
                   FROM sessions WHERE session_number=?""",
                (session_number,),
            ).fetchone()
        if row is None:
            raise ValueError("unknown session")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {
                "session_number": session.number,
                "started_at": session.started_at.isoformat(),
            }
        payload["phase"] = row["phase"]
        for key in ("finalized_at", "last_finalization_error"):
            if row[key] is None:
                payload.pop(key, None)
            else:
                payload[key] = row[key]
        _atomic_json(metadata_path, payload)

    def begin_finalization(self, session_number: int) -> None:
        changed = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown session")
            if row["phase"] == "slides":
                db.execute(
                    "UPDATE sessions SET phase='finalizing' WHERE session_number=?",
                    (session_number,),
                )
                changed = True
        if changed:
            try:
                self.reconcile_session_metadata(session_number)
            except Exception as exc:
                self.record_finalization_error(
                    session_number, f"Session metadata reconciliation failed: {exc}",
                    reconcile=False,
                )
                raise
            self._emit(session_number, "finalization_started", "Finalization started")

    def prepare_finalization(self, session_number: int) -> bool:
        """Verify/export, then durably enter cleanup_pending; safe to re-poll.

        Unlike the sibling phase transitions, the capture list read here is
        not held under the same `BEGIN IMMEDIATE` as the final write: hashing
        every retained file must not hold a write lock. This is safe only
        because `scan_block`/`record_slide_capture` already reject new
        captures once the phase leaves `blocks`/`slides`, so no capture can
        appear between the read and the write below.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown session")
            if row["phase"] in {"cleanup_pending", "finalized"}:
                return True
            if row["phase"] != "finalizing":
                return False
            pending = db.execute(
                """SELECT COUNT(*) AS count FROM sets WHERE session_number=?
                   AND preprocessing_status IN ('queued', 'processing')""",
                (session_number,),
            ).fetchone()["count"]
            if pending:
                return False
            captures = db.execute(
                """SELECT capture_path, checksum FROM sets
                   WHERE session_number=? AND capture_path IS NOT NULL""",
                (session_number,),
            ).fetchall()
            captures += db.execute(
                """SELECT capture_path, checksum FROM slide_captures
                   WHERE session_number=?""",
                (session_number,),
            ).fetchall()
        problems = tuple(
            str(row["capture_path"]) for row in captures
            if not Path(row["capture_path"]).is_file()
            or _sha256(Path(row["capture_path"])) != row["checksum"]
        )
        if problems:
            message = (
                f"{len(problems)} retained capture(s) failed verification: "
                f"{', '.join(problems)}"
            )
            self.record_finalization_error(session_number, message)
            self._emit(session_number, "finalization_verification_failed", message)
            return False
        session = self._session_identity(session_number)
        try:
            self._refresh_manifest_export(session)
            self._refresh_decisions_export(session)
        except Exception as exc:
            self.record_finalization_error(
                session_number, f"Finalization export failed: {exc}"
            )
            raise
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                """UPDATE sessions SET phase='cleanup_pending'
                   WHERE session_number=? AND phase='finalizing'""",
                (session_number,),
            ).rowcount
        if updated == 0:
            return False
        try:
            self.reconcile_session_metadata(session_number)
        except Exception as exc:
            self.record_finalization_error(
                session_number, f"Session metadata reconciliation failed: {exc}",
                reconcile=False,
            )
            raise
        return True

    def complete_finalization(self, session_number: int) -> bool:
        """Publish finalized only after cleanup; retry metadata after a crash."""
        with self._connect() as db:
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown session")
            if row["phase"] == "cleanup_pending":
                finalized_at = datetime.now(timezone.utc).isoformat()
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE sessions SET phase='finalized', finalized_at=?,
                       last_finalization_error=NULL WHERE session_number=?
                       AND phase='cleanup_pending'""",
                    (finalized_at, session_number),
                )
            elif row["phase"] != "finalized":
                return False
        try:
            self.reconcile_session_metadata(session_number)
        except Exception as exc:
            self.record_finalization_error(
                session_number, f"Session metadata reconciliation failed: {exc}",
                reconcile=False,
            )
            raise
        with self._connect() as db:
            db.execute(
                "UPDATE sessions SET last_finalization_error=NULL WHERE session_number=?",
                (session_number,),
            )
        self.reconcile_session_metadata(session_number)
        self._emit(session_number, "session_finalized", "Session finalized")
        return True

    def record_finalization_error(
        self, session_number: int, message: str, *, reconcile: bool = True
    ) -> None:
        """Durable failure reason: recovery information, not just an in-memory event."""
        with self._connect() as db:
            db.execute(
                "UPDATE sessions SET last_finalization_error=? WHERE session_number=?",
                (message, session_number),
            )
        if reconcile:
            self.reconcile_session_metadata(session_number)

    def _refresh_manifest_export(self, session: SessionIdentity) -> None:
        with self._connect() as db:
            rows = db.execute(
                """SELECT block_id, capture_id, preprocessing_status, slide_capture_id,
                   verdict, score, decision_stage, decision_reason, decided_at
                   FROM sets WHERE session_number=? ORDER BY rowid""",
                (session.number,),
            ).fetchall()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "block_id", "capture_id", "preprocessing_status", "slide_capture_id",
            "verdict", "score", "decision_stage", "decision_reason", "decided_at",
        ])
        for row in rows:
            writer.writerow([
                row["block_id"], row["capture_id"] or "", row["preprocessing_status"],
                row["slide_capture_id"] or "", row["verdict"] or "",
                "" if row["score"] is None else f"{row['score']:.4f}",
                row["decision_stage"] or "", row["decision_reason"] or "",
                row["decided_at"] or "",
            ])
        _atomic_bytes(
            session.directory / "manifest.csv", buffer.getvalue().encode("utf-8")
        )

    def receive_capture(
        self,
        session_number: int,
        *,
        capture_id: str,
        block_id: str,
        checksum: str,
        body: bytes,
        start_job: bool = True,
        recapture: bool = False,
        profile: bool = False,
    ) -> UploadReceipt:
        if hashlib.sha256(body).hexdigest() != checksum:
            raise ValueError("checksum verification failed")
        session = self._session_identity(session_number)
        profile_queued_ns = self._profile_clock_ns() if profile else None
        capture_dir = session.directory / "captures"
        capture_dir.mkdir(exist_ok=True)
        destination = capture_dir / f"{capture_id}.png"
        with self._connect() as db:
            # Serialize the idempotency check and set/receipt update. Without
            # this write lock, two retry requests could both observe "new".
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                """SELECT receipts.checksum, sets.capture_path
                   FROM receipts JOIN sets USING (session_number, block_id)
                   WHERE receipts.capture_id = ?""",
                (capture_id,),
            ).fetchone()
            if prior:
                if prior["checksum"] != checksum:
                    raise ValueError("capture ID was previously stored with another checksum")
                stored = Path(prior["capture_path"])
                if not stored.is_file() or _sha256(stored) != checksum:
                    raise ValueError("previously acknowledged capture is not durably stored")
                return UploadReceipt(capture_id, True, checksum)

            _atomic_bytes(destination, body)
            if _sha256(destination) != checksum:
                destination.unlink(missing_ok=True)
                raise ValueError("stored capture checksum verification failed")
            if recapture:
                updated = db.execute(
                    """UPDATE sets SET capture_id=?, capture_path=?, checksum=?,
                       preprocessing_status='queued', preprocessing_metadata=NULL,
                       mask_path=NULL, qc_path=NULL, failure_reason=NULL,
                       profile_enabled=?, profile_queued_ns=?
                       WHERE session_number=? AND block_id=?
                       AND preprocessing_status='failed'""",
                    (capture_id, str(destination), checksum, int(profile),
                     profile_queued_ns, session_number, block_id),
                ).rowcount
            else:
                updated = db.execute(
                    """UPDATE sets SET capture_id=?, capture_path=?, checksum=?,
                       preprocessing_status='queued', profile_enabled=?,
                       profile_queued_ns=?
                       WHERE session_number=? AND block_id=? AND capture_id IS NULL""",
                    (capture_id, str(destination), checksum, int(profile),
                     profile_queued_ns, session_number, block_id),
                ).rowcount
            if updated != 1:
                destination.unlink(missing_ok=True)
                raise ValueError("capture does not match one unique unfilled set")
            db.execute(
                "INSERT INTO receipts VALUES (?, ?, ?, ?)",
                (capture_id, checksum, session_number, block_id),
            )
        self._emit(
            session_number, "upload_acknowledged", "Capture durably stored",
            block_id, capture_id,
        )
        if recapture:
            self._emit(
                session_number, "block_recaptured", "Failed block capture replaced",
                block_id, capture_id,
            )
        if start_job:
            self._submit_preprocessing(
                session, block_id, capture_id, destination,
                profile=profile, profile_queued_ns=profile_queued_ns,
            )
        return UploadReceipt(capture_id, True, checksum)

    def _submit_preprocessing(
        self, session: SessionIdentity, block_id: str, capture_id: str, path: Path,
        *, profile: bool = False, profile_queued_ns: int | None = None,
    ) -> None:
        job = self._executor.submit(
            self._preprocess, session, block_id, capture_id, path,
            profile=profile, profile_queued_ns=profile_queued_ns,
        )
        with self._jobs_lock:
            self._jobs.append(job)

    def _recover_jobs(self) -> None:
        """Re-enqueue interrupted block and retrieval-slide jobs.
        """
        with self._connect() as db:
            rows = db.execute(
                """SELECT session_number, block_id, capture_id, capture_path,
                          profile_enabled, profile_queued_ns
                   FROM sets WHERE preprocessing_status IN ('queued', 'processing')"""
            ).fetchall()
            db.execute(
                """UPDATE sets SET preprocessing_status='queued'
                   WHERE preprocessing_status='processing'"""
            )
        for row in rows:
            session = self._session_identity(int(row["session_number"]))
            self._submit_preprocessing(
                session,
                str(row["block_id"]),
                str(row["capture_id"]),
                Path(row["capture_path"]),
                profile=bool(row["profile_enabled"]),
                profile_queued_ns=row["profile_queued_ns"],
            )
        self._recover_retrieval_jobs()

    def _recover_retrieval_jobs(self) -> None:
        """Restart recovery for the durable per-slide retrieval queue.

        Only `slide_captures.job_state` rows matter here. NORMAL and Hybrid
        out-of-pool rows never get a job state, so this cannot touch them.
        `_recover_claims`'s own `sc.work_order_id IS NULL` guard deliberately
        excludes every work-order-stamped capture from its recovery sweep;
        this method is what a Retrieval Slide Job gets instead.

        Disposition per durable state, run at `ProcessingStore.__init__`
        time (startup, before any worker job is running, so no live Future
        can race this UPDATE):

        - `complete`: PRESERVED, never touched, never rescored. Rescoring an
          already-decided verdict would silently change a result the
          operator may already have seen -- the single most important rule
          here.
        - `queued`: nothing ever ran; simply resubmitted.
        - `preparing`/`scoring`: the process restarted mid-job -- nothing
          about the SLIDE failed, so this returns to `queued` (never
          `error`) and is resubmitted exactly like a fresh `queued` row.
          Landing here as `error` would manufacture a phantom system
          failure out of an ordinary restart.
        - `error`: LEFT ALONE, deliberately not resubmitted. #256 gives the
          operator an explicit retry verb for a genuine system failure ("An
          ERROR retries processing from the durable capture before
          recapture is offered") -- silently re-running it here would
          bypass that operator-facing decision and could re-run a job whose
          failure needs a human to look at it first.
        - `superseded` (#256 is the first writer of this value; nothing in
          this slice ever produces it, but a database can already contain
          one by the time this runs): LEFT ALONE, never resubmitted, never
          resurrected -- the whole point of supersession is that this
          capture is no longer the active job for its slide.

        Resubmission order preserves scheduling order across the restart:
        `priority` first (NULL sorts last -- see its migration comment),
        then `captured_at`/`capture_id` FIFO among equal priority (NULL
        priority is always equal to other NULL priority for this purpose).
        This is the ONLY mechanism that can reliably re-express "recaptures
        first, then ordinary FIFO" after a restart, because the
        single-worker `ThreadPoolExecutor`'s own internal queue has no
        priority concept -- only the ORDER jobs are submitted in matters,
        and this method controls that order completely since nothing else
        has submitted anything yet at startup.

        A single bad row must never block every other row from recovering
        (this runs at startup, so a hard failure here is tolerable for a
        genuinely corrupt database, but not for one merely-odd row) --
        `_submit_hybrid_scoring` itself only calls `self._executor.submit`
        and appends to an in-memory list, neither of which is expected to
        raise for a well-formed row, but a missing/unreadable
        `capture_path` or an unknown `session_number` must still degrade
        this ONE row rather than aborting the whole recovery sweep.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Nothing about an interrupted mid-flight job failed -- the
            # PROCESS restarted -- so it returns to 'queued', never 'error'.
            # #258: a profiled row's mid-flight stage indicator is reset to
            # 'queue_wait' along with it -- the worker never actually reached
            # 'complete', so its last-recorded stage would otherwise show a
            # phantom "still scoring" for a job that has not even resumed
            # yet. `profile_queued_ns` itself is deliberately left untouched
            # (mirrors `_recover_jobs`'s own block precedent): this is the
            # SAME durable job resuming, not a new enqueue.
            db.execute(
                """UPDATE slide_captures SET job_state='queued',
                   profile_current_stage=CASE WHEN profile_enabled=1
                       THEN 'queue_wait' ELSE profile_current_stage END
                   WHERE job_state IN ('preparing', 'scoring')"""
            )
            rows = db.execute(
                """SELECT session_number, work_order_id, block_id, capture_id,
                          capture_path, profile_enabled, profile_queued_ns
                   FROM slide_captures
                   WHERE job_state='queued'
                   ORDER BY CASE WHEN priority IS NULL THEN 1 ELSE 0 END,
                            priority, captured_at, capture_id"""
            ).fetchall()
        for row in rows:
            capture_id = str(row["capture_id"])
            try:
                session = self._session_identity(int(row["session_number"]))
                submit = (
                    self._submit_retrieval_scoring
                    if session.session_mode == SessionMode.OPEN_RETRIEVAL.value
                    else self._submit_hybrid_scoring
                )
                submit(
                    session,
                    int(row["work_order_id"]),
                    str(row["block_id"]),
                    capture_id,
                    Path(row["capture_path"]),
                    profile=bool(row["profile_enabled"]),
                    profile_queued_ns=row["profile_queued_ns"],
                )
            except Exception as exc:  # one bad row must not block the rest
                self._log_durable_exception(
                    int(row["session_number"]), "hybrid_job_recovery_failed",
                    f"Failed to recover Hybrid slide job for capture "
                    f"{capture_id}; this row stays 'queued' and will be "
                    "retried on the next restart", exc,
                )

    def _preprocess(
        self, session: SessionIdentity, block_id: str, capture_id: str, path: Path,
        *, profile: bool = False, profile_queued_ns: int | None = None,
    ) -> None:
        started_ns = self._profile_clock_ns() if profile else None
        preparation_finished_ns: int | None = None
        self._set_status(session.number, block_id, "processing")
        self._emit(
            session.number, "preprocessing_started", "Block preprocessing started",
            block_id, capture_id,
        )
        try:
            with observed(self.runtime_observer, "block_setup", block_id):
                if profile and self._uses_default_block_preprocessor:
                    mask, metadata = preprocess_block(path, profile=True)
                else:
                    mask, metadata = self.preprocessor(path)
            preparation_finished_ns = self._profile_clock_ns() if profile else None
            if mask.dtype != np.uint8 or mask.ndim != 2:
                raise ValueError("preprocessor must return a 2-D uint8 mask")
            if not np.any(mask):
                raise ValueError("preprocessor returned an empty comparable mask")
            artifact_dir = session.directory / "block_artifacts"
            artifact_dir.mkdir(exist_ok=True)
            mask_path = artifact_dir / f"{capture_id}_mask.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError("could not write comparable mask")
            qc_path = artifact_dir / f"{capture_id}_qc.png"
            self._write_qc(path, mask, qc_path)
            with self._connect() as db:
                db.execute(
                    """UPDATE sets SET preprocessing_status='complete',
                       preprocessing_metadata=?, mask_path=?, qc_path=?,
                       failure_reason=NULL
                       WHERE session_number=? AND block_id=? AND capture_id=?""",
                    (json.dumps(dict(metadata), sort_keys=True), str(mask_path),
                     str(qc_path), session.number, block_id, capture_id),
                )
            self._emit(
                session.number,
                "preprocessing_complete",
                "Block preprocessing complete",
                block_id, capture_id,
            )
            if profile:
                assert started_ns is not None and preparation_finished_ns is not None
                completed_ns = self._profile_clock_ns()
                self._record_block_benchmark(
                    session,
                    capture_id,
                    queue_wait_ms=_elapsed_ms(started_ns, profile_queued_ns),
                    block_preparation_ms=_elapsed_ms(
                        preparation_finished_ns, started_ns
                    ),
                    segmentation_ms=metadata.get("segmentation_ms"),
                    artifact_write_ms=_elapsed_ms(completed_ns, preparation_finished_ns),
                    ready_after_receive_ms=_elapsed_ms(completed_ns, profile_queued_ns),
                    status="complete",
                )
        except Exception as exc:  # failure is durable and must not kill capture
            failed_ns = self._profile_clock_ns() if profile else None
            artifact_dir = session.directory / "block_artifacts"
            artifact_dir.mkdir(exist_ok=True)
            qc_path = artifact_dir / f"{capture_id}_failed_qc.png"
            self._write_failure_qc(path, str(exc), qc_path)
            with self._connect() as db:
                db.execute(
                    """UPDATE sets SET preprocessing_status='failed',
                       preprocessing_metadata=?, failure_reason=?, qc_path=?,
                       mask_path=NULL WHERE session_number=? AND block_id=?
                       AND capture_id=?""",
                    (json.dumps({"error": str(exc)}), str(exc), str(qc_path),
                     session.number, block_id, capture_id),
                )
            self._emit(
                session.number, "preprocessing_failed", str(exc), block_id, capture_id
            )
            self._emit(
                session.number, "failed_block_warning",
                f"Block preprocessing failed: {exc}", block_id, capture_id,
            )
            if profile:
                assert started_ns is not None and failed_ns is not None
                completed_ns = self._profile_clock_ns()
                preparation_end_ns = preparation_finished_ns or failed_ns
                self._record_block_benchmark(
                    session, capture_id,
                    queue_wait_ms=_elapsed_ms(started_ns, profile_queued_ns),
                    block_preparation_ms=_elapsed_ms(preparation_end_ns, started_ns),
                    segmentation_ms=None,
                    artifact_write_ms=_elapsed_ms(completed_ns, preparation_end_ns),
                    ready_after_receive_ms=_elapsed_ms(completed_ns, profile_queued_ns),
                    status="failed",
                )

    @staticmethod
    def _write_qc(capture: Path, mask: np.ndarray, destination: Path) -> None:
        image = cv2.imread(str(capture))
        if image is None:
            raise ValueError("stored capture could not be reopened for QC")
        overlay = image.copy()
        overlay[mask > 0] = (
            overlay[mask > 0].astype(np.float32) * 0.5
            + np.array([0, 255, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        panel = np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay))
        if not cv2.imwrite(str(destination), panel):
            raise OSError("could not write block QC artifact")
        del image, overlay, panel

    @staticmethod
    def _write_failure_qc(capture: Path, reason: str, destination: Path) -> None:
        image = cv2.imread(str(capture))
        if image is None:
            image = np.zeros((240, 640, 3), dtype=np.uint8)
        banner_height = min(160, max(60, image.shape[0] // 10))
        panel = image.copy()
        panel[:banner_height] = (0, 0, 180)
        cv2.putText(
            panel, f"PREPROCESSING FAILED: {reason}"[:120],
            (20, max(35, banner_height // 2)), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        if not cv2.imwrite(str(destination), panel):
            raise OSError("could not write failed-block QC artifact")
        del image, panel

    def _write_slide_artifacts(
        self,
        session: SessionIdentity,
        capture_id: str,
        capture: Path,
        image: np.ndarray | None,
        result: PreparedResult,
    ) -> None:
        """Persist slide preparation evidence parallel to ``block_artifacts``."""
        artifact_dir = session.directory / "slide_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(result, PreparedSpecimen) and image is not None:
            mask_path = artifact_dir / f"{capture_id}_mask.png"
            if not cv2.imwrite(str(mask_path), result.mask):
                raise OSError("could not write slide comparable mask")
            qc_path = artifact_dir / f"{capture_id}_qc.png"
            self._write_slide_qc(image, result.mask, qc_path)
            return

        reason = result.reason if isinstance(result, PreparationFailure) else (
            "slide image unavailable"
        )
        failure_image = cv2.imread(str(capture), cv2.IMREAD_GRAYSCALE)
        failure_mask = np.zeros(
            (1, 1) if failure_image is None else failure_image.shape, dtype=np.uint8
        )
        if not cv2.imwrite(str(artifact_dir / f"{capture_id}_mask.png"), failure_mask):
            raise OSError("could not write failed slide comparable mask")
        del failure_image, failure_mask
        self._write_failure_qc(
            capture, reason, artifact_dir / f"{capture_id}_failed_qc.png"
        )

    @staticmethod
    def _write_slide_qc(
        image: np.ndarray, mask: np.ndarray, destination: Path
    ) -> None:
        """Write the same original/mask/overlay layout used for block QC."""
        display_mask = mask
        if mask.shape != image.shape[:2]:
            display_mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay = image.copy()
        overlay[display_mask > 0] = (
            overlay[display_mask > 0].astype(np.float32) * 0.5
            + np.array([0, 255, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        panel = np.hstack((
            image,
            cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR),
            overlay,
        ))
        if not cv2.imwrite(str(destination), panel):
            raise OSError("could not write slide QC artifact")
        del overlay, panel

    def _set_status(self, session_number: int, block_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE sets SET preprocessing_status=? WHERE session_number=? AND block_id=?",
                (status, session_number, block_id),
            )

    def wait_for_jobs(self) -> None:
        with self._jobs_lock:
            jobs = list(self._jobs)
        wait(jobs)

    def get_set(self, session_number: int, block_id: str) -> dict[str, object]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM sets WHERE session_number=? AND block_id=?",
                (session_number, block_id),
            ).fetchone()
        if row is None:
            raise KeyError(block_id)
        return dict(row)

    def precheck_slide_scan(self, session_number: int, block_id: str) -> bool:
        """Scan-time duplicate guard, symmetric to ``scan_block``'s.

        Returns ``True`` when a handheld slide scan for ``block_id`` may be
        stashed for the next capture, and ``False`` when that block already
        carries a durable verdict -- the same ``verdict IS NOT NULL`` condition
        ``resolve_claim`` rejects post-capture (see :meth:`claim_slide_verdict`).
        On a duplicate it emits ``duplicate_slide_scan`` so the kiosk can flash
        the operator immediately, before a capture cycle is spent. A block id
        absent from the session inventory is not a duplicate.
        """
        try:
            row = self.get_set(session_number, block_id)
        except KeyError:
            return True
        if row["verdict"] is not None:
            self._emit(
                session_number, "duplicate_slide_scan", "Slide already processed",
                block_id,
            )
            return False
        return True

    def record_slide_capture(
        self,
        session_number: int,
        source: str | Path,
        *,
        captured_at: datetime,
        result: SlideQRResult,
        duration_ms: float,
        request_id: str | None = None,
        source_token: str | None = None,
        start_job: bool = True,
        priority: int | None = None,
        profile: bool = False,
    ) -> str:
        """Durably store one slide still and its complete identity audit.

        ``priority`` (#255) is the durable retrieval job scheduling-order key
        (see the ``slide_captures.priority`` migration comment): ``None``
        (the default) is an ordinary job, scheduled FIFO by
        ``captured_at``/``capture_id`` alongside every other ordinary job.
        A caller that needs its job to run ahead of the ordinary queue --
        #256's accepted recapture is the one real use -- passes a lower
        integer (e.g. ``0``). Open Retrieval uses the ordinary FIFO priority;
        NORMAL and Hybrid out-of-pool captures do not gain a ``job_state``.

        ``profile`` (#258) is this capture's ``--profile`` gate, mirroring
        ``receive_capture``'s own ``profile`` parameter for blocks. It is
        only ever meaningful for an accepted, in-pool Hybrid claim (see
        ``accepted_hybrid_claim`` below). Open Retrieval has a job lifecycle
        but does not collect Hybrid profiling stages. Collection is gated here,
        not by a later display filter: when ``profile`` is ``False`` (the
        default) ``profile_queued_ns`` is never read from the clock and the
        durable ``profile_enabled`` column is written 0, so
        ``list_hybrid_profile_rows`` can never surface this row.
        """
        if captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        session = self._session_identity(session_number)
        source_path = Path(source)
        # Taken before the durable transaction below (mirrors
        # `receive_capture`'s own early `profile_queued_ns` read) so queue
        # wait includes the time this call spent decoding/hashing the image,
        # not just the time spent inside the DB transaction.
        profile_queued_ns = self._profile_clock_ns() if profile else None
        receive_started = perf_counter_ns()
        with observed(
            self.runtime_observer,
            "decode_load",
            result.block_id or "unresolved-slide",
        ):
            body = source_path.read_bytes()
            image = cv2.imdecode(
                np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        if image is None or image.ndim != 3:
            raise ValueError("slide capture is not a readable color image")
        del image
        checksum = hashlib.sha256(body).hexdigest()
        utc = captured_at.astimezone(timezone.utc)
        capture_id = _slide_capture_id(utc, checksum, result)
        capture_dir = session.directory / "slide_captures"
        capture_dir.mkdir(exist_ok=True)
        destination = capture_dir / f"{capture_id}.png"
        # Idempotent overwrite: safe to redo whether or not this call turns
        # out to be a ledger replay below, so it stays outside the txn.
        _atomic_bytes(destination, body)
        attempts_json = json.dumps(
            [asdict(attempt) for attempt in result.attempts], sort_keys=True
        )
        fingerprint = (
            self._fingerprint({
                "source": source_token or str(source),
                "captured_at": utc.isoformat(),
                "result": store_wire.encode(result),
                "duration_ms": duration_ms,
                "priority": priority,
                "profile": profile,
            })
            if request_id is not None else None
        )
        with self._connect() as db:
            # BEGIN IMMEDIATE up front (mirrors receive_capture ~L890-963):
            # two concurrent retries of the same request_id must not both
            # observe "new".
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "record_slide_capture", session_number,
                    fingerprint,
                )
                if hit:
                    return json.loads(cached)
                pending = db.execute(
                    "SELECT status FROM request_ledger WHERE session_number=? "
                    "AND method='record_slide_capture' AND request_id=?",
                    (session_number, request_id),
                ).fetchone()
            else:
                pending = None
            session_row = db.execute(
                "SELECT phase, session_mode FROM sessions WHERE session_number=?",
                (session_number,),
            ).fetchone()
            if session_row is None or session_row["phase"] != "slides":
                destination.unlink(missing_ok=True)
                raise RuntimeError("slide actions require slide mode")
            session_mode = str(session_row["session_mode"])
            open_work_order = db.execute(
                """SELECT work_order_id FROM work_orders
                   WHERE session_number=? AND lifecycle_state='capturing'
                   ORDER BY work_order_id DESC LIMIT 1""",
                (session_number,),
            ).fetchone()
            stamped_work_order_id = (
                int(open_work_order["work_order_id"])
                if open_work_order is not None else None
            )
            # #251: an Out-of-Pool Claim -- a Hybrid slide whose claimed
            # block is absent from ITS OWN work order's frozen Hybrid
            # Candidate Pool. Decided HERE, inside the same durable
            # transaction that already resolved `stamped_work_order_id`,
            # using a narrow `hybrid_pools` read scoped to that ONE work
            # order -- never `get_set`/`self.hybrid_pool()` (see
            # `_hybrid_pool_block_ids_for_out_of_pool_guard`'s docstring for
            # why both are wrong here). Only decided for Hybrid/Hybrid
            # Shadow: NORMAL/OPEN_RETRIEVAL keep today's exact behavior.
            out_of_pool_claim = (
                bool(result.success)
                and bool(result.block_id)
                and stamped_work_order_id is not None
                and session_mode in (
                    SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
                )
                and result.block_id not in self._hybrid_pool_block_ids_for_out_of_pool_guard(
                    db, session_number, stamped_work_order_id,
                )
            )
            # #252: the accept branch -- a successful, in-pool Hybrid claim.
            # Neither the `stamped_work_order_id is None` fork above (that's
            # NORMAL/unstamped inline scoring) nor `out_of_pool_claim` (an
            # identity mismatch) covers this shape, so before this it was
            # silently dropped: no verdict, no job, ever. `job_state='queued'`
            # is written in THIS SAME transaction as the capture row itself --
            # image bytes, decoded identity, frozen-pool association, and the
            # job record commit together, exactly as the issue requires.
            accepted_hybrid_claim = (
                bool(result.success)
                and bool(result.block_id)
                and stamped_work_order_id is not None
                and session_mode in (
                    SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
                )
                and not out_of_pool_claim
            )
            accepted_open_retrieval_claim = (
                bool(result.success)
                and bool(result.block_id)
                and stamped_work_order_id is not None
                and session_mode == SessionMode.OPEN_RETRIEVAL.value
            )
            accepted_retrieval_claim = (
                accepted_hybrid_claim or accepted_open_retrieval_claim
            )
            job_state = "queued" if accepted_retrieval_claim else None
            # #258: only an accepted, in-pool Hybrid claim ever gets a
            # `job_state`/worker to profile -- gated identically to
            # `job_state`/`priority` immediately above, by construction, not
            # by a later display filter. `profile_shadow` is stamped from the
            # session's own durable `session_mode`, which cannot change
            # mid-session, so it is unmistakable in the PERSISTED row from
            # the moment it is queued, not only once the worker finishes.
            profile_enabled = bool(profile and accepted_hybrid_claim)
            profile_is_shadow = session_mode == SessionMode.HYBRID_SHADOW.value
            if pending is None:
                db.execute(
                    """INSERT INTO slide_captures (
                   capture_id, session_number, captured_at, capture_path,
                   checksum, success, reason, raw_payload, payload_format,
                   block_id, slide_num, stain, work_order, email, genotype,
                   engine, symbology, preprocessing, duration_ms, attempts_json,
                   work_order_id, job_state, priority, profile_enabled,
                   profile_queued_ns, profile_current_stage, profile_shadow
                   ) VALUES (
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?
                   )""",
                    (
                        capture_id, session_number, utc.isoformat(), str(destination),
                        checksum, int(result.success), result.reason,
                        result.raw_payload, result.format, result.block_id,
                        result.slide_num, result.stain, result.lab_work_order,
                        result.email, result.genotype, result.engine,
                        result.symbology, result.preprocessing, float(duration_ms),
                        attempts_json, stamped_work_order_id, job_state,
                        priority if accepted_hybrid_claim else None,
                        int(profile_enabled),
                        profile_queued_ns if profile_enabled else None,
                        "queue_wait" if profile_enabled else None,
                        int(profile_is_shadow) if profile_enabled else None,
                    ),
                )
                db.execute(
                    "UPDATE sessions SET slide_recovery_state=? "
                    "WHERE session_number=?",
                    (
                        "waiting_for_removal" if result.success else "reposition",
                        session_number,
                    ),
                )
                if request_id is not None:
                    self._ledger_record(
                        db, request_id, "record_slide_capture", session_number,
                        fingerprint, json.dumps(capture_id), status="pending",
                    )
        if pending is None:
            kind = "slide_identity_validated" if result.success else "slide_identity_failed"
            message = result.reason if result.success else "Reposition slide"
            self._emit(
                session_number, kind, message, result.block_id, capture_id
            )
        self._record_slide_profile_stage(
            capture_id, "receive_persist_ms", receive_started
        )
        if result.success and result.block_id and stamped_work_order_id is None:
            # An unstamped NORMAL capture keeps the immediate claimed-pair
            # verification path. Retrieval modes always stamp a work order and
            # dispatch through the durable per-slide queue below.
            cascade_request_id = (
                f"{request_id}:claim" if request_id is not None else None
            )
            self.resolve_claim(
                session_number, result.block_id, capture_id, destination,
                request_id=cascade_request_id,
            )
        elif out_of_pool_claim:
            # #251: identity mismatch alone -- never scoring -- gets its
            # REVIEW immediately, same as the `stamped_work_order_id is
            # None` branch above, instead of deferring like an in-pool
            # Hybrid claim does today (left untouched for #252).
            cascade_request_id = (
                f"{request_id}:claim" if request_id is not None else None
            )
            self._reject_out_of_pool_claim(
                session_number, result.block_id, capture_id, destination,
                request_id=cascade_request_id,
            )
        elif accepted_retrieval_claim:
            # #252: the durable `job_state='queued'` row already committed
            # above; only the disposable in-memory Future is submitted here,
            # AFTER that transaction closed (mirrors `freeze_hybrid_pool`'s
            # post-commit call to `_bounce_hybrid_pool_to_blocks`) -- the
            # station acknowledges as soon as the durable commit lands, never
            # waiting on preparation/quality-gates/scoring, which all now run
            # inside `_score_hybrid_slide` instead of on this path.
            #
            # Gated on `pending is None`, mirroring `receive_capture`'s own
            # `prior`-row early return: a request_id retried after an earlier
            # attempt already inserted this row must not submit a second job.
            # A genuine process crash between the commit above and this
            # submit leaves the row `job_state='queued'` with no live Future
            # -- #255 (restart recovery), not this slice, is what resumes
            # that; `_recover_jobs` is deliberately left untouched for
            # `job_state='queued'` rows (see its own docstring).
            if pending is None and start_job:
                # `accepted_hybrid_claim` already guarantees both are set;
                # narrow the static types rather than widening the helper's
                # signature to Optional for a case that can never occur.
                assert stamped_work_order_id is not None
                assert result.block_id is not None
                self._submit_retrieval_scoring(
                    session, stamped_work_order_id, result.block_id, capture_id,
                    destination,
                    profile=profile_enabled, profile_queued_ns=profile_queued_ns,
                )
        if request_id is not None:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE request_ledger SET status='ok' "
                    "WHERE session_number=? AND method='record_slide_capture' "
                    "AND request_id=?",
                    (session_number, request_id),
                )
        return capture_id

    def retry_hybrid_slide(
        self, session_number: int, capture_id: str, *, request_id: str | None = None,
    ) -> bool:
        """#256: operator-triggered retry of a Hybrid Processing Error.

        Re-runs `_score_hybrid_slide` against the SAME durably saved capture
        bytes already on disk (`capture_path`) -- never a new photo, exactly
        the issue's "retries processing from the durable capture" contract.
        Only a row currently `job_state='error'` is eligible: a CAS from
        'error' to 'queued' both selects and claims the row atomically, so
        two concurrent retry taps can only ever resubmit once. Anything else
        (no such capture in this session, or a `job_state` that is not
        'error' -- already queued/scoring/complete/superseded, or not a
        Hybrid job at all) is a harmless no-op that returns False without
        raising.

        Guarded by the same request_ledger mechanism as every other mutating
        store call: a replayed retry tap (same request_id) returns the
        identical cached True/False without resubmitting a second job.
        """
        fingerprint = (
            self._fingerprint({"capture_id": capture_id}) if request_id is not None else None
        )
        accepted = False
        work_order_id: int | None = None
        block_id: str | None = None
        capture_path: str | None = None
        profile_enabled = False
        profile_queued_ns: int | None = None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "retry_hybrid_slide", session_number, fingerprint,
                )
                if hit:
                    return bool(json.loads(cached))
            row = db.execute(
                """SELECT work_order_id, block_id, capture_path, profile_enabled,
                          profile_queued_ns
                   FROM slide_captures
                   WHERE session_number=? AND capture_id=? AND job_state='error'""",
                (session_number, capture_id),
            ).fetchone()
            if row is not None:
                # #258: the row's own durable `profile_enabled` (set once at
                # its original enqueue -- see `record_slide_capture`) carries
                # forward unchanged; a retry re-runs the SAME durable job, it
                # does not re-decide whether to profile it. Its stage
                # indicator resets to 'queue_wait' exactly like restart
                # recovery's own reset, so a retried row never shows a stale
                # in-flight stage from the failed attempt.
                updated = db.execute(
                    """UPDATE slide_captures SET job_state='queued',
                       profile_current_stage=CASE WHEN profile_enabled=1
                           THEN 'queue_wait' ELSE profile_current_stage END
                       WHERE session_number=? AND capture_id=? AND job_state='error'""",
                    (session_number, capture_id),
                ).rowcount
                accepted = updated == 1
                if accepted:
                    work_order_id = int(row["work_order_id"])
                    block_id = str(row["block_id"])
                    capture_path = str(row["capture_path"])
                    profile_enabled = bool(row["profile_enabled"])
                    profile_queued_ns = row["profile_queued_ns"]
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "retry_hybrid_slide", session_number,
                    fingerprint, json.dumps(accepted),
                )
        if accepted:
            assert (
                work_order_id is not None and block_id is not None
                and capture_path is not None
            )
            session = self._session_identity(session_number)
            self._submit_hybrid_scoring(
                session, work_order_id, block_id, capture_id, Path(capture_path),
                profile=profile_enabled, profile_queued_ns=profile_queued_ns,
            )
            self._emit(
                session_number, "hybrid_slide_retry_queued",
                f"Retrying Hybrid processing for capture {capture_id}",
                block_id, capture_id,
            )
        return accepted

    def recapture_hybrid_slide(
        self,
        session_number: int,
        superseded_capture_id: str,
        source: str | Path,
        *,
        captured_at: datetime,
        result: SlideQRResult,
        duration_ms: float,
        request_id: str | None = None,
        source_token: str | None = None,
    ) -> RecaptureOutcome:
        """#256: an operator-accepted recapture for one Hybrid slide capture.

        Supersession of ``superseded_capture_id`` happens ONLY when
        ``result``'s decoded claim equals that row's OWN claimed block -- a
        structural check performed inside this method, never a caller
        convention -- so a recapture that decodes to a DIFFERENT block can
        never replace the wrong Results row (#256's explicit safety
        requirement). On a mismatch (a failed decode, a different
        ``block_id``, or no such durable capture at all) this method creates
        NO new job and leaves the superseded row, its ``job_state``, and its
        verdict completely untouched; it only returns a rejection message.

        On a match: a brand-new ``slide_captures`` row is inserted for the
        SAME work order the superseded capture belonged to -- deliberately
        NOT "whichever work order happens to be open right now". The banner
        that offers this route (`kiosk.attention.project_attention_banner`)
        only ever makes it actionable when no OTHER work order is actively
        capturing, so whenever this method is actually reached the currently
        open work order (if any) is already this same one; stamping to the
        superseded row's own ``work_order_id`` is what lets a recapture stay
        correct even once no work order is open at all (results already
        showing, between orders) -- `record_slide_capture`'s own "whichever
        work order is open" stamping would silently misfile the job in that
        case. The identity match also means the new claim is trivially
        already a member of the SAME frozen pool the superseded row
        belonged to (a frozen pool cannot change), so no Out-of-Pool Claim
        check is needed or performed here, unlike `record_slide_capture`.

        The new row gets ``job_state='queued'`` and ``priority=0`` -- #255's
        durable scheduling column -- so it runs ahead of the ordinary FIFO
        queue once the currently active job (never interrupted; the
        single-worker executor has no preemption) finishes.

        The superseded row is compare-and-set from whatever ``job_state`` it
        currently holds to ``'superseded'`` in the SAME transaction as the
        new row's insert, EXCEPT when it is already ``'superseded'`` (never
        re-fired). This is deliberately not restricted to ``job_state=
        'error'`` alone: a retry the operator triggered moments earlier may
        still be genuinely mid-flight (`'preparing'`/`'scoring'`) when the
        recapture is accepted, and superseding it regardless of its current
        state is what lets `_score_hybrid_slide`'s EXISTING #255 stale-write
        CAS (conditioned on `job_state` still being what that worker
        expects) silently drop that job's late write instead of clobbering
        the new, active result -- the concrete race #256's own acceptance
        criteria describes.

        The new row's background job is submitted only AFTER the
        transaction commits (mirrors `record_slide_capture`'s commit-then-
        submit shape).

        Guarded by the request ledger like every other mutating store call:
        a replayed recapture request (same request_id) returns the
        identical cached outcome without inserting a second row or
        re-running the CAS.
        """
        fingerprint = (
            self._fingerprint({
                "superseded_capture_id": superseded_capture_id,
                "source": source_token or str(source),
                "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
                "result": store_wire.encode(result),
                "duration_ms": duration_ms,
            })
            if request_id is not None else None
        )
        session: SessionIdentity | None = None
        old_work_order_id: int | None = None
        old_block_id: str | None = None
        new_capture_id: str | None = None
        destination: Path | None = None
        superseded = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "recapture_hybrid_slide", session_number, fingerprint,
                )
                if hit:
                    return store_wire.loads_as(RecaptureOutcome, cached)
            old_row = db.execute(
                """SELECT work_order_id, block_id, job_state, profile_enabled
                   FROM slide_captures
                   WHERE session_number=? AND capture_id=?""",
                (session_number, superseded_capture_id),
            ).fetchone()
            if old_row is None or old_row["job_state"] is None:
                outcome = RecaptureOutcome(
                    False, "no Hybrid job exists for that capture", None,
                )
                if request_id is not None:
                    self._ledger_record(
                        db, request_id, "recapture_hybrid_slide", session_number,
                        fingerprint, store_wire.dumps(outcome),
                    )
                return outcome
            old_work_order_id = int(old_row["work_order_id"])
            old_block_id = str(old_row["block_id"])
            identity_matches = bool(result.success) and result.block_id == old_block_id
            if not identity_matches:
                outcome = RecaptureOutcome(
                    False,
                    "recapture identity does not match the original slide "
                    "claim; the original result was not replaced",
                    None,
                )
                if request_id is not None:
                    self._ledger_record(
                        db, request_id, "recapture_hybrid_slide", session_number,
                        fingerprint, store_wire.dumps(outcome),
                    )
                return outcome
            session_row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,),
            ).fetchone()
            if session_row is None or session_row["phase"] != "slides":
                raise RuntimeError("slide actions require slide mode")
            source_path = Path(source)
            body = source_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3:
                raise ValueError("slide capture is not a readable color image")
            del image
            checksum = hashlib.sha256(body).hexdigest()
            utc = captured_at.astimezone(timezone.utc)
            new_capture_id = _slide_capture_id(utc, checksum, result)
            session = self._session_identity(session_number)
            capture_dir = session.directory / "slide_captures"
            capture_dir.mkdir(exist_ok=True)
            destination = capture_dir / f"{new_capture_id}.png"
            _atomic_bytes(destination, body)
            attempts_json = json.dumps(
                [asdict(attempt) for attempt in result.attempts], sort_keys=True
            )
            # #258: the NEW row inherits whether the superseded capture was
            # profiled -- the operator did not get a fresh `--profile` choice
            # for a recapture -- but gets its OWN fresh enqueue timestamp,
            # since this is genuinely a new queue period for a brand-new
            # capture_id, never the old row's already-elapsed one.
            recapture_profile_enabled = bool(old_row["profile_enabled"])
            recapture_profile_queued_ns = (
                self._profile_clock_ns() if recapture_profile_enabled else None
            )
            recapture_profile_is_shadow = (
                session.session_mode == SessionMode.HYBRID_SHADOW.value
            )
            db.execute(
                """INSERT INTO slide_captures (
                   capture_id, session_number, captured_at, capture_path,
                   checksum, success, reason, raw_payload, payload_format,
                   block_id, slide_num, stain, work_order, email, genotype,
                   engine, symbology, preprocessing, duration_ms, attempts_json,
                   work_order_id, job_state, priority, profile_enabled,
                   profile_queued_ns, profile_current_stage, profile_shadow
                   ) VALUES (
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?
                   )""",
                (
                    new_capture_id, session_number, utc.isoformat(), str(destination),
                    checksum, int(result.success), result.reason,
                    result.raw_payload, result.format, result.block_id,
                    result.slide_num, result.stain, result.lab_work_order,
                    result.email, result.genotype, result.engine,
                    result.symbology, result.preprocessing, float(duration_ms),
                    attempts_json, old_work_order_id, "queued", 0,
                    int(recapture_profile_enabled),
                    recapture_profile_queued_ns,
                    "queue_wait" if recapture_profile_enabled else None,
                    (
                        int(recapture_profile_is_shadow)
                        if recapture_profile_enabled else None
                    ),
                ),
            )
            superseded = db.execute(
                """UPDATE slide_captures SET job_state='superseded'
                   WHERE session_number=? AND capture_id=? AND job_state IS NOT NULL
                   AND job_state != 'superseded'""",
                (session_number, superseded_capture_id),
            ).rowcount == 1
            outcome = RecaptureOutcome(True, "recapture accepted", new_capture_id)
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "recapture_hybrid_slide", session_number,
                    fingerprint, store_wire.dumps(outcome),
                )
        if not superseded:
            self._log_durable(
                session_number, "hybrid_recapture_supersession_skipped",
                f"Recapture {new_capture_id} accepted for block {old_block_id}, "
                f"but superseded capture {superseded_capture_id} was already "
                "'superseded' by commit time; its row was left untouched.",
            )
        self._emit(
            session_number, "hybrid_slide_recaptured",
            f"Recapture accepted for block {old_block_id}", old_block_id, new_capture_id,
        )
        assert session is not None and destination is not None
        self._submit_hybrid_scoring(
            session, old_work_order_id, old_block_id, new_capture_id, destination,
            profile=recapture_profile_enabled,
            profile_queued_ns=recapture_profile_queued_ns,
        )
        return outcome

    def _reject_out_of_pool_claim(
        self,
        session_number: int,
        block_id: str,
        slide_capture_id: str,
        slide_path: str | Path,
        *,
        request_id: str | None = None,
    ) -> ClaimOutcome:
        """#251 Out-of-Pool Claim: write the immediate REVIEW verdict for a
        Hybrid slide whose claimed block is absent from ITS OWN work
        order's frozen Hybrid Candidate Pool (CONTEXT.md "Out-of-Pool
        Claim").

        Deliberately mirrors `resolve_claim`'s own KeyError-path shape --
        same verdict, same stage, the exact same reason string, and the
        same `_finalize_claim` write machinery -- but is called with
        `row=None` UNCONDITIONALLY, even when `block_id` DOES have a
        `sets` row in this session under a DIFFERENT work order (the
        isolation test's scenario). That row belongs to the other work
        order and must never be read or written by this claim: doing so
        would either corrupt the other work order's block row or, worse,
        actually run scoring against it -- reopening the exact
        cross-work-order leak #269 closed. No CV/preparation ever runs on
        this path; the REVIEW is reached by identity mismatch alone.
        """
        slide_path = Path(slide_path)
        fingerprint = (
            self._fingerprint({
                "block_id": block_id,
                "slide_capture_id": slide_capture_id,
                "slide_path": str(slide_path),
            })
            if request_id is not None else None
        )
        decision = ClaimDecision(
            claim_id=block_id,
            block_path="",
            slide_path=str(slide_path),
            verdict=VERDICT_REVIEW,
            stage="identity_lookup",
            reason="block id not found in session inventory",
        )
        preparation_started = perf_counter_ns()
        slide_img, slide_result = self._prepare_slide_for_artifacts(
            slide_path, block_id
        )
        self._record_slide_profile_stage(
            slide_capture_id, "slide_preparation_ms", preparation_started
        )
        return self._finalize_claim(
            session_number, block_id, slide_capture_id, slide_path, None, decision,
            slide_result=slide_result, slide_img=slide_img,
            request_id=request_id, fingerprint=fingerprint,
        )

    def resolve_claim(
        self,
        session_number: int,
        block_id: str,
        slide_capture_id: str,
        slide_path: str | Path,
        *,
        request_id: str | None = None,
    ) -> ClaimOutcome:
        """Turn one valid slide identity into an immediate durable verdict."""
        slide_path = Path(slide_path)
        fingerprint = None
        if request_id is not None:
            fingerprint = self._fingerprint({
                "block_id": block_id,
                "slide_capture_id": slide_capture_id,
                "slide_path": str(slide_path),
            })
            # Cheap, overwrite-proof replay check BEFORE any CV/scorer work:
            # a ledger hit means this exact request already produced a
            # durable verdict, so the original ClaimOutcome is returned
            # without re-running scoring or touching QC artifacts.
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                hit, cached = self._ledger_hit(
                    db, request_id, "resolve_claim", session_number, fingerprint
                )
            if hit:
                return store_wire.loads_as(ClaimOutcome, cached)
        lookup_started = perf_counter_ns()
        try:
            row = self.get_set(session_number, block_id)
        except KeyError:
            decision = ClaimDecision(
                claim_id=block_id,
                block_path="",
                slide_path=str(slide_path),
                verdict=VERDICT_REVIEW,
                stage="identity_lookup",
                reason="block id not found in session inventory",
            )
            preparation_started = perf_counter_ns()
            slide_img, slide_result = self._prepare_slide_for_artifacts(
                slide_path, block_id
            )
            self._record_slide_profile_stage(
                slide_capture_id, "slide_preparation_ms", preparation_started
            )
            return self._finalize_claim(
                session_number, block_id, slide_capture_id, slide_path, None, decision,
                slide_result=slide_result, slide_img=slide_img,
                request_id=request_id, fingerprint=fingerprint,
            )
        self._record_slide_profile_stage(
            slide_capture_id, "identity_lookup_ms", lookup_started
        )

        if row["verdict"] is not None:
            if row["slide_capture_id"] == slide_capture_id:
                session = self._session_identity(session_number)
                self._refresh_decisions_export(session)
                return ClaimOutcome(
                    True,
                    f"{row['verdict']}: {row['decision_reason']}",
                    str(row["verdict"]),
                    None if row["score"] is None else float(row["score"]),
                    str(row["decision_stage"]),
                    str(row["decision_reason"]),
                )
            self._emit(
                session_number, "slide_already_processed", "Slide already processed",
                block_id, slide_capture_id,
            )
            return ClaimOutcome(False, "Slide already processed")

        readiness = self._readiness_from_row(row)
        block_result: PreparedResult | None = None
        if not readiness.evaluable:
            # No evaluable claimed block means slide preprocessing cannot affect
            # the fail-closed verdict. Persist a failure artifact without calling
            # the scorer/preprocessor seam.
            img: np.ndarray | None = None
            slide_result: PreparedResult = PreparationFailure(
                role="slide", reason="slide preparation skipped: block is unusable"
            )
            decision = ClaimDecision(
                claim_id=block_id,
                block_path=str(row["capture_path"] or ""),
                slide_path=str(slide_path),
                verdict=VERDICT_REVIEW,
                stage="block_unusable",
                reason=readiness.review_reason or "block is not evaluable",
            )
        else:
            block_result = self._load_block_result(row)
            preparation_started = perf_counter_ns()
            img, slide_result = self._prepare_slide_for_artifacts(slide_path, block_id)
            self._record_slide_profile_stage(
                slide_capture_id, "slide_preparation_ms", preparation_started
            )
            decision = decide_claim(
                block_id, block_result, slide_result,
                block_path=str(row["capture_path"] or ""), slide_path=str(slide_path),
                observer=_SlideBenchmarkObserver(self, slide_capture_id),
            )
        outcome = self._finalize_claim(
            session_number, block_id, slide_capture_id, slide_path, row, decision,
            block_result=block_result, slide_result=slide_result, slide_img=img,
            request_id=request_id, fingerprint=fingerprint,
        )
        del img
        return outcome

    def _prepare_slide_for_artifacts(
        self, slide_path: Path, item_id: str, *, fail_closed: bool = True
    ) -> tuple[np.ndarray | None, PreparedResult]:
        """Prepare a readable slide for QA, independent of identity lookup.

        `fail_closed=True` (the default -- `resolve_claim` and
        `claim_out_of_pool_block` both rely on it) swallows a
        `self.slide_preprocessor` exception into a `PreparationFailure`: "a
        crash must never skip a verdict" for those callers, since a
        `PreparationFailure` still flows into `decide_claim`'s quality
        gates and lands as an ordinary REVIEW.

        `_score_hybrid_slide` passes `fail_closed=False`. It draws a
        deliberate line `resolve_claim` does not need: `cv2.imread`
        returning `None` (an expected, common capture problem) still comes
        back as a `PreparationFailure` -- never raises -- but a genuine
        `self.slide_preprocessor` exception is left to propagate, so its
        caller's own try/except can keep recording that as the unexpected
        SYSTEM failure `job_state='error'`, not a gate-failure REVIEW. See
        `_score_hybrid_slide`'s docstring for why that split matters there.
        """
        image: np.ndarray | None = None

        def _decode_and_preprocess() -> tuple[np.ndarray | None, PreparedResult]:
            nonlocal image
            image = cv2.imread(str(slide_path))
            if image is None:
                return None, PreparationFailure(
                    role="slide", reason=f"could not read image: {slide_path}"
                )
            return image, self.slide_preprocessor(image)

        with observed(self.runtime_observer, "slide_preparation", item_id):
            if not fail_closed:
                return _decode_and_preprocess()
            try:
                return _decode_and_preprocess()
            except Exception as exc:  # fail closed: a crash must never skip a verdict
                return image, PreparationFailure(role="slide", reason=str(exc))

    @staticmethod
    def _load_block_result(row: Mapping[str, object]) -> PreparedResult:
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return PreparationFailure(
                role="block",
                reason="stored block comparable mask could not be reopened",
            )
        metadata = json.loads(str(row["preprocessing_metadata"] or "{}"))
        return PreparedSpecimen(
            role="block",
            mask=mask,
            roi_ok=bool(metadata.get("roi_ok", True)),
            roi_reason=str(metadata.get("roi_reason", "")),
            segmentation_backend=str(metadata.get("segmentation_backend", "classical")),
        )

    def _finalize_claim(
        self,
        session_number: int,
        block_id: str,
        slide_capture_id: str,
        slide_path: Path,
        row: Mapping[str, object] | None,
        decision: ClaimDecision,
        *,
        block_result: PreparedResult | None = None,
        slide_result: PreparedResult | None = None,
        slide_img: np.ndarray | None = None,
        request_id: str | None = None,
        fingerprint: str | None = None,
        expected_job_state: str | None = None,
    ) -> ClaimOutcome:
        # Write the QC evidence first: if it raises, no verdict is committed,
        # so a retry is still possible instead of being locked out by the
        # "already processed" guard with no artifact ever produced.
        session = self._session_identity(session_number)
        artifact_dir = session.directory / "claim_artifacts"
        artifact_dir.mkdir(exist_ok=True)
        qc_path = artifact_dir / f"{slide_capture_id}_claim_qc.png"
        block_path = row["capture_path"] if row is not None else None
        block_img = cv2.imread(str(block_path)) if block_path else None
        qc_started = perf_counter_ns()
        with observed(
            self.runtime_observer, "verdict_qc_serialization", block_id
        ):
            self._write_claim_qc(
                block_img,
                slide_img,
                block_result if block_result is not None
                else PreparationFailure(role="block", reason="not attempted"),
                slide_result if slide_result is not None
                else PreparationFailure(role="slide", reason="not attempted"),
                decision,
                qc_path,
            )
            self._write_slide_artifacts(
                session,
                slide_capture_id,
                slide_path,
                slide_img,
                slide_result if slide_result is not None
                else PreparationFailure(role="slide", reason="not attempted"),
            )
        del block_img, slide_img
        self._record_slide_profile_stage(slide_capture_id, "qc_render_ms", qc_started)

        commit_started = perf_counter_ns()
        decided_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if row is not None:
                # #255 stale-write protection: `expected_job_state` is None
                # for every caller except `_score_hybrid_slide`, so this is
                # byte-identical to the pre-#255 single UPDATE for
                # resolve_claim/_reject_out_of_pool_claim/_score_work_order.
                # A Hybrid worker passes its own capture's `job_state` at the
                # moment it started scoring (normally "scoring"); the
                # correlated subquery re-reads THAT row's current
                # `job_state` atomically as part of this UPDATE's own WHERE
                # clause -- if a concurrent supersession (#256) changed it
                # in the meantime, the subquery no longer matches and this
                # write affects zero rows, exactly like the existing
                # `verdict IS NULL` duplicate guard already does for a
                # genuine double-decide. Without this, a stale (superseded)
                # job that happens to finish BEFORE the active job could win
                # the `verdict IS NULL` race and permanently block the
                # active job's own later, correct write.
                if expected_job_state is None:
                    updated = db.execute(
                        """UPDATE sets SET verdict=?, score=?, decision_stage=?,
                           decision_reason=?, slide_capture_id=?, decided_at=?
                           WHERE session_number=? AND block_id=? AND verdict IS NULL""",
                        (decision.verdict, decision.score, decision.stage,
                         decision.reason, slide_capture_id, decided_at,
                         session_number, block_id),
                    ).rowcount
                else:
                    updated = db.execute(
                        """UPDATE sets SET verdict=?, score=?, decision_stage=?,
                           decision_reason=?, slide_capture_id=?, decided_at=?
                           WHERE session_number=? AND block_id=? AND verdict IS NULL
                             AND (SELECT job_state FROM slide_captures
                                  WHERE capture_id=?) = ?""",
                        (decision.verdict, decision.score, decision.stage,
                         decision.reason, slide_capture_id, decided_at,
                         session_number, block_id, slide_capture_id,
                         expected_job_state),
                    ).rowcount
                if updated != 1:
                    return ClaimOutcome(False, "Slide already processed")
                # `sets` is the canonical verdict home once a set exists; only
                # persist the qc pointer here to avoid a second, driftable copy.
                db.execute(
                    "UPDATE slide_captures SET claim_qc_path=? WHERE capture_id=?",
                    (str(qc_path), slide_capture_id),
                )
            else:
                # No set exists to hold this verdict durably, so slide_captures
                # is the only durable home for it. Same stale-write guard as
                # above, added directly to this row's own WHERE clause since
                # it IS the row being written (no correlated subquery needed).
                if expected_job_state is None:
                    db.execute(
                        """UPDATE slide_captures SET verdict=?, claim_score=?,
                           claim_stage=?, claim_reason=?, claim_qc_path=?,
                           claim_decided_at=?
                           WHERE capture_id=?""",
                        (decision.verdict, decision.score, decision.stage,
                         decision.reason, str(qc_path), decided_at,
                         slide_capture_id),
                    )
                else:
                    updated = db.execute(
                        """UPDATE slide_captures SET verdict=?, claim_score=?,
                           claim_stage=?, claim_reason=?, claim_qc_path=?,
                           claim_decided_at=?
                           WHERE capture_id=? AND job_state=?""",
                        (decision.verdict, decision.score, decision.stage,
                         decision.reason, str(qc_path), decided_at,
                         slide_capture_id, expected_job_state),
                    ).rowcount
                    if updated != 1:
                        return ClaimOutcome(False, "Slide already processed")
            outcome = ClaimOutcome(
                True, f"{decision.verdict}: {decision.reason}",
                decision.verdict, decision.score, decision.stage, decision.reason,
            )
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "resolve_claim", session_number,
                    fingerprint, store_wire.dumps(outcome),
                )
        self._refresh_decisions_export(session)
        self._record_slide_profile_stage(
            slide_capture_id, "verdict_commit_export_ms", commit_started
        )
        kind = "claim_pass" if decision.verdict == VERDICT_PASS else "claim_review"
        self._emit(
            session_number, kind, f"{decision.verdict}: {decision.reason}",
            block_id, slide_capture_id,
        )
        return outcome

    def _write_claim_qc(
        self,
        block_img: np.ndarray | None,
        slide_img: np.ndarray | None,
        block_result: PreparedResult,
        slide_result: PreparedResult,
        decision: ClaimDecision,
        destination: Path,
    ) -> None:
        self._contact_sheet_renderer(
            block_img=block_img,
            slide_img=slide_img,
            block_result=block_result,
            slide_result=slide_result,
            decision=decision,
            output_path=destination,
        )

    def _write_claim_slide_overlay(
        self,
        session: SessionIdentity,
        capture_id: str,
        *,
        block_img: np.ndarray | None,
        slide_img: np.ndarray | None,
        block_result: PreparedResult,
        slide_result: PreparedResult,
        decision: ClaimDecision,
    ) -> None:
        # Block/Slide thumbs+display need only raw captures (#236). Overlay
        # PNG/JPEG additionally need prepared masks + a locked pose.
        if block_img is None and slide_img is None:
            return

        from kiosk.images import (
            CLAIM_DISPLAY_MAX_LONG_EDGE,
            CLAIM_THUMB_MAX_LONG_EDGE,
            encode_downscaled_jpeg,
        )

        artifact_dir = session.directory / "claim_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        jpeg_jobs: list[tuple[str, np.ndarray, int]] = []
        if block_img is not None:
            jpeg_jobs.extend(
                (
                    (f"{capture_id}_block_thumb.jpg", block_img, CLAIM_THUMB_MAX_LONG_EDGE),
                    (
                        f"{capture_id}_block_display.jpg",
                        block_img,
                        CLAIM_DISPLAY_MAX_LONG_EDGE,
                    ),
                )
            )
        if slide_img is not None:
            jpeg_jobs.extend(
                (
                    (f"{capture_id}_slide_thumb.jpg", slide_img, CLAIM_THUMB_MAX_LONG_EDGE),
                    (
                        f"{capture_id}_slide_display.jpg",
                        slide_img,
                        CLAIM_DISPLAY_MAX_LONG_EDGE,
                    ),
                )
            )
        for filename, source, max_long_edge in jpeg_jobs:
            (artifact_dir / filename).write_bytes(
                encode_downscaled_jpeg(source, max_long_edge=max_long_edge)
            )

        if block_img is None or slide_img is None:
            return
        if not isinstance(block_result, PreparedSpecimen):
            return
        if not isinstance(slide_result, PreparedSpecimen):
            return
        if decision.best_angle is None or decision.best_flip is None:
            return

        overlay = build_slide_image_overlay(
            block_img,
            slide_img,
            block_result.mask,
            slide_result.mask,
            decision.best_angle,
            bool(decision.best_flip),
        )
        overlay_path = artifact_dir / f"{capture_id}_slide_overlay.png"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(f"could not write slide overlay: {overlay_path}")
        (artifact_dir / f"{capture_id}_overlay_display.jpg").write_bytes(
            encode_downscaled_jpeg(overlay, max_long_edge=CLAIM_DISPLAY_MAX_LONG_EDGE)
        )

    def _refresh_decisions_export(self, session: SessionIdentity) -> None:
        with self._connect() as db:
            rows = db.execute(
                """SELECT block_id, verdict, score, stage, reason, decided_at
                   FROM (
                       SELECT block_id, verdict, score, decision_stage AS stage,
                              decision_reason AS reason, decided_at
                       FROM sets
                       WHERE session_number=? AND verdict IS NOT NULL
                       UNION ALL
                       SELECT sc.block_id, sc.verdict, sc.claim_score,
                              sc.claim_stage, sc.claim_reason, sc.claim_decided_at
                       FROM slide_captures AS sc
                       WHERE sc.session_number=? AND sc.verdict IS NOT NULL
                         AND NOT EXISTS (
                             SELECT 1 FROM sets AS s
                             WHERE s.session_number=sc.session_number
                               AND s.block_id=sc.block_id
                         )
                   ) ORDER BY decided_at, block_id""",
                (session.number, session.number),
            ).fetchall()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["block_id", "verdict", "score", "stage", "reason", "decided_at"])
        for row in rows:
            writer.writerow([
                row["block_id"], row["verdict"],
                "" if row["score"] is None else f"{row['score']:.4f}",
                row["stage"], row["reason"], row["decided_at"],
            ])
        _atomic_bytes(
            session.directory / "decisions.csv", buffer.getvalue().encode("utf-8")
        )

    def _recover_claims(self) -> None:
        """Finish valid slide claims interrupted after durable capture storage."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT sc.session_number, sc.capture_id, sc.block_id,
                          sc.capture_path
                   FROM slide_captures AS sc
                   LEFT JOIN sets AS s
                     ON s.session_number=sc.session_number
                    AND s.block_id=sc.block_id
                   WHERE sc.success=1 AND sc.block_id IS NOT NULL
                     AND sc.verdict IS NULL AND sc.work_order_id IS NULL
                     AND (s.block_id IS NULL OR (
                         s.verdict IS NULL
                         AND s.preprocessing_status IN ('complete', 'unusable')
                     ))
                   ORDER BY sc.captured_at, sc.capture_id"""
            ).fetchall()
        for row in rows:
            self.resolve_claim(
                int(row["session_number"]),
                str(row["block_id"]),
                str(row["capture_id"]),
                Path(row["capture_path"]),
            )
        with self._connect() as db:
            decided_sessions = db.execute(
                """SELECT session_number FROM sets WHERE verdict IS NOT NULL
                   UNION
                   SELECT session_number FROM slide_captures
                   WHERE verdict IS NOT NULL"""
            ).fetchall()
        for row in decided_sessions:
            self._refresh_decisions_export(
                self._session_identity(int(row["session_number"]))
            )

    def slide_captures(self, session_number: int) -> tuple[dict[str, object], ...]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM slide_captures WHERE session_number=?
                   ORDER BY captured_at, capture_id""",
                (session_number,),
            ).fetchall()
        captures = []
        for row in rows:
            capture = dict(row)
            capture["attempts"] = json.loads(str(capture.pop("attempts_json")))
            captures.append(capture)
        return tuple(captures)

    def get_slide_capture(self, session_number: int, capture_id: str) -> dict[str, object]:
        """Mirror of ``get_set`` for one slide capture row (mirrors #93 pattern)."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM slide_captures WHERE session_number=? AND capture_id=?",
                (session_number, capture_id),
            ).fetchone()
        if row is None:
            raise KeyError(capture_id)
        capture = dict(row)
        capture["attempts"] = json.loads(str(capture.pop("attempts_json")))
        return capture

    # -----------------------------------------------------------------
    # Work-order-scoped N^2 retrieval mode (#149, ADR 0009).
    #
    # A work order is the operator's explicit start/finish capture bracket.
    # `scan_block`/`record_slide_capture` stamp every row created while one
    # is open (`lifecycle_state='capturing'`) with its id; captures under an
    # open work order defer their verdict until `finish_work_order` dispatches
    # a batch N^2 scoring job on the executor (mirrors `_submit_preprocessing`).
    # -----------------------------------------------------------------

    def start_work_order(self, session_number: int) -> int:
        """Open a new capture bracket; returns the durable work_order_id.

        Idempotent: a double-tap (retry, or a Pi restart landing mid-request)
        reuses the SAME id of an already-open bracket rather than orphaning a
        second `capturing` row, mirroring `finish_work_order`'s existing
        SELECT-before-write guard.

        #269 FIX3: `scan_block` stamps a new `sets` row's `work_order_id`
        from whichever work order happens to be open AT SCAN TIME -- `NULL`
        if none is. A block scanned (and even fully captured/prepared)
        before this call has no way back into a bracket on its own:
        re-scanning the same `block_id` fails ("Block already scanned", the
        `(session_number, block_id)` primary key), and `unscan_block`
        requires `capture_id IS NULL`, which a captured block never has --
        so without this backfill those rows are stranded, invisible to
        every freeze's `AND work_order_id=?` candidate filter, forever. This
        adopts this session's not-yet-verdicted (`verdict IS NULL`),
        unclaimed (`work_order_id IS NULL`) rows into the bracket this call
        just opened. It never re-parents a row that already belongs to
        ANY work order (open or closed) -- `work_order_id IS NULL` excludes
        those -- and never touches a row a slide has already claimed a
        verdict against (`verdict IS NULL` excludes those).
        """
        started_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """SELECT work_order_id FROM work_orders
                   WHERE session_number=? AND lifecycle_state='capturing'
                   ORDER BY work_order_id DESC LIMIT 1""",
                (session_number,),
            ).fetchone()
            if existing is not None:
                return int(existing["work_order_id"])
            previous = db.execute(
                "SELECT 1 FROM work_orders WHERE session_number=? LIMIT 1",
                (session_number,),
            ).fetchone()
            if previous is not None:
                # The ambient kiosk session remains alive between orders. A
                # newly opened bracket begins a fresh block -> slide capture
                # cycle instead of inheriting the prior order's slide phase.
                db.execute(
                    "UPDATE sessions SET phase='blocks' "
                    "WHERE session_number=? AND phase='slides'",
                    (session_number,),
                )
            cursor = db.execute(
                """INSERT INTO work_orders(session_number, lifecycle_state, started_at)
                   VALUES (?, 'capturing', ?)""",
                (session_number, started_at),
            )
            work_order_id = int(cursor.lastrowid)
            db.execute(
                """UPDATE sets SET work_order_id=?
                   WHERE session_number=? AND work_order_id IS NULL
                   AND verdict IS NULL""",
                (work_order_id, session_number),
            )
        self._emit(session_number, "work_order_started", "Work order started")
        return work_order_id

    def open_work_order_id(self, session_number: int) -> int | None:
        """Read helper: the currently `capturing` work order's id, or None."""
        with self._connect() as db:
            row = db.execute(
                """SELECT work_order_id FROM work_orders
                   WHERE session_number=? AND lifecycle_state='capturing'
                   ORDER BY work_order_id DESC LIMIT 1""",
                (session_number,),
            ).fetchone()
        return int(row["work_order_id"]) if row is not None else None

    def has_work_orders(self, session_number: int) -> bool:
        """Return whether this session has ever opened a work order."""
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM work_orders WHERE session_number=? LIMIT 1",
                (session_number,),
            ).fetchone()
        return row is not None

    def finish_work_order(
        self, session_number: int, *, start_job: bool = True,
        request_id: str | None = None,
    ) -> int:
        """Close the open capture bracket and dispatch mode-specific final work.

        NORMAL dispatches its legacy batch scorer. Open Retrieval slides were
        already scored by durable per-slide jobs, so Finish only queues the
        aggregate CSV/results-ready finalizer after those jobs. Hybrid modes
        already own their per-slide result lifecycle and remain finalized.

        #269: HYBRID/HYBRID_SHADOW sessions must never run the full N-by-N
        path -- the whole point of the Hybrid Candidate Pool. The
        `capturing -> finalized` commit always happens (closing the bracket
        is mode-agnostic), but the `finalized -> scoring` commit and the job
        submission are gated on the durable `sessions.session_mode`, read
        directly rather than inferred from whether a `hybrid_pools` row
        happens to exist (see the design-gap note this fixes: a Hybrid work
        order that failed to freeze -- fewer than 2 usable blocks -- has no
        `hybrid_pools` row at all, so an existence check would wrongly fall
        through to full N-by-N scoring in exactly the mode that exists to
        avoid it). A Hybrid work order therefore stays `finalized` after this
        call -- driving it to a scored/results-ready state is #251/#252's job.
        """
        fingerprint = (
            self._fingerprint({"start_job": bool(start_job)})
            if request_id is not None else None
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, cached = self._ledger_hit(
                    db, request_id, "finish_work_order", session_number, fingerprint
                )
                if hit:
                    return int(json.loads(cached))
            row = db.execute(
                """SELECT work_order_id FROM work_orders
                   WHERE session_number=? AND lifecycle_state='capturing'
                   ORDER BY work_order_id DESC LIMIT 1""",
                (session_number,),
            ).fetchone()
            if row is None:
                raise ValueError("no open work order to finish")
            work_order_id = int(row["work_order_id"])
            db.execute(
                "UPDATE work_orders SET lifecycle_state='finalized' "
                "WHERE work_order_id=?",
                (work_order_id,),
            )
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "finish_work_order", session_number,
                    fingerprint, json.dumps(work_order_id),
                )
        session_mode = self._session_mode(session_number)
        if session_mode == SessionMode.OPEN_RETRIEVAL.value:
            if start_job:
                self._submit_open_retrieval_finalization(
                    self._session_identity(session_number), work_order_id
                )
            return work_order_id
        if session_mode in (
            SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
        ):
            return work_order_id
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE work_orders SET lifecycle_state='scoring' "
                "WHERE work_order_id=? AND lifecycle_state='finalized'",
                (work_order_id,),
            )
        if start_job:
            session = self._session_identity(session_number)
            job = self._executor.submit(self._score_work_order, session, work_order_id)
            with self._jobs_lock:
                self._jobs.append(job)
        return work_order_id

    def _submit_open_retrieval_finalization(
        self, session: SessionIdentity, work_order_id: int
    ) -> None:
        """Queue audit finalization after this order's per-slide jobs."""
        job = self._executor.submit(
            self._finalize_open_retrieval_work_order, session, work_order_id
        )
        with self._jobs_lock:
            self._jobs.append(job)

    def _finalize_open_retrieval_work_order(
        self, session: SessionIdentity, work_order_id: int
    ) -> None:
        """Write the aggregate CSV without rescoring any completed slide."""
        try:
            with self._connect() as db:
                state = db.execute(
                    """SELECT lifecycle_state FROM work_orders
                       WHERE session_number=? AND work_order_id=?""",
                    (session.number, work_order_id),
                ).fetchone()
                active_jobs = db.execute(
                    """SELECT COUNT(*) AS count FROM slide_captures
                       WHERE session_number=? AND work_order_id=?
                         AND job_state IN ('queued', 'preparing', 'scoring')""",
                    (session.number, work_order_id),
                ).fetchone()["count"]
            if state is None or state["lifecycle_state"] != "finalized":
                return
            if int(active_jobs):
                return

            with self._connect() as db:
                rows = db.execute(
                    """SELECT sc.capture_id, sc.block_id,
                              COALESCE(s.verdict, sc.verdict) AS verdict,
                              COALESCE(s.decision_reason, sc.claim_reason) AS reason,
                              COALESCE(s.score, sc.claim_score) AS claim_score,
                              sc.match_margin, sc.top_block, sc.near_miss_blocks,
                              sc.job_state
                       FROM slide_captures AS sc
                       LEFT JOIN sets AS s
                         ON s.session_number=sc.session_number
                        AND s.slide_capture_id=sc.capture_id
                       WHERE sc.session_number=? AND sc.work_order_id=?
                         AND sc.success=1 AND sc.block_id IS NOT NULL
                         AND sc.job_state IS NOT 'superseded'
                       ORDER BY sc.captured_at, sc.capture_id""",
                    (session.number, work_order_id),
                ).fetchall()

            csv_rows: list[
                tuple[str, str, WorkOrderVerdict, float | None]
            ] = []
            for row in rows:
                durable_verdict = str(row["verdict"] or "ERROR")
                reason = str(
                    row["reason"]
                    or (
                        "retrieval slide job did not complete"
                        if row["job_state"] != "complete"
                        else ""
                    )
                )
                csv_rows.append((
                    str(row["capture_id"]),
                    str(row["block_id"]),
                    WorkOrderVerdict(
                        verdict=durable_verdict,
                        reason=reason,
                        match_margin=row["match_margin"],
                        top_block=row["top_block"],
                        near_miss_blocks=frozenset(
                            value for value in str(row["near_miss_blocks"] or "").split(",")
                            if value
                        ),
                    ),
                    row["claim_score"],
                ))

            csv_path = self._write_work_order_csv(session, work_order_id, csv_rows)
            sheets_dir = (
                session.directory / "work_orders"
                / f"work_order_{work_order_id:06d}_sheets"
            )
            contact_sheet_dir = (
                str(sheets_dir)
                if sheets_dir.is_dir() and any(sheets_dir.iterdir())
                else None
            )
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE work_orders SET lifecycle_state='results_ready',
                       finished_at=?, verdict_csv_path=?, contact_sheet_dir=?,
                       failure_reason=NULL
                       WHERE session_number=? AND work_order_id=?
                         AND lifecycle_state='finalized'""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        str(csv_path),
                        contact_sheet_dir,
                        session.number,
                        work_order_id,
                    ),
                )
            self._emit(
                session.number, "work_order_results_ready", "Work order results ready"
            )
        except Exception as exc:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE work_orders SET lifecycle_state='scoring_failed',
                       failure_reason=? WHERE session_number=? AND work_order_id=?""",
                    (str(exc), session.number, work_order_id),
                )
            self._emit(session.number, "work_order_scoring_failed", str(exc))

    def get_work_order(self, session_number: int, work_order_id: int) -> dict[str, object]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM work_orders WHERE session_number=? AND work_order_id=?",
                (session_number, work_order_id),
            ).fetchone()
        if row is None:
            raise KeyError(work_order_id)
        return dict(row)

    def list_results_ready_work_orders(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        """The kiosk results table's data source (#150): one row per slide,
        aggregated across EVERY ``results_ready`` work order in the session
        -- not a single-order picker -- so any order finished this session
        is viewable without re-scanning. Pure read of columns #149's
        ``_finalize_claim`` already populates; no new schema, no CSV
        re-parsing. ``_finalize_claim`` writes the verdict to ``sets`` (the
        canonical home) when a matching set row exists, and only falls back
        to ``slide_captures`` when it doesn't (no set to durably hold it) --
        so this reads both and prefers ``sets``. A work order still
        ``capturing``/``finalized``/``scoring`` contributes no rows until it
        reaches ``results_ready``.
        """
        with self._connect() as db:
            rows = db.execute(
                """SELECT sc.capture_id AS capture_id, sc.block_id AS block_id,
                          COALESCE(s.verdict, sc.verdict) AS verdict,
                          COALESCE(s.score, sc.claim_score) AS claim_score,
                          COALESCE(s.decision_reason, sc.claim_reason)
                              AS claim_reason,
                          sc.work_order_id AS work_order_id,
                          sc.work_order AS lab_work_order,
                          sc.work_order AS work_order,
                          sc.top_block AS top_block,
                          wo.contact_sheet_dir AS contact_sheet_dir
                   FROM slide_captures AS sc
                   JOIN work_orders AS wo
                     ON wo.work_order_id = sc.work_order_id
                    AND wo.session_number = sc.session_number
                   LEFT JOIN sets AS s
                     ON s.session_number = sc.session_number
                    AND s.slide_capture_id = sc.capture_id
                   WHERE sc.session_number=? AND wo.lifecycle_state='results_ready'
                   ORDER BY sc.captured_at, sc.capture_id""",
                (session_number,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def list_retrieval_results(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        """Live kiosk rows for per-slide retrieval jobs.

        One row per successfully decoded Open Retrieval, Hybrid, or Hybrid
        Shadow slide claim in the session,
        across EVERY work order (``capturing``, ``finalized``, or
        ``results_ready``), not just ``results_ready`` ones. Failed/unresolved
        captures remain durable capture-recovery audit data, not result rows.

        This is separate from `list_results_ready_work_orders`: live rows are
        intentionally visible while their work order is still capturing or
        finalizing, while the older method remains lifecycle-gated.

        Same projected key set as `list_results_ready_work_orders` (so
        ``code/kiosk/results_table.py:project_results_table`` consumes
        either unchanged): ``capture_id``, ``block_id``, ``verdict``,
        ``claim_score``, ``claim_reason``, ``work_order_id``,
        ``lab_work_order``, ``top_block``, ``contact_sheet_dir``. The legacy
        ``work_order`` projection remains as a backward-compatible alias for
        ``lab_work_order``.

        ``verdict`` is a PROJECTION of the row's internal ``job_state``,
        never the durable column itself::

            job_state='error'                             -> "ERROR"
            job_state in ('queued', 'preparing', 'scoring') -> "PENDING"
            otherwise (job_state IS NULL, e.g. an Out-of-Pool Claim, or
            job_state='complete')                         -> the durable
                                                              PASS/REVIEW
                                                              verdict

        ``sets.verdict``/``slide_captures.verdict`` are NEVER written as
        ``"ERROR"`` or ``"PENDING"`` -- those two strings exist only in this
        projection. Internal lifecycle states (``queued``, ``preparing``,
        ``scoring``, ``superseded``) are durable and must never reach a
        caller of this method; only ``"ERROR"``/``"PENDING"``/``"PASS"``/
        ``"REVIEW"`` ever appear under the ``verdict`` key. A row currently
        ``job_state='superseded'`` (#256 recapture supersession, which
        deliberately permits superseding an already-``complete`` row too --
        see ``recapture_hybrid_slide``'s own docstring) is EXCLUDED from
        this projection entirely by its own ``WHERE`` clause below, rather
        than being mapped to any of the four strings above: showing its
        stale ``PASS``/``REVIEW`` verdict would be indistinguishable from
        the active result and could read as a second, contradictory row for
        the same block once the recapture's own row lands. A block whose
        only capture is mid-recapture instead shows no row at all for one
        moment (until the new capture's row reaches at least ``queued``,
        which happens in the same transaction) -- a visibly incomplete
        table an operator will wait out, not a wrong verdict they could act
        on, so this is the safer of the two failure shapes for this
        projection.

        Gated on the durable ``sessions.session_mode`` (`_session_mode`,
        #269's fail-closed lookup) rather than any per-row signal. NORMAL
        always returns an empty tuple.

        MUST NOT RAISE. This is polled from the kiosk camera loop
        (mirrors `_hybrid_pool_block_ids_for_out_of_pool_guard`'s own
        documented blast-radius rule): a raise here would reach
        `_camera_loop`'s bare ``except Exception``, set ``self._stop``, and
        kill the camera loop. Any unexpected failure -- including
        `_session_mode`'s own deliberate ``ValueError`` for an unknown
        session, since this method cannot prove no poll path will ever pass
        a stale/unknown ``session_number`` -- degrades to an empty tuple via
        `_log_durable_exception`, so the failure stays diagnosable instead
        of silently swallowed.
        """
        try:
            mode = self._session_mode(session_number)
            if mode not in (
                SessionMode.OPEN_RETRIEVAL.value,
                SessionMode.HYBRID.value,
                SessionMode.HYBRID_SHADOW.value,
            ):
                return ()
            with self._connect() as db:
                rows = db.execute(
                    """SELECT sc.capture_id AS capture_id, sc.block_id AS block_id,
                              CASE
                                  WHEN sc.job_state='error' THEN 'ERROR'
                                  WHEN sc.job_state IN ('queued', 'preparing', 'scoring')
                                      THEN 'PENDING'
                                  ELSE COALESCE(s.verdict, sc.verdict)
                              END AS verdict,
                              COALESCE(s.score, sc.claim_score) AS claim_score,
                              COALESCE(s.decision_reason, sc.claim_reason)
                                  AS claim_reason,
                              sc.work_order_id AS work_order_id,
                              sc.work_order AS lab_work_order,
                              sc.work_order AS work_order,
                              sc.top_block AS top_block,
                              wo.contact_sheet_dir AS contact_sheet_dir
                       FROM slide_captures AS sc
                       JOIN work_orders AS wo
                         ON wo.work_order_id = sc.work_order_id
                        AND wo.session_number = sc.session_number
                       LEFT JOIN sets AS s
                         ON s.session_number = sc.session_number
                        AND s.slide_capture_id = sc.capture_id
                       WHERE sc.session_number=? AND sc.success=1
                         AND sc.block_id IS NOT NULL
                         AND sc.job_state IS NOT 'superseded'
                       ORDER BY sc.captured_at, sc.capture_id""",
                    (session_number,),
                ).fetchall()
            return tuple(dict(row) for row in rows)
        except Exception as exc:  # fail closed: never let a poll loop die on this
            self._log_durable_exception(
                session_number, "list_hybrid_results_failed",
                f"list_hybrid_results raised for session {session_number}; "
                "degrading to an empty tuple so the kiosk poll loop survives",
                exc,
            )
            return ()

    def list_hybrid_results(self, session_number: int) -> tuple[dict[str, object], ...]:
        """Backward-compatible Hybrid-only view of live retrieval rows."""
        try:
            if self._session_mode(session_number) not in (
                SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
            ):
                return ()
            return self.list_retrieval_results(session_number)
        except Exception as exc:
            self._log_durable_exception(
                session_number, "list_retrieval_results_failed",
                f"list_retrieval_results raised for session {session_number}; "
                "degrading to an empty tuple so the kiosk poll loop survives",
                exc,
            )
            return ()

    def list_hybrid_profile_rows(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        """#258: the shared pure formatter's (`code/session/profile_report.py`
        `project_profile_rows`) one durable source -- one row per Hybrid
        slide capture PROFILED in this session (`profile_enabled=1`), across
        every work order, exactly like `list_hybrid_results` above but
        scoped to `--profile` captures only. Copies that method's structure
        and hardening deliberately, rather than sharing a query, because the
        two projections diverge (`verdict`/`claim_score`/... vs raw
        `job_state`/timing columns) and each must stay independently correct.

        Returns an empty tuple -- never collects, never persists anything
        additional, never raises -- whenever:

        - the session is NORMAL/OPEN_RETRIEVAL (gated on the durable
          `sessions.session_mode`, mirroring `list_hybrid_results`'s own
          `_session_mode` gate exactly, not a per-row signal);
        - profiling was never turned on for a given row (`profile_enabled=1`
          is a WHERE clause here, not a display-time filter) -- this is what
          makes "no `--profile` means nothing is collected or persisted"
          true by construction: a row this method would otherwise return
          simply was never written with its timing columns populated, by
          `record_slide_capture`/`_score_hybrid_slide` (see their own
          docstrings).

        Each returned dict has exactly the keys `project_profile_rows`
        reads: `capture_id`, `block_id`, `job_state`, `verdict`, `stage`
        (this row's `profile_current_stage` while pending; NULL once
        complete or if never profiled), `queued_ns` (`profile_queued_ns`),
        `total_ms` (`profile_total_ms`), `stage_ms_json`
        (`profile_stage_ms_json`), and `shadow` (`profile_shadow`, stamped at
        enqueue from the session's own durable `session_mode` -- see the
        `slide_captures.profile_shadow` migration comment -- so a shadow
        row's complete-pool timing is unmistakable in this PERSISTED
        projection, not only once `project_profile_rows` labels it on
        screen).

        MUST NOT RAISE. This is polled from the kiosk camera loop exactly
        like `list_hybrid_results` (see that method's own docstring for the
        concrete blast radius: a raise here would reach `_camera_loop`'s
        bare ``except Exception``, set ``self._stop``, and kill the camera
        loop). Any unexpected failure degrades to an empty tuple via
        `self._log_durable_exception`, same as `list_hybrid_results`.
        """
        try:
            mode = self._session_mode(session_number)
            if mode not in (
                SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
            ):
                return ()
            with self._connect() as db:
                rows = db.execute(
                    """SELECT capture_id, block_id, job_state, verdict,
                              profile_current_stage AS stage,
                              profile_queued_ns AS queued_ns,
                              profile_total_ms AS total_ms,
                              profile_stage_ms_json AS stage_ms_json,
                              profile_shadow AS shadow
                       FROM slide_captures
                       WHERE session_number=? AND profile_enabled=1
                       ORDER BY captured_at, capture_id""",
                    (session_number,),
                ).fetchall()
            return tuple(dict(row) for row in rows)
        except Exception as exc:  # fail closed: never let a poll loop die on this
            self._log_durable_exception(
                session_number, "list_hybrid_profile_rows_failed",
                f"list_hybrid_profile_rows raised for session {session_number}; "
                "degrading to an empty tuple so the kiosk poll loop survives",
                exc,
            )
            return ()

    def _recover_work_orders(self) -> None:
        """Recover batch scoring or Open Retrieval audit finalization.

        #269: gated on the persisted `sessions.session_mode`, excluding
        HYBRID/HYBRID_SHADOW, for the exact reason `finish_work_order` is --
        this runs at `ProcessingStore.__init__`/`_initialize` time, in a
        freshly started process with no in-memory `SessionWorkflow`/
        `PiCaptureRuntime` (and therefore no in-memory `SessionMode`)
        surviving the restart. A Hybrid work order that never froze a pool
        (fewer than 2 usable blocks) is deliberately, permanently `finalized`
        -- never crashed mid-transition -- and the old blanket
        `WHERE lifecycle_state='finalized'` sweep would wrongly resume it
        into full N-by-N scoring on every restart. `sessions.session_mode` is
        the only signal available here that distinguishes NORMAL batch work,
        Open Retrieval's per-slide finalizer, and Hybrid work orders correctly
        waiting on their own per-slide lifecycle.
        """
        _per_slide_retrieval_modes = (
            SessionMode.OPEN_RETRIEVAL.value,
            SessionMode.HYBRID.value,
            SessionMode.HYBRID_SHADOW.value,
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE work_orders SET lifecycle_state='scoring'
                   WHERE lifecycle_state='finalized'
                     AND session_number IN (
                         SELECT session_number FROM sessions
                         WHERE session_mode NOT IN (?, ?, ?)
                     )""",
                _per_slide_retrieval_modes,
            )
            rows = db.execute(
                """SELECT work_orders.session_number AS session_number,
                          work_orders.work_order_id AS work_order_id
                   FROM work_orders
                   JOIN sessions
                     ON sessions.session_number = work_orders.session_number
                   WHERE work_orders.lifecycle_state='scoring'
                      AND sessions.session_mode NOT IN (?, ?, ?)""",
                _per_slide_retrieval_modes,
            ).fetchall()
            open_rows = db.execute(
                """SELECT work_orders.session_number AS session_number,
                          work_orders.work_order_id AS work_order_id
                   FROM work_orders
                   JOIN sessions
                     ON sessions.session_number = work_orders.session_number
                   WHERE work_orders.lifecycle_state='finalized'
                     AND sessions.session_mode=?""",
                (SessionMode.OPEN_RETRIEVAL.value,),
            ).fetchall()
        for row in rows:
            session = self._session_identity(int(row["session_number"]))
            work_order_id = int(row["work_order_id"])
            job = self._executor.submit(self._score_work_order, session, work_order_id)
            with self._jobs_lock:
                self._jobs.append(job)
        for row in open_rows:
            self._submit_open_retrieval_finalization(
                self._session_identity(int(row["session_number"])),
                int(row["work_order_id"]),
            )

    def _score_work_order(self, session: SessionIdentity, work_order_id: int) -> None:
        """Batch N^2 score one finished work order; never re-raises (mirrors
        `_preprocess`'s try/except -- a scoring failure must not kill the
        single-worker executor for future submissions)."""
        try:
            with self._connect() as db:
                block_rows = db.execute(
                    "SELECT * FROM sets WHERE session_number=? AND work_order_id=?",
                    (session.number, work_order_id),
                ).fetchall()
                slide_rows = db.execute(
                    """SELECT * FROM slide_captures
                       WHERE session_number=? AND work_order_id=? AND success=1
                         AND block_id IS NOT NULL""",
                    (session.number, work_order_id),
                ).fetchall()

            block_results: dict[str, PreparedResult] = {}
            block_rows_by_id: dict[str, sqlite3.Row] = {}
            for row in block_rows:
                block_id = str(row["block_id"])
                block_rows_by_id[block_id] = row
                readiness = self._readiness_from_row(row)
                if readiness.evaluable:
                    block_results[block_id] = self._load_block_result(row)
                else:
                    block_results[block_id] = PreparationFailure(
                        role="block",
                        reason=readiness.review_reason or "block is not evaluable",
                    )

            slide_results: dict[str, PreparedResult] = {}
            slide_rows_by_capture: dict[str, sqlite3.Row] = {}
            for row in slide_rows:
                capture_id = str(row["capture_id"])
                slide_rows_by_capture[capture_id] = row
                slide_path = Path(str(row["capture_path"]))
                img: np.ndarray | None = None
                try:
                    with observed(
                        self.runtime_observer, "slide_preparation",
                        str(row["block_id"]),
                    ):
                        img = cv2.imread(str(slide_path))
                        if img is None:
                            slide_results[capture_id] = PreparationFailure(
                                role="slide",
                                reason=f"could not read image: {slide_path}",
                            )
                        else:
                            slide_results[capture_id] = self.slide_preprocessor(img)
                except Exception as exc:  # fail closed: never skip a verdict
                    slide_results[capture_id] = PreparationFailure(
                        role="slide", reason=str(exc)
                    )
                # Decoded frame is used only to build slide_results above; it
                # is never retained past this iteration (#185 -- the batch
                # path must stay O(1) memory, not O(N) frames held across the
                # N^2 scoring barrier below).
                del img

            scoring_result = _normalize_work_order_scoring_result(
                self.work_order_scorer(block_results, slide_results)
            )
            scores_by_slide = scoring_result.scores
            pair_decisions_by_slide = scoring_result.pair_decisions

            sheets_dir = (
                session.directory / "work_orders"
                / f"work_order_{work_order_id:06d}_sheets"
            )
            any_sheets_written = False

            csv_rows: list[tuple[str, str, WorkOrderVerdict, float | None]] = []
            for capture_id, slide_row in slide_rows_by_capture.items():
                claimed_block = str(slide_row["block_id"])
                candidate_scores = dict(scores_by_slide.get(capture_id, {}))
                verdict = evaluate_work_order(candidate_scores, claimed_block)
                slide_path = Path(str(slide_row["capture_path"]))
                try:
                    row = self.get_set(session.number, claimed_block)
                except KeyError:
                    row = None
                decision = ClaimDecision(
                    claim_id=claimed_block,
                    block_path=str(row["capture_path"]) if row else "",
                    slide_path=str(slide_path),
                    verdict=verdict.verdict,
                    stage="work_order_scoring",
                    reason=verdict.reason,
                    score=candidate_scores.get(claimed_block),
                )
                scored_pair = pair_decisions_by_slide.get(capture_id, {}).get(
                    claimed_block
                )
                if scored_pair is not None:
                    decision = replace(
                        decision,
                        selected_metric=scored_pair.selected_metric,
                        router_size_signal=scored_pair.router_size_signal,
                        block_occupied_fraction=scored_pair.block_occupied_fraction,
                        slide_occupied_fraction=scored_pair.slide_occupied_fraction,
                        best_angle=scored_pair.best_angle,
                        best_flip=scored_pair.best_flip,
                        align_soft_iou=scored_pair.align_soft_iou,
                        mask_iou=scored_pair.mask_iou,
                    )
                # Re-decode on demand, one frame at a time: prep (loop above)
                # and this finalize/QC step are separated by the N^2 scoring
                # barrier, so a decode-once frame can't span both without
                # holding all N captures' frames resident through scoring
                # (#185 -- no batch-memory buildup). _finalize_claim writes
                # claim_qc.png for EVERY verdict (not just REVIEW) and no
                # longer decodes for itself, so this decode is unconditional
                # -- matching origin/main, where _finalize_claim always
                # decoded the slide. REVIEW additionally reuses this same
                # frame for the flagged-pair sheets below instead of a
                # second decode.
                slide_img = cv2.imread(str(slide_path))
                self._finalize_claim(
                    session.number, claimed_block, capture_id, slide_path, row, decision,
                    block_result=block_results.get(claimed_block),
                    slide_result=slide_results.get(capture_id),
                    slide_img=slide_img,
                )
                with self._connect() as db:
                    db.execute(
                        """UPDATE slide_captures SET top_block=?, near_miss_blocks=?
                           WHERE capture_id=?""",
                        (
                            verdict.top_block,
                            ",".join(sorted(verdict.near_miss_blocks)),
                            capture_id,
                        ),
                    )
                # QA temporary: claimed-pair slide overlay for PASS and REVIEW.
                # Revisit after overlay QA — production intent was REVIEW-only.
                claimed_block_row = block_rows_by_id.get(claimed_block)
                claimed_block_path = (
                    claimed_block_row["capture_path"]
                    if claimed_block_row is not None
                    else None
                )
                claimed_block_img = (
                    cv2.imread(str(claimed_block_path)) if claimed_block_path else None
                )
                self._write_claim_slide_overlay(
                    session,
                    capture_id,
                    block_img=claimed_block_img,
                    slide_img=slide_img,
                    block_result=block_results.get(
                        claimed_block,
                        PreparationFailure(role="block", reason="not scanned"),
                    ),
                    slide_result=slide_results.get(
                        capture_id,
                        PreparationFailure(role="slide", reason="not attempted"),
                    ),
                    decision=decision,
                )
                del claimed_block_img
                if verdict.verdict == "REVIEW":
                    for pair in flagged_pairs(claimed_block, verdict):
                        pair_block_id = pair["block_id"]
                        block_row = block_rows_by_id.get(pair_block_id)
                        block_path = (
                            block_row["capture_path"] if block_row is not None else None
                        )
                        block_img = cv2.imread(str(block_path)) if block_path else None
                        sheet_path = sheets_dir / f"{capture_id}__{pair_block_id}.png"
                        self._contact_sheet_renderer(
                            block_img=block_img,
                            slide_img=slide_img,
                            block_result=block_results.get(
                                pair_block_id,
                                PreparationFailure(role="block", reason="not scanned"),
                            ),
                            slide_result=slide_results.get(
                                capture_id,
                                PreparationFailure(role="slide", reason="not attempted"),
                            ),
                            decision=decision,
                            output_path=sheet_path,
                            slide_id=capture_id,
                            role_label=f"{pair['role']} {pair_block_id}",
                        )
                        any_sheets_written = True
                        del block_img
                del slide_img
                score = candidate_scores.get(claimed_block)
                csv_rows.append((capture_id, claimed_block, verdict, score))

            csv_path = self._write_work_order_csv(session, work_order_id, csv_rows)
            contact_sheet_dir = str(sheets_dir) if any_sheets_written else None
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE work_orders SET lifecycle_state='results_ready',
                       finished_at=?, verdict_csv_path=?, contact_sheet_dir=?
                       WHERE session_number=? AND work_order_id=?""",
                    (
                        datetime.now(timezone.utc).isoformat(), str(csv_path),
                        contact_sheet_dir, session.number, work_order_id,
                    ),
                )
            self._emit(
                session.number, "work_order_results_ready", "Work order results ready"
            )
        except Exception as exc:  # failure is durable and must not kill the executor
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE work_orders SET lifecycle_state='scoring_failed',
                       failure_reason=? WHERE session_number=? AND work_order_id=?""",
                    (str(exc), session.number, work_order_id),
                )
            self._emit(session.number, "work_order_scoring_failed", str(exc))

    def _submit_hybrid_scoring(
        self,
        session: SessionIdentity,
        work_order_id: int,
        block_id: str,
        capture_id: str,
        slide_path: Path,
        *,
        profile: bool = False,
        profile_queued_ns: int | None = None,
    ) -> None:
        """#252: submit one accepted, in-pool Hybrid slide's disposable
        in-memory Future, mirroring `_submit_preprocessing`'s shape exactly so
        `wait_for_jobs()` drains this job too. Runs on the SAME single-worker
        `self._executor` -- no second executor, no second thread; ordinary
        jobs (block preprocessing, batch work-order scoring, and this) all
        share one sequential, FIFO queue, per the issue's explicit
        "parallel workers are out of scope" call-out.

        ``profile``/``profile_queued_ns`` (#258) are threaded straight through
        to `_score_hybrid_slide`, mirroring `_submit_preprocessing`'s own
        pass-through of the block equivalents.
        """
        job = self._executor.submit(
            self._score_hybrid_slide, session, work_order_id, block_id, capture_id,
            slide_path, profile=profile, profile_queued_ns=profile_queued_ns,
        )
        with self._jobs_lock:
            self._jobs.append(job)

    def _submit_retrieval_scoring(
        self,
        session: SessionIdentity,
        work_order_id: int,
        block_id: str,
        capture_id: str,
        slide_path: Path,
        *,
        profile: bool = False,
        profile_queued_ns: int | None = None,
    ) -> None:
        """Submit one durable retrieval slide job using its mode strategy."""
        if session.session_mode == SessionMode.OPEN_RETRIEVAL.value:
            worker = self._score_open_retrieval_slide
            kwargs: dict[str, object] = {}
        else:
            worker = self._score_hybrid_slide
            kwargs = {
                "profile": profile,
                "profile_queued_ns": profile_queued_ns,
            }
        job = self._executor.submit(
            worker, session, work_order_id, block_id, capture_id, slide_path, **kwargs
        )
        with self._jobs_lock:
            self._jobs.append(job)

    def _set_slide_job_state(self, capture_id: str, job_state: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE slide_captures SET job_state=? WHERE capture_id=?",
                (job_state, capture_id),
            )

    def _cas_slide_job_state(
        self, capture_id: str, *, expected: str, new_state: str,
    ) -> bool:
        """CAS entry-transition write for `_score_hybrid_slide`'s own
        `job_state` advances (the #255-completion/#255-error CAS writes'
        sibling for the two ENTRY transitions those never covered): the row
        only advances to ``new_state`` when it is still exactly
        ``expected`` -- the state this worker's own submission (a fresh
        enqueue, a restart-recovery requeue, or an operator retry) left it
        in. A row a concurrent recapture has already superseded (or
        otherwise reassigned) out from under this worker fails this CAS and
        must never be resurrected back into an active state -- the same
        "0 rows affected -> abandon quietly, never raise" shape the
        completion/error CAS writes below already established. Unlike
        `_set_slide_job_state` (still used, unconditionally, by tests and by
        code that means to force a state regardless of what is currently
        there), this is the only path that must never stomp a state it did
        not expect.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                "UPDATE slide_captures SET job_state=? "
                "WHERE capture_id=? AND job_state=?",
                (new_state, capture_id, expected),
            ).rowcount
        return updated == 1

    def _set_hybrid_profile_stage(self, capture_id: str, stage: str) -> None:
        """#258: durable "which of `PROFILE_STAGE_ORDER` is the worker
        currently inside" marker for a still-pending profiled row -- the one
        `list_hybrid_profile_rows`/`project_profile_rows` read as a pending
        row's `stage`. Callers gate every call on `profile` themselves (this
        method has no opinion on whether profiling is on), mirroring
        `_set_slide_job_state`'s own unconditional-write shape.
        """
        with self._connect() as db:
            db.execute(
                "UPDATE slide_captures SET profile_current_stage=? WHERE capture_id=?",
                (stage, capture_id),
            )

    def _persist_candidate_selection(
        self,
        capture_id: str,
        selection: CandidateSelection | None,
        selection_error: str | None,
    ) -> None:
        """#253: durable audit evidence for one slide's Heuristic Candidate
        Band selection, so a pruned block can never be confused with a block
        that was never scanned in this work order -- ``evaluate_work_order``
        treats a missing key as "not scanned", and without this record that
        is indistinguishable from "correctly pruned by the band" from the
        outside. Operator-hidden: `slide_captures.candidate_selection_json`
        is a new column nothing in `code/kiosk/results_table.py` or any CSV
        export reads.

        ``selection`` is ``None`` only when selection itself raised (caught
        by ``_score_hybrid_slide``); ``selection_error`` names that failure
        so the audit trail distinguishes "selection ran and fell back"
        (``selection.fallback_required``) from "selection code raised"
        (this branch) -- both are permanent, expected fallback shapes, never
        an ``ERROR`` verdict.
        """
        if selection is None:
            payload: dict[str, object] = {"selection_error": selection_error}
        else:
            payload = asdict(selection)
            validation_error = validate_selection(selection)
            if validation_error is not None:
                payload["validation_error"] = validation_error
        with self._connect() as db:
            db.execute(
                "UPDATE slide_captures SET candidate_selection_json=? "
                "WHERE capture_id=?",
                (json.dumps(payload), capture_id),
            )

    def _persist_shadow_comparison(
        self,
        capture_id: str,
        complete_scores: Mapping[str, float | None],
        hybrid_scoring_ids: Sequence[str],
        claim_id: str,
        complete_verdict: WorkOrderVerdict,
    ) -> None:
        """#254: durable Hybrid Shadow safety-comparison evidence.

        ``complete_scores``/``complete_verdict`` are the SAME map and verdict
        `_score_hybrid_slide` already produced from its one complete accurate
        pass over the whole frozen pool (the durable, displayed verdict).
        This method derives the PROPOSED Hybrid verdict by FILTERING that
        already-computed map down to ``hybrid_scoring_ids`` -- exactly the
        set real Hybrid would have scored (`select_hybrid_candidates`'s
        `accurate_scoring_ids`, or the whole pool on fallback) -- and calling
        the unchanged ``evaluate_work_order`` a SECOND time on that subset.
        That is a second pure verdict computation, never a second call to
        the accurate scorer: no key in ``complete_scores`` is ever recomputed
        here.

        Records both verdicts, both reasons, both match margins, whether the
        verdicts differ, the margin delta, and the exact candidate/pruned
        block-id sets used -- enough context to analyze candidate-pruning
        safety across real work orders without re-deriving anything from
        ``candidate_selection_json`` separately.
        """
        subset_scores = {
            block_id: complete_scores[block_id]
            for block_id in hybrid_scoring_ids
            if block_id in complete_scores
        }
        proposed_verdict = evaluate_work_order(subset_scores, claim_id)
        pruned_ids = tuple(
            sorted(set(complete_scores) - set(hybrid_scoring_ids))
        )
        margin_delta = (
            None
            if complete_verdict.match_margin is None
            or proposed_verdict.match_margin is None
            else complete_verdict.match_margin - proposed_verdict.match_margin
        )
        payload = {
            "claim_id": claim_id,
            "candidate_ids": tuple(sorted(hybrid_scoring_ids)),
            "pruned_ids": pruned_ids,
            "complete_verdict": complete_verdict.verdict,
            "complete_reason": complete_verdict.reason,
            "complete_match_margin": complete_verdict.match_margin,
            "complete_top_block": complete_verdict.top_block,
            "proposed_verdict": proposed_verdict.verdict,
            "proposed_reason": proposed_verdict.reason,
            "proposed_match_margin": proposed_verdict.match_margin,
            "proposed_top_block": proposed_verdict.top_block,
            "verdict_differs": complete_verdict.verdict != proposed_verdict.verdict,
            "match_margin_delta": margin_delta,
        }
        with self._connect() as db:
            db.execute(
                "UPDATE slide_captures SET shadow_comparison_json=? "
                "WHERE capture_id=?",
                (json.dumps(payload), capture_id),
            )

    def _score_open_retrieval_slide(
        self,
        session: SessionIdentity,
        work_order_id: int,
        block_id: str,
        capture_id: str,
        slide_path: Path,
    ) -> None:
        """Score one Open Retrieval slide against every block in its work order."""
        if not self._cas_slide_job_state(
            capture_id, expected="queued", new_state="preparing"
        ):
            return
        try:
            with self._connect() as db:
                block_rows = db.execute(
                    """SELECT * FROM sets
                       WHERE session_number=? AND work_order_id=?
                       ORDER BY rowid""",
                    (session.number, work_order_id),
                ).fetchall()
            block_results: dict[str, PreparedResult] = {}
            block_rows_by_id: dict[str, sqlite3.Row] = {}
            for block_row in block_rows:
                pool_block_id = str(block_row["block_id"])
                block_rows_by_id[pool_block_id] = block_row
                readiness = self._readiness_from_row(block_row)
                block_results[pool_block_id] = (
                    self._load_block_result(block_row)
                    if readiness.evaluable
                    else PreparationFailure(
                        role="block",
                        reason=readiness.review_reason or "block is not evaluable",
                    )
                )

            img, slide_result = self._prepare_slide_for_artifacts(
                slide_path, block_id, fail_closed=False
            )
            if not self._cas_slide_job_state(
                capture_id, expected="preparing", new_state="scoring"
            ):
                return

            scoring_result = _normalize_work_order_scoring_result(
                self.work_order_scorer(block_results, {capture_id: slide_result})
            )
            candidate_scores = dict(scoring_result.scores.get(capture_id, {}))
            score_results = dict(
                scoring_result.pair_decisions.get(capture_id, {})
            )
            verdict = evaluate_work_order(candidate_scores, block_id)

            try:
                claimed_row = self.get_set(session.number, block_id)
            except KeyError:
                claimed_row = None
            decision = ClaimDecision(
                claim_id=block_id,
                block_path=(
                    str(claimed_row["capture_path"]) if claimed_row is not None else ""
                ),
                slide_path=str(slide_path),
                verdict=verdict.verdict,
                stage="open_retrieval_scoring",
                reason=verdict.reason,
                score=candidate_scores.get(block_id),
            )
            claimed_score = score_results.get(block_id)
            if claimed_score is not None:
                decision = replace(
                    decision,
                    selected_metric=claimed_score.selected_metric,
                    router_size_signal=claimed_score.router_size_signal,
                    block_occupied_fraction=claimed_score.block_occupied_fraction,
                    slide_occupied_fraction=claimed_score.slide_occupied_fraction,
                    best_angle=claimed_score.best_angle,
                    best_flip=claimed_score.best_flip,
                    align_soft_iou=claimed_score.align_soft_iou,
                    mask_iou=claimed_score.mask_iou,
                )

            block_result = block_results.get(block_id)
            outcome = self._finalize_claim(
                session.number,
                block_id,
                capture_id,
                slide_path,
                claimed_row,
                decision,
                block_result=block_result,
                slide_result=slide_result,
                slide_img=img,
                expected_job_state="scoring",
            )
            if not outcome.accepted:
                return
            if score_results:
                self._persist_hybrid_score_audit(
                    session.number,
                    work_order_id,
                    capture_id,
                    block_id,
                    score_results,
                    verdict,
                )

            claimed_block_img = (
                cv2.imread(str(claimed_row["capture_path"]))
                if claimed_row is not None else None
            )
            self._write_claim_slide_overlay(
                session,
                capture_id,
                block_img=claimed_block_img,
                slide_img=img,
                block_result=block_result or PreparationFailure(
                    role="block", reason="claimed block not captured in this work order"
                ),
                slide_result=slide_result,
                decision=decision,
            )
            del claimed_block_img
            if verdict.verdict == "REVIEW":
                sheets_dir = (
                    session.directory / "work_orders"
                    / f"work_order_{work_order_id:06d}_sheets"
                )
                sheets_written = False
                for pair in flagged_pairs(block_id, verdict):
                    pair_block_id = pair["block_id"]
                    pair_row = block_rows_by_id.get(pair_block_id)
                    pair_block_img = (
                        cv2.imread(str(pair_row["capture_path"]))
                        if pair_row is not None else None
                    )
                    self._contact_sheet_renderer(
                        block_img=pair_block_img,
                        slide_img=img,
                        block_result=block_results.get(
                            pair_block_id,
                            PreparationFailure(role="block", reason="not scanned"),
                        ),
                        slide_result=slide_result,
                        decision=decision,
                        output_path=(
                            sheets_dir / f"{capture_id}__{pair_block_id}.png"
                        ),
                        slide_id=capture_id,
                        role_label=f"{pair['role']} {pair_block_id}",
                    )
                    sheets_written = True
                    del pair_block_img
                if sheets_written:
                    with self._connect() as db:
                        db.execute(
                            """UPDATE work_orders SET contact_sheet_dir=?
                               WHERE session_number=? AND work_order_id=?""",
                            (str(sheets_dir), session.number, work_order_id),
                        )

            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE slide_captures
                       SET top_block=?, near_miss_blocks=?, job_state='complete'
                       WHERE capture_id=? AND job_state='scoring'""",
                    (
                        verdict.top_block,
                        ",".join(sorted(verdict.near_miss_blocks)),
                        capture_id,
                    ),
                )
            self._finalize_open_retrieval_work_order(session, work_order_id)
            del img
        except Exception as exc:
            self._write_slide_artifacts(
                session,
                capture_id,
                slide_path,
                None,
                PreparationFailure(role="slide", reason=str(exc)),
            )
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """UPDATE slide_captures SET job_state='error'
                       WHERE capture_id=? AND job_state IN ('preparing', 'scoring')""",
                    (capture_id,),
                )
            self._finalize_open_retrieval_work_order(session, work_order_id)
            self._log_durable_exception(
                session.number,
                "open_retrieval_slide_scoring_failed",
                f"Open Retrieval scoring failed for capture {capture_id} "
                f"(work order {work_order_id}, claimed block {block_id})",
                exc,
            )

    def _score_hybrid_slide(
        self,
        session: SessionIdentity,
        work_order_id: int,
        block_id: str,
        capture_id: str,
        slide_path: Path,
        *,
        profile: bool = False,
        profile_queued_ns: int | None = None,
    ) -> None:
        """#252: the sequential background worker for one accepted, in-pool
        Hybrid slide claim.

        Slide preparation and quality checks move IN HERE -- they used to run
        synchronously on the acknowledgment path inside `resolve_claim`
        (`cv2.imread`, `self.slide_preprocessor`, `decide_claim`'s internal
        `run_quality_gates`), which is exactly the wait #252 removes from
        capture's critical path. Decode + prepare now shares
        `_prepare_slide_for_artifacts` with `resolve_claim` and
        `claim_out_of_pool_block` (passing `fail_closed=False`, see that
        method's docstring for why); the gate check and everything after it
        is this worker's own, and it never calls `resolve_claim`.

        #253: scores only the Heuristic Candidate Band plus the claim
        (`self.select_hybrid_candidates`'s `accurate_scoring_ids`), not the
        entire frozen pool -- via the unchanged `score_routed_caches`, one
        call per selected pool block against a single slide
        `LockedScoreCache` built through the existing injectable
        `self.score_cache_builder` seam. Falls back to scoring the ENTIRE
        pool (the #252 behavior, unconditionally) whenever selection raises,
        reports `fallback_required`, or fails its own `validate_selection`
        self-check -- this is a PERMANENT fallback path, not scaffolding to
        be removed later. The resulting `block_id -> score | None` map (only
        selected/scored blocks are present as keys; a pruned block is simply
        absent, which `evaluate_work_order` reads as "not scanned in this
        order" -- `self._persist_candidate_selection` records the selection
        itself so a pruned block is never confused with a never-scanned one)
        is handed to the unchanged `evaluate_work_order` (verdict authority;
        no verdict logic lives here), and the verdict is written through the
        existing `self._finalize_claim`, the same writer `resolve_claim` and
        `_score_work_order` both use. This is `SessionMode.HYBRID`'s
        behavior; `SessionMode.HYBRID_SHADOW` diverges only in WHICH ids get
        handed to `score_routed_caches` -- see below.

        #254 (Hybrid Shadow): `select_hybrid_candidates` still runs (so the
        selection audit and the "what would real Hybrid have scored" set are
        identical to Hybrid's), but `is_shadow` forces `scoring_ids` to the
        WHOLE pool (`pool.block_ids`) instead of
        `hybrid_scoring_ids`/`selection.accurate_scoring_ids`. That single
        substitution is what guarantees NO PAIR IS EVER SCORED TWICE: the
        `score_routed_caches` dict comprehension below still runs exactly
        once per id in whichever `scoring_ids` was chosen, so a shadow slide
        gets exactly `len(pool.block_ids)` calls and a plain Hybrid slide
        still gets exactly `len(hybrid_scoring_ids)` calls -- never both.
        `candidate_scores`/`verdict` (the COMPLETE map/verdict for shadow) is
        the same variable that reaches `_finalize_claim` for both modes, so
        the durable, displayed, operator-visible verdict is unconditionally
        `evaluate_work_order` applied to whatever was actually scored -- for
        shadow, that is genuinely the complete pool, never the subset.  The
        PROPOSED Hybrid verdict is derived afterward by
        `self._persist_shadow_comparison`, which FILTERS `candidate_scores`
        down to `hybrid_scoring_ids` and calls `evaluate_work_order` a
        SECOND time on that subset -- a second verdict computation (pure,
        no I/O, no scorer call), never a second score. A comparison-
        persistence failure is caught locally (logged via
        `self._log_durable_exception`, `shadow_comparison_json` stays NULL)
        so it can never turn the already-decided complete verdict into
        `job_state='error'` -- only a genuine scoring/IO failure earns ERROR.

        #258: when ``profile`` is on, five stages are timed via the
        injectable ``self._profile_clock_ns`` ONLY -- never a raw clock read
        -- and persisted to ``slide_captures.profile_stage_ms_json``/
        ``profile_total_ms`` in the SAME final CAS write that already
        commits ``top_block``/``near_miss_blocks``/``job_state='complete'``
        below: queue wait (enqueue to worker start), preparation (worker
        start through the quality-gate check), heuristic selection
        (`select_hybrid_candidates` alone -- explicitly stopped before
        `score_routed_caches` runs, so the selection number is never
        inflated by scoring time), accurate scoring (`score_routed_caches`
        alone), and artifact writing (everything from there through this
        method's own completion, `_persist_hybrid_score_audit`/
        `_finalize_claim`/the slide overlay). A gate-failed slide never
        reaches selection or scoring, so those two keys are simply absent
        from the persisted JSON -- never a fabricated zero (mirrors
        `code/session/profile_report.py:_stage_ms_breakdown`'s documented
        tolerance for a stage that never ran). Shadow's row is tagged via
        the `profile_shadow` column stamped at ENQUEUE time
        (`record_slide_capture`/`recapture_hybrid_slide`) from the session's
        own durable `session_mode`, so a shadow slide's complete-pool timing
        can never be read as pruned Hybrid timing in the PERSISTED data --
        honoring the warning this docstring used to carry.

        Never re-raises (mirrors `_preprocess`/`_score_work_order`'s
        try/except): a single job's failure must not kill this single-worker
        executor for the next queued slide. A failure here -- an unreadable
        image the slide_preprocessor itself cannot even attempt, a missing
        frozen pool, or an injected seam raising -- is a SYSTEM/ARTIFACT
        failure: durably recorded as `job_state='error'` with NO verdict
        written (`verdict` stays NULL; `code/kiosk/results_table.py`'s
        ERROR-vs-REVIEW split is explicit that ERROR means a system failure,
        never a match failure). `verdict='ERROR'` is never written --
        `session.pipeline` only ever produces the strings `"PASS"`/
        `"REVIEW"`. An EXPECTED outcome -- the slide fails quality gates,
        candidate selection falls back, or scores disagree with the claim --
        goes through `evaluate_work_order`/`_finalize_claim` normally and
        lands as REVIEW with `job_state='complete'`, exactly like any other
        REVIEW. Candidate selection failing is deliberately caught INSIDE
        this method's own try block (not left to the outer except) so a
        selection bug can never masquerade as this job's ERROR outcome.

        #255: restart recovery for an interrupted job is `_recover_retrieval_jobs`'s
        job, run at `ProcessingStore.__init__` -- this method's own
        `job_state` transitions (`preparing` -> `scoring` -> `complete`/
        `error`) are exactly what that recovery reads. Stale-write
        protection is also #255: the final `_finalize_claim` call and the
        completion/error writes below are all conditioned on `job_state`
        still being what this worker expects (see the comments at each of
        those call sites), so a job superseded mid-flight (#256) cannot
        have its late write clobber the active result.
        """
        is_shadow = session.session_mode == SessionMode.HYBRID_SHADOW.value
        # #258: taken at the very top of the worker, before anything else --
        # this is "queue wait ends, preparation begins" for BOTH the
        # persisted stage breakdown and the durable `profile_current_stage`
        # indicator a pending row's console/screen line reads.
        worker_started_ns = self._profile_clock_ns() if profile else None
        # CAS entry transition (sibling of the completion/error CAS writes
        # below, closing the gap those left open): every path that submits
        # this worker -- a fresh `record_slide_capture`/`recapture_hybrid_
        # slide` enqueue, `_recover_retrieval_jobs`'s restart requeue, and
        # `retry_hybrid_slide`'s own CAS -- leaves the row at exactly
        # 'queued' immediately beforehand, so that is the only prior state
        # this transition admits. A row a concurrent recapture has already
        # superseded (or otherwise reassigned) out from under this worker
        # fails this CAS and must NEVER be resurrected back into 'preparing'
        # -- abandon quietly (log and return), never raise, never 'error'
        # (being superseded is a correct, expected outcome, not a failure).
        if not self._cas_slide_job_state(
            capture_id, expected="queued", new_state="preparing"
        ):
            self._log_durable(
                session.number, "hybrid_stale_write_dropped",
                f"Hybrid slide scoring for capture {capture_id} (work order "
                f"{work_order_id}, claimed block {block_id}) could not enter "
                "'preparing': job_state was no longer 'queued' (superseded "
                "or otherwise reassigned) by the time this worker started; "
                "abandoning without touching job_state.",
            )
            return
        if profile:
            self._set_hybrid_profile_stage(capture_id, "preparation")
        try:
            pool = self.hybrid_pool(work_order_id)
            if pool is None:
                raise ValueError(
                    f"no frozen Hybrid Candidate Pool for work order {work_order_id}"
                )

            img, slide_result = self._prepare_slide_for_artifacts(
                slide_path, block_id, fail_closed=False
            )

            gate_failure_reason: str | None = None
            if isinstance(slide_result, PreparationFailure):
                gate_failure_reason = f"slide preparation failed: {slide_result.reason}"
            else:
                mask_gate = _check_mask_quality("slide", slide_result.mask)
                if mask_gate is not None:
                    gate_failure_reason = mask_gate.reason
            # #258: preparation ends here for BOTH outcomes -- decode/prepare
            # plus the quality-gate check itself, whether or not it passed.
            preparation_finished_ns = self._profile_clock_ns() if profile else None

            # CAS entry transition, same shape and reasoning as the
            # 'queued' -> 'preparing' CAS above: the only legal prior state
            # here is 'preparing', the state this SAME worker's own CAS just
            # won. A concurrent recapture superseding this row during the
            # (potentially long) decode/gate-check window just finished
            # fails this CAS and must NEVER resurrect the row back into
            # 'scoring' -- this is precisely the corruption this fix closes:
            # without it, a superseded job that reaches this line can win
            # `sets.verdict` out from under the job that actually superseded
            # it. Abandon quietly on a loss: log and return, never raise,
            # never 'error'.
            if not self._cas_slide_job_state(
                capture_id, expected="preparing", new_state="scoring"
            ):
                self._log_durable(
                    session.number, "hybrid_stale_write_dropped",
                    f"Hybrid slide scoring for capture {capture_id} (work "
                    f"order {work_order_id}, claimed block {block_id}) "
                    "could not enter 'scoring': job_state was no longer "
                    "'preparing' (superseded or otherwise reassigned) by "
                    "the time slide preparation finished; abandoning "
                    "without touching job_state.",
                )
                return
            candidate_scores: dict[str, float | None]
            score_results: dict[str, ProductionScoreResult] = {}
            # Populated only in the no-gate-failure branch below; stays None
            # for a gate-failed slide (nothing was scored, so there is no
            # subset to derive a #254 shadow comparison from -- mirrors
            # `candidate_selection_json` staying NULL on gate failure too).
            hybrid_scoring_ids: tuple[str, ...] | None = None
            # #258: only populated in the no-gate-failure branch below -- a
            # gate-failed slide never reaches selection or scoring, so both
            # stay None and both stage keys are simply absent from the
            # persisted breakdown (never a fabricated zero).
            selection_finished_ns: int | None = None
            scoring_finished_ns: int | None = None
            if gate_failure_reason is not None:
                # Gate-failed: never scored against any pool block, exactly
                # like `resolve_claim`'s own gate-before-score boundary.
                # `evaluate_work_order` turns an all-None map into its own
                # "claimed pair gate-failed -- fail-closed" REVIEW.
                candidate_scores = {pool_block_id: None for pool_block_id in pool.block_ids}
                if profile:
                    self._set_hybrid_profile_stage(capture_id, "artifact_write")
            else:
                if profile:
                    self._set_hybrid_profile_stage(capture_id, "heuristic_selection")
                assert isinstance(slide_result, PreparedSpecimen)
                slide_cache = self.score_cache_builder(slide_result)

                # #253: rank the frozen pool cheaply first -- BOTH modes need
                # this: Hybrid uses it to choose what to score; Hybrid Shadow
                # (#254) uses it only to know what real Hybrid WOULD have
                # scored (`hybrid_scoring_ids`), never to choose what it
                # itself scores. `accurate_scoring_ids` structurally always
                # includes the claim, regardless of the claim's heuristic
                # rank. Selection failing (raising, or failing its own
                # `validate_selection` self-check) is a PERMANENT fallback to
                # complete pool scoring, never this job's `job_state='error'`
                # outcome: only a genuine scoring/IO failure earns ERROR (see
                # this method's docstring on that split).
                selection: CandidateSelection | None
                selection_error: str | None = None
                try:
                    selection = self.select_hybrid_candidates(
                        pool, block_id, slide_cache,
                    )
                except Exception as exc:  # selection failure != scoring failure
                    selection = None
                    selection_error = str(exc)
                    self._log_durable_exception(
                        session.number, "hybrid_candidate_selection_failed",
                        "Heuristic Candidate Band selection failed for "
                        f"capture {capture_id} (work order {work_order_id}); "
                        "falling back to complete Hybrid pool scoring", exc,
                    )

                if (
                    selection is None
                    or selection.fallback_required
                    or validate_selection(selection) is not None
                ):
                    hybrid_scoring_ids = pool.block_ids
                else:
                    hybrid_scoring_ids = selection.accurate_scoring_ids
                # #258: heuristic selection ends here, BEFORE `score_routed_
                # caches` runs below -- this is what keeps the selection
                # timing honest; it must never include scoring time.
                selection_finished_ns = self._profile_clock_ns() if profile else None
                if profile:
                    self._set_hybrid_profile_stage(capture_id, "accurate_scoring")

                # #254: Hybrid Shadow always scores the WHOLE frozen pool --
                # one complete accurate pass -- instead of the Heuristic
                # Candidate Band subset. This is the ONLY behavioral fork
                # between the two modes; everything else (selection,
                # gate-checking, audit persistence, `_finalize_claim`) is
                # identical code for both. Plain Hybrid's `scoring_ids` is
                # unchanged from #253: exactly `hybrid_scoring_ids`.
                scoring_ids: tuple[str, ...] = (
                    pool.block_ids if is_shadow else hybrid_scoring_ids
                )

                score_results = {
                    pool_block_id: score_routed_caches(
                        pool.score_caches[pool_block_id], slide_cache,
                    )
                    for pool_block_id in scoring_ids
                }
                # #258: accurate scoring ends here -- shadow's row genuinely
                # paid the complete O(pool) cost through this same line
                # (`scoring_ids` was the whole pool for shadow, above), so
                # its timing is real, just never comparable to pruned
                # Hybrid's -- that is exactly what `profile_shadow` tags.
                scoring_finished_ns = self._profile_clock_ns() if profile else None
                candidate_scores = {
                    pool_block_id: result.score
                    for pool_block_id, result in score_results.items()
                }
                self._persist_candidate_selection(capture_id, selection, selection_error)
                if profile:
                    self._set_hybrid_profile_stage(capture_id, "artifact_write")

            verdict = evaluate_work_order(candidate_scores, block_id)

            # #254: derive (never re-score) the proposed Hybrid verdict by
            # filtering the just-computed COMPLETE map down to what real
            # Hybrid would have scored, and record the safety comparison.
            # `hybrid_scoring_ids is None` only on a gate-failed slide (never
            # scored against anything either way -- no comparison to make).
            # A comparison-persistence failure is caught HERE, not left to
            # the outer except: `verdict` above is already the authoritative,
            # fully-scored complete-pool result, and it must still reach
            # `_finalize_claim` below even if recording the comparison fails.
            if is_shadow and hybrid_scoring_ids is not None:
                try:
                    self._persist_shadow_comparison(
                        capture_id, candidate_scores, hybrid_scoring_ids, block_id, verdict,
                    )
                except Exception as exc:
                    self._log_durable_exception(
                        session.number, "hybrid_shadow_comparison_failed",
                        "Hybrid Shadow comparison persistence failed for "
                        f"capture {capture_id} (work order {work_order_id}, "
                        f"claimed block {block_id}); the complete verdict "
                        "above is unaffected and still lands normally", exc,
                    )

            # Persist only pairs the accurate scorer actually evaluated.  A
            # pruned candidate has no score by design, and a gate-failed slide
            # never reached the scorer, so neither receives a made-up row.
            self._persist_hybrid_score_audit(
                session.number,
                work_order_id,
                capture_id,
                block_id,
                score_results,
                verdict,
            )

            try:
                row = self.get_set(session.number, block_id)
            except KeyError:
                row = None
            block_result = self._load_block_result(row) if row is not None else None
            decision = ClaimDecision(
                claim_id=block_id,
                block_path=str(row["capture_path"]) if row is not None else "",
                slide_path=str(slide_path),
                verdict=verdict.verdict,
                stage="hybrid_scoring",
                reason=verdict.reason,
                score=candidate_scores.get(block_id),
            )
            # The claimed pair was one of this worker's accurate scores, so
            # reuse its locked pose for the operator overlay.  Do not run a
            # second alignment search, and leave gate-failed/pruned claims
            # pose-less rather than inventing an overlay.
            claimed_score = score_results.get(block_id)
            if claimed_score is not None:
                decision = replace(
                    decision,
                    selected_metric=claimed_score.selected_metric,
                    router_size_signal=claimed_score.router_size_signal,
                    block_occupied_fraction=claimed_score.block_occupied_fraction,
                    slide_occupied_fraction=claimed_score.slide_occupied_fraction,
                    best_angle=claimed_score.best_angle,
                    best_flip=claimed_score.best_flip,
                    align_soft_iou=claimed_score.align_soft_iou,
                    mask_iou=claimed_score.mask_iou,
                )
            # `_finalize_claim`'s own docstring note on `expected_job_state`
            # for the concrete race this closes. A late completion from a
            # job that lost this race (e.g. superseded by #256 mid-flight)
            # is a durable, non-fatal outcome, never an exception: log it
            # and return without touching top_block/near_miss_blocks/
            # job_state below, so it cannot clobber whatever the active job
            # already wrote.
            claim_outcome = self._finalize_claim(
                session.number, block_id, capture_id, slide_path, row, decision,
                block_result=block_result, slide_result=slide_result, slide_img=img,
                expected_job_state="scoring",
            )
            if not claim_outcome.accepted:
                self._log_durable(
                    session.number, "hybrid_stale_write_dropped",
                    f"Hybrid slide scoring for capture {capture_id} (work "
                    f"order {work_order_id}, claimed block {block_id}) "
                    "completed after this job was no longer the active one "
                    "for its slide; the durable verdict write was skipped "
                    "so it could not overwrite the active result.",
                )
                return
            # Hybrid uses the same durable claim result projection as the
            # synchronous paths. Write its raw block/slide JPEG evidence too;
            # the helper only writes an overlay when this worker has a locked
            # pose, so Hybrid's evaluator verdict does not invent one.
            block_img = (
                cv2.imread(str(row["capture_path"]))
                if row is not None else None
            )
            self._write_claim_slide_overlay(
                session,
                capture_id,
                block_img=block_img,
                slide_img=img,
                block_result=block_result if block_result is not None else PreparationFailure(
                    role="block", reason="not scanned"
                ),
                slide_result=slide_result,
                decision=decision,
            )
            del block_img
            # #258: taken right before the final CAS write, so "artifact
            # write" covers everything from the end of scoring (or, on a
            # gate-failed slide, the end of preparation) through the score
            # audit, `_finalize_claim`, and the slide overlay above -- the
            # worker's own last stage before it marks itself complete.
            completed_ns = self._profile_clock_ns() if profile else None
            stage_ms_json: str | None = None
            total_ms: int | None = None
            if profile:
                stage_ms: dict[str, int] = {}
                queue_wait_ms = _elapsed_ms(worker_started_ns, profile_queued_ns)
                if queue_wait_ms is not None:
                    stage_ms["queue_wait"] = queue_wait_ms
                preparation_ms = _elapsed_ms(preparation_finished_ns, worker_started_ns)
                if preparation_ms is not None:
                    stage_ms["preparation"] = preparation_ms
                if selection_finished_ns is not None:
                    selection_ms = _elapsed_ms(
                        selection_finished_ns, preparation_finished_ns
                    )
                    if selection_ms is not None:
                        stage_ms["heuristic_selection"] = selection_ms
                if scoring_finished_ns is not None:
                    scoring_ms = _elapsed_ms(scoring_finished_ns, selection_finished_ns)
                    if scoring_ms is not None:
                        stage_ms["accurate_scoring"] = scoring_ms
                artifact_write_ms = _elapsed_ms(
                    completed_ns,
                    scoring_finished_ns
                    if scoring_finished_ns is not None else preparation_finished_ns,
                )
                if artifact_write_ms is not None:
                    stage_ms["artifact_write"] = artifact_write_ms
                stage_ms_json = json.dumps(stage_ms)
                total_ms = _elapsed_ms(completed_ns, profile_queued_ns)
            # Same CAS as `_finalize_claim` above, combined into one atomic
            # UPDATE: the job only transitions to 'complete' -- and only
            # gets its top_block/near_miss_blocks written -- if `job_state`
            # is still 'scoring' at the moment of this exact statement. A
            # supersession landing in the narrow window between the
            # `_finalize_claim` call above and this one is still caught: the
            # row simply keeps whichever state (e.g. 'superseded') a
            # concurrent writer already set, rather than being stomped back
            # to 'complete'.
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                completed = db.execute(
                    """UPDATE slide_captures SET top_block=?, near_miss_blocks=?,
                       job_state='complete', profile_stage_ms_json=?,
                       profile_total_ms=?
                       WHERE capture_id=? AND job_state='scoring'""",
                    (
                        verdict.top_block,
                        ",".join(sorted(verdict.near_miss_blocks)),
                        stage_ms_json,
                        total_ms,
                        capture_id,
                    ),
                ).rowcount
            if completed != 1:
                self._log_durable(
                    session.number, "hybrid_stale_write_dropped",
                    f"Hybrid slide scoring for capture {capture_id} (work "
                    f"order {work_order_id}, claimed block {block_id}) "
                    "verdict was already committed, but this job was no "
                    "longer the active one by the time it tried to mark "
                    "itself complete; job_state was left untouched.",
                )
        except Exception as exc:  # failure is durable and must not kill the worker
            # A worker-level failure can happen before `_finalize_claim`, which
            # normally persists slide preparation evidence. Preserve a failure
            # QC panel here as well, so every captured Hybrid slide remains
            # inspectable even when preprocessing or scoring aborts early.
            self._write_slide_artifacts(
                session,
                capture_id,
                slide_path,
                None,
                PreparationFailure(role="slide", reason=str(exc)),
            )
            # Same CAS as the completion path above: only a job that is
            # still genuinely mid-flight ('preparing'/'scoring') earns
            # 'error' here. A row already superseded (or, in principle,
            # already completed by a concurrent writer) must not be stomped
            # back to 'error' by a stale failure.
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                errored = db.execute(
                    """UPDATE slide_captures SET job_state='error'
                       WHERE capture_id=? AND job_state IN ('preparing', 'scoring')""",
                    (capture_id,),
                ).rowcount
            if errored != 1:
                self._log_durable(
                    session.number, "hybrid_stale_write_dropped",
                    f"Hybrid slide scoring for capture {capture_id} (work "
                    f"order {work_order_id}, claimed block {block_id}) "
                    "failed after this job was no longer the active one "
                    "for its slide; job_state was left untouched instead "
                    "of being marked 'error'.",
                )
            self._log_durable_exception(
                session.number, "hybrid_slide_scoring_failed",
                f"Hybrid slide scoring failed for capture {capture_id} "
                f"(work order {work_order_id}, claimed block {block_id})",
                exc,
            )

    def _write_work_order_csv(
        self,
        session: SessionIdentity,
        work_order_id: int,
        rows: list[tuple[str, str, WorkOrderVerdict, float | None]],
    ) -> Path:
        directory = session.directory / "work_orders"
        directory.mkdir(exist_ok=True)
        path = directory / f"work_order_{work_order_id:06d}_verdicts.csv"
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "capture_id", "claimed_block", "verdict", "reason", "score",
            "match_margin", "top_block", "near_miss_blocks",
        ])
        for capture_id, claimed_block, verdict, score in rows:
            writer.writerow([
                capture_id, claimed_block, verdict.verdict, verdict.reason,
                "" if score is None else f"{score:.4f}",
                "" if verdict.match_margin is None else f"{verdict.match_margin:.4f}",
                verdict.top_block or "",
                ",".join(sorted(verdict.near_miss_blocks)),
            ])
        _atomic_bytes(path, buffer.getvalue().encode("utf-8"))
        return path

    def slide_recovery_state(self, session_number: int) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT slide_recovery_state FROM sessions WHERE session_number=?",
                (session_number,),
            ).fetchone()
        if row is None:
            raise KeyError(session_number)
        return str(row["slide_recovery_state"])

    def skip_unreadable_slide(
        self, session_number: int, *, request_id: str | None = None
    ) -> None:
        fingerprint = self._fingerprint({}) if request_id is not None else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, _cached = self._ledger_hit(
                    db, request_id, "skip_unreadable_slide", session_number, fingerprint
                )
                if hit:
                    return None
            updated = db.execute(
                "UPDATE sessions SET slide_recovery_state='waiting_for_removal' "
                "WHERE session_number=? AND slide_recovery_state='reposition'",
                (session_number,),
            ).rowcount
            if updated != 1:
                # This is a deterministic command-state rejection, not an
                # infrastructure/runtime failure.  The RPC boundary maps
                # ValueError to HTTP 400 so remote callers do not retry it.
                # Not ledgered: the transaction rolls back, so a genuine
                # retry with the same request_id must see the same rejection.
                raise ValueError("Skip is only valid for an unreadable slide")
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "skip_unreadable_slide", session_number,
                    fingerprint, json.dumps(None),
                )
        self._emit(
            session_number,
            "unreadable_slide_skipped",
            "Unidentified slide skipped",
        )

    def mark_waiting_for_slide(
        self, session_number: int, *, request_id: str | None = None
    ) -> None:
        fingerprint = self._fingerprint({}) if request_id is not None else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, _cached = self._ledger_hit(
                    db, request_id, "mark_waiting_for_slide", session_number, fingerprint
                )
                if hit:
                    # Replay must not clobber a slide_recovery_state that has
                    # since moved on (e.g. a delayed retry arriving after the
                    # slide was actually captured) -- this is exactly what the
                    # ledger short-circuit protects against.
                    return None
            db.execute(
                "UPDATE sessions SET slide_recovery_state='waiting' "
                "WHERE session_number=?",
                (session_number,),
            )
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "mark_waiting_for_slide", session_number,
                    fingerprint, json.dumps(None),
                )
        self._emit(session_number, "waiting_for_slide", "Waiting for slide")

    def active_warnings(self, session_number: int) -> tuple[FailedBlockWarning, ...]:
        """Return presentation-independent recovery actions for failed blocks."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT block_id, failure_reason, qc_path FROM sets
                   WHERE session_number=? AND preprocessing_status='failed'
                   ORDER BY rowid""",
                (session_number,),
            ).fetchall()
        return tuple(
            FailedBlockWarning(
                str(row["block_id"]), str(row["failure_reason"]), Path(row["qc_path"])
            )
            for row in rows
        )

    def summarize(self, session: SessionIdentity) -> SessionSummary:
        """Compact processed/PASS/REVIEW counts plus expandable detail."""
        with self._connect() as db:
            counts = db.execute(
                """SELECT
                   COUNT(*) AS processed,
                   SUM(CASE WHEN verdict='PASS' THEN 1 ELSE 0 END) AS passed,
                   SUM(CASE WHEN verdict='REVIEW' THEN 1 ELSE 0 END) AS reviewed
                   FROM sets WHERE session_number=? AND verdict IS NOT NULL""",
                (session.number,),
            ).fetchall()
            finalization_error = db.execute(
                "SELECT last_finalization_error FROM sessions WHERE session_number=?",
                (session.number,),
            ).fetchone()["last_finalization_error"]
            missing = db.execute(
                """SELECT block_id FROM sets WHERE session_number=?
                   AND preprocessing_status IN ('complete', 'unusable')
                   AND verdict IS NULL ORDER BY rowid""",
                (session.number,),
            ).fetchall()
            pending = db.execute(
                """SELECT block_id FROM sets WHERE session_number=?
                   AND preprocessing_status IN ('queued', 'processing')
                   ORDER BY rowid""",
                (session.number,),
            ).fetchall()
            skipped = db.execute(
                """SELECT capture_id FROM slide_captures
                   WHERE session_number=? AND success=0 ORDER BY captured_at""",
                (session.number,),
            ).fetchall()
            # #188: a dismissed block (operator resolved a preprocessing
            # failure as unusable) is not real work left to resume -- exclude
            # it so an all-dismissed session doesn't misread as "unfinished."
            blocks_captured = db.execute(
                """SELECT COUNT(*) AS total FROM sets
                   WHERE session_number=? AND dismissed_at IS NULL""",
                (session.number,),
            ).fetchone()["total"]
        processed = sum(row["processed"] or 0 for row in counts)
        passed = sum(row["passed"] or 0 for row in counts)
        reviewed = sum(row["reviewed"] or 0 for row in counts)
        return SessionSummary(
            session.number, session.started_at, processed, passed, reviewed,
            missing_slides=tuple(row["block_id"] for row in missing),
            block_failures=self.active_warnings(session.number),
            skipped_decodes=tuple(row["capture_id"] for row in skipped),
            pending_blocks=tuple(row["block_id"] for row in pending),
            finalization_error=finalization_error,
            blocks_captured=int(blocks_captured or 0),
        )

    def dismiss_block(
        self, session_number: int, block_id: str, *, reason: str,
        request_id: str | None = None,
    ) -> None:
        if not reason.strip():
            raise ValueError("dismissal reason is required")
        reason = reason.strip()
        fingerprint = (
            self._fingerprint({"block_id": block_id, "reason": reason})
            if request_id is not None else None
        )
        dismissed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, _cached = self._ledger_hit(
                    db, request_id, "dismiss_block", session_number, fingerprint
                )
                if hit:
                    return None
            updated = db.execute(
                """UPDATE sets SET preprocessing_status='unusable', dismissed_at=?,
                   unusable_reason=?, mask_path=NULL
                   WHERE session_number=? AND block_id=?
                   AND preprocessing_status='failed'""",
                (dismissed_at, reason, session_number, block_id),
            ).rowcount
            if updated != 1:
                raise ValueError("only a failed block can be dismissed")
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "dismiss_block", session_number,
                    fingerprint, json.dumps(None),
                )
        self._emit(session_number, "block_dismissed", reason, block_id)

    def block_readiness(self, session_number: int, block_id: str) -> BlockReadiness:
        return self._readiness_from_row(self.get_set(session_number, block_id))

    @staticmethod
    def _readiness_from_row(row: Mapping[str, object]) -> BlockReadiness:
        status = str(row["preprocessing_status"])
        if status == "complete" and row["mask_path"]:
            return BlockReadiness(True)
        reason = row["unusable_reason"] or row["failure_reason"] or (
            f"block preprocessing is {status}"
        )
        return BlockReadiness(False, str(reason))

    def snapshot(self, session: SessionIdentity) -> WorkflowSnapshot:
        with self._connect() as db:
            phase = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session.number,)
            ).fetchone()["phase"]
            pending = db.execute(
                """SELECT COUNT(*) AS count FROM sets WHERE session_number=?
                   AND preprocessing_status IN ('queued', 'processing')""",
                (session.number,),
            ).fetchone()["count"]
            unresolved = db.execute(
                """SELECT COUNT(*) AS count FROM sets WHERE session_number=?
                   AND preprocessing_status NOT IN ('complete', 'unusable')""",
                (session.number,),
            ).fetchone()["count"]
            latest = db.execute(
                """SELECT block_id, preprocessing_status FROM sets
                   WHERE session_number=? AND capture_id IS NOT NULL
                   ORDER BY rowid DESC LIMIT 1""",
                (session.number,),
            ).fetchone()
        return WorkflowSnapshot(
            session.number, session.started_at, phase, "idle" if pending == 0 else "active",
            int(pending), latest["block_id"] if latest else None,
            latest["preprocessing_status"] if latest else None,
            unresolved_blocks=int(unresolved),
        )

    def events(self, session_number: int) -> tuple[WorkflowEvent, ...]:
        with self._events_lock:
            return tuple(self._events.get(session_number, ()))

    def record_event(
        self,
        session_number: int,
        kind: str,
        message: str,
        *,
        request_id: str | None = None,
    ) -> None:
        if request_id is not None:
            with self._events_lock:
                if request_id in self._seen_event_request_ids:
                    return
                self._seen_event_request_ids.add(request_id)
        self._emit(session_number, kind, message)

    def _emit(
        self, session_number: int, kind: str, message: str,
        block_id: str | None = None, capture_id: str | None = None,
    ) -> None:
        with self._connect() as db:
            row = db.execute(
                "SELECT phase FROM sessions WHERE session_number=?", (session_number,)
            ).fetchone()
        phase = str(row["phase"]) if row else "unknown"
        event = WorkflowEvent(kind, session_number, phase, message, block_id, capture_id)
        with self._events_lock:
            self._events.setdefault(session_number, []).append(event)

    def record_profile_capture(
        self,
        session_number: int,
        capture_id: str,
        fields: Mapping[str, object],
        *,
        request_id: str | None = None,
    ) -> None:
        """Append one row to `<session_dir>/profile_summary.csv` (#168).

        Opt-in tracer slice: called only when `PiCaptureRuntime.profile` is
        true, so profile-off sessions never create this file. One row per
        capture, one column per `CAPTURE_STAGE_TIMING_KEYS` stage, joined by
        `capture_id`.
        """
        fingerprint = (
            self._fingerprint({"capture_id": capture_id, "fields": dict(fields)})
            if request_id is not None else None
        )
        identity = self._session_identity(session_number)
        row = format_profile_summary_row(capture_id, fields)
        path = identity.directory / "profile_summary.csv"
        fieldnames = ["capture_id", *CAPTURE_STAGE_TIMING_KEYS, *SETTLING_STAGE_KEYS]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                hit, _cached = self._ledger_hit(
                    db, request_id, "record_profile_capture", session_number, fingerprint
                )
                if hit:
                    return None
            existing_rows: list[dict] = []
            if path.exists():
                with path.open("r", encoding="utf-8", newline="") as stream:
                    existing_rows = list(csv.DictReader(stream))
            existing_rows.append(row)
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
            # NOTE: this CSV write is a filesystem side effect that is NOT
            # transactional with SQLite; a crash between the atomic file
            # write and the ledger COMMIT could roll back the ledger row and
            # let a retry duplicate the row. Accepted for opt-in diagnostic
            # profiling data (#168) -- not worth a two-phase-commit dance.
            _atomic_bytes(path, buffer.getvalue().encode("utf-8"))
            if request_id is not None:
                self._ledger_record(
                    db, request_id, "record_profile_capture", session_number,
                    fingerprint, json.dumps(None),
                )

    def record_slide_benchmark(
        self, session_number: int, capture_id: str, fields: Mapping[str, object]
    ) -> None:
        """Upsert the opt-in, normal-mode per-slide timing report.

        The Pi supplies its local timing spans after the receiver has returned;
        PC stages may be supplied by the receiver before the same row is
        rewritten. Blank columns deliberately mean that a partial/failed
        capture never reached that phase.
        """
        identity = self._session_identity(session_number)
        path = identity.directory / "slide_benchmark.csv"
        fieldnames = ["capture_id", *SLIDE_BENCHMARK_COLUMNS]
        rows: dict[str, dict[str, object]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = {str(row["capture_id"]): row for row in csv.DictReader(stream)}
        row = rows.setdefault(capture_id, {"capture_id": capture_id})
        with self._slide_profile_lock:
            row.update(self._slide_profile_stages.pop(capture_id, {}))
        row.update({key: value for key, value in fields.items() if key in fieldnames})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
        _atomic_bytes(path, buffer.getvalue().encode("utf-8"))

    def _record_block_benchmark(
        self,
        session: SessionIdentity,
        capture_id: str,
        *,
        queue_wait_ms: int | None,
        block_preparation_ms: int | None,
        segmentation_ms: object | None,
        artifact_write_ms: int | None,
        ready_after_receive_ms: int | None,
        status: str,
    ) -> None:
        """Append one profiled block's PC-local readiness evidence.

        This file intentionally has no Pi capture or transfer duration: those
        clocks belong to a different machine and are already represented by
        ``profile_summary.csv``. Its ``ready_after_receive_ms`` answers when
        the receiver's queued block became usable for later slide scoring.
        """
        path = session.directory / "block_benchmark.csv"
        row = {
            "capture_id": capture_id,
            "queue_wait_ms": queue_wait_ms,
            "block_preparation_ms": block_preparation_ms,
            "segmentation_ms": segmentation_ms,
            "artifact_write_ms": artifact_write_ms,
            "ready_after_receive_ms": ready_after_receive_ms,
            "status": status,
        }
        with self._block_profile_lock:
            rows: dict[str, dict[str, object]] = {}
            if path.exists():
                with path.open("r", encoding="utf-8", newline="") as stream:
                    rows = {str(item["capture_id"]): item for item in csv.DictReader(stream)}
            rows[capture_id] = row
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer, fieldnames=["capture_id", *BLOCK_BENCHMARK_COLUMNS]
            )
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])
            _atomic_bytes(path, buffer.getvalue().encode("utf-8"))

    def _record_slide_profile_stage(self, capture_id: str, stage: str, started_ns: int) -> None:
        with self._slide_profile_lock:
            self._slide_profile_stages.setdefault(capture_id, {})[stage] = int(
                round((perf_counter_ns() - started_ns) / 1_000_000)
            )

    def latest_session_number(self) -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT session_number FROM sessions ORDER BY session_number DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError("no session is available")
        return int(row["session_number"])

    def _persist_hybrid_score_audit(
        self,
        session_number: int,
        work_order_id: int,
        slide_capture_id: str,
        claimed_block_id: str,
        score_results: Mapping[str, ProductionScoreResult],
        verdict: WorkOrderVerdict,
    ) -> None:
        """Persist one Hybrid slide's real scoring evidence and rank summary.

        ``score_results`` deliberately contains only calls that reached
        ``score_routed_caches``.  That makes the matching-corpus rows an
        honest audit record: a pruned candidate or a quality-gated slide is
        absent instead of being represented by a fabricated zero/NULL score.
        """
        ranked = sorted(
            score_results.items(), key=lambda item: (-item[1].score, item[0])
        )
        top_score = ranked[0][1].score if ranked else None
        runner_up_score = ranked[1][1].score if len(ranked) > 1 else None
        scored_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as db:
            capture_row = db.execute(
                "SELECT work_order FROM slide_captures WHERE capture_id=?",
                (slide_capture_id,),
            ).fetchone()
            if capture_row is None:
                raise KeyError(f"unknown slide capture: {slide_capture_id}")
            # Scan payloads normally carry an operator work-order label.  The
            # durable bracket id is the fallback for a compatible/manual
            # capture that has no label, so this NOT NULL corpus field is
            # still traceable to a single bracket.
            work_order = str(capture_row["work_order"] or work_order_id)

            for rank, (block_id, result) in enumerate(ranked, start=1):
                pair_id = make_pair_id(session_number, block_id, slide_capture_id)
                pair_source = "true_pair" if block_id == claimed_block_id else "candidate"
                db.execute(
                    """
                    INSERT INTO matching_pairs(
                        pair_id, session_number, work_order, block_id, slide_capture_id,
                        pair_source, is_match, classical_score, rank_for_block,
                        metric, scored_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pair_id) DO UPDATE SET
                        work_order=excluded.work_order,
                        pair_source=CASE
                            WHEN matching_pairs.pair_source = 'near_miss'
                                 AND excluded.pair_source = 'candidate'
                            THEN matching_pairs.pair_source
                            WHEN matching_pairs.pair_source = 'true_pair'
                            THEN matching_pairs.pair_source
                            ELSE excluded.pair_source
                        END,
                        is_match=COALESCE(excluded.is_match, matching_pairs.is_match),
                        classical_score=excluded.classical_score,
                        rank_for_block=excluded.rank_for_block,
                        metric=excluded.metric,
                        scored_at=excluded.scored_at
                    """,
                    (
                        pair_id, session_number, work_order, block_id, slide_capture_id,
                        pair_source, int(block_id == claimed_block_id), result.score,
                        rank, result.selected_metric, scored_at,
                    ),
                )

            db.execute(
                """UPDATE slide_captures
                      SET top_block=?, near_miss_blocks=?, top_score=?,
                          runner_up_score=?, match_margin=?
                    WHERE capture_id=?""",
                (
                    verdict.top_block,
                    ",".join(sorted(verdict.near_miss_blocks)),
                    top_score,
                    runner_up_score,
                    verdict.match_margin,
                    slide_capture_id,
                ),
            )

    def upsert_matching_pair(
        self,
        *,
        session_number: int,
        work_order: str,
        block_id: str,
        slide_capture_id: str,
        pair_source: str,
        is_match: int | None,
    ) -> str:
        pair_id = make_pair_id(session_number, block_id, slide_capture_id)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO matching_pairs(
                    pair_id, session_number, work_order, block_id, slide_capture_id,
                    pair_source, is_match
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    work_order=excluded.work_order,
                    pair_source=CASE
                        WHEN matching_pairs.pair_source = 'near_miss'
                             AND excluded.pair_source = 'candidate'
                        THEN matching_pairs.pair_source
                        WHEN matching_pairs.pair_source = 'true_pair'
                        THEN matching_pairs.pair_source
                        ELSE excluded.pair_source
                    END,
                    is_match=COALESCE(excluded.is_match, matching_pairs.is_match)
                """,
                (
                    pair_id, session_number, work_order, block_id, slide_capture_id,
                    pair_source, is_match,
                ),
            )
        return pair_id

    def sync_matching_pairs_for_work_order(
        self, session_number: int, work_order: str,
    ) -> int:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT sc.capture_id, sc.block_id, sc.work_order
                  FROM slide_captures AS sc
                  JOIN sets AS s
                    ON s.session_number = sc.session_number
                   AND s.block_id = sc.block_id
                 WHERE sc.session_number = ?
                   AND sc.work_order = ?
                   AND sc.success = 1
                   AND sc.block_id IS NOT NULL
                   AND s.mask_path IS NOT NULL
                 ORDER BY sc.captured_at, sc.capture_id
                """,
                (session_number, work_order),
            ).fetchall()
        true_pairs = [
            TruePairRef(session_number, str(row["work_order"]), str(row["block_id"]),
                        str(row["capture_id"]))
            for row in rows
        ]
        count = 0
        for pair in true_pairs:
            self.upsert_matching_pair(
                session_number=pair.session_number,
                work_order=pair.work_order,
                block_id=pair.block_id,
                slide_capture_id=pair.slide_capture_id,
                pair_source="true_pair",
                is_match=1,
            )
            count += 1
        for candidate in expand_same_work_order_candidates(true_pairs):
            self.upsert_matching_pair(
                session_number=candidate.session_number,
                work_order=candidate.work_order,
                block_id=candidate.block_id,
                slide_capture_id=candidate.slide_capture_id,
                pair_source="candidate",
                is_match=0,
            )
            count += 1
        return count

    def list_matching_pairs(
        self,
        session_number: int | None = None,
        *,
        unscored_only: bool = False,
    ) -> tuple[dict[str, object], ...]:
        clauses: list[str] = []
        params: list[object] = []
        if session_number is not None:
            clauses.append("session_number=?")
            params.append(session_number)
        if unscored_only:
            clauses.append("classical_score IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM matching_pairs {where} ORDER BY pair_id",
                params,
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def write_matching_pair_score(
        self,
        pair_id: str,
        *,
        classical_score: float,
        rank_for_block: int | None,
        metric: str,
        scored_at: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE matching_pairs
                   SET classical_score=?, rank_for_block=?, metric=?, scored_at=?
                 WHERE pair_id=?
                """,
                (classical_score, rank_for_block, metric, scored_at, pair_id),
            )

    def promote_matching_near_misses(
        self, session_number: int, work_order: str, *, margin: float,
    ) -> int:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT pair_id, block_id, is_match, classical_score
                  FROM matching_pairs
                 WHERE session_number=? AND work_order=?
                   AND classical_score IS NOT NULL
                """,
                (session_number, work_order),
            ).fetchall()
            scored = [
                ScoredPair(
                    r["pair_id"],
                    r["block_id"],
                    bool(r["is_match"]),
                    float(r["classical_score"]),
                )
                for r in rows
            ]
            promoted = promote_near_misses(scored, margin=margin)
            for pair_id in promoted:
                db.execute(
                    """
                    UPDATE matching_pairs
                       SET pair_source='near_miss'
                     WHERE pair_id=? AND pair_source != 'true_pair'
                    """,
                    (pair_id,),
                )
        return len(promoted)

    def _session_identity(self, number: int) -> SessionIdentity:
        with self._connect() as db:
            row = db.execute(
                "SELECT started_at, session_mode FROM sessions WHERE session_number=?",
                (number,),
            ).fetchone()
        if row is None:
            raise ValueError("unknown session")
        started = datetime.fromisoformat(row["started_at"])
        directory = next(self.root.glob(f"session_{number:06d}_*"), None)
        if directory is None:
            raise ValueError("session metadata directory is missing")
        return SessionIdentity(number, started, directory, str(row["session_mode"]))

    def results_evidence_bytes(
        self, session_number: int, artifact_path: str | Path
    ) -> bytes | None:
        """Return one session-owned claim-artifact JPEG, or ``None``.

        The processing computer owns these artifacts.  In deployed sessions the
        Pi may ask for a Windows path from a results row, so path validation
        happens here, on the machine that owns the file.  Resolving both paths
        before the containment check rejects ``..`` traversal and any artifact
        from another session.
        """
        session = self._session_identity(session_number)
        artifacts = (session.directory / "claim_artifacts").resolve()
        requested = Path(artifact_path).resolve()
        try:
            requested.relative_to(artifacts)
        except ValueError:
            return None
        if requested.suffix.lower() not in {".jpg", ".jpeg"}:
            return None
        try:
            return requested.read_bytes()
        except OSError:
            return None


def _elapsed_ms(finished_ns: int | None, started_ns: int | None) -> int | None:
    if finished_ns is None or started_ns is None:
        return None
    return int(round((finished_ns - started_ns) / 1_000_000))


def preprocess_block(
    path: Path, *, profile: bool = False
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Run the existing production block preparation behind the queue seam."""
    stage_timings: dict[str, int] | None = {} if profile else None
    if stage_timings is None:
        prepared = prepare_specimen(path, "block")
    else:
        prepared = prepare_specimen(path, "block", stage_timings=stage_timings)
    if isinstance(prepared, PreparationFailure):
        raise ValueError(prepared.reason)
    metadata: dict[str, object] = {
        "role": prepared.role,
        "roi_ok": prepared.roi_ok,
        "roi_reason": prepared.roi_reason,
        "segmentation_backend": prepared.segmentation_backend,
    }
    if stage_timings is not None:
        metadata.update(stage_timings)
    return prepared.mask, metadata


def preprocess_slide(img: np.ndarray) -> PreparedResult:
    """Run the existing production slide preparation behind the claim seam.

    Unlike ``preprocess_block``, failure is returned rather than raised: the
    shared gates/scorer composition already turns a ``PreparationFailure``
    into a fail-closed REVIEW, so the claim seam must not treat it as an
    unexpected exception.
    """
    return prepare_specimen_from_image(img, "slide")


def default_work_order_scorer(
    block_results: Mapping[str, PreparedResult],
    slide_results: Mapping[str, PreparedResult],
) -> WorkOrderScoringResult:
    """Score every slide-block pair in one work order (#149, ADR 0009).

    Reuses ``pipeline.decide_claim`` (``pair_composition.compose_prepared_pair``
    under the hood) -- the same production scorer ``resolve_claim`` uses for
    one claimed pair -- looped over the full bipartite set. Deliberately does
    NOT import ``tools/scoring_diagnostics/pair_diagnostics.py``: that would
    violate the ``code/`` -> ``tools/`` architecture boundary.

    Per-item / per-pair boundary (ADR 0011, issue #158): the 256x256
    normalization + component-feature cache (``build_locked_score_cache``)
    depends only on one specimen, so a pre-pass builds it once per item
    (M+K) instead of once per pair (2*M*K). Cache lookup keys on object
    identity (``id(result)``), not the id string, because the loop below
    passes these exact ``PreparedSpecimen`` objects down through
    ``decide_claim`` -> ``compose_prepared_pair`` -> the injected scorer.
    ``PreparationFailure`` results are skipped -- they never reach the
    scorer (the gate short-circuits first).
    """
    block_caches = {
        id(result): build_locked_score_cache(result)
        for result in block_results.values()
        if isinstance(result, PreparedSpecimen)
    }
    slide_caches = {
        id(result): build_locked_score_cache(result)
        for result in slide_results.values()
        if isinstance(result, PreparedSpecimen)
    }

    def cached_scorer(
        block_result: PreparedSpecimen,
        slide_result: PreparedSpecimen,
        *,
        observer: RuntimeObserver | None = None,
        item_id: str = "",
    ) -> ProductionScoreResult:
        return score_routed_caches(
            block_caches[id(block_result)],
            slide_caches[id(slide_result)],
            observer=observer,
            item_id=item_id,
        )

    scores: dict[str, dict[str, float | None]] = {}
    pair_decisions: dict[str, dict[str, ClaimDecision]] = {}
    for slide_capture_id, slide_result in slide_results.items():
        row_scores: dict[str, float | None] = {}
        row_decisions: dict[str, ClaimDecision] = {}
        for block_id, block_result in block_results.items():
            decision = decide_claim(
                f"{slide_capture_id}:{block_id}", block_result, slide_result,
                scorer=cached_scorer,
            )
            row_scores[block_id] = decision.score
            row_decisions[block_id] = decision
        scores[slide_capture_id] = row_scores
        pair_decisions[slide_capture_id] = row_decisions
    return WorkOrderScoringResult(scores=scores, pair_decisions=pair_decisions)


def format_profile_summary_row(
    capture_id: str, fields: Mapping[str, object]
) -> dict[str, object]:
    """Pure dict-to-row shaping for `ProcessingStore.record_profile_capture` (#168).

    `fields` is the stage-timing subset of `SuccessfulCapture.metadata`;
    only the keys present in `CAPTURE_STAGE_TIMING_KEYS` are carried over.
    """
    row: dict[str, object] = {"capture_id": capture_id}
    for key in (*CAPTURE_STAGE_TIMING_KEYS, *SETTLING_STAGE_KEYS):
        if key in fields:
            row[key] = fields[key]
    return row
