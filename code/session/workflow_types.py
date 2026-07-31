"""Shared session-workflow dataclasses, protocols, and wire registration (#201 slice 2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol

from session.pipeline import ClaimDecision
from slide.qr import DecodeAttempt, SlideQRResult
import store.wire as store_wire


@dataclass(frozen=True)
class SessionIdentity:
    number: int
    started_at: datetime
    directory: Path
    # #269 startup-mismatch fix: the durable ``sessions.session_mode`` value,
    # carried over the SAME `start_session`/`resume_session` round trip that
    # already crosses `/rpc` -- not a new store method. Defaults to "normal"
    # so old callers that still construct `SessionIdentity(number, started,
    # directory)` positionally (and any `decode()` payload from before this
    # field existed) keep working unchanged (see `store.wire.decode`'s
    # "keys absent from data are omitted... dataclass defaults apply").
    # Consumed by `tools/run_pi_session.py::main` to refuse startup when the
    # Pi's own resolved `SessionMode` disagrees with this value.
    session_mode: str = "normal"


@dataclass(frozen=True)
class WorkOrderScoringResult:
    """N^2 work-order scoring output: float map for the evaluator plus optional
    per-pair ``ClaimDecision`` rows (pose fields) from the same scorer pass."""

    scores: Mapping[str, Mapping[str, float | None]]
    pair_decisions: Mapping[str, Mapping[str, ClaimDecision]] = field(
        default_factory=dict
    )


def normalize_work_order_scoring_result(
    result: WorkOrderScoringResult | Mapping[str, Mapping[str, float | None]],
) -> WorkOrderScoringResult:
    if isinstance(result, WorkOrderScoringResult):
        return result
    return WorkOrderScoringResult(scores=result)


@dataclass(frozen=True)
class WorkflowEvent:
    kind: str
    session_number: int
    phase: str
    message: str
    block_id: str | None = None
    capture_id: str | None = None


@dataclass(frozen=True)
class ScanOutcome:
    accepted: bool
    message: str


@dataclass(frozen=True)
class ClaimOutcome:
    """Result of resolving one valid slide identity against the block inventory."""

    accepted: bool
    message: str
    verdict: str | None = None
    score: float | None = None
    stage: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class UploadReceipt:
    capture_id: str
    acknowledged: bool
    checksum: str


@dataclass(frozen=True)
class SlideStageTimings:
    """#171: per-stage slide-capture durations, mirroring the sub-durations
    `PiCaptureRuntime._consume_capture` forwards to `CaptureController`.

    ``outbox_ms``/``send_ms`` are ``None`` when decode failed and the
    unreadable-slide reposition path skipped instrumenting the outbox/send
    stages (only ``decode_ms`` is meaningful there)."""

    decode_ms: float
    outbox_ms: float | None
    send_ms: float | None
    capture_id: str | None = None


@dataclass(frozen=True)
class FailedBlockWarning:
    block_id: str
    reason: str
    qc_path: Path
    can_recapture: bool = True
    can_dismiss: bool = True


@dataclass(frozen=True)
class SessionSummary:
    """Compact processed/PASS/REVIEW counts plus expandable detail categories."""

    session_number: int
    started_at: datetime
    sets_processed: int
    pass_count: int
    review_count: int
    missing_slides: tuple[str, ...] = ()
    # #188: total non-dismissed block captures this session regardless of
    # verdict status -- distinct from sets_processed (verdicted-only). Lets
    # the boot router tell "no work yet" from "blocks captured, none scored
    # yet". Dismissed (resolved-unusable) blocks are excluded: they are not
    # work left to resume.
    blocks_captured: int = 0
    block_failures: tuple[FailedBlockWarning, ...] = ()
    skipped_decodes: tuple[str, ...] = ()
    pending_blocks: tuple[str, ...] = ()
    pending_uploads: tuple[str, ...] = ()
    finalization_error: str | None = None


@dataclass(frozen=True)
class WorkflowSnapshot:
    session_number: int
    started_at: datetime
    phase: str
    upload_state: str
    preprocessing_pending: int
    latest_block_id: str | None
    latest_block_status: str | None
    pending_transfers: int = 0
    unresolved_blocks: int = 0


@dataclass(frozen=True)
class FramingCalibration:
    image_path: Path
    approved_at: datetime


@dataclass(frozen=True)
class BlockReadiness:
    evaluable: bool
    review_reason: str | None = None


@dataclass(frozen=True)
class RecaptureOutcome:
    """#256: result of one accepted-recapture attempt against an existing
    Hybrid slide capture (``ProcessingStore.recapture_hybrid_slide``).

    Supersession of the prior row is gated STRUCTURALLY on decoded-identity
    equality -- the newly decoded claim's ``block_id`` must equal the
    superseded capture's own claimed block -- inside the store method
    itself, never left to a caller convention. ``accepted`` is True only
    when that check passed and a new Hybrid job now exists for the fresh
    capture; on a mismatch (or no such durable capture at all) this is
    False, ``new_capture_id`` is ``None``, and the original row, its
    ``job_state``, and its verdict are left completely untouched.
    """

    accepted: bool
    message: str
    new_capture_id: str | None = None


@dataclass(frozen=True)
class HybridPoolFreezeResult:
    """Outcome of one `ProcessingStore.freeze_hybrid_pool` call (#250).

    ``frozen`` is the one-way signal: once ``True`` for a session, a repeat
    call returns the identical result (idempotent) and no further freeze
    attempt for that session can ever change it. ``usable_block_ids`` is
    empty only when block work is still resolving; when block work resolved
    with fewer than two usable blocks, it still names whichever 0/1 blocks
    were judged usable, so the operator message is concrete.
    """

    frozen: bool
    usable_block_ids: tuple[str, ...]
    message: str


# Register every wire type the /rpc surface can return or accept so
# store_wire's envelope dispatch (`dumps`/`loads`) can round-trip them
# without the caller importing this module's or slide_qr's/pipeline's
# dataclasses directly.
for _wire_type in (
    SessionIdentity,
    ScanOutcome,
    UploadReceipt,
    ClaimOutcome,
    WorkflowSnapshot,
    SessionSummary,
    FailedBlockWarning,
    BlockReadiness,
    HybridPoolFreezeResult,
    RecaptureOutcome,
    WorkflowEvent,
    SlideQRResult,
    DecodeAttempt,
    ClaimDecision,
):
    store_wire.register(_wire_type)
del _wire_type


@dataclass(frozen=True)
class OutboxCapture:
    capture_id: str
    path: Path
    block_id: str
    checksum: str
    captured_at: datetime
    state: str = "pending"
    recapture: bool = False
    profile: bool = False


@dataclass(frozen=True)
class OutboxSlide:
    capture_id: str
    path: Path
    captured_at: datetime
    result: SlideQRResult
    duration_ms: float
    state: str = "pending"
    profile: bool = False
    # #256 follow-up: the durable "this capture supersedes an existing
    # Hybrid attention item" tag -- set only when `SessionWorkflow.
    # capture_slide` was armed (see `arm_hybrid_recapture`) at the moment
    # THIS capture was taken. Baked into the outbox entry at publish time
    # (never re-derived at replay time) so the routing decision survives a
    # crash/restart between publish and replay exactly like every other
    # durable outbox field. `None` (the default, every pre-existing caller)
    # is the ordinary path: `PiOutbox.replay_slides` calls
    # `store.record_slide_capture`. Non-``None`` routes that SAME replay to
    # `store.recapture_hybrid_slide` instead, naming this as the superseded
    # capture id.
    supersedes: str | None = None


class CaptureTransport(Protocol):
    def status(self, session_number: int) -> Mapping[str, object]: ...

    def upload(
        self, session_number: int, capture: OutboxCapture
    ) -> UploadReceipt: ...
