"""#258: the shared pure Hybrid ``--profile`` formatter.

Both the touchscreen (``code/kiosk/``) and the console render Hybrid queue
count / per-slide stage timing from THIS module and nothing else, so they
can never disagree about a number or a label. Pure data in, data/strings
out: no I/O, no store access, no ``cv2``, no clock reads except the
``now_ns`` value a caller passes in -- that is what makes ``elapsed_ms``
deterministic under a controlled clock in tests.

``ProcessingStore.list_hybrid_profile_rows`` is the one durable source this
module's ``project_profile_rows`` consumes; see that method's docstring for
the raw row shape (``job_state``, ``profile_current_stage``, etc.). This
module is what turns that raw, partly-internal shape into the
internal-lifecycle-free ``ProfileRow`` both renderers below read.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

# The five Hybrid worker stages #258 instruments, in pipeline order. This is
# the single name authority: `ProcessingStore._score_hybrid_slide`/
# `_persist_hybrid_profile_result` write these exact keys into
# `slide_captures.profile_stage_ms_json`, and every reader here (and the
# touchscreen/console callers of this module) matches against the same
# tuple rather than a second hardcoded list.
PROFILE_STAGE_ORDER = (
    "queue_wait", "preparation", "heuristic_selection", "accurate_scoring",
    "artifact_write",
)

# Durable `slide_captures.job_state` values that mean "still running" --
# these, and ONLY these, ever project to the visible "PENDING" state. Every
# other internal lifecycle string (`complete`, `superseded`, and `job_state`
# itself being absent/NULL for a non-Hybrid row) is handled by the
# else-branch in `_visible_state` below, never leaked as-is.
_PENDING_JOB_STATES = frozenset({"queued", "preparing", "scoring"})

# A `queued` job has not reached the worker yet, so the store never stamps
# `profile_current_stage` for it -- this is the one default a pending row's
# stage falls back to when that column is still NULL.
_QUEUED_STAGE = "queue_wait"

_SHADOW_TAG = "SHADOW"
_SHADOW_NOTE = "Hybrid Shadow: complete-pool cost, not deployed Hybrid timing"


@dataclass(frozen=True)
class ProfileRow:
    """One Hybrid slide's profiled state, ready for direct display.

    ``state`` is a VISIBLE state only -- never a durable ``job_state``
    string. ``stage``/``elapsed_ms`` are populated only while ``state`` is
    ``"PENDING"``; ``total_ms``/``stage_ms`` are populated only once the job
    is no longer pending. ``shadow=True`` marks a ``hybrid_shadow`` row's
    COMPLETE-pool cost, which must never be read as pruned Hybrid timing.
    """

    capture_id: str
    block_id: str
    state: str
    stage: str | None
    elapsed_ms: int | None
    total_ms: int | None
    stage_ms: Mapping[str, int]
    shadow: bool


def _visible_state(job_state: object, verdict: object) -> str:
    if job_state == "error":
        return "ERROR"
    if job_state in _PENDING_JOB_STATES:
        return "PENDING"
    if verdict in ("PASS", "REVIEW"):
        return str(verdict)
    # Defensive-only fallback: a row this method was never meant to see (no
    # durable verdict, no pending/error job_state) still renders as a
    # visible state instead of leaking whatever the raw value was.
    return "REVIEW"


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _stage_ms_breakdown(raw: object) -> dict[str, int]:
    """Parse a persisted ``profile_stage_ms_json`` value into ints only.

    Tolerates a missing/NULL column (pre-#258 row), unparsable JSON, a
    non-mapping payload, and individual stage values that are ``null``
    (a stage that never ran, e.g. a gate-failed slide's selection/scoring)
    -- every one of those degrades to that key simply being absent from the
    returned mapping, never a raised exception.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    breakdown: dict[str, int] = {}
    for name in PROFILE_STAGE_ORDER:
        value = _as_int(parsed.get(name))
        if value is not None:
            breakdown[name] = value
    return breakdown


def project_profile_rows(
    rows: Iterable[Mapping[str, Any]], *, now_ns: int
) -> tuple[ProfileRow, ...]:
    """Translate raw ``ProcessingStore.list_hybrid_profile_rows`` dicts into
    the display-ready, internal-lifecycle-free ``ProfileRow`` shape.

    ``now_ns`` is the ONLY clock value this function ever sees -- it never
    reads a clock itself. A missing/NULL ``queued_ns`` (a pre-#258 row, or
    one that never captured a queue timestamp for any reason) degrades
    ``elapsed_ms`` to ``None`` rather than raising or fabricating a number.
    """
    projected: list[ProfileRow] = []
    for row in rows:
        job_state = row.get("job_state")
        state = _visible_state(job_state, row.get("verdict"))
        pending = state == "PENDING"
        stage: str | None = None
        elapsed_ms: int | None = None
        total_ms: int | None = None
        stage_ms: dict[str, int] = {}
        if pending:
            raw_stage = row.get("stage")
            stage = (
                str(raw_stage) if raw_stage in PROFILE_STAGE_ORDER else _QUEUED_STAGE
            )
            queued_ns = _as_int(row.get("queued_ns"))
            if queued_ns is not None:
                elapsed_ms = int(round((now_ns - queued_ns) / 1_000_000))
        else:
            total_ms = _as_int(row.get("total_ms"))
            stage_ms = _stage_ms_breakdown(row.get("stage_ms_json"))
        projected.append(
            ProfileRow(
                capture_id=str(row.get("capture_id", "")),
                block_id=str(row.get("block_id", "")),
                state=state,
                stage=stage,
                elapsed_ms=elapsed_ms,
                total_ms=total_ms,
                stage_ms=stage_ms,
                shadow=bool(row.get("shadow")),
            )
        )
    return tuple(projected)


def profile_screen_fields(
    rows: Sequence[ProfileRow], *, queue_count: int
) -> dict[str, Any]:
    """Touchscreen field payload: the queue count plus one entry per row.

    Every entry carries only visible-state data. A shadow row additionally
    carries an explicit ``shadow_note`` string, so a shadow row's numbers
    can never be mistaken for pruned Hybrid timing on the screen either.
    """
    entries: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "capture_id": row.capture_id,
            "block_id": row.block_id,
            "state": row.state,
            "shadow": row.shadow,
        }
        if row.shadow:
            entry["shadow_note"] = _SHADOW_NOTE
        if row.stage is not None:
            entry["stage"] = row.stage
        if row.elapsed_ms is not None:
            entry["elapsed_ms"] = row.elapsed_ms
        if row.total_ms is not None:
            entry["total_ms"] = row.total_ms
        if row.stage_ms:
            entry["stage_breakdown"] = dict(row.stage_ms)
        entries.append(entry)
    return {"queue_count": queue_count, "rows": tuple(entries)}


def format_profile_console(rows: Sequence[ProfileRow], *, queue_count: int) -> str:
    """One human-readable line per row plus a queue-count header.

    Renders the SAME ``ProfileRow`` data ``profile_screen_fields`` does, so
    the console and the touchscreen can never disagree about a number or
    whether a row is Hybrid Shadow.
    """
    lines = [f"Hybrid profile: queue={queue_count}"]
    for row in rows:
        tag = f" [{_SHADOW_TAG}]" if row.shadow else ""
        if row.state == "PENDING":
            elapsed = "?" if row.elapsed_ms is None else f"{row.elapsed_ms}ms"
            stage = row.stage or "?"
            lines.append(
                f"  {row.capture_id} {row.block_id} {row.state}{tag} "
                f"stage={stage} elapsed={elapsed}"
            )
        else:
            total = "?" if row.total_ms is None else f"{row.total_ms}ms"
            breakdown = " ".join(
                f"{name}={row.stage_ms[name]}ms"
                for name in PROFILE_STAGE_ORDER
                if name in row.stage_ms
            )
            suffix = f" [{breakdown}]" if breakdown else ""
            lines.append(
                f"  {row.capture_id} {row.block_id} {row.state}{tag} "
                f"total={total}{suffix}"
            )
    return "\n".join(lines)
