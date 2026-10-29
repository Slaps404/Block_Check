"""Explicit, mutually exclusive session scoring mode.

Independent mode booleans on the Pi capture runtime (``open_retrieval``,
``hybrid``, ``hybrid_shadow``) could be true at once, a meaningless state.
This module expresses the mode as one explicit, exhaustive, four-member
:class:`SessionMode` plus a pure resolver that rejects any conflicting
combination loudly instead of picking a winner.

``--profile`` is a deliberately separate, orthogonal instrumentation axis
(see CONTEXT.md "Profile Mode") -- it is not a scoring mode and has no
member here and no parameter on :func:`resolve_session_mode`.

Pure data in, enum out -- no cv2/numpy/scipy, no I/O, no argparse
dependency. Mirrors the ``evaluate_work_order``/``WorkOrderVerdict`` pattern
in ``code/retrieval/work_order_evaluator.py``.
"""
from __future__ import annotations

from enum import Enum


class SessionMode(Enum):
    """The four mutually exclusive session-wide scoring modes.

    ``NORMAL`` is closed-set per-slide verification (the default).
    ``OPEN_RETRIEVAL`` is work-order-scoped retrieval (ADR 0009 / ADR 0016).
    ``HYBRID`` and ``HYBRID_SHADOW`` freeze a Hybrid Candidate Pool at
    Finish Blocks and score slides against it.
    """

    NORMAL = "normal"
    OPEN_RETRIEVAL = "open_retrieval"
    HYBRID = "hybrid"
    HYBRID_SHADOW = "hybrid_shadow"


# Flag NAMES may be revised while mode semantics stay fixed. Every call
# site (argparse, the PowerShell launcher help text, error messages) must
# read this constant rather than the literal string, so a rename stays a
# one-line change here.
OPEN_RETRIEVAL_FLAG = "--open-retrieval"
HYBRID_FLAG = "--hybrid"
HYBRID_SHADOW_FLAG = "--hybrid-shadow"

_FLAG_BY_MODE: dict[SessionMode, str] = {
    SessionMode.OPEN_RETRIEVAL: OPEN_RETRIEVAL_FLAG,
    SessionMode.HYBRID: HYBRID_FLAG,
    SessionMode.HYBRID_SHADOW: HYBRID_SHADOW_FLAG,
}


class SessionModeConflictError(ValueError):
    """Raised when two or more scoring-mode flags are set at once.

    Callers must reject this loudly at startup -- before capture can begin
    -- rather than silently resolving to one of the conflicting modes.
    """


def resolve_session_mode(
    *,
    open_retrieval: bool = False,
    hybrid: bool = False,
    hybrid_shadow: bool = False,
) -> SessionMode:
    """Resolve the one active :class:`SessionMode` from independent launch
    booleans.

    Exactly zero or one of the three keyword flags may be ``True``; with
    none set the result is ``SessionMode.NORMAL`` (today's default,
    unchanged). Two or more set raises :class:`SessionModeConflictError`
    naming every conflicting flag, so callers can print the message and
    exit non-zero before any camera/store/workflow side effect runs.
    """
    selected = [
        mode
        for mode, flag in (
            (SessionMode.OPEN_RETRIEVAL, open_retrieval),
            (SessionMode.HYBRID, hybrid),
            (SessionMode.HYBRID_SHADOW, hybrid_shadow),
        )
        if flag
    ]
    if len(selected) > 1:
        flags = ", ".join(_FLAG_BY_MODE[mode] for mode in selected)
        raise SessionModeConflictError(
            "Only one session scoring-mode flag may be set at a time, "
            f"got: {flags}. --profile is independent and may combine with "
            "any one of them."
        )
    if not selected:
        return SessionMode.NORMAL
    return selected[0]
