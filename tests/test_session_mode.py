"""Explicit mutually-exclusive session scoring mode.

Pure mode type + resolver used instead of separate
``open_retrieval``/``hybrid``/``hybrid_shadow`` booleans in the Pi capture
runtime. See docs/adr/0009 + docs/adr/0016 for the Open Retrieval
pattern this generalizes, and CONTEXT.md for the Hybrid/Hybrid Shadow
glossary entries.
"""
from __future__ import annotations

import pytest

from session.session_mode import (
    HYBRID_FLAG,
    OPEN_RETRIEVAL_FLAG,
    SessionMode,
    SessionModeConflictError,
    resolve_session_mode,
)


# --------------------------------------------------------------------------
# resolve_session_mode: default + single-flag resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, SessionMode.NORMAL),
        ({"open_retrieval": True}, SessionMode.OPEN_RETRIEVAL),
        ({"hybrid": True}, SessionMode.HYBRID),
        ({"hybrid_shadow": True}, SessionMode.HYBRID_SHADOW),
    ],
)
def test_resolve_session_mode_single_flag(kwargs, expected):
    assert resolve_session_mode(**kwargs) == expected


# --------------------------------------------------------------------------
# Conflicting flags: rejected loudly, never silently resolved
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"open_retrieval": True, "hybrid": True},
        {"open_retrieval": True, "hybrid_shadow": True},
        {"hybrid": True, "hybrid_shadow": True},
        {"open_retrieval": True, "hybrid": True, "hybrid_shadow": True},
    ],
)
def test_resolve_session_mode_rejects_any_two_scoring_mode_flags(kwargs):
    with pytest.raises(SessionModeConflictError):
        resolve_session_mode(**kwargs)


def test_session_mode_conflict_error_is_a_value_error():
    """Callers (argparse-style startup code) commonly catch ValueError; the
    conflict must still be catchable that way, not just by its subclass."""
    assert issubclass(SessionModeConflictError, ValueError)


def test_conflict_message_names_every_conflicting_flag():
    with pytest.raises(SessionModeConflictError) as exc_info:
        resolve_session_mode(open_retrieval=True, hybrid=True)
    message = str(exc_info.value)
    assert OPEN_RETRIEVAL_FLAG in message
    assert HYBRID_FLAG in message


# --------------------------------------------------------------------------
# --profile orthogonality: not a scoring mode, rejected by the resolver
# --------------------------------------------------------------------------


def test_resolve_session_mode_rejects_unexpected_profile_kwarg():
    with pytest.raises(TypeError):
        resolve_session_mode(profile=True)  # type: ignore[call-arg]
