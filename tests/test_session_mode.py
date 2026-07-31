"""Explicit mutually-exclusive session scoring mode (#247 prefactor).

TDD for the pure mode type + resolver that replaces the ad hoc
``open_retrieval``/``hybrid``/``hybrid_shadow`` booleans threaded around the
Pi capture runtime. See docs/adr/0009 + docs/adr/0016 for the Open Retrieval
pattern this generalizes, and CONTEXT.md for the Hybrid/Hybrid Shadow
glossary entries.
"""
from __future__ import annotations

import inspect

import pytest

from session.session_mode import (
    HYBRID_FLAG,
    HYBRID_SHADOW_FLAG,
    OPEN_RETRIEVAL_FLAG,
    SessionMode,
    SessionModeConflictError,
    resolve_session_mode,
)


# --------------------------------------------------------------------------
# The mode type itself: exhaustive, mutually exclusive members
# --------------------------------------------------------------------------


def test_session_mode_has_exactly_four_members():
    assert {member.name for member in SessionMode} == {
        "NORMAL",
        "OPEN_RETRIEVAL",
        "HYBRID",
        "HYBRID_SHADOW",
    }


def test_session_mode_members_have_distinct_values():
    values = [member.value for member in SessionMode]
    assert len(values) == len(set(values))


def test_flag_constants_match_the_literal_cli_flag_strings():
    """Named constants, not literals, are what argparse/PowerShell must read
    (#245: --hybrid-shadow's exact name may still be revised)."""
    assert OPEN_RETRIEVAL_FLAG == "--open-retrieval"
    assert HYBRID_FLAG == "--hybrid"
    assert HYBRID_SHADOW_FLAG == "--hybrid-shadow"


# --------------------------------------------------------------------------
# resolve_session_mode: default + single-flag resolution
# --------------------------------------------------------------------------


def test_resolve_session_mode_defaults_to_normal_with_no_flags():
    assert resolve_session_mode() == SessionMode.NORMAL


def test_resolve_session_mode_open_retrieval_flag_alone():
    assert (
        resolve_session_mode(open_retrieval=True) == SessionMode.OPEN_RETRIEVAL
    )


def test_resolve_session_mode_hybrid_flag_alone():
    assert resolve_session_mode(hybrid=True) == SessionMode.HYBRID


def test_resolve_session_mode_hybrid_shadow_flag_alone():
    assert resolve_session_mode(hybrid_shadow=True) == SessionMode.HYBRID_SHADOW


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
# --profile orthogonality: not a scoring mode, no parameter on the resolver
# --------------------------------------------------------------------------


def test_resolve_session_mode_has_no_profile_parameter():
    """--profile is instrumentation, not a scoring mode (CONTEXT.md "Profile
    Mode"): it must never be wired into the mode resolver as a fourth
    competing flag."""
    params = inspect.signature(resolve_session_mode).parameters
    assert "profile" not in params


def test_resolve_session_mode_rejects_unexpected_profile_kwarg():
    with pytest.raises(TypeError):
        resolve_session_mode(profile=True)  # type: ignore[call-arg]
