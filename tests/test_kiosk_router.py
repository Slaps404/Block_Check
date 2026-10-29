"""The event-driven screen->state router.

`select_screen(state)` is the single tested source of truth for WHICH screen is
active. It is a pure function of a state snapshot (workflow signals + the client's
own UI flags), so screen selection is driven by state -- never by deck navigation
(the whole reason ADR 0004 dropped the deck runtime). Rule order matters:
first match wins, and this suite pins the load-bearing precedence decisions.
"""
from __future__ import annotations

import pytest

from kiosk.router import effective_engaged, select_screen


def _state(**overrides):
    base = dict(
        engaged=True,
        session_present=True,
        phase="blocks",
        capture_state="WAITING_FOR_SCAN",
        capture_mode="block",
        latest_verdict=None,
        current_capture_id=None,
        finish_blocks_guard=False,
        finish_slides_guard=False,
    )
    base.update(overrides)
    return base


# All slide verdict screens share WAITING_FOR_REMOVAL.
_REMOVAL = dict(
    phase="slides", capture_mode="slide",
    capture_state="WAITING_FOR_REMOVAL", current_capture_id="cap-7",
)
_SLIDE_IDLE = dict(phase="slides", capture_mode="slide", capture_state="EMPTY")


def _verdict(kind, capture_id="cap-7"):
    return {"kind": kind, "capture_id": capture_id}


_ROUTES = [
    # -- boot chooser (client not yet engaged) --
    ("boot_no_session", dict(engaged=False, session_present=False), "01"),
    # Launcher always attaches a session number; captured==0 means no work yet.
    ("boot_empty_session", dict(engaged=False, captured=0), "01"),
    ("boot_missing_captured_is_zero", dict(engaged=False), "01"),
    ("boot_unfinished_resumes", dict(engaged=False, captured=1), "02"),
    # Captured counts verdicted pairs only; captured blocks still resume.
    ("boot_blocks_captured_none_scored",
     dict(engaged=False, captured=0, blocks_captured=65), "02"),
    ("boot_finalized_starts_fresh", dict(engaged=False, phase="finalized"), "01"),
    # -- lifecycle / transitional --
    ("finalized", dict(phase="finalized"), "21"),
    # Precedence: lifecycle wins over a leftover capture state.
    ("finalized_beats_stale_capture",
     dict(phase="finalized", capture_state="SETTLING"), "21"),
    ("draining_blocks", dict(phase="draining_blocks"), "processing"),
    ("finalizing", dict(phase="finalizing", capture_mode="slide",
                        capture_state="EMPTY"), "processing"),
    # -- confirm guards, gated on an idle capture state --
    ("finish_blocks_guard_idle", dict(finish_blocks_guard=True), "10"),
    # A late placement must never be masked by a stale confirm modal.
    ("finish_blocks_guard_defers_to_live_capture",
     dict(finish_blocks_guard=True, capture_state="SETTLING"), "hold_still"),
    ("finish_slides_guard_idle",
     dict(finish_slides_guard=True, **_SLIDE_IDLE), "20"),
    # -- transients / errors --
    ("capture_error", dict(capture_state="CAPTURE_ERROR"), "19"),
    ("calibration_failed", dict(capture_state="CALIBRATION_FAILED"), "04"),
    ("reposition_slide", dict(phase="slides", capture_mode="slide",
                              capture_state="REPOSITION_SLIDE"), "17"),
    # -- calibration: both awaiting and building show Calibrating --
    ("awaiting_baseline", dict(capture_state="AWAITING_BASELINE_CONFIRMATION"), "05"),
    ("building_baseline", dict(capture_state="BUILDING_BASELINE"), "05"),
    # -- capture review gate (before removal rules) --
    ("review_block", dict(capture_state="AWAITING_ACCEPT"), "capture_review"),
    ("review_slide", dict(phase="slides", capture_mode="slide",
                          capture_state="AWAITING_ACCEPT"), "capture_review"),
    ("review_beats_id_matched_verdict",
     dict(_REMOVAL, capture_state="AWAITING_ACCEPT",
          latest_verdict=_verdict("claim_pass")), "capture_review"),
    # -- slide verdict split --
    ("id_matched_pass", dict(_REMOVAL, latest_verdict=_verdict("claim_pass")), "15"),
    ("id_matched_review", dict(_REMOVAL, latest_verdict=_verdict("claim_review")), "16"),
    # A PASS for a DIFFERENT capture id must not paint this slide green.
    ("stale_verdict_does_not_bleed",
     dict(_REMOVAL, latest_verdict=_verdict("claim_pass", "cap-6")), "14"),
    ("removal_without_verdict_is_verifying", dict(_REMOVAL), "14"),
    # -- removal / capturing: mode picks the variant --
    ("block_removal", dict(capture_state="WAITING_FOR_REMOVAL"), "09"),
    ("block_capture_requested", dict(capture_state="CAPTURE_REQUESTED"), "08"),
    ("slide_capture_requested", dict(phase="slides", capture_mode="slide",
                                     capture_state="CAPTURE_REQUESTED"), "13"),
    # -- idle resting screens --
    ("block_place", dict(capture_state="EMPTY"), "07"),
    ("block_scan", dict(), "06"),
    # ADR 0005: SETTLING is Hold Still, not Place and not session Processing.
    ("block_settling", dict(capture_state="SETTLING"), "hold_still"),
    ("slide_settling", dict(phase="slides", capture_mode="slide",
                            capture_state="SETTLING"), "hold_still"),
    ("slide_place", dict(_SLIDE_IDLE), "12"),
    # -- degraded: no capture session (bare workflow / pre-attach) --
    ("degraded_blocks", dict(capture_state=None, capture_mode=None), "06"),
    ("degraded_slides",
     dict(capture_state=None, capture_mode=None, phase="slides"), "12"),
    # -- results table: guard + results-ready data, either alone no-ops --
    ("results_guard_with_data",
     dict(view_results_guard=True, results_ready_work_orders=(7,)), "results_table"),
    ("results_guard_without_data",
     dict(view_results_guard=True, results_ready_work_orders=()), "06"),
    ("results_data_without_guard",
     dict(view_results_guard=False, results_ready_work_orders=(7,)), "06"),
    # Hybrid rows are available after FINISH WORK ORDER even while scoring.
    ("finished_hybrid_order_escapes_to_results",
     dict(mode="hybrid", work_order_open=False, has_work_orders=True,
          engaged=False, phase="slides", view_results_guard=True,
          results_ready_work_orders=(7,)), "results_table"),
    # -- attention correction flow: same guard+data shape --
    ("recapture_guard_with_attention",
     dict(recapture_guard=True, attention={"capture_id": "cap-1"}), "hybrid_attention"),
    ("recapture_guard_without_attention",
     dict(recapture_guard=True, attention=None), "06"),
    # The passive banner never takes over the active screen on its own.
    ("attention_without_recapture_guard",
     dict(recapture_guard=False, attention={"capture_id": "cap-1"}), "06"),
]


@pytest.mark.parametrize(
    "overrides, expected", [(o, e) for _, o, e in _ROUTES], ids=[i for i, _, _ in _ROUTES]
)
def test_select_screen_routes(overrides, expected):
    assert select_screen(_state(**overrides)) == expected


# -- work-order bracket modes (Open Retrieval, Hybrid) ---------------------
# Open Retrieval gates on the open_retrieval flag; Hybrid/Hybrid Shadow gate
# on the `mode` string relay.py publishes (open_retrieval stays False there).

_BRACKET = {
    "open_retrieval": dict(open_retrieval=True),
    "hybrid": dict(mode="hybrid", open_retrieval=False),
    "hybrid_shadow": dict(mode="hybrid_shadow", open_retrieval=False),
}
bracket_modes = pytest.mark.parametrize("bracket", list(_BRACKET))


def _wo(bracket, **overrides):
    return _state(**_BRACKET[bracket], **overrides)


@bracket_modes
def test_bracket_without_any_work_orders_shows_first_work_order(bracket):
    # A durable work-order-history decision, not a guess from captured sets.
    assert select_screen(_wo(
        bracket, work_order_open=False, engaged=False, has_work_orders=False,
        captured=0,
    )) == "first_work_order"
    assert select_screen(_wo(
        bracket, work_order_open=False, engaged=True, has_work_orders=False,
    )) == "first_work_order"


@bracket_modes
def test_bracket_closed_after_a_work_order_shows_between_orders_and_results_gate(bracket):
    closed = _wo(bracket, work_order_open=False, engaged=False,
                 has_work_orders=True, captured=0)
    assert select_screen(closed) == "between_orders"
    # The results gate (guard + ready data) still escapes.
    assert select_screen(dict(
        closed, view_results_guard=True, results_ready_work_orders=(7,),
    )) == "results_table"
    # Either half alone stays on between_orders.
    assert select_screen(dict(
        closed, view_results_guard=True, results_ready_work_orders=(),
    )) == "between_orders"
    assert select_screen(dict(
        closed, view_results_guard=False, results_ready_work_orders=(7,),
    )) == "between_orders"


@bracket_modes
def test_bracket_open_routes_idle_to_work_order_screens(bracket):
    assert select_screen(
        _wo(bracket, work_order_open=True)
    ) == "block_scan_work_order"
    assert select_screen(
        _wo(bracket, work_order_open=True, **_SLIDE_IDLE)
    ) == "slide_capture_work_order"


@bracket_modes
def test_bracket_open_reaches_calibrating_while_disengaged(bracket):
    # An open work order counts as engaged even with the client latch
    # off, so AWAITING_BASELINE routes to Calibrating, not the boot chooser.
    assert select_screen(_wo(
        bracket, work_order_open=True, engaged=False,
        capture_state="AWAITING_BASELINE_CONFIRMATION",
    )) == "05"


@bracket_modes
def test_effective_engaged_derivation(bracket):
    # An open bracket engages even with the client latch off; no bracket
    # does not engage on its own.
    assert effective_engaged(
        _wo(bracket, work_order_open=True, engaged=False)
    ) is True
    assert effective_engaged(
        _wo(bracket, work_order_open=False, engaged=False)
    ) is False
    # Closed-set: falls back to the raw client latch.
    assert effective_engaged(_state(engaged=True)) is True
    assert effective_engaged(_state(engaged=False)) is False


@bracket_modes
@pytest.mark.parametrize("verdict_kind", ("claim_pass", "claim_review"))
def test_bracket_open_defers_completed_verdict_to_results(bracket, verdict_kind):
    """Fast background scoring must not show PASS/REVIEW during capture."""
    assert select_screen(
        _wo(bracket, work_order_open=True, capture_state="SETTLING")
    ) == "hold_still"
    assert select_screen(
        _wo(bracket, work_order_open=True, capture_state="AWAITING_ACCEPT")
    ) == "capture_review"
    assert select_screen(_wo(
        bracket, work_order_open=True,
        **dict(_REMOVAL, latest_verdict=_verdict(verdict_kind)),
    )) == "hybrid_slide_queued"
    assert select_screen(
        _wo(bracket, work_order_open=True, **_REMOVAL)
    ) == "hybrid_slide_queued"


@bracket_modes
def test_bracket_slide_capture_never_falls_back_to_preview_on_stale_closed_bracket(
    bracket,
):
    """An active capture instruction outranks a one-poll work-order mismatch."""
    common = dict(
        work_order_open=False, engaged=False, phase="slides", capture_mode="slide",
    )
    assert select_screen(_wo(
        bracket, **common, has_work_orders=True, capture_state="CAPTURE_REQUESTED",
    )) == "13"
    assert select_screen(_wo(
        bracket, **common, has_work_orders=True, capture_state="WAITING_FOR_REMOVAL",
    )) == "hybrid_slide_queued"
    # A placed slide keeps Hold Still even before any work order exists.
    assert select_screen(_wo(
        bracket, **common, has_work_orders=False, capture_state="SETTLING",
    )) == "hold_still"


@bracket_modes
def test_bracket_open_slide_capture_escapes_to_results_and_go_back_resumes(bracket):
    """Results is reachable from an OPEN slide-capture work order, and
    clearing the guard (Go Back) re-derives the SAME open slide-capture
    screen rather than closing slide intake."""
    common = dict(work_order_open=True, results_ready_work_orders=(7,), **_SLIDE_IDLE)
    assert select_screen(_wo(bracket, **common, view_results_guard=True)) == "results_table"
    assert select_screen(
        _wo(bracket, **common, view_results_guard=False)
    ) == "slide_capture_work_order"


def test_select_screen_never_mutates_the_state_it_is_given():
    """The screen router remains a pure
    projection with no side effects on the mapping it reads."""
    state = _wo(
        "hybrid", work_order_open=True, view_results_guard=True,
        results_ready_work_orders=(7,), **_SLIDE_IDLE,
    )
    before = dict(state)

    select_screen(state)

    assert state == before
