"""The event-driven screen->state router (#122 / 119b).

`select_screen(state)` is the single tested source of truth for WHICH screen is
active. It is a pure function of a state snapshot (workflow signals + the client's
own UI flags), so screen selection is driven by state -- never by deck navigation
(the whole reason ADR 0004 dropped the deck runtime). Rule order matters:
first match wins, and this suite pins the load-bearing precedence decisions.
"""
from __future__ import annotations

import pytest

from kiosk.router import select_screen


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


# -- boot chooser (client not yet engaged) ---------------------------------


def test_boot_no_session_shows_startup():
    assert select_screen(_state(engaged=False, session_present=False)) == "01"


def test_boot_empty_attached_session_shows_startup_not_resume():
    # Launcher always attaches a session number; captured==0 means no work yet.
    assert select_screen(
        _state(engaged=False, session_present=True, phase="blocks", captured=0)
    ) == "01"
    assert select_screen(
        _state(engaged=False, session_present=True, phase="blocks")
    ) == "01"  # missing captured treated as 0


def test_boot_unfinished_session_with_captures_shows_resume():
    assert select_screen(
        _state(
            engaged=False,
            session_present=True,
            phase="blocks",
            captured=1,
        )
    ) == "02"


def test_boot_blocks_captured_but_none_scored_shows_resume():
    # #188: sets_processed (captured) counts verdicted pairs only. A session
    # with real blocks captured but zero slides scored yet must still resume,
    # not misread as an empty/no-work session.
    assert select_screen(
        _state(
            engaged=False,
            session_present=True,
            phase="blocks",
            captured=0,
            blocks_captured=65,
        )
    ) == "02"


def test_boot_finalized_session_is_nothing_to_resume_so_startup():
    assert select_screen(
        _state(engaged=False, session_present=True, phase="finalized")
    ) == "01"


# -- lifecycle / transitional (engaged) ------------------------------------


def test_finalized_shows_session_complete():
    assert select_screen(_state(phase="finalized")) == "21"


def test_finalized_beats_any_stale_capture_state():
    # Precedence: lifecycle wins over a leftover capture state.
    assert select_screen(_state(phase="finalized", capture_state="SETTLING")) == "21"


def test_draining_and_finalizing_show_the_processing_screen():
    assert select_screen(_state(phase="draining_blocks")) == "processing"
    assert select_screen(_state(phase="finalizing", capture_mode="slide",
                                capture_state="EMPTY")) == "processing"


# -- confirm guards, gated on an idle capture state ------------------------


def test_finish_blocks_guard_shows_confirm_only_while_idle():
    assert select_screen(
        _state(finish_blocks_guard=True, capture_state="WAITING_FOR_SCAN")
    ) == "10"


def test_finish_blocks_guard_defers_to_a_live_capture():
    # A late placement must never be masked by a stale confirm modal.
    assert select_screen(
        _state(finish_blocks_guard=True, capture_state="SETTLING")
    ) == "hold_still"


def test_finish_slides_guard_shows_confirm_only_while_idle():
    assert select_screen(
        _state(finish_slides_guard=True, phase="slides",
               capture_mode="slide", capture_state="EMPTY")
    ) == "20"


# -- transients / errors override the idle screen --------------------------


def test_capture_error_shows_retry_screen():
    assert select_screen(_state(capture_state="CAPTURE_ERROR")) == "19"


def test_calibration_failed_shows_clear_backlight_screen():
    assert select_screen(_state(capture_state="CALIBRATION_FAILED")) == "04"


def test_calibration_failed_does_not_use_capture_error_screen():
    assert select_screen(_state(capture_state="CALIBRATION_FAILED")) != "19"


def test_reposition_shows_unreadable_screen():
    assert select_screen(
        _state(capture_mode="slide", phase="slides", capture_state="REPOSITION_SLIDE")
    ) == "17"


# -- calibration -----------------------------------------------------------


def test_awaiting_baseline_shows_calibrating():
    # Manual CALIBRATE NOW (screen 04) retired; both awaiting and building
    # show Calibrating… while the client auto-fires confirm_empty.
    assert select_screen(_state(capture_state="AWAITING_BASELINE_CONFIRMATION")) == "05"


def test_building_baseline_shows_calibrating():
    assert select_screen(_state(capture_state="BUILDING_BASELINE")) == "05"


# -- capture review gate (before removal rules) ----------------------------


def test_awaiting_accept_shows_capture_review_for_block_and_slide():
    assert select_screen(
        _state(capture_state="AWAITING_ACCEPT", capture_mode="block")
    ) == "capture_review"
    assert select_screen(
        _state(
            phase="slides",
            capture_state="AWAITING_ACCEPT",
            capture_mode="slide",
        )
    ) == "capture_review"


def test_awaiting_accept_beats_waiting_for_removal():
    # Review gate must win over the generic removal/verdict screens.
    assert select_screen(
        _state(
            phase="slides",
            capture_mode="slide",
            capture_state="AWAITING_ACCEPT",
            latest_verdict={"kind": "claim_pass", "capture_id": "cap-7"},
            current_capture_id="cap-7",
        )
    ) == "capture_review"
    assert select_screen(
        _state(capture_mode="block", capture_state="AWAITING_ACCEPT")
    ) == "capture_review"


# -- slide verdict split (all share WAITING_FOR_REMOVAL) -------------------


def _slide_removal(**overrides):
    return _state(
        phase="slides", capture_mode="slide",
        capture_state="WAITING_FOR_REMOVAL",
        current_capture_id="cap-7", **overrides
    )


def test_id_matched_pass_shows_green():
    assert select_screen(
        _slide_removal(latest_verdict={"kind": "claim_pass", "capture_id": "cap-7"})
    ) == "15"


def test_id_matched_review_shows_red():
    assert select_screen(
        _slide_removal(latest_verdict={"kind": "claim_review", "capture_id": "cap-7"})
    ) == "16"


def test_stale_prior_verdict_does_not_bleed_onto_next_slide():
    # A PASS for a DIFFERENT capture id must not paint this slide green.
    assert select_screen(
        _slide_removal(latest_verdict={"kind": "claim_pass", "capture_id": "cap-6"})
    ) == "14"


def test_removal_without_verdict_yet_is_verifying():
    assert select_screen(_slide_removal(latest_verdict=None)) == "14"


@pytest.mark.parametrize("verdict_kind", ("claim_pass", "claim_review"))
def test_open_hybrid_work_order_suppresses_per_slide_verdict(verdict_kind):
    """#252: background Hybrid completions cannot interrupt slide capture."""
    assert select_screen(
        _slide_removal(
            mode="hybrid", work_order_open=True,
            latest_verdict={"kind": verdict_kind, "capture_id": "cap-7"},
        )
    ) == "hybrid_slide_queued"


# -- removal / capturing -- mode picks the variant -------------------------


def test_block_removal_shows_remove_block():
    assert select_screen(
        _state(capture_mode="block", capture_state="WAITING_FOR_REMOVAL")
    ) == "09"


def test_capture_requested_picks_block_or_slide_capturing():
    assert select_screen(
        _state(capture_mode="block", capture_state="CAPTURE_REQUESTED")
    ) == "08"
    assert select_screen(
        _state(capture_mode="slide", phase="slides", capture_state="CAPTURE_REQUESTED")
    ) == "13"


# -- idle resting screens --------------------------------------------------


def test_block_place_and_scan_screens():
    assert select_screen(_state(capture_state="EMPTY")) == "07"
    assert select_screen(_state(capture_state="WAITING_FOR_SCAN")) == "06"


def test_settling_shows_hold_still_for_block_and_slide():
    # ADR 0005: SETTLING is Hold Still, not Place and not session Processing.
    assert select_screen(_state(capture_state="SETTLING")) == "hold_still"
    assert select_screen(
        _state(phase="slides", capture_mode="slide", capture_state="SETTLING")
    ) == "hold_still"


def test_slide_place_screen():
    assert select_screen(
        _state(phase="slides", capture_mode="slide", capture_state="EMPTY")
    ) == "12"


# -- degraded mode: no capture session (bare workflow / pre-attach) --------


def test_degrades_to_phase_only_when_capture_state_absent():
    assert select_screen(
        _state(capture_state=None, capture_mode=None, phase="blocks")
    ) == "06"
    assert select_screen(
        _state(capture_state=None, capture_mode=None, phase="slides")
    ) == "12"


# -- results table (#150): client-owned guard gated on results-ready data --


def test_results_ready_work_orders_route_to_results_table():
    assert select_screen(
        _state(view_results_guard=True, results_ready_work_orders=(7,))
    ) == "results_table"


def test_results_table_guard_without_any_results_ready_work_order_is_a_no_op():
    # Mirrors finish_blocks_guard/finish_slides_guard: the guard flag alone
    # must not paint the screen -- it needs the data condition too.
    assert select_screen(
        _state(view_results_guard=True, results_ready_work_orders=())
    ) != "results_table"
    assert select_screen(
        _state(view_results_guard=False, results_ready_work_orders=(7,))
    ) != "results_table"


# -- #256 attention correction flow: guard+data pattern, same shape --------


def test_recapture_guard_with_attention_routes_to_hybrid_attention():
    assert select_screen(
        _state(recapture_guard=True, attention={"capture_id": "cap-1"})
    ) == "hybrid_attention"


def test_recapture_guard_without_attention_is_a_no_op():
    assert select_screen(
        _state(recapture_guard=True, attention=None)
    ) != "hybrid_attention"


def test_attention_without_recapture_guard_does_not_route_away():
    """The passive banner never automatically takes over the active
    capture screen -- only the explicit guard tap does."""
    assert select_screen(
        _state(recapture_guard=False, attention={"capture_id": "cap-1"})
    ) != "hybrid_attention"


# -- Open Retrieval work-order controls (#155) -----------------------------


def test_open_retrieval_without_any_work_orders_shows_first_session_screen():
    # The first work order keeps the original startup-preview layout.  This is
    # a durable work-order-history decision, not a guess from captured sets.
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=False,
               has_work_orders=False, session_present=True, phase="blocks",
               captured=0)
    ) == "first_work_order"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=True,
               has_work_orders=False,
               phase="blocks", capture_state="WAITING_FOR_SCAN",
               capture_mode="block")
    ) == "first_work_order"


def test_open_retrieval_after_a_work_order_shows_between_orders():
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=False,
               has_work_orders=True, session_present=True, phase="blocks")
    ) == "between_orders"


def test_open_retrieval_between_orders_escapes_to_results_table():
    # #155 gap fix: with no open work order the operator sits on between_orders,
    # but the existing results gate (view_results_guard + a non-empty
    # results_ready_work_orders) must still be able to pull them to the table.
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=False,
               has_work_orders=True,
               session_present=True, phase="blocks", captured=0,
               view_results_guard=True, results_ready_work_orders=(7,))
    ) == "results_table"


def test_finished_hybrid_work_order_escapes_to_results_table_with_pending_rows():
    """Hybrid rows are available after FINISH WORK ORDER even while scoring."""
    assert select_screen(
        _state(
            mode="hybrid", work_order_open=False, has_work_orders=True,
            engaged=False, session_present=True, phase="slides",
            view_results_guard=True, results_ready_work_orders=(7,),
        )
    ) == "results_table"


def test_open_retrieval_between_orders_stays_without_results_gate():
    # Guard flag alone (no ready orders) or ready orders alone (no guard) must
    # NOT escape -- the operator stays on between_orders, mirroring the
    # results_table data+guard condition everywhere else.
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=False,
               has_work_orders=True,
               session_present=True, phase="blocks", captured=0,
               view_results_guard=True, results_ready_work_orders=())
    ) == "between_orders"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=False, engaged=False,
               has_work_orders=True,
               session_present=True, phase="blocks", captured=0,
               view_results_guard=False, results_ready_work_orders=(7,))
    ) == "between_orders"


def test_open_retrieval_with_work_order_open_shows_block_scan_work_order_not_06():
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, phase="blocks",
               capture_mode="block", capture_state="WAITING_FOR_SCAN")
    ) == "block_scan_work_order"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, phase="blocks",
               capture_mode="block", capture_state="WAITING_FOR_SCAN")
    ) != "06"


def test_open_retrieval_open_bracket_reaches_calibrating_while_disengaged():
    # #159 bug: an open work order must be treated as engaged even when the
    # client latch is false, so an AWAITING_BASELINE session routes to
    # Calibrating (05) instead of the boot chooser. Regression for the hang.
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, engaged=False,
               phase="blocks", capture_mode="block",
               capture_state="AWAITING_BASELINE_CONFIRMATION")
    ) == "05"


def test_effective_engaged_derivation():
    from kiosk.router import effective_engaged
    # open bracket engages even with the client latch off
    assert effective_engaged(
        _state(open_retrieval=True, work_order_open=True, engaged=False)
    ) is True
    # open-retrieval with NO bracket does not engage on its own
    assert effective_engaged(
        _state(open_retrieval=True, work_order_open=False, engaged=False)
    ) is False
    # closed-set: falls back to the raw client latch
    assert effective_engaged(_state(engaged=True)) is True
    assert effective_engaged(_state(engaged=False)) is False


def test_open_retrieval_with_work_order_open_routes_slide_idle_to_finish_screen():
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, phase="slides",
               capture_mode="slide", capture_state="EMPTY")
    ) == "slide_capture_work_order"


@pytest.mark.parametrize("verdict_kind", ("claim_pass", "claim_review"))
def test_open_retrieval_defers_completed_verdict_to_results(verdict_kind):
    """Fast background scoring must not show PASS/REVIEW during capture."""
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, capture_state="SETTLING")
    ) == "hold_still"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True, capture_mode="block",
               capture_state="AWAITING_ACCEPT")
    ) == "capture_review"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True,
               **_slide_removal(latest_verdict={"kind": verdict_kind,
                                                "capture_id": "cap-7"}))
    ) == "hybrid_slide_queued"
    assert select_screen(
        _state(open_retrieval=True, work_order_open=True,
               **_slide_removal(latest_verdict=None))
    ) == "hybrid_slide_queued"


def test_absence_of_open_retrieval_keys_leaves_existing_router_behavior_unchanged():
    # Regression: the normal closed-set state (no open_retrieval/work_order_open
    # keys at all) must route byte-for-byte identically to today.
    assert select_screen(_state()) == "06"
    assert select_screen(_state(engaged=False, session_present=False)) == "01"
    assert select_screen(_state(phase="finalized")) == "21"
    assert select_screen(
        _state(phase="slides", capture_mode="slide", capture_state="EMPTY")
    ) == "12"


def test_normal_mode_state_still_gets_no_work_order_screen():
    # Negative control (must not be vacuous): a plain closed-set state --
    # mode="normal", no open_retrieval, no work_order_open -- must route
    # exactly as it did before #269. This is the regression bar for every
    # Hybrid change below.
    boot_state = _state(
        mode="normal", open_retrieval=False, work_order_open=False,
        has_work_orders=False, engaged=False, session_present=True,
        phase="blocks", captured=0,
    )
    assert select_screen(boot_state) == "01"
    assert select_screen(boot_state) != "first_work_order"

    idle_block_state = _state(mode="normal")
    assert select_screen(idle_block_state) == "06"
    assert select_screen(idle_block_state) != "block_scan_work_order"

    idle_slide_state = _state(
        mode="normal", phase="slides", capture_mode="slide",
        capture_state="EMPTY",
    )
    assert select_screen(idle_slide_state) == "12"
    assert select_screen(idle_slide_state) != "slide_capture_work_order"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_without_a_work_order_shows_first_work_order(hybrid_mode):
    # #269: today this state falls through to the ordinary boot chooser (01)
    # since open_retrieval is deliberately False for Hybrid. Reusing Open
    # Retrieval's screens means the same first-run work-order screen must
    # appear, gated on the `mode` string relay.py publishes.
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=False,
               engaged=False, has_work_orders=False, session_present=True,
               phase="blocks", captured=0)
    ) == "first_work_order"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_between_orders_after_a_closed_bracket(hybrid_mode):
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=False,
               engaged=False, has_work_orders=True, session_present=True,
               phase="blocks")
    ) == "between_orders"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_slide_capture_never_falls_back_to_preview_on_stale_closed_bracket(
    hybrid_mode,
):
    """An active capture instruction outranks a one-poll work-order mismatch."""
    common = dict(
        mode=hybrid_mode, open_retrieval=False, work_order_open=False,
        engaged=False, has_work_orders=True, session_present=True, phase="slides",
        capture_mode="slide",
    )
    assert select_screen(_state(
        **common, capture_state="CAPTURE_REQUESTED",
    )) == "13"
    assert select_screen(_state(
        **common, capture_state="WAITING_FOR_REMOVAL",
    )) == "hybrid_slide_queued"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_slide_settling_never_falls_back_to_start_on_stale_bracket(
    hybrid_mode,
):
    """A placed slide must keep Hold Still during a stale status poll."""
    assert select_screen(_state(
        mode=hybrid_mode, open_retrieval=False, work_order_open=False,
        engaged=False, has_work_orders=False, session_present=True, phase="slides",
        capture_mode="slide", capture_state="SETTLING",
    )) == "hold_still"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_between_orders_escapes_to_results_table(hybrid_mode):
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=False,
               engaged=False, has_work_orders=True, session_present=True,
               phase="blocks", captured=0,
               view_results_guard=True, results_ready_work_orders=(7,))
    ) == "results_table"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_with_work_order_open_shows_block_scan_work_order(hybrid_mode):
    # Before #269 this was PROCESSING/NORMAL's screen 06 -- an operator could
    # scan/capture blocks but never see the work-order-aware screen or its
    # bracket controls.
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               phase="blocks", capture_mode="block",
               capture_state="WAITING_FOR_SCAN")
    ) == "block_scan_work_order"
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               phase="blocks", capture_mode="block",
               capture_state="WAITING_FOR_SCAN")
    ) != "06"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_with_work_order_open_shows_slide_capture_work_order(hybrid_mode):
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               phase="slides", capture_mode="slide", capture_state="EMPTY")
    ) == "slide_capture_work_order"
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               phase="slides", capture_mode="slide", capture_state="EMPTY")
    ) != "12"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_mode_open_bracket_reaches_calibrating_while_disengaged(hybrid_mode):
    # Mirrors the #159 Open Retrieval regression: an open Hybrid bracket must
    # also count as engaged even when the client latch is off, so a restart
    # mid-session routes to Calibrating (05) instead of hanging on the boot
    # chooser.
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               engaged=False, phase="blocks", capture_mode="block",
               capture_state="AWAITING_BASELINE_CONFIRMATION")
    ) == "05"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_effective_engaged_true_for_hybrid_open_bracket(hybrid_mode):
    from kiosk.router import effective_engaged

    assert effective_engaged(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               engaged=False)
    ) is True
    assert effective_engaged(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=False,
               engaged=False)
    ) is False


# -- #257: Results reachable during an OPEN Hybrid slide-capture work order -


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_open_slide_capture_escapes_to_results_table(hybrid_mode):
    """#257 core criterion: Results must be reachable from an OPEN Hybrid
    slide-capture work order (idle "Place Slide" screen), not only between
    orders -- the router's results gate has no mode condition, so this
    requires no router change at all, only the state the entry point sets."""
    assert select_screen(
        _state(mode=hybrid_mode, open_retrieval=False, work_order_open=True,
               phase="slides", capture_mode="slide", capture_state="EMPTY",
               view_results_guard=True, results_ready_work_orders=(7,))
    ) == "results_table"


@pytest.mark.parametrize("hybrid_mode", ["hybrid", "hybrid_shadow"])
def test_hybrid_go_back_from_results_resumes_the_same_open_slide_capture(hybrid_mode):
    """#257: clearing the guard (Go Back) with everything else unchanged
    (same open bracket, same idle capture state) must re-derive the SAME
    open slide-capture screen -- proving Go Back resumes the work order
    rather than closing slide intake. Asserts the actual screen identity,
    not merely "not results_table"."""
    common = dict(
        mode=hybrid_mode, open_retrieval=False, work_order_open=True,
        phase="slides", capture_mode="slide", capture_state="EMPTY",
        results_ready_work_orders=(7,),
    )
    assert select_screen(_state(**common, view_results_guard=True)) == "results_table"
    assert select_screen(
        _state(**common, view_results_guard=False)
    ) == "slide_capture_work_order"


def test_select_screen_never_mutates_the_state_it_is_given():
    """#257 acceptance criterion: the screen router remains a pure
    projection with no new side effects. Proven generically (not just for
    the new feature): select_screen must never write into the mapping it
    reads, for any state shape covering the new Hybrid open-bracket path."""
    state = _state(
        mode="hybrid", open_retrieval=False, work_order_open=True,
        phase="slides", capture_mode="slide", capture_state="EMPTY",
        view_results_guard=True, results_ready_work_orders=(7,),
    )
    before = dict(state)

    select_screen(state)

    assert state == before


def test_open_retrieval_results_route_unaffected_by_pause_capture_addition():
    """#257 acceptance criterion, non-vacuous control: Open Retrieval's
    existing between_orders -> results_table -> back route (the ONLY
    pre-#257 entry point) must still route exactly as before. Flips the
    guard both ways so a broken OR path fails this test, not just a missing
    one."""
    ready = _state(
        open_retrieval=True, work_order_open=False, engaged=False,
        has_work_orders=True, session_present=True, phase="blocks", captured=0,
        view_results_guard=True, results_ready_work_orders=(7,),
    )
    assert select_screen(ready) == "results_table"
    assert select_screen(dict(ready, view_results_guard=False)) == "between_orders"


def test_results_table_screen_renders_synthetic_rows_without_a_workflow_or_camera():
    """The results-table screen is driven purely by state + the pure
    projection helper -- no images, camera, or server round-trip needed."""
    from kiosk.results_table import project_results_table

    synthetic_rows = [
        {"capture_id": "cap-1", "block_id": "b1", "verdict": "PASS",
         "claim_reason": "", "claim_score": 0.91},
        {"capture_id": "cap-2", "block_id": "b2", "verdict": "REVIEW",
         "claim_reason": "ambiguous near-miss", "claim_score": 0.40},
    ]
    state = _state(
        view_results_guard=True,
        results_ready_work_orders=(7,),
        results_rows=synthetic_rows,
    )

    assert select_screen(state) == "results_table"
    rendered = project_results_table(synthetic_rows)
    assert [row["capture_id"] for row in rendered] == ["cap-2", "cap-1"]
    assert rendered[0]["color"] == "red"
    assert rendered[1]["color"] == "green"
