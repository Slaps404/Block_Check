"""Event-driven screen->state router for the kiosk (#122 / 119b, ADR 0004).

`select_screen(state)` maps one state snapshot to exactly one screen id. It is a
pure function -- no I/O, no deck navigation -- so the active screen is a
deterministic consequence of workflow signals plus the client's own UI flags.
Rules are ordered; the FIRST match wins, and the ordering encodes the precedence
decided for kiosk 119b (lifecycle before capture; guards gated on an idle
capture state; id-matched verdict before the generic removal screen).

State keys consumed (all optional; absent capture_* => degraded phase-only mode):
  engaged, session_present, phase, capture_state, capture_mode,
  latest_verdict {kind, capture_id}, current_capture_id,
  finish_blocks_guard, finish_slides_guard, captured (sets_processed),
  blocks_captured (any block captured this session, regardless of verdict),
  view_results_guard, results_ready_work_orders, recapture_guard, attention
  (#256's non-interrupting Hybrid attention-banner projection, or None).
"""
from __future__ import annotations

from typing import Mapping

# The transitional screen has no wireframe number (screen 11, the intended
# "Preparing Slides", was excluded as a DO-NOT-USE testbed). This id names the
# generic "Processing..." screen shown during draining_blocks / finalizing.
PROCESSING = "processing"

# Shared SETTLING screen (ADR 0005): specimen detected, waiting for stillness.
# Not the session-level PROCESSING screen.
HOLD_STILL = "hold_still"

_TRANSITIONAL_PHASES = {"draining_blocks", "finalizing", "cleanup_pending"}


_WORK_ORDER_BRACKET_MODES = ("hybrid", "hybrid_shadow")


def _has_work_order_bracket(state: Mapping) -> bool:
    """True when this session uses a real work-order bracket: Open Retrieval,
    or Hybrid/Hybrid Shadow reusing Open Retrieval's screens (#269, ADR gives
    Hybrid the same start_work_order/finish_work_order lifecycle). Reads the
    `mode` string relay.py publishes (KioskRelay._mode_value), never an
    inferred/artifact-based signal. `.get` throughout: this runs on every
    /state poll, so a state dict missing either key must degrade, not raise."""
    return bool(state.get("open_retrieval", False)) or (
        state.get("mode") in _WORK_ORDER_BRACKET_MODES
    )


def effective_engaged(state: Mapping) -> bool:
    """The authoritative 'engaged' value: the client's boot latch OR (when a
    work-order bracket applies -- open-retrieval, hybrid, hybrid_shadow) an
    open work-order bracket. ADR-0016 (open-retrieval) #2: work_order_open is
    the engaged gate in that mode. Pure function of state so both the router
    and the relay derive the SAME value from one place."""
    return bool(state.get("engaged", False)) or (
        _has_work_order_bracket(state)
        and bool(state.get("work_order_open", False))
    )


def select_screen(state: Mapping) -> str:
    engaged = effective_engaged(state)
    phase = state.get("phase")
    cstate = state.get("capture_state")
    cmode = state.get("capture_mode")

    # The Pi capture state is the immediate operator-safety instruction. A
    # delayed work-order status poll must never replace an active retrieval slide
    # capture or removal prompt with the between-orders preview. Retrieval modes
    # also defer PASS/REVIEW presentation to the Results table.
    if (
        _has_work_order_bracket(state)
        and phase == "slides"
        and cmode == "slide"
    ):
        if cstate == "SETTLING":
            return HOLD_STILL
        if cstate == "CAPTURE_REQUESTED":
            return "13"
        if cstate == "WAITING_FOR_REMOVAL":
            return "hybrid_slide_queued"

    # --- Open Retrieval (#155): the work-order bracket gates the boot latch.
    # The first work order uses the original startup-preview layout. After any
    # durable work-order row exists, a closed bracket uses between_orders.
    # Once one is open, the closed-set boot latch is treated as satisfied.
    if _has_work_order_bracket(state):
        if not state.get("work_order_open", False):
            if not state.get("has_work_orders", False):
                return "first_work_order"
            # ADR 0016 decision #2: between-orders must be able to reach the
            # already-built results_table. The ordinary results gate sits below
            # this early return, so re-check it here (same guard+data condition)
            # before falling through to the chooser -- otherwise the N^2 results
            # are unreachable once every bracket is closed.
            if (state.get("view_results_guard")
                    and state.get("results_ready_work_orders")):
                return "results_table"
            return "between_orders"
        # work_order_open == True: fall through. `engaged` is already True via
        # effective_engaged() above, so every capture-state branch runs as before.

    # --- boot chooser: the operator has not engaged a session yet ---------
    if not engaged:
        if not state.get("session_present"):
            return "01"  # Startup Preview
        if phase == "finalized":
            return "01"  # nothing to resume -> start fresh
        # Empty attached session (launcher just created N, no sets yet): treat
        # as Start, not Resume. captured == sets_processed (verdicted-only) --
        # #188: that alone misreads "blocks captured, none scored yet" as no
        # session, so also check blocks_captured (any block ever captured,
        # regardless of verdict status).
        try:
            captured = int(state.get("captured") or 0)
        except (TypeError, ValueError):
            captured = 0
        try:
            blocks_captured = int(state.get("blocks_captured") or 0)
        except (TypeError, ValueError):
            blocks_captured = 0
        if captured == 0 and blocks_captured == 0:
            return "01"  # Start Session (empty continue into this N)
        return "02"  # Unfinished Session (real work to resume)

    # --- lifecycle / transitional gates (must precede capture screens) ----
    if phase == "finalized":
        return "21"  # Session Complete
    if phase in _TRANSITIONAL_PHASES:
        return PROCESSING  # generic progress screen (screen 11 was excluded)

    # --- results table (#150/#252): client-owned guard gated on results data
    # Mirrors finish_blocks_guard/finish_slides_guard's guard+data-condition
    # pattern, but the guard flag alone is a no-op -- there must be at least
    # one work order with rows for the table to be shown. NORMAL remains
    # `results_ready`-gated. Retrieval modes source this key from every live
    # Retrieval Slide Job, scored or not, so the same key admits the table
    # when a work order has any row. No
    # separate gate needed here, only `SessionWorkflow.results_status`'s
    # per-mode row source (see its docstring).
    if state.get("view_results_guard") and state.get("results_ready_work_orders"):
        return "results_table"

    # --- #256 attention correction flow: guard+data pattern, mirrors
    # results_table/finish_blocks_guard immediately above/below. Reached
    # only by an explicit operator tap (setting recapture_guard) on the
    # banner's own nav button -- the passive banner itself is a
    # client-rendered overlay layer, never a router-driven screen swap, so
    # it never automatically takes over whatever screen is active.
    if state.get("recapture_guard") and state.get("attention"):
        return "hybrid_attention"

    # --- operator confirm modals, honored only over an idle capture -------
    if state.get("finish_blocks_guard") and phase == "blocks" and cstate == "WAITING_FOR_SCAN":
        return "10"  # Finish Blocks Confirm
    if state.get("finish_slides_guard") and phase == "slides" and cstate == "EMPTY":
        return "20"  # Finish Slides Confirm

    # --- transients / errors override whatever idle screen they sit on ----
    if cstate == "CAPTURE_ERROR":
        return "19"  # Capture Error (RETRY)
    if cstate == "CALIBRATION_FAILED":
        return "04"  # Clear Backlight for Calibration (RETRY)
    if cstate == "REPOSITION_SLIDE":
        return "17"  # Unreadable Code

    # --- calibration ------------------------------------------------------
    # Both AWAITING (waiting to start) and BUILDING show Calibrating…; the
    # client auto-fires confirm_empty on engage / when AWAITING appears while
    # already engaged (slide-phase rebuild). CALIBRATION_FAILED → screen 04.
    if cstate in ("AWAITING_BASELINE_CONFIRMATION", "BUILDING_BASELINE"):
        return "05"  # Calibrating…

    # --- capture review gate (before removal / verdict screens) ------------
    if cstate == "AWAITING_ACCEPT":
        return "capture_review"

    # --- slide verdict split: 14/15/16 all share WAITING_FOR_REMOVAL ------
    if cmode == "slide" and cstate == "WAITING_FOR_REMOVAL":
        verdict = state.get("latest_verdict")
        if verdict and verdict.get("capture_id") == state.get("current_capture_id"):
            if verdict.get("kind") == "claim_pass":
                return "15"  # PASS
            if verdict.get("kind") == "claim_review":
                return "16"  # REVIEW
        return "14"  # Verifying Slide (no id-matched verdict yet)

    # --- removal & capture-in-flight: mode picks the variant --------------
    if cmode == "block" and cstate == "WAITING_FOR_REMOVAL":
        return "09"  # Remove Block
    if cstate == "CAPTURE_REQUESTED":
        return "13" if cmode == "slide" else "08"  # Capturing Slide / Block

    # --- settle (specimen present, waiting for stillness) -----------------
    # ADR 0005: SETTLING is Hold Still, not Place and not session Processing.
    if cstate == "SETTLING":
        return HOLD_STILL

    # --- idle resting screens ---------------------------------------------
    if cmode == "block" and cstate == "EMPTY":
        return "07"  # Place Block
    if cmode == "slide" and cstate == "EMPTY":
        if (_has_work_order_bracket(state)
                and state.get("work_order_open", False)):
            return "slide_capture_work_order"
        return "12"  # Place Slide
    if cmode == "block" and cstate == "WAITING_FOR_SCAN":
        if _has_work_order_bracket(state):
            return "block_scan_work_order"
        return "06"  # Scan Block

    # --- degraded: no capture session attached (bare workflow / pre-attach)
    if phase == "blocks":
        return "06"
    if phase == "slides":
        return "12"
    return "01"
