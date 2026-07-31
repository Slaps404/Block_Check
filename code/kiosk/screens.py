"""The screen catalog -- single source of truth for the generic renderer (119).

Pure static data. Each entry describes how one screen is painted; the client
picks the active entry with ``/state.screen`` (= :func:`kiosk.router.select_screen`
output) and overlays entry ``"18"`` on top of ``"06"`` when a duplicate scan
event arrives. There is NO domain logic here and nothing is imported from the
session layer: the catalog is state-independent, safe to serve once at boot.

Keys of ``CATALOG["screens"]`` are EXACTLY the ids ``select_screen`` can return
plus the client-overlay id ``"18"``. Button ``action`` values are drawn only
from the frozen vocabulary (``dispatch``/``guard``/``back``/``engage``/
``disengage``); ``verb`` appears only on ``dispatch`` and ``target`` only on
``guard``/``back``.

Copy + render fields are ported from the DesignSync kiosk wireframe (lab
design artifacts moved out of-repo in #206). Headlines are authored title-case
and the renderer upper-cases them via ``text-transform``. Two render fields
beyond the 119a shape carry design-faithful layout signals the renderer reads:

* ``status_bar`` (bool) -- whether the capture status line shows on this screen.
  The renderer builds the text ("BLOCK CAPTURE ..." / "SLIDE CAPTURE ...") from
  live ``capture_mode`` + counts, so one flag drives both phases.
* ``sub2`` (str, optional) -- a second, smaller instruction line (only screen
  16 REVIEW, which shows the verdict reason AND "Remove Slide").
"""
from __future__ import annotations

import json
from typing import Any, Final

BUTTON_ACTIONS: Final = ("dispatch", "guard", "back", "engage", "disengage")

CATALOG: Final[dict[str, Any]] = {
    "screens": {
        "01": {
            "headline": "LJI Block Check",
            "sub": "Preview",
            "variant": "idle",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "START SESSION", "action": "engage"},
            ],
            "special": (
                "Boot chooser (router R1/R2/R3-empty). Design 01 shows NO "
                "headline: a large START SESSION button in the top third over a "
                "contained camera-preview placeholder captioned 'Preview' "
                "(the `sub`). Also shown when session_present but captured==0 "
                "(empty continue into the launcher-attached N — K11). "
                "START SESSION flips the client-owned engaged latch to true; it "
                "does NOT dispatch a backend verb (session creation is 119c)."
            ),
        },
        "first_work_order": {
            "headline": "LJI Block Check",
            "sub": "Preview",
            "variant": "idle",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "START SESSION", "action": "dispatch",
                 "verb": "start_work_order"},
            ],
            "special": (
                "Open Retrieval first-run screen. It deliberately matches "
                "screen 01's full-size preview and single START SESSION "
                "button, but opens the first durable work-order bracket."
            ),
        },
        "02": {
            "headline": "Unfinished Session Found",
            "sub": "",
            "variant": "chooser",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "RESUME SESSION {session_number}", "action": "engage"},
                {"label": "START NEW SESSION", "action": "engage"},
            ],
            "special": (
                "Boot chooser (router R3): !engaged && session_present && "
                "phase != finalized && captured > 0. Empty attached sessions "
                "(captured==0) use screen 01 instead (K11). Design 02: 92px "
                "nowrap headline with a large primary 'RESUME SESSION "
                "{session_number}' button centred under it and a smaller "
                "'START NEW SESSION' button pinned bottom. Both buttons flip "
                "the engaged latch (real new-session rebind still 119c). START "
                "NEW SESSION stays a full-opacity no-op guard (no new verb)."
            ),
        },
        "04": {
            "headline": "Clear Backlight for Calibration",
            "sub": "",
            "variant": "error",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {
                    "label": "RETRY",
                    "action": "dispatch",
                    "verb": "confirm_empty",
                    "size": "lg",
                },
            ],
            "special": (
                "Router: capture_state == CALIBRATION_FAILED. Calibration Failure "
                "recovery after Empty-Backlight Setup fails (engage or slide start). "
                "RETRY is the large primary footer button (size lg, same as screen "
                "19) and dispatches confirm_empty (full setup again). Client paints "
                "screen 05 immediately on tap. Not Capture Error (screen 19). "
                "Fixed copy — no reason-specific sub."
            ),
        },
        "05": {
            "headline": "Calibrating…",
            "sub": "",
            "variant": "processing",
            "bg": "default",
            "progress_bar": True,
            "status_bar": False,
            "buttons": [],
            "special": (
                "Router: capture_state in {AWAITING_BASELINE_CONFIRMATION, "
                "BUILDING_BASELINE}. Shown at session start (after START "
                "SESSION) and again when slides begin. Client auto-fires "
                "confirm_empty; no CALIBRATE NOW button. Auto-advances to "
                "Scan Block (block) or Place Slide (slide) when the baseline "
                "is ready. PROGRESS BAR = rectangular with a SMALL bevel."
            ),
        },
        "06": {
            "headline": "Scan Block",
            "sub": "{hybrid_pool_bounce_message}",
            "variant": "capture",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "FINISH BLOCKS", "action": "guard",
                 "target": "finish_blocks_guard"},
            ],
            "special": (
                "Router R21: capture_mode == block && capture_state == "
                "WAITING_FOR_SCAN (also the degraded fallback for phase == "
                "blocks). Status bar reads: 'BLOCK CAPTURE · WO: "
                "{work_order|—} · {captured} CAPTURED'. Barcode "
                "scanner self-dispatches the unified scan_qr front door via the "
                "keydown->chars->Enter path (runtime routes: scan_block in block "
                "mode, slide-payload stash in slide mode); the scanner is "
                "NOT a catalog button. FINISH BLOCKS (design's large action-"
                "button size, lg) is client nav that SETS finish_blocks_guard "
                "(routes to screen 10), not a dispatch. Screen 18 duplicate "
                "overlay layers on top of this screen. #250: sub renders "
                "KioskRelay.state()'s `hybrid_pool_bounce_message` (empty "
                "string/falls back to blank whenever the most recent event "
                "is not a Hybrid Finish-Blocks <2-usable-blocks bounce) so "
                "the operator sees WHY they landed back on Scan Block "
                "instead of freezing, rather than a bare unexplained bounce."
            ),
        },
        "07": {
            "headline": "Place Block",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "place",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R19: capture_mode == block && capture_state == EMPTY. "
                "Auto-capture, no button. BLOCK ID sub (38px) reads the scanned "
                "id (pending_block_id / block_scanned event) - note 119a gotcha: "
                "snapshot.latest_block_id fills at capture time not scan time. "
                "SETTLING is hold_still (ADR 0005), not this screen."
            ),
        },
        "hold_still": {
            "headline": "Hold Still",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "place",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router ADR 0005: capture_state == SETTLING (block or slide). "
                "Shared centred Place-style layout (variant place) -- NOT the "
                "session processing screen. No progress bar, no buttons. BLOCK "
                "ID sub only in block mode (client clears sub in slide mode)."
            ),
        },
        "capture_review": {
            "headline": "Review Capture",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "still",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "RETAKE", "action": "dispatch",
                 "verb": "retry_capture"},
                {"label": "ACCEPT", "action": "dispatch",
                 "verb": "accept_capture"},
            ],
            "special": (
                "Router ADR 0006: capture_state == AWAITING_ACCEPT (block or "
                "slide). Shows the held still fit-to-screen via GET "
                "/review-still (cache-busted by pending_capture_id). RETAKE "
                "dispatches retry_capture (discard + immediate re-shoot); "
                "ACCEPT dispatches accept_capture (deferred commit). BLOCK ID "
                "sub only in block mode (client clears sub in slide mode)."
            ),
        },
        "08": {
            "headline": "Capturing…",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "capture",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R17: capture_mode == block && capture_state == "
                "CAPTURE_REQUESTED. In-flight shot, auto-advances to 09 "
                "(removal) or 19 (error). No button. BLOCK ID sub persists "
                "from scan through remove (ADR 0005 grill; diverges from "
                "design_spec empty-sub on 08)."
            ),
        },
        "09": {
            "headline": "Remove Block",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "reposition",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R16: capture_mode == block && capture_state == "
                "WAITING_FOR_REMOVAL. Auto (removal detection), no button. "
                "BLOCK ID sub persists until Scan Block returns (ADR 0005)."
            ),
        },
        "10": {
            "headline": "Finish Block Capture?",
            "sub": "",
            "variant": "confirm",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "GO BACK", "action": "back", "target": "finish_blocks_guard"},
                {"label": "FINISH BLOCKS", "action": "dispatch", "verb": "finish_blocks"},
            ],
            "special": (
                "Router R6: finish_blocks_guard && phase == blocks && "
                "capture_state == WAITING_FOR_SCAN. Design 10 has NO sub and no "
                "counts -- it is only an accidental-tap guard. GO BACK clears "
                "finish_blocks_guard (falls back to 06). FINISH BLOCKS "
                "dispatches finish_blocks (phase -> draining_blocks, R5 "
                "processing screen takes over). Guard is gated on idle "
                "capture_state so a live scan/capture drops the modal "
                "automatically."
            ),
        },
        "12": {
            "headline": "Place Slide",
            "sub": "{pending_slide_label}",
            "variant": "place",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "FINISH SLIDES", "action": "guard", "target": "finish_slides_guard"},
            ],
            "special": (
                "Router R20: capture_mode == slide && capture_state == EMPTY "
                "(also degraded fallback for phase == slides). SETTLING is "
                "hold_still (ADR 0005). Status bar reads 'SLIDE CAPTURE · WO · "
                "{n} remaining'. FINISH SLIDES is client nav that SETS "
                "finish_slides_guard (routes to screen 20), not a dispatch. "
                "Sub shows the scanned id via pending_slide_label (\"SCANNED: "
                "<id> · place slide\") once a slide is scanned; blank until "
                "then."
            ),
        },
        "13": {
            "headline": "Capturing…",
            "sub": "",
            "variant": "capture",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R18: capture_mode == slide && capture_state == "
                "CAPTURE_REQUESTED. Automatic; no 'Hold still' sub and no "
                "button (design 13). Advances to 14 (verify) or 19 (error)."
            ),
        },
        "14": {
            "headline": "Verifying…",
            "sub": "Remove Slide",
            "variant": "verify",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R15: capture_mode == slide && capture_state == "
                "WAITING_FOR_REMOVAL with NO id-matched verdict yet. Design 14 "
                "shows a 'Remove Slide' sub -- the operator may remove the "
                "slide while the verdict is pending. Auto; becomes 15/16 once "
                "latest_verdict.capture_id == current_capture_id."
            ),
        },
        "15": {
            "headline": "PASS",
            "sub": "Remove Slide",
            "variant": "pass",
            "bg": "pass-full",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R13: slide && WAITING_FOR_REMOVAL && "
                "latest_verdict.kind == claim_pass && "
                "latest_verdict.capture_id == current_capture_id. FULL-SCREEN "
                "GREEN (#20b15a) with white text and a 'Remove Slide' sub. "
                "Status bar text is rgba(255,255,255,0.82). Auto, awaiting "
                "removal, no button. Id-match prevents a stale prior-slide PASS "
                "bleeding onto the next slide."
            ),
        },
        "16": {
            "headline": "REVIEW",
            "sub": "{latest_verdict.reason}",
            "sub2": "Remove Slide",
            "variant": "review",
            "bg": "review-full",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "Router R14: slide && WAITING_FOR_REMOVAL && "
                "latest_verdict.kind == claim_review && id-match. FULL-SCREEN "
                "RED (#c93535) with white text. Two lines: the verdict reason "
                "(`sub`, 58px) and a smaller 'Remove Slide' line (`sub2`, "
                "44px). Auto, awaiting removal, no button."
            ),
        },
        "17": {
            "headline": "Reposition Slide",
            "sub": "Code could not be read",
            "variant": "reposition",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "RETAKE SLIDE", "action": "dispatch",
                 "verb": "retry_capture"},
                {"label": "SKIP SLIDE", "action": "dispatch", "verb": "skip_slide"},
            ],
            "special": (
                "Router R10: capture_state == REPOSITION_SLIDE. Design 17: "
                "WHITE bg (not red); the headline is the ACTION 'Reposition "
                "Slide' and the sub is the reason 'Code could not be read'. "
                "RETAKE SLIDE dispatches retry_capture for an immediate still "
                "(no settle/motion wait). SKIP SLIDE dispatches skip_slide "
                "(wraps skip_unreadable_slide)."
            ),
        },
        "18": {
            "headline": "Block Already Scanned",
            "sub": "BLOCK ID: {latest_block_id}",
            "variant": "duplicate-flash",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [],
            "special": (
                "CLIENT-SIDE OVERLAY, NOT a router-returned screen "
                "(select_screen never returns 18). Triggers: "
                "/state.last_event.kind == 'duplicate_block_scan' on routed "
                "screen 06, OR 'duplicate_slide_scan' on routed screen 12 "
                "(scan-time slide duplicate guard; headline swaps to 'Slide "
                "Already Scanned'). Design 18: WHITE bg (not red), dark "
                "'Block Already Scanned' headline, a 'BLOCK ID' sub, and a "
                "dimmed (#999) status bar. Shown as a ~2s timed flash overlay "
                "ON TOP OF the routed screen, then auto-dismiss and reveal it "
                "again. Key the timer on the event (client wall-clock ~2s TTL) "
                "so a re-delivered/out-of-order duplicate event neither "
                "re-triggers a finished flash nor latches it open. No button."
            ),
        },
        "19": {
            "headline": "Capture Failed",
            "sub": "",
            "variant": "error",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "RETRY", "action": "dispatch",
                 "verb": "retry_capture"},
            ],
            "special": (
                "Router R9: capture_state == CAPTURE_ERROR. Design 19: WHITE bg "
                "(NOT red) and a 'Capture Failed' headline. Per Zeke's kiosk "
                "review the reason sub is dropped (sub empty) so ONLY the "
                "headline shows. RETRY is the design's large primary action "
                "button (size lg) and dispatches retry_capture (-> "
                "CAPTURE_REQUESTED -> back to 08/13)."
            ),
        },
        "20": {
            "headline": "Finish Slide Capture?",
            "sub": "",
            "variant": "confirm",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "GO BACK", "action": "back", "target": "finish_slides_guard"},
                {"label": "FINISH SLIDES", "action": "dispatch", "verb": "end_session"},
            ],
            "special": (
                "Router R7: finish_slides_guard && phase == slides && "
                "capture_state == EMPTY. Design 20 has NO sub and no counts -- "
                "the summary appears only after finishing. GO BACK clears "
                "finish_slides_guard (falls back to 12). FINISH SLIDES is a "
                "CONFIRM that dispatches end_session (there is no phantom "
                "finish_slides verb; the reconciled contract maps the "
                "slide-phase finish onto end_session). end_session drives phase "
                "-> finalizing/finalized, then R4 shows screen 21."
            ),
        },
        "21": {
            "headline": "Session Complete",
            "sub": "",
            "variant": "counts",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "END SESSION", "action": "disengage"},
            ],
            "special": (
                "Router R4: phase == finalized (only while engaged). COUNTS "
                "GRID (design 21): a 2-col left-aligned grid PASS / REVIEW / "
                "TOTAL (= pass_count / review_count / captured, from /state), "
                "with a 'Debug details collapsed' caption below. END SESSION "
                "here DISENGAGES the boot latch (engaged -> false), returning "
                "to the boot chooser (01) - it does NOT dispatch end_session "
                "again (that already ran at screen 20's confirm)."
            ),
        },
        "results_table": {
            "headline": "Results",
            "sub": "",
            "variant": "table",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "GO BACK", "action": "back",
                 "target": "view_results_guard"},
            ],
            "special": (
                "Router (#150, ADR 0009 follow-on): view_results_guard && a "
                "non-empty results_ready_work_orders (client-owned guard "
                "gated on a data condition, mirrors finish_blocks_guard/"
                "finish_slides_guard). No wireframe number exists for this "
                "list/summary shape (design_spec has none), so it takes the "
                "same non-numeric treatment as hold_still/capture_review/"
                "processing. Renders one row per slide from "
                "kiosk.results_table.project_results_table(results_rows): "
                "REVIEW rows sorted to the top, each row colour-coded green "
                "(PASS) / red (REVIEW) with an expand_target (capture_id) "
                "wired to the per-slide inspection route in a later slice. "
                "Rows aggregate every results_ready work order in the "
                "session, not a single-order picker, so any order finished "
                "this session is viewable without re-scanning."
            ),
        },
        "between_orders": {
            "headline": "Between Work Orders",
            "sub": "Preview",
            "variant": "preview_actions",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "START NEW WORK ORDER", "action": "dispatch",
                 "verb": "start_work_order"},
                {"label": "VIEW RESULTS", "action": "guard",
                 "target": "view_results_guard"},
                {"label": "END SESSION", "action": "dispatch",
                 "verb": "end_session"},
            ],
            "special": (
                "Router (#155, ADR 0016): open_retrieval && !work_order_open, "
                "regardless of engaged/phase -- the Open Retrieval work-order "
                "gate takes precedence over the ordinary boot chooser. START "
                "NEW WORK ORDER dispatches start_work_order (opens a new "
                "capture bracket; idempotent on double-tap). This screen "
                "appears only after at least one durable work order exists "
                "and keeps the full-size live preview with two small actions "
                "at the bottom. END SESSION is available only after the "
                "current work order closes, so it cannot abandon an active "
                "block/slide bracket."
            ),
        },
        "block_scan_work_order": {
            "headline": "Scan Block",
            "sub": "",
            "variant": "capture",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "FINISH BLOCKS", "action": "guard",
                 "target": "finish_blocks_guard"},
            ],
            "special": (
                "Router (#155, ADR 0016): open_retrieval && work_order_open && "
                "capture_mode == block && capture_state == WAITING_FOR_SCAN -- "
                "the screen 06 sibling for an open work-order bracket. FINISH "
                "BLOCKS is the unchanged guard-nav button (same target/routing "
                "as screen 06's, -> screen 10). The work-order bracket remains "
                "open through slide capture."
            ),
        },
        "slide_capture_work_order": {
            "headline": "Place Slide",
            "sub": "{pending_slide_label}",
            "variant": "place",
            "bg": "default",
            "progress_bar": False,
            "status_bar": True,
            "buttons": [
                {"label": "FINISH WORK ORDER", "action": "dispatch",
                 "verb": "finish_work_order"},
                {"label": "VIEW RESULTS", "action": "dispatch",
                 "verb": "pause_capture"},
            ],
            "special": (
                "Router (#155, ADR 0016): open_retrieval && work_order_open && "
                "capture_mode == slide && capture_state == EMPTY. FINISH WORK "
                "ORDER closes the bracket only after its block and slide "
                "captures, dispatches async N^2 scoring, and returns to the "
                "between-orders preview. #257: VIEW RESULTS is the entry point "
                "for inspecting Results DURING an open slide-capture work "
                "order (Hybrid/Hybrid Shadow, and -- as a side effect of this "
                "screen being shared per #269 -- Open Retrieval too). Unlike "
                "every other guard-nav button, this one is a real `dispatch` "
                "(not `guard`): pausing automatic capture is a side effect, "
                "and #257 requires the router to stay a pure projection, so "
                "the pause itself must live in the verb layer. The client "
                "special-cases the `pause_capture` verb to ALSO set "
                "view_results_guard=true in the same tap (mirrors the "
                "existing finish_blocks guard-clearing special case), and "
                "GO BACK from the results table dispatches resume_capture "
                "when this screen was the entry point -- never for the "
                "unrelated between_orders (Open Retrieval closed-bracket) "
                "VIEW RESULTS/GO BACK pair, which stays pure guard/back with "
                "no dispatch at all."
            ),
        },
        "hybrid_attention": {
            "headline": "Slide Needs Attention",
            "sub": "{attention.message}",
            "variant": "confirm",
            "bg": "default",
            "progress_bar": False,
            "status_bar": False,
            "buttons": [
                {"label": "GO BACK", "action": "back", "target": "recapture_guard"},
                {"label": "RETRY", "action": "dispatch", "verb": "retry_hybrid_slide"},
                {"label": "RECAPTURE", "action": "dispatch",
                 "verb": "arm_hybrid_recapture"},
            ],
            "special": (
                "#256: the EXPLICIT correction-flow destination reached only "
                "by tapping the passive attention banner's own nav button, "
                "which sets recapture_guard (guard+data pattern, mirrors "
                "finish_blocks_guard/view_results_guard -- shown only when "
                "recapture_guard is set AND state.attention is non-null). "
                "The passive amber banner itself is a THIRD layer alongside "
                "the WS-A offline banner (design_spec §22): it renders "
                "whenever state.attention is present, layered over whatever "
                "screen is already active, and never tears the routed "
                "screen down -- explicitly NOT the screen-18 duplicate-scan "
                "overlay (a full-screen takeover), which is the wrong shape "
                "for a late background-job outcome. RETRY dispatches "
                "retry_hybrid_slide against attention.capture_id: a system "
                "ERROR retries from the durable capture before recapture is "
                "offered. When attention.can_recapture is false, another "
                "work order is actively capturing and the client should "
                "hold off offering further recapture action until an "
                "available transition -- GO BACK always remains available. "
                "RECAPTURE (follow-up to #256) dispatches "
                "arm_hybrid_recapture against attention.capture_id -- the "
                "photo-driven trigger itself still belongs to the live "
                "capture runtime (`SessionWorkflow.capture_slide`), but this "
                "is what tells it the NEXT physically captured slide "
                "supersedes THIS attention item instead of starting an "
                "ordinary claim. The operator taps RECAPTURE, then GO BACK "
                "to return to live capture, then places the new slide."
            ),
        },
        "processing": {
            "headline": "Processing…",
            "sub": "",
            "variant": "processing",
            "bg": "default",
            "progress_bar": True,
            "status_bar": False,
            "buttons": [],
            "special": (
                "Router R5: phase in {draining_blocks, finalizing, "
                "cleanup_pending}. The generic transitional/hold-frame screen "
                "(wireframe 11 was excluded as a DO-NOT-USE testbed; id string "
                "is literally 'processing'). Background drivers "
                "poll_drain/poll_finalization advance the phase; no button. "
                "PROGRESS BAR = rectangular with a SMALL bevel (same restyled "
                "bar as screen 05, per Zeke - both bars rectangular like the "
                "buttons)."
            ),
        },
    },
}


def catalog_json() -> str:
    """The catalog as a compact JSON string -- what ``GET /catalog`` serves."""
    return json.dumps(CATALOG)
