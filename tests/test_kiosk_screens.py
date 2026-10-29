"""The screen catalog is the single source of truth for the generic renderer.

These tests pin the catalog to the frozen 119 contract: it must cover EXACTLY
the ids ``select_screen`` can emit (plus the client-overlay id ``18``), each
entry must carry the render fields the frontend reads, and every button action
must be drawn only from the frozen five-verb vocabulary. The catalog is pure
static data, so it must be JSON-serializable with no per-request state.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from kiosk import router
from kiosk.screens import CATALOG, catalog_json

# The router-returned ids, plus the client-only overlay id 18 (select_screen
# never returns 18 -- it is layered on 06 by the client).
_ROUTER_IDS = {
    "led_calibration_place", "led_calibration_remove", "led_calibration_failed",
    "01", "02", "04", "05", "06", "07", "08", "09", "10", "12",
    "13", "14", "15", "16", "17", "19", "20", "21", "processing", "hold_still",
    "capture_review", "results_table", "first_work_order", "between_orders",
    "block_scan_work_order", "slide_capture_work_order", "hybrid_attention",
    "hybrid_slide_queued",
}
_RETIRED_IDS: set[str] = set()
_EXPECTED_IDS = _ROUTER_IDS | _RETIRED_IDS | {"18"}

_ACTION_VOCAB = {"dispatch", "guard", "back", "engage", "disengage"}


def test_catalog_covers_exactly_the_router_ids_plus_overlay():
    assert set(CATALOG["screens"].keys()) == _EXPECTED_IDS


def test_router_emits_no_id_outside_the_catalog():
    """Guard against router drift: every literal id returned by select_screen
    must have a catalog entry (18 is the one entry the router never emits)."""
    source = inspect.getsource(router.select_screen)
    # Every quoted string literal the function returns (ternary arms included),
    # plus PROCESSING which is returned by name (its value is "processing").
    quoted = set(re.findall(r'return\b[^\n#]*', source))
    returned = {tok for line in quoted for tok in re.findall(r'"([^"]+)"', line)}
    if "return PROCESSING" in source:
        returned.add(router.PROCESSING)
    if "return HOLD_STILL" in source:
        returned.add(router.HOLD_STILL)
    # Drop capture-mode literals from ternary conditions; keep unknown ids.
    literal_ids = returned - {"slide", "block"}
    # Every screen id literal the router returns is catalogued.
    assert literal_ids <= set(CATALOG["screens"].keys())
    # And the router genuinely reaches the full live router-id set.
    assert literal_ids == _ROUTER_IDS
    assert literal_ids.isdisjoint(_RETIRED_IDS)


def test_every_entry_has_the_required_render_fields():
    for screen_id, entry in CATALOG["screens"].items():
        assert isinstance(entry["headline"], str) and entry["headline"], screen_id
        assert isinstance(entry["sub"], str), screen_id
        assert isinstance(entry["variant"], str) and entry["variant"], screen_id
        assert entry["bg"] in ("default", "pass-full", "review-full", "error"), screen_id
        assert isinstance(entry["progress_bar"], bool), screen_id
        assert isinstance(entry["buttons"], list), screen_id


def test_button_actions_are_only_from_the_frozen_vocab():
    for screen_id, entry in CATALOG["screens"].items():
        for button in entry["buttons"]:
            assert button["action"] in _ACTION_VOCAB, (screen_id, button)
            assert isinstance(button["label"], str) and button["label"], screen_id
            # verb only on dispatch; target only on guard/back.
            if button["action"] == "dispatch":
                assert isinstance(button.get("verb"), str) and button["verb"], screen_id
            if button["action"] in ("guard", "back"):
                assert isinstance(button.get("target"), str) and button["target"], screen_id


def test_screen_04_is_calibration_failure_recovery():
    """Screen 04 is the CALIBRATION_FAILED recovery screen with RETRY."""
    entry = CATALOG["screens"]["04"]
    assert entry["headline"] == "Clear Backlight for Calibration"
    assert entry["buttons"][0]["label"] == "RETRY"
    assert entry["buttons"][0]["verb"] == "confirm_empty"
    assert entry["buttons"][0]["size"] == "lg"
    assert "CALIBRATION_FAILED" in entry["special"]
    assert "RETIRED" not in entry["special"]


def test_screen_05_is_indeterminate_progress_with_no_buttons():
    """Screen 05 'Calibrating…' is the only live calibration screen: no
    button, indeterminate progress, auto-advances after confirm_empty."""
    entry = CATALOG["screens"]["05"]
    assert entry["headline"] == "Calibrating…"
    assert entry["progress_bar"] is True
    assert entry["buttons"] == []
    assert "auto-fires confirm_empty" in entry["special"]


def test_hold_still_is_centred_place_layout_without_progress():
    """ADR 0005: SETTLING uses shared Hold Still, not Place or Processing."""
    entry = CATALOG["screens"]["hold_still"]
    assert entry["headline"] == "Hold Still"
    assert entry["variant"] == "place"
    assert entry["progress_bar"] is False
    assert entry["status_bar"] is True
    assert "BLOCK ID" in entry["sub"]


def test_block_id_persists_on_place_capture_remove_subs():
    for sid in ("07", "hold_still", "08", "09", "capture_review"):
        assert "BLOCK ID" in CATALOG["screens"][sid]["sub"]


@pytest.mark.parametrize("screen_id", ("09", "14", "15", "16"))
def test_removal_screens_offer_next_capture_override(screen_id):
    entry = CATALOG["screens"][screen_id]
    button = next(
        button for button in entry["buttons"] if button["label"] == "NEXT CAPTURE"
    )

    assert button["action"] == "dispatch"
    assert button["verb"] == "next_capture"


@pytest.mark.parametrize("screen_id", ("07", "hold_still"))
def test_placement_screens_offer_a_manual_capture_override(screen_id):
    # Data-collection throughput: the operator triggers the capture instead of
    # waiting for presence confirmation and the stable-duration clock.
    entry = CATALOG["screens"][screen_id]
    button = next(
        button for button in entry["buttons"] if button["label"] == "CAPTURE NOW"
    )

    assert button["action"] == "dispatch"
    assert button["verb"] == "capture_now"


def test_catalog_is_json_serializable_and_helper_round_trips():
    dumped = json.dumps(CATALOG)  # raises if any value is not JSON-safe
    assert json.loads(catalog_json()) == json.loads(dumped)


def test_capture_review_offers_accept_and_retake_with_still_variant():
    entry = CATALOG["screens"]["capture_review"]
    assert entry["variant"] == "still"
    assert entry["status_bar"] is True
    assert "BLOCK ID" in entry["sub"]
    labels = [button["label"] for button in entry["buttons"]]
    assert labels == ["RETAKE", "ACCEPT"]
    retake = entry["buttons"][0]
    accept = entry["buttons"][1]
    assert retake["action"] == "dispatch" and retake["verb"] == "retry_capture"
    assert accept["action"] == "dispatch" and accept["verb"] == "accept_capture"


def test_between_orders_offers_start_results_and_session_exit_actions():
    """A closed work order leaves a safe path to start, inspect, or end."""
    entry = CATALOG["screens"]["between_orders"]
    labels = {button["label"]: button for button in entry["buttons"]}
    assert entry["variant"] == "preview_actions"
    assert set(labels) == {"START NEW WORK ORDER", "VIEW RESULTS", "END SESSION"}
    start = labels["START NEW WORK ORDER"]
    assert start["action"] == "dispatch" and start["verb"] == "start_work_order"
    end = labels["END SESSION"]
    assert end["action"] == "dispatch" and end["verb"] == "end_session"
    # VIEW RESULTS stays a pure client guard, never a dispatch.
    view = labels["VIEW RESULTS"]
    assert view["action"] == "guard"
    assert view["target"] == "view_results_guard"
    assert "verb" not in view


def test_first_work_order_reuses_startup_layout_and_opens_work_order():
    startup = CATALOG["screens"]["01"]
    first = CATALOG["screens"]["first_work_order"]

    assert first["headline"] == startup["headline"]
    assert first["sub"] == startup["sub"]
    assert first["variant"] == startup["variant"] == "idle"
    assert first["buttons"] == [
        {"label": "START SESSION", "action": "dispatch",
         "verb": "start_work_order"}
    ]


def test_results_table_can_return_to_the_between_orders_preview():
    assert CATALOG["screens"]["results_table"]["buttons"] == [
        {"label": "GO BACK", "action": "back", "target": "view_results_guard"}
    ]


def test_work_order_finish_is_available_after_blocks_on_the_slide_screen():
    """The bracket must stay open while both its blocks and slides capture."""
    block_entry = CATALOG["screens"]["block_scan_work_order"]
    block_labels = {
        button["label"]: button for button in block_entry["buttons"]
    }
    assert set(block_labels) == {"RECALIBRATE", "FINISH BLOCKS"}
    finish_blocks = block_labels["FINISH BLOCKS"]
    assert finish_blocks["action"] == "guard"
    assert finish_blocks["target"] == "finish_blocks_guard"

    slide_entry = CATALOG["screens"]["slide_capture_work_order"]
    slide_labels = {
        button["label"]: button for button in slide_entry["buttons"]
    }
    # VIEW RESULTS joins FINISH WORK ORDER on this screen.
    assert set(slide_labels) == {"FINISH WORK ORDER", "VIEW RESULTS"}
    finish_wo = slide_labels["FINISH WORK ORDER"]
    assert finish_wo["action"] == "dispatch"
    assert finish_wo["verb"] == "finish_work_order"


@pytest.mark.parametrize("screen_id", ["06", "block_scan_work_order"])
def test_scan_block_screens_offer_recalibrate(screen_id):
    """After the camera moves, rebuild the baseline from the idle scan screen."""
    buttons = CATALOG["screens"][screen_id]["buttons"]
    assert {"label": "RECALIBRATE", "action": "dispatch",
            "verb": "confirm_empty"} in buttons


def test_fallback_catalog_matches_every_server_rendered_entry():
    """Every offline screen matches the server's rendered fields."""
    html = _index_html_source()
    match = re.search(
        r"const FALLBACK_CATALOG = (?P<catalog>\{.*?\});\n\n  let catalog",
        html,
        flags=re.DOTALL,
    )
    assert match, "could not find FALLBACK_CATALOG"
    fallback_json = re.sub(r"/\*.*?\*/", "", match.group("catalog"))
    fallback_json = re.sub(
        r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)",
        r'\1"\2"\3',
        fallback_json,
    )
    fallback_screens = json.loads(fallback_json)["screens"]
    server_screens = {
        screen_id: {
            key: value for key, value in entry.items() if key != "special"
        }
        for screen_id, entry in CATALOG["screens"].items()
    }

    assert fallback_screens == server_screens


def test_hybrid_mode_slide_capture_renders_view_results_button():
    """The open slide-capture screen's entry point actually RENDERS
    for a real Hybrid state (routes through select_screen, not just a
    literal catalog lookup), and its verb is a `dispatch` (a real side
    effect: pause_capture), never a `guard`/`back` no-op."""
    state = dict(
        mode="hybrid", open_retrieval=False, work_order_open=True,
        phase="slides", capture_mode="slide", capture_state="EMPTY",
    )
    screen_id = router.select_screen(state)
    assert screen_id == "slide_capture_work_order"
    entry = CATALOG["screens"][screen_id]
    labels = {button["label"]: button for button in entry["buttons"]}
    assert "VIEW RESULTS" in labels
    view_results = labels["VIEW RESULTS"]
    assert view_results["action"] == "dispatch"
    assert view_results["verb"] == "pause_capture"


def test_screen_17_offers_retake_and_skip_slide():
    entry = CATALOG["screens"]["17"]
    labels = [button["label"] for button in entry["buttons"]]
    assert "SKIP SLIDE" in labels
    assert "RETAKE SLIDE" in labels
    retake = next(
        button for button in entry["buttons"] if button["label"] == "RETAKE SLIDE"
    )
    assert retake["action"] == "dispatch"
    assert retake["verb"] == "retry_capture"
    assert "retry_capture" in entry["special"]


def _index_html_source() -> str:
    from pathlib import Path

    return (
        Path(__file__).resolve().parents[1]
        / "code" / "kiosk" / "static" / "index.html"
    ).read_text(encoding="utf-8")


def test_finish_blocks_dispatch_clears_the_guard_latch_client_side():
    """The guard-latch loop trap. Screen 10 (Finish Block Capture?) shows
    only while `finish_blocks_guard` is true AND the router is idle on
    phase=='blocks' (see router.select_screen). A Hybrid <2-usable-blocks
    bounce returns phase to 'blocks', so a guard left true by the dispatch
    would route straight back to screen 10 and loop forever.

    Checked textually (no JS test harness in this repo): the "dispatch" case
    must clear `finish_blocks_guard` when the dispatched
    verb is "finish_blocks", so the operator always lands back on live block
    capture (or advances into the drain) rather than a re-latched confirm
    modal.
    """
    html = _index_html_source()
    dispatch_start = html.index('case "dispatch":')
    dispatch_end = html.index('case "guard":', dispatch_start)
    dispatch_block = html[dispatch_start:dispatch_end]

    assert 'b.verb === "finish_blocks"' in dispatch_block
    assert "ui.finish_blocks_guard = false" in dispatch_block


def test_view_results_dispatch_pauses_capture_and_opens_results_client_side():
    """Tapping VIEW RESULTS on the open slide-capture screen must, in
    ONE tap, (a) dispatch pause_capture (the real server-side side effect --
    the router itself makes no side effects) and (b) set the client-owned
    view_results_guard flag so the next poll routes to results_table.
    Mirrors the existing finish_blocks guard-clearing special case."""
    html = _index_html_source()
    dispatch_start = html.index('case "dispatch":')
    dispatch_end = html.index('case "guard":', dispatch_start)
    dispatch_block = html[dispatch_start:dispatch_end]

    assert 'b.verb === "pause_capture"' in dispatch_block
    assert "ui.view_results_guard = true" in dispatch_block
    assert "viewResultsPausedCapture = true" in dispatch_block


def test_go_back_from_results_resumes_capture_only_when_this_client_paused_it():
    """GO BACK from Results must dispatch resume_capture when THIS
    client is the one that paused capture (the open slide-capture entry
    point) -- and must be conditioned on that local flag, not fire
    unconditionally, so Open Retrieval's between_orders VIEW RESULTS/GO BACK
    pair (which never dispatches pause_capture) stays completely untouched."""
    html = _index_html_source()
    back_start = html.index('case "back":')
    back_end = html.index('case "engage":', back_start)
    back_block = html[back_start:back_end]

    # Every back tap clears its own guard flag unconditionally.
    assert "setFlag(b.target, false)" in back_block
    assert 'b.target === "view_results_guard"' in back_block
    assert "viewResultsPausedCapture" in back_block
    assert 'command("resume_capture")' in back_block


def test_hybrid_pool_bounce_message_is_rendered_not_just_used_as_a_dedupe_key():
    """The bounce reason must be RENDERED on screen 06, not
    merely carried as a dedupe key the way `last_event` is used for the
    duplicate-scan flash (`handleDuplicate`)."""
    html = _index_html_source()
    screen_06_start = html.index('"06": {')
    screen_06_end = html.index('"07": {')
    screen_06_block = html[screen_06_start:screen_06_end]

    assert "{hybrid_pool_bounce_message}" in screen_06_block
