"""Preview-tool scene coverage for the redesigned results screen (#232).

``tools/preview_kiosk.py`` is the hardware-free browser preview: each SCENE
feeds the REAL relay + router deterministic fake signals so a human can paint
any screen with no camera/PC attached. These tests guard the new grouped
results scene the way ``verify_scenes`` already guards routing -- pure, no
server started -- plus assert the scene actually carries the multi-work-order
fake data the redesigned front end groups on.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import preview_kiosk  # noqa: E402
from kiosk.relay import KioskRelay  # noqa: E402


def _results_scene():
    scene = next(
        (s for s in preview_kiosk.SCENES if s.screen == "results_table"), None
    )
    assert scene is not None, "a results_table scene must exist for the preview"
    return scene


def test_every_scene_still_routes_to_its_expected_screen():
    # The new scene rides the same self-check every other scene does; a routing
    # regression anywhere (incl. the results scene) fails here.
    for check in preview_kiosk.verify_scenes():
        assert check.ok, f"{check.label}: expected {check.expected}, got {check.actual}"


def test_results_scene_routes_to_the_results_table():
    scene = _results_scene()
    state = KioskRelay(scene.handle()).state(scene.ui)
    assert state["screen"] == "results_table"


def test_results_scene_carries_multiple_work_orders_each_with_pass_and_review():
    scene = _results_scene()
    state = KioskRelay(scene.handle()).state(scene.ui)
    rows = state["results_rows"]

    # Every row carries the fields the redesigned front end reads.
    for row in rows:
        assert {"work_order", "block_id", "verdict", "claim_reason",
                "claim_score", "evidence"} <= row.keys()

    # At least two work orders, each with at least one PASS and one REVIEW.
    by_wo: dict[str, set[str]] = {}
    for row in rows:
        by_wo.setdefault(str(row["work_order"]), set()).add(row["verdict"])
    assert len(by_wo) >= 2
    for verdicts in by_wo.values():
        assert {"PASS", "REVIEW"} <= verdicts
