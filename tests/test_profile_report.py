"""#258: pure-formatter tests for `code/session/profile_report.py`.

This module has NO clock, NO I/O, NO store access -- every test here drives
it with a literal `now_ns` and literal raw row dicts (the shape
`ProcessingStore.list_hybrid_profile_rows` produces), so every assertion
below is an exact integer or exact string, never a "some positive number"
fuzzy check.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import session.profile_report as profile_report  # noqa: E402
from session.profile_report import (  # noqa: E402
    PROFILE_STAGE_ORDER,
    format_profile_console,
    profile_screen_fields,
    project_profile_rows,
)


# The exact durable-lifecycle strings that must NEVER reach a caller of
# either renderer -- `sets`/`slide_captures.job_state` values other than the
# four visible states, plus 'superseded' (#256).
_INTERNAL_LIFECYCLE_STRINGS = ("queued", "preparing", "scoring", "complete", "superseded")


def _flatten_string_values(value: object) -> list[str]:
    """Every leaf string value reachable from a dict/tuple/list, recursively."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_string_values(item))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(_flatten_string_values(item))
        return out
    return []


# ---------------------------------------------------------------------------
# 1. Pending row: stage + elapsed derived from the injected now_ns. Finished
#    row: total plus the five-stage breakdown. Exact integers throughout.
# ---------------------------------------------------------------------------


def test_pending_row_reports_stage_and_elapsed_from_injected_clock():
    raw_rows = [{
        "capture_id": "cap-1",
        "block_id": "11111111",
        "job_state": "scoring",
        "verdict": None,
        "stage": "accurate_scoring",
        "queued_ns": 2_000_000,
        "shadow": 0,
    }]

    rows = project_profile_rows(raw_rows, now_ns=9_000_000)

    assert len(rows) == 1
    row = rows[0]
    assert row.capture_id == "cap-1"
    assert row.block_id == "11111111"
    assert row.state == "PENDING"
    assert row.stage == "accurate_scoring"
    assert row.elapsed_ms == 7
    assert row.total_ms is None
    assert row.stage_ms == {}
    assert row.shadow is False


def test_finished_row_reports_total_and_five_stage_breakdown():
    stage_ms = {
        "queue_wait": 1, "preparation": 3, "heuristic_selection": 2,
        "accurate_scoring": 4, "artifact_write": 3,
    }
    raw_rows = [{
        "capture_id": "cap-2",
        "block_id": "22222222",
        "job_state": "complete",
        "verdict": "PASS",
        "total_ms": 13,
        "stage_ms_json": json.dumps(stage_ms),
        "shadow": 0,
    }]

    rows = project_profile_rows(raw_rows, now_ns=999_999_999)

    assert len(rows) == 1
    row = rows[0]
    assert row.state == "PASS"
    assert row.stage is None
    assert row.elapsed_ms is None
    assert row.total_ms == 13
    assert row.stage_ms == stage_ms
    assert tuple(row.stage_ms) == PROFILE_STAGE_ORDER  # exact stage order

    fields = profile_screen_fields(rows, queue_count=0)
    assert fields["rows"][0]["total_ms"] == 13
    assert fields["rows"][0]["stage_breakdown"] == stage_ms

    console = format_profile_console(rows, queue_count=0)
    assert "total=13ms" in console
    for name, ms in stage_ms.items():
        assert f"{name}={ms}ms" in console


def test_gate_failed_row_omits_missing_selection_and_scoring_stages():
    """A gate-failed slide never reaches selection/scoring -- those two keys
    are simply absent, never a fabricated zero (mirrors
    `_stage_ms_breakdown`'s documented tolerance)."""
    stage_ms = {"queue_wait": 1, "preparation": 2, "artifact_write": 1}
    raw_rows = [{
        "capture_id": "cap-3",
        "block_id": "33333333",
        "job_state": "complete",
        "verdict": "REVIEW",
        "total_ms": 4,
        "stage_ms_json": json.dumps(stage_ms),
        "shadow": 0,
    }]

    rows = project_profile_rows(raw_rows, now_ns=0)

    assert rows[0].stage_ms == stage_ms
    assert "heuristic_selection" not in rows[0].stage_ms
    assert "accurate_scoring" not in rows[0].stage_ms


# ---------------------------------------------------------------------------
# 2. A shadow row is labeled shadow in BOTH renderers.
# ---------------------------------------------------------------------------


def test_shadow_row_is_labeled_in_both_screen_fields_and_console():
    raw_rows = [{
        "capture_id": "cap-shadow",
        "block_id": "44444444",
        "job_state": "complete",
        "verdict": "REVIEW",
        "total_ms": 20,
        "stage_ms_json": json.dumps({"queue_wait": 5, "preparation": 15}),
        "shadow": 1,
    }]

    rows = project_profile_rows(raw_rows, now_ns=0)
    assert rows[0].shadow is True

    fields = profile_screen_fields(rows, queue_count=1)
    entry = fields["rows"][0]
    assert entry["shadow"] is True
    assert entry["shadow_note"] == profile_report._SHADOW_NOTE

    console = format_profile_console(rows, queue_count=1)
    assert f"[{profile_report._SHADOW_TAG}]" in console


def test_non_shadow_row_is_never_tagged_shadow():
    raw_rows = [{
        "capture_id": "cap-plain",
        "block_id": "55555555",
        "job_state": "complete",
        "verdict": "PASS",
        "total_ms": 9,
        "stage_ms_json": json.dumps({"queue_wait": 9}),
        "shadow": 0,
    }]

    rows = project_profile_rows(raw_rows, now_ns=0)
    fields = profile_screen_fields(rows, queue_count=0)
    entry = fields["rows"][0]
    assert entry["shadow"] is False
    assert "shadow_note" not in entry
    assert profile_report._SHADOW_TAG not in format_profile_console(rows, queue_count=0)


# ---------------------------------------------------------------------------
# 3. No internal lifecycle string ever appears in any rendered value.
# ---------------------------------------------------------------------------


def test_no_internal_lifecycle_string_appears_in_any_rendered_value():
    raw_rows = [
        {
            "capture_id": "s001", "block_id": "11111111",
            "job_state": "queued", "verdict": None, "queued_ns": 0, "shadow": 0,
        },
        {
            "capture_id": "s002", "block_id": "22222222",
            "job_state": "preparing", "verdict": None, "queued_ns": 0, "shadow": 0,
        },
        {
            "capture_id": "s003", "block_id": "33333333",
            "job_state": "scoring", "verdict": None, "queued_ns": 0, "shadow": 0,
        },
        {
            "capture_id": "s004", "block_id": "44444444",
            "job_state": "complete", "verdict": "PASS", "total_ms": 1,
            "stage_ms_json": json.dumps({"queue_wait": 1}), "shadow": 0,
        },
        {
            "capture_id": "s005", "block_id": "55555555",
            "job_state": "error", "verdict": None, "shadow": 0,
        },
        {
            "capture_id": "s006", "block_id": "66666666",
            "job_state": "superseded", "verdict": None, "shadow": 0,
        },
    ]

    rows = project_profile_rows(raw_rows, now_ns=5_000_000)
    fields = profile_screen_fields(rows, queue_count=6)
    console = format_profile_console(rows, queue_count=6)

    rendered_values = _flatten_string_values(fields)
    console_lines = console.splitlines()
    for leaked in _INTERNAL_LIFECYCLE_STRINGS:
        assert leaked not in rendered_values, (
            f"internal lifecycle string {leaked!r} leaked as an exact "
            f"rendered field value"
        )
        # The console renders one line per row as free text (not a single
        # opaque token), so check whole-line equality is meaningless there;
        # instead confirm the raw lifecycle word never appears standalone
        # (surrounded by word boundaries) on any line.
        for line in console_lines:
            assert re.search(rf"\b{leaked}\b", line) is None, (
                f"internal lifecycle string {leaked!r} leaked into console "
                f"line: {line!r}"
            )
    assert any(row.state == "ERROR" for row in rows)


def test_missing_or_null_stage_values_on_an_old_row_do_not_raise():
    """A pre-#258 row (or one with a NULL stage_ms_json/queued_ns) must
    degrade gracefully, never raise."""
    raw_rows = [
        {"capture_id": "cap-old", "block_id": "77777777", "job_state": "queued"},
        {
            "capture_id": "cap-old-2", "block_id": "88888888",
            "job_state": "complete", "verdict": "PASS", "stage_ms_json": None,
            "total_ms": None,
        },
    ]

    rows = project_profile_rows(raw_rows, now_ns=1_000_000)

    assert rows[0].elapsed_ms is None
    assert rows[0].stage == "queue_wait"
    assert rows[1].stage_ms == {}
    assert rows[1].total_ms is None

    profile_screen_fields(rows, queue_count=0)
    format_profile_console(rows, queue_count=0)
