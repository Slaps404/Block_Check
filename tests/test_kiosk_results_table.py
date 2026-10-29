"""Pure results-table projection (ADR 0009).

``project_results_table(rows)`` is the sort/color/expand-target seam between
the durable per-slide verdict rows (``SessionWorkflow.
list_results_ready_work_orders``) and the kiosk's results-table screen. It is
a pure function -- no I/O -- exactly the shape of ``kiosk.router.select_screen``,
so it is unit-tested here directly with synthetic dicts, the same way
``test_kiosk_router.py`` tests ``select_screen`` with synthetic ``_state(...)``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from kiosk.results_table import (
    _COLOR_BY_VERDICT,
    _SORT_RANK_BY_VERDICT,
    _UNKNOWN_SORT_RANK,
    evidence_paths_for_capture,
    project_results_table,
)

_INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "code" / "kiosk" / "static" / "index.html"
).read_text(encoding="utf-8")


def _row(capture_id, block_id, verdict, *, reason="", score=0.9, **overrides):
    row = dict(
        capture_id=capture_id,
        block_id=block_id,
        verdict=verdict,
        claim_reason=reason,
        claim_score=score,
    )
    row.update(overrides)
    return row


def test_project_results_table_row_carries_an_expand_target():
    rows = project_results_table([_row("cap-9", "b9", "PASS")])

    assert rows[0]["expand_target"] == "cap-9"


def test_project_results_table_returns_empty_list_for_no_rows():
    assert project_results_table([]) == []


def test_project_results_table_passes_extra_columns_through_unchanged():
    # The human work-order number is the results screen's grouping key,
    # and evidence refs feed the expanded row; project_results_table deep-copies
    # each row, so extra SELECT columns must survive untouched.
    evidence = {
        "block_thumb": "/artifacts/cap-1_block_thumb.jpg",
        "slide_thumb": "/artifacts/cap-1_slide_thumb.jpg",
        "block_display": "/artifacts/cap-1_block_display.jpg",
        "slide_display": "/artifacts/cap-1_slide_display.jpg",
        "overlay_display": "/artifacts/cap-1_overlay_display.jpg",
    }
    rows = project_results_table([
        _row("cap-1", "b1", "PASS", work_order="12080", evidence=evidence),
        _row("cap-2", "b2", "REVIEW", work_order="12094"),
    ])
    by_capture = {row["capture_id"]: row for row in rows}

    assert by_capture["cap-1"]["work_order"] == "12080"
    assert by_capture["cap-2"]["work_order"] == "12094"
    assert by_capture["cap-1"]["evidence"] == evidence


@pytest.mark.parametrize(
    "verdict, has_overlay",
    [("PASS", True), ("REVIEW", True), ("pending", False), ("error", False)],
)
def test_evidence_paths_for_capture(tmp_path, verdict, has_overlay):
    paths = evidence_paths_for_capture(tmp_path, "cap-1", verdict)

    for ref in ("block_thumb", "slide_thumb", "block_display", "slide_display"):
        assert paths[ref] == str(tmp_path / f"cap-1_{ref}.jpg")
    expected_overlay = str(tmp_path / "cap-1_overlay_display.jpg") if has_overlay else None
    assert paths["overlay_display"] == expected_overlay


def test_project_results_table_color_codes_all_four_states():
    # Hybrid adds two non-verdict row states -- ERROR (amber, a system
    # failure) and PENDING (gray, still scoring) -- alongside the existing
    # PASS/REVIEW verdicts.
    rows = project_results_table([
        _row("cap-1", "b1", "PASS"),
        _row("cap-2", "b2", "REVIEW"),
        _row("cap-3", "b3", "ERROR"),
        _row("cap-4", "b4", "PENDING"),
    ])
    by_capture = {row["capture_id"]: row for row in rows}

    assert by_capture["cap-1"]["color"] == "green"
    assert by_capture["cap-2"]["color"] == "red"
    assert by_capture["cap-3"]["color"] == "amber"
    assert by_capture["cap-4"]["color"] == "gray"


def test_project_results_table_sorts_error_review_pass_pending_in_order():
    # Order is ERROR, REVIEW, PASS, PENDING so a system failure sorts ahead of even a REVIEW match failure.
    rows = project_results_table([
        _row("cap-1", "b1", "PENDING"),
        _row("cap-2", "b2", "PASS"),
        _row("cap-3", "b3", "REVIEW"),
        _row("cap-4", "b4", "ERROR"),
        _row("cap-5", "b5", "PENDING"),
        _row("cap-6", "b6", "ERROR"),
    ])

    assert [row["verdict"] for row in rows] == [
        "ERROR", "ERROR", "REVIEW", "PASS", "PENDING", "PENDING",
    ]
    # Stable within each group: original relative order is preserved.
    assert [row["capture_id"] for row in rows] == [
        "cap-4", "cap-6", "cap-3", "cap-2", "cap-1", "cap-5",
    ]


def test_project_results_table_bad_row_does_not_discard_good_rows_in_batch():
    # Degradation must be per-row: the sole production caller (kiosk.relay's
    # /state poll) is unguarded, so one malformed row must not take down the
    # operator's entire results table.
    rows = project_results_table([
        _row("cap-1", "b1", "PASS"),
        _row("cap-2", "b2", "queued"),  # leaked internal job state
        _row("cap-3", "b3", "REVIEW"),
    ])

    by_capture = {row["capture_id"]: row for row in rows}
    assert by_capture["cap-1"]["verdict"] == "PASS"
    assert by_capture["cap-1"]["color"] == "green"
    assert by_capture["cap-3"]["verdict"] == "REVIEW"
    assert by_capture["cap-3"]["color"] == "red"


@pytest.mark.parametrize("verdict", ["queued", None], ids=["unrecognized", "missing"])
def test_project_results_table_bad_verdict_surfaces_as_loud_unknown_row(verdict):
    # An unrecognized or missing verdict must not be silently
    # rendered as REVIEW/red, and must not abort the batch. It gets its own
    # loud color, and the raw value is left on the row for operators/logs.
    row = _row("cap-1", "b1", verdict)
    if verdict is None:
        del row["verdict"]

    projected = project_results_table([row])

    assert projected[0].get("verdict") == verdict
    assert projected[0]["color"] not in _COLOR_BY_VERDICT.values()
    assert projected[0]["color"] == "purple"


def test_project_results_table_unknown_state_sorts_at_or_above_error():
    # An unknown row must sort AT OR ABOVE ERROR (never after
    # PENDING, and never buried at the bottom of a long table).
    rows = project_results_table([
        _row("cap-1", "b1", "PENDING"),
        _row("cap-2", "b2", "ERROR"),
        _row("cap-3", "b3", "retrying"),  # leaked internal job state
    ])

    assert rows[0]["capture_id"] == "cap-3"
    assert [row["verdict"] for row in rows] == ["retrying", "ERROR", "PENDING"]


def test_project_results_table_logs_a_warning_naming_the_capture_id(caplog):
    with caplog.at_level(logging.WARNING, logger="kiosk.results_table"):
        project_results_table([_row("cap-42", "b1", "queued")])

    assert any(
        "cap-42" in record.getMessage() and "queued" in record.getMessage()
        for record in caplog.records
    )


def test_python_and_js_result_state_vocabularies_match():
    # The Python projection and the browser's mirror
    # (index.html RT_STATE_RANK / rtStateRank) must agree exactly on the
    # verdict set, the sort ranks, and the unknown-row fallback rank.
    match = re.search(r"const RT_STATE_RANK = \{([^}]*)\};", _INDEX_HTML)
    assert match, "RT_STATE_RANK literal not found in index.html"
    js_rank = {
        name: int(value)
        for name, value in re.findall(r"(\w+):\s*(-?\d+)", match.group(1))
    }
    assert js_rank == _SORT_RANK_BY_VERDICT

    unknown_match = re.search(r"const RT_UNKNOWN_RANK = (-?\d+);", _INDEX_HTML)
    assert unknown_match, "RT_UNKNOWN_RANK literal not found in index.html"
    assert int(unknown_match.group(1)) == _UNKNOWN_SORT_RANK

    # The unknown fallback must sort at or above ERROR (rank 0) in both.
    assert _UNKNOWN_SORT_RANK <= _SORT_RANK_BY_VERDICT["ERROR"]

    color_match = re.search(r"const RT_STATE_COLOR = \{([^}]*)\};", _INDEX_HTML)
    assert color_match, "RT_STATE_COLOR literal not found in index.html"
    js_color_class = dict(re.findall(r'(\w+):\s*"([\w-]+)"', color_match.group(1)))
    assert set(js_color_class) == set(_COLOR_BY_VERDICT)
    # Same four verdicts named the same way, each mapped to the matching
    # browser CSS class suffix (rt-error/rt-review/rt-pass/rt-pending).
    expected_class_by_python_color = {
        "amber": "rt-error",
        "red": "rt-review",
        "green": "rt-pass",
        "gray": "rt-pending",
    }
    for verdict, python_color in _COLOR_BY_VERDICT.items():
        assert js_color_class[verdict] == expected_class_by_python_color[python_color]
