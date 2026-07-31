"""Pure results-table projection (#150, ADR 0009 follow-on).

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


def test_project_results_table_color_codes_pass_and_review_rows():
    rows = project_results_table([
        _row("cap-1", "b1", "PASS"),
        _row("cap-2", "b2", "REVIEW", reason="low score"),
    ])
    by_capture = {row["capture_id"]: row for row in rows}

    assert by_capture["cap-1"]["verdict"] == "PASS"
    assert by_capture["cap-1"]["color"] == "green"
    assert by_capture["cap-2"]["verdict"] == "REVIEW"
    assert by_capture["cap-2"]["color"] == "red"


def test_project_results_table_sorts_review_rows_to_the_top():
    rows = project_results_table([
        _row("cap-1", "b1", "PASS"),
        _row("cap-2", "b2", "REVIEW"),
        _row("cap-3", "b3", "PASS"),
        _row("cap-4", "b4", "REVIEW"),
    ])

    assert [row["verdict"] for row in rows] == ["REVIEW", "REVIEW", "PASS", "PASS"]
    # Stable within each group: original relative order is preserved.
    assert [row["capture_id"] for row in rows] == ["cap-2", "cap-4", "cap-1", "cap-3"]


def test_project_results_table_row_carries_an_expand_target():
    rows = project_results_table([_row("cap-9", "b9", "PASS")])

    assert rows[0]["expand_target"] == "cap-9"


def test_project_results_table_returns_empty_list_for_no_rows():
    assert project_results_table([]) == []


def test_project_results_table_passes_work_order_through():
    # #232: the human work-order number (e.g. "12080") is the results screen's
    # section grouping key. project_results_table deep-copies each row, so any
    # extra column the SELECT emits -- work_order among them -- must survive the
    # projection untouched (not just the two render fields it adds).
    rows = project_results_table([
        _row("cap-1", "b1", "PASS", work_order="12080"),
        _row("cap-2", "b2", "REVIEW", work_order="12094"),
    ])
    by_capture = {row["capture_id"]: row for row in rows}

    assert by_capture["cap-1"]["work_order"] == "12080"
    assert by_capture["cap-2"]["work_order"] == "12094"


def test_evidence_paths_for_capture_pass_includes_all_five_refs(tmp_path):
    paths = evidence_paths_for_capture(tmp_path, "cap-1", "PASS")

    assert paths["block_thumb"] == str(tmp_path / "cap-1_block_thumb.jpg")
    assert paths["slide_thumb"] == str(tmp_path / "cap-1_slide_thumb.jpg")
    assert paths["block_display"] == str(tmp_path / "cap-1_block_display.jpg")
    assert paths["slide_display"] == str(tmp_path / "cap-1_slide_display.jpg")
    assert paths["overlay_display"] == str(tmp_path / "cap-1_overlay_display.jpg")


def test_evidence_paths_for_capture_review_includes_overlay(tmp_path):
    paths = evidence_paths_for_capture(tmp_path, "cap-2", "REVIEW")

    assert paths["overlay_display"] == str(tmp_path / "cap-2_overlay_display.jpg")


def test_evidence_paths_for_capture_pending_omits_overlay(tmp_path):
    paths = evidence_paths_for_capture(tmp_path, "cap-3", "pending")

    assert paths["block_thumb"] == str(tmp_path / "cap-3_block_thumb.jpg")
    assert paths["slide_display"] == str(tmp_path / "cap-3_slide_display.jpg")
    assert paths["overlay_display"] is None


def test_evidence_paths_for_capture_error_omits_overlay(tmp_path):
    paths = evidence_paths_for_capture(tmp_path, "cap-4", "error")

    assert paths["overlay_display"] is None
    assert paths["block_display"] is not None


def test_project_results_table_color_codes_all_four_states():
    # #248: Hybrid adds two non-verdict row states -- ERROR (amber, a system
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
    # #248: ordering changes from REVIEW-first to ERROR, REVIEW, PASS, PENDING
    # so a system failure sorts ahead of even a REVIEW match failure.
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
    # #248-fix: the old behavior raised UnknownResultStateError on the FIRST
    # bad row, discarding every good row from the same call. The sole
    # production caller (kiosk.relay's /state poll) is unguarded, so one
    # malformed row used to take down the operator's entire results table.
    # Degradation must now be per-row: good rows in the same batch still
    # render.
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


def test_project_results_table_unrecognized_verdict_surfaces_as_loud_unknown_row():
    # #248-fix: an unrecognized verdict must not be silently rendered as
    # REVIEW/red (the original #150 bug #248 fixed), and must not abort the
    # batch (the #248 regression this fix addresses). It gets its own loud,
    # distinct color, and the raw unexpected value is left on the row so an
    # operator or a log line can see exactly what it was.
    rows = project_results_table([_row("cap-1", "b1", "queued")])

    assert rows[0]["verdict"] == "queued"
    assert rows[0]["color"] not in _COLOR_BY_VERDICT.values()
    assert rows[0]["color"] == "purple"


def test_project_results_table_missing_verdict_surfaces_as_loud_unknown_row():
    rows = [_row("cap-1", "b1", "PASS")]
    del rows[0]["verdict"]

    projected = project_results_table(rows)

    assert projected[0].get("verdict") is None
    assert projected[0]["color"] == "purple"


def test_project_results_table_unknown_state_sorts_at_or_above_error():
    # #248-fix: an unknown row must sort AT OR ABOVE ERROR (never after
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
    # #248-fix requirement 5: the Python projection and the browser's mirror
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


def test_project_results_table_passes_evidence_through_unchanged():
    evidence = {
        "block_thumb": "/artifacts/cap-1_block_thumb.jpg",
        "slide_thumb": "/artifacts/cap-1_slide_thumb.jpg",
        "block_display": "/artifacts/cap-1_block_display.jpg",
        "slide_display": "/artifacts/cap-1_slide_display.jpg",
        "overlay_display": "/artifacts/cap-1_overlay_display.jpg",
    }
    rows = project_results_table([
        _row("cap-1", "b1", "PASS", evidence=evidence),
    ])

    assert rows[0]["evidence"] == evidence
