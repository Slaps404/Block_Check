"""Pure inspection-sheet projection tests (#151, ADR 0009 follow-on).

``kiosk.inspection.project_inspection(row)`` turns one results-table row
(already carrying ``top_block``/``contact_sheet_dir`` from #151's extended
``list_results_ready_work_orders`` SELECT) into the ordered contact-sheet
descriptors the kiosk fetches when an operator expands a REVIEW row: the top
match always, and the claimed block appended only when it disagrees --
mirrors ``work_order_evaluator.flagged_pairs``. Dict-in/dict-out, no I/O --
styled exactly like ``kiosk.results_table.project_results_table``.
"""
from __future__ import annotations

from kiosk.inspection import project_inspection


def _row(**overrides):
    row = {
        "capture_id": "capture_9",
        "block_id": "62626262",
        "verdict": "REVIEW",
        "top_block": "51151378",
        "contact_sheet_dir": "/sessions/1/work_orders/work_order_000003_sheets",
    }
    row.update(overrides)
    return row


def test_project_inspection_orders_top_match_first_then_claim():
    descriptors = project_inspection(_row())

    assert [d["role"] for d in descriptors] == ["TOP MATCH", "CLAIMED"]
    assert descriptors[0]["unique_id"] == "capture_9__51151378"
    assert descriptors[1]["unique_id"] == "capture_9__62626262"
    assert descriptors[0]["path"] == (
        "/sessions/1/work_orders/work_order_000003_sheets/capture_9__51151378.png"
    )
    assert descriptors[1]["path"] == (
        "/sessions/1/work_orders/work_order_000003_sheets/capture_9__62626262.png"
    )


def test_project_inspection_returns_single_descriptor_when_claim_is_top_match():
    descriptors = project_inspection(_row(top_block="62626262"))

    assert len(descriptors) == 1
    assert descriptors[0]["role"] == "TOP MATCH"
    assert descriptors[0]["unique_id"] == "capture_9__62626262"


def test_project_inspection_returns_empty_for_pass_rows():
    assert project_inspection(_row(verdict="PASS")) == []


def test_project_inspection_returns_empty_for_pending_rows():
    # #248: PENDING (still scoring) has no verdict evidence to inspect yet.
    assert project_inspection(_row(verdict="PENDING")) == []


def test_project_inspection_returns_empty_for_error_rows():
    # #248: ERROR (a system failure) has no verdict evidence either.
    assert project_inspection(_row(verdict="ERROR")) == []


def test_project_inspection_returns_only_claim_when_top_block_is_none():
    """ADR 0009 boundary rule: when the claimed block wasn't scanned in this
    order (or the claimed pair gate-failed), evaluate_work_order leaves
    top_block=None. project_inspection must not fabricate a TOP MATCH
    descriptor for a block that was never scored -- only the claim shows."""
    descriptors = project_inspection(_row(top_block=None))

    assert len(descriptors) == 1
    assert descriptors[0]["role"] == "CLAIMED"
    assert descriptors[0]["unique_id"] == "capture_9__62626262"


def test_project_inspection_does_not_mutate_input_row():
    row = _row()
    original = dict(row)
    project_inspection(row)
    assert row == original
