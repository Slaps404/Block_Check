"""Pure inspection-sheet projection (#151, ADR 0009 follow-on).

``project_inspection(row)`` turns one results-table row (already carrying
``top_block``/``contact_sheet_dir`` from #151's extended
``list_results_ready_work_orders`` SELECT) into the ordered contact-sheet
descriptors the kiosk fetches when an operator expands a REVIEW row: the top
match always, and the claimed block appended only when it disagrees --
mirrors ``work_order_evaluator.flagged_pairs``. Dict-in/dict-out, no I/O --
styled exactly like ``kiosk.results_table.project_results_table``.
"""
from __future__ import annotations

from typing import Any, Mapping, TypedDict


class InspectionDescriptor(TypedDict):
    unique_id: str
    role: str
    path: str


def project_inspection(row: Mapping[str, Any]) -> list[InspectionDescriptor]:
    """Ordered contact-sheet descriptors for one results-table row.

    Returns an empty list for a PASS row (no flagged pairs to inspect).
    Never mutates ``row``.
    """
    if row.get("verdict") != "REVIEW":
        return []

    capture_id = row["capture_id"]
    top_block = row["top_block"]
    claimed_block = row["block_id"]
    contact_sheet_dir = row["contact_sheet_dir"]

    descriptors: list[InspectionDescriptor] = []
    if top_block is not None:
        descriptors.append(
            _descriptor(contact_sheet_dir, capture_id, top_block, "TOP MATCH")
        )
    if claimed_block != top_block:
        descriptors.append(
            _descriptor(contact_sheet_dir, capture_id, claimed_block, "CLAIMED")
        )
    return descriptors


def _descriptor(
    contact_sheet_dir: str, capture_id: str, block_id: str, role: str
) -> InspectionDescriptor:
    unique_id = f"{capture_id}__{block_id}"
    return {
        "unique_id": unique_id,
        "role": role,
        "path": f"{contact_sheet_dir}/{unique_id}.png",
    }
