"""#256: the Hybrid "slide needs attention" banner projection.

A THIRD, non-interrupting layer alongside the existing WS-A offline banner
(``.offline-banner`` in ``code/kiosk/static/index.html``) -- the shape this
deliberately copies: a static, layered notice that never tears down the
routed screen beneath it. Explicitly NOT the screen-18 duplicate-scan
overlay (``.overlay``), which is a full-screen takeover -- the wrong shape
for a late background-job outcome, which must never hijack an active
capture screen.

Pure function of already-projected result rows plus the two live
work-order signals the router already reads (``work_order_open``,
``open_work_order_id``) -- no I/O, no store call of its own, so it can
never be the thing that raises on the kiosk poll path. ``KioskRelay``
wraps the call in its own try/except (blast-radius rule: a raise here must
degrade to "no banner", never reach ``_camera_loop``'s bare ``except
Exception``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def project_attention_banner(
    rows: Sequence[Mapping[str, Any]],
    *,
    work_order_open: bool,
    open_work_order_id: int | None,
) -> dict[str, Any] | None:
    """The single most urgent ERROR row as a banner projection, or ``None``.

    Only ``verdict == "ERROR"`` rows are attention items (CONTEXT.md
    "Hybrid Processing Error" -- a system/artifact failure, never a match
    failure; REVIEW/PASS/PENDING rows never surface here). The FIRST such
    row in ``rows`` is picked -- both ``list_hybrid_results`` and
    ``kiosk.results_table.project_results_table`` already order/sort their
    output so this stays a stable choice rather than flickering between
    multiple simultaneous errors on every poll.

    ``can_recapture`` is the "waits for an available transition" gate
    (#256 acceptance criterion): False only when a work order IS actively
    capturing (``work_order_open``) and it is a DIFFERENT work order than
    the one the attention item belongs to. If the attention item's OWN
    work order is the one open, or no work order is open at all,
    recapture is immediately actionable -- there is nothing to wait for.
    """
    for row in rows:
        if row.get("verdict") != "ERROR":
            continue
        item_work_order_id = row.get("work_order_id")
        waiting = bool(work_order_open) and open_work_order_id != item_work_order_id
        return {
            "capture_id": row.get("capture_id"),
            "block_id": row.get("block_id"),
            "work_order_id": item_work_order_id,
            "message": f"Slide needs attention: block {row.get('block_id')}",
            "can_recapture": not waiting,
        }
    return None
