"""Pure results-table projection (#150, ADR 0009 follow-on; #248 4-state;
#248-fix per-row degrade).

``project_results_table(rows)`` is the sort/color/expand-target seam between
the durable per-slide row state (``SessionWorkflow.
list_results_ready_work_orders``) and the kiosk's results-table screen. Like
``kiosk.router.select_screen`` it is a pure function -- no I/O -- so it is
unit-tested directly with synthetic dicts.

Each input row carries at least ``capture_id``, ``block_id``, ``verdict``,
``claim_reason``, ``claim_score``. ``verdict`` normally carries one of four
operator-visible states (#248, CONTEXT.md "Hybrid Result State"): the two
verdicts ``"PASS"``/``"REVIEW"``, plus two non-verdict states Hybrid adds --
``"ERROR"`` (a system/artifact failure, never a match failure) and
``"PENDING"`` (still scoring). Internal job states (queued, preparing,
scoring, retrying, superseded) must never reach this function -- they stay
durable and operator-hidden upstream.

The output is a NEW list of dicts (the input rows are never mutated) with two
render fields added -- ``color`` (amber/red/green/gray/purple for
ERROR/REVIEW/PASS/PENDING/unknown) and ``expand_target`` (the capture id,
wired to the per-slide inspection route) -- stable-sorted ERROR, REVIEW,
PASS, PENDING so anything needing attention sorts first.

A row carrying anything outside those four states (or missing ``verdict``
entirely) degrades PER ROW instead of aborting the whole batch: it gets its
own loud, distinct color (``"purple"``, mirroring the browser's
``--rt-unknown`` token) and a sort rank at/above ERROR so it can never hide
at the bottom of a long table. The raw unexpected value is left on the row
untouched, and a warning is logged naming the capture_id and the value, so an
operator or a log line can see exactly what it was. Earlier revisions of this
projection raised ``UnknownResultStateError`` on the first bad row, which
discarded every OTHER (good) row from the same call -- the sole production
caller, ``kiosk.relay.RelayHandle.state``, builds this list outside its
``TransportError`` guard, so one malformed row used to take down the
operator's entire results table. Silently rendering the row as REVIEW/red
(the #150 behavior #248 replaced) would be just as wrong -- a system failure
disguised as a match failure -- so the fix is a loud per-row fallback, not a
silent default and not a batch-wide exception.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger(__name__)

_COLOR_BY_VERDICT = {
    "ERROR": "amber",
    "REVIEW": "red",
    "PASS": "green",
    "PENDING": "gray",
}

# Stable sort rank: ERROR, REVIEW, PASS, PENDING (#248).
_SORT_RANK_BY_VERDICT = {
    "ERROR": 0,
    "REVIEW": 1,
    "PASS": 2,
    "PENDING": 3,
}

# An unrecognized or missing verdict (#248-fix) is loud, not disguised as
# REVIEW/red and not batch-aborting: a distinct color, never used by the four
# known states, and a rank at/above ERROR's (0) so it can never sort to the
# bottom. Mirrors index.html's RT_UNKNOWN_RANK / rt-unknown.
_UNKNOWN_COLOR = "purple"
_UNKNOWN_SORT_RANK = -1

_VERDICTS_WITH_OVERLAY = frozenset({"PASS", "REVIEW"})

_EVIDENCE_FILENAMES = {
    "block_thumb": "{capture_id}_block_thumb.jpg",
    "slide_thumb": "{capture_id}_slide_thumb.jpg",
    "block_display": "{capture_id}_block_display.jpg",
    "slide_display": "{capture_id}_slide_display.jpg",
    "overlay_display": "{capture_id}_overlay_display.jpg",
}


def evidence_paths_for_capture(
    claim_artifacts_dir: Path | str,
    capture_id: str,
    verdict: str | None,
) -> dict[str, str | None]:
    """Expected claim_artifact paths for one results row (#236 Seam 2).

    PASS/REVIEW rows include all five refs; pending/error/unknown verdicts
    omit ``overlay_display``. Paths are always emitted (no existence check).
    """
    base = Path(claim_artifacts_dir)
    include_overlay = verdict in _VERDICTS_WITH_OVERLAY
    evidence: dict[str, str | None] = {}
    for key, pattern in _EVIDENCE_FILENAMES.items():
        if key == "overlay_display" and not include_overlay:
            evidence[key] = None
        else:
            evidence[key] = str(base / pattern.format(capture_id=capture_id))
    return evidence


def project_results_table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort ERROR, REVIEW, PASS, PENDING (stable within each group) and
    attach render fields.

    A row whose ``verdict`` is not one of the four known states (or is
    missing) degrades to the loud "unknown" fallback instead of raising --
    see module docstring. It still gets ``color``/``expand_target``, a
    warning is logged naming the capture_id and the offending value, and
    every OTHER row in ``rows`` is projected normally.
    """
    projected = []
    for row in rows:
        row = dict(row)
        verdict = row.get("verdict")
        if verdict not in _COLOR_BY_VERDICT:
            log.warning(
                "project_results_table: row capture_id=%r carries an "
                "unrecognized verdict %r (expected one of %s); rendering as "
                "the loud unknown state instead of guessing REVIEW/red",
                row.get("capture_id"), verdict, sorted(_COLOR_BY_VERDICT),
            )
            row["color"] = _UNKNOWN_COLOR
        else:
            row["color"] = _COLOR_BY_VERDICT[verdict]
        row["expand_target"] = row.get("capture_id")
        projected.append(row)
    projected.sort(
        key=lambda row: _SORT_RANK_BY_VERDICT.get(row.get("verdict"), _UNKNOWN_SORT_RANK)
    )
    return projected
