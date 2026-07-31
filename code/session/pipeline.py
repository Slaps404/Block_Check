"""Production pipeline for v2 claimed-pair verification.

Flow: load manifest → prepare block + slide → gates → score → PASS/REVIEW CSV.

Code map
--------
ClaimDecision
    Per-claim verdict row (score, stage, reason).
process_claim(claim)
    Prepare both sides, run gates, score, decide; returns decision + results.
run_claim_pipeline(manifest_path, ...)   ← CLI/tools entry
    Batch all claims; writes decisions CSV (+ optional contact sheets).
_write_decisions
    Internal CSV writer for decision rows.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from contact_sheet import write_contact_sheet
from session.manifest import load_manifest, ClaimRow
from verify.pair_composition import ScoreFn, compose_prepared_pair
from session.preparation import PreparedResult, PreparationFailure, prepare_specimen_from_image
from runtime_observer import RuntimeObserver
from verify.scorer import decide

VERDICT_PASS = "PASS"
VERDICT_REVIEW = "REVIEW"
DECISION_COLUMNS = [
    "claim_id", "block_path", "slide_path",
    "score", "selected_metric", "router_size_signal",
    "block_occupied_fraction", "slide_occupied_fraction",
    "best_angle", "best_flip", "align_soft_iou", "mask_iou",
    "verdict", "stage", "reason",
]


@dataclass
class ClaimDecision:
    claim_id: str
    block_path: str
    slide_path: str
    verdict: str
    stage: str
    reason: str
    score: float | None = None
    selected_metric: str = ""
    router_size_signal: float | None = None
    block_occupied_fraction: float | None = None
    slide_occupied_fraction: float | None = None
    best_angle: float | None = None
    best_flip: bool | None = None
    align_soft_iou: float | None = None
    mask_iou: float | None = None


def decide_claim(
    claim_id: str,
    block_result: PreparedResult,
    slide_result: PreparedResult,
    *,
    block_path: str = "",
    slide_path: str = "",
    observer: RuntimeObserver | None = None,
    scorer: ScoreFn | None = None,
) -> ClaimDecision:
    """Gate, score, and decide one already-prepared claimed pair.

    Shared by the batch manifest pipeline and the live single-claim workflow
    so both routes run the identical gate-before-score, PASS_THRESHOLD logic.

    `scorer`, when provided, is forwarded to `compose_prepared_pair` in place
    of its default (per-pair-rebuilding) scorer -- e.g. the work-order N^2
    loop injects a cache-lookup closure so per-item normalization is computed
    once (ADR 0011). Leaving it None keeps the live single-claim path on
    `compose_prepared_pair`'s own default, unchanged.
    """
    compose_kwargs = {} if scorer is None else {"scorer": scorer}
    pair = compose_prepared_pair(
        block_result, slide_result, observer=observer, item_id=claim_id,
        **compose_kwargs,
    )

    if not pair.gate.passed:
        return ClaimDecision(
            claim_id=claim_id,
            block_path=block_path,
            slide_path=slide_path,
            verdict=VERDICT_REVIEW,
            stage=pair.gate.stage,
            reason=pair.gate.reason,
        )

    if pair.score is None:
        return ClaimDecision(
            claim_id=claim_id,
            block_path=block_path,
            slide_path=slide_path,
            verdict=VERDICT_REVIEW,
            stage="scoring",
            reason="score unavailable after preparation",
        )

    score_result = pair.score_result
    assert score_result is not None
    score = score_result.score
    verdict, verdict_reason = decide(score)

    return ClaimDecision(
        claim_id=claim_id,
        block_path=block_path,
        slide_path=slide_path,
        verdict=verdict,
        stage="scoring",
        reason=verdict_reason,
        score=score,
        selected_metric=score_result.selected_metric,
        router_size_signal=score_result.router_size_signal,
        block_occupied_fraction=score_result.block_occupied_fraction,
        slide_occupied_fraction=score_result.slide_occupied_fraction,
        best_angle=score_result.best_angle,
        best_flip=score_result.best_flip,
        align_soft_iou=score_result.align_soft_iou,
        mask_iou=score_result.mask_iou,
    )


def process_claim(
    claim: ClaimRow,
) -> tuple[ClaimDecision, PreparedResult, PreparedResult, np.ndarray | None, np.ndarray | None]:
    """Process one claimed pair: prepare → quality gates → score → verdict.

    Returns (decision, block_result, slide_result, block_img, slide_img) so the
    caller can write a contact sheet without re-decoding the source JPEGs.
    """
    block_img = cv2.imread(str(claim.block_path))
    if block_img is None:
        block_result: PreparedResult = PreparationFailure(
            role="block",
            reason=f"could not read image: {claim.block_path}",
        )
    else:
        block_result = prepare_specimen_from_image(block_img, "block")

    slide_img = cv2.imread(str(claim.slide_path))
    if slide_img is None:
        slide_result: PreparedResult = PreparationFailure(
            role="slide",
            reason=f"could not read image: {claim.slide_path}",
        )
    else:
        slide_result = prepare_specimen_from_image(slide_img, "slide")

    decision = decide_claim(
        claim.claim_id, block_result, slide_result,
        block_path=claim.block_path, slide_path=claim.slide_path,
    )
    return decision, block_result, slide_result, block_img, slide_img


def run_claim_pipeline(
    manifest_path: str | Path,
    output_path: str | Path,
    sheets_dir: str | Path | None = None,
) -> List[ClaimDecision]:
    """Run the pipeline: manifest CSV -> decision CSV, optionally writing contact sheets."""
    claims = load_manifest(manifest_path)
    decisions: List[ClaimDecision] = []

    for claim in claims:
        decision, block_result, slide_result, block_img, slide_img = process_claim(claim)
        decisions.append(decision)

        if sheets_dir is not None:
            sheet_path = Path(sheets_dir) / f"{claim.claim_id}_sheet.png"
            write_contact_sheet(
                block_img=block_img,
                slide_img=slide_img,
                block_result=block_result,
                slide_result=slide_result,
                decision=decision,
                output_path=sheet_path,
            )

    _write_decisions(decisions, output_path)
    return decisions


def _write_decisions(decisions: List[ClaimDecision], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        for d in decisions:
            writer.writerow({
                "claim_id": d.claim_id,
                "block_path": d.block_path,
                "slide_path": d.slide_path,
                "score": "" if d.score is None else f"{d.score:.4f}",
                "selected_metric": d.selected_metric,
                "router_size_signal": _format_optional(d.router_size_signal),
                "block_occupied_fraction": _format_optional(d.block_occupied_fraction),
                "slide_occupied_fraction": _format_optional(d.slide_occupied_fraction),
                "best_angle": _format_optional(d.best_angle, digits=1),
                "best_flip": "" if d.best_flip is None else str(d.best_flip),
                "align_soft_iou": _format_optional(d.align_soft_iou),
                "mask_iou": _format_optional(d.mask_iou),
                "verdict": d.verdict,
                "stage": d.stage,
                "reason": d.reason,
            })


def _format_optional(value: float | None, *, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"
