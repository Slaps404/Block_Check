"""Pre-scoring quality gates for v2 claimed-pair verification.

Run before scorer; failed gate → REVIEW (fail-closed, never PASS on uncertainty).

Code map
--------
GateResult
    passed flag + stage + reason.
run_quality_gates(block_result, slide_result)   ← pipeline entry
    Prep failures, mask coverage, block ROI.
check_block_quality(block_result)
    Block-only subset of the above, for judging a block before any slide
    exists to pair it with (#250 Hybrid Candidate Pool usability).
_check_mask_quality
    Per-side mask sanity.
_is_sliver_like
    Archived (ADR 0011, issue #158): thin/low-coverage shape detector, no
    longer wired into run_quality_gates. Kept for reversibility; sliverness
    was a proxy for indiscriminability, which open retrieval detects
    directly. Do not re-wire without revisiting that ADR.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from constants import (
    MIN_MASK_COVERAGE_FRAC,
    PAIR_SLIVER_COVERAGE_FRAC,
    SLIVER_ASPECT_RATIO,
)
from session.preparation import PreparationFailure, PreparedResult

_STAGE_PREP = "preparation"
_STAGE_MASK = "mask_quality"
_STAGE_ROI = "roi"
_STAGE_PASS = "gates"


@dataclass
class GateResult:
    passed: bool
    stage: str
    reason: str


def check_block_quality(block_result: PreparedResult) -> GateResult:
    """Judge a block's own usability, independent of any slide (#250).

    Finish Blocks must decide whether a Hybrid block is usable before any
    slide exists to pair it with, so the block-side checks `run_quality_gates`
    normally runs alongside a slide are also exposed standalone here. This is
    additive: `run_quality_gates`'s own inline block checks (and its existing,
    tested block-vs-slide failure-ordering) are untouched by this function.
    """
    if isinstance(block_result, PreparationFailure):
        return GateResult(
            passed=False,
            stage=_STAGE_PREP,
            reason=f"block preparation failed: {block_result.reason}",
        )
    gate = _check_mask_quality("block", block_result.mask)
    if gate is not None:
        return gate
    if not block_result.roi_ok:
        return GateResult(
            passed=False,
            stage=_STAGE_ROI,
            reason=f"block ROI failed: {block_result.roi_reason}",
        )
    return GateResult(passed=True, stage=_STAGE_PASS, reason="block quality gates passed")


def run_quality_gates(
    block_result: PreparedResult,
    slide_result: PreparedResult,
) -> GateResult:
    """Run all pre-scoring quality gates in order. Return on first failure."""
    if isinstance(block_result, PreparationFailure):
        return GateResult(
            passed=False,
            stage=_STAGE_PREP,
            reason=f"block preparation failed: {block_result.reason}",
        )
    if isinstance(slide_result, PreparationFailure):
        return GateResult(
            passed=False,
            stage=_STAGE_PREP,
            reason=f"slide preparation failed: {slide_result.reason}",
        )

    for label, specimen in (("block", block_result), ("slide", slide_result)):
        gate = _check_mask_quality(label, specimen.mask)
        if gate is not None:
            return gate

    if not block_result.roi_ok:
        return GateResult(
            passed=False,
            stage=_STAGE_ROI,
            reason=f"block ROI failed: {block_result.roi_reason}",
        )

    return GateResult(passed=True, stage=_STAGE_PASS, reason="all gates passed")


def _check_mask_quality(label: str, mask: np.ndarray) -> GateResult | None:
    """Return a failed GateResult if the mask is non-evaluable, else None."""
    nonzero = int(np.count_nonzero(mask))
    if nonzero == 0:
        return GateResult(
            passed=False,
            stage=_STAGE_MASK,
            reason=f"{label} mask is empty",
        )
    coverage = nonzero / mask.size
    if coverage < MIN_MASK_COVERAGE_FRAC:
        return GateResult(
            passed=False,
            stage=_STAGE_MASK,
            reason=f"{label} mask too sparse: {coverage:.4%} coverage",
        )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return GateResult(
            passed=False,
            stage=_STAGE_MASK,
            reason=f"{label} mask has no contours",
        )
    return None


def _is_sliver_like(mask: np.ndarray) -> bool:
    """Return True when the mask has only thin, low-coverage visual evidence."""
    coverage = np.count_nonzero(mask) / mask.size
    if coverage >= PAIR_SLIVER_COVERAGE_FRAC:
        return False
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    _, _, width, height = cv2.boundingRect(largest)
    aspect = max(width, height) / max(min(width, height), 1)
    return aspect >= SLIVER_ASPECT_RATIO
