"""Pure open-retrieval work-order verdict evaluator.

Given a work order's claimed block-slide pairing and the scorer's per-block
candidate scores for one slide, decide PASS/REVIEW using a match-margin
comparison against the runner-up block rather than an absolute threshold
alone. See docs/adr/0009-workorder-scoped-n2-retrieval-mode.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Mapping, Optional, TypedDict

from constants import MATCH_MARGIN, PASS_THRESHOLD


class FlaggedPair(TypedDict):
    block_id: str
    role: str


@dataclass(frozen=True)
class WorkOrderVerdict:
    verdict: str
    reason: str
    match_margin: Optional[float]
    top_block: Optional[str]
    near_miss_blocks: FrozenSet[str] = field(default_factory=frozenset)


def evaluate_work_order(
    candidate_scores: Mapping[str, Optional[float]],
    claimed_block: str,
    *,
    match_margin: float = MATCH_MARGIN,
    pass_threshold: float = PASS_THRESHOLD,
) -> WorkOrderVerdict:
    """Decide PASS/REVIEW for a claimed block against this order's candidates.

    ``candidate_scores`` maps block_id -> score for every block scanned in
    this work order. A missing key means the block was not scanned in this
    order; a ``None`` value means the pair could not be prepared for scoring.
    """
    if claimed_block not in candidate_scores:
        return WorkOrderVerdict(
            verdict="REVIEW",
            reason=f"Claimed block {claimed_block} not in this order.",
            match_margin=None,
            top_block=None,
            near_miss_blocks=frozenset(),
        )

    if candidate_scores[claimed_block] is None:
        return WorkOrderVerdict(
            verdict="REVIEW",
            reason="Preparation failed.",
            match_margin=None,
            top_block=None,
            near_miss_blocks=frozenset(),
        )

    ranked = sorted(
        (
            (block, score)
            for block, score in candidate_scores.items()
            if score is not None
        ),
        key=lambda kv: (-kv[1], kv[0]),
    )

    if len(ranked) == 1:
        block, score = ranked[0]
        verdict = "PASS" if score >= pass_threshold else "REVIEW"
        return WorkOrderVerdict(
            verdict=verdict,
            reason="unverified - threshold only",
            match_margin=None,
            top_block=block,
            near_miss_blocks=frozenset(),
        )

    top_block, top_score = ranked[0]
    near_miss_blocks = frozenset(
        block
        for block, score in ranked[1:]
        if (top_score - score) < match_margin
    )

    runner_up_score = ranked[1][1]
    computed_margin = top_score - runner_up_score

    if top_block == claimed_block:
        if not near_miss_blocks:
            return WorkOrderVerdict(
                verdict="PASS",
                reason="Claimed block clearly scored highest.",
                match_margin=computed_margin,
                top_block=top_block,
                near_miss_blocks=near_miss_blocks,
            )
        return WorkOrderVerdict(
            verdict="REVIEW",
            reason=(
                f"Near miss with block {sorted(near_miss_blocks)[0]}. Review both."
            ),
            match_margin=computed_margin,
            top_block=top_block,
            near_miss_blocks=near_miss_blocks,
        )

    return WorkOrderVerdict(
        verdict="REVIEW",
        reason=f"{top_block} scored higher than claimed block {claimed_block}.",
        match_margin=computed_margin,
        top_block=top_block,
        near_miss_blocks=near_miss_blocks,
    )


def flagged_pairs(claimed_block: str, verdict: WorkOrderVerdict) -> List[FlaggedPair]:
    """Return the claim and strongest alternative for a review contact sheet."""
    pairs: List[FlaggedPair] = []
    if verdict.top_block is not None:
        pairs.append({"block_id": verdict.top_block, "role": "TOP MATCH"})
    if claimed_block != verdict.top_block:
        pairs.append({"block_id": claimed_block, "role": "CLAIMED"})
    return pairs
