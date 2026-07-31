"""Shared preparation/gate/score composition for block-slide pairs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from verify.gates import GateResult, run_quality_gates
from session.preparation import PreparedResult, PreparedSpecimen, prepare_specimen
from runtime_observer import RuntimeObserver, observed
from verify.scorer import ProductionScoreResult, score_pair_result_routed


ScoreFn = Callable[..., ProductionScoreResult]


@dataclass(frozen=True)
class ComposedPair:
    block_result: PreparedResult
    slide_result: PreparedResult
    gate: GateResult
    score_result: ProductionScoreResult | None

    @property
    def score(self) -> float | None:
        if self.score_result is None:
            return None
        return self.score_result.score


def compose_pair(
    block_path: str | Path,
    slide_path: str | Path,
    *,
    scorer: ScoreFn = score_pair_result_routed,
    score_gated_pairs: bool = False,
    observer: RuntimeObserver | None = None,
    item_id: str = "",
) -> ComposedPair:
    """Prepare, gate, and score a pair from paths.

    Production callers leave score_gated_pairs as False, preserving the
    fail-closed gate-before-score boundary. Diagnostics set it to True so
    gated-but-preparable pairs still have calibration scores.
    """
    block_result = prepare_specimen(block_path, role="block")
    slide_result = prepare_specimen(slide_path, role="slide")
    return compose_prepared_pair(
        block_result,
        slide_result,
        scorer=scorer,
        score_gated_pairs=score_gated_pairs,
        observer=observer,
        item_id=item_id,
    )


def compose_prepared_pair(
    block_result: PreparedResult,
    slide_result: PreparedResult,
    *,
    scorer: ScoreFn = score_pair_result_routed,
    score_gated_pairs: bool = False,
    observer: RuntimeObserver | None = None,
    item_id: str = "",
) -> ComposedPair:
    """Gate and score already-prepared specimens."""
    with observed(observer, "quality_gates", item_id):
        gate = run_quality_gates(block_result, slide_result)
    score_result = None
    should_score = gate.passed or score_gated_pairs
    if (
        should_score
        and isinstance(block_result, PreparedSpecimen)
        and isinstance(slide_result, PreparedSpecimen)
    ):
        score_result = scorer(block_result, slide_result,
                              observer=observer, item_id=item_id)
    return ComposedPair(
        block_result=block_result,
        slide_result=slide_result,
        gate=gate,
        score_result=score_result,
    )
