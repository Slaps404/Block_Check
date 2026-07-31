"""Compact, deliberately non-promotional report for candidate retrieval evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, TypedDict

from candidate_retrieval_analysis import (
    HybridAudit, NestedEvaluation, RecallSummary, VetoCalibration,
)


REQUIRED_SECTIONS = (
    "Corpus and provenance", "Descriptor catalog and timing", "Competitor Recall@k",
    "Architecture comparison", "Outer-fold coverage", "Adaptive candidate band",
    "Hybrid reranking safety audit", "Heuristic veto", "Miss diagnostics", "Recommendation",
)


class VetoFoldResult(TypedDict):
    held_out_order: str
    enabled: bool
    training_enabled: bool
    threshold: float | None
    reason: str
    vetoed_claims: list[str]
    training_false_reviews: list[str]
    heldout_vetoed_slides: list[str]
    heldout_false_reviews: list[str]
    heldout_safe: bool


def render_report(
    *, provenance: Mapping[str, object],
    descriptor_catalog: Sequence[Mapping[str, object]],
    recall_summaries: Sequence[RecallSummary], audit: HybridAudit,
    veto: VetoCalibration, misses: Sequence[Mapping[str, object]],
    recommendation: str,
    nested_evaluation: NestedEvaluation | None = None,
    timing_summary: Mapping[str, object] | None = None,
    subgroup_cuts: (
        Mapping[str, Sequence[Mapping[str, object]]]
        | Sequence[Mapping[str, object]]
    ) = (),
    worst_subgroup_summary: Mapping[str, object] | None = None,
    efficiency_summary: Mapping[str, object] | None = None,
    veto_fold_results: Sequence[VetoFoldResult] = (),
) -> str:
    """Render human QA material. This cannot claim production readiness by design."""
    safe_recommendation = recommendation.replace(
        "not production-ready", "not production-promoted",
    ).replace("production-ready", "not production-promoted")
    lines = [
        "# Candidate Retrieval Proof-of-Concept", "",
        "**Status: not production-promoted.**", "", "## Corpus and provenance",
        str(dict(provenance)), "", "## Descriptor catalog and timing",
        *[str(dict(row)) for row in descriptor_catalog], "", "## Competitor Recall@k",
    ]
    lines.insert(
        lines.index("## Competitor Recall@k"),
        f"Timing summary: {dict(timing_summary or {})}",
    )
    lines.extend(
        f"- {item.method}: k={item.k}, recall={item.recall}, "
        f"misses={list(item.missed_slide_ids)}"
        for item in recall_summaries
    )
    fold_lines = ["No nested held-out evaluation supplied."]
    if nested_evaluation is not None:
        fold_lines = [
            f"- holdout={fold.held_out_order}; selected={fold.selected.name}; "
            f"thresholds={dict(fold.thresholds)}; "
            f"train_coverage={fold.training.coverage}; "
            f"held_out_coverage={fold.held_out.coverage}; "
            f"candidates(mean={fold.held_out.mean_candidate_count}, "
            f"median={fold.held_out.median_candidate_count}, "
            f"p95={fold.held_out.p95_candidate_count}, "
            f"max={fold.held_out.max_candidate_count})"
            for fold in nested_evaluation.folds
        ]
        for fold in nested_evaluation.folds:
            fold_lines.extend(
                f"  - {architecture.kind.value}:{architecture.name}; "
                f"coverage={metric.coverage}; "
                f"mean_candidates={metric.mean_candidate_count}"
                for architecture, metric in fold.architecture_comparison
            )
        if nested_evaluation.insufficient_generalization_warning:
            fold_lines.append(nested_evaluation.insufficient_generalization_warning)
    safety_line = (
        f"New false PASS: {audit.new_false_pass_count}; inherited false PASS: "
        f"{audit.inherited_false_pass_count}; confirmed wrong claims: "
        f"{audit.confirmed_wrong_claim_count}."
        if audit.safety_evaluable
        else "Safety not evaluable: no unambiguous confirmed-wrong claims."
    )
    if isinstance(subgroup_cuts, Mapping):
        subgroup_lines = [
            f"Subgroup {field}: {dict(row)}"
            for field, rows in subgroup_cuts.items()
            for row in rows
        ]
    else:
        subgroup_lines = [f"Subgroup: {dict(row)}" for row in subgroup_cuts]
    subgroup_lines.append(
        f"Worst subgroup: {dict(worst_subgroup_summary or {})}"
    )
    efficiency = dict(efficiency_summary or {})
    veto_fold_lines = [
        f"- holdout={row['held_out_order']}; "
        f"training_enabled={row['training_enabled']}; "
        f"frozen_threshold={row['threshold']}; "
        f"heldout_vetoed={row['heldout_vetoed_slides']}; "
        f"heldout_false_reviews={row['heldout_false_reviews']}; "
        f"heldout_safe={row['heldout_safe']}; enabled={row['enabled']}"
        for row in veto_fold_results
    ]
    if not veto_fold_lines:
        veto_fold_lines = ["No per-fold held-out veto results supplied."]
    lines += [
        "", "## Architecture comparison", "Selection must use nested held-out work orders.",
        "", "## Outer-fold coverage", "Coverage is per individual evaluable slide.",
        *fold_lines,
        "Candidate counts report mean, median, p95, and maximum per fold.",
        *subgroup_lines,
        "", "## Adaptive candidate band",
        "Claim insertion is independent of the heuristic band.", "",
        "## Hybrid reranking safety audit", safety_line,
        f"Verdict parity: {audit.verdict_parity_count}/{len(audit.rows)}; "
        f"reason parity: {audit.reason_parity_count}/{len(audit.rows)}; "
        f"top-block parity: {audit.top_block_parity_count}/{len(audit.rows)}; "
        f"Match Margin drift: {list(audit.match_margin_drifts)}.",
        "Efficiency: "
        f"accurate_rerank_calls={efficiency.get('accurate_rerank_calls')}; "
        f"estimated_runtime={efficiency.get('estimated_runtime')}; "
        f"observed_runtime={efficiency.get('observed_runtime')}; "
        f"full_comparison_reduction={efficiency.get('full_comparison_reduction')}.",
        "", "## Heuristic veto", f"{veto.reason}; enabled={veto.enabled}.",
        *veto_fold_lines, "", "## Miss diagnostics",
        *[str(dict(row)) for row in misses], "", "## Recommendation",
        safe_recommendation, "",
        "This evidence is proof-of-concept only and is not a production promotion.",
    ]
    return "\n".join(lines) + "\n"


def write_report(path: Path, **kwargs: object) -> Path:
    """The only I/O in this module, intentionally kept at the output boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(**kwargs), encoding="utf-8")
    return path
