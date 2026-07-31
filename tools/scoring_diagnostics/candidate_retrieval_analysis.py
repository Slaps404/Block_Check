"""Pure calibration helpers for issue #242 candidate-retrieval diagnostics.

These functions consume cached slide-to-block score mappings.  They deliberately
do not prepare images or score pairs, which keeps threshold/architecture sweeps
from accidentally re-running the expensive matcher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations, product
from math import ceil
from statistics import mean, median
from typing import Iterable, Mapping, Optional, Sequence

from verify.work_order_evaluator import WorkOrderVerdict, evaluate_work_order


@dataclass(frozen=True)
class CompetitorTarget:
    slide_id: str
    claim: str
    block_id: Optional[str]
    score: Optional[float]
    evaluable: bool
    reason: str


@dataclass(frozen=True)
class RecallSummary:
    method: str
    k: int
    evaluable_slides: int
    covered_slides: int
    non_evaluable_slides: int
    target_ranks: tuple[Optional[int], ...]
    missed_slide_ids: tuple[str, ...]

    @property
    def recall(self) -> Optional[float]:
        return None if not self.evaluable_slides else self.covered_slides / self.evaluable_slides


@dataclass(frozen=True)
class CandidateBand:
    members: tuple[str, ...]
    maximum_score: Optional[float]
    threshold: float


@dataclass(frozen=True)
class HybridAuditRow:
    slide_id: str
    claim: str
    baseline: WorkOrderVerdict
    hybrid: WorkOrderVerdict
    candidates: tuple[str, ...]
    strongest_nonclaim: Optional[str]
    confirmed_wrong_claim: Optional[bool]
    new_false_pass: bool
    inherited_false_pass: bool
    missing_competitor: Optional[str]


@dataclass(frozen=True)
class HybridAudit:
    rows: tuple[HybridAuditRow, ...]

    @property
    def new_false_pass_count(self) -> int:
        return sum(row.new_false_pass for row in self.rows)

    @property
    def inherited_false_pass_count(self) -> int:
        return sum(row.inherited_false_pass for row in self.rows)

    @property
    def confirmed_wrong_claim_count(self) -> int:
        return sum(row.confirmed_wrong_claim is True for row in self.rows)

    @property
    def safety_evaluable(self) -> bool:
        """False prevents a vacuous zero-regression safety claim."""
        return self.confirmed_wrong_claim_count > 0

    @property
    def verdict_parity_count(self) -> int:
        return sum(row.baseline.verdict == row.hybrid.verdict for row in self.rows)

    @property
    def reason_parity_count(self) -> int:
        return sum(row.baseline.reason == row.hybrid.reason for row in self.rows)

    @property
    def top_block_parity_count(self) -> int:
        return sum(row.baseline.top_block == row.hybrid.top_block for row in self.rows)

    @property
    def match_margin_drifts(self) -> tuple[Optional[float], ...]:
        return tuple(
            None
            if row.baseline.match_margin is None or row.hybrid.match_margin is None
            else row.hybrid.match_margin - row.baseline.match_margin
            for row in self.rows
        )


class ArchitectureKind(str, Enum):
    INDIVIDUAL = "individual"
    FUSION = "fusion"
    UNION = "union"
    ROUTER = "router"


_ARCHITECTURE_PRIORITY = {
    ArchitectureKind.INDIVIDUAL: 0,
    ArchitectureKind.FUSION: 1,
    ArchitectureKind.UNION: 2,
    ArchitectureKind.ROUTER: 3,
}


@dataclass(frozen=True)
class Architecture:
    """One fixed retrieval shape selected without looking at an outer holdout."""

    kind: ArchitectureKind
    name: str
    methods: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ArchitectureKind(self.kind))


@dataclass(frozen=True)
class BandEvaluation:
    evaluable_slides: int
    covered_slides: int
    candidate_counts: tuple[int, ...]
    missed_slide_ids: tuple[str, ...]

    @property
    def coverage(self) -> Optional[float]:
        return None if not self.evaluable_slides else self.covered_slides / self.evaluable_slides

    @property
    def mean_candidate_count(self) -> float:
        return mean(self.candidate_counts) if self.candidate_counts else 0.0

    @property
    def median_candidate_count(self) -> float:
        return median(self.candidate_counts) if self.candidate_counts else 0.0

    @property
    def p95_candidate_count(self) -> int:
        if not self.candidate_counts:
            return 0
        ordered = sorted(self.candidate_counts)
        return ordered[max(0, ceil(.95 * len(ordered)) - 1)]

    @property
    def max_candidate_count(self) -> int:
        return max(self.candidate_counts, default=0)


@dataclass(frozen=True)
class OuterFoldEvaluation:
    held_out_order: str
    training_orders: tuple[str, ...]
    training_slide_ids: tuple[str, ...]
    held_out_slide_ids: tuple[str, ...]
    architecture_comparison: tuple[tuple[Architecture, BandEvaluation], ...]
    selected: Architecture
    thresholds: tuple[tuple[str, float], ...]
    router_by_slide: tuple[tuple[str, Optional[str]], ...]
    training: BandEvaluation
    held_out: BandEvaluation


@dataclass(frozen=True)
class NestedEvaluation:
    folds: tuple[OuterFoldEvaluation, ...]
    held_out_evaluable_slides: int
    held_out_covered_slides: int
    held_out_candidate_counts: tuple[int, ...]
    insufficient_generalization_warning: Optional[str]

    @property
    def held_out_coverage(self) -> Optional[float]:
        if not self.held_out_evaluable_slides:
            return None
        return self.held_out_covered_slides / self.held_out_evaluable_slides


@dataclass(frozen=True)
class VetoCalibration:
    enabled: bool
    threshold: Optional[float]
    vetoed_claims: tuple[str, ...]
    false_reviews: tuple[str, ...]
    reason: str


def _rank_scored_blocks(
    scores: Mapping[str, Optional[float]], *, exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    excluded = set(exclude)
    return sorted(
        ((str(block), score) for block, score in scores.items()
         if str(block) not in excluded and score is not None),
        key=lambda item: (-item[1], item[0]),
    )


def strongest_nonclaim_competitor(
    slide_id: str, accurate_scores: Mapping[str, Optional[float]], claim: str,
) -> CompetitorTarget:
    """Return the complete matcher’s best scored nonclaim, tie-broken by ID."""
    if claim not in accurate_scores:
        return CompetitorTarget(slide_id, claim, None, None, False, "claimed block absent")
    ranked = _rank_scored_blocks(accurate_scores, exclude=(claim,))
    if not ranked:
        return CompetitorTarget(slide_id, claim, None, None, False, "no scored nonclaim")
    block, score = ranked[0]
    return CompetitorTarget(slide_id, claim, block, score, True, "evaluable")


def nonclaim_ranking(scores: Mapping[str, Optional[float]], claim: str) -> tuple[str, ...]:
    """Higher-is-better deterministic heuristic ranking, excluding the claim."""
    return tuple(block for block, _ in _rank_scored_blocks(scores, exclude=(claim,)))


def recall_at_k(
    method: str,
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    k: int,
) -> RecallSummary:
    """Measure strongest-nonclaim coverage. The separately inserted claim uses no slot."""
    if k < 1:
        raise ValueError("k must be at least one")
    targets = []
    ranks: list[Optional[int]] = []
    misses: list[str] = []
    covered = 0
    for slide_id in sorted(accurate_by_slide):
        if slide_id not in claims:
            raise ValueError(f"missing claim for slide {slide_id!r}")
        target = strongest_nonclaim_competitor(
            slide_id, accurate_by_slide[slide_id], claims[slide_id],
        )
        targets.append(target)
        if not target.evaluable:
            ranks.append(None)
            continue
        ranking = nonclaim_ranking(heuristic_by_slide.get(slide_id, {}), claims[slide_id])
        rank = ranking.index(target.block_id) + 1 if target.block_id in ranking else None
        ranks.append(rank)
        if rank is not None and rank <= k:
            covered += 1
        else:
            misses.append(slide_id)
    evaluable = sum(target.evaluable for target in targets)
    return RecallSummary(method, k, evaluable, covered, len(targets) - evaluable,
                         tuple(ranks), tuple(misses))


def recall_curve(
    method: str,
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_slide: Mapping[str, Mapping[str, Optional[float]]],
) -> tuple[RecallSummary, ...]:
    """Return every meaningful fixed-k ruler, never used as the selected runtime band."""
    widest = max(
        (len(nonclaim_ranking(scores, claims[slide]))
         for slide, scores in heuristic_by_slide.items() if slide in claims),
        default=0,
    )
    return tuple(recall_at_k(method, accurate_by_slide, claims, heuristic_by_slide, k)
                 for k in range(1, widest + 1))


def candidate_union(rankings: Sequence[Sequence[str]], k: int) -> tuple[str, ...]:
    """Union top-k lists, retaining deterministic first-list then ID ordering."""
    if k < 1:
        raise ValueError("k must be at least one")
    return tuple(sorted({block for ranking in rankings for block in ranking[:k]}))


def adaptive_candidate_band(
    scores: Mapping[str, Optional[float]], gap: float, *, claim: Optional[str] = None,
) -> CandidateBand:
    """Select all heuristic candidates within ``gap`` of the maximum score.

    The claim is deliberately excluded here: callers insert it into the accurate
    rerank set independently, so it cannot be pruned or consume a band slot.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")
    ranked = _rank_scored_blocks(scores)
    if not ranked:
        return CandidateBand((), None, gap)
    maximum = ranked[0][1]
    members = tuple(
        block for block, score in ranked
        if block != claim and maximum - score <= gap
    )
    return CandidateBand(members, maximum, gap)


def nested_work_order_folds(
    work_order_by_slide: Mapping[str, str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Outer folds that hold out whole work orders, never random slides."""
    orders = sorted(set(work_order_by_slide.values()))
    if len(orders) < 2:
        return ()
    return tuple(
        (held_out, tuple(order for order in orders if order != held_out))
        for held_out in orders
    )


def architecture_candidates(
    method_names: Iterable[str], *, include_router: bool = False,
    pairwise_methods: Optional[Iterable[str]] = None,
    pairwise_top_n: int = 6,
    include_all_method_baselines: bool = True,
) -> tuple[Architecture, ...]:
    """Declare candidates in the required simple-to-complex evaluation order."""
    methods = tuple(sorted(set(method_names)))
    if not methods:
        raise ValueError("at least one heuristic method is required")
    if pairwise_top_n < 1:
        raise ValueError("pairwise_top_n must be positive")
    candidates = [
        Architecture(ArchitectureKind.INDIVIDUAL, method, (method,))
        for method in methods
    ]
    if len(methods) > 1:
        screened = (
            tuple(sorted(set(pairwise_methods)))
            if pairwise_methods is not None
            else methods[:pairwise_top_n]
        )
        if not set(screened).issubset(methods):
            raise ValueError("pairwise methods must belong to the descriptor catalog")
        # Pairwise complements plus one all-method baseline keep the catalog
        # broad but polynomial when dozens of descriptors are predeclared.
        subsets = list(combinations(screened, 2))
        if include_all_method_baselines and len(methods) > 2:
            subsets.append(methods)
        for subset in subsets:
            suffix = "+".join(subset)
            candidates.extend((
                Architecture(
                    ArchitectureKind.FUSION, f"equal_rank_fusion:{suffix}", subset,
                ),
                Architecture(
                    ArchitectureKind.FUSION,
                    f"equal_normalized_fusion:{suffix}", subset,
                ),
            ))
            # Union gap calibration is a Cartesian product across member score
            # scales. Keep the all-method union only when that sweep is bounded.
            if len(subset) <= pairwise_top_n:
                candidates.append(Architecture(
                    ArchitectureKind.UNION, f"candidate_union:{suffix}", subset,
                ))
        if include_router:
            candidates.append(
                Architecture(ArchitectureKind.ROUTER, "slide_router", screened)
            )
    return tuple(candidates)


def screen_pairwise_methods(
    individual_results: Sequence[tuple[Architecture, BandEvaluation]],
    *, top_n: int = 6, complementary_count: int = 2,
) -> tuple[str, ...]:
    """Bound pair search using training-only recall and complementary misses."""
    if top_n < 1 or complementary_count < 0:
        raise ValueError("screening limits must be non-negative and top_n positive")
    ranked = sorted(
        individual_results,
        key=lambda row: (
            -(row[1].coverage if row[1].coverage is not None else -1.0),
            row[1].mean_candidate_count,
            row[0].name,
        ),
    )
    if not ranked:
        return ()
    selected = [row[0].methods[0] for row in ranked[:top_n]]
    primary_misses = set(ranked[0][1].missed_slide_ids)
    complementary = sorted(
        (
            (len(primary_misses - set(metric.missed_slide_ids)), architecture.name)
            for architecture, metric in ranked[top_n:]
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected.extend(
        method for gain, method in complementary[:complementary_count] if gain > 0
    )
    return tuple(dict.fromkeys(selected))


def _equal_rank_fusion(
    slide_scores: Mapping[str, Mapping[str, Optional[float]]],
    methods: Sequence[str],
) -> dict[str, float]:
    """Fuse deterministic percentile ranks, avoiding incompatible raw scales."""
    blocks = sorted({block for method in methods for block in slide_scores.get(method, {})})
    totals = {block: 0.0 for block in blocks}
    counts = {block: 0 for block in blocks}
    for method in methods:
        ranked = _rank_scored_blocks(slide_scores.get(method, {}))
        denominator = max(1, len(ranked) - 1)
        for index, (block, _score) in enumerate(ranked):
            totals[block] += 1.0 - index / denominator
            counts[block] += 1
    return {
        block: totals[block] / counts[block]
        for block in blocks if counts[block]
    }


def _equal_normalized_fusion(
    slide_scores: Mapping[str, Mapping[str, Optional[float]]],
    methods: Sequence[str],
) -> dict[str, float]:
    """Average per-method min-max scores with fixed equal weights."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for method in methods:
        ranked = _rank_scored_blocks(slide_scores.get(method, {}))
        if not ranked:
            continue
        values = [score for _block, score in ranked]
        low, high = min(values), max(values)
        span = high - low
        for block, score in ranked:
            normalized = 1.0 if span == 0 else (score - low) / span
            totals[block] = totals.get(block, 0.0) + normalized
            counts[block] = counts.get(block, 0) + 1
    return {block: totals[block] / counts[block] for block in sorted(totals)}


def _fusion_scores(
    architecture: Architecture,
    slide_scores: Mapping[str, Mapping[str, Optional[float]]],
) -> dict[str, float]:
    if architecture.name.startswith("equal_rank_fusion"):
        return _equal_rank_fusion(slide_scores, architecture.methods)
    if architecture.name.startswith("equal_normalized_fusion"):
        return _equal_normalized_fusion(slide_scores, architecture.methods)
    raise ValueError(f"unknown fusion architecture {architecture.name!r}")


def _slide_method_scores(
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    slide_id: str,
) -> dict[str, Mapping[str, Optional[float]]]:
    return {
        method: by_slide.get(slide_id, {})
        for method, by_slide in heuristic_by_method.items()
    }


def _fixed_k_members(
    architecture: Architecture,
    slide_scores: Mapping[str, Mapping[str, Optional[float]]],
    claim: str,
    k: int,
    routed_method: Optional[str],
) -> tuple[str, ...]:
    if architecture.kind is ArchitectureKind.INDIVIDUAL:
        return nonclaim_ranking(slide_scores[architecture.methods[0]], claim)[:k]
    if architecture.kind is ArchitectureKind.FUSION:
        return nonclaim_ranking(
            _fusion_scores(architecture, slide_scores), claim,
        )[:k]
    if (architecture.kind is ArchitectureKind.ROUTER
            and routed_method in architecture.methods):
        return nonclaim_ranking(slide_scores[routed_method], claim)[:k]
    rankings = [nonclaim_ranking(slide_scores[method], claim)
                for method in architecture.methods]
    return candidate_union(rankings, k)


def _band_members(
    architecture: Architecture,
    slide_scores: Mapping[str, Mapping[str, Optional[float]]],
    claim: Optional[str],
    thresholds: Mapping[str, float],
    routed_method: Optional[str],
) -> tuple[str, ...]:
    if architecture.kind is ArchitectureKind.INDIVIDUAL:
        method = architecture.methods[0]
        return adaptive_candidate_band(
            slide_scores[method], thresholds[method], claim=claim,
        ).members
    if architecture.kind is ArchitectureKind.FUSION:
        scores = _fusion_scores(architecture, slide_scores)
        return adaptive_candidate_band(
            scores, thresholds["fusion"], claim=claim,
        ).members
    methods = architecture.methods
    if architecture.kind is ArchitectureKind.ROUTER and routed_method in methods:
        methods = (routed_method,)
    members = {
        block
        for method in methods
        for block in adaptive_candidate_band(
            slide_scores[method], thresholds[method], claim=claim,
        ).members
    }
    return tuple(sorted(members))


def _evaluate_members(
    slide_ids: Iterable[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    members_by_slide: Mapping[str, Sequence[str]],
) -> BandEvaluation:
    evaluable = covered = 0
    counts: list[int] = []
    misses: list[str] = []
    for slide_id in sorted(slide_ids):
        target = strongest_nonclaim_competitor(
            slide_id, accurate_by_slide[slide_id], claims[slide_id],
        )
        if not target.evaluable:
            continue
        evaluable += 1
        members = tuple(members_by_slide.get(slide_id, ()))
        counts.append(len(set(members)))
        if target.block_id in members:
            covered += 1
        else:
            misses.append(slide_id)
    return BandEvaluation(evaluable, covered, tuple(counts), tuple(misses))


def compare_architectures(
    slide_ids: Iterable[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
    k: int = 1,
    screening_slide_ids: Optional[Iterable[str]] = None,
    pairwise_top_n: int = 6,
    complementary_method_count: int = 2,
) -> tuple[tuple[Architecture, BandEvaluation], ...]:
    """Compare candidates on fixed-k training evidence before gap calibration."""
    slides = tuple(sorted(slide_ids))
    screening_slides = tuple(sorted(
        slides if screening_slide_ids is None else screening_slide_ids
    ))
    individual_architectures = tuple(
        Architecture(ArchitectureKind.INDIVIDUAL, method, (method,))
        for method in sorted(heuristic_by_method)
    )
    individual_screening = []
    for architecture in individual_architectures:
        method = architecture.methods[0]
        members = {
            slide: nonclaim_ranking(
                heuristic_by_method[method].get(slide, {}), claims[slide],
            )[:k]
            for slide in screening_slides
        }
        individual_screening.append((architecture, _evaluate_members(
            screening_slides, accurate_by_slide, claims, members,
        )))
    screened_methods = screen_pairwise_methods(
        individual_screening, top_n=pairwise_top_n,
        complementary_count=complementary_method_count,
    )
    candidates = architecture_candidates(
        heuristic_by_method, include_router=router_by_slide is not None,
        pairwise_methods=screened_methods, pairwise_top_n=pairwise_top_n,
    )
    rows = []
    for architecture in candidates:
        members = {}
        for slide_id in slides:
            slide_scores = _slide_method_scores(heuristic_by_method, slide_id)
            route = None if router_by_slide is None else router_by_slide.get(slide_id)
            members[slide_id] = _fixed_k_members(
                architecture, slide_scores, claims[slide_id], k, route,
            )
        rows.append((architecture, _evaluate_members(
            slides, accurate_by_slide, claims, members,
        )))
    return tuple(rows)


def fit_slide_group_router(
    training_slide_ids: Iterable[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    router_group_by_slide: Mapping[str, str],
) -> dict[str, Optional[str]]:
    """Fit one descriptor per slide-only group; tied or unseen groups are uncertain."""
    training = tuple(sorted(training_slide_ids))
    groups = sorted({router_group_by_slide[slide] for slide in training
                     if slide in router_group_by_slide})
    choice_by_group: dict[str, Optional[str]] = {}
    for group in groups:
        slides = tuple(
            slide for slide in training if router_group_by_slide.get(slide) == group
        )
        scored = []
        for method, by_slide in sorted(heuristic_by_method.items()):
            summary = recall_at_k(
                method,
                {slide: accurate_by_slide[slide] for slide in slides},
                claims, by_slide, 1,
            )
            scored.append((summary.recall if summary.recall is not None else -1.0,
                           method))
        best = max(score for score, _method in scored)
        winners = [method for score, method in scored if score == best]
        choice_by_group[group] = winners[0] if len(winners) == 1 else None
    return {
        slide: choice_by_group.get(router_group_by_slide.get(slide, ""))
        for slide in router_group_by_slide
    }


def _inner_validated_comparison(
    training_orders: Sequence[str],
    training_slide_ids: Sequence[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    work_order_by_slide: Mapping[str, str],
    router_group_by_slide: Optional[Mapping[str, str]],
    fixed_router_by_slide: Optional[Mapping[str, Optional[str]]],
) -> tuple[tuple[Architecture, BandEvaluation], ...]:
    """Aggregate complete inner-work-order holdouts for architecture selection."""
    if len(training_orders) < 2:
        routes = fixed_router_by_slide
        if router_group_by_slide is not None:
            routes = fit_slide_group_router(
                training_slide_ids, accurate_by_slide, claims,
                heuristic_by_method, router_group_by_slide,
            )
        return compare_architectures(
            training_slide_ids, accurate_by_slide, claims, heuristic_by_method,
            router_by_slide=routes, screening_slide_ids=training_slide_ids,
        )
    accumulated: dict[str, tuple[Architecture, list[BandEvaluation]]] = {}
    for inner_holdout in training_orders:
        inner_train = tuple(
            slide for slide in training_slide_ids
            if work_order_by_slide[slide] != inner_holdout
        )
        inner_test = tuple(
            slide for slide in training_slide_ids
            if work_order_by_slide[slide] == inner_holdout
        )
        routes = fixed_router_by_slide
        if router_group_by_slide is not None:
            routes = fit_slide_group_router(
                inner_train, accurate_by_slide, claims,
                heuristic_by_method, router_group_by_slide,
            )
        for architecture, metric in compare_architectures(
            inner_test, accurate_by_slide, claims, heuristic_by_method,
            router_by_slide=routes, screening_slide_ids=inner_train,
        ):
            accumulated.setdefault(architecture.name, (architecture, []))[1].append(metric)
    rows = []
    for architecture, metrics in accumulated.values():
        if len(metrics) != len(training_orders):
            continue
        rows.append((architecture, BandEvaluation(
            sum(item.evaluable_slides for item in metrics),
            sum(item.covered_slides for item in metrics),
            tuple(count for item in metrics for count in item.candidate_counts),
            tuple(slide for item in metrics for slide in item.missed_slide_ids),
        )))
    return tuple(sorted(
        rows,
        key=lambda row: (_ARCHITECTURE_PRIORITY[row[0].kind], row[0].name),
    ))


def select_architecture(
    comparison: Sequence[tuple[Architecture, BandEvaluation]],
) -> Architecture:
    """Prefer coverage, then cost, then the required simplicity order.

    The router is eligible only when it Pareto-improves both unified fusion and
    union baselines. This makes it a challenger rather than a default.
    """
    if not comparison:
        raise ValueError("architecture comparison cannot be empty")
    eligible = list(comparison)
    router = next(
        (row for row in eligible if row[0].kind is ArchitectureKind.ROUTER), None,
    )

    def best_baseline(kind: ArchitectureKind) -> Optional[
        tuple[Architecture, BandEvaluation]
    ]:
        rows = [row for row in eligible if row[0].kind is kind]
        if not rows:
            return None
        return max(rows, key=lambda row: (
            row[1].coverage if row[1].coverage is not None else -1.0,
            -row[1].mean_candidate_count,
            row[0].name,
        ))

    baselines = (
        best_baseline(ArchitectureKind.FUSION),
        best_baseline(ArchitectureKind.UNION),
    )
    if router is not None:
        router_metric = router[1]
        dominates = all(baseline is not None for baseline in baselines) and all(
            (router_metric.coverage or 0) >= (metric.coverage or 0)
            and router_metric.mean_candidate_count <= metric.mean_candidate_count
            and ((router_metric.coverage or 0) > (metric.coverage or 0)
                 or router_metric.mean_candidate_count < metric.mean_candidate_count)
            for _architecture, metric in baselines if metric is not None
        )
        if not dominates:
            eligible.remove(router)
    return max(
        eligible,
        key=lambda row: (
            row[1].coverage if row[1].coverage is not None else -1.0,
            -row[1].mean_candidate_count,
            -_ARCHITECTURE_PRIORITY[row[0].kind],
            row[0].name,
        ),
    )[0]


def _threshold_keys(architecture: Architecture) -> tuple[str, ...]:
    if architecture.kind is ArchitectureKind.FUSION:
        return ("fusion",)
    return architecture.methods


def calibrate_architecture_thresholds(
    architecture: Architecture,
    training_slide_ids: Iterable[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    gap_grid: Mapping[str, Sequence[float]],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
) -> tuple[tuple[tuple[str, float], ...], BandEvaluation]:
    """Fit score-scale-specific gaps using training slides only."""
    slides = tuple(sorted(training_slide_ids))
    keys = _threshold_keys(architecture)
    grids = []
    for key in keys:
        values = tuple(sorted(set(gap_grid.get(key, ()))))
        if not values or values[0] < 0:
            raise ValueError(f"non-negative gap grid required for {key!r}")
        grids.append(values)
    evaluated = []

    def evaluate_values(values: Sequence[float]) -> tuple[
        tuple[tuple[str, float], ...], BandEvaluation,
    ]:
        thresholds = dict(zip(keys, values))
        members = {}
        for slide_id in slides:
            route = None if router_by_slide is None else router_by_slide.get(slide_id)
            members[slide_id] = _band_members(
                architecture,
                _slide_method_scores(heuristic_by_method, slide_id),
                claims[slide_id], thresholds, route,
            )
        metric = _evaluate_members(slides, accurate_by_slide, claims, members)
        return tuple(sorted(thresholds.items())), metric

    combination_count = 1
    for grid in grids:
        combination_count *= len(grid)
    if combination_count <= 4096:
        evaluated.extend(evaluate_values(values) for values in product(*grids))
    else:
        # Coordinate ascent preserves independent score-scale thresholds while
        # bounding large union/router sweeps to polynomial work.
        indices = [0] * len(grids)
        evaluated.append(evaluate_values([grid[0] for grid in grids]))
        while any(index + 1 < len(grids[position])
                  for position, index in enumerate(indices)):
            neighbors = []
            for position, index in enumerate(indices):
                if index + 1 >= len(grids[position]):
                    continue
                candidate = list(indices)
                candidate[position] += 1
                values = [grids[i][candidate[i]] for i in range(len(grids))]
                neighbors.append((candidate, evaluate_values(values)))
            chosen_indices, chosen_result = max(
                neighbors,
                key=lambda item: (
                    item[1][1].coverage
                    if item[1][1].coverage is not None else -1.0,
                    -item[1][1].mean_candidate_count,
                    -sum(value for _key, value in item[1][0]),
                    tuple(-index for index in item[0]),
                ),
            )
            indices = chosen_indices
            evaluated.append(chosen_result)
    return max(
        evaluated,
        key=lambda row: (
            row[1].coverage if row[1].coverage is not None else -1.0,
            -row[1].mean_candidate_count,
            -sum(value for _key, value in row[0]),
        ),
    )


def evaluate_frozen_architecture(
    architecture: Architecture,
    slide_ids: Iterable[str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    thresholds: Mapping[str, float],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
) -> BandEvaluation:
    """Evaluate a preselected architecture and gaps without fitting anything."""
    slides = tuple(sorted(slide_ids))
    members = {}
    for slide_id in slides:
        route = None if router_by_slide is None else router_by_slide.get(slide_id)
        members[slide_id] = _band_members(
            architecture, _slide_method_scores(heuristic_by_method, slide_id),
            claims[slide_id], thresholds, route,
        )
    return _evaluate_members(slides, accurate_by_slide, claims, members)


def candidate_bands_for_architecture(
    architecture: Architecture,
    slide_ids: Iterable[str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    thresholds: Mapping[str, float],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, tuple[str, ...]]:
    """Build raw bands for safety simulation, before any claim is inserted."""
    bands = {}
    for slide_id in sorted(slide_ids):
        route = None if router_by_slide is None else router_by_slide.get(slide_id)
        bands[slide_id] = _band_members(
            architecture, _slide_method_scores(heuristic_by_method, slide_id),
            None, thresholds, route,
        )
    return bands


def nested_leave_one_work_order_out(
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    work_order_by_slide: Mapping[str, str],
    gap_grid: Mapping[str, Sequence[float]],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
    router_group_by_slide: Optional[Mapping[str, str]] = None,
) -> NestedEvaluation:
    """Select architecture and thresholds on training work orders per outer fold."""
    folds_spec = nested_work_order_folds(work_order_by_slide)
    if not folds_spec:
        return NestedEvaluation(
            (), 0, 0, (),
            "insufficient independent work orders for leave-one-work-order-out",
        )
    order_count = len(set(work_order_by_slide.values()))
    warning = None
    if order_count < 3:
        warning = (
            "insufficient independent work orders for inner work-order "
            "validation; outer results are exploratory"
        )
    folds = []
    for held_out, training_orders in folds_spec:
        training_slides = tuple(sorted(
            slide for slide, order in work_order_by_slide.items()
            if order in training_orders
        ))
        held_out_slides = tuple(sorted(
            slide for slide, order in work_order_by_slide.items()
            if order == held_out
        ))
        comparison = _inner_validated_comparison(
            training_orders, training_slides, accurate_by_slide, claims,
            heuristic_by_method, work_order_by_slide, router_group_by_slide,
            router_by_slide,
        )
        selected = select_architecture(comparison)
        fold_routes = router_by_slide
        if router_group_by_slide is not None:
            fold_routes = fit_slide_group_router(
                training_slides, accurate_by_slide, claims,
                heuristic_by_method, router_group_by_slide,
            )
        thresholds, training_metric = calibrate_architecture_thresholds(
            selected, training_slides, accurate_by_slide, claims,
            heuristic_by_method, gap_grid, router_by_slide=fold_routes,
        )
        held_out_metric = evaluate_frozen_architecture(
            selected, held_out_slides, accurate_by_slide, claims,
            heuristic_by_method, dict(thresholds),
            router_by_slide=fold_routes,
        )
        folds.append(OuterFoldEvaluation(
            held_out, training_orders, training_slides, held_out_slides,
            comparison, selected, thresholds,
            tuple(sorted((fold_routes or {}).items())),
            training_metric, held_out_metric,
        ))
    return NestedEvaluation(
        tuple(folds),
        sum(fold.held_out.evaluable_slides for fold in folds),
        sum(fold.held_out.covered_slides for fold in folds),
        tuple(count for fold in folds for count in fold.held_out.candidate_counts),
        warning,
    )


def hybrid_audit(
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    heuristic_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    gap: float,
    *,
    confirmed_correct_by_slide: Optional[Mapping[str, str]] = None,
    simulate_all_claims: bool = False,
    candidate_members_by_slide: Optional[Mapping[str, Sequence[str]]] = None,
) -> HybridAudit:
    """Compare claim-plus-band reranking with complete evaluator behavior.

    ``confirmed_correct_by_slide`` is optional. Only a present, different value
    labels a claim as confirmed wrong, preventing identity ambiguity from being
    turned into a false-PASS assertion.
    """
    rows = []
    for slide_id in sorted(accurate_by_slide):
        complete = accurate_by_slide[slide_id]
        simulated_claims = (
            tuple(sorted(complete)) if simulate_all_claims else (claims[slide_id],)
        )
        for claim in simulated_claims:
            baseline = evaluate_work_order(complete, claim)
            if candidate_members_by_slide is None:
                band = adaptive_candidate_band(
                    heuristic_by_slide.get(slide_id, {}), gap, claim=claim,
                )
                band_members = band.members
            else:
                band_members = candidate_members_by_slide.get(slide_id, ())
            candidates = tuple(sorted(set(band_members) | {claim}))
            hybrid = evaluate_work_order(
                {block: complete.get(block) for block in candidates}, claim,
            )
            target = strongest_nonclaim_competitor(slide_id, complete, claim)
            correct = (
                None if confirmed_correct_by_slide is None
                else confirmed_correct_by_slide.get(slide_id)
            )
            confirmed_wrong = None if correct is None else correct != claim
            new_false_pass = bool(
                confirmed_wrong is True
                and hybrid.verdict == "PASS"
                and baseline.verdict != "PASS"
            )
            inherited = bool(
                confirmed_wrong is True
                and hybrid.verdict == "PASS"
                and baseline.verdict == "PASS"
            )
            missing = (
                target.block_id
                if target.evaluable and target.block_id not in candidates
                else None
            )
            rows.append(HybridAuditRow(
                slide_id, claim, baseline, hybrid, candidates, target.block_id,
                confirmed_wrong, new_false_pass, inherited, missing,
            ))
    return HybridAudit(tuple(rows))


def calibrate_review_veto(
    heuristic_by_slide: Mapping[str, Mapping[str, Optional[float]]], claims: Mapping[str, str],
    baseline_by_slide: Mapping[str, WorkOrderVerdict],
    confirmed_correct_by_slide: Mapping[str, str],
    thresholds: Iterable[float],
) -> VetoCalibration:
    """Choose the most useful REVIEW-only veto with zero observed false reviews.

    A veto fires when the heuristic winner exceeds the claim by at least its
    threshold. No safe threshold means disabled, rather than weakening rerank.
    """
    candidates: list[VetoCalibration] = []
    for threshold in sorted(set(thresholds)):
        if threshold < 0:
            raise ValueError("veto thresholds must be non-negative")
        vetoed, false_reviews = [], []
        for slide_id, scores in heuristic_by_slide.items():
            claim = claims[slide_id]
            claim_score = scores.get(claim)
            ranked = _rank_scored_blocks(scores)
            if claim_score is None or not ranked or ranked[0][0] == claim:
                continue
            if ranked[0][1] - claim_score >= threshold:
                vetoed.append(slide_id)
                if (confirmed_correct_by_slide.get(slide_id) == claim
                        and baseline_by_slide[slide_id].verdict == "PASS"):
                    false_reviews.append(slide_id)
        if not false_reviews and vetoed:
            candidates.append(
                VetoCalibration(True, threshold, tuple(vetoed), (), "safe REVIEW-only veto")
            )
    if not candidates:
        return VetoCalibration(
            False, None, (), (),
            "disabled: no useful threshold has zero observed false reviews",
        )
    return max(candidates, key=lambda result: (len(result.vetoed_claims), -result.threshold))


def architecture_veto_gaps(
    architecture: Architecture,
    slide_ids: Iterable[str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    claims: Mapping[str, str],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, Optional[float]]:
    """Return conservative claim-to-winner gaps for the selected architecture."""
    gaps = {}
    for slide_id in sorted(slide_ids):
        claim = claims[slide_id]
        slide_scores = _slide_method_scores(heuristic_by_method, slide_id)
        if architecture.kind is ArchitectureKind.FUSION:
            active_scores = (_fusion_scores(architecture, slide_scores),)
        elif architecture.kind is ArchitectureKind.INDIVIDUAL:
            active_scores = (slide_scores[architecture.methods[0]],)
        else:
            route = None if router_by_slide is None else router_by_slide.get(slide_id)
            active_methods = architecture.methods
            if architecture.kind is ArchitectureKind.ROUTER and route in active_methods:
                active_methods = (route,)
            active_scores = tuple(slide_scores[method] for method in active_methods)
        method_gaps = []
        for scores in active_scores:
            claim_score = scores.get(claim)
            ranked = _rank_scored_blocks(scores)
            if claim_score is None or not ranked:
                continue
            method_gaps.append(max(0.0, ranked[0][1] - claim_score))
        # A union/uncertain router vetoes only when every available method agrees.
        gaps[slide_id] = min(method_gaps) if method_gaps else None
    return gaps


def calibrate_architecture_veto(
    architecture: Architecture,
    training_slide_ids: Iterable[str],
    heuristic_by_method: Mapping[
        str, Mapping[str, Mapping[str, Optional[float]]]
    ],
    claims: Mapping[str, str],
    baseline_by_slide: Mapping[str, WorkOrderVerdict],
    confirmed_correct_by_slide: Mapping[str, str],
    thresholds: Iterable[float],
    *, router_by_slide: Optional[Mapping[str, Optional[str]]] = None,
) -> VetoCalibration:
    """Calibrate the optional REVIEW-only veto on one outer-training fold."""
    slides = tuple(sorted(training_slide_ids))
    gaps = architecture_veto_gaps(
        architecture, slides, heuristic_by_method, claims,
        router_by_slide=router_by_slide,
    )
    synthetic_scores = {
        slide: ({claims[slide]: 0.0, "__heuristic_winner__": gap}
                if gap is not None else {claims[slide]: None})
        for slide, gap in gaps.items()
    }
    return calibrate_review_veto(
        synthetic_scores, claims, baseline_by_slide,
        confirmed_correct_by_slide, thresholds,
    )


def audit_rows_as_dicts(audit: HybridAudit) -> list[dict[str, object]]:
    """Small machine-readable adapter retained separately from report rendering."""
    return [asdict(row) for row in audit.rows]


MISS_REASON_CLASSIFICATIONS = frozenset({
    "data_defect", "low_information_mask", "genuine_heuristic_failure",
    "unknown_needs_manual_classification",
})


def hybrid_miss_diagnostics(
    audit: HybridAudit,
    reason_classification_by_case: Optional[
        Mapping[tuple[str, str], str]
    ] = None,
) -> list[dict[str, object]]:
    """Classify every omitted competitor or changed verdict for machine reports."""
    diagnostics = []
    for row in audit.rows:
        if row.missing_competitor is not None:
            event = "strongest_nonclaim_omitted"
        elif row.baseline.verdict != row.hybrid.verdict:
            event = "verdict_changed_without_target_omission"
        else:
            continue
        classification = (
            (reason_classification_by_case or {}).get(
                (row.slide_id, row.claim),
                "unknown_needs_manual_classification",
            )
        )
        if classification not in MISS_REASON_CLASSIFICATIONS:
            raise ValueError(f"invalid miss reason classification {classification!r}")
        diagnostics.append({
            "slide_id": row.slide_id,
            "claim": row.claim,
            "event": event,
            "reason_classification": classification,
            "missing_competitor": row.missing_competitor,
            "baseline_verdict": row.baseline.verdict,
            "hybrid_verdict": row.hybrid.verdict,
            "baseline_reason": row.baseline.reason,
            "hybrid_reason": row.hybrid.reason,
        })
    return diagnostics


def subgroup_band_evaluations(
    evaluation_slide_ids: Iterable[str],
    group_by_slide: Mapping[str, str],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    members_by_slide: Mapping[str, Sequence[str]],
) -> dict[str, BandEvaluation]:
    """Optional metadata cuts, retaining the individual-slide denominator."""
    slides = tuple(evaluation_slide_ids)
    return {
        group: _evaluate_members(
            (slide for slide in slides if group_by_slide.get(slide) == group),
            accurate_by_slide, claims, members_by_slide,
        )
        for group in sorted({group_by_slide[slide] for slide in slides
                             if slide in group_by_slide})
    }


def standard_subgroup_band_evaluations(
    evaluation_slide_ids: Iterable[str],
    metadata_by_slide: Mapping[str, Mapping[str, str]],
    accurate_by_slide: Mapping[str, Mapping[str, Optional[float]]],
    claims: Mapping[str, str],
    members_by_slide: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, BandEvaluation]]:
    """Build the issue's standard optional tissue/morphology/capture cuts."""
    slides = tuple(evaluation_slide_ids)
    fields = ("tissue", "morphology", "sparse_dense", "capture_status")
    return {
        field: subgroup_band_evaluations(
            slides,
            {slide: metadata[field] for slide, metadata in metadata_by_slide.items()
             if field in metadata},
            accurate_by_slide, claims, members_by_slide,
        )
        for field in fields
    }


def worst_subgroup(
    cuts: Mapping[str, Mapping[str, BandEvaluation]],
) -> Optional[tuple[str, str, BandEvaluation]]:
    """Return the lowest-coverage evaluable subgroup with deterministic ties."""
    rows = [
        (field, group, metric)
        for field, groups in cuts.items()
        for group, metric in groups.items()
        if metric.coverage is not None
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: (
        row[2].coverage, row[0], row[1],
    ))
