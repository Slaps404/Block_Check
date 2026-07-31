"""Heuristic Candidate Band: cheap, alignment-free pool ranking (issue #253).

Ranks a frozen Hybrid Candidate Pool's blocks against one slide using only
cheap, alignment-free descriptor comparisons (`verify.invariant_descriptors
.compare_descriptor_values`), keeps every block within a shape-class-specific
gap of the best fused rank as the Heuristic Candidate Band, and always hands
the caller a structural record of what happened -- never a bare list of ids.

Pure data in, frozen-dataclass out: no cv2/numpy image work, no I/O, no store
access, no logging. Callers build one `SpecimenFingerprint` per block/slide
from data already computed on the production scoring path (`build_locked
_score_cache`'s `normalized_mask.mean()` for `occupied_fraction`, `verify
.invariant_descriptors.build_descriptor_values` for `descriptor_values`) --
this module never builds a mask or a descriptor itself.

Recall target -- read before wiring this into anything
----------------------------------------------------------------------------
The barcode-claimed block is ALWAYS inserted into accurate scoring separately
from the band (CONTEXT.md "Out-of-Pool Claim"/"Hybrid Reranking"; #245/#253):
its recall is guaranteed by construction here, via `CandidateSelection
.accurate_scoring_ids` (`candidate_ids` unioned with `claim_id`), not by a
caller convention a later edit could quietly drop. `claim_id` never appears
inside `candidate_ids`.

What the band width actually protects is the STRONGEST NON-CLAIM COMPETITOR:
`verify.work_order_evaluator.evaluate_work_order` only inspects the top-2
scores it is given, gated by `MATCH_MARGIN`. If the band omits the block that
would have been the true runner-up, the evaluator sees an artificially weak
field and a genuine near-miss REVIEW silently becomes a PASS. Band width and
the floor below exist to protect that competitor, not the claim.

Fallback, structurally
----------------------------------------------------------------------------
`evaluate_work_order` treats a missing block_id as "not scanned in this
order," not as an error -- so a bare list of candidate ids can never tell a
caller "correctly pruned" apart from "buggy or incomplete." `CandidateSelection
.pruned_ids` records exactly which pool blocks were examined and not
selected; `fallback_required`/`fallback_reason` record when selection could
not be performed at all (missing descriptor evidence, or a mask-quality
signal a caller already computed via `verify.gates`), in which case
`candidate_ids`/`pruned_ids` are both empty and the caller must fall back to
complete accurate scoring of the whole pool for that slide -- this module
never invents partial evidence to avoid an honest fallback.
`select_candidate_band`'s docstring below is the source of truth for exactly
when each fallback fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence

from constants import SHAPE_ROUTER_SIZE_THRESHOLD
from verify.invariant_descriptors import (
    DescriptorSpec,
    DescriptorValue,
    compare_descriptor_values,
    descriptor_catalog,
)

# ---------------------------------------------------------------------------
# UNCALIBRATED HYPOTHESIS -- every weight, weight set, gap fraction, and floor
# below is a placeholder. No real calibration data exists: issue #242's only
# committed artifact is a 2-block/2-slide smoke test
# (`outputs/issue242_qa_smoke/report.md`) whose recall@1 = 1.0 is meaningless
# with one possible competitor. What would validate these: a real #242
# retrieval proof of concept run across many multi-block work orders,
# measuring Heuristic Competitor Coverage (CONTEXT.md) for the strongest
# non-claim competitor specifically, not overall recall. Tune by editing this
# block only -- every public function below takes these as keyword defaults,
# never as inline literals, so tuning never requires touching logic.
# ---------------------------------------------------------------------------

SIZE_SIGNAL_WEIGHT = 0.20
"""Standalone weight for the occupied-fraction size_signal ranking term.

Today `score_routed_caches` (verify/scorer.py) computes this same value
(`min(block_fraction, slide_fraction)`) only to gate mask_iou vs point_layout
for one already-claimed pair, and `global_morphology_v1` dilutes an
occupied-fraction dimension to one-in-ten alongside eccentricity/solidity.
Lab mechanism-label evidence (vault) shows it separates all 41 corpus pairs by
a 4.7x empty gap (0.0058-0.0236 sparse vs 0.1106-0.3602 dense) -- the
best-validated cheap axis in this codebase -- which justifies pulling it out
as its own weighted rank-fusion term instead of leaving it buried. Weighted
above every individual descriptor in both sets below (dense top descriptor is
0.35, sparse top is 0.30) to reflect that strength, but kept well under 1.0
so it cannot alone dominate the fused ranking the way a raw weighted sum
would let a wide-dynamic-range descriptor dominate regardless of its
assigned weight -- rank fusion is scale-free by construction (see
`_borda_rank_scores`), so this weight only ever competes on equal footing
with every other term's Borda rank score.
"""

DENSE_DESCRIPTOR_WEIGHTS: Mapping[str, float] = {
    "global_morphology_v1": 0.35,
    "radial_foreground_histogram_v1": 0.30,
    "boundary_radius_histogram_v1": 0.15,
    "distance_transform_histogram_v1": 0.10,
    "fourier_radial_power_v1": 0.05,
    "hu_absolute_moments_v1": 0.05,
    # component_* descriptors are explicit zeros, not omissions: dense lobes
    # merge on the slide but stay separate on the block, so component-based
    # descriptors mislead exactly where mask_iou (dense's accurate metric)
    # tolerates the difference. `test_candidate_band.py` asserts perturbing
    # one genuinely does not move the dense ranking.
    "component_radial_histogram_v1": 0.0,
    "component_distance_histogram_v1": 0.0,
    "component_area_histogram_v1": 0.0,
}
"""Dense (mask_iou-routed) rank-fusion weights: rank by coarse silhouette."""

SPARSE_DESCRIPTOR_WEIGHTS: Mapping[str, float] = {
    "component_radial_histogram_v1": 0.30,
    "component_distance_histogram_v1": 0.25,
    "component_area_histogram_v1": 0.20,
    "global_morphology_v1": 0.15,
    "radial_foreground_histogram_v1": 0.05,
    # Remaining descriptors share 0.05 combined, per the approved design.
    "boundary_radius_histogram_v1": 0.0125,
    "distance_transform_histogram_v1": 0.0125,
    "fourier_radial_power_v1": 0.0125,
    "hu_absolute_moments_v1": 0.0125,
}
"""Sparse (point_layout-routed) rank-fusion weights: rank by component
constellation. Lab mechanism-label evidence (vault) records zero orphan
fragments across the 7 esophagus true pairs, so component descriptors are
reliable precisely where sparse routes."""

DENSE_GAP_FRACTION = 0.40
SPARSE_GAP_FRACTION = 0.65
"""Relative score-gap fraction of the observed fused-score range, per shape
class. Sparse is wider: lab mechanism-label evidence (vault) shows sparse/
esophagus accurate-metric margins are the thinnest in the corpus (set 018:
-0.0048, 014: +0.0022, 019: +0.09) versus dense mask_iou margins mostly +0.18
to +0.34, so an uncertain sparse heuristic must widen its band further to
avoid pruning a genuine near-tie competitor."""

DENSE_FLOOR_FRACTION = 0.25
DENSE_FLOOR_MINIMUM = 3
SPARSE_FLOOR_FRACTION = 0.40
SPARSE_FLOOR_MINIMUM = 4
"""Minimum non-claim candidate count, per shape class: `max(minimum, ceil(
fraction * pool_size))`. Independent of the gap -- protects the strongest
non-claim competitor even when the fused-score range is small enough that the
gap alone would keep almost nothing."""


class ShapeClass(str, Enum):
    """Which rank-fusion weight set and band parameters apply to one slide."""

    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True)
class SpecimenFingerprint:
    """One block or slide's pure, pair-independent retrieval fingerprint.

    Built once per specimen (at Hybrid Candidate Pool freeze for a block, at
    slide accept for a slide) from data the production scoring path already
    computes: `occupied_fraction` is the same value `score_routed_caches`
    calls `block_fraction`/`slide_fraction` (`normalized_mask.mean()`), and
    `descriptor_values` is `verify.invariant_descriptors
    .build_descriptor_values(normalized_mask)`'s return value. This module
    never builds either itself -- no cv2, no mask I/O.
    """

    specimen_id: str
    occupied_fraction: float
    descriptor_values: Mapping[str, DescriptorValue]


@dataclass(frozen=True)
class ScoreBand:
    """Faithful port of `tools/scoring_diagnostics/candidate_retrieval_analysis
    .adaptive_candidate_band`'s return shape (issue #246 precedent: relocate
    a diagnostics-proven pure function into `code/verify`, do not import it --
    `tests/test_architecture_boundaries.py` forbids `code/` importing `tools`,
    including by the bare module name `tests/conftest.py` puts on `sys.path`).
    """

    members: tuple[str, ...]
    maximum_score: Optional[float]
    threshold: float


@dataclass(frozen=True)
class CandidateSelection:
    """Everything a caller needs to know about one slide's Heuristic
    Candidate Band -- never a bare list of ids.

    `candidate_ids` never includes `claim_id`: use `accurate_scoring_ids`
    (candidates plus the always-separately-inserted claim) as the exact set
    to hand to accurate gates/scoring. `pruned_ids` is every other pool block
    that was examined and not selected -- explicit, so a caller can never
    confuse "correctly pruned this run" with "never scanned in this work
    order" (a bare missing key in `evaluate_work_order`'s input means the
    latter, and looks identical to the former unless recorded here).

    When `fallback_required` is True, candidate selection could not be
    performed at all: `candidate_ids` and `pruned_ids` are both empty,
    `fallback_reason` names why, and the caller must fall back to complete
    accurate scoring of the whole pool for this slide
    (`code/session/hybrid_configuration.REQUIRED_FALLBACK_IDS`'s
    `"complete_accurate_scoring_on_missing_fingerprint"`). `shape_class` may
    still be populated even on a fallback (routing only needs the slide's own
    occupied fraction, which is never what's missing when a *descriptor* is
    missing) -- `gap_threshold`/`floor_count`/`weight_set_name` are not, since
    they are never computed on a fallback path.
    """

    claim_id: str
    candidate_ids: tuple[str, ...]
    pruned_ids: tuple[str, ...]
    shape_class: Optional[ShapeClass]
    gap_threshold: Optional[float]
    floor_count: Optional[int]
    weight_set_name: Optional[str]
    fallback_required: bool
    fallback_reason: Optional[str]
    claim_inserted_separately: bool = field(default=True)

    @property
    def accurate_scoring_ids(self) -> tuple[str, ...]:
        """`candidate_ids` plus the always-separately-inserted claim.

        This is the exact set a caller must hand to accurate gates/scoring --
        the claim's guaranteed recall is structural only if callers use this
        property (or replicate its union) instead of scoring `candidate_ids`
        alone.
        """
        return tuple(sorted({self.claim_id, *self.candidate_ids}))


def route_shape_class(
    slide_occupied_fraction: float,
    *,
    threshold: float = SHAPE_ROUTER_SIZE_THRESHOLD,
) -> ShapeClass:
    """Route to dense/sparse rank-fusion weights, reusing the production
    shape router (`verify.scorer.score_routed_caches`'s
    `SHAPE_ROUTER_SIZE_THRESHOLD` gate) rather than inventing a new one.

    `score_routed_caches` computes `size_signal = min(block_fraction,
    slide_fraction)` per already-claimed PAIR to choose mask_iou (dense) vs
    point_layout (sparse). Candidate retrieval ranks one slide against an
    entire pool before any pair is chosen, so there is no single "the block"
    fraction to take a min against yet -- and CONTEXT.md's "Heuristic Router"
    is explicitly a slide-level choice, not a pair-level one
    (`_Avoid_: pair-level routing that mixes uncalibrated score scales`).
    Lab mechanism-label evidence (vault) shows the dense/sparse split is clean
    on EACH side independently, not only the min: esophagus (sparse) sits
    0.0058-0.0236 on both block and slide sides; every other tissue (dense)
    sits 0.1106-0.3602 on both sides. Routing on the slide's own occupied
    fraction alone reuses the identical constant and reproduces the same split
    without a new router or a new threshold.
    """
    return (
        ShapeClass.DENSE
        if slide_occupied_fraction >= threshold
        else ShapeClass.SPARSE
    )


def adaptive_score_band(
    scores: Mapping[str, Optional[float]],
    gap: float,
    *,
    claim: Optional[str] = None,
) -> ScoreBand:
    """Select all candidates within ``gap`` of the maximum score.

    Faithful port of `tools/scoring_diagnostics/candidate_retrieval_analysis
    .adaptive_candidate_band` -- same behavior, unchanged. The claim is
    deliberately excluded here: callers insert it into the accurate rerank
    set independently (`CandidateSelection.accurate_scoring_ids`), so it can
    never be pruned or consume a band slot.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")
    ranked = sorted(
        ((str(block), score) for block, score in scores.items() if score is not None),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return ScoreBand((), None, gap)
    maximum = ranked[0][1]
    members = tuple(
        block for block, score in ranked if block != claim and maximum - score <= gap
    )
    return ScoreBand(members, maximum, gap)


def validate_selection(selection: CandidateSelection) -> Optional[str]:
    """Structural self-check a caller should run before trusting a selection.

    Returns ``None`` when internally consistent, else a reason a caller must
    treat as an invalid candidate-selection result
    (`code/session/hybrid_configuration.REQUIRED_FALLBACK_IDS`'s
    `"complete_accurate_scoring_on_invalid_candidate_output"`) and use to
    fall back to complete accurate scoring of the whole pool for this slide,
    rather than trusting a `CandidateSelection` this module never actually
    produces in normal operation but a future edit could.
    """
    if selection.fallback_required:
        return None
    if selection.claim_id in selection.candidate_ids:
        return "claim_id must never appear inside candidate_ids"
    if set(selection.candidate_ids) & set(selection.pruned_ids):
        return "candidate_ids and pruned_ids must be disjoint"
    if selection.claim_id not in selection.accurate_scoring_ids:
        return "claim_id missing from accurate_scoring_ids"
    if not selection.claim_inserted_separately:
        return "claim must be flagged as separately inserted"
    if selection.gap_threshold is None or selection.gap_threshold < 0:
        return "gap_threshold must be a non-negative number"
    if selection.floor_count is None or selection.floor_count < 0:
        return "floor_count must be a non-negative number"
    return None


_DESCRIPTOR_SPECS_BY_NAME: Mapping[str, DescriptorSpec] = {
    spec.name: spec for spec in descriptor_catalog()
}


def _required_descriptor_names(weights: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(name for name, weight in weights.items() if weight > 0))


def _missing_descriptors(
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    slide_fingerprint: SpecimenFingerprint,
    required_descriptors: Sequence[str],
) -> tuple[str, ...]:
    missing: set[str] = set()
    for name in required_descriptors:
        if name not in slide_fingerprint.descriptor_values:
            missing.add(name)
            continue
        for fingerprint in pool_fingerprints.values():
            if name not in fingerprint.descriptor_values:
                missing.add(name)
    return tuple(sorted(missing))


def _descriptor_raw_scores(
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    slide_fingerprint: SpecimenFingerprint,
    descriptor_name: str,
) -> dict[str, float]:
    spec = _DESCRIPTOR_SPECS_BY_NAME[descriptor_name]
    slide_value = slide_fingerprint.descriptor_values[descriptor_name]
    return {
        block_id: compare_descriptor_values(
            spec, fingerprint.descriptor_values[descriptor_name], slide_value
        )
        for block_id, fingerprint in pool_fingerprints.items()
    }


def _size_signal_raw_scores(
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    slide_fingerprint: SpecimenFingerprint,
) -> dict[str, float]:
    """Same comparison shape as an `exp_l1` descriptor, over a 1-dim value.

    Not part of the Heuristic Descriptor Catalog (`known_descriptor_names`
    only covers `build_descriptor_values`'s mask-shaped vectors): occupied
    fraction is a scalar the scorer already derives directly from the
    normalized mask, not a cacheable catalog descriptor, so it is compared
    here rather than routed through `compare_descriptor_values`.
    """
    slide_fraction = slide_fingerprint.occupied_fraction
    return {
        block_id: float(math.exp(-abs(fingerprint.occupied_fraction - slide_fraction)))
        for block_id, fingerprint in pool_fingerprints.items()
    }


def _borda_rank_scores(raw_scores: Mapping[str, float]) -> dict[str, float]:
    """Convert one descriptor's raw comparator scores into scale-free ranks.

    Weighted Borda over per-descriptor rankings, not `min()` and not a raw
    weighted sum of the comparator scores themselves: comparators have
    different dynamic ranges inside [0, 1] (`histogram_intersection` on 16
    bins vs `exp_l1` on a 7-dim Hu vector), so a raw weighted sum would let
    the widest-spread descriptor dominate regardless of its assigned weight.
    Converting to a rank first, before any weight is applied, is what makes
    the fusion scale-free: any order-preserving rescaling of one descriptor's
    raw scores produces the identical rank scores here.

    Ties (including the common case of every block scoring identically on an
    irrelevant descriptor) break by block id, matching `_rank_scored_blocks`
    in the ported diagnostics precedent.
    """
    ranked = sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))
    denominator = max(1, len(ranked) - 1)
    return {
        block: 1.0 - index / denominator for index, (block, _score) in enumerate(ranked)
    }


def _fuse_rank_scores(
    per_term_rank_scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    total_weight = sum(weights.get(name, 0.0) for name in per_term_rank_scores)
    if total_weight <= 0:
        raise ValueError("rank fusion requires at least one positive-weight term")
    block_ids: set[str] = set()
    for scores in per_term_rank_scores.values():
        block_ids.update(scores)
    return {
        block: sum(
            weights.get(name, 0.0) * per_term_rank_scores[name].get(block, 0.0)
            for name in per_term_rank_scores
        )
        / total_weight
        for block in block_ids
    }


def _floor_extended_members(
    ranked_non_claim: Sequence[tuple[str, float]],
    gap_band_members: tuple[str, ...],
    floor_count: int,
) -> tuple[str, ...]:
    """Extend the gap band up to ``floor_count`` members, never past it.

    `ranked_non_claim` is sorted descending by fused score, so the gap band
    (everything within the gap of the max) is always a prefix of it; widening
    to the floor is therefore just taking a longer prefix.
    """
    target = max(len(gap_band_members), min(floor_count, len(ranked_non_claim)))
    return tuple(block for block, _score in ranked_non_claim[:target])


def select_candidate_band(
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    slide_fingerprint: SpecimenFingerprint,
    claim_id: str,
    *,
    dense_descriptor_weights: Mapping[str, float] = DENSE_DESCRIPTOR_WEIGHTS,
    sparse_descriptor_weights: Mapping[str, float] = SPARSE_DESCRIPTOR_WEIGHTS,
    size_signal_weight: float = SIZE_SIGNAL_WEIGHT,
    shape_router_threshold: float = SHAPE_ROUTER_SIZE_THRESHOLD,
    dense_gap_fraction: float = DENSE_GAP_FRACTION,
    sparse_gap_fraction: float = SPARSE_GAP_FRACTION,
    dense_floor_fraction: float = DENSE_FLOOR_FRACTION,
    dense_floor_minimum: int = DENSE_FLOOR_MINIMUM,
    sparse_floor_fraction: float = SPARSE_FLOOR_FRACTION,
    sparse_floor_minimum: int = SPARSE_FLOOR_MINIMUM,
    mask_quality_fallback_reason: Optional[str] = None,
) -> CandidateSelection:
    """Rank a frozen pool against one slide and select the Heuristic
    Candidate Band, or return an explicit fallback reason.

    Fallback fires (candidate_ids/pruned_ids both empty) when:

    - ``mask_quality_fallback_reason`` is given: a caller already ran
      `verify.gates` and found the slide or a pool block's mask untrustworthy
      (e.g. faint-paraffin colon tail, blue-cast brain -- segmentation
      defects the band's gap/floor cannot compensate for, since the
      heuristic would be fed a wrong mask). This module does not detect mask
      quality itself; it only exposes the reason the caller supplies.
    - a descriptor the routed shape class's weight set requires is missing
      from the slide's fingerprint or from any pool block's fingerprint.

    Otherwise, weighted Borda rank fusion (see `_borda_rank_scores`,
    `_fuse_rank_scores`) selects the routed shape class's descriptors plus
    the standalone `size_signal` term, then `adaptive_score_band` plus a
    per-shape-class floor (`_floor_extended_members`) selects the band.
    ``claim_id`` must be a member of ``pool_fingerprints`` -- an out-of-pool
    claim is rejected upstream by identity mismatch alone (CONTEXT.md
    "Out-of-Pool Claim"), never reaches candidate selection.
    """
    if mask_quality_fallback_reason is not None:
        return CandidateSelection(
            claim_id=claim_id,
            candidate_ids=(),
            pruned_ids=(),
            shape_class=None,
            gap_threshold=None,
            floor_count=None,
            weight_set_name=None,
            fallback_required=True,
            fallback_reason=mask_quality_fallback_reason,
        )
    if claim_id not in pool_fingerprints:
        raise ValueError(
            f"claim {claim_id!r} must be a member of the frozen pool being ranked"
        )

    pool_size = len(pool_fingerprints)
    shape_class = route_shape_class(
        slide_fingerprint.occupied_fraction, threshold=shape_router_threshold
    )
    if shape_class is ShapeClass.DENSE:
        descriptor_weights = dense_descriptor_weights
        gap_fraction = dense_gap_fraction
        floor_fraction, floor_minimum = dense_floor_fraction, dense_floor_minimum
        weight_set_name = "dense"
    else:
        descriptor_weights = sparse_descriptor_weights
        gap_fraction = sparse_gap_fraction
        floor_fraction, floor_minimum = sparse_floor_fraction, sparse_floor_minimum
        weight_set_name = "sparse"

    required_descriptors = _required_descriptor_names(descriptor_weights)
    missing = _missing_descriptors(pool_fingerprints, slide_fingerprint, required_descriptors)
    if missing:
        return CandidateSelection(
            claim_id=claim_id,
            candidate_ids=(),
            pruned_ids=(),
            shape_class=shape_class,
            gap_threshold=None,
            floor_count=None,
            weight_set_name=weight_set_name,
            fallback_required=True,
            fallback_reason=(
                f"missing descriptor(s) required for {weight_set_name} ranking: "
                f"{', '.join(missing)}"
            ),
        )

    per_term_rank_scores = {
        name: _borda_rank_scores(
            _descriptor_raw_scores(pool_fingerprints, slide_fingerprint, name)
        )
        for name in required_descriptors
    }
    per_term_rank_scores["__size_signal__"] = _borda_rank_scores(
        _size_signal_raw_scores(pool_fingerprints, slide_fingerprint)
    )
    weights = {**descriptor_weights, "__size_signal__": size_signal_weight}
    fused = _fuse_rank_scores(per_term_rank_scores, weights)

    ranked_non_claim = sorted(
        ((block, score) for block, score in fused.items() if block != claim_id),
        key=lambda item: (-item[1], item[0]),
    )
    floor_count = max(floor_minimum, math.ceil(floor_fraction * pool_size))
    if not ranked_non_claim:
        return CandidateSelection(
            claim_id=claim_id,
            candidate_ids=(),
            pruned_ids=(),
            shape_class=shape_class,
            gap_threshold=0.0,
            floor_count=floor_count,
            weight_set_name=weight_set_name,
            fallback_required=False,
            fallback_reason=None,
        )

    scores_only = [score for _block, score in ranked_non_claim]
    score_range = max(scores_only) - min(scores_only)
    gap = gap_fraction * score_range
    band = adaptive_score_band(fused, gap, claim=claim_id)
    candidate_ids = _floor_extended_members(ranked_non_claim, band.members, floor_count)
    pruned_ids = tuple(
        sorted(
            block
            for block in pool_fingerprints
            if block != claim_id and block not in candidate_ids
        )
    )
    return CandidateSelection(
        claim_id=claim_id,
        candidate_ids=candidate_ids,
        pruned_ids=pruned_ids,
        shape_class=shape_class,
        gap_threshold=gap,
        floor_count=floor_count,
        weight_set_name=weight_set_name,
        fallback_required=False,
        fallback_reason=None,
    )


def select_configured_candidate_band(
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    slide_fingerprint: SpecimenFingerprint,
    claim_id: str,
    *,
    architecture_kind: str,
    architecture_name: str,
    architecture_methods: Sequence[str],
    candidate_band_thresholds: Mapping[str, float],
) -> CandidateSelection:
    """Select using the architecture and score-gap thresholds in the handoff.

    This is deliberately separate from :func:`select_candidate_band`: that
    function is the earlier uncalibrated rank-fusion hypothesis.  A #249
    handoff instead records the diagnostic architecture and its *absolute*
    comparator-score gap(s), so treating those values as rank-fusion fractions
    would silently change the calibration's meaning.

    ``router`` cannot run safely yet: its handoff lacks the per-slide routing
    rule that chose a descriptor in diagnostics.  It returns an explicit
    full-pool fallback rather than guessing a route.
    """
    if claim_id not in pool_fingerprints:
        raise ValueError(
            f"claim {claim_id!r} must be a member of the frozen pool being ranked"
        )
    methods = tuple(architecture_methods)
    if not methods:
        return _configuration_fallback(claim_id, "handoff architecture has no methods")
    missing = _missing_descriptors(pool_fingerprints, slide_fingerprint, methods)
    if missing:
        return _configuration_fallback(
            claim_id, "missing handoff descriptor(s): " + ", ".join(missing)
        )
    method_scores = {
        method: _descriptor_raw_scores(pool_fingerprints, slide_fingerprint, method)
        for method in methods
    }
    if architecture_kind == "individual":
        method = methods[0]
        scores = method_scores[method]
        threshold = candidate_band_thresholds.get(method)
    elif architecture_kind == "union":
        selected: set[str] = set()
        for method in methods:
            threshold = candidate_band_thresholds.get(method)
            if threshold is None:
                return _configuration_fallback(
                    claim_id, f"handoff missing candidate-band threshold for {method!r}"
                )
            selected.update(
                adaptive_score_band(method_scores[method], threshold, claim=claim_id).members
            )
        return _configured_selection(claim_id, tuple(sorted(selected)), pool_fingerprints, None)
    elif architecture_kind == "fusion":
        if architecture_name.startswith("equal_rank_fusion"):
            scores = _equal_rank_fusion(method_scores, methods)
        elif architecture_name.startswith("equal_normalized_fusion"):
            scores = _equal_normalized_fusion(method_scores, methods)
        else:
            return _configuration_fallback(
                claim_id, f"unsupported handoff fusion architecture {architecture_name!r}"
            )
        threshold = candidate_band_thresholds.get("fusion")
    elif architecture_kind == "router":
        return _configuration_fallback(
            claim_id,
            "handoff router architecture has no production routing rule; "
            "scoring complete pool",
        )
    else:
        return _configuration_fallback(
            claim_id, f"unsupported handoff architecture kind {architecture_kind!r}"
        )
    if threshold is None:
        return _configuration_fallback(
            claim_id, "handoff missing candidate-band threshold"
        )
    members = adaptive_score_band(scores, threshold, claim=claim_id).members
    return _configured_selection(claim_id, members, pool_fingerprints, threshold)


def _configured_selection(
    claim_id: str,
    candidate_ids: tuple[str, ...],
    pool_fingerprints: Mapping[str, SpecimenFingerprint],
    gap_threshold: float | None,
) -> CandidateSelection:
    return CandidateSelection(
        claim_id=claim_id,
        candidate_ids=candidate_ids,
        pruned_ids=tuple(
            sorted(
                block for block in pool_fingerprints
                if block != claim_id and block not in candidate_ids
            )
        ),
        shape_class=None,
        gap_threshold=gap_threshold,
        floor_count=0,
        weight_set_name="handoff",
        fallback_required=False,
        fallback_reason=None,
    )


def _configuration_fallback(claim_id: str, reason: str) -> CandidateSelection:
    return CandidateSelection(
        claim_id=claim_id,
        candidate_ids=(),
        pruned_ids=(),
        shape_class=None,
        gap_threshold=None,
        floor_count=None,
        weight_set_name=None,
        fallback_required=True,
        fallback_reason=reason,
    )


def _equal_rank_fusion(
    method_scores: Mapping[str, Mapping[str, float]], methods: Sequence[str],
) -> dict[str, float]:
    blocks = sorted({block for method in methods for block in method_scores[method]})
    totals = {block: 0.0 for block in blocks}
    for method in methods:
        ranked = sorted(method_scores[method].items(), key=lambda item: (-item[1], item[0]))
        denominator = max(1, len(ranked) - 1)
        for index, (block, _score) in enumerate(ranked):
            totals[block] += 1.0 - index / denominator
    return {block: totals[block] / len(methods) for block in blocks}


def _equal_normalized_fusion(
    method_scores: Mapping[str, Mapping[str, float]], methods: Sequence[str],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for method in methods:
        values = method_scores[method]
        low, high = min(values.values()), max(values.values())
        span = high - low
        for block, score in values.items():
            totals[block] = totals.get(block, 0.0) + (1.0 if span == 0 else (score - low) / span)
    return {block: total / len(methods) for block, total in totals.items()}
