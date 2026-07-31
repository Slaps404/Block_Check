"""Pure, fast tests for the #253 Heuristic Candidate Band (code/verify/candidate_band.py).

No cv2, no ProcessingStore, no threads, no temp databases, no real images:
every fingerprint below is hand-built so a descriptor's comparator output is a
directly chosen ``quality`` in ``[0, 1]`` (see ``_quality_vector``), letting
each test engineer an exact ranking outcome instead of reasoning about real
mask geometry.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from constants import SHAPE_ROUTER_SIZE_THRESHOLD
from verify.candidate_band import (
    DENSE_DESCRIPTOR_WEIGHTS,
    DENSE_FLOOR_FRACTION,
    DENSE_FLOOR_MINIMUM,
    SPARSE_DESCRIPTOR_WEIGHTS,
    SPARSE_FLOOR_FRACTION,
    SPARSE_FLOOR_MINIMUM,
    ShapeClass,
    SpecimenFingerprint,
    adaptive_score_band,
    route_shape_class,
    select_candidate_band,
    select_configured_candidate_band,
    validate_selection,
)
from verify.invariant_descriptors import DescriptorValue, descriptor_catalog

_CATALOG = {spec.name: spec for spec in descriptor_catalog()}
_DENSE_REQUIRED = tuple(sorted(n for n, w in DENSE_DESCRIPTOR_WEIGHTS.items() if w > 0))
_SPARSE_REQUIRED = tuple(sorted(n for n, w in SPARSE_DESCRIPTOR_WEIGHTS.items() if w > 0))
_DENSE_SLIDE_FRACTION = 0.2
_SPARSE_SLIDE_FRACTION = 0.01

assert _DENSE_SLIDE_FRACTION >= SHAPE_ROUTER_SIZE_THRESHOLD
assert _SPARSE_SLIDE_FRACTION < SHAPE_ROUTER_SIZE_THRESHOLD


def _dv(vector: np.ndarray) -> DescriptorValue:
    return DescriptorValue(vector=np.asarray(vector, dtype=np.float64), construction_ns=0)


def _quality_vector(dimension: int, comparison: str, quality: float) -> np.ndarray:
    """Build a vector whose comparator score vs `_reference_vector` is exactly `quality`."""
    if comparison == "exp_l1":
        magnitude = -math.log(quality) / dimension
        return np.full(dimension, magnitude, dtype=np.float64)
    if comparison == "histogram_intersection":
        vector = np.zeros(dimension, dtype=np.float64)
        vector[0] = quality
        vector[1] = 1.0 - quality
        return vector
    raise ValueError(f"unsupported comparison {comparison!r}")


def _reference_vector(dimension: int, comparison: str) -> np.ndarray:
    if comparison == "exp_l1":
        return np.zeros(dimension, dtype=np.float64)
    if comparison == "histogram_intersection":
        vector = np.zeros(dimension, dtype=np.float64)
        vector[0] = 1.0
        return vector
    raise ValueError(f"unsupported comparison {comparison!r}")


def _slide_fingerprint(
    occupied_fraction: float, names: Sequence[str] = _DENSE_REQUIRED + _SPARSE_REQUIRED,
) -> SpecimenFingerprint:
    values = {
        name: _dv(_reference_vector(_CATALOG[name].dimension, _CATALOG[name].comparison))
        for name in dict.fromkeys(names)
    }
    return SpecimenFingerprint("slide", occupied_fraction, values)


def _quality_fingerprint(
    specimen_id: str,
    slide_occupied_fraction: float,
    quality: float,
    names: Sequence[str],
    *,
    size_signal_quality: Optional[float] = None,
    per_descriptor_quality: Optional[Mapping[str, float]] = None,
) -> SpecimenFingerprint:
    """Build a fingerprint whose comparator score is `quality` on every named
    descriptor (or an explicit per-descriptor override), plus a `size_signal`
    comparator score of `size_signal_quality` (defaults to `quality`)."""
    overrides = per_descriptor_quality or {}
    values = {
        name: _dv(
            _quality_vector(
                _CATALOG[name].dimension, _CATALOG[name].comparison, overrides.get(name, quality)
            )
        )
        for name in names
    }
    size_quality = quality if size_signal_quality is None else size_signal_quality
    occupied_fraction = slide_occupied_fraction + (-math.log(size_quality))
    return SpecimenFingerprint(specimen_id, occupied_fraction, values)


def test_route_shape_class_reuses_the_production_threshold():
    assert route_shape_class(_DENSE_SLIDE_FRACTION) is ShapeClass.DENSE
    assert route_shape_class(_SPARSE_SLIDE_FRACTION) is ShapeClass.SPARSE
    assert route_shape_class(SHAPE_ROUTER_SIZE_THRESHOLD) is ShapeClass.DENSE  # boundary is dense


# ---------------------------------------------------------------------------
# 1. Faithful port of adaptive_candidate_band, including claim exclusion.
# ---------------------------------------------------------------------------


def test_ported_band_matches_original_including_claim_excluded_from_count():
    from candidate_retrieval_analysis import adaptive_candidate_band

    scores = {"claim": 0.95, "b1": 0.90, "b2": 0.85, "b3": 0.50}
    original = adaptive_candidate_band(scores, 0.1, claim="claim")
    ported = adaptive_score_band(scores, 0.1, claim="claim")

    assert ported.members == original.members == ("b1", "b2")
    assert ported.maximum_score == original.maximum_score == 0.95
    assert ported.threshold == original.threshold == 0.1
    assert "claim" not in ported.members  # claim never consumes a band slot


def test_handoff_threshold_changes_individual_candidate_selection():
    """#253: calibrated handoff gaps, not uncalibrated runtime defaults, drive pruning."""
    method = _DENSE_REQUIRED[0]
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, (method,))
    pool = {
        "claim": _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, .99, (method,)),
        "near": _quality_fingerprint("near", _DENSE_SLIDE_FRACTION, .90, (method,)),
        "far": _quality_fingerprint("far", _DENSE_SLIDE_FRACTION, .60, (method,)),
    }
    narrow = select_configured_candidate_band(
        pool, slide, "claim", architecture_kind="individual",
        architecture_name="individual", architecture_methods=(method,),
        candidate_band_thresholds={method: .1},
    )
    wide = select_configured_candidate_band(
        pool, slide, "claim", architecture_kind="individual",
        architecture_name="individual", architecture_methods=(method,),
        candidate_band_thresholds={method: .4},
    )
    assert narrow.candidate_ids == ("near",)
    assert wide.candidate_ids == ("near", "far")
    assert narrow.pruned_ids == ("far",)
    assert wide.pruned_ids == ()


def test_router_handoff_fails_closed_to_complete_pool_scoring():
    method = _DENSE_REQUIRED[0]
    selection = select_configured_candidate_band(
        {"claim": _quality_fingerprint("claim", .2, .99, (method,)),
         "other": _quality_fingerprint("other", .2, .8, (method,))},
        _slide_fingerprint(.2, (method,)), "claim",
        architecture_kind="router", architecture_name="router", architecture_methods=(method,),
        candidate_band_thresholds={method: .1},
    )
    assert selection.fallback_required
    assert "no production routing rule" in selection.fallback_reason


# ---------------------------------------------------------------------------
# 2. Claim always structurally recoverable, even at the extreme (dead last).
# ---------------------------------------------------------------------------


def test_claim_never_in_candidate_ids_but_always_in_accurate_scoring_ids_at_dead_last():
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION)
    claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 0.01, _DENSE_REQUIRED)
    others = {
        f"block_{i}": _quality_fingerprint(
            f"block_{i}", _DENSE_SLIDE_FRACTION, 0.99, _DENSE_REQUIRED
        )
        for i in range(4)
    }
    pool = {"claim": claim, **others}

    selection = select_candidate_band(pool, slide, "claim")

    assert not selection.fallback_required
    assert "claim" not in selection.candidate_ids
    assert "claim" in selection.accurate_scoring_ids
    assert selection.accurate_scoring_ids == tuple(sorted({"claim", *selection.candidate_ids}))
    assert validate_selection(selection) is None


# ---------------------------------------------------------------------------
# 3. Strongest non-claim competitor survives a near-tie band, beyond the floor.
# ---------------------------------------------------------------------------


def test_strongest_competitor_survives_band_beyond_the_floor_on_near_tie_scores():
    """Sparse margins in the real corpus are razor thin (lab mechanism labels:
    set 018 -0.0048, 014 +0.0022, 019 +0.09) -- much thinner than Borda rank
    fusion's discretized spacing can literally reproduce. This test proves the
    *mechanism* those numbers motivate: a competitor several ranks below the
    floor still survives because the wide sparse gap (0.65x range), not the
    floor, protects it.
    """
    slide = _slide_fingerprint(_SPARSE_SLIDE_FRACTION)
    claim = _quality_fingerprint("claim", _SPARSE_SLIDE_FRACTION, 0.5, _SPARSE_REQUIRED)
    ranked_blocks = {
        f"rank{i}": _quality_fingerprint(
            f"rank{i}", _SPARSE_SLIDE_FRACTION, 0.99 - i * 0.01, _SPARSE_REQUIRED
        )
        for i in range(9)  # rank0 best .. rank8 worst; identical order on every term
    }
    pool = {"claim": claim, **ranked_blocks}

    selection = select_candidate_band(pool, slide, "claim")

    assert not selection.fallback_required
    assert selection.shape_class is ShapeClass.SPARSE
    expected_floor = max(SPARSE_FLOOR_MINIMUM, math.ceil(SPARSE_FLOOR_FRACTION * len(pool)))
    assert selection.floor_count == expected_floor == 4
    # The competitor sits at rank index 5 (6th best of 9) -- past the floor's
    # reach (indices 0-3) -- and is retained only because the sparse gap
    # (0.65x the full [0,1] Borda range) reaches index 5 (diff 0.625 <= 0.65).
    assert "rank5" in selection.candidate_ids
    assert {"rank6", "rank7", "rank8"} <= set(selection.pruned_ids)
    assert len(selection.candidate_ids) == 6  # gap admits more than the bare floor


# ---------------------------------------------------------------------------
# 4. Dense vs sparse differ on the same pool; floors honored on pools of 4/5.
# ---------------------------------------------------------------------------


def test_dense_and_sparse_select_different_candidates_on_the_same_pool():
    all_names = _DENSE_REQUIRED + tuple(n for n in _SPARSE_REQUIRED if n not in _DENSE_REQUIRED)
    component_only = tuple(n for n in _SPARSE_REQUIRED if n not in _DENSE_REQUIRED)

    def build(specimen_id: str, shape_quality: float, component_quality: float, fraction: float):
        overrides = {name: component_quality for name in component_only}
        for name in _DENSE_REQUIRED:
            overrides.setdefault(name, shape_quality)
        return _quality_fingerprint(
            specimen_id, fraction, shape_quality, all_names,
            size_signal_quality=0.5, per_descriptor_quality=overrides,
        )

    dense_slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, all_names)
    claim = build("claim", 0.5, 0.5, _DENSE_SLIDE_FRACTION)
    shape_favored = build("shape_favored", 0.99, 0.02, _DENSE_SLIDE_FRACTION)
    component_favored = build("component_favored", 0.02, 0.99, _DENSE_SLIDE_FRACTION)
    pool = {"claim": claim, "shape_favored": shape_favored, "component_favored": component_favored}

    dense_selection = select_candidate_band(pool, dense_slide, "claim")
    sparse_slide = _slide_fingerprint(_SPARSE_SLIDE_FRACTION, all_names)
    sparse_selection = select_candidate_band(pool, sparse_slide, "claim")

    assert dense_selection.shape_class is ShapeClass.DENSE
    assert sparse_selection.shape_class is ShapeClass.SPARSE
    # Dense weights ignore component_* entirely -> shape_favored ranks first.
    assert dense_selection.candidate_ids[0] == "shape_favored"
    # Sparse weights lean heavily on component_* -> component_favored ranks first.
    assert sparse_selection.candidate_ids[0] == "component_favored"
    assert dense_selection.candidate_ids != sparse_selection.candidate_ids


def _extreme_pool(required_names: Sequence[str], slide_fraction: float, pool_size: int):
    """One clearly-best non-claim block plus (pool_size - 2) clearly-bad ones."""
    slide = _slide_fingerprint(slide_fraction, required_names)
    claim = _quality_fingerprint("claim", slide_fraction, 0.5, required_names)
    pool = {"claim": claim}
    pool["best"] = _quality_fingerprint("best", slide_fraction, 0.999, required_names)
    for i in range(pool_size - 2):
        pool[f"far_{i}"] = _quality_fingerprint(
            f"far_{i}", slide_fraction, 0.001, required_names
        )
    return pool, slide


def test_dense_floor_is_honored_on_pools_of_4_and_5():
    for pool_size in (4, 5):
        pool, slide = _extreme_pool(_DENSE_REQUIRED, _DENSE_SLIDE_FRACTION, pool_size)
        selection = select_candidate_band(pool, slide, "claim")
        expected_floor = max(DENSE_FLOOR_MINIMUM, math.ceil(DENSE_FLOOR_FRACTION * pool_size))
        non_claim_count = pool_size - 1
        assert selection.floor_count == expected_floor
        assert len(selection.candidate_ids) >= min(expected_floor, non_claim_count)


def test_sparse_floor_is_honored_on_pools_of_4_and_5():
    for pool_size in (4, 5):
        pool, slide = _extreme_pool(_SPARSE_REQUIRED, _SPARSE_SLIDE_FRACTION, pool_size)
        selection = select_candidate_band(pool, slide, "claim")
        expected_floor = max(SPARSE_FLOOR_MINIMUM, math.ceil(SPARSE_FLOOR_FRACTION * pool_size))
        non_claim_count = pool_size - 1
        assert selection.floor_count == expected_floor
        assert len(selection.candidate_ids) >= min(expected_floor, non_claim_count)


# ---------------------------------------------------------------------------
# 5. Rank fusion is scale-free.
# ---------------------------------------------------------------------------


def test_rank_fusion_is_scale_free_under_rescaling_one_descriptor():
    """Halving one descriptor's raw comparator outputs (order-preserving,
    since all qualities stay positive) must not change the fused ranking --
    this is the property that justifies Borda fusion over a raw weighted sum.
    """
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, _DENSE_REQUIRED)
    varied_descriptor = "global_morphology_v1"
    other_descriptors = tuple(n for n in _DENSE_REQUIRED if n != varied_descriptor)
    qualities = {"b0": 0.9, "b1": 0.6, "b2": 0.3}

    def build_pool(scale: float):
        claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 1.0, _DENSE_REQUIRED)
        pool = {"claim": claim}
        for block_id, quality in qualities.items():
            overrides = {name: 1.0 for name in other_descriptors}
            overrides[varied_descriptor] = quality * scale
            pool[block_id] = _quality_fingerprint(
                block_id, _DENSE_SLIDE_FRACTION, 1.0, _DENSE_REQUIRED,
                size_signal_quality=1.0, per_descriptor_quality=overrides,
            )
        return pool

    full_scale_selection = select_candidate_band(build_pool(1.0), slide, "claim")
    half_scale_selection = select_candidate_band(build_pool(0.5), slide, "claim")

    assert full_scale_selection.candidate_ids == half_scale_selection.candidate_ids
    assert full_scale_selection.pruned_ids == half_scale_selection.pruned_ids
    assert full_scale_selection.shape_class == half_scale_selection.shape_class


# ---------------------------------------------------------------------------
# 6. min() aggregation would have excluded a block that weighted Borda keeps.
# ---------------------------------------------------------------------------


def test_min_aggregation_would_exclude_a_competitor_weighted_fusion_keeps():
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, _DENSE_REQUIRED)
    claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 0.5, _DENSE_REQUIRED)
    # competitor: excellent on 5/6 descriptors, but one noisy descriptor
    # (hu_absolute_moments_v1, weight 0.05 -- the smallest dense weight) tanks it.
    competitor_overrides = {name: 0.95 for name in _DENSE_REQUIRED}
    competitor_overrides["hu_absolute_moments_v1"] = 0.05
    competitor = _quality_fingerprint(
        "competitor", _DENSE_SLIDE_FRACTION, 0.95, _DENSE_REQUIRED,
        size_signal_quality=0.95, per_descriptor_quality=competitor_overrides,
    )
    uniform = _quality_fingerprint(
        "uniform", _DENSE_SLIDE_FRACTION, 0.6, _DENSE_REQUIRED, size_signal_quality=0.6,
    )
    bad = _quality_fingerprint(
        "bad", _DENSE_SLIDE_FRACTION, 0.1, _DENSE_REQUIRED, size_signal_quality=0.1,
    )
    pool = {"claim": claim, "competitor": competitor, "uniform": uniform, "bad": bad}

    selection = select_candidate_band(pool, slide, "claim")
    assert "competitor" in selection.candidate_ids

    # Naive min-aggregation over the SAME raw per-descriptor comparator scores:
    min_scores = {
        "competitor": min([0.95] * 5 + [0.05]),  # hu tanks the min to 0.05
        "uniform": 0.6,
        "bad": 0.1,
    }
    naive_band = adaptive_score_band(min_scores, 0.1, claim=None)
    assert "competitor" not in naive_band.members  # min() would have vetoed it
    assert naive_band.members == ("uniform",)


# ---------------------------------------------------------------------------
# 7. Pruned blocks are explicit and distinguishable from never-scanned blocks.
# ---------------------------------------------------------------------------


def test_pruned_blocks_are_explicit_and_distinct_from_never_scanned_blocks():
    pool, slide = _extreme_pool(_DENSE_REQUIRED, _DENSE_SLIDE_FRACTION, 5)
    selection = select_candidate_band(pool, slide, "claim")

    assert set(selection.pruned_ids)  # at least one block was actually pruned
    for pruned_id in selection.pruned_ids:
        assert pruned_id not in selection.candidate_ids
        assert pruned_id != "claim"

    ghost_id = "block_never_in_this_work_order"
    assert ghost_id not in selection.candidate_ids
    assert ghost_id not in selection.pruned_ids
    assert ghost_id not in selection.accurate_scoring_ids
    # A pruned block is explicit (present in pruned_ids); a never-scanned
    # block is wholly absent from every field -- that absence, not a shared
    # "not selected" bucket, is what a caller must use to tell them apart.


# ---------------------------------------------------------------------------
# 8. Fallback-required reasons: missing descriptor, and mask quality.
# ---------------------------------------------------------------------------


def test_fallback_required_when_a_descriptor_is_missing():
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, _DENSE_REQUIRED)
    claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 0.9, _DENSE_REQUIRED)
    incomplete_names = tuple(n for n in _DENSE_REQUIRED if n != "global_morphology_v1")
    other = _quality_fingerprint("other", _DENSE_SLIDE_FRACTION, 0.9, incomplete_names)
    pool = {"claim": claim, "other": other}

    selection = select_candidate_band(pool, slide, "claim")

    assert selection.fallback_required
    assert selection.candidate_ids == ()
    assert selection.pruned_ids == ()
    assert selection.fallback_reason is not None
    assert "global_morphology_v1" in selection.fallback_reason
    assert validate_selection(selection) is None  # an honest fallback is valid


def test_fallback_required_on_a_caller_supplied_mask_quality_reason():
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, _DENSE_REQUIRED)
    claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 0.9, _DENSE_REQUIRED)
    pool = {"claim": claim}

    selection = select_candidate_band(
        pool, slide, "claim",
        mask_quality_fallback_reason="block mask failed verify.gates quality check",
    )

    assert selection.fallback_required
    assert selection.fallback_reason == "block mask failed verify.gates quality check"
    assert selection.candidate_ids == () and selection.pruned_ids == ()
    assert selection.shape_class is None
    assert validate_selection(selection) is None


# ---------------------------------------------------------------------------
# 9. Zero-weight component descriptors genuinely do not influence dense ranking.
# ---------------------------------------------------------------------------


def test_zero_weight_component_descriptor_does_not_move_dense_ranking():
    zero_weight_name = "component_radial_histogram_v1"
    assert DENSE_DESCRIPTOR_WEIGHTS[zero_weight_name] == 0.0
    names_with_extra = _DENSE_REQUIRED + (zero_weight_name,)
    slide = _slide_fingerprint(_DENSE_SLIDE_FRACTION, names_with_extra)

    def build_pool(perturbed_quality: float):
        claim = _quality_fingerprint("claim", _DENSE_SLIDE_FRACTION, 0.5, _DENSE_REQUIRED)
        b0 = _quality_fingerprint(
            "b0", _DENSE_SLIDE_FRACTION, 0.9, names_with_extra,
            per_descriptor_quality={zero_weight_name: perturbed_quality},
        )
        b1 = _quality_fingerprint(
            "b1", _DENSE_SLIDE_FRACTION, 0.4, names_with_extra,
            per_descriptor_quality={zero_weight_name: perturbed_quality},
        )
        return {"claim": claim, "b0": b0, "b1": b1}

    baseline = select_candidate_band(build_pool(0.99), slide, "claim")
    perturbed = select_candidate_band(build_pool(0.001), slide, "claim")

    assert baseline.candidate_ids == perturbed.candidate_ids
    assert baseline.pruned_ids == perturbed.pruned_ids
