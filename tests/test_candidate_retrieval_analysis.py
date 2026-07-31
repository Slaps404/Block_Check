"""High-seam behavioral contracts for pure candidate retrieval analysis (#242)."""

from pathlib import Path

from candidate_retrieval_analysis import (
    Architecture, ArchitectureKind, BandEvaluation, adaptive_candidate_band,
    architecture_candidates, calibrate_architecture_veto,
    calibrate_architecture_thresholds, calibrate_review_veto, candidate_union,
    compare_architectures, evaluate_frozen_architecture, fit_slide_group_router,
    hybrid_audit, hybrid_miss_diagnostics,
    candidate_bands_for_architecture, nested_leave_one_work_order_out,
    nested_work_order_folds, recall_at_k, select_architecture,
    standard_subgroup_band_evaluations, strongest_nonclaim_competitor,
    subgroup_band_evaluations, worst_subgroup,
)
from candidate_retrieval_report import REQUIRED_SECTIONS, render_report, write_report
from verify.work_order_evaluator import evaluate_work_order


def test_competitor_excludes_claim_and_breaks_accurate_tie_by_block_id():
    target = strongest_nonclaim_competitor("s", {"claim": .9, "B": .7, "A": .7}, "claim")
    assert (target.block_id, target.score, target.evaluable) == ("A", .7, True)


def test_recall_excludes_inserted_claim_and_counts_non_evaluable_separately():
    accurate = {"s1": {"C": .9, "X": .8, "Y": .1}, "s2": {"C": .9}}
    heuristic = {"s1": {"C": 100, "X": 2, "Y": 1}, "s2": {"C": 1}}
    summary = recall_at_k("h", accurate, {"s1": "C", "s2": "C"}, heuristic, 1)
    assert (summary.covered_slides, summary.evaluable_slides,
            summary.non_evaluable_slides) == (1, 1, 1)


def test_union_is_unique_and_band_keeps_boundary_ties_but_not_claim():
    assert candidate_union((("A", "B"), ("B", "C")), 2) == ("A", "B", "C")
    band = adaptive_candidate_band({"C": 1.0, "A": .8, "B": .8}, .2, claim="C")
    assert band.members == ("A", "B")


def test_nested_folds_hold_out_entire_work_orders_and_never_fall_back_to_slides():
    assert nested_work_order_folds({"s1": "wo1", "s2": "wo1", "s3": "wo2"}) == (
        ("wo1", ("wo2",)), ("wo2", ("wo1",)),
    )
    assert nested_work_order_folds({"s": "only"}) == ()


def test_hybrid_audit_separates_new_and_inherited_false_passes():
    accurate = {"new": {"C": .9, "X": .95}, "old": {"C": .99, "X": .8}}
    claims = {"new": "C", "old": "C"}
    heuristic = {"new": {"C": 1, "X": 0}, "old": {"C": 1, "X": 0}}
    audit = hybrid_audit(
        accurate, claims, heuristic, 0,
        confirmed_correct_by_slide={"new": "X", "old": "X"},
    )
    assert audit.new_false_pass_count == 1
    assert audit.inherited_false_pass_count == 1
    assert audit.rows[0].missing_competitor == "X"


def test_hybrid_audit_simulates_every_block_claim_when_identity_is_unambiguous():
    accurate = {"s": {"A": .99, "B": .9, "C": .1}}
    heuristic = {"s": {"A": 1, "B": .8, "C": 0}}
    audit = hybrid_audit(
        accurate, {"s": "A"}, heuristic, 0,
        confirmed_correct_by_slide={"s": "A"}, simulate_all_claims=True,
    )
    assert [row.claim for row in audit.rows] == ["A", "B", "C"]
    assert audit.confirmed_wrong_claim_count == 2
    assert audit.safety_evaluable is True


def test_missing_identity_evidence_does_not_make_zero_false_pass_a_safety_claim():
    audit = hybrid_audit(
        {"s": {"A": .9, "B": .8}}, {"s": "A"},
        {"s": {"A": 1, "B": 0}}, 0, simulate_all_claims=True,
    )
    assert audit.new_false_pass_count == 0
    assert audit.confirmed_wrong_claim_count == 0
    assert audit.safety_evaluable is False


def test_frozen_architecture_band_can_drive_all_claim_hybrid_audit():
    scores = {"h": {"s": {"A": 1, "B": .9, "C": 0}}}
    nested = nested_leave_one_work_order_out(
        {"s": {"A": .99, "B": .9, "C": .1},
         "t": {"A": .99, "B": .9, "C": .1}},
        {"s": "A", "t": "A"},
        {"h": {"s": scores["h"]["s"], "t": scores["h"]["s"]}},
        {"s": "one", "t": "two"}, {"h": (.1,)},
    )
    selected = nested.folds[0]
    bands = candidate_bands_for_architecture(
        selected.selected, ("s",), scores, dict(selected.thresholds),
    )
    audit = hybrid_audit(
        {"s": {"A": .99, "B": .9, "C": .1}}, {"s": "A"}, {}, 0,
        confirmed_correct_by_slide={"s": "A"}, simulate_all_claims=True,
        candidate_members_by_slide=bands,
    )
    assert audit.confirmed_wrong_claim_count == 2


def test_nested_selection_never_uses_held_out_work_order_for_method_choice():
    work_orders = {"a": "wo1", "b": "wo2", "sentinel": "wo3"}
    claims = {slide: "C" for slide in work_orders}
    accurate = {
        "a": {"C": .9, "X": .8, "Y": .1},
        "b": {"C": .9, "X": .8, "Y": .1},
        "sentinel": {"C": .9, "X": .8, "Y": .1},
    }
    methods = {
        "stable": {
            "a": {"C": 0, "X": 2, "Y": 1},
            "b": {"C": 0, "X": 2, "Y": 1},
            "sentinel": {"C": 0, "X": 1, "Y": 2},
        },
        "sentinel_only": {
            "a": {"C": 0, "X": 1, "Y": 2},
            "b": {"C": 0, "X": 1, "Y": 2},
            "sentinel": {"C": 0, "X": 2, "Y": 1},
        },
    }
    result = nested_leave_one_work_order_out(
        accurate, claims, methods, work_orders, gap_grid={
            "stable": (0,), "sentinel_only": (0,), "fusion": (0,),
        },
    )
    sentinel_fold = next(fold for fold in result.folds if fold.held_out_order == "wo3")
    assert sentinel_fold.selected.kind == "individual"
    assert sentinel_fold.selected.methods == ("stable",)
    assert "sentinel" not in sentinel_fold.training_slide_ids
    assert result.held_out_evaluable_slides == 3


def test_nested_selection_warns_when_outer_training_cannot_support_inner_orders():
    result = nested_leave_one_work_order_out(
        {"a": {"C": .9, "X": .8}, "b": {"C": .9, "X": .8}},
        {"a": "C", "b": "C"},
        {"h": {"a": {"C": 0, "X": 1}, "b": {"C": 0, "X": 1}}},
        {"a": "one", "b": "two"}, {"h": (0,)},
    )
    assert "inner work-order validation" in result.insufficient_generalization_warning


def test_router_dispatch_is_eligible_only_when_it_beats_fusion_and_union():
    accurate = {
        "s1": {"C": .9, "Y": .8, "X": .1},
        "s2": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {"s1": "C", "s2": "C"}
    methods = {
        "h1": {
            "s1": {"C": 0, "X": 2, "Y": 1},
            "s2": {"C": 0, "X": 2, "Y": 1},
        },
        "h2": {
            "s1": {"C": 0, "X": 1, "Y": 2},
            "s2": {"C": 0, "X": 1, "Y": 2},
        },
    }
    comparison = compare_architectures(
        ("s1", "s2"), accurate, claims, methods,
        router_by_slide={"s1": "h2", "s2": "h1"},
    )
    router = next(row for row in comparison if row[0].kind is ArchitectureKind.ROUTER)
    assert router[1].coverage == 1.0
    assert router[1].candidate_counts == (1, 1)
    assert select_architecture(comparison).kind is ArchitectureKind.ROUTER


def test_uncertain_router_dispatch_falls_back_to_unique_candidate_union():
    comparison = compare_architectures(
        ("s",), {"s": {"C": .9, "Y": .8, "X": .1}}, {"s": "C"},
        {
            "h1": {"s": {"C": 0, "X": 2, "Y": 1}},
            "h2": {"s": {"C": 0, "X": 1, "Y": 2}},
        },
        router_by_slide={"s": None},
    )
    union = next(row for row in comparison if row[0].kind is ArchitectureKind.UNION)
    router = next(row for row in comparison if row[0].kind is ArchitectureKind.ROUTER)
    assert router[1] == union[1]
    assert router[1].candidate_counts == (2,)


def test_router_must_beat_best_fusion_and_union_among_multiple_baselines():
    metric = lambda covered, counts: BandEvaluation(  # noqa: E731
        2, covered, counts, () if covered == 2 else ("miss",),
    )
    comparison = (
        (Architecture(ArchitectureKind.FUSION, "fusion_weak", ("a", "b")),
         metric(1, (1, 1))),
        (Architecture(ArchitectureKind.FUSION, "fusion_best", ("a", "c")),
         metric(2, (1, 1))),
        (Architecture(ArchitectureKind.UNION, "union_weak", ("a", "b")),
         metric(1, (2, 2))),
        (Architecture(ArchitectureKind.UNION, "union_best", ("a", "c")),
         metric(2, (2, 2))),
        (Architecture(ArchitectureKind.ROUTER, "router", ("a", "b", "c")),
         metric(2, (1, 1))),
    )
    assert select_architecture(comparison).name == "fusion_best"


def test_fusion_gap_is_calibrated_on_training_then_frozen_for_holdout():
    architecture = Architecture(
        ArchitectureKind.FUSION, "equal_rank_fusion", ("h1", "h2"),
    )
    accurate = {
        "train": {"C": .9, "X": .8, "Y": .1},
        "held": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {"train": "C", "held": "C"}
    methods = {
        method: {
            "train": {"C": 3, "X": 2, "Y": 1},
            "held": {"C": 3, "X": 2, "Y": 1},
        }
        for method in ("h1", "h2")
    }
    thresholds, training = calibrate_architecture_thresholds(
        architecture, ("train",), accurate, claims, methods,
        {"fusion": (0, .5)},
    )
    held_out = evaluate_frozen_architecture(
        architecture, ("held",), accurate, claims, methods, dict(thresholds),
    )
    assert thresholds == (("fusion", .5),)
    assert training.coverage == held_out.coverage == 1.0
    assert held_out.candidate_counts == (1,)


def test_union_uses_per_method_training_gaps_then_freezes_them_for_holdout():
    architecture = Architecture(
        ArchitectureKind.UNION, "candidate_union", ("h1", "h2"),
    )
    accurate = {
        "train_x": {"C": .9, "X": .8, "Y": .1},
        "train_y": {"C": .9, "Y": .8, "X": .1},
        "held": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {slide: "C" for slide in accurate}
    methods = {
        "h1": {
            "train_x": {"C": 1, "X": .8, "Y": 0},
            "train_y": {"C": 1, "X": 0, "Y": 0},
            "held": {"C": 1, "X": .8, "Y": 0},
        },
        "h2": {
            "train_x": {"C": 1, "X": 0, "Y": 0},
            "train_y": {"C": 1, "X": 0, "Y": .7},
            "held": {"C": 1, "X": 0, "Y": 0},
        },
    }
    thresholds, training = calibrate_architecture_thresholds(
        architecture, ("train_x", "train_y"), accurate, claims, methods,
        {"h1": (0, .2), "h2": (0, .31)},
    )
    held_out = evaluate_frozen_architecture(
        architecture, ("held",), accurate, claims, methods, dict(thresholds),
    )
    assert thresholds == (("h1", .2), ("h2", .31))
    assert training.coverage == held_out.coverage == 1.0
    assert held_out.candidate_counts == (1,)


def test_large_threshold_grid_is_bounded_reproducible_and_deterministic(monkeypatch):
    methods = tuple(f"m{i}" for i in range(7))
    architecture = Architecture(
        ArchitectureKind.UNION, "candidate_union", methods,
    )
    accurate = {"s": {"C": .9, "X": .8}}
    claims = {"s": "C"}
    heuristic = {
        method: {"s": {"C": 1, "X": (.5 if method == "m6" else 0)}}
        for method in methods
    }
    grid = {method: (0, .25, .5, .75) for method in methods}
    assert 4 ** len(methods) > 4096

    def forbidden_cartesian_product(*_args, **_kwargs):
        raise AssertionError("large grids must use bounded coordinate search")

    monkeypatch.setattr(
        "candidate_retrieval_analysis.product", forbidden_cartesian_product,
    )
    first = calibrate_architecture_thresholds(
        architecture, ("s",), accurate, claims, heuristic, grid,
    )
    second = calibrate_architecture_thresholds(
        architecture, ("s",), accurate, claims, heuristic, grid,
    )
    expected = tuple(
        (method, .5 if method == "m6" else 0) for method in methods
    )
    assert first == second
    assert first[0] == expected
    assert first[1].coverage == 1.0


def test_subgroup_evaluations_group_rows_and_skip_missing_metadata():
    accurate = {
        "covered": {"C": .9, "X": .8, "Y": .1},
        "missed": {"C": .9, "X": .8, "Y": .1},
        "unlabeled": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {slide: "C" for slide in accurate}
    cuts = subgroup_band_evaluations(
        accurate, {"covered": "lung", "missed": "lung"},
        accurate, claims,
        {"covered": ("X",), "missed": ("Y",), "unlabeled": ("X",)},
    )
    assert set(cuts) == {"lung"}
    assert cuts["lung"].evaluable_slides == 2
    assert cuts["lung"].covered_slides == 1
    assert cuts["lung"].missed_slide_ids == ("missed",)


def test_standard_subgroup_cuts_include_worst_tissue_morphology_and_capture_status():
    accurate = {
        "a": {"C": .9, "X": .8, "Y": .1},
        "b": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {"a": "C", "b": "C"}
    cuts = standard_subgroup_band_evaluations(
        accurate,
        {
            "a": {"tissue": "lung", "morphology": "single",
                  "sparse_dense": "dense", "capture_status": "primary"},
            "b": {"tissue": "skin", "morphology": "ribbon",
                  "sparse_dense": "sparse", "capture_status": "adversarial"},
        },
        accurate, claims, {"a": ("X",), "b": ("Y",)},
    )
    assert set(cuts) == {"tissue", "morphology", "sparse_dense", "capture_status"}
    field, group, metric = worst_subgroup(cuts)
    assert metric.coverage == 0.0
    assert (field, group) == ("capture_status", "adversarial")


def test_invalid_architecture_kind_is_rejected_at_construction():
    import pytest

    with pytest.raises(ValueError, match="not a valid ArchitectureKind"):
        Architecture("accidental_fallback", "bad", ("h",))


def test_search_includes_pair_subsets_and_both_simple_fusion_baselines():
    candidates = architecture_candidates(("a", "b", "c"))
    names = {candidate.name for candidate in candidates}
    assert "equal_rank_fusion:a+b" in names
    assert "equal_normalized_fusion:a+b" in names
    assert "candidate_union:a+b" in names
    assert "candidate_union:a+b+c" in names


def test_39_method_pair_search_is_bounded_and_screened_on_training_only():
    method_names = tuple(f"m{i:02d}" for i in range(39))
    assert len(architecture_candidates(method_names)) <= 87
    accurate = {
        "train": {"C": .9, "X": .8, "Y": .1},
        "held": {"C": .9, "X": .8, "Y": .1},
    }
    claims = {"train": "C", "held": "C"}
    methods = {}
    for method in method_names:
        methods[method] = {
            "train": {"C": 0, "X": 1, "Y": 2},
            "held": {"C": 0, "X": 1, "Y": 2},
        }
    methods["m00"]["train"] = {"C": 0, "X": 2, "Y": 1}
    methods["m38"]["held"] = {"C": 0, "X": 2, "Y": 1}
    comparison = compare_architectures(
        ("held",), accurate, claims, methods,
        screening_slide_ids=("train",), pairwise_top_n=2,
        complementary_method_count=1,
    )
    pairwise = [
        architecture for architecture, _metric in comparison
        if len(architecture.methods) == 2
    ]
    assert pairwise
    assert all("m38" not in architecture.methods for architecture in pairwise)


def test_slide_group_router_is_fit_only_from_training_and_unseen_group_is_uncertain():
    accurate = {
        "train": {"C": .9, "X": .8, "Y": .1},
        "held": {"C": .9, "X": .8, "Y": .1},
        "unseen": {"C": .9, "X": .8, "Y": .1},
    }
    methods = {
        "good": {slide: {"C": 0, "X": 2, "Y": 1} for slide in accurate},
        "bad": {slide: {"C": 0, "X": 1, "Y": 2} for slide in accurate},
    }
    routes = fit_slide_group_router(
        ("train",), accurate, {slide: "C" for slide in accurate}, methods,
        {"train": "dense", "held": "dense", "unseen": "sparse"},
    )
    assert routes == {"train": "good", "held": "good", "unseen": None}


def test_hybrid_miss_rows_have_reason_classification():
    audit = hybrid_audit(
        {"s": {"C": .9, "X": .95}}, {"s": "C"},
        {"s": {"C": 1, "X": 0}}, 0,
        confirmed_correct_by_slide={"s": "X"},
    )
    row = hybrid_miss_diagnostics(audit)[0]
    assert row["event"] == "strongest_nonclaim_omitted"
    assert row["reason_classification"] == "unknown_needs_manual_classification"
    classified = hybrid_miss_diagnostics(
        audit, {("s", "C"): "low_information_mask"},
    )
    assert classified[0]["reason_classification"] == "low_information_mask"


def test_report_exposes_timing_subgroups_parity_and_clean_nonpromotion_wording():
    audit = hybrid_audit(
        {"s": {"C": .9, "X": .8}}, {"s": "C"},
        {"s": {"C": 1, "X": 0}}, 0,
    )
    text = render_report(
        provenance={}, descriptor_catalog=[], recall_summaries=[], audit=audit,
        veto=calibrate_review_veto({}, {}, {}, {}, []), misses=[],
        recommendation="not production-ready", timing_summary={"construction_ns": 3},
        subgroup_cuts={
            "tissue": ({"name": "lung", "coverage": 1.0},),
            "capture_status": ({"name": "primary", "coverage": 1.0},),
        },
        worst_subgroup_summary={"field": "tissue", "name": "lung"},
        efficiency_summary={
            "accurate_rerank_calls": 2,
            "estimated_runtime": .2,
            "observed_runtime": .3,
            "full_comparison_reduction": .8,
        },
        veto_fold_results=({
            "held_out_order": "wo2",
            "enabled": False,
            "training_enabled": True,
            "threshold": .2,
            "reason": "safe on training",
            "vetoed_claims": ["train-slide"],
            "training_false_reviews": [],
            "heldout_vetoed_slides": ["held-slide"],
            "heldout_false_reviews": ["held-slide"],
            "heldout_safe": False,
        },),
    )
    assert "construction_ns" in text
    assert "Subgroup tissue:" in text
    assert "Worst subgroup:" in text
    assert "accurate_rerank_calls=2" in text
    assert "holdout=wo2" in text
    assert "frozen_threshold=0.2" in text
    assert "heldout_false_reviews=['held-slide']" in text
    assert "Verdict parity:" in text
    assert "not not production-promoted" not in text


def test_veto_disables_when_every_useful_threshold_false_reviews_a_confirmed_correct_pass():
    scores = {"s": {"C": .5, "X": .9}}
    baseline = {"s": evaluate_work_order({"C": .9, "X": .1}, "C")}
    result = calibrate_review_veto(scores, {"s": "C"}, baseline, {"s": "C"}, [.1, .4])
    assert result.enabled is False


def test_veto_selects_review_only_threshold_at_exact_boundary_when_safe():
    scores = {
        "wrong": {"C": .5, "X": .9},
        "correct": {"C": .9, "X": .6},
    }
    baseline = {
        "wrong": evaluate_work_order({"C": .5, "X": .9}, "C"),
        "correct": evaluate_work_order({"C": .9, "X": .6}, "C"),
    }
    result = calibrate_review_veto(
        scores, {"wrong": "C", "correct": "C"}, baseline,
        {"wrong": "X", "correct": "C"}, (.4,),
    )
    assert result.enabled is True
    assert result.threshold == .4
    assert result.vetoed_claims == ("wrong",)
    assert result.false_reviews == ()


def test_selected_union_veto_requires_all_member_methods_to_agree():
    architecture = Architecture(
        ArchitectureKind.UNION, "candidate_union", ("h1", "h2"),
    )
    methods = {
        "h1": {
            "wrong": {"C": .5, "X": .9},
            "correct": {"C": .9, "X": .5},
        },
        "h2": {
            "wrong": {"C": .5, "X": .71},
            "correct": {"C": .8, "X": .9},
        },
    }
    claims = {"wrong": "C", "correct": "C"}
    baseline = {
        "wrong": evaluate_work_order({"C": .5, "X": .9}, "C"),
        "correct": evaluate_work_order({"C": .9, "X": .5}, "C"),
    }
    result = calibrate_architecture_veto(
        architecture, claims, methods, claims, baseline,
        {"wrong": "X", "correct": "C"}, (.2,),
    )
    assert result.enabled is True
    assert result.vetoed_claims == ("wrong",)


def test_report_has_required_sections_and_cannot_promote_production(tmp_path: Path):
    audit = hybrid_audit({"s": {"C": .9, "X": .1}}, {"s": "C"}, {"s": {"C": 1, "X": 0}}, 0)
    veto = calibrate_review_veto({}, {}, {}, {}, [])
    text = render_report(provenance={}, descriptor_catalog=[], recall_summaries=[], audit=audit,
                         veto=veto, misses=[], recommendation="production-ready based on nothing")
    assert all(section in text for section in REQUIRED_SECTIONS)
    assert "production-ready" not in text
    assert "Safety not evaluable" in text
    assert write_report(
        tmp_path / "report.md", provenance={}, descriptor_catalog=[],
        recall_summaries=[], audit=audit, veto=veto, misses=[],
        recommendation="exploratory",
    ).exists()
