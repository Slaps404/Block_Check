"""Exact CSV schema lock for production-parity diagnostics."""

from pair_diagnostics import DIAGNOSTIC_COLUMNS


def test_diagnostic_schema_names_are_unique():
    assert len(DIAGNOSTIC_COLUMNS) == len(set(DIAGNOSTIC_COLUMNS))


def test_diagnostic_schema_rejects_removed_legacy_fields():
    assert not set(DIAGNOSTIC_COLUMNS).intersection({
        "score_d4", "score_invariant_only", "score_rotation_search",
        "best_d4_transform", "component_count_score", "router_method",
        "score_routed", "scorer_profile",
    })


def test_diagnostic_schema_requires_agreed_production_names():
    assert {
        "score", "selected_metric", "best_angle", "best_flip",
        "align_soft_iou", "mask_iou", "block_occupied_fraction",
        "slide_occupied_fraction", "router_size_signal",
        "true_vs_best_wrong_margin",
    } <= set(DIAGNOSTIC_COLUMNS)
