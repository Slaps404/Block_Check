"""TDD for the #249 diagnostic-side Hybrid integration handoff emitter.

`tools/scoring_diagnostics/integration_handoff.py` serializes an
already-selected `Architecture` + its calibration evidence (objects #242's
tooling already computes) into one versioned JSON artifact. Building it must
never touch images, the accurate matcher, or manifest loading -- see
`test_building_handoff_never_touches_image_preparation_or_scoring` below --
so adding the emitter cannot change any existing #242 analysis result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from candidate_retrieval_analysis import (
    Architecture,
    ArchitectureKind,
    BandEvaluation,
    VetoCalibration,
)
from integration_handoff import (
    HANDOFF_SCHEMA_VERSION,
    PROOF_OF_CONCEPT_STATUS,
    REQUIRED_FALLBACK_IDS,
    build_integration_handoff,
    write_integration_handoff,
)
from verify.invariant_descriptors import descriptor_catalog


def _architecture():
    spec = descriptor_catalog()[0]
    return spec, Architecture(ArchitectureKind.INDIVIDUAL, spec.name, (spec.name,))


def _build(**overrides):
    spec, architecture = _architecture()
    kwargs = dict(
        architecture=architecture,
        descriptor_catalog=descriptor_catalog(),
        candidate_band_thresholds={spec.name: 0.2},
        veto=VetoCalibration(True, 0.3, ("W::S1",), (), "safe REVIEW-only veto"),
        candidate_evidence=BandEvaluation(8, 7, (1, 2, 3), ("W::S9",)),
        efficiency={
            "observed_runtime": 1.23, "estimated_runtime": 0.98,
            "full_comparison_reduction": 0.65,
        },
        known_misses=["W::S9"],
        weak_stratum={"field": "tissue", "group": "esophagus", "coverage": 0.7},
        provenance={
            "manifest_path": "manifest.csv", "manifest_hash": "c" * 64,
            "code_revision": "abc123", "implementation_hashes": {"scorer": "hash"},
        },
        calibration_run_id="run-42",
    )
    kwargs.update(overrides)
    return build_integration_handoff(**kwargs), spec


# --------------------------------------------------------------------------
# Field-by-field schema contract
# --------------------------------------------------------------------------


def test_handoff_carries_an_explicit_schema_version():
    handoff, _spec = _build()
    assert handoff["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert isinstance(handoff["schema_version"], int)


def test_handoff_carries_selected_architecture_and_descriptor_recipe():
    handoff, spec = _build()
    assert handoff["architecture"] == {
        "kind": "individual", "name": spec.name, "methods": [spec.name],
    }
    assert len(handoff["descriptor_recipe"]) == 1
    assert handoff["descriptor_recipe"][0]["name"] == spec.name
    assert handoff["descriptor_recipe"][0]["version"] == spec.version


def test_handoff_omits_descriptor_recipe_entries_not_referenced_by_architecture():
    handoff, spec = _build()
    catalog_names = {item.name for item in descriptor_catalog()}
    assert len(catalog_names) > 1, "fixture assumes more than one catalog descriptor"
    assert {entry["name"] for entry in handoff["descriptor_recipe"]} == {spec.name}


def test_handoff_carries_fitted_candidate_band_thresholds():
    handoff, spec = _build()
    assert handoff["candidate_band_thresholds"] == {spec.name: 0.2}


def test_handoff_carries_veto_status_and_threshold():
    handoff, _spec = _build()
    assert handoff["veto"] == {
        "enabled": True, "threshold": 0.3, "reason": "safe REVIEW-only veto",
    }


def test_handoff_carries_candidate_count_and_runtime_evidence():
    handoff, _spec = _build()
    evidence = handoff["candidate_evidence"]
    assert evidence["mean_candidate_count"] == pytest.approx(2.0)
    assert evidence["p95_candidate_count"] == 3
    assert evidence["max_candidate_count"] == 3
    assert evidence["observed_runtime_seconds"] == 1.23
    assert evidence["estimated_runtime_seconds"] == 0.98
    assert evidence["full_comparison_reduction"] == 0.65


def test_handoff_carries_known_misses_and_weak_strata():
    handoff, _spec = _build()
    assert handoff["known_misses"] == ["W::S9"]
    assert handoff["weak_stratum"] == {
        "field": "tissue", "group": "esophagus", "coverage": 0.7,
    }


def test_handoff_weak_stratum_is_null_when_none_provided():
    handoff, _spec = _build(weak_stratum=None)
    assert handoff["weak_stratum"] is None


def test_handoff_carries_required_fallbacks():
    handoff, _spec = _build()
    assert set(handoff["required_fallbacks"]) == set(REQUIRED_FALLBACK_IDS)


def test_handoff_carries_manifest_identity_scorer_identity_and_code_provenance():
    handoff, _spec = _build()
    provenance = handoff["provenance"]
    assert provenance["manifest_path"] == "manifest.csv"
    assert provenance["manifest_hash"] == "c" * 64
    assert provenance["code_revision"] == "abc123"
    assert provenance["implementation_hashes"] == {"scorer": "hash"}
    assert provenance["calibration_run_id"] == "run-42"


def test_handoff_status_is_explicitly_not_production_approved():
    handoff, _spec = _build()
    assert handoff["status"] == PROOF_OF_CONCEPT_STATUS
    assert "not_production_approved" in handoff["status"]


def test_handoff_is_json_serializable_round_trip(tmp_path):
    handoff, _spec = _build()
    path = tmp_path / "handoff.json"
    write_integration_handoff(path, handoff)
    assert json.loads(path.read_text(encoding="utf-8")) == handoff


def test_write_integration_handoff_writes_atomically(tmp_path, monkeypatch):
    """F10 (#249 review): must use `session.atomic_io.atomic_json`, not a
    plain `write_text`, so a crash/power-loss mid-write can never leave a
    truncated handoff at the path production's loader reads from."""
    import integration_handoff

    calls = []
    monkeypatch.setattr(
        integration_handoff,
        "atomic_json",
        lambda path, value: calls.append((path, value)),
    )
    handoff, _spec = _build()
    path = tmp_path / "handoff.json"
    write_integration_handoff(path, handoff)
    assert len(calls) == 1
    written_path, written_value = calls[0]
    assert Path(written_path) == path
    assert written_value == handoff


# --------------------------------------------------------------------------
# Emitting the handoff must not change any existing #242 analysis result
# --------------------------------------------------------------------------


def test_building_handoff_never_touches_image_preparation_or_scoring(monkeypatch):
    def _fail(*_args, **_kwargs):
        pytest.fail("building the handoff touched image preparation or scoring")

    monkeypatch.setattr("retrieval_evidence.prepare_specimen", _fail)
    monkeypatch.setattr("retrieval_evidence.build_locked_score_cache", _fail)
    monkeypatch.setattr("retrieval_evidence.score_routed_caches", _fail)
    handoff, _spec = _build()
    assert handoff["schema_version"] == HANDOFF_SCHEMA_VERSION
