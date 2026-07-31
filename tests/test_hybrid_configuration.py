"""TDD for the #249 Hybrid Configuration loader/validator (production side).

CONTEXT.md "Hybrid Configuration": missing or incompatible configuration
must prevent Hybrid startup, loudly. `code/session/hybrid_configuration.py`
is the pure parser (`parse_hybrid_configuration`) plus its thin file-reading
shell (`load_hybrid_configuration`); this module never imports
`tools/scoring_diagnostics` (see `tests/test_architecture_boundaries.py`).

#245/#242: the curated calibration manifest and real handoff do not exist
yet, so every test here uses a synthetic handoff built from the SAME
production descriptor catalog and implementation-hash helper the loader
itself calls -- proving the loader's OWN contract, not a value that merely
matches a hardcoded expectation.
"""
from __future__ import annotations

import json
import math

import pytest

from session.hybrid_configuration import (
    HYBRID_CONFIGURATION_SCHEMA_VERSION,
    REQUIRED_FALLBACK_IDS,
    HybridConfigurationError,
    current_implementation_hashes,
    known_descriptor_names,
    load_hybrid_configuration,
    parse_hybrid_configuration,
)
from session.hybrid_configuration import _ALLOWED_HANDOFF_STATUSES, _ARCHITECTURE_KINDS
from verify.invariant_descriptors import descriptor_catalog

from integration_handoff import (
    HANDOFF_SCHEMA_VERSION,
    build_integration_handoff,
    write_integration_handoff,
)
from integration_handoff import REQUIRED_FALLBACK_IDS as DIAGNOSTIC_REQUIRED_FALLBACK_IDS
from candidate_retrieval_analysis import (
    Architecture,
    ArchitectureKind,
    BandEvaluation,
    VetoCalibration,
)


def _real_descriptor_spec():
    return descriptor_catalog()[0]


def _valid_payload() -> dict:
    spec = _real_descriptor_spec()
    return {
        "schema_version": HYBRID_CONFIGURATION_SCHEMA_VERSION,
        "architecture": {"kind": "individual", "name": spec.name, "methods": [spec.name]},
        "descriptor_recipe": [
            {
                "name": spec.name, "version": spec.version, "dimension": spec.dimension,
                "comparison": spec.comparison, "prior_evidence": spec.prior_evidence,
            }
        ],
        "candidate_band_thresholds": {spec.name: 0.1},
        "veto": {"enabled": False, "threshold": None, "reason": "disabled: no safe threshold"},
        "candidate_evidence": {
            "mean_candidate_count": 2.0, "median_candidate_count": 2.0,
            "p95_candidate_count": 3, "max_candidate_count": 4,
            "observed_runtime_seconds": 0.02, "estimated_runtime_seconds": 0.02,
            "full_comparison_reduction": 0.6,
        },
        "known_misses": ["W::S3"],
        "weak_stratum": {"field": "tissue", "group": "lung", "coverage": 0.8},
        "required_fallbacks": list(REQUIRED_FALLBACK_IDS),
        "provenance": {
            "manifest_path": "retrieval_manifest.csv", "manifest_hash": "a" * 64,
            "code_revision": "deadbeef",
            "implementation_hashes": current_implementation_hashes(),
            "calibration_run_id": "run-1",
        },
        "status": "proof_of_concept_not_production_approved",
    }


# --------------------------------------------------------------------------
# Schema-version agreement between the two independent halves of the seam
# --------------------------------------------------------------------------


def test_diagnostic_and_production_schema_version_constants_agree():
    assert HANDOFF_SCHEMA_VERSION == HYBRID_CONFIGURATION_SCHEMA_VERSION


# --------------------------------------------------------------------------
# Pure parser: valid payload
# --------------------------------------------------------------------------


def test_parse_valid_synthetic_handoff_returns_hybrid_configuration():
    config = parse_hybrid_configuration(
        _valid_payload(),
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
    )
    spec = _real_descriptor_spec()
    assert config.schema_version == HYBRID_CONFIGURATION_SCHEMA_VERSION
    assert config.architecture_kind == "individual"
    assert config.architecture_methods == (spec.name,)
    assert config.candidate_band_thresholds == {spec.name: 0.1}
    assert config.veto.enabled is False
    assert config.candidate_evidence.p95_candidate_count == 3
    assert config.known_misses == ("W::S3",)
    assert config.weak_stratum.field == "tissue"
    assert set(REQUIRED_FALLBACK_IDS) <= set(config.required_fallbacks)
    assert config.provenance.calibration_run_id == "run-1"
    assert config.status == "proof_of_concept_not_production_approved"


def test_parse_valid_synthetic_handoff_with_null_weak_stratum():
    payload = _valid_payload()
    payload["weak_stratum"] = None
    config = parse_hybrid_configuration(
        payload,
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
    )
    assert config.weak_stratum is None


# --------------------------------------------------------------------------
# Emitter -> loader round trip (the actual production/diagnostic seam)
# --------------------------------------------------------------------------


def test_synthetic_handoff_round_trips_emitter_to_loader(tmp_path):
    spec = _real_descriptor_spec()
    architecture = Architecture(ArchitectureKind.INDIVIDUAL, spec.name, (spec.name,))
    veto = VetoCalibration(False, None, (), (), "disabled: synthetic")
    candidate_evidence = BandEvaluation(10, 10, (1, 2, 2, 3), ())
    provenance = {
        "manifest_path": "retrieval_manifest.csv", "manifest_hash": "b" * 64,
        "code_revision": "cafebabe",
        "implementation_hashes": current_implementation_hashes(),
    }
    handoff = build_integration_handoff(
        architecture=architecture,
        descriptor_catalog=descriptor_catalog(),
        candidate_band_thresholds={spec.name: 0.15},
        veto=veto,
        candidate_evidence=candidate_evidence,
        efficiency={
            "observed_runtime": 0.5, "estimated_runtime": 0.4,
            "full_comparison_reduction": 0.7,
        },
        known_misses=["W::S9"],
        weak_stratum=None,
        provenance=provenance,
        calibration_run_id="run-xyz",
    )
    path = tmp_path / "handoff.json"
    write_integration_handoff(path, handoff)

    config = load_hybrid_configuration(path)

    assert config.architecture_name == spec.name
    assert config.architecture_methods == (spec.name,)
    assert config.candidate_band_thresholds == {spec.name: 0.15}
    assert (
        config.candidate_evidence.mean_candidate_count
        == candidate_evidence.mean_candidate_count
    )
    assert config.known_misses == ("W::S9",)
    assert config.provenance.calibration_run_id == "run-xyz"
    assert config.status == "proof_of_concept_not_production_approved"


# --------------------------------------------------------------------------
# Missing file / malformed JSON
# --------------------------------------------------------------------------


def test_load_missing_file_raises_hybrid_configuration_error(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(HybridConfigurationError) as exc_info:
        load_hybrid_configuration(missing)
    assert str(missing) in str(exc_info.value)


def test_load_malformed_json_raises_hybrid_configuration_error(tmp_path):
    path = tmp_path / "handoff.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HybridConfigurationError) as exc_info:
        load_hybrid_configuration(path)
    assert "not valid JSON" in str(exc_info.value)


def test_load_truncated_json_raises_hybrid_configuration_error(tmp_path):
    path = tmp_path / "handoff.json"
    full = json.dumps(_valid_payload())
    path.write_text(full[: len(full) // 2], encoding="utf-8")
    with pytest.raises(HybridConfigurationError):
        load_hybrid_configuration(path)


# --------------------------------------------------------------------------
# Schema version mismatch: names both expected and found
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_version", [0, 2, 999])
def test_wrong_schema_version_names_expected_and_found(bad_version):
    payload = _valid_payload()
    payload["schema_version"] = bad_version
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    message = str(exc_info.value)
    assert str(HYBRID_CONFIGURATION_SCHEMA_VERSION) in message
    assert str(bad_version) in message


# --------------------------------------------------------------------------
# Required top-level fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "architecture", "descriptor_recipe", "candidate_band_thresholds", "veto",
        "candidate_evidence", "known_misses", "weak_stratum", "required_fallbacks",
        "provenance", "status",
    ],
)
def test_missing_required_top_level_field_is_rejected(field):
    payload = _valid_payload()
    del payload[field]
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    assert field in str(exc_info.value)


# --------------------------------------------------------------------------
# Descriptor absent from the production catalog
# --------------------------------------------------------------------------


def test_unknown_descriptor_in_architecture_methods_is_rejected():
    payload = _valid_payload()
    payload["architecture"]["methods"] = ["not_a_real_descriptor"]
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    assert "not_a_real_descriptor" in str(exc_info.value)


def test_unknown_descriptor_in_recipe_is_rejected():
    payload = _valid_payload()
    payload["descriptor_recipe"][0]["name"] = "not_a_real_descriptor"
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    assert "not_a_real_descriptor" in str(exc_info.value)


# --------------------------------------------------------------------------
# Provenance mismatch: stale implementation hashes
# --------------------------------------------------------------------------


def test_stale_implementation_hash_is_rejected():
    payload = _valid_payload()
    payload["provenance"]["implementation_hashes"]["scorer"] = "stale-hash"
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    message = str(exc_info.value)
    assert "stale" in message.lower()
    assert "scorer" in message


def test_missing_implementation_hash_key_is_rejected():
    payload = _valid_payload()
    del payload["provenance"]["implementation_hashes"]["evaluator"]
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    assert "evaluator" in str(exc_info.value)


# --------------------------------------------------------------------------
# Required safety fallback declarations
# --------------------------------------------------------------------------


def test_missing_required_fallback_declaration_is_rejected():
    payload = _valid_payload()
    payload["required_fallbacks"] = [
        fallback for fallback in payload["required_fallbacks"]
        if fallback != "claim_inserted_separately_from_candidate_band"
    ]
    with pytest.raises(HybridConfigurationError) as exc_info:
        parse_hybrid_configuration(
            payload,
            known_descriptor_names=known_descriptor_names(),
            current_implementation_hashes=current_implementation_hashes(),
        )
    assert "claim_inserted_separately_from_candidate_band" in str(exc_info.value)


# --------------------------------------------------------------------------
# Extra-fields / forward-compatibility policy: unknown fields tolerated
# --------------------------------------------------------------------------


def test_unknown_top_level_field_is_tolerated_not_rejected():
    payload = _valid_payload()
    payload["a_future_diagnostic_field_not_yet_defined"] = {"anything": 1}
    config = parse_hybrid_configuration(
        payload,
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
    )
    assert config.extra_fields["a_future_diagnostic_field_not_yet_defined"] == {"anything": 1}


def test_unknown_nested_veto_field_is_tolerated_not_rejected():
    payload = _valid_payload()
    payload["veto"]["a_future_field"] = "ignored"
    config = parse_hybrid_configuration(
        payload,
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
    )
    assert config.veto.enabled is False


def _parse(payload):
    return parse_hybrid_configuration(
        payload,
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
    )


def _two_real_descriptor_specs():
    specs = descriptor_catalog()[:2]
    assert len(specs) >= 2, "fixture assumes at least two catalog descriptors"
    return specs


def _recipe_entry(spec):
    return {
        "name": spec.name, "version": spec.version, "dimension": spec.dimension,
        "comparison": spec.comparison, "prior_evidence": spec.prior_evidence,
    }


def _fusion_payload() -> dict:
    """A FUSION-kind payload: thresholds are keyed by the literal "fusion",
    never by the individual method names (`candidate_retrieval_analysis
    ._threshold_keys`) -- used to prove F2's per-kind invariant is actually
    kind-aware rather than blindly requiring one threshold per method name.
    """
    spec_a, spec_b = _two_real_descriptor_specs()
    payload = _valid_payload()
    payload["architecture"] = {
        "kind": "fusion", "name": "equal_rank_fusion", "methods": [spec_a.name, spec_b.name],
    }
    payload["descriptor_recipe"] = [_recipe_entry(spec_a), _recipe_entry(spec_b)]
    payload["candidate_band_thresholds"] = {"fusion": 0.1}
    return payload


def _union_payload() -> dict:
    """A UNION-kind payload: thresholds ARE keyed by every method name
    (unlike FUSION), per `candidate_retrieval_analysis._threshold_keys`.
    """
    spec_a, spec_b = _two_real_descriptor_specs()
    payload = _valid_payload()
    payload["architecture"] = {
        "kind": "union", "name": "candidate_union", "methods": [spec_a.name, spec_b.name],
    }
    payload["descriptor_recipe"] = [_recipe_entry(spec_a), _recipe_entry(spec_b)]
    payload["candidate_band_thresholds"] = {spec_a.name: 0.1, spec_b.name: 0.2}
    return payload


# --------------------------------------------------------------------------
# F1: every malformed payload raises HybridConfigurationError, never a bare
# ValueError/TypeError from an unguarded int()/float()/iteration coercion.
# --------------------------------------------------------------------------


def test_hybrid_configuration_error_no_longer_subclasses_value_error():
    assert not issubclass(HybridConfigurationError, ValueError)
    assert issubclass(HybridConfigurationError, Exception)


def test_string_candidate_band_threshold_raises_configuration_error_not_value_error():
    payload = _valid_payload()
    spec = _real_descriptor_spec()
    payload["candidate_band_thresholds"] = {spec.name: "abc"}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "candidate_band_thresholds" in str(exc_info.value)


def test_known_misses_non_iterable_raises_configuration_error_not_type_error():
    payload = _valid_payload()
    payload["known_misses"] = 5
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "known_misses" in str(exc_info.value)


def test_architecture_methods_non_iterable_raises_configuration_error_not_type_error():
    payload = _valid_payload()
    payload["architecture"]["methods"] = 5
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "architecture.methods" in str(exc_info.value)


def test_required_fallbacks_non_iterable_raises_configuration_error():
    payload = _valid_payload()
    payload["required_fallbacks"] = 5
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "required_fallbacks" in str(exc_info.value)


def test_descriptor_recipe_non_iterable_raises_configuration_error():
    payload = _valid_payload()
    payload["descriptor_recipe"] = 5
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "descriptor_recipe" in str(exc_info.value)


def test_empty_descriptor_recipe_is_rejected():
    """#250 review F2: an empty `descriptor_recipe` was previously accepted
    (only `architecture.methods` was checked non-empty), and flowed all the
    way to `ProcessingStore.freeze_hybrid_pool` building fingerprints of `{}`
    for every block -- a pool that silently freezes with zero comparable
    evidence. Must fail loudly here, at startup, like every other malformed
    handoff field."""
    payload = _valid_payload()
    payload["descriptor_recipe"] = []
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "descriptor_recipe" in str(exc_info.value)


def test_descriptor_recipe_dimension_non_numeric_raises_configuration_error():
    payload = _valid_payload()
    payload["descriptor_recipe"][0]["dimension"] = "abc"
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "dimension" in str(exc_info.value)


def test_veto_threshold_non_numeric_raises_configuration_error():
    payload = _valid_payload()
    payload["veto"] = {"enabled": False, "threshold": "abc", "reason": "bad"}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "veto.threshold" in str(exc_info.value)


def test_candidate_evidence_field_non_numeric_raises_configuration_error():
    payload = _valid_payload()
    payload["candidate_evidence"]["mean_candidate_count"] = "abc"
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "mean_candidate_count" in str(exc_info.value)


# --------------------------------------------------------------------------
# F2: candidate_band_thresholds contents (empty / non-finite / non-positive /
# per-ArchitectureKind key coverage)
# --------------------------------------------------------------------------


def test_empty_candidate_band_thresholds_is_rejected():
    payload = _valid_payload()
    payload["candidate_band_thresholds"] = {}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "candidate_band_thresholds" in str(exc_info.value)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, 0.0, -0.1])
def test_non_finite_or_non_positive_candidate_band_threshold_is_rejected(bad_value):
    payload = _valid_payload()
    spec = _real_descriptor_spec()
    payload["candidate_band_thresholds"] = {spec.name: bad_value}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "candidate_band_thresholds" in str(exc_info.value)


def test_numeric_string_candidate_band_threshold_is_rejected():
    """F2 judgment call: a threshold must be a real JSON number, not a
    numeric string -- the handoff is machine-emitted, so a string here is
    either hand-editing or a serialization bug, not a legitimate value."""
    payload = _valid_payload()
    spec = _real_descriptor_spec()
    payload["candidate_band_thresholds"] = {spec.name: "0.1"}
    with pytest.raises(HybridConfigurationError):
        _parse(payload)


def test_individual_kind_requires_threshold_named_for_its_one_method():
    payload = _valid_payload()
    payload["candidate_band_thresholds"] = {"some_other_descriptor": 0.1}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    spec = _real_descriptor_spec()
    assert spec.name in str(exc_info.value)


def test_union_kind_requires_a_threshold_per_method_name():
    payload = _union_payload()
    spec_a, _spec_b = _two_real_descriptor_specs()
    del payload["candidate_band_thresholds"][spec_a.name]
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert spec_a.name in str(exc_info.value)


def test_union_kind_accepts_a_threshold_per_method_name():
    config = _parse(_union_payload())
    assert config.architecture_kind == "union"


def test_fusion_kind_accepts_a_single_fusion_threshold_not_per_method_names():
    """F2 "verify before enforcing": `candidate_retrieval_analysis
    ._threshold_keys` keys a FUSION architecture's thresholds by the literal
    "fusion", never by its member method names -- a rule that required a
    threshold per `architecture.methods` name would wrongly reject this
    valid, actually-emitted handoff shape.
    """
    config = _parse(_fusion_payload())
    assert config.architecture_kind == "fusion"
    assert config.candidate_band_thresholds == {"fusion": 0.1}


def test_fusion_kind_missing_fusion_threshold_is_rejected():
    payload = _fusion_payload()
    payload["candidate_band_thresholds"] = {}
    with pytest.raises(HybridConfigurationError):
        _parse(payload)


# --------------------------------------------------------------------------
# F3: veto.enabled=true coupled to a finite, positive veto.threshold
# --------------------------------------------------------------------------


def test_veto_enabled_with_null_threshold_is_rejected():
    payload = _valid_payload()
    payload["veto"] = {"enabled": True, "threshold": None, "reason": "on"}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "veto" in str(exc_info.value)


@pytest.mark.parametrize("bad_threshold", [math.nan, math.inf, -math.inf, 0.0, -0.2])
def test_veto_enabled_with_non_finite_or_non_positive_threshold_is_rejected(bad_threshold):
    payload = _valid_payload()
    payload["veto"] = {"enabled": True, "threshold": bad_threshold, "reason": "on"}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "veto" in str(exc_info.value)


def test_veto_enabled_with_finite_positive_threshold_is_accepted():
    payload = _valid_payload()
    payload["veto"] = {"enabled": True, "threshold": 0.3, "reason": "on"}
    config = _parse(payload)
    assert config.veto.enabled is True
    assert config.veto.threshold == 0.3


def test_veto_disabled_with_null_threshold_is_still_accepted():
    config = _parse(_valid_payload())
    assert config.veto.enabled is False
    assert config.veto.threshold is None


# --------------------------------------------------------------------------
# F4: status must be on an explicit allowlist
# --------------------------------------------------------------------------


def test_unknown_status_is_rejected():
    payload = _valid_payload()
    payload["status"] = "approved_for_production"
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "approved_for_production" in str(exc_info.value)


def test_status_allowlist_matches_the_emitter_proof_of_concept_status():
    from integration_handoff import PROOF_OF_CONCEPT_STATUS

    assert _ALLOWED_HANDOFF_STATUSES == frozenset({PROOF_OF_CONCEPT_STATUS})


# --------------------------------------------------------------------------
# F5: boolean/float coercion cannot flip a field to mean its opposite
# --------------------------------------------------------------------------


def test_schema_version_true_is_rejected_not_coerced_to_one():
    payload = _valid_payload()
    payload["schema_version"] = True
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "schema_version" in str(exc_info.value)


def test_schema_version_float_is_rejected():
    payload = _valid_payload()
    payload["schema_version"] = 1.0
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "schema_version" in str(exc_info.value)


def test_veto_enabled_string_no_is_rejected_not_coerced_to_true():
    payload = _valid_payload()
    payload["veto"] = {"enabled": "no", "threshold": None, "reason": "off"}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "veto.enabled" in str(exc_info.value)


# --------------------------------------------------------------------------
# F8: candidate_evidence count fields are required, not defaulted to 0
# --------------------------------------------------------------------------


def test_empty_candidate_evidence_is_rejected():
    payload = _valid_payload()
    payload["candidate_evidence"] = {}
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert "candidate_evidence" in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    [
        "mean_candidate_count", "median_candidate_count",
        "p95_candidate_count", "max_candidate_count",
    ],
)
def test_missing_candidate_evidence_count_field_is_rejected(field):
    payload = _valid_payload()
    del payload["candidate_evidence"][field]
    with pytest.raises(HybridConfigurationError) as exc_info:
        _parse(payload)
    assert field in str(exc_info.value)


def test_missing_candidate_evidence_runtime_fields_are_tolerated_as_optional():
    """observed/estimated runtime and full_comparison_reduction are legitimately
    optional: `retrieval_evidence._efficiency_summary` can genuinely have no
    runtime measurement, unlike the always-computed count fields above."""
    payload = _valid_payload()
    del payload["candidate_evidence"]["observed_runtime_seconds"]
    del payload["candidate_evidence"]["estimated_runtime_seconds"]
    del payload["candidate_evidence"]["full_comparison_reduction"]
    config = _parse(payload)
    assert config.candidate_evidence.observed_runtime_seconds is None
    assert config.candidate_evidence.estimated_runtime_seconds is None
    assert config.candidate_evidence.full_comparison_reduction is None


# --------------------------------------------------------------------------
# Constant-drift binding: production-side vocabulary vs diagnostic-side
# --------------------------------------------------------------------------


def test_required_fallback_ids_constants_agree():
    assert set(DIAGNOSTIC_REQUIRED_FALLBACK_IDS) == set(REQUIRED_FALLBACK_IDS)


def test_architecture_kind_vocabulary_agrees_with_diagnostic_enum():
    assert _ARCHITECTURE_KINDS == frozenset(kind.value for kind in ArchitectureKind)
