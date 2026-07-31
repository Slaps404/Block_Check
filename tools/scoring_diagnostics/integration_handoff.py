"""Versioned, machine-readable Hybrid integration handoff (#249).

#242's tooling already computes a selected retrieval `Architecture`, its
fitted candidate-band thresholds, an optional `VetoCalibration`, and
provenance-linked evidence -- but only as in-memory dataclasses that end up
rendered into `candidate_retrieval_report`'s prose. This module is the
missing serialization step: a pure function that packs those already-computed
objects into one versioned JSON artifact production can load and validate
(`code/session/hybrid_configuration.py`) without scraping a report or
hardcoding numbers.

Building the handoff never touches images, the accurate matcher, or manifest
loading -- it only reads objects `retrieval_evidence.calibrate_cached_evidence`
(or `select_hybrid_handoff_inputs`) already produced from cached evidence, so
calling it cannot change any existing #242 analysis result.

Dependency direction: this module lives under `tools/scoring_diagnostics`
(diagnostics) and imports the production-owned descriptor catalog
(`verify.invariant_descriptors`) -- the arrow stays diagnostics -> production,
never the reverse (#245/#246).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from candidate_retrieval_analysis import Architecture, VetoCalibration
from session.atomic_io import atomic_json
from verify.invariant_descriptors import DescriptorSpec

# Must equal `code/session/hybrid_configuration.HYBRID_CONFIGURATION_SCHEMA_VERSION`.
# Deliberately duplicated rather than imported -- production must not import
# diagnostic tooling, and diagnostics must not import production's *session*
# package (only its data/scoring modules) to keep this one small integer
# genuinely independent on both sides. `tests/test_integration_handoff.py`
# and `tests/test_hybrid_configuration.py` both assert the two constants
# agree, so drift is caught immediately rather than silently at Hybrid
# startup.
HANDOFF_SCHEMA_VERSION = 1

# Safety-critical fallback behaviors #245 requires (CONTEXT.md "Hybrid
# Configuration"). Must equal
# `code/session/hybrid_configuration.REQUIRED_FALLBACK_IDS` for the same
# independence reason as `HANDOFF_SCHEMA_VERSION` above.
REQUIRED_FALLBACK_IDS = (
    "complete_accurate_scoring_on_missing_fingerprint",
    "complete_accurate_scoring_on_invalid_candidate_output",
    "claim_inserted_separately_from_candidate_band",
)

# #242's proof-of-concept decision rule: passing its gates proves feasibility
# only. Every handoff this module builds carries this status literally, so
# no downstream consumer can present current evidence as production-approved.
PROOF_OF_CONCEPT_STATUS = "proof_of_concept_not_production_approved"


def build_integration_handoff(
    *,
    architecture: Architecture,
    descriptor_catalog: Sequence[DescriptorSpec],
    candidate_band_thresholds: Mapping[str, float],
    veto: VetoCalibration,
    candidate_evidence: Any,
    efficiency: Mapping[str, Any],
    known_misses: Sequence[str],
    weak_stratum: Optional[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    calibration_run_id: str,
) -> dict[str, Any]:
    """Serialize one selected architecture + its calibration evidence.

    ``candidate_evidence`` is a
    `candidate_retrieval_analysis.BandEvaluation` (duck-typed here so this
    module need not import it just for a type hint). ``provenance`` is the
    evidence-build provenance mapping already produced by
    `retrieval_evidence._provenance` (``evidence["provenance"]``);
    ``calibration_run_id`` additionally pins the handoff to one specific
    calibration invocation (e.g. a hash of the evidence file), so rerunning
    calibration against the same code/manifest still yields a distinguishable
    handoff.
    """
    referenced = set(architecture.methods)
    recipe = [asdict(spec) for spec in descriptor_catalog if spec.name in referenced]
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "architecture": {
            "kind": architecture.kind.value,
            "name": architecture.name,
            "methods": list(architecture.methods),
        },
        "descriptor_recipe": recipe,
        "candidate_band_thresholds": dict(candidate_band_thresholds),
        "veto": {
            "enabled": veto.enabled,
            "threshold": veto.threshold,
            "reason": veto.reason,
        },
        "candidate_evidence": {
            "mean_candidate_count": candidate_evidence.mean_candidate_count,
            "median_candidate_count": candidate_evidence.median_candidate_count,
            "p95_candidate_count": candidate_evidence.p95_candidate_count,
            "max_candidate_count": candidate_evidence.max_candidate_count,
            "observed_runtime_seconds": efficiency.get("observed_runtime"),
            "estimated_runtime_seconds": efficiency.get("estimated_runtime"),
            "full_comparison_reduction": efficiency.get("full_comparison_reduction"),
        },
        "known_misses": list(known_misses),
        "weak_stratum": dict(weak_stratum) if weak_stratum is not None else None,
        "required_fallbacks": list(REQUIRED_FALLBACK_IDS),
        "provenance": {
            "manifest_path": provenance.get("manifest_path"),
            "manifest_hash": provenance.get("manifest_hash"),
            "code_revision": provenance.get("code_revision"),
            "implementation_hashes": dict(provenance.get("implementation_hashes", {})),
            "calibration_run_id": calibration_run_id,
        },
        "status": PROOF_OF_CONCEPT_STATUS,
    }


def write_integration_handoff(path: str | Path, handoff: Mapping[str, Any]) -> None:
    """Write ``handoff`` (see :func:`build_integration_handoff`) as pretty JSON.

    Uses `session.atomic_io.atomic_json` (F10 of #249's review) rather than a
    plain `write_text`: a crash or power loss mid-write must never leave a
    truncated/partial handoff at the path production's loader reads from.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, handoff)
