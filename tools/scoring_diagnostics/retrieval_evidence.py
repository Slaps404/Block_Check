"""Build and validate self-contained raw evidence for retrieval experiments.

This module is deliberately the seam between image work and calibration.  The
builder may touch images and the production matcher.  Calibrators consume only
the JSON produced here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import zlib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from candidate_retrieval_analysis import (
    HybridAudit,
    VetoCalibration,
    architecture_veto_gaps,
    candidate_bands_for_architecture,
    calibrate_architecture_veto,
    hybrid_audit,
    hybrid_miss_diagnostics,
    nested_leave_one_work_order_out,
    recall_curve,
    standard_subgroup_band_evaluations,
    worst_subgroup,
)
from candidate_retrieval_report import write_report
from verify.invariant_descriptors import (
    DescriptorValue,
    build_descriptor_values,
    compare_descriptor_values,
    descriptor_catalog,
)
from retrieval_manifest import RetrievalManifest, Specimen
from session.preparation import PreparedSpecimen, prepare_specimen
from verify.gates import run_quality_gates
from verify.scorer import build_locked_score_cache, score_routed_caches
from verify.work_order_evaluator import evaluate_work_order


SCHEMA_VERSION = "retrieval-evidence-v1"
NORMALIZATION_MODE = "rms"


class CachedEvidenceError(ValueError):
    """A cache cannot safely support the requested analysis."""


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_hash(module_file: str | Path) -> str:
    path = Path(module_file)
    return _sha256(path) if path.is_file() else "unavailable"


def _code_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).parents[2]
    files = {
        "preparation": root / "code" / "session" / "preparation.py",
        "normalization": Path(__file__).with_name("robust_normalization.py"),
        "descriptors": root / "code" / "verify" / "invariant_descriptors.py",
        "gates": root / "code" / "verify" / "gates.py",
        "scorer": root / "code" / "verify" / "scorer.py",
        "evaluator": root / "code" / "verify" / "work_order_evaluator.py",
    }
    return {name: _module_hash(path) for name, path in files.items()}


def _descriptor_provenance() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in descriptor_catalog()]


def _provenance(manifest: RetrievalManifest) -> dict[str, Any]:
    specimens = (*manifest.blocks, *manifest.slides)
    return {
        "manifest_path": manifest.source_path,
        "manifest_hash": manifest.source_hash,
        "code_revision": _code_revision(),
        "normalization": NORMALIZATION_MODE,
        "descriptor_catalog": _descriptor_provenance(),
        "implementation_hashes": _implementation_hashes(),
        "work_order_identities": _work_order_identities(manifest),
        "image_identities": [
            {"role": item.role, "id": item.specimen_id,
             "work_order": item.work_order, "path": item.path,
             "sha256": _sha256(item.path)}
            for item in specimens
        ],
    }


def _work_order_identities(manifest: RetrievalManifest) -> dict[str, str]:
    blocks = {item.specimen_id: item for item in manifest.blocks}
    slides = {item.specimen_id: item for item in manifest.slides}
    identities = {}
    for name, order in manifest.work_orders.items():
        payload = {
            "blocks": [
                {"id": item_id, "path": blocks[item_id].path,
                 "sha256": _sha256(blocks[item_id].path),
                 "metadata": dict(blocks[item_id].metadata)}
                for item_id in order.block_ids
            ],
            "slides": [
                {"id": item_id, "path": slides[item_id].path,
                 "sha256": _sha256(slides[item_id].path),
                 "metadata": dict(slides[item_id].metadata)}
                for item_id in order.slide_ids
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        identities[name] = hashlib.sha256(encoded).hexdigest()
    return identities


def _serialized_prepared(prepared: object) -> dict[str, Any]:
    if not hasattr(prepared, "mask"):
        return {"kind": "failure", "reason": str(getattr(prepared, "reason", "unknown"))}
    mask = np.asarray(prepared.mask, dtype=np.uint8)
    return {
        "kind": "prepared", "shape": list(mask.shape),
        "mask_zlib_base64": base64.b64encode(zlib.compress(mask.tobytes())).decode("ascii"),
        "roi_ok": bool(getattr(prepared, "roi_ok", True)),
        "roi_reason": str(getattr(prepared, "roi_reason", "")),
    }


def _deserialized_prepared(payload: Mapping[str, Any], role: str) -> object:
    cached = payload["prepared_cache"]
    if cached["kind"] == "failure":
        from session.preparation import PreparationFailure
        return PreparationFailure(role, cached["reason"])
    raw = zlib.decompress(base64.b64decode(cached["mask_zlib_base64"]))
    mask = np.frombuffer(raw, dtype=np.uint8).reshape(cached["shape"]).copy()
    return PreparedSpecimen(role, mask, bool(cached["roi_ok"]), str(cached["roi_reason"]))


def _specimen_payload(
    item: Specimen, prepared: object, locked_cache: object | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    payload: dict[str, Any] = {
        "id": item.specimen_id, "role": item.role,
        "work_order": item.work_order, "path": item.path,
        "metadata": dict(item.metadata), "prepared": False,
        "descriptors": {}, "construction_ns": {},
        "prepared_cache": _serialized_prepared(prepared),
    }
    if not isinstance(prepared, PreparedSpecimen) and not hasattr(prepared, "mask"):
        return payload, {}
    values = build_descriptor_values(locked_cache.normalized_mask)
    payload["prepared"] = True
    payload["descriptors"] = {
        name: value.vector.tolist() for name, value in values.items()
    }
    payload["construction_ns"] = {
        name: value.construction_ns for name, value in values.items()
    }
    return payload, values


def _accurate_score(
    block: object, slide: object, block_cache: object | None, slide_cache: object | None,
) -> tuple[float | None, str, str, int]:
    gate = run_quality_gates(block, slide)
    if not gate.passed or not hasattr(block, "mask") or not hasattr(slide, "mask"):
        return None, gate.stage, gate.reason, 0
    start = perf_counter_ns()
    result = score_routed_caches(block_cache, slide_cache)
    return float(result.score), gate.stage, gate.reason, perf_counter_ns() - start


def evidence_is_compatible(
    evidence: Mapping[str, Any], *, manifest: RetrievalManifest | None = None,
    expected_manifest_hash: str | None = None,
) -> bool:
    """Check all identities whose changes make cached calibration unsafe."""
    if evidence.get("schema_version") != SCHEMA_VERSION:
        return False
    if "heuristic_scores" not in evidence or "comparison_timing" not in evidence:
        return False
    provenance = evidence.get("provenance", {})
    manifest_hash = expected_manifest_hash or (manifest.source_hash if manifest else None)
    if manifest_hash and provenance.get("manifest_hash") != manifest_hash:
        return False
    if provenance.get("normalization") != NORMALIZATION_MODE:
        return False
    if provenance.get("descriptor_catalog") != _descriptor_provenance():
        return False
    if provenance.get("code_revision") != _code_revision():
        return False
    if provenance.get("implementation_hashes") != _implementation_hashes():
        return False
    if manifest:
        try:
            current = _provenance(manifest)["image_identities"]
        except OSError:
            return False
        if provenance.get("image_identities") != current:
            return False
    return True


def _implementation_compatible(evidence: Mapping[str, Any]) -> bool:
    provenance = evidence.get("provenance", {})
    return (
        evidence.get("schema_version") == SCHEMA_VERSION
        and provenance.get("normalization") == NORMALIZATION_MODE
        and provenance.get("descriptor_catalog") == _descriptor_provenance()
        and provenance.get("code_revision") == _code_revision()
        and provenance.get("implementation_hashes") == _implementation_hashes()
    )


def build_evidence(manifest: RetrievalManifest, output_path: str | Path) -> dict[str, Any]:
    """Prepare every unique specimen once, then save full gate/score evidence."""
    destination = Path(output_path)
    existing: dict[str, Any] = {}
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if evidence_is_compatible(existing, manifest=manifest):
            return existing
        if not _implementation_compatible(existing):
            existing = {}
    current_provenance = _provenance(manifest)
    old_images = {
        (item["role"], item["id"]): item
        for item in existing.get("provenance", {}).get("image_identities", [])
    }
    old_payloads = {
        (item["role"], item["id"]): item
        for item in existing.get("specimens", [])
    }
    current_images = {
        (item["role"], item["id"]): item
        for item in current_provenance["image_identities"]
    }
    prepared: dict[tuple[str, str], object] = {}
    locked_caches: dict[tuple[str, str], object] = {}
    payloads: list[dict[str, Any]] = []
    descriptor_values: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in (*manifest.blocks, *manifest.slides):
        key = (item.role, item.specimen_id)
        old_payload = old_payloads.get(key)
        reusable = (
            old_payload is not None
            and "prepared_cache" in old_payload
            and old_images.get(key) == current_images.get(key)
        )
        if reusable:
            prepared[key] = _deserialized_prepared(old_payload, item.role)
            payload = dict(old_payload)
            payload.update({"path": item.path, "work_order": item.work_order,
                            "metadata": dict(item.metadata)})
            values = {
                name: DescriptorValue(
                    np.asarray(vector, dtype=np.float64),
                    int(payload.get("construction_ns", {}).get(name, 0)),
                )
                for name, vector in payload.get("descriptors", {}).items()
            }
        else:
            prepared[key] = prepare_specimen(item.path, role=item.role)
            cache = None
            if hasattr(prepared[key], "mask"):
                cache = build_locked_score_cache(prepared[key])
                locked_caches[key] = cache
            payload, values = _specimen_payload(item, prepared[key], cache)
        payloads.append(payload)
        descriptor_values[key] = values
    scores: list[dict[str, Any]] = []
    heuristic_scores: list[dict[str, Any]] = []
    current_order_ids = current_provenance["work_order_identities"]
    old_order_ids = existing.get("provenance", {}).get("work_order_identities", {})
    unchanged_orders = {
        name for name, identity in current_order_ids.items()
        if old_order_ids.get(name) == identity
    }
    scores.extend(
        row for row in existing.get("accurate_scores", [])
        if row["work_order"] in unchanged_orders
    )
    heuristic_scores.extend(
        row for row in existing.get("heuristic_scores", [])
        if row["work_order"] in unchanged_orders
    )

    def locked_cache(key: tuple[str, str]) -> object | None:
        specimen = prepared[key]
        if not hasattr(specimen, "mask"):
            return None
        if key not in locked_caches:
            locked_caches[key] = build_locked_score_cache(specimen)
        return locked_caches[key]

    for work_order in manifest.work_orders.values():
        if work_order.work_order in unchanged_orders:
            continue
        for slide_id in work_order.slide_ids:
            slide = prepared[("slide", slide_id)]
            for block_id in work_order.block_ids:
                block = prepared[("block", block_id)]
                score, stage, reason, duration = _accurate_score(
                    block, slide, locked_cache(("block", block_id)),
                    locked_cache(("slide", slide_id)),
                )
                scores.append({"work_order": work_order.work_order, "slide_id": slide_id,
                               "block_id": block_id, "score": score, "gate_stage": stage,
                               "gate_reason": reason, "comparison_ns": duration})
                for spec in descriptor_catalog():
                    start = perf_counter_ns()
                    block_value = descriptor_values[
                        ("block", block_id)
                    ].get(spec.name)
                    slide_value = descriptor_values[
                        ("slide", slide_id)
                    ].get(spec.name)
                    heuristic = None
                    if block_value is not None and slide_value is not None:
                        heuristic = compare_descriptor_values(spec, slide_value, block_value)
                    elapsed = perf_counter_ns() - start
                    heuristic_scores.append({
                        "descriptor": spec.name, "work_order": work_order.work_order,
                        "slide_id": slide_id, "block_id": block_id, "score": heuristic,
                        "comparison_ns": elapsed,
                    })
    comparison_timing = {
        spec.name: {
            "comparison_ns": sum(
                row["comparison_ns"] for row in heuristic_scores
                if row["descriptor"] == spec.name
            ),
            "comparison_count": sum(
                row["descriptor"] == spec.name for row in heuristic_scores
            ),
        }
        for spec in descriptor_catalog()
    }
    evidence = {"schema_version": SCHEMA_VERSION, "provenance": current_provenance,
                "exclusions": [asdict(item) for item in manifest.exclusions],
                "specimens": payloads, "accurate_scores": scores,
                "heuristic_scores": heuristic_scores,
                "comparison_timing": comparison_timing}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def calibrate_cached_evidence(
    evidence_path: str | Path, *, manifest: RetrievalManifest | None = None,
    expected_manifest_hash: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return cached-matrix totals without preparing images or invoking scoring."""
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    if not evidence_is_compatible(evidence, manifest=manifest,
                                  expected_manifest_hash=expected_manifest_hash):
        raise CachedEvidenceError("cached evidence is incompatible with manifest or provenance")
    scores = evidence.get("accurate_scores", [])
    accurate, claims, heuristics = _cached_analysis_maps(evidence)
    router_groups = _router_groups(evidence)
    curves = [
        item
        for name, values in heuristics.items()
        for item in recall_curve(name, accurate, claims, values)
    ]
    work_orders = {slide_id: slide_id.split("::", maxsplit=1)[0] for slide_id in accurate}
    gaps = {name: (0.0, 0.05, 0.1, 0.2, 0.5, 1.0) for name in heuristics}
    gaps["fusion"] = gaps[next(iter(heuristics))] if heuristics else ()
    nested = nested_leave_one_work_order_out(
        accurate, claims, heuristics, work_orders, gaps,
        router_group_by_slide=router_groups or None,
    )
    audit, candidate_members = _nested_safety_audit(
        nested, accurate, claims, heuristics
    )
    vetoes = _calibrate_fold_vetoes(nested, accurate, claims, heuristics)
    veto = _veto_summary(vetoes)
    efficiency = _efficiency_summary(evidence, nested)
    result = {
        "score_count": len(scores),
        "valid_score_count": sum(row.get("score") is not None for row in scores),
        "gate_failed_count": sum(row.get("score") is None for row in scores),
        "recall_curves": [
            {"method": item.method, "k": item.k, "recall": item.recall,
             "evaluable_slides": item.evaluable_slides,
             "missed_slide_ids": list(item.missed_slide_ids)}
            for item in curves
        ],
        "new_false_pass_count": audit.new_false_pass_count,
        "inherited_false_pass_count": audit.inherited_false_pass_count,
        "safety_evaluable": audit.safety_evaluable,
        "nested_held_out_coverage": nested.held_out_coverage,
        "insufficient_generalization_warning": nested.insufficient_generalization_warning,
        "veto_by_fold": vetoes,
        "efficiency": efficiency,
        "evidence": evidence,
    }
    if report_path:
        misses = hybrid_miss_diagnostics(audit)
        metadata = _metadata_by_slide(evidence)
        subgroup_metrics = standard_subgroup_band_evaluations(
            accurate, metadata, accurate, claims, candidate_members,
        )
        subgroup_cuts = {
            field: [
                {"group": group, "coverage": metric.coverage,
                 "evaluable_slides": metric.evaluable_slides,
                 "mean_candidate_count": metric.mean_candidate_count}
                for group, metric in groups.items()
            ]
            for field, groups in subgroup_metrics.items()
        }
        worst = worst_subgroup(subgroup_metrics)
        worst_summary = None if worst is None else {
            "field": worst[0], "group": worst[1],
            "coverage": worst[2].coverage,
            "evaluable_slides": worst[2].evaluable_slides,
        }
        write_report(
            Path(report_path), provenance=evidence["provenance"],
            descriptor_catalog=evidence["provenance"]["descriptor_catalog"],
            recall_summaries=curves, audit=audit, veto=veto, misses=misses,
            recommendation="Exploratory calibration only, not production-ready.",
            nested_evaluation=nested,
            timing_summary=_timing_summary(evidence),
            subgroup_cuts=subgroup_cuts,
            worst_subgroup_summary=worst_summary,
            efficiency_summary=efficiency,
            veto_fold_results=vetoes,
        )
        result["report_path"] = str(Path(report_path))
    return result


def select_hybrid_handoff_inputs(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Pick one architecture/threshold set + supporting evidence for a #249 handoff.

    Reuses the same nested leave-one-work-order-out selection
    `calibrate_cached_evidence` already computes for calibration review, read
    from cached evidence only -- no image preparation or accurate scoring
    runs here, so calling this (or the `--handoff` CLI flag that wraps it)
    cannot change any existing #242 analysis result.

    Proof-of-concept scope only: today's tiny corpus has no separately
    "promoted" production fold, so this uses the LAST outer
    leave-one-work-order-out fold's selection as the one architecture/
    threshold set the handoff describes. Raises `CachedEvidenceError` when
    fewer than two independent work orders make any outer fold available --
    the same insufficiency `nested_leave_one_work_order_out` itself reports.

    Statistics honesty note (#249 review, F9): the returned artifact mixes
    two different aggregation scopes and does not label them as such.
    ``candidate_evidence`` and ``weak_stratum`` are computed from ONLY the
    last fold's held-out slides (`fold.held_out`, `fold.held_out_slide_ids`
    above) -- a single leave-one-work-order-out fold's worth of data. ``veto``
    (`_calibrate_fold_vetoes`/`_veto_summary`) and ``efficiency``
    (`_efficiency_summary`) are aggregated across ALL outer folds in
    `nested.folds`. So a caller reading the emitted handoff is looking at a
    single-fold candidate-evidence/weak-stratum number next to an all-fold
    veto/efficiency number, not four numbers computed the same way. One
    concrete consequence: `weak_stratum: None` can mean either "no weak
    stratum exists" OR "this one fold's held-out slice had too little data
    in any stratum to compute one" -- the two are indistinguishable from the
    artifact alone.
    """
    accurate, claims, heuristics = _cached_analysis_maps(evidence)
    router_groups = _router_groups(evidence)
    work_orders = {slide_id: slide_id.split("::", maxsplit=1)[0] for slide_id in accurate}
    gaps = {name: (0.0, 0.05, 0.1, 0.2, 0.5, 1.0) for name in heuristics}
    gaps["fusion"] = gaps[next(iter(heuristics))] if heuristics else ()
    nested = nested_leave_one_work_order_out(
        accurate, claims, heuristics, work_orders, gaps,
        router_group_by_slide=router_groups or None,
    )
    if not nested.folds:
        raise CachedEvidenceError(
            "cannot select a Hybrid architecture for a #249 handoff: at least "
            "two independent work orders (one outer leave-one-work-order-out "
            "fold) are required"
        )
    fold = nested.folds[-1]
    vetoes = _calibrate_fold_vetoes(nested, accurate, claims, heuristics)
    veto = _veto_summary(vetoes)
    efficiency = _efficiency_summary(evidence, nested)
    known_misses = sorted({
        slide for other_fold in nested.folds
        for slide in other_fold.held_out.missed_slide_ids
    })
    metadata = _metadata_by_slide(evidence)
    bands = candidate_bands_for_architecture(
        fold.selected, fold.held_out_slide_ids, heuristics, dict(fold.thresholds),
        router_by_slide=dict(fold.router_by_slide),
    )
    subgroup_metrics = standard_subgroup_band_evaluations(
        fold.held_out_slide_ids, metadata, accurate, claims, bands,
    )
    worst = worst_subgroup(subgroup_metrics)
    weak_stratum = None if worst is None else {
        "field": worst[0], "group": worst[1], "coverage": worst[2].coverage,
    }
    return {
        "architecture": fold.selected,
        "thresholds": dict(fold.thresholds),
        "veto": veto,
        "candidate_evidence": fold.held_out,
        "efficiency": efficiency,
        "known_misses": known_misses,
        "weak_stratum": weak_stratum,
        "provenance": evidence["provenance"],
    }


def _calibrate_fold_vetoes(
    nested: Any,
    accurate: Mapping[str, Mapping[str, float | None]],
    claims: Mapping[str, str],
    heuristics: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> list[dict[str, Any]]:
    baseline = {
        slide: evaluate_work_order(scores, claims[slide])
        for slide, scores in accurate.items()
    }
    rows = []
    for fold in nested.folds:
        calibrated = calibrate_architecture_veto(
            fold.selected, fold.training_slide_ids, heuristics, claims,
            baseline, claims, (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            router_by_slide=dict(fold.router_by_slide),
        )
        heldout_gaps = architecture_veto_gaps(
            fold.selected, fold.held_out_slide_ids, heuristics, claims,
            router_by_slide=dict(fold.router_by_slide),
        )
        heldout_vetoed = []
        heldout_false_reviews = []
        if calibrated.enabled and calibrated.threshold is not None:
            for slide in fold.held_out_slide_ids:
                gap = heldout_gaps.get(slide)
                if gap is None or gap < calibrated.threshold:
                    continue
                heldout_vetoed.append(slide)
                if baseline[slide].verdict == "PASS":
                    heldout_false_reviews.append(slide)
        safe = calibrated.enabled and not heldout_false_reviews
        rows.append({
            "held_out_order": fold.held_out_order, "enabled": safe,
            "training_enabled": calibrated.enabled,
            "threshold": calibrated.threshold, "reason": calibrated.reason,
            "vetoed_claims": list(calibrated.vetoed_claims),
            "training_false_reviews": list(calibrated.false_reviews),
            "heldout_vetoed_slides": heldout_vetoed,
            "heldout_false_reviews": heldout_false_reviews,
            "heldout_safe": safe,
        })
    return rows


def _veto_summary(rows: list[dict[str, Any]]) -> VetoCalibration:
    unsafe = [row for row in rows if row["heldout_false_reviews"]]
    if unsafe:
        return VetoCalibration(
            False, None, (),
            tuple(sorted({slide for row in unsafe
                          for slide in row["heldout_false_reviews"]})),
            "disabled: frozen outer-held-out veto caused False REVIEW",
        )
    enabled = [row for row in rows if row["enabled"]]
    if not enabled:
        return VetoCalibration(
            False, None, (), (),
            "disabled: no outer-training fold found a safe useful threshold",
        )
    return VetoCalibration(
        True, None,
        tuple(sorted({claim for row in enabled for claim in row["vetoed_claims"]})),
        tuple(sorted({slide for row in rows for slide in row["heldout_false_reviews"]})),
        f"enabled in {len(enabled)}/{len(rows)} outer-training folds",
    )


def _nested_safety_audit(
    nested: Any,
    accurate: Mapping[str, Mapping[str, float | None]],
    claims: Mapping[str, str],
    heuristics: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> tuple[HybridAudit, dict[str, tuple[str, ...]]]:
    """Simulate every claim using each outer fold's frozen selected architecture."""
    audits = []
    all_members = {}
    for fold in nested.folds:
        held_out = fold.held_out_slide_ids
        bands = candidate_bands_for_architecture(
            fold.selected, held_out, heuristics, dict(fold.thresholds),
            router_by_slide=dict(fold.router_by_slide),
        )
        all_members.update(bands)
        subset = {slide: accurate[slide] for slide in held_out}
        subset_claims = {slide: claims[slide] for slide in held_out}
        audits.append(hybrid_audit(
            subset, subset_claims, {}, 0.0,
            confirmed_correct_by_slide=subset_claims, simulate_all_claims=True,
            candidate_members_by_slide=bands,
        ))
    if audits:
        audit = HybridAudit(tuple(row for item in audits for row in item.rows))
        return audit, all_members
    return HybridAudit(()), all_members


def _router_groups(evidence: Mapping[str, Any]) -> dict[str, str]:
    groups = {}
    for item in evidence["specimens"]:
        if item["role"] != "slide":
            continue
        metadata = item.get("metadata", {})
        group = metadata.get("tissue_raw") or metadata.get("tissue")
        if group:
            groups[f"{item['work_order']}::{item['id']}"] = str(group)
    return groups


def _metadata_by_slide(evidence: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for item in evidence["specimens"]:
        if item["role"] != "slide":
            continue
        raw = item.get("metadata", {})
        projected = {
            "tissue": raw.get("tissue_raw") or raw.get("tissue"),
            "morphology": raw.get("morphology"),
            "sparse_dense": raw.get("sparse_dense"),
            "capture_status": raw.get("capture_status"),
        }
        result[f"{item['work_order']}::{item['id']}"] = {
            key: str(value) for key, value in projected.items() if value not in (None, "")
        }
    return result


def _efficiency_summary(evidence: Mapping[str, Any], nested: Any) -> dict[str, Any]:
    full_calls = len(evidence.get("accurate_scores", []))
    rerank_calls = sum(
        count + 1 for fold in nested.folds
        for count in fold.held_out.candidate_counts
    )
    accurate_ns = sum(
        int(row.get("comparison_ns", 0)) for row in evidence.get("accurate_scores", [])
    )
    timing = _timing_summary(evidence)
    heuristic_ns = sum(timing["construction_ns_by_descriptor"].values()) + sum(
        row["comparison_ns"]
        for row in timing["comparison_by_descriptor"].values()
    )
    per_call = accurate_ns / full_calls if full_calls else 0.0
    estimated_ns = heuristic_ns + int(per_call * rerank_calls)
    return {
        "accurate_rerank_calls": rerank_calls,
        "full_accurate_calls": full_calls,
        "observed_runtime": accurate_ns / 1_000_000_000,
        "estimated_runtime": estimated_ns / 1_000_000_000,
        "full_comparison_reduction": (
            None if not full_calls else 1.0 - rerank_calls / full_calls
        ),
    }


def _timing_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    construction: dict[str, int] = {}
    for item in evidence["specimens"]:
        for name, duration in item.get("construction_ns", {}).items():
            construction[name] = construction.get(name, 0) + int(duration)
    return {
        "construction_ns_by_descriptor": construction,
        "comparison_by_descriptor": evidence.get("comparison_timing", {}),
    }


def _cached_analysis_maps(
    evidence: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, float | None]],
    dict[str, str],
    dict[str, dict[str, dict[str, float | None]]],
]:
    """Recreate slide-to-block maps from JSON without preparing or scoring images."""
    def specimen_key(item: Mapping[str, Any]) -> str:
        return f"{item['work_order']}::{item['id']}"

    slides = {
        specimen_key(item): item for item in evidence["specimens"]
        if item["role"] == "slide"
    }
    claims = {
        slide_key: f"{slide['work_order']}::{slide['metadata']['claim_block_id']}"
        for slide_key, slide in slides.items()
        if slide["metadata"].get("claim_block_id")
    }
    accurate = {slide_key: {} for slide_key in slides}
    for row in evidence["accurate_scores"]:
        slide_key = f"{row['work_order']}::{row['slide_id']}"
        block_key = f"{row['work_order']}::{row['block_id']}"
        accurate.setdefault(slide_key, {})[block_key] = row["score"]
    heuristics: dict[str, dict[str, dict[str, float | None]]] = {}
    for row in evidence.get("heuristic_scores", []):
        descriptor = row["descriptor"]
        slide_key = f"{row['work_order']}::{row['slide_id']}"
        block_key = f"{row['work_order']}::{row['block_id']}"
        heuristics.setdefault(descriptor, {}).setdefault(slide_key, {})[block_key] = row["score"]
    return accurate, claims, heuristics
