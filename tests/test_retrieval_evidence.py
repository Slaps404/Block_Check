# flake8: noqa
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retrieval_evidence import (
    CachedEvidenceError,
    _calibrate_fold_vetoes,
    _veto_summary,
    build_evidence,
    calibrate_cached_evidence,
    select_hybrid_handoff_inputs,
)
from verify.invariant_descriptors import descriptor_catalog
from retrieval_manifest import load_retrieval_manifest
from candidate_retrieval_analysis import VetoCalibration


def _manifest(tmp_path, cross_orders=False):
    paths = []
    for name in ("b.png", "s1.png", "s2.png"):
        path = tmp_path / name; path.touch(); paths.append(path)
    manifest = tmp_path / "manifest.csv"
    header = (
        "work_order,block_id,block_path,slide_id,slide_path,row_id,claim_id,"
        "set_id,label_source,inclusion_status,capture_profile,capture_status,"
        "tissue_source,tissue_confidence\n"
    )
    common = ",set-1,manual,included,pi-v3,accepted,manual,1.0\n"
    manifest.write_text(
        header + f"W,B,{paths[0]},S1,{paths[1]},row-1,claim-1" + common
        + f"{'W2' if cross_orders else 'W'},B,{paths[0]},S2,{paths[2]},row-2,claim-2"
        + common
    )
    return load_retrieval_manifest(manifest)


def test_evidence_prepares_each_unique_specimen_once_and_preserves_none_score(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, cross_orders=True); calls = []; cache_calls = []
    descriptor_masks = []
    monkeypatch.setattr("retrieval_evidence.prepare_specimen", lambda path, role: calls.append((path, role)) or SimpleNamespace(mask=np.ones((4, 4), dtype=np.uint8)))
    monkeypatch.setattr(
        "retrieval_evidence.build_locked_score_cache",
        lambda specimen: cache_calls.append(specimen)
        or SimpleNamespace(normalized_mask=np.full((4, 4), 7, dtype=np.uint8)),
    )
    monkeypatch.setattr("retrieval_evidence.run_quality_gates", lambda *_: SimpleNamespace(passed=False, stage="pair", reason="bad"))
    from verify.invariant_descriptors import build_descriptor_values as real_build
    monkeypatch.setattr(
        "retrieval_evidence.build_descriptor_values",
        lambda mask: descriptor_masks.append(mask.copy()) or real_build(mask),
    )
    path = tmp_path / "evidence.json"
    evidence = build_evidence(manifest, path)
    assert len(calls) == 3
    assert len(cache_calls) == 3
    assert len(descriptor_masks) == 3
    assert all(np.all(mask == 7) for mask in descriptor_masks)
    assert [row["score"] for row in evidence["accurate_scores"]] == [None, None]
    assert len(evidence["heuristic_scores"]) == 2 * len(descriptor_catalog())
    assert all(row["comparison_ns"] >= 0 for row in evidence["heuristic_scores"])
    provenance = evidence["provenance"]
    assert provenance["code_revision"]
    assert set(provenance["implementation_hashes"]) == {
        "preparation", "normalization", "descriptors", "gates", "scorer", "evaluator",
    }
    assert json.loads(path.read_text())["schema_version"] == "retrieval-evidence-v1"


def test_calibration_reads_only_cached_evidence_and_provenance_change_rebuilds(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path); path = tmp_path / "evidence.json"
    monkeypatch.setattr("retrieval_evidence.prepare_specimen", lambda path, role: SimpleNamespace(mask=np.ones((4, 4), dtype=np.uint8)))
    monkeypatch.setattr("retrieval_evidence.run_quality_gates", lambda *_: SimpleNamespace(passed=False, stage="pair", reason="bad"))
    build_evidence(manifest, path)
    monkeypatch.setattr("retrieval_evidence.prepare_specimen", lambda *a, **k: pytest.fail("calibration prepared an image"))
    monkeypatch.setattr("retrieval_evidence.compare_descriptor_values", lambda *a: pytest.fail("calibration compared vectors"))
    summary = calibrate_cached_evidence(path)
    assert summary["score_count"] == 2
    report = tmp_path / "report.md"
    summary = calibrate_cached_evidence(path, report_path=report)
    assert report.is_file()
    assert "not production-promoted" in report.read_text()
    assert "recall_curves" in summary
    with monkeypatch.context() as context:
        context.setattr(
            "retrieval_evidence._implementation_hashes", lambda: {"changed": "hash"}
        )
        with pytest.raises(CachedEvidenceError, match="provenance"):
            calibrate_cached_evidence(path)
    changed = tmp_path / "changed.png"; changed.write_bytes(b"changed")
    with pytest.raises(CachedEvidenceError, match="manifest"):
        calibrate_cached_evidence(path, manifest=manifest, expected_manifest_hash="other")


def test_incremental_update_reuses_specimens_and_rebuilds_changed_work_order(
    tmp_path, monkeypatch,
):
    manifest = _manifest(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    calls = []
    monkeypatch.setattr(
        "retrieval_evidence.prepare_specimen",
        lambda path, role: calls.append((path, role))
        or SimpleNamespace(mask=np.ones((4, 4), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "retrieval_evidence.run_quality_gates",
        lambda *_: SimpleNamespace(passed=False, stage="pair", reason="bad"),
    )
    first = build_evidence(manifest, evidence_path)
    assert len(calls) == 3
    source = Path(manifest.source_path)
    source.write_text(source.read_text().replace("accepted", "reviewed", 1))
    changed_manifest = load_retrieval_manifest(source)
    calls.clear()
    second = build_evidence(changed_manifest, evidence_path)
    assert calls == []
    assert first["provenance"]["manifest_hash"] != second["provenance"]["manifest_hash"]
    assert len(second["accurate_scores"]) == 2


def test_veto_is_frozen_then_disabled_by_heldout_false_review(monkeypatch):
    fold = SimpleNamespace(
        selected=object(), training_slide_ids=("train",),
        held_out_slide_ids=("held",), held_out_order="WO-2", router_by_slide=(),
    )
    nested = SimpleNamespace(folds=(fold,))
    monkeypatch.setattr(
        "retrieval_evidence.evaluate_work_order",
        lambda scores, claim: SimpleNamespace(verdict="PASS"),
    )
    monkeypatch.setattr(
        "retrieval_evidence.calibrate_architecture_veto",
        lambda *args, **kwargs: VetoCalibration(
            True, 0.2, ("train",), (), "training-safe",
        ),
    )
    monkeypatch.setattr(
        "retrieval_evidence.architecture_veto_gaps",
        lambda *args, **kwargs: {"held": 0.3},
    )
    rows = _calibrate_fold_vetoes(
        nested, {"train": {"B": 0.9}, "held": {"B": 0.9}},
        {"train": "B", "held": "B"}, {},
    )
    assert rows[0]["threshold"] == 0.2
    assert rows[0]["heldout_false_reviews"] == ["held"]
    assert rows[0]["enabled"] is False
    assert _veto_summary(rows).enabled is False


def _hand_authored_evidence(order_count: int = 2) -> dict:
    """Small evidence dict shaped like `build_evidence`'s output, hand-authored
    so `select_hybrid_handoff_inputs` (#249) can be tested without preparing
    real images -- it only reads cached matrices, exactly like
    `calibrate_cached_evidence`."""
    orders = [("W1", "S1", "B1", "B2", 0.9, 0.5, 0.9, 0.1),
              ("W2", "S2", "B3", "B4", 0.8, 0.2, 0.7, 0.05)][:order_count]
    specimens = []
    accurate_scores = []
    heuristic_scores = []
    for order, slide, claim, other, claim_score, other_score, h_claim, h_other in orders:
        specimens.append({
            "role": "slide", "work_order": order, "id": slide,
            "metadata": {"claim_block_id": claim},
        })
        accurate_scores.extend([
            {"work_order": order, "slide_id": slide, "block_id": claim, "score": claim_score},
            {"work_order": order, "slide_id": slide, "block_id": other, "score": other_score},
        ])
        heuristic_scores.extend([
            {"descriptor": "d1", "work_order": order, "slide_id": slide,
             "block_id": claim, "score": h_claim},
            {"descriptor": "d1", "work_order": order, "slide_id": slide,
             "block_id": other, "score": h_other},
        ])
    return {
        "provenance": {
            "manifest_path": "manifest.csv", "manifest_hash": "f" * 64,
            "code_revision": "rev1", "implementation_hashes": {"scorer": "h1"},
        },
        "specimens": specimens,
        "accurate_scores": accurate_scores,
        "heuristic_scores": heuristic_scores,
    }


def test_select_hybrid_handoff_inputs_picks_last_outer_fold_architecture():
    evidence = _hand_authored_evidence(order_count=2)
    inputs = select_hybrid_handoff_inputs(evidence)
    assert inputs["architecture"].methods == ("d1",)
    assert set(inputs["thresholds"]) == {"d1"}
    assert inputs["candidate_evidence"].evaluable_slides >= 1
    assert "estimated_runtime" in inputs["efficiency"]
    assert isinstance(inputs["known_misses"], list)
    assert inputs["provenance"] is evidence["provenance"]


def test_select_hybrid_handoff_inputs_requires_at_least_two_work_orders():
    evidence = _hand_authored_evidence(order_count=1)
    with pytest.raises(CachedEvidenceError, match="two independent work orders"):
        select_hybrid_handoff_inputs(evidence)


def test_select_hybrid_handoff_inputs_docstring_discloses_mixed_fold_aggregation():
    """F9 (#249 review): the artifact mixes a single fold's
    candidate_evidence/weak_stratum with all-fold veto/efficiency, and
    `weak_stratum: None` is ambiguous between "none exists" and "too little
    data in that one fold" -- the docstring must say so, not just the code."""
    doc = select_hybrid_handoff_inputs.__doc__
    assert "single" in doc.lower() and "fold" in doc.lower()
    assert "all outer folds" in doc or "ALL outer folds" in doc
    assert "too little data" in doc
