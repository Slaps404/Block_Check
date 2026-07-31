# flake8: noqa
import json
from pathlib import Path
import subprocess
import sys

import pytest

from run_retrieval_diagnostics import main


def test_direct_script_help_bootstraps_repo_import_paths():
    script = (
        Path(__file__).parents[1]
        / "tools" / "scoring_diagnostics" / "run_retrieval_diagnostics.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "build" in result.stdout and "calibrate" in result.stdout


def test_cli_has_separate_build_and_calibrate_operations(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"; manifest.write_text("x")
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr("run_retrieval_diagnostics.load_retrieval_manifest", lambda path: "manifest")
    monkeypatch.setattr("run_retrieval_diagnostics.build_evidence", lambda manifest, output: {"operation": "built", "output": str(output)})
    assert main(["build", "--manifest", str(manifest), "--evidence", str(evidence)]) == 0
    assert json.loads(capsys.readouterr().out)["operation"] == "built"
    monkeypatch.setattr("run_retrieval_diagnostics.calibrate_cached_evidence", lambda path: {"operation": "calibrated", "score_count": 3})
    assert main(["calibrate", "--evidence", str(evidence)]) == 0
    assert json.loads(capsys.readouterr().out)["score_count"] == 3


def test_calibrate_without_handoff_flag_leaves_existing_result_unchanged(
    tmp_path, monkeypatch, capsys,
):
    """#249: adding --handoff must not change the default calibrate result."""
    evidence = tmp_path / "evidence.json"; evidence.write_text("{}")
    called = []
    monkeypatch.setattr(
        "run_retrieval_diagnostics.calibrate_cached_evidence",
        lambda path, **kwargs: called.append(kwargs) or {"score_count": 3},
    )
    monkeypatch.setattr(
        "run_retrieval_diagnostics.select_hybrid_handoff_inputs",
        lambda evidence: pytest.fail("select_hybrid_handoff_inputs called without --handoff"),
    )
    assert main(["calibrate", "--evidence", str(evidence)]) == 0
    assert json.loads(capsys.readouterr().out) == {"score_count": 3}


def test_calibrate_with_handoff_flag_writes_versioned_handoff_file(
    tmp_path, monkeypatch, capsys,
):
    from candidate_retrieval_analysis import Architecture, ArchitectureKind, BandEvaluation, VetoCalibration

    evidence = tmp_path / "evidence.json"; evidence.write_text(json.dumps({"provenance": {}}))
    handoff_path = tmp_path / "handoff.json"
    monkeypatch.setattr(
        "run_retrieval_diagnostics.calibrate_cached_evidence",
        lambda path, **kwargs: {"score_count": 3},
    )
    monkeypatch.setattr(
        "run_retrieval_diagnostics.select_hybrid_handoff_inputs",
        lambda evidence: {
            "architecture": Architecture(ArchitectureKind.INDIVIDUAL, "d1", ("d1",)),
            "thresholds": {"d1": 0.2},
            "veto": VetoCalibration(False, None, (), (), "disabled"),
            "candidate_evidence": BandEvaluation(4, 3, (1, 2), ("W::S1",)),
            "efficiency": {"observed_runtime": 0.1, "estimated_runtime": 0.1,
                            "full_comparison_reduction": 0.5},
            "known_misses": ["W::S1"],
            "weak_stratum": None,
            "provenance": {"manifest_path": "m.csv", "manifest_hash": "a" * 64,
                           "code_revision": "rev", "implementation_hashes": {}},
        },
    )
    result_code = main([
        "calibrate", "--evidence", str(evidence), "--handoff", str(handoff_path),
    ])
    assert result_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["score_count"] == 3
    assert printed["handoff_path"] == str(handoff_path)
    written = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert written["architecture"]["methods"] == ["d1"]
    assert written["status"] == "proof_of_concept_not_production_approved"
