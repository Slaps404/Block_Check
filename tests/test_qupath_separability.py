"""Behavior tests for the RTrees-versus-classical separability comparison."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import qupath_separability  # noqa: E402
from qupath_separability import build_comparison_report  # noqa: E402
from verify import segmentation  # noqa: E402


def _record(
    block: str,
    slide: str,
    block_set: str,
    slide_set: str,
    tissue: str,
    score: float,
):
    return SimpleNamespace(
        diagnostic_label="true_pair" if block_set == slide_set else "wrong_pair",
        block_path=block,
        slide_path=slide,
        block_set=block_set,
        slide_set=slide_set,
        score=score,
        gate_passed=True,
        gate_stage="ok",
        gate_reason="ok",
        columns={"slide_tissue_raw": tissue},
    )


def test_comparison_reports_same_tissue_margins_misranks_and_hard_cases():
    classical = (
        _record("block-a", "slide-a", "01", "01", "lung", 0.80),
        _record("block-a", "slide-b", "01", "02", "lung", 0.60),
        _record("block-b", "slide-b", "02", "02", "lung", 0.50),
        _record("block-b", "slide-a", "02", "01", "lung", 0.55),
    )
    rtrees = (
        _record("block-a", "slide-a", "01", "01", "lung", 0.90),
        _record("block-a", "slide-b", "01", "02", "lung", 0.50),
        _record("block-b", "slide-b", "02", "02", "lung", 0.40),
        _record("block-b", "slide-a", "02", "01", "lung", 0.60),
    )

    claims = (
        {"claim_id": "a", "block_path": "block-a", "slide_path": "slide-a",
         "set_id": "01", "slide_tissue": "lung"},
        {"claim_id": "b", "block_path": "block-b", "slide_path": "slide-b",
         "set_id": "02", "slide_tissue": "lung"},
    )
    negatives = {"a": (("block-a", "slide-b"),), "b": (("block-b", "slide-a"),)}
    report, rows = build_comparison_report(
        classical, rtrees, claims, negatives,
        {"01": ("depth-of-section",), "02": ("faint-tail",)},
        {"manifest_sha256": "manifest", "rtrees_model_sha256": "model"},
    )

    assert [(row["backend"], row["misrank"]) for row in rows] == [
        ("classical", False), ("classical", True), ("rtrees", False), ("rtrees", True),
    ]
    assert [row["margin"] for row in rows] == pytest.approx([0.2, -0.05, 0.4, -0.2])
    assert report["hard_case_evidence"]["depth-of-section"]["classical"]["claims"] == 1
    assert report["hard_case_evidence"]["faint-tail"]["rtrees"]["claims"] == 1
    assert report["per_tissue"]["lung"]["rtrees"]["misranks"] == 1
    assert report["recommendation"] == "reject"


def test_comparison_requires_identical_claims_between_backends():
    classical = (_record("block-a", "slide-a", "01", "01", "lung", 0.8),)
    rtrees = (_record("block-a", "slide-b", "01", "01", "lung", 0.8),)

    try:
        build_comparison_report(
            classical, rtrees,
            [{"claim_id": "a", "block_path": "block-a", "slide_path": "slide-a",
              "set_id": "01", "slide_tissue": "lung"}],
            {"a": ()}, {}, {},
        )
    except ValueError as error:
        assert "missing claimed-pair score" in str(error)
    else:
        raise AssertionError("comparison accepted different backend claims")


def test_runner_freezes_provenance_and_restores_the_default_backend(tmp_path, monkeypatch):
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    for name in ("block-a.png", "block-b.png", "slide-a.png", "slide-b.png"):
        assert cv2.imwrite(str(tmp_path / name), image)
    manifest = tmp_path / "benchmark.csv"
    manifest.write_text(
        "claim_id,block_path,slide_path,set_id,block_tissue,slide_tissue\n"
        f"a,{tmp_path / 'block-a.png'},{tmp_path / 'slide-a.png'},01,lung,lung\n"
        f"b,{tmp_path / 'block-b.png'},{tmp_path / 'slide-b.png'},02,lung,lung\n",
        encoding="utf-8",
    )
    hard_cases = tmp_path / "hard-cases.csv"
    hard_cases.write_text(
        "set_id,hard_case\n01,depth-of-section\n02,faint-tail\n", encoding="utf-8"
    )
    negatives = tmp_path / "hard-negatives.csv"
    negatives.write_text(
        "claim_id,block_path,slide_path\n"
        f"a,{tmp_path / 'block-a.png'},{tmp_path / 'slide-b.png'}\n"
        f"b,{tmp_path / 'block-b.png'},{tmp_path / 'slide-a.png'}\n",
        encoding="utf-8",
    )
    model, sidecar = tmp_path / "model.yml.gz", tmp_path / "recipe.json"
    model.write_bytes(b"model")
    sidecar.write_text("{}", encoding="utf-8")
    calls = []

    def fake_collect(pairs, *, path_metadata):
        calls.append(segmentation.BLOCK_SEGMENTER)
        scores = (0.8, 0.6) if segmentation.BLOCK_SEGMENTER == "classical" else (0.9, 0.5)
        return (
            _record(str(pairs[0][0]), str(pairs[0][1]), "01", "01", "lung", scores[0]),
            _record(str(pairs[2][0]), str(pairs[2][1]), "01", "02", "lung", scores[1]),
            _record(str(pairs[1][0]), str(pairs[1][1]), "02", "02", "lung", scores[0]),
            _record(str(pairs[3][0]), str(pairs[3][1]), "02", "01", "lung", scores[1]),
        )

    monkeypatch.setattr(qupath_separability, "collect_selected_pair_records", fake_collect)
    report = qupath_separability.run_separability_comparison(
        manifest, hard_cases, negatives, model, sidecar, tmp_path / "out"
    )

    assert calls == ["classical", "rtrees"]
    assert segmentation.BLOCK_SEGMENTER == "classical"
    assert report["provenance"]["manifest_sha256"]
    assert report["provenance"]["rtrees_model_sha256"]
    assert (tmp_path / "out" / "pair_comparison.csv").is_file()
