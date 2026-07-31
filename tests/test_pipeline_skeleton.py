"""Tests for the manifest-first pipeline skeleton (issue #15).

Tests the external behavior: manifest CSV in -> decision CSV out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from session.pipeline import run_claim_pipeline, VERDICT_PASS, VERDICT_REVIEW, DECISION_COLUMNS

VALID_VERDICTS = {VERDICT_PASS, VERDICT_REVIEW}


@pytest.fixture
def manifest_csv(tmp_path):
    """Three claims: block_01 repeated across two slides, then block_02."""
    path = tmp_path / "manifest.csv"
    path.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,images/block_01.jpg,images/slide_01.jpg\n"
        "C002,images/block_01.jpg,images/slide_02.jpg\n"
        "C003,images/block_02.jpg,images/slide_03.jpg\n"
    )
    return path


@pytest.fixture
def output_csv(tmp_path):
    return tmp_path / "decisions.csv"


def test_manifest_drives_end_to_end_run(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    assert output_csv.exists()
    assert len(decisions) == 3


def test_one_row_is_one_claim(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    ids = [d.claim_id for d in decisions]
    assert ids == ["C001", "C002", "C003"]


def test_repeated_block_across_multiple_rows(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    block_paths = [d.block_path for d in decisions]
    assert block_paths.count("images/block_01.jpg") == 2


def test_decision_csv_has_required_columns(manifest_csv, output_csv):
    run_claim_pipeline(manifest_csv, output_csv)
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        for col in DECISION_COLUMNS:
            assert col in (reader.fieldnames or []), f"Missing column: {col}"


def test_decision_csv_exposes_production_score_traceability(manifest_csv, output_csv):
    run_claim_pipeline(manifest_csv, output_csv)
    with open(output_csv, newline="") as f:
        fields = set(csv.DictReader(f).fieldnames or [])
    assert {
        "score", "selected_metric", "router_size_signal",
        "block_occupied_fraction", "slide_occupied_fraction",
        "best_angle", "best_flip", "align_soft_iou", "mask_iou",
    } <= fields


def test_verdicts_are_pass_or_review_only(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    for d in decisions:
        assert d.verdict in VALID_VERDICTS


def test_decision_csv_rows_match_manifest_rows(manifest_csv, output_csv):
    run_claim_pipeline(manifest_csv, output_csv)
    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_decision_includes_block_and_slide_references(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    for d in decisions:
        assert d.block_path
        assert d.slide_path


def test_decision_includes_stage_and_reason(manifest_csv, output_csv):
    decisions = run_claim_pipeline(manifest_csv, output_csv)
    for d in decisions:
        assert d.stage
        assert d.reason


def test_gate_failed_production_claim_writes_review_with_blank_score(tmp_path):
    block_img = tmp_path / "block.jpg"
    slide_img = tmp_path / "slide.jpg"
    block = np.full((1000, 1000, 3), 230, dtype=np.uint8)
    slide = np.full((1000, 1000, 3), 230, dtype=np.uint8)
    # A long, thin tissue strip on the block produces a degenerate aspect-ratio
    # mask at native resolution (4x downsample → strip is 100 wide × 2 tall,
    # aspect=50), which fails the block ROI gate. The slide uses the same
    # color-segmentation path; the test exercises the gate-before-score
    # boundary, not a specific gate stage.
    cv2.rectangle(block, (300, 496), (700, 504), (40, 90, 140), -1)
    cv2.rectangle(slide, (300, 496), (700, 504), (200, 150, 230), -1)
    cv2.imwrite(str(block_img), block)
    cv2.imwrite(str(slide_img), slide)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "claim_id,block_path,slide_path\n"
        f"C001,{block_img},{slide_img}\n"
    )
    output = tmp_path / "decisions.csv"

    decisions = run_claim_pipeline(manifest, output)

    assert len(decisions) == 1
    assert decisions[0].verdict == VERDICT_REVIEW
    assert decisions[0].stage  # some gate fired
    assert decisions[0].score is None
    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["score"] == ""
