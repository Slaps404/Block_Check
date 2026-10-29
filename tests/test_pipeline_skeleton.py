"""Tests for the manifest-first pipeline skeleton.

Tests the external behavior: manifest CSV in -> decision CSV out.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

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


def test_manifest_drives_end_to_end_run(manifest_csv, tmp_path):
    """One manifest row is one claim; decisions and CSV mirror the manifest."""
    output_csv = tmp_path / "decisions.csv"
    decisions = run_claim_pipeline(manifest_csv, output_csv)

    assert [d.claim_id for d in decisions] == ["C001", "C002", "C003"]
    assert [d.block_path for d in decisions].count("images/block_01.jpg") == 2
    for d in decisions:
        assert d.verdict in VALID_VERDICTS
        assert d.block_path and d.slide_path
        assert d.stage and d.reason

    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    assert len(rows) == 3
    assert set(DECISION_COLUMNS) <= fields
    assert {
        "score", "selected_metric", "router_size_signal",
        "block_occupied_fraction", "slide_occupied_fraction",
        "best_angle", "best_flip", "align_soft_iou", "mask_iou",
    } <= fields

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
