"""Tests for manifest loading and validation (issue #16).

Tests cover valid manifests, missing fields, duplicate IDs, missing files,
and repeated-block/multiple-slide claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from session.manifest import ClaimRow, ManifestValidationError, load_manifest


# ---------------------------------------------------------------------------
# Valid manifests
# ---------------------------------------------------------------------------

def test_load_valid_manifest(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,b1.jpg,s1.jpg\n"
    )
    rows = load_manifest(csv)
    assert len(rows) == 1
    assert rows[0].claim_id == "C001"
    assert rows[0].block_path == "b1.jpg"
    assert rows[0].slide_path == "s1.jpg"


def test_repeated_block_across_multiple_slides(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,block.jpg,slide_a.jpg\n"
        "C002,block.jpg,slide_b.jpg\n"
    )
    rows = load_manifest(csv)
    assert len(rows) == 2
    assert rows[0].block_path == rows[1].block_path == "block.jpg"
    assert rows[0].slide_path != rows[1].slide_path


def test_auto_generated_claim_ids_are_unique(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "block_path,slide_path\n"
        "b1.jpg,s1.jpg\n"
        "b2.jpg,s2.jpg\n"
    )
    rows = load_manifest(csv)
    ids = [r.claim_id for r in rows]
    assert len(ids) == len(set(ids)), "auto-generated claim IDs must be unique"


def test_empty_manifest_returns_empty_list(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("claim_id,block_path,slide_path\n")
    assert load_manifest(csv) == []


# ---------------------------------------------------------------------------
# Validation errors (fatal)
# ---------------------------------------------------------------------------

def test_missing_block_path_column_raises(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("claim_id,slide_path\nb1.jpg,s1.jpg\n")
    with pytest.raises(ManifestValidationError, match="block_path"):
        load_manifest(csv)


def test_missing_slide_path_column_raises(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("claim_id,block_path\nC001,b1.jpg\n")
    with pytest.raises(ManifestValidationError, match="slide_path"):
        load_manifest(csv)


def test_duplicate_claim_ids_raise(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,b1.jpg,s1.jpg\n"
        "C001,b2.jpg,s2.jpg\n"
    )
    with pytest.raises(ManifestValidationError, match="C001"):
        load_manifest(csv)


# ---------------------------------------------------------------------------
# Missing files (non-fatal — reported in ClaimRow, not a crash)
# ---------------------------------------------------------------------------

def test_missing_block_file_reported_not_crashed(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,nonexistent_block.jpg,nonexistent_slide.jpg\n"
    )
    rows = load_manifest(csv, check_files=True)
    assert len(rows) == 1
    assert rows[0].missing_files, "missing files should be recorded on the ClaimRow"


def test_existing_files_have_no_missing_files(tmp_path):
    block = tmp_path / "block.jpg"
    slide = tmp_path / "slide.jpg"
    block.write_bytes(b"fake")
    slide.write_bytes(b"fake")
    csv = tmp_path / "m.csv"
    csv.write_text(
        f"claim_id,block_path,slide_path\n"
        f"C001,{block},{slide}\n"
    )
    rows = load_manifest(csv, check_files=True)
    assert rows[0].missing_files == ()


def test_missing_files_not_checked_by_default(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text(
        "claim_id,block_path,slide_path\n"
        "C001,ghost_block.jpg,ghost_slide.jpg\n"
    )
    rows = load_manifest(csv)
    assert rows[0].missing_files == ()


def test_partial_missing_files_records_which_ones(tmp_path):
    block = tmp_path / "real_block.jpg"
    block.write_bytes(b"fake")
    csv = tmp_path / "m.csv"
    csv.write_text(
        f"claim_id,block_path,slide_path\n"
        f"C001,{block},ghost_slide.jpg\n"
    )
    rows = load_manifest(csv, check_files=True)
    assert len(rows[0].missing_files) == 1
    assert "ghost_slide.jpg" in rows[0].missing_files[0]
