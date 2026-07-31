"""Tests for the active compact PNG dataset manifest contract."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "manifest"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from build_png_manifest import (  # noqa: E402
    DatasetContractError,
    LABEL_SOURCE_FILENAME_SET_ID,
    build_png_manifest_rows,
    write_png_manifest,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_build_png_manifest_contract_rows(tmp_path):
    dataset = tmp_path / "images" / "pi_images_v3"
    _touch(dataset / "block_001_lung_N2_01_HE.png")
    _touch(dataset / "slide_001_lung_N2_01_HE.png")

    rows = build_png_manifest_rows(dataset, repo_root=tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["claim_id"] == "set_001_01_HE"
    assert row["block_path"] == (
        "images/pi_images_v3/block_001_lung_N2_01_HE.png"
    )
    assert row["slide_path"] == (
        "images/pi_images_v3/slide_001_lung_N2_01_HE.png"
    )
    assert row["set_id"] == "set_001"
    assert row["block_tissue"] == "lung"
    assert row["slide_tissue"] == "lung"
    assert row["slide_no"] == "01"

    assert row["label_source"] == LABEL_SOURCE_FILENAME_SET_ID
    assert row["block_identity_source"] == "not_decoded"
    assert row["slide_identity_source"] == "not_decoded"
    assert row["block_barcode_id"] == ""
    assert row["slide_barcode_block_id"] == ""
    assert row["provenance_dataset"] == "images/pi_images_v3"
    assert row["provenance_builder"] == "tools/manifest/build_png_manifest.py"


def test_build_png_manifest_separates_filename_labels_from_barcode_identity(
    tmp_path,
):
    dataset = tmp_path / "images" / "pi_images_v3"
    _touch(dataset / "block_002_lungs_NAIVE_01_HE.png")
    _touch(dataset / "slide_002_lungs_NAIVE_01_HE.png")

    row = build_png_manifest_rows(dataset, repo_root=tmp_path)[0]

    assert row["label_source"] == "temporary_filename_set_id"
    assert row["block_identity_source"] != row["label_source"]
    assert row["slide_identity_source"] != row["label_source"]


def test_write_png_manifest_outputs_contract_columns(tmp_path):
    dataset = tmp_path / "images" / "pi_images_v3"
    _touch(dataset / "block_003_lungs_WT1_01_HE.png")
    _touch(dataset / "slide_003_lungs_WT1_01_HE.png")
    out = tmp_path / "manifest.csv"

    count = write_png_manifest(dataset, out, repo_root=tmp_path)

    assert count == 1
    rows = _read_csv(out)
    assert rows[0]["claim_id"] == "set_003_01_HE"
    assert {
        "block_path",
        "slide_path",
        "set_id",
        "block_tissue",
        "slide_tissue",
        "label_source",
        "provenance_dataset",
        "provenance_builder",
    } <= set(rows[0])


def test_build_png_manifest_missing_dataset_fails_clearly(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(
        DatasetContractError,
        match="PNG dataset path not found",
    ):
        build_png_manifest_rows(missing, repo_root=tmp_path)


def test_build_png_manifest_empty_dataset_fails_clearly(tmp_path):
    dataset = tmp_path / "images" / "pi_images_v3"
    dataset.mkdir(parents=True)

    with pytest.raises(
        DatasetContractError,
        match="no usable block/slide PNG rows",
    ):
        build_png_manifest_rows(dataset, repo_root=tmp_path)


def test_build_png_manifest_tissue_mismatch_fails_closed(tmp_path):
    dataset = tmp_path / "images" / "pi_images_v3"
    _touch(dataset / "block_004_lung_N1_01_HE.png")
    _touch(dataset / "slide_004_lungs_N1_01_HE.png")

    with pytest.raises(DatasetContractError, match="tissue mismatch"):
        build_png_manifest_rows(dataset, repo_root=tmp_path)
