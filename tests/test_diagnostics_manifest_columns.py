"""Tests for pair_diagnostics manifest-column preference (TDD).

When set_id / block_tissue / slide_tissue are present on the row (from the
new v2 manifest), _diagnostic_label must use those directly.
When they are absent, the existing filename-regex fallback must still work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from pair_diagnostics import _diagnostic_label_from_row


# ---------------------------------------------------------------------------
# Manifest-column path (v2 opaque filenames)
# ---------------------------------------------------------------------------

def test_true_pair_via_manifest_set_id():
    # Both block and slide row carry the same set_id -> true_pair
    block_row = {"block_path": "images/pi_images_v2/capture_001.jpg", "set_id": "set_01",
                 "block_tissue": "lung", "slide_tissue": "lung"}
    slide_row = {"slide_path": "images/pi_images_v2/capture_002.jpg", "set_id": "set_01",
                 "slide_tissue": "lung"}
    label = _diagnostic_label_from_row(
        block_path=block_row["block_path"],
        slide_path=slide_row["slide_path"],
        block_set_id=block_row.get("set_id"),
        slide_set_id=slide_row.get("set_id"),
        block_tissue=block_row.get("block_tissue"),
        slide_tissue=slide_row.get("slide_tissue"),
    )
    assert label == "true_pair"


def test_same_tissue_near_miss_via_manifest_columns():
    # Different set_id, same tissue
    label = _diagnostic_label_from_row(
        block_path="images/pi_images_v2/capture_001.jpg",
        slide_path="images/pi_images_v2/capture_002.jpg",
        block_set_id="set_01",
        slide_set_id="set_02",
        block_tissue="lung",
        slide_tissue="lung",
    )
    assert label == "wrong_pair"


def test_cross_tissue_mismatch_via_manifest_columns():
    label = _diagnostic_label_from_row(
        block_path="images/pi_images_v2/capture_001.jpg",
        slide_path="images/pi_images_v2/capture_002.jpg",
        block_set_id="set_01",
        slide_set_id="set_02",
        block_tissue="lung",
        slide_tissue="esophagus",
    )
    assert label == "wrong_pair"


# ---------------------------------------------------------------------------
# Filename-regex fallback (v1 images/pi_images/ with structured names)
# ---------------------------------------------------------------------------

def test_true_pair_via_filename_fallback():
    # Classic images/pi_images/ filenames with set_id in name, no manifest columns supplied
    label = _diagnostic_label_from_row(
        block_path="images/pi_images/set_01_block_silhouette_lung_HE_wt_W01.jpg",
        slide_path="images/pi_images/set_01_slide_lung_HE_wt_W01.jpg",
        block_set_id=None,
        slide_set_id=None,
        block_tissue=None,
        slide_tissue=None,
    )
    assert label == "true_pair"


def test_same_tissue_near_miss_via_filename_fallback():
    label = _diagnostic_label_from_row(
        block_path="images/pi_images/set_01_block_silhouette_lung_HE_wt_W01.jpg",
        slide_path="images/pi_images/set_02_slide_lung_HE_wt_W02.jpg",
        block_set_id=None,
        slide_set_id=None,
        block_tissue=None,
        slide_tissue=None,
    )
    assert label == "wrong_pair"


def test_cross_tissue_mismatch_via_filename_fallback():
    label = _diagnostic_label_from_row(
        block_path="images/pi_images/set_01_block_silhouette_lung_HE_wt_W01.jpg",
        slide_path="images/pi_images/set_02_slide_liver_HE_wt_W02.jpg",
        block_set_id=None,
        slide_set_id=None,
        block_tissue=None,
        slide_tissue=None,
    )
    assert label == "wrong_pair"
