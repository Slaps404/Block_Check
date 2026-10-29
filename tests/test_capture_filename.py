"""Shared filename parsing for legacy, compact, and triple-capture names."""
from __future__ import annotations

from capture_filename import capture_tag, parse_dataset_filename


def test_compact_filename_preserves_optional_capture_index():
    parsed = parse_dataset_filename(
        "block_001_capture_02_lung_N2_01_HE.png"
    )

    assert parsed is not None
    assert parsed.role == "block"
    assert parsed.set_number == "001"
    assert parsed.capture_index == "02"
    assert parsed.tissue == "lung"
    assert parsed.stain == "HE"


def test_legacy_filename_without_capture_index_keeps_existing_metadata():
    parsed = parse_dataset_filename(
        "set_01_block_silhouette_lung_HE_wt_W01.jpg"
    )

    assert parsed is not None
    assert parsed.capture_index is None
    assert parsed.role == "block_silhouette"
    assert parsed.workorder == "W01"


def test_capture_tag_is_one_based_and_zero_padded():
    assert capture_tag(1) == "capture_01"
    assert capture_tag(3) == "capture_03"
