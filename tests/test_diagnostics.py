"""Production-parity diagnostic behavior and schema."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from pair_diagnostics import (
    DIAGNOSTIC_LABEL_COLUMN,
    collect_all_pair_records,
    _extract_metadata,
    run_all_pairs_diagnostic,
    write_diagnostic_csv,
)


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    image = np.full((200, 200, 3), 230, dtype=np.uint8)
    cv2.circle(image, (100, 100), 18, color, -1)
    noise = np.random.default_rng(0).normal(0, 6, image.shape)
    cv2.imwrite(str(path), np.clip(image + noise, 0, 255).astype(np.uint8))
    return path


def _fixture(tmp_path: Path):
    blocks = [
        _image(tmp_path / "set_01_block_lung_HE_wt_W01.jpg", (40, 90, 140)),
        _image(tmp_path / "set_02_block_liver_HE_wt_W02.jpg", (40, 90, 140)),
    ]
    slides = [
        _image(tmp_path / "set_01_slide_lung_HE_wt_W01.jpg", (200, 150, 230)),
        _image(tmp_path / "set_02_slide_liver_HE_wt_W02.jpg", (200, 150, 230)),
        _image(tmp_path / "set_03_slide_lung_HE_wt_W03.jpg", (200, 150, 230)),
    ]
    return blocks, slides


def _read(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def test_all_pairs_labels_one_best_gate_passing_wrong_as_near_miss(tmp_path):
    blocks, slides = _fixture(tmp_path)
    out = tmp_path / "diagnostic.csv"
    run_all_pairs_diagnostic(blocks, slides, out)
    _, rows = _read(out)
    assert len(rows) == len(blocks) * len(slides)
    assert {r[DIAGNOSTIC_LABEL_COLUMN] for r in rows} == {
        "true_pair", "near_miss", "wrong_pair"
    }
    for block in blocks:
        own = [r for r in rows if r["block_path"] == str(block)]
        assert sum(r[DIAGNOSTIC_LABEL_COLUMN] == "near_miss" for r in own) == 1


def test_true_and_near_miss_rows_share_margin(tmp_path):
    blocks, slides = _fixture(tmp_path)
    out = tmp_path / "diagnostic.csv"
    run_all_pairs_diagnostic(blocks, slides, out)
    _, rows = _read(out)
    for block in blocks:
        own = [r for r in rows if r["block_path"] == str(block)]
        true = next(r for r in own if r[DIAGNOSTIC_LABEL_COLUMN] == "true_pair")
        near = next(r for r in own if r[DIAGNOSTIC_LABEL_COLUMN] == "near_miss")
        assert true["true_vs_best_wrong_margin"] == near["true_vs_best_wrong_margin"]
        assert true["true_vs_best_wrong_margin"]


def test_schema_has_production_trace_and_no_legacy_scorers(tmp_path):
    blocks, slides = _fixture(tmp_path)
    out = tmp_path / "diagnostic.csv"
    run_all_pairs_diagnostic(blocks, slides, out)
    fields, _ = _read(out)
    assert {
        "score", "selected_metric", "best_angle", "best_flip",
        "align_soft_iou", "mask_iou", "block_occupied_fraction",
        "slide_occupied_fraction", "router_size_signal", "point_layout",
        "soft_correlation", "distance_signature",
    } <= set(fields)
    assert not set(fields).intersection({
        "score_d4", "score_invariant_only", "score_rotation_search",
        "best_d4_transform", "component_count_score", "router_method",
        "score_routed", "scorer_profile",
    })


def test_manifest_metadata_labels_opaque_filenames(tmp_path):
    block = _image(tmp_path / "capture_1.jpg", (40, 90, 140))
    slides = [
        _image(tmp_path / "capture_2.jpg", (200, 150, 230)),
        _image(tmp_path / "capture_3.jpg", (200, 150, 230)),
    ]
    metadata = {
        block.as_posix(): ("01", "lung"),
        slides[0].as_posix(): ("01", "lung"),
        slides[1].as_posix(): ("02", "lung"),
    }
    out = tmp_path / "diagnostic.csv"
    run_all_pairs_diagnostic([block], slides, out, path_metadata=metadata)
    _, rows = _read(out)
    assert [r[DIAGNOSTIC_LABEL_COLUMN] for r in rows] == ["true_pair", "near_miss"]


def test_v3_png_filename_metadata_is_parsed():
    meta = _extract_metadata("images/pi_images_v3/block_001_lung_N2_01_HE.png")
    assert meta | {} == {
        "set": "001", "tissue_raw": "lung", "tissue_bucket": "lung",
        "stain": "HE", "genotype": "N2", "workorder": "",
    }


def test_raw_records_write_the_same_formatted_diagnostic_csv(tmp_path):
    blocks, slides = _fixture(tmp_path)
    wrapper_out = tmp_path / "wrapper.csv"
    raw_out = tmp_path / "raw.csv"

    run_all_pairs_diagnostic(blocks, slides, wrapper_out)
    records = collect_all_pair_records(blocks, slides)
    write_diagnostic_csv(records, raw_out)

    assert raw_out.read_bytes() == wrapper_out.read_bytes()
