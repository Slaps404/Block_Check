"""Tests for run_diagnostics.py manifest mode (Task 3).

load_manifest_paths sources the block/slide image lists and the ground-truth
(set_id, tissue) map from pair_manifest.csv, since the glob cannot match the
opaque timestamp filenames of the v2 dataset.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "scoring_diagnostics"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


_MANIFEST_HEADER = [
    "claim_id",
    "block_path",
    "slide_path",
    "set_id",
    "block_tissue",
    "slide_tissue",
    "slide_no",
    "slide_n_clusters",
]


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_manifest_paths(tmp_path):
    """Two sets; set_01's block recurs across two slide rows, so the block
    list must de-duplicate (2 blocks) while the slide list keeps all (3 slides).
    The metadata map must key on the resolved posix path and carry
    (set_id, block_tissue) for blocks and (set_id, slide_tissue) for slides.
    """
    from run_diagnostics import load_manifest_paths

    rows = [
        {
            "claim_id": "c1",
            "block_path": "images/pi_images_v2/capture_b01.jpg",
            "slide_path": "images/pi_images_v2/capture_s01.jpg",
            "set_id": "set_01",
            "block_tissue": "lung",
            "slide_tissue": "lung",
            "slide_no": "1",
            "slide_n_clusters": "1",
        },
        {
            "claim_id": "c2",
            "block_path": "images/pi_images_v2/capture_b01.jpg",
            "slide_path": "images/pi_images_v2/capture_s02.jpg",
            "set_id": "set_01",
            "block_tissue": "lung",
            "slide_tissue": "lung",
            "slide_no": "2",
            "slide_n_clusters": "1",
        },
        {
            "claim_id": "c3",
            "block_path": "images/pi_images_v2/capture_b02.jpg",
            "slide_path": "images/pi_images_v2/capture_s03.jpg",
            "set_id": "set_02",
            "block_tissue": "liver",
            "slide_tissue": "liver",
            "slide_no": "1",
            "slide_n_clusters": "1",
        },
    ]
    manifest = _write_manifest(tmp_path / "pair_manifest.csv", rows)

    block_paths, slide_paths, path_metadata = load_manifest_paths(
        manifest, root=tmp_path
    )

    # set_01's block appears twice in the manifest -> de-duplicated to one.
    assert len(block_paths) == 2, f"expected 2 unique blocks, got {block_paths}"
    assert len(slide_paths) == 3, f"expected 3 slides, got {slide_paths}"

    # Spot-check the map: keys are resolved (root / relpath).as_posix(),
    # blocks -> (set_id, block_tissue), slides -> (set_id, slide_tissue).
    block_key = (tmp_path / "images/pi_images_v2/capture_b01.jpg").as_posix()
    slide_key = (tmp_path / "images/pi_images_v2/capture_s03.jpg").as_posix()
    assert path_metadata[block_key] == ("set_01", "lung")
    assert path_metadata[slide_key] == ("set_02", "liver")


def test_discover_dataset_paths_supports_v3_png_names(tmp_path):
    from run_diagnostics import discover_dataset_paths

    block = tmp_path / "block_001_lung_N2_01_HE.png"
    slide = tmp_path / "slide_001_lung_N2_01_HE.png"
    block.write_bytes(b"placeholder")
    slide.write_bytes(b"placeholder")

    block_paths, slide_paths = discover_dataset_paths(tmp_path)

    assert block_paths == [block]
    assert slide_paths == [slide]
