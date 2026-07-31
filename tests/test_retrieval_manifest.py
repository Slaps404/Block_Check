# flake8: noqa
import csv

import pytest

from retrieval_manifest import ManifestValidationError, load_retrieval_manifest


def _write(tmp_path, rows):
    for index, row in enumerate(rows, start=1):
        omit_tissue = row.pop("_omit_tissue", False)
        row.update({
            "row_id": row.get("row_id", f"row-{index}"),
            "claim_id": row.get("claim_id", f"claim-{index}"),
            "set_id": row.get("set_id", "set-1"),
            "label_source": row.get("label_source", "manual"),
            "inclusion_status": row.get(
                "inclusion_status", "excluded" if row.get("exclusion_reason") else "included"
            ),
            "capture_profile": row.get("capture_profile", "pi-v3"),
            "capture_status": row.get("capture_status", "accepted"),
            "tissue_source": row.get("tissue_source", "manual"),
            "tissue_confidence": row.get("tissue_confidence", "1.0"),
        })
        if omit_tissue:
            row.pop("tissue_source")
            row.pop("tissue_confidence")
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manifest_deduplicates_repeated_block_and_preserves_metadata(tmp_path):
    block = tmp_path / "block.png"; block.touch()
    first = tmp_path / "slide-1.png"; first.touch()
    second = tmp_path / "slide-2.png"; second.touch()
    path = _write(tmp_path, [
        {"work_order": "WO-1", "block_id": "B-1", "block_path": block,
         "slide_id": "S-1", "slide_path": first, "tissue": "lung", "source": "manual"},
        {"work_order": "WO-1", "block_id": "B-1", "block_path": block,
         "slide_id": "S-2", "slide_path": second, "tissue": "lung", "source": "manual"},
    ])
    manifest = load_retrieval_manifest(path)
    assert len(manifest.blocks) == 1
    assert len(manifest.slides) == 2
    assert manifest.slides[0].metadata["tissue"] == "lung"
    assert manifest.work_orders["WO-1"].block_ids == ("B-1",)


def test_manifest_rejects_conflicting_repeated_block_and_duplicate_slide(tmp_path):
    a = tmp_path / "a.png"; a.touch(); b = tmp_path / "b.png"; b.touch(); s = tmp_path / "s.png"; s.touch()
    conflict = _write(tmp_path, [
        {"work_order": "1", "block_id": "B", "block_path": a, "slide_id": "S1", "slide_path": s},
        {"work_order": "1", "block_id": "B", "block_path": b, "slide_id": "S2", "slide_path": s},
    ])
    with pytest.raises(ManifestValidationError, match="conflicting block"):
        load_retrieval_manifest(conflict)
    duplicate = _write(tmp_path, [
        {"work_order": "1", "block_id": "B1", "block_path": a, "slide_id": "S", "slide_path": s},
        {"work_order": "1", "block_id": "B2", "block_path": b, "slide_id": "S", "slide_path": s},
    ])
    with pytest.raises(ManifestValidationError, match="duplicate slide"):
        load_retrieval_manifest(duplicate)


def test_manifest_rejects_global_slide_path_duplicate_and_block_metadata_conflict(tmp_path):
    block = tmp_path / "block.png"
    block.touch()
    slide = tmp_path / "slide.png"
    slide.touch()
    duplicate_path = _write(tmp_path, [
        {"work_order": "1", "block_id": "B1", "block_path": block,
         "slide_id": "S1", "slide_path": slide},
        {"work_order": "2", "block_id": "B2", "block_path": block,
         "slide_id": "S2", "slide_path": slide},
    ])
    with pytest.raises(ManifestValidationError, match="duplicate slide"):
        load_retrieval_manifest(duplicate_path)
    changed_metadata = _write(tmp_path, [
        {"work_order": "1", "block_id": "B", "block_path": block,
         "slide_id": "S1", "slide_path": slide, "block_tissue": "lung"},
        {"work_order": "1", "block_id": "B", "block_path": block,
         "slide_id": "S2", "slide_path": tmp_path / "other.png",
         "block_tissue": "liver"},
    ])
    (tmp_path / "other.png").touch()
    with pytest.raises(ManifestValidationError, match="conflicting block"):
        load_retrieval_manifest(changed_metadata)


def test_manifest_requires_work_order_and_existing_paths_and_keeps_exclusion(tmp_path):
    missing = _write(tmp_path, [{"work_order": "", "block_id": "B", "block_path": "missing.png", "slide_id": "S", "slide_path": "missing-s.png", "exclusion_reason": "bad capture"}])
    with pytest.raises(ManifestValidationError, match="work_order"):
        load_retrieval_manifest(missing)
    block = tmp_path / "b.png"; block.touch(); slide = tmp_path / "s.png"; slide.touch()
    path = _write(tmp_path, [{"work_order": "1", "block_id": "B", "block_path": block, "slide_id": "S", "slide_path": slide, "exclusion_reason": "bad capture"}])
    manifest = load_retrieval_manifest(path)
    assert manifest.slides == ()
    assert manifest.exclusions[0].reason == "bad capture"


def test_tissue_provenance_is_optional_and_stable_block_identity_is_global(tmp_path):
    block = tmp_path / "block.png"
    block.touch()
    slide_1 = tmp_path / "slide-1.png"
    slide_1.touch()
    slide_2 = tmp_path / "slide-2.png"
    slide_2.touch()
    path = _write(tmp_path, [
        {"work_order": "WO-1", "block_id": "B", "block_path": block,
         "slide_id": "S1", "slide_path": slide_1, "_omit_tissue": True},
        {"work_order": "WO-2", "block_id": "B", "block_path": block,
         "slide_id": "S2", "slide_path": slide_2, "_omit_tissue": True},
    ])
    manifest = load_retrieval_manifest(path)
    assert len(manifest.blocks) == 1
    assert manifest.work_orders["WO-1"].block_ids == ("B",)
    assert manifest.work_orders["WO-2"].block_ids == ("B",)
    assert "tissue_source" not in manifest.slides[0].metadata
