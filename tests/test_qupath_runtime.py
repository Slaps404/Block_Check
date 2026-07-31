"""Runtime seam tests for the opt-in block RTrees backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from constants import BLOCK_QUPATH_MODEL_PATH
from session.preparation import PreparedSpecimen
from session.processing_store import ProcessingStore, preprocess_block
from verify import segmentation
from verify.qupath_features import FeatureRecipe, FeatureSpec
from verify.qupath_rtrees import train_rtrees_segmenter


def test_live_qupath_backend_uses_requested_block_tissue_classifier():
    assert BLOCK_QUPATH_MODEL_PATH.name == "block_tissue_v001_g-wg-gm_s1-2-8_t25.json"


def _artifact(tmp_path):
    recipe = FeatureRecipe(
        channels=("Red",),
        features=(FeatureSpec("GAUSSIAN", 1.0),),
        downsample=1.0,
        class_map={0: "Background", 1: "Tissue"},
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, 6:, 2] = 230
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[:, 6:] = 1
    model_path = tmp_path / "block.yml.gz"
    sidecar_path = tmp_path / "block.recipe.json"
    train_rtrees_segmenter(
        recipe,
        [(image, labels)],
        positive_class_id=1,
        model_path=model_path,
        sidecar_path=sidecar_path,
        rng_seed=4,
    )
    return image, model_path, sidecar_path


def test_block_rtrees_backend_uses_saved_artifact_and_reports_active_backend(
    tmp_path, monkeypatch
):
    image, model_path, sidecar_path = _artifact(tmp_path)
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "rtrees")
    monkeypatch.setattr(segmentation, "BLOCK_RTREES_MODEL_PATH", model_path)
    monkeypatch.setattr(segmentation, "BLOCK_RTREES_SIDECAR_PATH", sidecar_path)
    segmentation._load_block_rtrees.cache_clear()

    mask = segmentation.segment_tissue(image, "block")

    assert segmentation.active_segmentation_backend("block") == "rtrees"
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask[:, :6]) == 0
    assert np.count_nonzero(mask[:, 6:]) == 12 * 6


def test_block_qupath_backend_loads_the_saved_classifier(monkeypatch):
    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "QuPath"
        / "block_tissue_v001_g-wg-gm_s1-2-8_t25.json"
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:, 2] = 220
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "qupath")
    monkeypatch.setattr(segmentation, "MIN_BLOCK_COMPONENT_AREA", 1)
    monkeypatch.setattr(segmentation, "BLOCK_QUPATH_MODEL_PATH", model_path)
    segmentation._load_block_qupath.cache_clear()

    mask = segmentation.segment_tissue(image, "block")

    assert segmentation.active_segmentation_backend("block") == "qupath"
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})


def test_block_qupath_backend_applies_standard_block_cleanup(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[40:55, 40:55] = 255
    raw[47, 47] = 0  # The normal fill stage must restore this tissue hole.
    raw[1, 1] = 255  # The normal open/component cleanup must remove this speck.
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "qupath")
    monkeypatch.setattr(
        segmentation,
        "_load_block_qupath",
        lambda _path: SimpleNamespace(predict=lambda _image: raw.copy()),
    )
    monkeypatch.setattr(segmentation, "MIN_BLOCK_COMPONENT_AREA", 1)

    mask = segmentation.segment_tissue(image, "block")

    assert mask[1, 1] == 0
    assert mask[47, 47] == 255


def test_block_qupath_cleanup_keeps_elongated_classifier_tissue(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[40:44, 30:70] = 255
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "qupath")
    monkeypatch.setattr(
        segmentation,
        "_load_block_qupath",
        lambda _path: SimpleNamespace(predict=lambda _image: raw.copy()),
    )
    monkeypatch.setattr(segmentation, "MIN_BLOCK_COMPONENT_AREA", 1)

    mask = segmentation.segment_tissue(image, "block")

    assert np.count_nonzero(mask) > 0


def test_native_qupath_import_rejects_non_identity_feature_normalization(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "QuPath"
        / "surf_e001_rt_g-log-wd-gm_25t_4px.json"
    )
    document = json.loads(source.read_text(encoding="utf-8"))
    document["op"]["op"]["ops"][0]["ops"][1]["preprocessor"]["normalizer"][
        "offsets"
    ][0] = 1.0
    altered = tmp_path / "normalized.json"
    altered.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization.*unsupported"):
        segmentation.load_qupath_rtrees_segmenter(altered)


@pytest.mark.parametrize("missing", ("model", "sidecar"))
def test_block_rtrees_missing_artifact_has_actionable_error(tmp_path, monkeypatch, missing):
    image, model_path, sidecar_path = _artifact(tmp_path)
    if missing == "model":
        model_path.unlink()
    else:
        sidecar_path.unlink()
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "rtrees")
    monkeypatch.setattr(segmentation, "BLOCK_RTREES_MODEL_PATH", model_path)
    monkeypatch.setattr(segmentation, "BLOCK_RTREES_SIDECAR_PATH", sidecar_path)
    segmentation._load_block_rtrees.cache_clear()

    with pytest.raises(ValueError, match=f"RTrees {missing} artifact is missing"):
        segmentation.segment_tissue(image, "block")


def test_block_rtrees_rejects_sidecar_with_unknown_positive_class(tmp_path):
    _image, model_path, sidecar_path = _artifact(tmp_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["positive_class_id"] = 9
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="positive_class_id.*not in the recipe class map"):
        segmentation.load_rtrees_segmenter(model_path, sidecar_path)


def test_block_backend_survives_durable_preprocessing_metadata_reload(tmp_path, monkeypatch):
    capture = tmp_path / "block.png"
    assert cv2.imwrite(str(capture), np.full((10, 10, 3), 120, dtype=np.uint8))
    mask = np.full((10, 10), 255, dtype=np.uint8)
    prepared = PreparedSpecimen(
        role="block", mask=mask, roi_ok=True, roi_reason="ok", segmentation_backend="rtrees"
    )
    monkeypatch.setattr(
        "session.processing_store.prepare_specimen",
        lambda *_args, slide_close_ksize=None, stage_timings=None: prepared,
    )

    persisted_mask, metadata = preprocess_block(capture)
    mask_path = tmp_path / "persisted_mask.png"
    assert cv2.imwrite(str(mask_path), persisted_mask)
    restored = ProcessingStore._load_block_result({
        "mask_path": str(mask_path), "preprocessing_metadata": json.dumps(metadata),
    })

    assert isinstance(restored, PreparedSpecimen)
    assert restored.segmentation_backend == "rtrees"
