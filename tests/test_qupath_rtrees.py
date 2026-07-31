"""End-to-end behavior tests for the retrained QuPath RTrees artifact."""

from __future__ import annotations

import json

import numpy as np

from verify.qupath_features import parse_feature_recipe
from verify.qupath_rtrees import (
    load_rtrees_segmenter,
    train_rtrees_segmenter,
)


def _recipe() -> dict:
    return {
        "pixel_classifier_type": "OpenCVPixelClassifier",
        "metadata": {
            "inputResolution": {
                "pixelWidth": {"value": 2.0, "unit": "px"},
                "pixelHeight": {"value": 2.0, "unit": "px"},
            },
            "classificationLabels": {
                "0": {"name": "Background"},
                "1": {"name": "Tissue"},
            },
        },
        "op": {
            "type": "data.op.channels",
            "colorTransforms": [{"channelName": "Red"}],
            "op": {
                "type": "op.core.sequential",
                "ops": [
                    {
                        "type": "op.core.sequential",
                        "ops": [
                            {
                                "type": "op.core.split-merge",
                                "ops": [
                                    {
                                        "type": "op.filters.multiscale",
                                        "features": ["GAUSSIAN"],
                                        "sigmaX": 1.0,
                                        "sigmaY": 1.0,
                                    }
                                ],
                            },
                            {"type": "op.ml.feature-preprocessor"},
                        ],
                    },
                    {"type": "op.ml.opencv-statmodel"},
                    {"type": "op.core.convert", "pixelType": "UINT8"},
                ],
            },
        },
    }


def _training_pair() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :6, 2] = 20
    image[:, 6:, 2] = 220
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[:, 6:] = 1
    return image, labels


def test_trains_saves_reloads_and_predicts_binary_mask(tmp_path):
    recipe = parse_feature_recipe(_recipe())
    image, labels = _training_pair()
    model_path = tmp_path / "block_rtrees.yml.gz"
    sidecar_path = tmp_path / "block_rtrees.recipe.json"

    train_rtrees_segmenter(
        recipe,
        [(image, labels)],
        positive_class_id=1,
        model_path=model_path,
        sidecar_path=sidecar_path,
        rng_seed=123,
    )

    assert model_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["rng_seed"] == 123
    assert sidecar["positive_class_id"] == 1
    segmenter = load_rtrees_segmenter(model_path, sidecar_path)
    mask = segmenter.predict(image)
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask[:, :6]) == 0
    assert np.count_nonzero(mask[:, 6:]) == 12 * 6


def test_training_can_cap_samples_per_class_without_losing_a_class(tmp_path):
    recipe = parse_feature_recipe(_recipe())
    image, labels = _training_pair()
    model_path = tmp_path / "block_rtrees.yml.gz"
    sidecar_path = tmp_path / "block_rtrees.recipe.json"

    train_rtrees_segmenter(
        recipe,
        [(image, labels)],
        positive_class_id=1,
        model_path=model_path,
        sidecar_path=sidecar_path,
        rng_seed=123,
        max_samples_per_class=4,
    )

    mask = load_rtrees_segmenter(model_path, sidecar_path).predict(image)
    assert np.count_nonzero(mask[:, :6]) == 0
    assert np.count_nonzero(mask[:, 6:]) == 12 * 6
