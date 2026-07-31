"""Contract tests for the small directory-based QuPath retraining tool."""

from __future__ import annotations

import json

import cv2
import numpy as np

from tools.train_qupath_rtrees import train_from_directories
from verify.qupath_rtrees import load_rtrees_segmenter


def _recipe() -> dict:
    return {
        "pixel_classifier_type": "OpenCVPixelClassifier",
        "metadata": {
            "inputResolution": {
                "pixelWidth": {"value": 1.0, "unit": "px"},
                "pixelHeight": {"value": 1.0, "unit": "px"},
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


def test_trains_from_matching_image_and_class_mask_directories(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    recipe_path = tmp_path / "classifier.json"
    recipe_path.write_text(json.dumps(_recipe()), encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:, 2] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 1
    cv2.imwrite(str(image_dir / "block_a.png"), image)
    cv2.imwrite(str(mask_dir / "block_a.png"), mask)
    model_path = tmp_path / "models" / "block.yml.gz"
    sidecar_path = tmp_path / "models" / "block.recipe.json"

    count = train_from_directories(
        recipe_path,
        image_dir,
        mask_dir,
        positive_class_id=1,
        model_path=model_path,
        sidecar_path=sidecar_path,
        rng_seed=4,
    )

    assert count == 1
    mask_result = load_rtrees_segmenter(model_path, sidecar_path).predict(image)
    assert np.count_nonzero(mask_result[:, :4]) == 0
    assert np.count_nonzero(mask_result[:, 4:]) == 32
