"""Behavior tests for the reusable QuPath feature-recipe engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from verify.qupath_features import (
    FeatureRecipeError,
    _downsample,
    build_feature_planes,
    parse_feature_recipe,
)


def _export(*, features: list[str], sigma: float = 1.0) -> dict:
    return {
        "pixel_classifier_type": "OpenCVPixelClassifier",
        "metadata": {
            "inputResolution": {
                "pixelWidth": {"value": 2.0, "unit": "px"},
                "pixelHeight": {"value": 2.0, "unit": "px"},
            },
            "classificationLabels": {
                "0": {"name": "Ignore*"},
                "1": {"name": "Tissue"},
            },
        },
        "op": {
            "type": "data.op.channels",
            "colorTransforms": [
                {"channelName": "Red"},
                {"channelName": "Green"},
                {"channelName": "Blue"},
            ],
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
                                        "features": features,
                                        "sigmaX": sigma,
                                        "sigmaY": sigma,
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


def test_parses_declared_channels_operations_scales_resolution_and_classes():
    recipe = parse_feature_recipe(
        _export(features=["GAUSSIAN", "LAPLACIAN", "GRADIENT_MAGNITUDE"])
    )

    assert recipe.channels == ("Red", "Green", "Blue")
    assert tuple(spec.operation for spec in recipe.features) == (
        "GAUSSIAN",
        "LAPLACIAN_OF_GAUSSIAN",
        "GRADIENT_MAGNITUDE",
    )
    assert tuple(spec.sigma for spec in recipe.features) == (1.0, 1.0, 1.0)
    assert recipe.downsample == 2.0
    assert recipe.class_map == {0: "Ignore*", 1: "Tissue"}


def test_builds_ordered_finite_planes_at_recipe_resolution():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, :, 2] = np.arange(8, dtype=np.uint8)  # red ramp in BGR input
    recipe = parse_feature_recipe(_export(features=["GAUSSIAN", "LAPLACIAN"]))

    planes = build_feature_planes(image, recipe)

    assert planes.shape == (3, 4, 6)
    assert planes.dtype == np.float32
    assert np.isfinite(planes).all()
    assert planes[:, :, 0].max() > 0  # Gaussian Red is the first declared plane.
    assert planes[:, :, 1].min() < 0  # LoG Red follows Gaussian Red.
    assert np.allclose(planes[:, :, 2], 0)  # Gaussian Green follows Red's features.
    assert np.allclose(planes[:, :, 3], 0)  # LoG Green follows Gaussian Green.


def test_preserves_recipe_order_across_multiple_scales_and_operation_families():
    export = _export(features=["GAUSSIAN"], sigma=1.0)
    export["op"]["op"]["ops"][0]["ops"][0]["ops"] = [
        {
            "type": "op.filters.multiscale",
            "features": ["GAUSSIAN", "GRADIENT_MAGNITUDE"],
            "sigmaX": 1.0,
            "sigmaY": 1.0,
        },
        {
            "type": "op.filters.multiscale",
            "features": ["LAPLACIAN"],
            "sigmaX": 3.0,
            "sigmaY": 3.0,
        },
    ]
    recipe = parse_feature_recipe(export)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:, 2] = 255

    planes = build_feature_planes(image, recipe)

    assert [(spec.operation, spec.sigma) for spec in recipe.features] == [
        ("GAUSSIAN", 1.0),
        ("GRADIENT_MAGNITUDE", 1.0),
        ("LAPLACIAN_OF_GAUSSIAN", 3.0),
    ]
    assert planes.shape == (4, 4, 9)
    # QuPath builds every requested feature for one channel before advancing to
    # the next channel.  The saved RTrees therefore expects scale -> channel
    # -> feature, not scale -> feature -> channel.
    assert planes[:, :, 0].max() > 0  # Gaussian Red
    assert planes[:, :, 1].max() > 0  # Gradient Red
    assert np.allclose(planes[:, :, 2], 0)  # Gaussian Green
    assert np.allclose(planes[:, :, 3], 0)  # Gradient Green
    assert np.allclose(planes[:, :, 4], 0)  # Gaussian Blue
    assert np.allclose(planes[:, :, 5], 0)  # Gradient Blue
    assert planes[:, :, 6].min() < 0  # LoG Red at the next scale


def _qupath_derivative_kernel(sigma: float, order: int) -> np.ndarray:
    n = int(sigma * 4)
    x = np.arange(-n, n + 1, dtype=np.float64)
    gaussian = np.exp(-(x * x) / (2 * sigma * sigma))
    denom = sigma * np.sqrt(2 * np.pi)
    if order == 0:
        values = gaussian / denom
    elif order == 1:
        values = -x * gaussian / (denom * sigma * sigma)
    elif order == 2:
        values = -(sigma * sigma - x * x) * gaussian / (denom * sigma**4)
    else:
        raise ValueError("only QuPath derivative orders 0-2 are supported")
    return values.astype(np.float32)


def test_uses_qupath_gaussian_derivative_kernels_for_gradient_and_log():
    export = _export(features=["GRADIENT_MAGNITUDE", "LAPLACIAN"], sigma=1.5)
    export["op"]["colorTransforms"] = [{"channelName": "Red"}]
    image = np.zeros((15, 17, 3), dtype=np.uint8)
    image[4:12, 7:15, 2] = 200
    recipe = parse_feature_recipe(export)

    planes = build_feature_planes(image, recipe)

    channel = cv2.resize(image[:, :, 2], (9, 8), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    k0 = _qupath_derivative_kernel(1.5, 0)
    k1 = _qupath_derivative_kernel(1.5, 1)
    k2 = _qupath_derivative_kernel(1.5, 2)
    dx = cv2.sepFilter2D(channel, cv2.CV_32F, k1, k0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.sepFilter2D(channel, cv2.CV_32F, k0, k1, borderType=cv2.BORDER_REPLICATE)
    expected_gradient = cv2.magnitude(dx, dy)
    dxx = cv2.sepFilter2D(channel, cv2.CV_32F, k2, k0, borderType=cv2.BORDER_REPLICATE)
    dyy = cv2.sepFilter2D(channel, cv2.CV_32F, k0, k2, borderType=cv2.BORDER_REPLICATE)

    np.testing.assert_allclose(planes[:, :, 0], expected_gradient, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(planes[:, :, 1], dxx + dyy, rtol=1e-5, atol=1e-5)


def test_builds_qupath_weighted_standard_deviation_with_replicated_border():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2, 2] = 255
    recipe = parse_feature_recipe(_export(features=["WEIGHTED_STD_DEV"], sigma=1.0))

    planes = build_feature_planes(image, recipe)

    channel = cv2.resize(
        image[:, :, 2], (3, 3), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    kernel = _qupath_derivative_kernel(1.0, 0)
    mean = cv2.sepFilter2D(channel, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REPLICATE)
    mean_squared = cv2.sepFilter2D(
        channel * channel, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REPLICATE
    )
    expected = np.sqrt(np.maximum(mean_squared - mean * mean, 0))

    np.testing.assert_allclose(planes[:, :, 0], expected, rtol=1e-5, atol=1e-5)
    assert np.allclose(planes[:, :, 1:], 0)


def test_honors_declared_channel_order_and_ceiling_downsample_shape():
    export = _export(features=["GAUSSIAN"])
    export["op"]["colorTransforms"] = [
        {"channelName": "Blue"},
        {"channelName": "Red"},
    ]
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 2] = 80

    planes = build_feature_planes(image, parse_feature_recipe(export))

    assert planes.shape == (3, 4, 2)
    assert planes[:, :, 0].mean() < planes[:, :, 1].mean()


def test_downsamples_like_qupaths_linear_image_server_request():
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    result = _downsample(image, 2.0)

    expected = cv2.resize(image, (4, 3), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result, expected)


def test_rejects_unrecognized_operation_graph_siblings():
    export = _export(features=["GAUSSIAN"])
    export["op"]["op"]["ops"].append({"type": "op.core.identity"})

    with pytest.raises(FeatureRecipeError, match="expected feature sequence"):
        parse_feature_recipe(export)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (["MEDIAN"], "unsupported QuPath feature operation"),
        (["GAUSSIAN"], "sigmaX and sigmaY"),
    ],
)
def test_rejects_unsupported_or_malformed_feature_recipes(features, message):
    export = _export(features=features, sigma=0.0 if features == ["GAUSSIAN"] else 1.0)

    with pytest.raises(FeatureRecipeError, match=message):
        parse_feature_recipe(export)


def test_rejects_non_bgr_image_input():
    recipe = parse_feature_recipe(_export(features=["GAUSSIAN"]))

    with pytest.raises(FeatureRecipeError, match="uint8 BGR"):
        build_feature_planes(np.zeros((5, 5), dtype=np.uint8), recipe)
