"""Parse supported QuPath pixel-classifier recipes and build OpenCV features.

This module deliberately reads only the feature recipe, never QuPath's saved
classifier.  Fresh OpenCV models are trained on these planes by later slices.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


class FeatureRecipeError(ValueError):
    """A QuPath export or feature-engine input is outside the supported contract."""


@dataclass(frozen=True)
class FeatureSpec:
    """One operation at one scale, applied to every declared color channel."""

    operation: str
    sigma: float


@dataclass(frozen=True)
class FeatureRecipe:
    """Validated feature contract shared by training and prediction."""

    channels: tuple[str, ...]
    features: tuple[FeatureSpec, ...]
    downsample: float
    class_map: dict[int, str]
    feature_groups: tuple[tuple[FeatureSpec, ...], ...] = ()


_FEATURE_NAMES = {
    "GAUSSIAN": "GAUSSIAN",
    "LAPLACIAN": "LAPLACIAN_OF_GAUSSIAN",
    "LAPLACIAN_OF_GAUSSIAN": "LAPLACIAN_OF_GAUSSIAN",
    "GRADIENT_MAGNITUDE": "GRADIENT_MAGNITUDE",
    "WEIGHTED_STD_DEV": "WEIGHTED_STD_DEV",
}
_BGR_INDEX = {"Blue": 0, "Green": 1, "Red": 2}


def load_feature_recipe(path: str | Path) -> FeatureRecipe:
    """Read one QuPath JSON export and return its validated feature recipe."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FeatureRecipeError(f"could not read QuPath recipe {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FeatureRecipeError(f"invalid QuPath JSON in {source}: {exc.msg}") from exc
    return parse_feature_recipe(document)


def parse_feature_recipe(document: Any) -> FeatureRecipe:
    """Extract the supported QuPath channel/operation/class feature contract."""
    root = _mapping(document, "QuPath recipe root")
    if root.get("pixel_classifier_type") != "OpenCVPixelClassifier":
        raise FeatureRecipeError(
            "unsupported pixel classifier type; expected OpenCVPixelClassifier"
        )

    metadata = _mapping(root.get("metadata"), "metadata")
    downsample = _parse_downsample(metadata)
    class_map = _parse_class_map(metadata.get("classificationLabels"))

    channel_op = _mapping(root.get("op"), "channel operation")
    if channel_op.get("type") != "data.op.channels":
        raise FeatureRecipeError("unsupported channel operation; expected data.op.channels")
    channels = _parse_channels(channel_op.get("colorTransforms"))
    feature_ops = _find_feature_ops(channel_op.get("op"))

    groups: list[tuple[FeatureSpec, ...]] = []
    for index, operation in enumerate(feature_ops):
        op = _mapping(operation, f"feature operation {index}")
        if op.get("type") != "op.filters.multiscale":
            raise FeatureRecipeError(
                f"unsupported feature operation family at index {index}: {op.get('type')!r}"
            )
        sigma = _parse_sigma(op, index)
        names = op.get("features")
        if not isinstance(names, list) or not names:
            raise FeatureRecipeError(
                f"feature operation {index} must declare a non-empty features list"
            )
        group: list[FeatureSpec] = []
        for name in names:
            canonical = _FEATURE_NAMES.get(name) if isinstance(name, str) else None
            if canonical is None:
                raise FeatureRecipeError(
                    f"unsupported QuPath feature operation at index {index}: {name!r}"
                )
            group.append(FeatureSpec(operation=canonical, sigma=sigma))
        groups.append(tuple(group))

    specs = [spec for group in groups for spec in group]
    if not specs:
        raise FeatureRecipeError("QuPath recipe must declare at least one feature operation")
    return FeatureRecipe(
        channels=channels,
        features=tuple(specs),
        feature_groups=tuple(groups),
        downsample=downsample,
        class_map=class_map,
    )


def build_feature_planes(bgr_image: np.ndarray, recipe: FeatureRecipe) -> np.ndarray:
    """Return finite float32 planes in QuPath scale-then-channel feature order.

    The input is BGR because OpenCV capture/read APIs produce BGR arrays.  The
    recipe's explicit channel names determine the plane order, so a QuPath RGB
    recipe is correctly reordered without callers doing that bookkeeping.
    """
    if not isinstance(recipe, FeatureRecipe):
        raise FeatureRecipeError("recipe must be a FeatureRecipe")
    if (
        not isinstance(bgr_image, np.ndarray)
        or bgr_image.dtype != np.uint8
        or bgr_image.ndim != 3
        or bgr_image.shape[2] != 3
        or bgr_image.shape[0] == 0
        or bgr_image.shape[1] == 0
    ):
        raise FeatureRecipeError(
            "feature input must be a non-empty uint8 BGR image with shape "
            "(height, width, 3)"
        )

    resized = _downsample(bgr_image, recipe.downsample)
    channels = [
        resized[:, :, _BGR_INDEX[name]].astype(np.float32)
        for name in recipe.channels
    ]
    planes: list[np.ndarray] = []
    feature_groups = recipe.feature_groups or (recipe.features,)
    for group in feature_groups:
        for channel in channels:
            planes.extend(_apply_feature(channel, spec) for spec in group)
    output = np.stack(planes, axis=2).astype(np.float32, copy=False)
    # cv2/NumPy buffers can be image-scale; drop temporary references before the
    # caller retains the returned feature stack.
    del planes, channels, resized
    if not np.isfinite(output).all():
        raise FeatureRecipeError("feature builder produced non-finite values")
    return output


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureRecipeError(f"malformed QuPath recipe: {label} must be an object")
    return value


def _parse_downsample(metadata: Mapping[str, Any]) -> float:
    resolution = _mapping(metadata.get("inputResolution"), "metadata.inputResolution")
    width = _mapping(resolution.get("pixelWidth"), "metadata.inputResolution.pixelWidth")
    height = _mapping(resolution.get("pixelHeight"), "metadata.inputResolution.pixelHeight")
    try:
        x, y = float(width["value"]), float(height["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureRecipeError("input resolution must contain numeric pixel widths") from exc
    if width.get("unit") != "px" or height.get("unit") != "px" or x <= 0 or x != y:
        raise FeatureRecipeError(
            "input resolution must be one positive, square px downsample factor"
        )
    return x


def _parse_class_map(value: Any) -> dict[int, str]:
    labels = _mapping(value, "metadata.classificationLabels")
    classes: dict[int, str] = {}
    for raw_id, raw_label in labels.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise FeatureRecipeError(
                f"classification class ID must be an integer: {raw_id!r}"
            ) from exc
        label = _mapping(raw_label, f"classification label {raw_id!r}").get("name")
        if not isinstance(label, str) or not label:
            raise FeatureRecipeError(
                f"classification label {raw_id!r} must have a non-empty name"
            )
        if class_id in classes:
            raise FeatureRecipeError(f"duplicate classification class ID: {class_id}")
        classes[class_id] = label
    if not classes:
        raise FeatureRecipeError("classificationLabels must not be empty")
    return classes


def _parse_channels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise FeatureRecipeError("colorTransforms must be a non-empty list")
    channels = tuple(
        item.get("channelName") if isinstance(item, Mapping) else None
        for item in value
    )
    if any(channel not in _BGR_INDEX for channel in channels):
        raise FeatureRecipeError(
            "only Red, Green, and Blue color channels are supported"
        )
    if len(set(channels)) != len(channels):
        raise FeatureRecipeError("colorTransforms must not repeat a channel")
    return channels  # type: ignore[return-value]


def _find_feature_ops(value: Any) -> list[Any]:
    """Validate the supported QuPath graph prefix and return its multiscale ops."""
    sequential = _mapping(value, "operation graph")
    if sequential.get("type") != "op.core.sequential":
        raise FeatureRecipeError("unsupported operation graph; expected op.core.sequential")
    ops = sequential.get("ops")
    if not isinstance(ops, list) or not ops:
        raise FeatureRecipeError("operation graph must contain feature operations")
    if len(ops) != 3:
        raise FeatureRecipeError(
            "unsupported operation graph; expected feature sequence, classifier, and conversion"
        )
    feature_sequence = _mapping(ops[0], "feature sequence")
    if feature_sequence.get("type") != "op.core.sequential":
        raise FeatureRecipeError("unsupported feature graph; expected sequential feature sequence")
    sequence_ops = feature_sequence.get("ops")
    if not isinstance(sequence_ops, list) or len(sequence_ops) != 2:
        raise FeatureRecipeError(
            "feature sequence must contain split-merge and feature-preprocessor operations"
        )
    split_merge = _mapping(sequence_ops[0], "split-merge operation")
    if split_merge.get("type") != "op.core.split-merge":
        raise FeatureRecipeError("unsupported feature graph; expected op.core.split-merge")
    feature_ops = split_merge.get("ops")
    if not isinstance(feature_ops, list) or not feature_ops:
        raise FeatureRecipeError("split-merge operation must contain feature operations")
    preprocessor = _mapping(sequence_ops[1], "feature preprocessor")
    if preprocessor.get("type") != "op.ml.feature-preprocessor":
        raise FeatureRecipeError("unsupported feature graph; expected op.ml.feature-preprocessor")
    classifier = _mapping(ops[1], "classifier operation")
    if classifier.get("type") != "op.ml.opencv-statmodel":
        raise FeatureRecipeError(
            "unsupported classifier operation; expected op.ml.opencv-statmodel"
        )
    conversion = _mapping(ops[2], "output conversion")
    if conversion.get("type") != "op.core.convert" or conversion.get("pixelType") != "UINT8":
        raise FeatureRecipeError("unsupported output conversion; expected UINT8 conversion")
    return feature_ops


def _parse_sigma(operation: Mapping[str, Any], index: int) -> float:
    try:
        sigma_x = float(operation["sigmaX"])
        sigma_y = float(operation["sigmaY"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureRecipeError(
            f"feature operation {index} must declare numeric sigmaX and sigmaY"
        ) from exc
    if not np.isfinite(sigma_x) or sigma_x <= 0 or sigma_x != sigma_y:
        raise FeatureRecipeError(
            f"feature operation {index} sigmaX and sigmaY must be equal positive finite values"
        )
    return sigma_x


def _downsample(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    height, width = image.shape[:2]
    target_width = max(1, int(np.ceil(width / factor)))
    target_height = max(1, int(np.ceil(height / factor)))
    # QuPath's image-server request at the classifier resolution uses linear
    # resampling.  RTrees decisions near tissue boundaries are sensitive to
    # this choice, so INTER_AREA is not a compatible substitute here.
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _apply_feature(channel: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    k0 = _qupath_derivative_kernel(spec.sigma, 0)
    if spec.operation == "GAUSSIAN":
        return _separable(channel, k0, k0)
    if spec.operation == "LAPLACIAN_OF_GAUSSIAN":
        k2 = _qupath_derivative_kernel(spec.sigma, 2)
        return _separable(channel, k2, k0) + _separable(channel, k0, k2)
    if spec.operation == "GRADIENT_MAGNITUDE":
        k1 = _qupath_derivative_kernel(spec.sigma, 1)
        return cv2.magnitude(_separable(channel, k1, k0), _separable(channel, k0, k1))
    if spec.operation == "WEIGHTED_STD_DEV":
        mean = _separable(channel, k0, k0)
        mean_squared = _separable(channel * channel, k0, k0)
        return np.sqrt(np.maximum(mean_squared - mean * mean, 0))
    raise FeatureRecipeError(f"unsupported normalized feature operation: {spec.operation!r}")


def _separable(channel: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> np.ndarray:
    return cv2.sepFilter2D(
        channel,
        cv2.CV_32F,
        kernel_x,
        kernel_y,
        borderType=cv2.BORDER_REPLICATE,
    )


def _qupath_derivative_kernel(sigma: float, order: int) -> np.ndarray:
    """Reproduce OpenCVTools.getGaussianDerivKernel from QuPath exactly."""
    radius = int(sigma * 4)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian = np.exp(-(x * x) / (2 * sigma * sigma))
    denominator = sigma * np.sqrt(2 * np.pi)
    if order == 0:
        values = gaussian / denominator
    elif order == 1:
        values = -x * gaussian / (denominator * sigma * sigma)
    elif order == 2:
        values = -(sigma * sigma - x * x) * gaussian / (denominator * sigma**4)
    else:
        raise FeatureRecipeError(f"unsupported QuPath derivative order: {order}")
    # QuPath stores each computed coefficient as a float before filtering.
    return values.astype(np.float32)
