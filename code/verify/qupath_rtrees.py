"""Train and run OpenCV RTrees from a parsed QuPath feature recipe.

The QuPath export supplies a feature family only.  The trees are always fitted
in OpenCV, so prediction uses exactly the same feature planes as training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Iterable

import cv2
import numpy as np

from verify.qupath_features import (
    FeatureRecipe,
    FeatureSpec,
    build_feature_planes,
    parse_feature_recipe,
)


@dataclass(frozen=True)
class RtreesSegmenter:
    """Loaded model and the recipe required to make its input rows."""

    model: cv2.ml_RTrees
    recipe: FeatureRecipe
    positive_class_id: int

    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return a full-resolution binary uint8 mask for one BGR image."""
        planes = build_feature_planes(bgr_image, self.recipe)
        height, width = planes.shape[:2]
        rows = planes.reshape(-1, planes.shape[2])
        _, classes = self.model.predict(rows)
        small_mask = (classes.reshape(height, width) == self.positive_class_id).astype(
            np.uint8
        ) * 255
        if small_mask.shape == bgr_image.shape[:2]:
            return small_mask
        return cv2.resize(
            small_mask,
            (bgr_image.shape[1], bgr_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )


def train_rtrees_segmenter(
    recipe: FeatureRecipe,
    training_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    positive_class_id: int,
    model_path: str | Path,
    sidecar_path: str | Path,
    rng_seed: int,
    max_samples_per_class: int | None = None,
) -> RtreesSegmenter:
    """Fit, persist, and return a retrained segmenter from image/class-mask pairs."""
    if max_samples_per_class is not None and max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive when provided")
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    sampler = np.random.default_rng(rng_seed)
    for image, labels in training_pairs:
        planes = build_feature_planes(image, recipe)
        resized_labels = _labels_at_feature_resolution(labels, planes.shape[:2])
        samples = planes.reshape(-1, planes.shape[2])
        responses = resized_labels.reshape(-1, 1).astype(np.int32, copy=False)
        sampled_rows = _sample_class_rows(
            responses, max_samples_per_class=max_samples_per_class, sampler=sampler
        )
        feature_rows.append(samples[sampled_rows])
        label_rows.append(responses[sampled_rows])
    if not feature_rows:
        raise ValueError("training requires at least one image/mask pair")

    samples = np.concatenate(feature_rows, axis=0)
    responses = np.concatenate(label_rows, axis=0)
    if positive_class_id not in set(np.unique(responses).tolist()):
        raise ValueError(f"positive class ID {positive_class_id} is absent from training masks")

    cv2.setRNGSeed(rng_seed)
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(2)
    model.setMaxCategories(max(2, len(recipe.class_map)))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    if not model.train(samples, cv2.ml.ROW_SAMPLE, responses):
        raise RuntimeError("OpenCV RTrees training failed")

    destination = Path(model_path)
    sidecar_destination = Path(sidecar_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar_destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    sidecar_destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rng_seed": rng_seed,
                "positive_class_id": positive_class_id,
                "max_samples_per_class": max_samples_per_class,
                "recipe": _recipe_to_dict(recipe),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RtreesSegmenter(model, recipe, positive_class_id)


def load_rtrees_segmenter(
    model_path: str | Path, sidecar_path: str | Path
) -> RtreesSegmenter:
    """Load a persisted OpenCV model and its feature-recipe sidecar."""
    model_source = Path(model_path)
    sidecar_source = Path(sidecar_path)
    if not model_source.is_file():
        raise ValueError(f"RTrees model artifact is missing: {model_source}")
    if not sidecar_source.is_file():
        raise ValueError(f"RTrees sidecar artifact is missing: {sidecar_source}")
    try:
        payload = json.loads(sidecar_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read RTrees sidecar artifact: {sidecar_source}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported RTrees recipe sidecar schema")
    recipe = _recipe_from_dict(payload.get("recipe"))
    try:
        positive_class_id = int(payload["positive_class_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RTrees recipe sidecar lacks positive_class_id") from exc
    if positive_class_id not in recipe.class_map:
        raise ValueError("RTrees positive_class_id is not in the recipe class map")
    try:
        model = cv2.ml.RTrees_load(str(model_source))
    except cv2.error as exc:
        raise ValueError(f"could not load RTrees model artifact: {model_source}") from exc
    if model.empty():
        raise ValueError(f"could not load RTrees model artifact: {model_source}")
    expected_feature_count = len(recipe.channels) * len(recipe.features)
    if model.getVarCount() != expected_feature_count:
        raise ValueError(
            "RTrees model/sidecar feature count mismatch: "
            f"model has {model.getVarCount()}, recipe requires {expected_feature_count}"
        )
    return RtreesSegmenter(model, recipe, positive_class_id)


def load_qupath_rtrees_segmenter(path: str | Path) -> RtreesSegmenter:
    """Load a saved QuPath OpenCV RTrees classifier without retraining it.

    QuPath nests OpenCV's ``opencv_ml_rtrees`` node inside its classifier JSON.
    OpenCV Python expects that node at the document root, so a temporary
    standards-compatible wrapper lets it load the exact saved tree ensemble.
    """
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"QuPath classifier artifact is missing: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not read QuPath classifier artifact: {source}") from exc

    recipe = parse_feature_recipe(document)
    try:
        statmodel = document["op"]["op"]["ops"][1]["model"]["statmodel"]
        rtrees = statmodel["opencv_ml_rtrees"]
        normalizer = document["op"]["op"]["ops"][0]["ops"][1]["preprocessor"][
            "normalizer"
        ]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("QuPath classifier does not contain an OpenCV RTrees model") from exc
    if not isinstance(rtrees, dict):
        raise ValueError("QuPath classifier RTrees model must be an object")
    try:
        offsets = [float(value) for value in normalizer["offsets"]]
        scales = [float(value) for value in normalizer["scales"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("QuPath classifier has an invalid feature normalizer") from exc
    if any(offset != 0.0 for offset in offsets) or any(scale != 1.0 for scale in scales):
        raise ValueError(
            "QuPath classifier feature normalization is not identity and is unsupported"
        )

    tissue_ids = [
        class_id for class_id, name in recipe.class_map.items() if name.lower() == "tissue"
    ]
    if len(tissue_ids) != 1:
        raise ValueError("QuPath classifier must define exactly one Tissue class")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump({"opencv_ml_rtrees": rtrees}, handle)
    try:
        model = cv2.ml.RTrees_load(str(temporary_path))
    except cv2.error as exc:
        raise ValueError(f"could not load QuPath RTrees model: {source}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    if model.empty():
        raise ValueError(f"could not load QuPath RTrees model: {source}")
    expected_feature_count = len(recipe.channels) * len(recipe.features)
    if model.getVarCount() != expected_feature_count:
        raise ValueError(
            "QuPath RTrees/recipe feature count mismatch: "
            f"model has {model.getVarCount()}, recipe requires {expected_feature_count}"
        )
    return RtreesSegmenter(model, recipe, tissue_ids[0])


def _labels_at_feature_resolution(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("training masks must be 2D uint8 class-ID arrays")
    height, width = shape
    if labels.shape == shape:
        return labels
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def _sample_class_rows(
    responses: np.ndarray,
    *,
    max_samples_per_class: int | None,
    sampler: np.random.Generator,
) -> np.ndarray:
    """Select every row, or a deterministic cap for each represented class."""
    if max_samples_per_class is None:
        return np.arange(responses.shape[0])
    selected: list[np.ndarray] = []
    flattened = responses.reshape(-1)
    for class_id in np.unique(flattened):
        rows = np.flatnonzero(flattened == class_id)
        if rows.size > max_samples_per_class:
            rows = sampler.choice(rows, size=max_samples_per_class, replace=False)
        selected.append(rows)
    return np.concatenate(selected)


def _recipe_to_dict(recipe: FeatureRecipe) -> dict:
    return {
        "channels": list(recipe.channels),
        "features": [
            {"operation": spec.operation, "sigma": spec.sigma}
            for spec in recipe.features
        ],
        "feature_groups": [
            [
                {"operation": spec.operation, "sigma": spec.sigma}
                for spec in group
            ]
            for group in recipe.feature_groups
        ],
        "downsample": recipe.downsample,
        "class_map": {str(key): value for key, value in recipe.class_map.items()},
    }


def _recipe_from_dict(value: object) -> FeatureRecipe:
    if not isinstance(value, dict):
        raise ValueError("RTrees recipe sidecar lacks recipe")
    try:
        channels = tuple(value["channels"])
        features = tuple(
            FeatureSpec(operation=item["operation"], sigma=float(item["sigma"]))
            for item in value["features"]
        )
        raw_groups = value.get("feature_groups")
        feature_groups = (
            tuple(
                tuple(
                    FeatureSpec(operation=item["operation"], sigma=float(item["sigma"]))
                    for item in group
                )
                for group in raw_groups
            )
            if isinstance(raw_groups, list)
            else ()
        )
        class_map = {int(key): name for key, name in value["class_map"].items()}
        recipe = FeatureRecipe(
            channels=channels,
            features=features,
            downsample=float(value["downsample"]),
            class_map=class_map,
            feature_groups=feature_groups,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid RTrees recipe sidecar") from exc
    if not recipe.channels or not recipe.features or recipe.downsample <= 0:
        raise ValueError("invalid RTrees recipe sidecar")
    return recipe
