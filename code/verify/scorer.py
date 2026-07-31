"""Deployed claimed-pair scorer: normalize, lock one pose, route one metric."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from constants import (
    PASS_THRESHOLD,
    POINT_LAYOUT_AREA_WEIGHT,
    POINT_LAYOUT_SHAPE_WEIGHT,
    SHAPE_ROUTER_SIZE_THRESHOLD,
)
from verify.locked_alignment import align_normalized_masks, radial_normalize_mask
from session.preparation import PreparedSpecimen
from runtime_observer import RuntimeObserver, observed


@dataclass(frozen=True)
class _ComponentFeatures:
    points: np.ndarray
    areas: np.ndarray
    shapes: np.ndarray


@dataclass(frozen=True)
class LockedScoreCache:
    """Pair-independent production facts for one prepared specimen."""

    normalized_mask: np.ndarray
    component_features: _ComponentFeatures


@dataclass(frozen=True)
class ProductionScoreResult:
    score: float
    selected_metric: str
    router_size_signal: float
    block_occupied_fraction: float
    slide_occupied_fraction: float
    best_angle: float
    best_flip: bool
    align_soft_iou: float
    mask_iou: float
    point_layout: float | None = None


def _component_features(mask: np.ndarray) -> _ComponentFeatures:
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        width = float(stats[label, cv2.CC_STAT_WIDTH])
        height = float(stats[label, cv2.CC_STAT_HEIGHT])
        components.append((
            area,
            float(centroids[label][0]), float(centroids[label][1]),
            width, height, area / max(width * height, 1.0),
        ))
    if not components:
        return _ComponentFeatures(np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3)))
    x0 = min(c[1] - c[3] / 2 for c in components)
    y0 = min(c[2] - c[4] / 2 for c in components)
    x1 = max(c[1] + c[3] / 2 for c in components)
    y1 = max(c[2] + c[4] / 2 for c in components)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    total = sum(c[0] for c in components)
    return _ComponentFeatures(
        points=np.asarray([((c[1] - x0) / width, (c[2] - y0) / height) for c in components]),
        areas=np.asarray([c[0] / total for c in components]),
        shapes=np.asarray([(c[3] / width, c[4] / height, c[5]) for c in components]),
    )


def point_layout_similarity(
    block_features: _ComponentFeatures,
    slide_features: _ComponentFeatures,
) -> float:
    """Compare component constellations at the already locked pose."""
    a, b = block_features.points, slide_features.points
    if len(a) == 0 or len(b) == 0:
        return 0.0
    padded, rows, columns = _point_layout_assignment(block_features, slide_features)
    size = max(len(a), len(b))
    return float(math.exp(-2.0 * padded[rows, columns].sum() / size))


def _point_layout_assignment(
    block_features: _ComponentFeatures,
    slide_features: _ComponentFeatures,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact padded costs and assignment used by point layout."""
    a, b = block_features.points, slide_features.points
    size = max(len(a), len(b))
    if size == 0:
        empty_indices = np.zeros(0, dtype=int)
        return np.zeros((0, 0), dtype=float), empty_indices, empty_indices

    padded = np.full((size, size), 0.95, dtype=float)
    if len(a) == 0 or len(b) == 0:
        rows, columns = linear_sum_assignment(padded)
        return padded, rows, columns

    costs = cdist(a, b)
    costs += POINT_LAYOUT_AREA_WEIGHT * np.abs(np.log(
        (block_features.areas[:, None] + 1e-4)
        / (slide_features.areas[None, :] + 1e-4)
    ))
    costs += POINT_LAYOUT_SHAPE_WEIGHT * cdist(block_features.shapes, slide_features.shapes)
    padded[:len(a), :len(b)] = costs
    rows, columns = linear_sum_assignment(padded)
    return padded, rows, columns


def build_locked_score_cache(specimen: PreparedSpecimen) -> LockedScoreCache:
    normalized = radial_normalize_mask(specimen.mask)
    return LockedScoreCache(normalized, _component_features(normalized))


def score_routed_caches(
    block: LockedScoreCache,
    slide: LockedScoreCache,
    *,
    observer: RuntimeObserver | None = None,
    item_id: str = "",
) -> ProductionScoreResult:
    block_fraction = float(block.normalized_mask.mean())
    slide_fraction = float(slide.normalized_mask.mean())
    size_signal = min(block_fraction, slide_fraction)
    with observed(observer, "alignment_scoring", item_id):
        alignment = align_normalized_masks(
            block.normalized_mask, slide.normalized_mask
        )
    if size_signal >= SHAPE_ROUTER_SIZE_THRESHOLD:
        selected_metric, point_layout, score = "mask_iou", None, alignment.mask_iou
    else:
        point_layout = point_layout_similarity(
            block.component_features,
            _component_features(alignment.aligned_slide_mask),
        )
        selected_metric, score = "point_layout", point_layout
    return ProductionScoreResult(
        score=float(max(0.0, min(1.0, score))),
        selected_metric=selected_metric,
        router_size_signal=size_signal,
        block_occupied_fraction=block_fraction,
        slide_occupied_fraction=slide_fraction,
        best_angle=alignment.best_angle,
        best_flip=alignment.best_flip,
        align_soft_iou=alignment.align_soft_iou,
        mask_iou=alignment.mask_iou,
        point_layout=point_layout,
    )


def score_pair_result_routed(
    block: PreparedSpecimen,
    slide: PreparedSpecimen,
    *,
    observer: RuntimeObserver | None = None,
    item_id: str = "",
) -> ProductionScoreResult:
    with observed(observer, "locked_cache", item_id):
        block_cache = build_locked_score_cache(block)
        slide_cache = build_locked_score_cache(slide)
    return score_routed_caches(
        block_cache, slide_cache, observer=observer, item_id=item_id
    )


def decide(score: float) -> tuple[str, str]:
    if score >= PASS_THRESHOLD:
        return "PASS", f"score {score:.3f} >= threshold {PASS_THRESHOLD}"
    return "REVIEW", f"score {score:.3f} below threshold {PASS_THRESHOLD}"
