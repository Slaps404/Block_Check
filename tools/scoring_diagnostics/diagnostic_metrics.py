"""Pure diagnostic metrics and per-specimen caches for ablation experiments.

Nothing in this module participates in production PASS/REVIEW scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

SILHOUETTE_BASED = "Silhouette-Based Descriptor"
COMPONENT_BASED = "Component-Based Descriptor"
SOFT_CORRELATION_SIGMA = 8.0

NEW_METRIC_NAMES = (
    "soft_dilated_iou",
    "symmetric_chamfer_mean",
    "symmetric_chamfer_p90",
    "modified_hausdorff",
    "polar_histogram_similarity",
    "hu_moment_similarity",
)
BASELINE_METRIC_NAMES = (
    "point_layout",
    "soft_correlation",
    "mask_iou",
    "distance_signature",
)
METRIC_TAGS = {name: SILHOUETTE_BASED for name in NEW_METRIC_NAMES} | {
    "point_layout": COMPONENT_BASED,
    "soft_correlation": SILHOUETTE_BASED,
    "mask_iou": SILHOUETTE_BASED,
    "distance_signature": COMPONENT_BASED,
}


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("metric masks must be two-dimensional")
    return (mask > 0).astype(np.uint8)


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _same_or_empty(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.array_equal(a, b):
        return 1.0
    if not np.any(a) or not np.any(b):
        return 0.0
    return None


def soft_dilated_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU after a small dilation makes boundary jitter less expensive."""
    a, b = _binary(mask_a), _binary(mask_b)
    special = _same_or_empty(a, b)
    if special is not None:
        return special
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    a, b = cv2.dilate(a, kernel), cv2.dilate(b, kernel)
    union = np.count_nonzero(a | b)
    return _clamp(np.count_nonzero(a & b) / max(union, 1))


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask > eroded).astype(np.uint8)


def _directed_boundary_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_boundary = _boundary(target)
    distance = cv2.distanceTransform(1 - target_boundary, cv2.DIST_L2, 3)
    return distance[_boundary(source) > 0]


def _distance_similarity(mask_a: np.ndarray, mask_b: np.ndarray, reducer) -> float:
    a, b = _binary(mask_a), _binary(mask_b)
    special = _same_or_empty(a, b)
    if special is not None:
        return special
    forward = reducer(_directed_boundary_distances(a, b))
    reverse = reducer(_directed_boundary_distances(b, a))
    diagonal = math.hypot(*a.shape)
    return _clamp(math.exp(-4.0 * float((forward + reverse) / 2.0) / diagonal))


def symmetric_chamfer_mean(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    return _distance_similarity(mask_a, mask_b, np.mean)


def symmetric_chamfer_p90(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    return _distance_similarity(mask_a, mask_b, lambda x: np.percentile(x, 90))


def modified_hausdorff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Similarity from the worse directed mean boundary distance."""
    a, b = _binary(mask_a), _binary(mask_b)
    special = _same_or_empty(a, b)
    if special is not None:
        return special
    distance = max(
        float(np.mean(_directed_boundary_distances(a, b))),
        float(np.mean(_directed_boundary_distances(b, a))),
    )
    return _clamp(math.exp(-4.0 * distance / math.hypot(*a.shape)))


def polar_histogram_descriptor(mask: np.ndarray, bins: int = 32) -> np.ndarray:
    """Rotation-invariant foreground-radius histogram around its centroid."""
    binary = _binary(mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return np.zeros(bins, dtype=np.float32)
    radii = np.hypot(xs - xs.mean(), ys - ys.mean())
    maximum = max(float(radii.max()), 1.0)
    hist, _ = np.histogram(radii / maximum, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def hu_moment_descriptor(mask: np.ndarray) -> np.ndarray:
    hu = cv2.HuMoments(cv2.moments(_binary(mask))).flatten()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).astype(np.float32)


def _descriptor_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    if not np.any(a) and not np.any(b):
        return 1.0
    return _clamp(math.exp(-float(np.linalg.norm(a - b)) / scale))


def polar_histogram_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = _binary(mask_a), _binary(mask_b)
    special = _same_or_empty(a, b)
    if special is not None:
        return special
    return _descriptor_similarity(
        polar_histogram_descriptor(a), polar_histogram_descriptor(b), 0.25
    )


def hu_moment_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = _binary(mask_a), _binary(mask_b)
    special = _same_or_empty(a, b)
    if special is not None:
        return special
    return _descriptor_similarity(hu_moment_descriptor(a), hu_moment_descriptor(b), 8.0)


@dataclass(frozen=True)
class SpecimenMetricCache:
    normalized_mask: np.ndarray
    hu_moments: np.ndarray
    polar_histogram: np.ndarray
    distance_signature: np.ndarray


def build_specimen_metric_cache(mask: np.ndarray) -> SpecimenMetricCache:
    """Compute every reusable specimen fact exactly once before the N x M loop."""
    from verify.locked_alignment import radial_normalize_mask
    normalized = radial_normalize_mask(mask)
    return SpecimenMetricCache(
        normalized_mask=normalized,
        hu_moments=hu_moment_descriptor(normalized),
        polar_histogram=polar_histogram_descriptor(normalized),
        distance_signature=_distance_signature(normalized),
    )


def build_specimen_metric_cache_from_locked(locked) -> SpecimenMetricCache:
    """Derive ablation metric facts from an existing ``LockedScoreCache``.

    Avoids a second radial normalize + ``_build_features`` pass when diagnostics
    already built the locked cache for production-path scoring.
    """
    normalized = locked.normalized_mask
    return SpecimenMetricCache(
        normalized_mask=normalized,
        hu_moments=hu_moment_descriptor(normalized),
        polar_histogram=polar_histogram_descriptor(normalized),
        distance_signature=_distance_signature(normalized),
    )


def score_locked_metrics(
    block: SpecimenMetricCache,
    slide: SpecimenMetricCache,
    aligned_slide_mask: np.ndarray,
    *,
    block_soft_blur: np.ndarray | None = None,
) -> dict[str, float]:
    """Return all raw metric scores at one pair's locked alignment."""
    from verify.scorer import _component_features, point_layout_similarity
    if block_soft_blur is not None:
        block_soft = block_soft_blur
    else:
        block_soft = cv2.GaussianBlur(
            block.normalized_mask.astype(np.float32), (0, 0), SOFT_CORRELATION_SIGMA
        )
    slide_soft = cv2.GaussianBlur(
        aligned_slide_mask.astype(np.float32), (0, 0), SOFT_CORRELATION_SIGMA
    )
    distance_diff = float(np.linalg.norm(
        block.distance_signature - slide.distance_signature
    ))
    scores = {
        "point_layout": point_layout_similarity(
            _component_features(block.normalized_mask),
            _component_features(aligned_slide_mask),
        ),
        "soft_correlation": _safe_correlation(block_soft, slide_soft),
        "mask_iou": _iou(block.normalized_mask, aligned_slide_mask),
        "distance_signature": math.exp(-1.5 * distance_diff),
        "soft_dilated_iou": soft_dilated_iou(
            block.normalized_mask, aligned_slide_mask
        ),
        "symmetric_chamfer_mean": symmetric_chamfer_mean(
            block.normalized_mask, aligned_slide_mask
        ),
        "symmetric_chamfer_p90": symmetric_chamfer_p90(
            block.normalized_mask, aligned_slide_mask
        ),
        "modified_hausdorff": modified_hausdorff(
            block.normalized_mask, aligned_slide_mask
        ),
        "polar_histogram_similarity": _descriptor_similarity(
            block.polar_histogram, slide.polar_histogram, 0.25
        ),
        "hu_moment_similarity": _descriptor_similarity(
            block.hu_moments, slide.hu_moments, 8.0
        ),
    }
    return {name: _clamp(float(value)) for name, value in scores.items()}


def _distance_signature(mask: np.ndarray, bins: int = 8) -> np.ndarray:
    count, _, _, centroids = cv2.connectedComponentsWithStats(_binary(mask), 8)
    points = centroids[1:count]
    if len(points) < 2:
        return np.zeros(bins, dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    values = distances[np.triu_indices(len(points), 1)]
    maximum = max(float(values.max()), 1.0)
    histogram, _ = np.histogram(values / maximum, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    left, right = a.astype(float).ravel(), b.astype(float).ravel()
    if left.std() < 1e-9 or right.std() < 1e-9:
        return 0.0
    return float((np.corrcoef(left, right)[0, 1] + 1.0) / 2.0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero((a > 0) | (b > 0))
    return float(np.count_nonzero((a > 0) & (b > 0)) / union) if union else 0.0
