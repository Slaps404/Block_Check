"""Cheap, pair-independent invariant descriptors for retrieval experiments.

These descriptors deliberately summarize a single already-normalized binary mask.
They do not rotate, align, or otherwise inspect a second specimen.  Their scores
are diagnostic retrieval evidence, never a production verification decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class DescriptorSpec:
    """One cacheable descriptor contract used by the retrieval evidence builder."""

    name: str
    version: str
    dimension: int
    comparison: str
    prior_evidence: str


@dataclass(frozen=True)
class DescriptorValue:
    """A descriptor vector plus the one-specimen construction duration."""

    vector: np.ndarray
    construction_ns: int


@dataclass(frozen=True)
class _DescriptorDefinition:
    """Internal single source of truth for a descriptor and its builder."""

    spec: DescriptorSpec
    builder: Callable[[np.ndarray, int], np.ndarray]


def descriptor_catalog() -> tuple[DescriptorSpec, ...]:
    """Return the predeclared, stable descriptor catalog."""
    return tuple(definition.spec for definition in _DESCRIPTORS)


def build_descriptor_values(normalized_mask: np.ndarray) -> dict[str, DescriptorValue]:
    """Build every fixed-size vector from one normalized binary mask.

    ``normalized_mask`` is expected to be the production radial-normalized mask.
    It is binarized defensively here so callers cannot accidentally make a color
    or brightness descriptor part of this experiment.
    """
    binary = _binary_mask(normalized_mask)
    values = {}
    for definition in _DESCRIPTORS:
        started = perf_counter_ns()
        vector = definition.builder(binary, definition.spec.dimension)
        elapsed = perf_counter_ns() - started
        values[definition.spec.name] = DescriptorValue(
            _finite_vector(vector, definition.spec.dimension), elapsed
        )
    return values


def compare_descriptor_values(
    spec: DescriptorSpec, left: DescriptorValue, right: DescriptorValue
) -> float:
    """Return a direct, deterministic higher-is-better similarity in ``[0, 1]``."""
    left_vector = _finite_vector(left.vector, spec.dimension)
    right_vector = _finite_vector(right.vector, spec.dimension)
    if np.array_equal(left_vector, right_vector):
        score = 1.0
    elif spec.comparison == "histogram_intersection":
        score = float(np.minimum(left_vector, right_vector).sum())
    elif spec.comparison == "exp_l1":
        score = float(np.exp(-np.abs(left_vector - right_vector).sum()))
    else:
        raise ValueError(f"unsupported descriptor comparison: {spec.comparison}")
    return float(np.clip(score, 0.0, 1.0))


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("normalized_mask must be two-dimensional")
    return (array > 0).astype(np.uint8)


def _finite_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    result = np.nan_to_num(np.asarray(vector, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape != (dimension,):
        raise ValueError(f"descriptor vector must have shape ({dimension},), got {result.shape}")
    return result


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        height, width = mask.shape
        return (float(width - 1) / 2.0, float(height - 1) / 2.0)
    return float(xs.mean()), float(ys.mean())


def _radial_distances(shape: tuple[int, int], centroid: tuple[float, float]) -> np.ndarray:
    ys, xs = np.indices(shape, dtype=np.float64)
    return np.hypot(xs - centroid[0], ys - centroid[1])


def _histogram(values: np.ndarray, bins: int, maximum: float | None = None) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float64)
    upper = maximum if maximum is not None else float(np.max(values))
    if upper <= 1e-12:
        result = np.zeros(bins, dtype=np.float64)
        result[0] = 1.0
        return result
    result, _ = np.histogram(values, bins=bins, range=(0.0, upper))
    result = result.astype(np.float64)
    return result / result.sum()


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return count - 1, stats[1:], centers[1:]


def _boundary(mask: np.ndarray) -> np.ndarray:
    return (mask > cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))).astype(np.uint8)


def _global_morphology(mask: np.ndarray, stats: np.ndarray, area_fraction: float) -> np.ndarray:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = float(mask.sum())
    hull_area = sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours)
    compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
    solidity = area / max(hull_area, 1.0)
    covariance = _coordinate_covariance(mask)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-12))))
    component_count = len(stats)
    component_areas = stats[:, cv2.CC_STAT_AREA] if component_count else np.empty(0)
    largest_share = float(component_areas.max() / area) if area else 0.0
    euler = _euler_number(hierarchy)
    normalized_perimeter = perimeter / max(2.0 * (mask.shape[0] + mask.shape[1]), 1.0)
    return np.array((
        area_fraction, normalized_perimeter, compactness, solidity, eccentricity,
        eigenvalues[0] / max(eigenvalues[1], 1e-12), np.log1p(component_count) / 8.0,
        largest_share, np.tanh(euler / 8.0),
        float(component_areas.std() / max(area, 1.0)) if component_count else 0.0,
    ))


def _coordinate_covariance(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return np.eye(2, dtype=np.float64)
    return np.cov(np.stack((xs, ys)).astype(np.float64))


def _euler_number(hierarchy: np.ndarray | None) -> int:
    if hierarchy is None:
        return 0
    parents = hierarchy[0, :, 3]
    return int(np.count_nonzero(parents < 0) - np.count_nonzero(parents >= 0))


def _hu_absolute(mask: np.ndarray) -> np.ndarray:
    hu = cv2.HuMoments(cv2.moments(mask)).ravel()
    return -np.log10(np.abs(hu) + 1e-30)


def _distance_histogram(mask: np.ndarray, bins: int) -> np.ndarray:
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return _histogram(distances[mask > 0], bins)


def _component_area_histogram(stats: np.ndarray, foreground: int, bins: int) -> np.ndarray:
    if not len(stats) or foreground == 0:
        return np.zeros(bins, dtype=np.float64)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64) / foreground
    return _histogram(areas, bins, maximum=1.0)


def _component_radial_histogram(
    centers: np.ndarray,
    centroid: tuple[float, float],
    shape: tuple[int, int],
    bins: int,
) -> np.ndarray:
    if not len(centers):
        return np.zeros(bins, dtype=np.float64)
    distances = np.hypot(centers[:, 0] - centroid[0], centers[:, 1] - centroid[1])
    return _histogram(distances, bins, maximum=float(np.hypot(*shape)))


def _component_distance_histogram(
    centers: np.ndarray, shape: tuple[int, int], bins: int
) -> np.ndarray:
    if len(centers) < 2:
        return np.zeros(bins, dtype=np.float64)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    values = distances[np.triu_indices(len(centers), 1)]
    return _histogram(values, bins, maximum=float(np.hypot(*shape)))


def _fourier_radial_power(mask: np.ndarray, bins: int) -> np.ndarray:
    power = np.abs(np.fft.fftshift(np.fft.fft2(mask.astype(np.float64)))) ** 2
    ys, xs = np.indices(mask.shape, dtype=np.float64)
    radii = np.hypot(xs - (mask.shape[1] - 1) / 2.0, ys - (mask.shape[0] - 1) / 2.0)
    edges = np.linspace(0.0, float(radii.max()), bins + 1)
    result = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        selector = (radii >= edges[index]) & (radii < edges[index + 1])
        result[index] = float(power[selector].mean()) if np.any(selector) else 0.0
    return result / max(float(result.sum()), 1e-12)


def _curvature_histogram(mask: np.ndarray, bins: int) -> np.ndarray:
    """Histogram unsigned contour turns, discarding pose and traversal direction."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    turns = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) < 3:
            continue
        incoming = points - np.roll(points, 1, axis=0)
        outgoing = np.roll(points, -1, axis=0) - points
        denominator = np.linalg.norm(incoming, axis=1) * np.linalg.norm(
            outgoing, axis=1
        )
        valid = denominator > 0.0
        cosine = np.zeros(len(points), dtype=np.float64)
        cosine[valid] = np.sum(incoming[valid] * outgoing[valid], axis=1) / (
            denominator[valid]
        )
        turns.extend(np.arccos(np.clip(cosine[valid], -1.0, 1.0)) / np.pi)
    return _histogram(np.asarray(turns), bins, maximum=1.0)


def _autocorrelation_radial(mask: np.ndarray, bins: int) -> np.ndarray:
    """Radially summarize phase-free mask autocorrelation."""
    spectrum = np.fft.fft2(mask.astype(np.float64))
    autocorrelation = np.fft.fftshift(
        np.fft.ifft2(np.abs(spectrum) ** 2).real
    )
    autocorrelation = np.maximum(autocorrelation, 0.0)
    ys, xs = np.indices(mask.shape, dtype=np.float64)
    radii = np.hypot(
        xs - (mask.shape[1] - 1) / 2.0,
        ys - (mask.shape[0] - 1) / 2.0,
    )
    edges = np.linspace(0.0, float(radii.max()), bins + 1)
    result = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        selector = (radii >= edges[index]) & (radii < edges[index + 1])
        result[index] = (
            float(autocorrelation[selector].mean()) if np.any(selector) else 0.0
        )
    return result / max(float(result.sum()), 1e-12)


def _build_global_morphology(mask: np.ndarray, dimension: int) -> np.ndarray:
    del dimension
    _, stats, _ = _components(mask)
    foreground = int(mask.sum())
    area_fraction = foreground / float(mask.shape[0] * mask.shape[1])
    return _global_morphology(mask, stats, area_fraction)


def _build_hu_absolute(mask: np.ndarray, dimension: int) -> np.ndarray:
    del dimension
    return _hu_absolute(mask)


def _build_radial_foreground(mask: np.ndarray, dimension: int) -> np.ndarray:
    radial = _radial_distances(mask.shape, _centroid(mask))
    return _histogram(radial[mask > 0], dimension)


def _build_boundary_radius(mask: np.ndarray, dimension: int) -> np.ndarray:
    radial = _radial_distances(mask.shape, _centroid(mask))
    return _histogram(radial[_boundary(mask) > 0], dimension)


def _build_distance_transform(mask: np.ndarray, dimension: int) -> np.ndarray:
    return _distance_histogram(mask, dimension)


def _build_component_area(mask: np.ndarray, dimension: int) -> np.ndarray:
    _, stats, _ = _components(mask)
    return _component_area_histogram(stats, int(mask.sum()), dimension)


def _build_component_radial(mask: np.ndarray, dimension: int) -> np.ndarray:
    _, _, centers = _components(mask)
    return _component_radial_histogram(
        centers, _centroid(mask), mask.shape, dimension
    )


def _build_component_distance(mask: np.ndarray, dimension: int) -> np.ndarray:
    _, _, centers = _components(mask)
    return _component_distance_histogram(centers, mask.shape, dimension)


def _build_fourier_power(mask: np.ndarray, dimension: int) -> np.ndarray:
    return _fourier_radial_power(mask, dimension)


def _build_curvature(mask: np.ndarray, dimension: int) -> np.ndarray:
    return _curvature_histogram(mask, dimension)


def _build_autocorrelation(mask: np.ndarray, dimension: int) -> np.ndarray:
    return _autocorrelation_radial(mask, dimension)


def _definition(
    name: str,
    dimension: int,
    comparison: str,
    prior_evidence: str,
    builder: Callable[[np.ndarray, int], np.ndarray],
    *,
    input_size: int | None = None,
) -> _DescriptorDefinition:
    selected_builder = builder
    if input_size is not None:
        def resized_builder(mask: np.ndarray, output_dimension: int) -> np.ndarray:
            if (
                mask.shape[0] % input_size == 0
                and mask.shape[1] % input_size == 0
            ):
                row_factor = mask.shape[0] // input_size
                column_factor = mask.shape[1] // input_size
                resized = mask.reshape(
                    input_size, row_factor, input_size, column_factor
                ).max(axis=(1, 3))
            else:
                resized = cv2.resize(
                    mask,
                    (input_size, input_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            return builder(resized, output_dimension)

        selected_builder = resized_builder
    return _DescriptorDefinition(
        DescriptorSpec(name, "1", dimension, comparison, prior_evidence),
        selected_builder,
    )


def _distribution_definitions(
    stem: str,
    canonical_name: str,
    dimensions: tuple[int, ...],
    canonical_dimension: int,
    builder: Callable[[np.ndarray, int], np.ndarray],
    prior_evidence: str,
    *,
    comparison: str = "histogram_intersection",
) -> tuple[_DescriptorDefinition, ...]:
    definitions = tuple(
        _definition(
            canonical_name if dimension == canonical_dimension else f"{stem}_{dimension}_v1",
            dimension,
            comparison,
            prior_evidence,
            builder,
        )
        for dimension in dimensions
    )
    return definitions + (
        _definition(
            f"{stem}_{canonical_dimension}_128_v1",
            canonical_dimension,
            comparison,
            f"128x128 resolution ablation. {prior_evidence}",
            builder,
            input_size=128,
        ),
    )


_COMPONENT_EVIDENCE = (
    "Component descriptors are fragile under merge/split; retrieval value unproven."
)


_DESCRIPTORS = (
    _definition(
        "global_morphology_v1",
        10,
        "exp_l1",
        "New retrieval-only summary; no prior scoring result.",
        _build_global_morphology,
    ),
    _definition(
        "global_morphology_128_v1",
        10,
        "exp_l1",
        "128x128 resolution ablation; no prior scoring result.",
        _build_global_morphology,
        input_size=128,
    ),
    _definition(
        "hu_absolute_moments_v1",
        7,
        "exp_l1",
        "Hu descriptor path was weak for cross-modal scoring; retrieval value unproven.",
        _build_hu_absolute,
    ),
    *_distribution_definitions(
        "radial_foreground_histogram",
        "radial_foreground_histogram_v1",
        (8, 16, 32),
        16,
        _build_radial_foreground,
        "Extends the prior polar histogram; retrieval value unproven.",
    ),
    *_distribution_definitions(
        "boundary_radius_histogram",
        "boundary_radius_histogram_v1",
        (8, 16, 32),
        16,
        _build_boundary_radius,
        "New retrieval-only boundary distribution; no prior scoring result.",
    ),
    *_distribution_definitions(
        "distance_transform_histogram",
        "distance_transform_histogram_v1",
        (8, 16, 32),
        16,
        _build_distance_transform,
        "New retrieval-only interior-thickness distribution; no prior scoring result.",
    ),
    *_distribution_definitions(
        "component_area_histogram",
        "component_area_histogram_v1",
        (8, 12, 16),
        12,
        _build_component_area,
        _COMPONENT_EVIDENCE,
    ),
    *_distribution_definitions(
        "component_radial_histogram",
        "component_radial_histogram_v1",
        (8, 12, 16),
        12,
        _build_component_radial,
        _COMPONENT_EVIDENCE,
    ),
    *_distribution_definitions(
        "component_distance_histogram",
        "component_distance_histogram_v1",
        (8, 12, 16),
        12,
        _build_component_distance,
        "Distance-signature-like component summary; prior scoring signal was unproven.",
    ),
    *_distribution_definitions(
        "curvature_histogram",
        "curvature_histogram_16_v1",
        (8, 16, 32),
        16,
        _build_curvature,
        "Unsigned curvature is new retrieval-only evidence; no prior scoring result.",
    ),
    *_distribution_definitions(
        "fourier_radial_power",
        "fourier_radial_power_v1",
        (8, 16, 32),
        16,
        _build_fourier_power,
        "Phase-free spectral power is new retrieval evidence; no prior scoring result.",
        comparison="exp_l1",
    ),
    *_distribution_definitions(
        "autocorrelation_radial",
        "autocorrelation_radial_16_v1",
        (8, 16, 32),
        16,
        _build_autocorrelation,
        "Phase-free autocorrelation is new retrieval evidence; no prior scoring result.",
        comparison="exp_l1",
    ),
)
