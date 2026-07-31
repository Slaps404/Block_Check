"""Contract tests for cheap, pair-independent retrieval descriptors."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from verify.invariant_descriptors import (
    build_descriptor_values,
    compare_descriptor_values,
    descriptor_catalog,
)


def _asymmetric_mask() -> np.ndarray:
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (96, 132), (42, 22), 24, 0, 360, 1, -1)
    cv2.rectangle(mask, (139, 62), (168, 112), 1, -1)
    cv2.circle(mask, (61, 181), 13, 1, -1)
    return mask


def test_catalog_names_are_unique_and_values_are_finite_fixed_size():
    catalog = descriptor_catalog()
    values = build_descriptor_values(_asymmetric_mask())

    assert len({spec.name for spec in catalog}) == len(catalog)
    assert set(values) == {spec.name for spec in catalog}
    for spec in catalog:
        value = values[spec.name]
        assert value.vector.shape == (spec.dimension,)
        assert np.isfinite(value.vector).all()
        assert value.construction_ns >= 0
        assert spec.version
        assert spec.prior_evidence


def test_catalog_predeclares_bin_resolution_curvature_and_autocorrelation_ablation():
    dimensions = {spec.name: spec.dimension for spec in descriptor_catalog()}

    assert dimensions["radial_foreground_histogram_8_v1"] == 8
    assert dimensions["radial_foreground_histogram_v1"] == 16
    assert dimensions["radial_foreground_histogram_32_v1"] == 32
    assert dimensions["curvature_histogram_16_v1"] == 16
    assert dimensions["autocorrelation_radial_16_v1"] == 16
    assert dimensions["radial_foreground_histogram_16_128_v1"] == 16
    assert dimensions["curvature_histogram_16_128_v1"] == 16
    assert dimensions["autocorrelation_radial_16_128_v1"] == 16


def test_catalog_is_deterministic_and_pair_comparisons_are_bounded():
    catalog = descriptor_catalog()
    first = build_descriptor_values(_asymmetric_mask())
    second = build_descriptor_values(_asymmetric_mask())

    for spec in catalog:
        assert np.array_equal(first[spec.name].vector, second[spec.name].vector)
        assert compare_descriptor_values(
            spec, first[spec.name], second[spec.name]
        ) == pytest.approx(1.0)


@pytest.mark.parametrize("transform", (np.rot90, np.fliplr))
def test_every_descriptor_is_rotation_and_reflection_invariant(transform):
    original = build_descriptor_values(_asymmetric_mask())
    transformed = build_descriptor_values(transform(_asymmetric_mask()).copy())

    for spec in descriptor_catalog():
        similarity = compare_descriptor_values(spec, original[spec.name], transformed[spec.name])
        assert similarity >= 0.97, spec.name


def test_empty_mask_has_valid_finite_descriptors_and_self_similarity():
    values = build_descriptor_values(np.zeros((256, 256), dtype=np.uint8))
    for spec in descriptor_catalog():
        assert np.isfinite(values[spec.name].vector).all()
        assert compare_descriptor_values(
            spec, values[spec.name], values[spec.name]
        ) == pytest.approx(1.0)
