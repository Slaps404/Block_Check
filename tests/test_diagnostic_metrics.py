"""Behavior tests for the diagnostic-only ablation metric battery."""

import cv2
import numpy as np
import pytest

from diagnostic_metrics import (
    METRIC_TAGS,
    SILHOUETTE_BASED,
    hu_moment_similarity,
    modified_hausdorff,
    polar_histogram_similarity,
    soft_dilated_iou,
    symmetric_chamfer_mean,
    symmetric_chamfer_p90,
)


METRICS = (
    soft_dilated_iou,
    symmetric_chamfer_mean,
    symmetric_chamfer_p90,
    modified_hausdorff,
    polar_histogram_similarity,
    hu_moment_similarity,
)


def _shape() -> np.ndarray:
    mask = np.zeros((96, 96), dtype=np.uint8)
    cv2.rectangle(mask, (18, 22), (55, 66), 1, -1)
    cv2.circle(mask, (66, 31), 12, 1, -1)
    return mask


@pytest.mark.parametrize("metric", METRICS)
def test_metric_is_bounded_and_maximal_for_identical_masks(metric):
    mask = _shape()
    assert metric(mask, mask) == pytest.approx(1.0)
    assert 0.0 <= metric(mask, np.zeros_like(mask)) <= 1.0


@pytest.mark.parametrize("metric", METRICS)
def test_metric_decreases_as_a_piece_is_moved_farther(metric):
    base = _shape()
    near = base.copy()
    near[:, 72:] = 0
    far = base.copy()
    far[:, 52:] = 0
    assert metric(base, near) >= metric(base, far)


@pytest.mark.parametrize(
    "metric", (polar_histogram_similarity, hu_moment_similarity)
)
def test_rotation_invariant_descriptors_ignore_quarter_turn(metric):
    mask = _shape()
    assert metric(mask, np.rot90(mask)) >= 0.99


def test_every_new_metric_exposes_a_static_harness_tag():
    names = {metric.__name__ for metric in METRICS}
    assert names <= set(METRIC_TAGS)
    assert {METRIC_TAGS[name] for name in names} == {SILHOUETTE_BASED}
