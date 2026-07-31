"""Behavior tests for the diagnostic-only locked shared alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

CODE = Path(__file__).resolve().parent.parent / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from verify.locked_alignment import (  # noqa: E402
    align_masks,
    radial_normalize_mask,
    transform_mask,
)
from diagnostic_metrics import (  # noqa: E402
    build_specimen_metric_cache,
    score_locked_metrics,
)
from session.preparation import PreparedSpecimen  # noqa: E402
from verify.scorer import score_pair_result_routed  # noqa: E402


def _asymmetric_mask() -> np.ndarray:
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.rectangle(mask, (42, 42), (58, 122), 1, -1)
    cv2.rectangle(mask, (42, 104), (112, 122), 1, -1)
    cv2.circle(mask, (95, 58), 10, 1, -1)
    return mask


def test_aligner_recovers_rotated_flipped_mask_at_locked_pose():
    block = _asymmetric_mask()
    slide = transform_mask(block, angle=34.0, flip=True)

    alignment = align_masks(block, slide)

    assert alignment.best_flip is True
    assert min(abs(alignment.best_angle - 34.0), abs(alignment.best_angle - 326.0)) <= 2.0
    assert alignment.align_soft_iou > 0.90
    assert alignment.mask_iou > 0.85


def test_aligner_soft_iou_is_low_for_shape_mismatch():
    block = _asymmetric_mask()
    mismatch = np.zeros_like(block)
    cv2.circle(mismatch, (80, 80), 42, 1, -1)

    alignment = align_masks(block, mismatch)

    assert alignment.align_soft_iou < 0.65


def test_production_selected_metric_matches_diagnostic_locked_recipe():
    block_mask = _asymmetric_mask()
    slide_mask = transform_mask(block_mask, angle=34.0, flip=True)
    block = PreparedSpecimen("block", block_mask, True, "ok")
    slide = PreparedSpecimen("slide", slide_mask, True, "ok")

    production = score_pair_result_routed(block, slide)
    block_cache = build_specimen_metric_cache(block_mask)
    slide_cache = build_specimen_metric_cache(slide_mask)
    alignment = align_masks(block_mask, slide_mask)
    diagnostic = score_locked_metrics(
        block_cache,
        slide_cache,
        alignment.aligned_slide_mask,
    )

    assert production.score == pytest.approx(diagnostic[production.selected_metric])


def test_production_normalization_defaults_to_area_weighted_rms_radius():
    mask = np.zeros((640, 640), dtype=np.uint8)
    cv2.ellipse(mask, (235, 340), (78, 52), 20, 0, 360, 1, -1)
    cv2.ellipse(mask, (405, 325), (72, 50), -15, 0, 360, 1, -1)
    cv2.circle(mask, (600, 70), 12, 1, -1)

    default = radial_normalize_mask(mask)
    rms = radial_normalize_mask(mask, mode="rms")
    maximum = radial_normalize_mask(mask, mode="max")

    assert np.array_equal(default, rms)
    assert not np.array_equal(default, maximum)
