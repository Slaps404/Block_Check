"""Behavior tests for the deployed two-branch production scorer."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from constants import PASS_THRESHOLD, SHAPE_ROUTER_SIZE_THRESHOLD
from session.preparation import PreparedSpecimen
from verify.scorer import build_locked_score_cache, decide, score_pair_result_routed


def _specimen(mask: np.ndarray, role: str = "block") -> PreparedSpecimen:
    return PreparedSpecimen(role, mask, True, "ok")


def _sparse() -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.circle(mask, (112, 512), 15, 255, -1)
    cv2.circle(mask, (912, 512), 15, 255, -1)
    return mask


def _dense() -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.circle(mask, (512, 512), 300, 255, -1)
    return mask


def test_sparse_pair_calculates_point_layout_and_traceability():
    block = _specimen(_sparse())
    slide = _specimen(_sparse(), "slide")
    result = score_pair_result_routed(block, slide)
    assert result.router_size_signal < SHAPE_ROUTER_SIZE_THRESHOLD
    assert result.selected_metric == "point_layout"
    assert result.point_layout is not None
    assert result.score == pytest.approx(result.point_layout)
    assert result.router_size_signal == pytest.approx(min(
        result.block_occupied_fraction, result.slide_occupied_fraction
    ))
    assert isinstance(result.best_angle, float)
    assert isinstance(result.best_flip, bool)
    assert 0 <= result.align_soft_iou <= 1
    assert 0 <= result.mask_iou <= 1


def test_dense_pair_calculates_only_mask_iou():
    result = score_pair_result_routed(
        _specimen(_dense()), _specimen(_dense(), "slide")
    )
    assert result.router_size_signal >= SHAPE_ROUTER_SIZE_THRESHOLD
    assert result.selected_metric == "mask_iou"
    assert result.point_layout is None
    assert result.score == pytest.approx(result.mask_iou)


def test_cache_uses_production_radial_grid():
    cache = build_locked_score_cache(_specimen(_dense()))
    assert cache.normalized_mask.shape == (256, 256)


def test_decide_has_only_pass_and_review():
    assert decide(PASS_THRESHOLD)[0] == "PASS"
    assert decide(PASS_THRESHOLD - 0.001)[0] == "REVIEW"
