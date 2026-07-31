"""Tests for the shared pair-composition seam."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import inspect  # noqa: E402

from verify.pair_composition import compose_pair, compose_prepared_pair  # noqa: E402
from session.preparation import PreparationFailure, PreparedSpecimen  # noqa: E402
from verify.scorer import score_pair_result_routed  # noqa: E402


def _specimen(mask: np.ndarray, role: str) -> PreparedSpecimen:
    return PreparedSpecimen(role=role, mask=mask, roi_ok=True, roi_reason="ok")


def test_compose_prepared_pair_scores_preparable_pairs_even_when_gate_fails():
    """Diagnostics need gated-but-preparable scores for calibration evidence."""
    block_mask = np.zeros((1000, 1000), dtype=np.uint8)
    slide_mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(block_mask, (100, 100), (110, 110), 255, -1)
    cv2.rectangle(slide_mask, (100, 100), (110, 110), 255, -1)

    pair = compose_prepared_pair(
        _specimen(block_mask, "block"),
        _specimen(slide_mask, "slide"),
        score_gated_pairs=True,
    )

    assert not pair.gate.passed
    assert pair.gate.stage == "mask_quality"
    assert pair.score_result is not None
    assert pair.score is not None


def test_compose_prepared_pair_skips_gated_scores_by_default():
    """Production keeps a fail-closed gate-before-score boundary."""
    block_mask = np.zeros((1000, 1000), dtype=np.uint8)
    slide_mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(block_mask, (100, 100), (110, 110), 255, -1)
    cv2.rectangle(slide_mask, (100, 100), (110, 110), 255, -1)

    pair = compose_prepared_pair(
        _specimen(block_mask, "block"),
        _specimen(slide_mask, "slide"),
    )

    assert not pair.gate.passed
    assert pair.score_result is None
    assert pair.score is None


def test_compose_prepared_pair_does_not_score_preparation_failures():
    pair = compose_prepared_pair(
        PreparationFailure(role="block", reason="could not read image"),
        _specimen(np.ones((100, 100), dtype=np.uint8), "slide"),
    )

    assert not pair.gate.passed
    assert pair.score_result is None
    assert pair.score is None


# ---------------------------------------------------------------------------
# Issue #72: the routed scorer is the compose_pair / compose_prepared_pair
# default ScoreFn.
# ---------------------------------------------------------------------------

def test_compose_pair_default_scorer_is_the_shape_router():
    default_scorer = inspect.signature(compose_pair).parameters["scorer"].default
    assert default_scorer is score_pair_result_routed


def test_compose_prepared_pair_default_scorer_is_the_shape_router():
    default_scorer = inspect.signature(compose_prepared_pair).parameters["scorer"].default
    assert default_scorer is score_pair_result_routed
