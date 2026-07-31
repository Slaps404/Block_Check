"""Tests for pre-scoring quality gates (issue #18)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from verify.gates import GateResult, check_block_quality, run_quality_gates
from session.preparation import PreparedSpecimen, PreparationFailure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_mask(h: int = 200, w: int = 200, coverage: float = 0.15) -> np.ndarray:
    """Binary mask with a central filled circle giving ~coverage fraction."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(((coverage * h * w) / np.pi) ** 0.5)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return mask


def _good_block(mask: np.ndarray | None = None) -> PreparedSpecimen:
    m = mask if mask is not None else _good_mask()
    return PreparedSpecimen(role="block", mask=m, roi_ok=True, roi_reason="ok")


def _good_slide(mask: np.ndarray | None = None) -> PreparedSpecimen:
    m = mask if mask is not None else _good_mask()
    return PreparedSpecimen(role="slide", mask=m, roi_ok=True, roi_reason="ok")


def _bad_roi_block() -> PreparedSpecimen:
    return PreparedSpecimen(
        role="block",
        mask=_good_mask(),
        roi_ok=False,
        roi_reason="ROI aspect ratio degenerate: 15.0",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_gates_pass_when_both_specimens_valid():
    result = run_quality_gates(_good_block(), _good_slide())
    assert isinstance(result, GateResult)
    assert result.passed is True


def test_passing_gate_has_stage_and_reason():
    result = run_quality_gates(_good_block(), _good_slide())
    assert result.stage
    assert result.reason


# ---------------------------------------------------------------------------
# Preparation failure gates
# ---------------------------------------------------------------------------

def test_block_preparation_failure_routes_to_review():
    fail = PreparationFailure(role="block", reason="could not read image")
    result = run_quality_gates(fail, _good_slide())
    assert result.passed is False
    assert "block" in result.reason.lower() or "preparation" in result.stage.lower()


def test_slide_preparation_failure_routes_to_review():
    fail = PreparationFailure(role="slide", reason="corrupt JPEG")
    result = run_quality_gates(_good_block(), fail)
    assert result.passed is False
    assert "slide" in result.reason.lower() or "preparation" in result.stage.lower()


def test_preparation_gate_has_stage_field():
    fail = PreparationFailure(role="block", reason="unreadable")
    result = run_quality_gates(fail, _good_slide())
    assert result.stage


# ---------------------------------------------------------------------------
# Mask quality gates
# ---------------------------------------------------------------------------

def test_empty_block_mask_routes_to_review():
    empty_block = PreparedSpecimen(
        role="block", mask=np.zeros((200, 200), dtype=np.uint8), roi_ok=True, roi_reason="ok"
    )
    result = run_quality_gates(empty_block, _good_slide())
    assert result.passed is False
    assert "mask" in result.stage.lower() or "mask" in result.reason.lower()


def test_empty_slide_mask_routes_to_review():
    empty_slide = PreparedSpecimen(
        role="slide", mask=np.zeros((200, 200), dtype=np.uint8), roi_ok=True, roi_reason="ok"
    )
    result = run_quality_gates(_good_block(), empty_slide)
    assert result.passed is False


def test_low_info_mask_routes_to_review():
    tiny_mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(tiny_mask, (250, 250), 2, 255, -1)   # just a few pixels
    sparse_block = PreparedSpecimen(
        role="block", mask=tiny_mask, roi_ok=True, roi_reason="ok"
    )
    result = run_quality_gates(sparse_block, _good_slide())
    assert result.passed is False


def test_small_real_mask_above_sparse_floor_can_be_scored():
    small_real_mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(small_real_mask, (490, 490), (503, 503), 255, -1)

    result = run_quality_gates(_good_block(small_real_mask), _good_slide())

    assert result.passed is True


def test_pair_of_sliver_masks_now_passes_gates():
    """The sliver-AND rule is retired (ADR 0011, issue #158): sliver-like is a
    shape proxy, not the correct indiscriminability signal. A both-sliver pair
    that otherwise clears mask-quality + ROI now passes the gate suite."""
    block_mask = np.zeros((500, 500), dtype=np.uint8)
    slide_mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(block_mask, (120, 240), (380, 241), 255, -1)
    cv2.rectangle(slide_mask, (130, 245), (390, 246), 255, -1)

    result = run_quality_gates(_good_block(block_mask), _good_slide(slide_mask))

    assert result.passed is True


# ---------------------------------------------------------------------------
# ROI gates
# ---------------------------------------------------------------------------

def test_bad_block_roi_routes_to_review():
    result = run_quality_gates(_bad_roi_block(), _good_slide())
    assert result.passed is False
    assert "roi" in result.stage.lower() or "roi" in result.reason.lower()


# ---------------------------------------------------------------------------
# Scorer is skipped when gate fails (tested via pipeline integration)
# ---------------------------------------------------------------------------

def test_gate_failure_reason_is_non_empty():
    fail = PreparationFailure(role="block", reason="unreadable")
    result = run_quality_gates(fail, _good_slide())
    assert len(result.reason) > 0


def test_gate_stage_is_non_empty_on_failure():
    fail = PreparationFailure(role="block", reason="unreadable")
    result = run_quality_gates(fail, _good_slide())
    assert len(result.stage) > 0


# ---------------------------------------------------------------------------
# check_block_quality: block-only usability, no slide required (#250)
# ---------------------------------------------------------------------------

def test_check_block_quality_passes_a_good_block():
    result = check_block_quality(_good_block())
    assert result.passed is True


def test_check_block_quality_fails_on_preparation_failure():
    fail = PreparationFailure(role="block", reason="could not read image")
    result = check_block_quality(fail)
    assert result.passed is False
    assert "preparation" in result.stage.lower()
    assert "could not read image" in result.reason


def test_check_block_quality_fails_on_empty_mask():
    empty_block = PreparedSpecimen(
        role="block", mask=np.zeros((200, 200), dtype=np.uint8), roi_ok=True, roi_reason="ok"
    )
    result = check_block_quality(empty_block)
    assert result.passed is False
    assert "mask" in result.stage.lower() or "mask" in result.reason.lower()


def test_check_block_quality_fails_on_bad_roi():
    result = check_block_quality(_bad_roi_block())
    assert result.passed is False
    assert "roi" in result.stage.lower() or "roi" in result.reason.lower()


def test_check_block_quality_does_not_require_a_slide():
    """Distinct from run_quality_gates: no slide argument exists at all."""
    import inspect
    signature = inspect.signature(check_block_quality)
    assert list(signature.parameters) == ["block_result"]
