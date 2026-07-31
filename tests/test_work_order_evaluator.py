"""Behavior tests for the pure work-order (open-retrieval) verdict evaluator."""

from __future__ import annotations

import ast
from pathlib import Path

from constants import MATCH_MARGIN, PASS_THRESHOLD
from verify.work_order_evaluator import evaluate_work_order, flagged_pairs


def test_match_margin_constant_default_and_pass_threshold_only_used_in_fallback():
    assert MATCH_MARGIN == 0.02
    # Single-block fallback should compare against PASS_THRESHOLD directly.
    below = PASS_THRESHOLD - 0.10
    result = evaluate_work_order({"B1": below}, claimed_block="B1")
    assert result.verdict == "REVIEW"

    above = min(PASS_THRESHOLD + 0.10, 1.0)
    result = evaluate_work_order({"B1": above}, claimed_block="B1")
    assert result.verdict == "PASS"


def test_result_shape_has_verdict_reason_margin_top_block_near_miss():
    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert hasattr(result, "verdict")
    assert hasattr(result, "reason")
    assert hasattr(result, "match_margin")
    assert hasattr(result, "top_block")
    assert hasattr(result, "near_miss_blocks")
    assert isinstance(result.verdict, str)
    assert isinstance(result.reason, str)
    assert result.top_block == "B1"
    assert result.near_miss_blocks == frozenset()


def test_clear_win_is_pass():
    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.verdict == "PASS"
    assert result.top_block == "B1"
    assert result.near_miss_blocks == frozenset()
    assert result.reason == "Claimed block clearly scored highest."


def test_ambiguous_near_miss_is_review():
    # B2 is within MATCH_MARGIN (0.02) of B1 -> near-miss set non-empty.
    scores = {"B1": 0.90, "B2": 0.89, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.verdict == "REVIEW"
    assert result.top_block == "B1"
    assert "B2" in result.near_miss_blocks
    assert result.reason == "Near miss with block B2. Review both."


def test_claim_disagreement_is_review():
    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B2")
    assert result.verdict == "REVIEW"
    assert result.top_block == "B1"
    assert result.reason == "B1 scored higher than claimed block B2."


def test_single_block_fallback_uses_pass_threshold_and_is_tagged_unverified():
    passing_score = min(PASS_THRESHOLD + 0.05, 1.0)
    result = evaluate_work_order({"B1": passing_score}, claimed_block="B1")
    assert result.verdict == "PASS"
    assert result.match_margin is None
    assert "unverified" in result.reason.lower()
    assert "threshold only" in result.reason.lower()

    failing_score = PASS_THRESHOLD - 0.05
    result_fail = evaluate_work_order({"B1": failing_score}, claimed_block="B1")
    assert result_fail.verdict == "REVIEW"
    assert result_fail.match_margin is None
    assert "unverified" in result_fail.reason.lower()


def test_claimed_block_not_in_order_is_review():
    scores = {"B1": 0.90, "B2": 0.40}
    result = evaluate_work_order(scores, claimed_block="B99")
    assert result.verdict == "REVIEW"
    assert result.reason == "Claimed block B99 not in this order."


def test_claimed_pair_gate_failed_none_score_is_review_fail_closed():
    scores = {"B1": None, "B2": 0.90, "B3": 0.40}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.verdict == "REVIEW"
    assert result.reason == "Preparation failed."


def test_exact_tie_margin_is_near_miss():
    # B1 - B2 == 0.0, which is strictly less than MATCH_MARGIN -> near-miss.
    scores = {"B1": 0.90, "B2": 0.90, "B3": 0.10}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.verdict == "REVIEW"
    assert "B2" in result.near_miss_blocks


def test_clear_win_match_margin_is_computed_from_scores():
    import pytest

    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.match_margin == pytest.approx(0.50)


def test_ambiguous_near_miss_match_margin_is_computed_from_scores():
    import pytest

    scores = {"B1": 0.90, "B2": 0.89, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B1")
    assert result.match_margin == pytest.approx(0.01)


def test_claim_disagreement_match_margin_is_relative_to_winner():
    import pytest

    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    result = evaluate_work_order(scores, claimed_block="B2")
    assert result.top_block == "B1"
    assert result.match_margin == pytest.approx(0.50)


def test_flagged_pairs_includes_top_match_and_appends_differing_claim():
    """#151: on claim disagreement, the operator's contact sheet must show
    BOTH the top match and the block that was actually claimed."""
    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    verdict = evaluate_work_order(scores, claimed_block="B2")
    assert verdict.top_block == "B1"

    pairs = flagged_pairs("B2", verdict)

    assert [pair["block_id"] for pair in pairs] == ["B1", "B2"]
    assert [pair["role"] for pair in pairs] == ["TOP MATCH", "CLAIMED"]


def test_flagged_pairs_dedupes_when_top_match_equals_claim():
    """When the claim IS the top match (clear win or ambiguous near-miss),
    only one pair is flagged -- no duplicate sheet for the same block."""
    scores = {"B1": 0.90, "B2": 0.40, "B3": 0.30}
    verdict = evaluate_work_order(scores, claimed_block="B1")
    assert verdict.top_block == "B1"

    pairs = flagged_pairs("B1", verdict)

    assert len(pairs) == 1
    assert pairs[0]["block_id"] == "B1"
    assert pairs[0]["role"] == "TOP MATCH"


def test_flagged_pairs_only_claim_when_top_block_is_none():
    """ADR 0009 boundary rule: when the claimed block isn't in this work
    order (or the claimed pair gate-failed), evaluate_work_order returns
    top_block=None. flagged_pairs must not fabricate a TOP MATCH pair for a
    block that was never scored -- only the claimed block is flagged."""
    scores = {"B1": 0.90, "B3": 0.30}
    verdict = evaluate_work_order(scores, claimed_block="B2")
    assert verdict.top_block is None

    pairs = flagged_pairs("B2", verdict)

    assert pairs == [{"block_id": "B2", "role": "CLAIMED"}]


def test_evaluator_module_has_no_cv2_numpy_io_imports():
    module_path = (
        Path(__file__).resolve().parents[1] / "code" / "verify" / "work_order_evaluator.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    banned = {"cv2", "numpy", "scipy", "os", "io", "socket", "threading"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            top = name.split(".")[0]
            assert top not in banned, f"banned import {name} in work_order_evaluator.py"
