"""Work-order slide-image overlay wiring (PASS+REVIEW for QA)."""

from __future__ import annotations

import contact_sheet
import cv2
import numpy as np
import pytest

import session.processing_store as processing_store_module
import session.workflow as session_workflow_module
from session.pipeline import ClaimDecision
from session.preparation import PreparedSpecimen
from session.workflow import ProcessingStore, WorkOrderScoringResult
from verify.locked_alignment import ALIGN_SIZE, align_masks
from verify.slide_image_overlay import build_slide_image_overlay

from tests.test_session_workflow import (  # noqa: F401 -- fixture import
    STARTED_AT,
    FastPreprocessor,
    StubContactSheetRenderer,
    _capture,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
    _valid_slide_result,
    lightweight_qc_artifacts,
)


class PoseAwareStubWorkOrderScorer:
    """Returns scores plus a fixed pose for integration tests."""

    def __init__(self, *, best_angle: float = 42.0, best_flip: bool = True):
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}
        self.best_angle = best_angle
        self.best_flip = best_flip

    def __call__(self, block_results, slide_results):
        pair_decisions: dict[str, dict[str, ClaimDecision]] = {}
        scores: dict[str, dict[str, float | None]] = {}
        for capture_id in slide_results:
            row_scores = dict(self.scores_by_slide.get(capture_id, {}))
            scores[capture_id] = row_scores
            row_decisions: dict[str, ClaimDecision] = {}
            for block_id, block_score in row_scores.items():
                row_decisions[block_id] = ClaimDecision(
                    claim_id=block_id,
                    block_path="",
                    slide_path="",
                    verdict="REVIEW",
                    stage="scoring",
                    reason="stub",
                    score=block_score,
                    best_angle=self.best_angle,
                    best_flip=self.best_flip,
                    align_soft_iou=0.5,
                    mask_iou=0.4,
                )
            pair_decisions[capture_id] = row_decisions
        return WorkOrderScoringResult(scores=scores, pair_decisions=pair_decisions)


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_review_work_order_writes_claimed_slide_overlay_png(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        result=_valid_slide_result(block_a),
        duration_ms=10.0,
    )

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_a)
    assert row["verdict"] == "REVIEW"

    overlay_path = (
        session.directory / "claim_artifacts" / f"{slide_id}_slide_overlay.png"
    )
    assert overlay_path.is_file()
    overlay = cv2.imread(str(overlay_path))
    assert overlay is not None
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    # Native block-mask crop, not the 256 alignment canvas.
    assert overlay.shape[0] != ALIGN_SIZE or overlay.shape[1] != ALIGN_SIZE


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_review_overlay_uses_scored_pose_not_align_masks(tmp_path, monkeypatch):
    root = tmp_path / "processing"
    scorer = PoseAwareStubWorkOrderScorer(best_angle=33.0, best_flip=False)
    renderer = StubContactSheetRenderer()
    overlay_calls: list[tuple] = []
    real_build_overlay = build_slide_image_overlay

    def spy_build_overlay(*args, **kwargs):
        overlay_calls.append((args, kwargs))
        return real_build_overlay(*args, **kwargs)

    align_calls: list[tuple] = []
    real_align_masks = align_masks

    def spy_align_masks(*args, **kwargs):
        align_calls.append((args, kwargs))
        return real_align_masks(*args, **kwargs)

    monkeypatch.setattr(
        processing_store_module, "build_slide_image_overlay", spy_build_overlay
    )
    monkeypatch.setattr(contact_sheet, "align_masks", spy_align_masks)

    store = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
        contact_sheet_renderer=renderer,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_b = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        result=_valid_slide_result(block_b),
        duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_a: 0.9, block_b: 0.2}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    overlay_path = (
        session.directory / "claim_artifacts" / f"{slide_id}_slide_overlay.png"
    )
    assert overlay_path.is_file()
    assert overlay_calls, "overlay must be built from locked pose"
    _, _, _, _, best_angle, best_flip = overlay_calls[0][0]
    assert best_angle == pytest.approx(33.0)
    assert best_flip is False
    assert align_calls == []


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_pass_work_order_writes_claimed_slide_overlay_png(tmp_path):
    """QA temporary: overlays write on PASS as well as REVIEW."""
    root = tmp_path / "processing"
    scorer = PoseAwareStubWorkOrderScorer(best_angle=12.0, best_flip=True)
    store = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
        contact_sheet_renderer=StubContactSheetRenderer(),
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        result=_valid_slide_result(block_id),
        duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_id: 0.95}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"

    overlay_path = (
        session.directory / "claim_artifacts" / f"{slide_id}_slide_overlay.png"
    )
    assert overlay_path.is_file()
    overlay = cv2.imread(str(overlay_path))
    assert overlay is not None
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert overlay.dtype == np.uint8


def test_default_work_order_scorer_retains_pair_decisions_with_pose():
    block_results = {
        "51151378": PreparedSpecimen(
            role="block", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    }
    slide_results = {
        "slide_capture_1": PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    }

    result = session_workflow_module.default_work_order_scorer(
        block_results, slide_results
    )

    decision = result.pair_decisions["slide_capture_1"]["51151378"]
    assert decision.best_angle is not None
    assert decision.best_flip is not None
