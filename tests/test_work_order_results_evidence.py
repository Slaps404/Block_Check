"""Score-time claim artifact JPEG evidence (issue #236 seam 1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from kiosk.images import CLAIM_DISPLAY_MAX_LONG_EDGE, CLAIM_THUMB_MAX_LONG_EDGE
from session.pipeline import ClaimDecision
from session.preparation import PreparationFailure
from session.workflow import ProcessingStore

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
from tests.test_work_order_slide_overlay import PoseAwareStubWorkOrderScorer


def _claim_evidence_paths(artifact_dir: Path, capture_id: str) -> dict[str, Path]:
    return {
        "block_thumb": artifact_dir / f"{capture_id}_block_thumb.jpg",
        "slide_thumb": artifact_dir / f"{capture_id}_slide_thumb.jpg",
        "block_display": artifact_dir / f"{capture_id}_block_display.jpg",
        "slide_display": artifact_dir / f"{capture_id}_slide_display.jpg",
        "overlay_display": artifact_dir / f"{capture_id}_overlay_display.jpg",
        "overlay_png": artifact_dir / f"{capture_id}_slide_overlay.png",
    }


def _assert_claim_evidence_assets(artifact_dir: Path, capture_id: str) -> None:
    paths = _claim_evidence_paths(artifact_dir, capture_id)
    for key, max_edge in (
        ("block_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("slide_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("block_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
        ("slide_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
        ("overlay_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
    ):
        path = paths[key]
        assert path.is_file(), f"missing {key}: {path}"
        assert path.stat().st_size > 0, f"empty {key}: {path}"
        image = cv2.imread(str(path))
        assert image is not None, f"unreadable {key}: {path}"
        long_edge = max(image.shape[0], image.shape[1])
        assert long_edge <= max_edge, (
            f"{key} long edge {long_edge} exceeds {max_edge}"
        )
    overlay_png = paths["overlay_png"]
    assert overlay_png.is_file(), f"missing overlay PNG: {overlay_png}"
    assert overlay_png.stat().st_size > 0


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_review_work_order_writes_claim_evidence_jpegs(tmp_path):
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

    artifact_dir = session.directory / "claim_artifacts"
    _assert_claim_evidence_assets(artifact_dir, slide_id)


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_pass_work_order_writes_claim_evidence_jpegs(tmp_path):
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

    artifact_dir = session.directory / "claim_artifacts"
    _assert_claim_evidence_assets(artifact_dir, slide_id)


def test_missing_pose_still_writes_block_slide_evidence_without_overlay(tmp_path):
    """#236: block/slide JPEGs must not depend on pose or PreparedSpecimen."""
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    capture_id = "slide_missing_pose_test"

    block_img = np.full((80, 120, 3), 90, dtype=np.uint8)
    slide_img = np.full((100, 140, 3), 110, dtype=np.uint8)
    decision = ClaimDecision(
        claim_id="51151378",
        block_path="",
        slide_path="",
        verdict="REVIEW",
        stage="scoring",
        reason="no pose",
        score=0.5,
        best_angle=None,
        best_flip=None,
    )
    block_result = PreparationFailure(role="block", reason="stub")
    slide_result = PreparationFailure(role="slide", reason="stub")

    store._write_claim_slide_overlay(
        session,
        capture_id,
        block_img=block_img,
        slide_img=slide_img,
        block_result=block_result,
        slide_result=slide_result,
        decision=decision,
    )

    artifact_dir = session.directory / "claim_artifacts"
    paths = _claim_evidence_paths(artifact_dir, capture_id)
    for key, max_edge in (
        ("block_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("slide_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("block_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
        ("slide_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
    ):
        path = paths[key]
        assert path.is_file(), f"missing {key}: {path}"
        assert path.stat().st_size > 0, f"empty {key}: {path}"
        image = cv2.imread(str(path))
        assert image is not None, f"unreadable {key}: {path}"
        long_edge = max(image.shape[0], image.shape[1])
        assert long_edge <= max_edge, (
            f"{key} long edge {long_edge} exceeds {max_edge}"
        )

    assert not paths["overlay_png"].is_file(), (
        f"overlay PNG must not be written without pose: {paths['overlay_png']}"
    )
    assert not paths["overlay_display"].is_file(), (
        "overlay display JPEG must not be written without pose: "
        f"{paths['overlay_display']}"
    )
