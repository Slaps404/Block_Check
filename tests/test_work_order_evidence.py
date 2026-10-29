"""Score-time claim evidence for work orders.

Covers the claim artifact JPEGs plus the slide-image overlay PNG (written on
PASS and REVIEW for QA), and that the overlay is built from the scored pose.
"""

from __future__ import annotations

from pathlib import Path

import contact_sheet
import cv2
import numpy as np
import pytest

import session.processing_store as processing_store_module
from kiosk.images import CLAIM_DISPLAY_MAX_LONG_EDGE, CLAIM_THUMB_MAX_LONG_EDGE
from session.pipeline import ClaimDecision
from session.preparation import PreparationFailure
from session.workflow import ProcessingStore
from verify.locked_alignment import ALIGN_SIZE, align_masks
from verify.slide_image_overlay import build_slide_image_overlay

from store_harness import (  # noqa: F401
    WORKFLOW_STARTED_AT as STARTED_AT,
    FastPreprocessor,
    PoseAwareStubWorkOrderScorer,
    StubContactSheetRenderer,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
)
from store_harness import (  # noqa: F401
    full_res_capture as _capture,
    lightweight_qc_artifacts,
    valid_slide_result as _valid_slide_result,
    lightweight_slide_qc,
)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts, lightweight_slide_qc):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into both shared harness QC fixtures."""


@pytest.fixture(autouse=True)
def _captures_above_display_size(monkeypatch):
    # Captures must exceed CLAIM_DISPLAY_MAX_LONG_EDGE so the size check bites.
    import constants

    monkeypatch.setitem(constants.CAPTURE_DIMENSIONS, "block", (1280, 960))


def _claim_evidence_paths(artifact_dir: Path, capture_id: str) -> dict[str, Path]:
    return {
        "block_thumb": artifact_dir / f"{capture_id}_block_thumb.jpg",
        "slide_thumb": artifact_dir / f"{capture_id}_slide_thumb.jpg",
        "block_display": artifact_dir / f"{capture_id}_block_display.jpg",
        "slide_display": artifact_dir / f"{capture_id}_slide_display.jpg",
        "overlay_display": artifact_dir / f"{capture_id}_overlay_display.jpg",
        "overlay_png": artifact_dir / f"{capture_id}_slide_overlay.png",
    }


def _assert_claim_evidence_assets(
    artifact_dir: Path, capture_id: str, *, with_overlay: bool = True
) -> None:
    paths = _claim_evidence_paths(artifact_dir, capture_id)
    sized = [
        ("block_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("slide_thumb", CLAIM_THUMB_MAX_LONG_EDGE),
        ("block_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
        ("slide_display", CLAIM_DISPLAY_MAX_LONG_EDGE),
    ]
    if with_overlay:
        sized.append(("overlay_display", CLAIM_DISPLAY_MAX_LONG_EDGE))
    for key, max_edge in sized:
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
    if not with_overlay:
        assert not overlay_png.is_file(), (
            f"overlay PNG must not be written without pose: {overlay_png}"
        )
        assert not paths["overlay_display"].is_file(), (
            "overlay display JPEG must not be written without pose: "
            f"{paths['overlay_display']}"
        )
        return
    assert overlay_png.is_file(), f"missing overlay PNG: {overlay_png}"
    assert overlay_png.stat().st_size > 0
    overlay = cv2.imread(str(overlay_png))
    assert overlay is not None
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    # Native block-mask crop, not the 256 alignment canvas.
    assert overlay.shape[0] != ALIGN_SIZE or overlay.shape[1] != ALIGN_SIZE


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_review_work_order_writes_claim_evidence(tmp_path):
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
def test_pass_work_order_writes_claim_evidence(tmp_path):
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

    artifact_dir = session.directory / "claim_artifacts"
    _assert_claim_evidence_assets(artifact_dir, slide_id)


def test_missing_pose_still_writes_block_slide_evidence_without_overlay(tmp_path):
    """Block/slide JPEGs must not depend on pose or PreparedSpecimen."""
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    capture_id = "slide_missing_pose_test"

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

    store._write_claim_slide_overlay(
        session,
        capture_id,
        block_img=np.full((80, 120, 3), 90, dtype=np.uint8),
        slide_img=np.full((100, 140, 3), 110, dtype=np.uint8),
        block_result=PreparationFailure(role="block", reason="stub"),
        slide_result=PreparationFailure(role="slide", reason="stub"),
        decision=decision,
    )

    _assert_claim_evidence_assets(
        session.directory / "claim_artifacts", capture_id, with_overlay=False
    )


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
