"""Slide preparation artifacts mirror the existing block artifact evidence."""

from pathlib import Path

import cv2
import numpy as np

from session.workflow import ProcessingStore
from session.preparation import PreparationFailure
from tests.test_session_workflow import (  # noqa: F401 -- fixture import
    STARTED_AT,
    FastPreprocessor,
    StubWorkOrderScorer,
    _capture,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
    _valid_slide_result,
    lightweight_qc_artifacts,
)


def _assert_slide_artifacts(session_dir: Path, capture_id: str) -> None:
    artifact_dir = session_dir / "slide_artifacts"
    mask_path = artifact_dir / f"{capture_id}_mask.png"
    qc_path = artifact_dir / f"{capture_id}_qc.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert qc_path.is_file()


def _assert_failed_slide_artifact(session_dir: Path, capture_id: str) -> None:
    artifact_dir = session_dir / "slide_artifacts"
    mask = cv2.imread(
        str(artifact_dir / f"{capture_id}_mask.png"), cv2.IMREAD_UNCHANGED
    )
    assert mask is not None
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert not np.any(mask)
    assert (artifact_dir / f"{capture_id}_failed_qc.png").is_file()


def test_closed_set_claim_writes_slide_mask_and_qc(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    capture_id = "slide_capture_1"

    store.resolve_claim(
        session.number,
        block_id,
        capture_id,
        _capture(tmp_path / "slide.png", 120),
    )

    _assert_slide_artifacts(session.directory, capture_id)


def test_open_retrieval_writes_slide_mask_and_qc(tmp_path):
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path)
    _drain_to_slides(store, session)
    capture_id = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        result=_valid_slide_result(block_id),
        duration_ms=10.0,
    )
    scorer.scores_by_slide[capture_id] = {block_id: 0.95}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    _assert_slide_artifacts(session.directory, capture_id)


def test_preparation_failure_writes_slide_failure_artifact(tmp_path):
    def failing_slide_preprocessor(_image):
        return PreparationFailure(role="slide", reason="no tissue found")

    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=failing_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    capture_id = "slide_capture_1"

    store.resolve_claim(
        session.number,
        block_id,
        capture_id,
        _capture(tmp_path / "slide.png", 120),
    )

    _assert_failed_slide_artifact(session.directory, capture_id)


def test_unknown_block_writes_slide_mask_and_qc_for_qa(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    capture_id = "slide_capture_1"

    outcome = store.resolve_claim(
        session.number,
        "99999999",
        capture_id,
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.verdict == "REVIEW"
    assert outcome.stage == "identity_lookup"
    _assert_slide_artifacts(session.directory, capture_id)
