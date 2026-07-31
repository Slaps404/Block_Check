"""Decode-once regression tests for the slide-claim path (#185 PR2).

``resolve_claim`` used to decode the same slide raster twice: once inside
``prepare_specimen`` (path-in) for Specimen Preprocessing, and again inside
``_finalize_claim`` purely to feed the claim QC contact sheet. The
decode-once refactor (ADR 0014) makes ``resolve_claim`` own a single
``cv2.imread`` and thread that frame through both consumers.

These tests spy on ``cv2.imread`` (patched on the shared ``cv2`` module
object, so every caller -- ``workflow.py``, ``preparation.py`` -- observes
the same patched attribute) and count invocations keyed on the exact slide
path, so a stray decode of the block photo or the stored block mask can't be
conflated with a slide re-decode.
"""
from __future__ import annotations

import cv2
import pytest

from session.workflow import ProcessingStore

# #185 PR2: reuse test_session_workflow's real-workflow fixtures/helpers
# rather than re-deriving a second harness. Imported at module level (not
# locally inside the test) so pytest can see ``lightweight_qc_artifacts`` as
# a requestable fixture -- autouse fixtures don't cross module boundaries,
# so the test here must ask for it explicitly.
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

# Body layout mirrors contact_sheet.py: a 60px header, then five 256x256
# panels hstacked in order (block, slide, block-mask, slide-mask, overlay).
_HEADER_H = 60
_PANEL_H = 256
_PANEL_W = 256


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_resolve_claim_decodes_slide_raster_exactly_once(tmp_path, monkeypatch):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)

    calls: list[str] = []
    real_imread = cv2.imread

    def spy_imread(path, *args, **kwargs):
        calls.append(str(path))
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", spy_imread)

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1", slide_path,
    )
    assert outcome.accepted

    slide_decodes = [call for call in calls if call == str(slide_path)]
    assert len(slide_decodes) == 1, (
        "expected exactly one main-side cv2.imread of the slide raster across "
        f"resolve_claim + _finalize_claim, got {len(slide_decodes)}: {calls}"
    )


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_resolve_claim_unreadable_slide_fails_closed_to_review(tmp_path):
    """A slide raster that fails to decode must reproduce the exact
    ``PreparationFailure(role="slide", reason="could not read image: ...")``
    the old path-in ``prepare_specimen`` produced, and the same fail-closed
    REVIEW verdict, now that the guard lives in ``resolve_claim`` instead."""
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    corrupt_slide = tmp_path / "corrupt_slide.png"
    corrupt_slide.write_bytes(b"")  # zero-byte file: cv2.imread returns None

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1", corrupt_slide,
    )

    assert outcome.accepted
    assert outcome.verdict == "REVIEW"
    assert outcome.stage == "preparation"
    assert outcome.reason == (
        f"slide preparation failed: could not read image: {corrupt_slide}"
    )


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_score_work_order_pass_verdict_claim_qc_keeps_real_slide_pixels(tmp_path):
    """#185 regression check: ``_finalize_claim`` writes ``claim_qc.png`` for
    EVERY work-order verdict, not just REVIEW. A prior draft of the batch
    decode-once fix gated the loop-2 re-decode on ``verdict == "REVIEW"``,
    which silently swapped the real slide pixels for the None-placeholder
    panel on every PASS claim. Guard against that regressing again: build a
    clear-win PASS verdict through the real (uninjected) contact sheet
    renderer and assert the slide panel's pixels are the real fill value
    (120), not the placeholder's (60)."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path)
    _drain_to_slides(store, session)
    slide_capture_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_capture_id] = {block_id: 0.95}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"

    qc_path = session.directory / "claim_artifacts" / f"{slide_capture_id}_claim_qc.png"
    assert qc_path.is_file()
    sheet = cv2.imread(str(qc_path))
    assert sheet is not None

    slide_panel = sheet[_HEADER_H:_HEADER_H + _PANEL_H, _PANEL_W:2 * _PANEL_W]
    mean_value = float(slide_panel.mean())
    assert mean_value > 100, (
        "expected real (fill-value 120) slide pixels in claim_qc.png for a "
        f"PASS work-order verdict, got mean={mean_value:.1f} -- this is what "
        "the None-placeholder panel (fill-value 60) would produce"
    )
