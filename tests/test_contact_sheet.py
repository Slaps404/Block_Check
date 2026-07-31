"""Tests for visual review contact sheet generation (issue #20).

Contact sheet tests verify file creation and basic dimensions/existence.
Pixel-level correctness is verified by manual inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import contact_sheet
from contact_sheet import _overlay_panel, id_line_text, write_contact_sheet
from session.preparation import PreparedSpecimen, PreparationFailure
from session.pipeline import ClaimDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dark_blob(h: int = 300, w: int = 400) -> np.ndarray:
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 4, (30, 30, 30), -1)
    return img


def _circle_mask(h: int = 300, w: int = 400) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)
    return mask


def _asymmetric_mask(h: int = 300, w: int = 400) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (45, 60), (180, 95), 255, -1)
    cv2.circle(mask, (260, 205), 28, 255, -1)
    cv2.rectangle(mask, (80, 165), (110, 260), 255, -1)
    return mask


def _good_block(img=None, mask=None) -> PreparedSpecimen:
    return PreparedSpecimen(
        role="block",
        mask=mask if mask is not None else _circle_mask(),
        roi_ok=True,
        roi_reason="ok",
    )


def _good_slide(img=None, mask=None) -> PreparedSpecimen:
    return PreparedSpecimen(
        role="slide",
        mask=mask if mask is not None else _circle_mask(),
        roi_ok=True,
        roi_reason="ok",
    )


def _decision(verdict: str = "REVIEW", stage: str = "scoring", reason: str = "test") -> ClaimDecision:
    return ClaimDecision(
        claim_id="C001",
        block_path="block.jpg",
        slide_path="slide.jpg",
        verdict=verdict,
        stage=stage,
        reason=reason,
        score=0.42,
    )


def _scored_decision(angle, flip, align_soft_iou, mask_iou) -> ClaimDecision:
    return ClaimDecision(
        claim_id="C001",
        block_path="block.jpg",
        slide_path="slide.jpg",
        verdict="REVIEW",
        stage="scoring",
        reason="test",
        score=0.42,
        best_angle=angle,
        best_flip=flip,
        align_soft_iou=align_soft_iou,
        mask_iou=mask_iou,
    )


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_contact_sheet_creates_file(tmp_path):
    out = tmp_path / "sheet_C001.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=out,
    )
    assert out.exists()


def test_contact_sheet_is_readable_image(tmp_path):
    out = tmp_path / "sheet.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=out,
    )
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_contact_sheet_has_minimum_width(tmp_path):
    out = tmp_path / "sheet.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=out,
    )
    img = cv2.imread(str(out))
    # Sheet should be wide enough to show multiple panels side by side
    assert img.shape[1] >= 600, f"sheet width {img.shape[1]} too narrow for panels"


def test_overlay_panel_uses_normalized_d4_alignment():
    block_mask = _asymmetric_mask()
    slide_mask = cv2.rotate(block_mask, cv2.ROTATE_90_CLOCKWISE)

    panel = _overlay_panel(_good_block(mask=block_mask), _good_slide(mask=slide_mask))

    block_pixels = panel[:, :, 0] > 0
    slide_pixels = panel[:, :, 2] > 0
    overlap = block_pixels & slide_pixels
    overlap_ratio = overlap.sum() / max(min(block_pixels.sum(), slide_pixels.sum()), 1)
    assert overlap_ratio > 0.70


# ---------------------------------------------------------------------------
# #199: reuse the scored decision's locked pose instead of re-running
# align_masks's full rotation+flip search when a pose is already known.
# ---------------------------------------------------------------------------

def test_overlay_panel_reuses_pose_skips_align_masks(monkeypatch):
    block_mask = _asymmetric_mask()
    slide_mask = cv2.rotate(block_mask, cv2.ROTATE_90_CLOCKWISE)

    # Compute the real pose BEFORE patching, so the decision carries a
    # genuine locked alignment.
    reference = contact_sheet.align_masks(block_mask, slide_mask)
    decision = _scored_decision(
        reference.best_angle,
        reference.best_flip,
        reference.align_soft_iou,
        reference.mask_iou,
    )

    def _spy(*args, **kwargs):
        raise AssertionError("align_masks should not be called when decision has a pose")

    monkeypatch.setattr(contact_sheet, "align_masks", _spy)

    panel = _overlay_panel(_good_block(mask=block_mask), _good_slide(mask=slide_mask), decision)

    assert panel is not None
    assert panel.size > 0
    assert panel.any()


def test_overlay_panel_no_pose_calls_align_masks(monkeypatch):
    block_mask = _asymmetric_mask()
    slide_mask = cv2.rotate(block_mask, cv2.ROTATE_90_CLOCKWISE)

    real_align_masks = contact_sheet.align_masks
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_align_masks(*args, **kwargs)

    monkeypatch.setattr(contact_sheet, "align_masks", _spy)

    # Pose-less decision (no best_angle set).
    _overlay_panel(_good_block(mask=block_mask), _good_slide(mask=slide_mask), _decision())
    assert len(calls) == 1, "align_masks must run when decision has no locked pose"

    # No decision at all.
    _overlay_panel(_good_block(mask=block_mask), _good_slide(mask=slide_mask), None)
    assert len(calls) == 2, "align_masks must run when there is no decision"


def test_overlay_panel_scored_path_pixel_equivalent():
    block_mask = _asymmetric_mask()
    slide_mask = cv2.rotate(block_mask, cv2.ROTATE_90_CLOCKWISE)

    ref = contact_sheet.align_masks(block_mask, slide_mask)
    decision = _scored_decision(ref.best_angle, ref.best_flip, ref.align_soft_iou, ref.mask_iou)

    block = _good_block(mask=block_mask)
    slide = _good_slide(mask=slide_mask)
    panel_reuse = _overlay_panel(block, slide, decision)
    panel_fresh = _overlay_panel(block, slide)

    assert np.array_equal(panel_reuse, panel_fresh), (
        f"pixel mismatch: {int(np.sum(panel_reuse != panel_fresh))} differing values"
    )


def test_write_contact_sheet_header_includes_slide_id_and_role_label(tmp_path):
    """#151: each specimen in a flagged pair must be labeled with its unique
    id (slide capture id) and a role tag (TOP MATCH / CLAIMED) so the
    operator can tell the two sheets for a disagreement apart. The new
    params are optional -- omitting them must reproduce the pre-#151 sheet
    exactly (same header height), so existing single-pair callers stay
    byte-identical."""
    baseline_out = tmp_path / "sheet_baseline.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=baseline_out,
    )
    labeled_out = tmp_path / "sheet_labeled.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=labeled_out,
        slide_id="capture_9",
        role_label="TOP MATCH",
    )

    baseline = cv2.imread(str(baseline_out))
    labeled = cv2.imread(str(labeled_out))

    # Body width (panel count) is unaffected by the new label line.
    assert labeled.shape[1] == baseline.shape[1]
    # The header must grow to fit the new id/role line -- proof the label
    # was actually rendered, not just accepted and dropped.
    assert labeled.shape[0] > baseline.shape[0]


def test_id_line_text_includes_block_id_via_role_label():
    """#151 AC3: each specimen must be labeled with its unique id. The
    caller (session_workflow._score_work_order) folds the block id into
    role_label (e.g. "TOP MATCH 51151378"); this pins that the rendered
    header text actually carries it through, not just the filename."""
    text = id_line_text("capture_9", "TOP MATCH 51151378")
    assert "51151378" in text
    assert "capture_9" in text
    assert "TOP MATCH" in text


def test_contact_sheet_creates_parent_dirs(tmp_path):
    out = tmp_path / "sheets" / "subdir" / "sheet.png"
    write_contact_sheet(
        block_img=_dark_blob(),
        slide_img=_dark_blob(),
        block_result=_good_block(),
        slide_result=_good_slide(),
        decision=_decision(),
        output_path=out,
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Works with preparation failures (gates-failed claims)
# ---------------------------------------------------------------------------

def test_contact_sheet_accepts_preparation_failure(tmp_path):
    out = tmp_path / "sheet_fail.png"
    write_contact_sheet(
        block_img=None,
        slide_img=_dark_blob(),
        block_result=PreparationFailure(role="block", reason="could not read"),
        slide_result=_good_slide(),
        decision=_decision(stage="preparation", reason="block preparation failed"),
        output_path=out,
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Pipeline integration: sheets written alongside decision CSV
# ---------------------------------------------------------------------------

def test_pipeline_writes_contact_sheets(tmp_path):
    from session.pipeline import run_claim_pipeline

    block_img = tmp_path / "block.jpg"
    slide_img = tmp_path / "slide.jpg"
    blob = _dark_blob()
    cv2.imwrite(str(block_img), blob)
    cv2.imwrite(str(slide_img), blob)

    manifest = tmp_path / "m.csv"
    manifest.write_text(
        f"claim_id,block_path,slide_path\n"
        f"C001,{block_img},{slide_img}\n"
    )
    out_csv = tmp_path / "decisions.csv"
    sheets_dir = tmp_path / "sheets"

    run_claim_pipeline(manifest, out_csv, sheets_dir=sheets_dir)

    sheet_files = list(sheets_dir.glob("*.png"))
    assert len(sheet_files) >= 1, "pipeline should write at least one contact sheet"
