"""Visual review contact sheet for v2 claimed-pair verification.

Single PNG per claim: block/slide images, masks, overlay, verdict, reason.

Code map
--------
write_contact_sheet(...)   ← pipeline entry
    Compose and write the review PNG.
_resize, _placeholder, _image_panel, _mask_panel, _overlay_panel, _header_panel
    Panel builders and layout helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from session.preparation import PreparedResult, PreparedSpecimen, PreparationFailure
from verify.locked_alignment import LockedAlignment, align_masks

if TYPE_CHECKING:
    from session.pipeline import ClaimDecision

_PANEL_H = 256
_PANEL_W = 256
_HEADER_H = 60
_ID_LINE_H = 24
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


class ContactSheetRenderer(Protocol):
    """Injectable seam over :func:`write_contact_sheet` (mirrors the
    ``work_order_scorer`` callable seam in ``session_workflow.py``)."""

    def __call__(
        self,
        block_img: np.ndarray | None,
        slide_img: np.ndarray | None,
        block_result: PreparedResult,
        slide_result: PreparedResult,
        decision: "ClaimDecision",
        output_path: str | Path,
        *,
        slide_id: str | None = None,
        role_label: str | None = None,
    ) -> None:
        ...


def write_contact_sheet(
    block_img: np.ndarray | None,
    slide_img: np.ndarray | None,
    block_result: PreparedResult,
    slide_result: PreparedResult,
    decision: ClaimDecision,
    output_path: str | Path,
    *,
    slide_id: str | None = None,
    role_label: str | None = None,
) -> None:
    """Write a contact sheet PNG for one claim row.

    ``slide_id``/``role_label`` are optional (#151): when supplied, an extra
    header line identifies which specimen this sheet is for (the slide's
    unique capture id) and its role in a flagged pair ("TOP MATCH" /
    "CLAIMED"). Omitting both reproduces the pre-#151 sheet exactly.
    """
    panels = [
        _image_panel(block_img, "block"),
        _image_panel(slide_img, "slide"),
        _mask_panel(block_result, color=(255, 80, 30)),   # blue-ish (BGR)
        _mask_panel(slide_result, color=(30, 80, 255)),   # red-ish (BGR)
        _overlay_panel(block_result, slide_result, decision),
    ]
    body = np.hstack(panels)

    header = _header_panel(
        decision, width=body.shape[1], slide_id=slide_id, role_label=role_label,
    )
    sheet = np.vstack([header, body])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise OSError(f"could not write contact sheet: {out}")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_AREA)


def _placeholder(label: str = "") -> np.ndarray:
    panel = np.full((_PANEL_H, _PANEL_W, 3), 60, dtype=np.uint8)
    if label:
        cv2.putText(panel, label, (10, _PANEL_H // 2), _FONT, _FONT_SCALE,
                    (180, 180, 180), _FONT_THICKNESS)
    return panel


def _image_panel(img: np.ndarray | None, label: str) -> np.ndarray:
    if img is None or img.size == 0:
        return _placeholder(f"no {label} image")
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return _resize(bgr)


def _mask_panel(result: PreparedResult, color: tuple) -> np.ndarray:
    if isinstance(result, PreparationFailure):
        return _placeholder("prep failed")
    mask_small = cv2.resize(result.mask, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
    panel[mask_small > 0] = color
    return panel


def _overlay_panel(
    block: PreparedResult,
    slide: PreparedResult,
    decision: ClaimDecision | None = None,
) -> np.ndarray:
    panel = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
    if isinstance(block, PreparedSpecimen) and isinstance(slide, PreparedSpecimen):
        if decision is not None and decision.best_angle is not None:
            from verify.locked_alignment import radial_normalize_mask, transform_mask
            block_mask = radial_normalize_mask(block.mask)
            slide_mask = transform_mask(
                radial_normalize_mask(slide.mask),
                decision.best_angle,
                bool(decision.best_flip),
            )
            alignment = LockedAlignment(
                decision.best_angle,
                bool(decision.best_flip),
                decision.align_soft_iou or 0.0,
                decision.mask_iou or 0.0,
                block_mask,
                slide_mask,
            )
        else:
            alignment = align_masks(block.mask, slide.mask)
        bm = cv2.resize(
            alignment.block_mask,
            (_PANEL_W, _PANEL_H),
            interpolation=cv2.INTER_NEAREST,
        )
        sm = cv2.resize(
            alignment.aligned_slide_mask,
            (_PANEL_W, _PANEL_H),
            interpolation=cv2.INTER_NEAREST,
        )
        panel[bm > 0, 0] = 200   # B channel -> block = blue
        panel[sm > 0, 2] = 200   # R channel -> slide = red
        cv2.putText(
            panel,
            (
                f"angle={alignment.best_angle:.0f} "
                f"flip={alignment.best_flip} IoU={alignment.mask_iou:.2f}"
            ),
            (8, 20),
            _FONT,
            _FONT_SCALE,
            (180, 180, 180),
            _FONT_THICKNESS,
        )
    elif isinstance(block, PreparedSpecimen):
        bm = cv2.resize(block.mask, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_NEAREST)
        panel[bm > 0, 0] = 200   # B channel -> block = blue
    elif isinstance(slide, PreparedSpecimen):
        sm = cv2.resize(slide.mask, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_NEAREST)
        panel[sm > 0, 2] = 200   # R channel -> slide = red
    return panel


def _header_panel(
    decision: ClaimDecision,
    width: int,
    *,
    slide_id: str | None = None,
    role_label: str | None = None,
) -> np.ndarray:
    has_id_line = bool(slide_id or role_label)
    height = _HEADER_H + (_ID_LINE_H if has_id_line else 0)
    panel = np.full((height, width, 3), 30, dtype=np.uint8)
    verdict_color = (80, 220, 80) if decision.verdict == "PASS" else (80, 80, 220)
    score_str = f"  score={decision.score:.3f}" if decision.score is not None else ""
    metric_str = f"  metric={decision.selected_metric}" if decision.selected_metric else ""
    line1 = (
        f"{decision.claim_id}  [{decision.verdict}]{score_str}{metric_str}  "
        f"stage={decision.stage}"
    )
    line2 = f"  {decision.reason[:90]}"
    cv2.putText(panel, line1, (6, 20), _FONT, _FONT_SCALE, verdict_color, _FONT_THICKNESS)
    cv2.putText(panel, line2, (6, 44), _FONT, _FONT_SCALE, (180, 180, 180), _FONT_THICKNESS)
    if has_id_line:
        cv2.putText(
            panel, id_line_text(slide_id, role_label), (6, _HEADER_H + 18),
            _FONT, _FONT_SCALE, (200, 200, 80), _FONT_THICKNESS,
        )
    return panel


def id_line_text(slide_id: str | None, role_label: str | None) -> str:
    """Build the header's id/role text (#151 AC3: each rendered sheet must
    self-identify its slide capture id and role, which in turn carries the
    block id -- e.g. ``role_label="TOP MATCH 51151378"``). Extracted so
    tests can assert on the exact string without OCR-ing the rendered PNG.
    """
    return f"  {slide_id or ''}  [{role_label or ''}]"
