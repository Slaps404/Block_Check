"""Tests for the green-iridescence (coverslip rainbow) slide component filter.

Background
----------
Some H&E slides carry a coverslip/mountant rainbow ("Newton's-rings") artifact.
Its fringe produces *green-dominant* pixels (g > r and g > b) which H&E stain
can never make, so green-dominance marks the artifact. The filter drops a kept
slide component when most of it sits near green AND its own median stain is low.

The guard (the bug fix)
-----------------------
A real tissue dot on slide_017 sat right next to the rainbow band, so proximity
alone wrongly condemned it. The fix: never remove a component whose own median
stain is strong (>= STAIN_GUARD) — iridescent fringe is pale, real tissue is
saturated. test_guard_spares_stained_component_near_green locks this down.

Feature flag
------------
SLIDE_IRIDESCENCE_FILTER_ENABLED is OFF by default; production behaviour is
unchanged until it is deliberately turned on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_CODE = _REPO / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import constants  # noqa: E402
import verify.segmentation as seg  # noqa: E402
from session.preparation import prepare_specimen_from_image, PreparedSpecimen  # noqa: E402

_DATASET = _REPO / "images" / "pi_images_v3"
_GREEN_BGR = (100, 200, 100)   # b,g,r: green-dominant -> "iridescence" source
_BG_BGR = (235, 235, 235)


# ---------------------------------------------------------------------------
# Function-level mechanism tests (synthetic, deterministic, always run)
# ---------------------------------------------------------------------------

def _scene():
    """Build (mask, bgr, stain) with three components and two green bands.

    Layout (cols): [comp1 pale] [GREEN] [comp2 stained] [GREEN] ...  comp3 far.
    comp1 sits next to green and is pale            -> should be REMOVED
    comp2 sits between green bands but is stained    -> guard KEEPS it
    comp3 is far from any green                      -> KEEPS it
    """
    h, w = 200, 200
    bgr = np.full((h, w, 3), _BG_BGR, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    stain = np.zeros((h, w), dtype=np.uint8)

    # green bands (NOT part of the mask -- green pixels are never kept tissue)
    bgr[20:40, 41:61] = _GREEN_BGR
    bgr[20:40, 81:101] = _GREEN_BGR

    # comp1: pale, adjacent to first green band -> removed
    mask[20:40, 20:40] = 255
    stain[20:40, 20:40] = 10

    # comp2: strongly stained, sandwiched by green bands -> guard keeps it
    mask[20:40, 61:81] = 255
    stain[20:40, 61:81] = 120

    # comp3: far from green -> kept
    mask[150:170, 150:170] = 255
    stain[150:170, 150:170] = 200
    return mask, bgr, stain


def test_removes_pale_component_near_green():
    mask, bgr, stain = _scene()
    out = seg._remove_green_iridescence(mask, bgr, stain)
    assert out[30, 30] == 0, "pale component next to green iridescence was not removed"


def test_guard_spares_stained_component_near_green():
    """THE regression: a strongly stained dot adjacent to iridescence survives."""
    mask, bgr, stain = _scene()
    out = seg._remove_green_iridescence(mask, bgr, stain)
    assert out[30, 70] == 255, (
        "stain guard failed: a real (strongly stained) tissue dot was removed "
        "merely because it sits near the iridescent band"
    )


def test_keeps_component_far_from_green():
    mask, bgr, stain = _scene()
    out = seg._remove_green_iridescence(mask, bgr, stain)
    assert out[160, 160] == 255, "component far from any green was wrongly removed"


# ---------------------------------------------------------------------------
# Feature-flag gating (synthetic, always run)
# ---------------------------------------------------------------------------

def test_filter_enabled_in_production():
    """The filter is ON in production: it removes the coverslip rainbow fans
    so SLIDE_LAB_A_MIN can stay at the slide_008-safe 130."""
    assert constants.SLIDE_IRIDESCENCE_FILTER_ENABLED is True
    assert constants.SLIDE_LAB_A_MIN == 130, (
        "LAB_A_MIN must stay 130 (008-safe); artifacts are handled by the filter"
    )


def _slide_image():
    """Synthetic slide with a pink tissue blob that passes the H&E stain gate."""
    img = np.full((600, 800, 3), 230, dtype=np.uint8)
    cv2.circle(img, (400, 300), 80, (200, 150, 230), -1)
    return img


def test_flag_gates_the_filter_call(monkeypatch):
    """segment_tissue calls the filter only when the flag is on."""
    calls = []
    real = seg._remove_green_iridescence

    def recorder(mask, bgr, stain, *, pixel_scale=1.0):
        calls.append(True)
        return real(mask, bgr, stain, pixel_scale=pixel_scale)

    monkeypatch.setattr(seg, "_remove_green_iridescence", recorder)

    monkeypatch.setattr(seg, "SLIDE_IRIDESCENCE_FILTER_ENABLED", False)
    seg.segment_tissue(_slide_image(), "slide")
    assert not calls, "filter ran even though the flag is OFF"

    monkeypatch.setattr(seg, "SLIDE_IRIDESCENCE_FILTER_ENABLED", True)
    seg.segment_tissue(_slide_image(), "slide")
    assert calls, "filter did not run even though the flag is ON"


# ---------------------------------------------------------------------------
# Real-image regression tests (skip if dataset absent)
# ---------------------------------------------------------------------------

_SLIDE_017 = _DATASET / "slide_017_esophagus_WT3_01_HE.png"
_SLIDE_014 = _DATASET / "slide_014_esophagus_WT2_01_HE.png"


def _prep_mask(path, *, filter_on):
    img = cv2.imread(str(path))
    assert img is not None, f"could not read {path}"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(seg, "SLIDE_IRIDESCENCE_FILTER_ENABLED", filter_on)
        result = prepare_specimen_from_image(img, "slide")
    assert isinstance(result, PreparedSpecimen)
    return result.mask


@pytest.mark.skipif(not _SLIDE_017.exists(), reason="slide_017 not in images/pi_images_v3/")
def test_slide_017_real_dot_survives():
    """The esophagus dot near the rainbow band must NOT be removed (the bug)."""
    mask_on = _prep_mask(_SLIDE_017, filter_on=True)
    # dot centroid measured at ~(2222, 1284); assert tissue survives in its window
    window = mask_on[1284 - 40:1284 + 40, 2222 - 40:2222 + 40]
    assert np.count_nonzero(window) > 0, (
        "slide_017 esophagus dot was deleted by the iridescence filter"
    )


@pytest.mark.skipif(not _SLIDE_014.exists(), reason="slide_014 not in images/pi_images_v3/")
def test_slide_014_fan_is_removed():
    """The slide_014 iridescent fan must be substantially removed when ON."""
    off = int(np.count_nonzero(_prep_mask(_SLIDE_014, filter_on=False)))
    on = int(np.count_nonzero(_prep_mask(_SLIDE_014, filter_on=True)))
    assert on < off * 0.9, (
        f"expected the iridescent fan to be removed (>=10% fewer px); "
        f"off={off} on={on}"
    )
