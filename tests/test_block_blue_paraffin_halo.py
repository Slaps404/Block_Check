"""Synthetic tests for the blue-paraffin-gated halo relaxation (iteration 047).

On blue-wax blocks (brain 039/040/041) the translucent olive tissue rim sits
below the default halo Lab-b floor and runs to greener hues, so the default gate
(`hue<=60, Lab-b>=170`) captures only the dense core. `_grow_tissue_halo` detects
blue paraffin from the wax region (`_is_blue_paraffin_wax`: median wax sat>=165
AND median wax Lab-b<=100) and, only then, relaxes the gate to
`hue<=95, Lab-b>=115` (sat unchanged). Purple/pale-wax blocks keep the strict
gate, so the same olive rim is NOT grown there.

The discriminating pixel is an olive rim at hue~72: it fails the default hue<=60
clause but passes the relaxed hue<=95 clause. Whether it is grown therefore
depends entirely on which wax the block sits in.

Covers:
  - Blue wax  -> detector fires  -> olive rim IS grown, core preserved.
  - Purple wax -> detector off   -> same olive rim is NOT grown.
  - Blue wax itself is not admitted (no wax leak).
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

from verify import segmentation  # noqa: E402
from verify.segmentation import segment_tissue  # noqa: E402


@pytest.fixture(autouse=True)
def _use_classical_backend(monkeypatch):
    """This module verifies the classical color-segmentation branch only."""
    monkeypatch.setattr(segmentation, "BLOCK_SEGMENTER", "classical")


_IMG = 1000
_CENTER = (500, 500)
_CORE_R = 50
_RIM_R = 75  # rim band 50-75 px: within 30px of the core (seed-connected) so the
#             seed-dilation used by the detector excludes it from the wax median.

# Yellow dense core -> fires brown_hsv, so it is a seed.
_CORE_BGR = (0, 210, 210)          # H30 S255 -> seed
# Olive rim -> H72 (above the default 60 cutoff, below the relaxed 95), Lab-b 165,
# sat 197. Not a seed (hue>55). Grown only under the relaxed (blue) gate.
_RIM_BGR = (120, 220, 50)          # H72 S197 Lab-b165
# Blue paraffin wax -> sat 179, Lab-b 87: fires the blue detector; itself not
# grown (hue 102 > 95 and Lab-b 87 < 115).
_WAX_BLUE = (235, 170, 70)         # H102 S179 Lab-b87
# Purple paraffin wax -> sat 102 (< 165): detector stays off. hue 146 not grown.
_WAX_PURPLE = (200, 120, 190)      # H146 S102 Lab-b97


def _make_image(wax_bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.full((_IMG, _IMG, 3), wax_bgr, dtype=np.uint8)
    cv2.circle(img, _CENTER, _RIM_R, _RIM_BGR, thickness=-1)
    cv2.circle(img, _CENTER, _CORE_R, _CORE_BGR, thickness=-1)
    return img


def _disc(center, radius) -> np.ndarray:
    m = np.zeros((_IMG, _IMG), dtype=np.uint8)
    cv2.circle(m, center, radius, 255, thickness=-1)
    return m > 0


_WIN = (0, 0, _IMG, _IMG)
_CORE = _disc(_CENTER, _CORE_R)
_RIM = _disc(_CENTER, _RIM_R) & ~_disc(_CENTER, _CORE_R + 3)


def test_blue_paraffin_rim_is_grown():
    """On blue wax the detector fires and the olive rim is recovered."""
    result = segment_tissue(_make_image(_WAX_BLUE), "block", block_window=_WIN)

    core_cov = (result[_CORE] == 255).mean()
    rim_cov = (result[_RIM] == 255).mean()

    assert core_cov >= 0.99, f"grow eroded the seed core (only {core_cov:.1%})"
    assert rim_cov >= 0.90, (
        f"olive rim not grown on blue wax (only {rim_cov:.1%}); the blue detector "
        "should have relaxed the gate to hue<=95, Lab-b>=115"
    )


def test_purple_paraffin_rim_not_grown():
    """The identical olive rim on purple wax stays out (default gate)."""
    result = segment_tissue(_make_image(_WAX_PURPLE), "block", block_window=_WIN)

    rim_cov = (result[_RIM] == 255).mean()
    assert rim_cov <= 0.10, (
        f"olive rim (hue~72) was grown on PURPLE wax ({rim_cov:.1%}); the relaxed "
        "gate must fire only on blue paraffin, not purple"
    )


def test_blue_wax_is_not_admitted():
    """Blue wax pixels far from the tissue are never grown (no wax leak)."""
    result = segment_tissue(_make_image(_WAX_BLUE), "block", block_window=_WIN)
    assert result[100, 100] == 0, "blue paraffin wax was admitted into the mask"
