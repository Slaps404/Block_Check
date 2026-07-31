"""Synthetic seam tests for the translucent-halo region grow.

Tissue is bright yellow only at its dense core; its thinner edges are a
translucent olive halo (hue just above the brown gate's cutoff) that the chroma
gates drop. `_grow_tissue_halo` recovers it as a hysteresis grow: weak olive
pixels are kept only where they connect to a confident seed, and only inside the
cassette window.

Covers:
  - Halo touching a seed IS grown, and the seed is preserved (purely additive).
  - A floating halo blob NOT touching any seed is NOT grown.
  - The grow is window-scoped: halo outside the window stays dark.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from verify.segmentation import segment_tissue  # noqa: E402

# Canvas is large so the grown disc stays under BLOCK_AREA_MAX_FRAC (0.04 of the
# image); on a small canvas the merged blob would trip the "too large" component
# filter, which is a canvas artifact, not the behavior under test.
_IMG = 1000
_CORE_CENTER = (500, 500)
_CORE_R = 45
_HALO_R = 80
_FLOAT_CENTER = (820, 150)
_FLOAT_R = 30

# Confident yellow core -> fires brown_hsv (a seed). HSV H30 S255.
_CORE_BGR = (0, 210, 210)
# Olive halo -> hue 48 (above brown's 45 cutoff) but inside the weak band
# (hue<=60, Lab-b>=170, sat>=60). Not a seed on its own.
_HALO_BGR = (80, 210, 130)
# Green wax -> hue 65, fails every gate including the weak band.
_WAX_BGR = (90, 200, 70)


def _make_image() -> np.ndarray:
    """Wax bg + olive halo disc + yellow core on top + a detached olive blob."""
    img = np.full((_IMG, _IMG, 3), _WAX_BGR, dtype=np.uint8)
    cv2.circle(img, _CORE_CENTER, _HALO_R, _HALO_BGR, thickness=-1)
    cv2.circle(img, _CORE_CENTER, _CORE_R, _CORE_BGR, thickness=-1)
    cv2.circle(img, _FLOAT_CENTER, _FLOAT_R, _HALO_BGR, thickness=-1)
    return img


def _disc(center, radius) -> np.ndarray:
    m = np.zeros((_IMG, _IMG), dtype=np.uint8)
    cv2.circle(m, center, radius, 255, thickness=-1)
    return m > 0


def test_halo_grown_when_touching_seed():
    """Olive halo around a yellow core is captured, and the core is preserved."""
    img = _make_image()
    win = (0, 0, _IMG, _IMG)

    result = segment_tissue(img, "block", block_window=win)

    core = _disc(_CORE_CENTER, _CORE_R)
    halo = _disc(_CORE_CENTER, _HALO_R) & ~_disc(_CORE_CENTER, _CORE_R + 3)

    core_cov = (result[core] == 255).mean()
    halo_cov = (result[halo] == 255).mean()

    assert core_cov >= 0.99, f"grow eroded the seed core (only {core_cov:.1%} kept)"
    assert halo_cov >= 0.90, f"olive halo not grown (only {halo_cov:.1%} captured)"


def test_floating_halo_not_grown():
    """An olive blob with no seed in its component must stay dark."""
    img = _make_image()
    win = (0, 0, _IMG, _IMG)

    result = segment_tissue(img, "block", block_window=win)

    fx, fy = _FLOAT_CENTER
    assert result[fy, fx] == 0, (
        "detached olive blob was grown despite touching no seed — "
        "the grow is not anchored to seeds."
    )


def test_halo_grow_scoped_to_window():
    """Halo outside the window is not grown even though it touches the seed."""
    img = _make_image()
    win = (0, 0, 100, 100)  # corner window, excludes the centred blob

    result = segment_tissue(img, "block", block_window=win)

    # A point on the outer halo ring, well outside the window and far from the
    # core edge (so morphological closing cannot bridge to it).
    hx, hy = 500, 578
    assert result[hy, hx] == 0, (
        f"outer-halo pixel ({hx},{hy}) was grown even though it lies outside the "
        "window — the grow's weak band is not window-scoped."
    )
