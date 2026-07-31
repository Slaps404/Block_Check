"""Tests for block ROI masking."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import block.roi_mask as block_roi_mask  # noqa: E402
from block.roi_mask import (  # noqa: E402
    find_cassette_bbox,
    find_cassette_window,
    keep_components_in_window,
    expand_window_for_segmentation,
    window_relative_to_crop,
)


def test_window_masked_uses_layer1_only_even_when_layer2_is_found(monkeypatch):
    gray = np.zeros((10, 10), dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    monkeypatch.setattr(block_roi_mask, "find_cassette_window", lambda _gray: (1, 1, 6, 6))
    monkeypatch.setattr(
        block_roi_mask,
        "find_paraffin_window",
        lambda _gray: (3, 3, 2, 2),
        raising=False,
    )

    out = block_roi_mask.window_masked(gray, mask)

    assert np.count_nonzero(out) == 36
    assert out[1:7, 1:7].min() == 255
    assert out[:1, :].max() == 0
    assert out[:, :1].max() == 0


# --- clip_components_in_window: connectivity-aware clip --------------------
# The hard rectangle clip amputates tissue tips that poke past the inset line.
# A component-aware clip keeps any component that is mostly inside the window
# (so a straddling tissue piece is kept WHOLE, tail included) while still
# dropping isolated specks that sit fully outside. Window is (x, y, w, h);
# the inside region is rows y:y+h, cols x:x+w.


def test_straddling_component_kept_whole_including_outside_tail():
    """A component >= frac inside the window is kept entirely, tail and all."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    # cols 3..12 (10 wide): 7 cols inside window (3..9), 3 cols outside (10..12).
    mask[5:9, 3:13] = 255
    window = (0, 0, 10, 20)  # left half

    out = keep_components_in_window(mask, window, 0.5)

    assert np.array_equal(out, mask), "mostly-inside component was not kept whole"


def test_fully_outside_component_dropped():
    """An isolated speck wholly outside the window is removed."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:9, 12:16] = 255  # entirely right of the window (cols 0..9)
    window = (0, 0, 10, 20)

    out = keep_components_in_window(mask, window, 0.5)

    assert np.count_nonzero(out) == 0, "fully-outside speck survived the clip"


def test_mostly_outside_component_dropped():
    """A component with < frac of its area inside the window is removed."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    # cols 7..16 (10 wide): 3 cols inside (7..9), 7 outside -> 30% inside.
    mask[5:9, 7:17] = 255
    window = (0, 0, 10, 20)

    out = keep_components_in_window(mask, window, 0.5)

    assert np.count_nonzero(out) == 0, "mostly-outside component was kept"


def test_passthrough_when_window_none():
    """No window found -> mask is returned unchanged."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:9, 3:13] = 255

    out = keep_components_in_window(mask, None, 0.5)

    assert np.array_equal(out, mask)


def test_expand_window_for_segmentation_adds_buffer_and_clamps():
    window = (100, 100, 200, 100)
    expanded = expand_window_for_segmentation(window, (500, 500), buffer_frac=0.20)

    assert expanded == (60, 80, 280, 140)


def test_window_relative_to_crop_maps_inner_bbox():
    inner = (100, 100, 200, 100)
    crop = (60, 80, 280, 140)

    assert window_relative_to_crop(inner, crop) == (40, 20, 200, 100)


def test_find_cassette_bbox_is_full_blob_window_is_inset():
    """Full cassette bbox is the dark blob; window insets that bbox."""
    gray = np.full((200, 200), 220, dtype=np.uint8)
    gray[40:160, 30:170] = 40  # dark cassette body

    bbox = find_cassette_bbox(gray)
    window = find_cassette_window(gray)

    assert bbox is not None and window is not None
    bx, by, bw, bh = bbox
    wx, wy, ww, wh = window
    assert bx <= wx and by <= wy
    assert bx + bw >= wx + ww and by + bh >= wy + wh
    assert bw > ww and bh > wh
