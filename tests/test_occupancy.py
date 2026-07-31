from __future__ import annotations

import numpy as np

from occupancy import OCCUPANCY_AREA_FRAC_MAX, detect_occupancy


def _bright_neutral(h=48, w=64):
    # Uniform lit backlight: R == G == B, so saturation is ~0.
    return np.full((h, w, 3), 220, dtype=np.float32)


def test_empty_bright_field_is_not_occupied():
    result = detect_occupancy(_bright_neutral())
    assert result.occupied is False
    assert result.chromatic_fraction == 0.0


def test_chromatic_blob_is_occupied():
    frame = _bright_neutral()
    # ~12.5% of the frame painted purple (high R+B, low G -> saturated).
    frame[12:36, 16:32] = (120, 20, 130)
    result = detect_occupancy(frame)
    assert result.occupied is True
    assert result.chromatic_fraction > OCCUPANCY_AREA_FRAC_MAX


def test_tiny_neutral_dust_is_not_occupied():
    frame = _bright_neutral()
    # A couple of dim-gray specks: still neutral (R == G == B) -> saturation 0.
    frame[0, 0] = (60, 60, 60)
    frame[5, 7] = (40, 40, 40)
    result = detect_occupancy(frame)
    assert result.occupied is False


def test_neutral_dark_jig_blob_is_not_occupied():
    frame = _bright_neutral()
    # A big BLACK corner region (the jig): dark but neutral -> ignored by chroma.
    frame[0:24, 48:64] = (5, 5, 5)
    result = detect_occupancy(frame)
    assert result.occupied is False
    assert result.chromatic_fraction == 0.0
