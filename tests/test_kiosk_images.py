"""JPEG downscale/encode helpers for kiosk still display (#137)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from kiosk.images import (
    DEFAULT_STILL_MAX_LONG_EDGE,
    encode_image_jpeg,
    encode_preview_jpeg,
    encode_still_jpeg,
)


def test_encode_still_jpeg_downscales_wide_png(tmp_path):
    path = tmp_path / "capture_1_block_20260709T120000Z.png"
    wide = np.zeros((3040, 4056, 3), dtype=np.uint8)
    wide[:, :, 2] = 200  # red channel for decode sanity
    cv2.imwrite(str(path), wide)

    jpeg = encode_still_jpeg(path, max_long_edge=1920)

    assert jpeg[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    h, w = decoded.shape[:2]
    assert max(h, w) <= DEFAULT_STILL_MAX_LONG_EDGE
    assert w > h  # landscape preserved


def test_encode_still_jpeg_raises_when_file_missing(tmp_path):
    with pytest.raises(ValueError, match="cannot read still"):
        encode_still_jpeg(tmp_path / "missing.png")


def test_encode_image_jpeg_returns_valid_bytes():
    patch = np.full((120, 160, 3), 128, dtype=np.uint8)
    jpeg = encode_image_jpeg(patch, quality=80)
    assert jpeg[:2] == b"\xff\xd8"
    round_trip = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert round_trip.shape == patch.shape


def test_encode_preview_jpeg_downscales_live_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = 90  # green channel for decode sanity

    jpeg = encode_preview_jpeg(frame, max_long_edge=320)

    assert jpeg[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 320
