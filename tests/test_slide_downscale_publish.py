"""Store-time slide downscale in ``CaptureStore.publish`` (issue #186, Task B4).

The Pi sensor still shoots full resolution (``STILL_SIZE`` unchanged); this
only shrinks the *stored/sent* slide PNG down to ``CAPTURE_DIMENSIONS["slide"]``
when the source is larger. Blocks and already-correct-size slides pass
through unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

import constants
from capture_storage import CaptureStore


NOW = datetime(2026, 7, 1, 19, 30, 45, tzinfo=timezone.utc)


@pytest.fixture
def out_store(tmp_path):
    return CaptureStore(tmp_path / "published")


def _full_png(tmp_path, name="capture_0001.png"):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((3040, 4056, 3), 200, dtype=np.uint8))
    return p


def test_slide_is_downscaled_to_configured_dims(tmp_path, monkeypatch, out_store):
    # force half-res slide config for this test
    monkeypatch.setitem(constants.CAPTURE_DIMENSIONS, "slide", (2028, 1520))
    src = _full_png(tmp_path)
    record = out_store.publish(src, "slide", captured_at=NOW)
    persisted = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
    assert persisted.shape[:2] == (1520, 2028)


def test_block_is_not_downscaled(tmp_path, out_store):
    src = _full_png(tmp_path)
    record = out_store.publish(
        src, "block", block_id="51151378", captured_at=NOW
    )
    persisted = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
    assert persisted.shape[:2] == (3040, 4056)
