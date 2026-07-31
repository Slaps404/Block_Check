import cv2
import numpy as np
import pytest

from capture_session import CaptureResult, CaptureSession
from capture_storage import CaptureStore, PublicationError, ValidatedStill
from constants import CAPTURE_DIMENSIONS, NATIVE_CAPTURE_DIMENSIONS


def test_roles_have_dimensions():
    assert CAPTURE_DIMENSIONS["block"] == NATIVE_CAPTURE_DIMENSIONS
    assert NATIVE_CAPTURE_DIMENSIONS == (4056, 3040)
    # Slide uses a qualified downscale policy, distinct from native capture.
    assert CAPTURE_DIMENSIONS["slide"] == (2028, 1520)
    # slide shares the sensor (block) aspect ratio at any resolution tier
    bw, bh = CAPTURE_DIMENSIONS["block"]
    sw, sh = CAPTURE_DIMENSIONS["slide"]
    assert sw * bh == sh * bw


def _write_png(tmp_path, w, h):
    p = tmp_path / "capture_0001.png"
    cv2.imwrite(str(p), np.zeros((h, w, 3), dtype=np.uint8))
    return p


def test_validate_still_accepts_role_dimensions(tmp_path):
    bw, bh = NATIVE_CAPTURE_DIMENSIONS
    p = _write_png(tmp_path, bw, bh)
    CaptureStore._validate_still(p, "block")  # must not raise


def test_validate_still_rejects_wrong_dimensions(tmp_path):
    p = _write_png(tmp_path, 1234, 567)
    with pytest.raises(PublicationError):
        CaptureStore._validate_still(p, "block")


def test_validate_still_rejects_unknown_role(tmp_path):
    p = _write_png(tmp_path, *NATIVE_CAPTURE_DIMENSIONS)
    with pytest.raises(PublicationError):
        CaptureStore._validate_still(p, "not_a_role")


def test_storage_validation_facts_are_accepted_by_session_for_same_role(tmp_path):
    width, height = CAPTURE_DIMENSIONS["slide"]
    path = _write_png(tmp_path, width, height)

    validated = CaptureStore._validate_still(path, "slide")
    result = CaptureResult.success(path, validated=validated)

    assert validated == ValidatedStill(width=width, height=height, format=".png")
    assert CaptureSession._still_acceptable(result, "slide")
