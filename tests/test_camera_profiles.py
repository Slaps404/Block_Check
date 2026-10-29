"""Camera still-size profiles for HQ vs Camera Module v2.1 smoke."""
from __future__ import annotations

import pytest

import constants
from camera_profiles import (
    apply_camera_profile,
    normalize_camera_profile_name,
    reset_camera_profile,
)
from capture_storage import CaptureStore, PublicationError


@pytest.fixture(autouse=True)
def _restore_hq_profile():
    reset_camera_profile()
    yield
    reset_camera_profile()


def test_default_profile_is_imx477_hq_sizes():
    profile = apply_camera_profile("imx477")
    assert profile.name == "imx477"
    assert profile.smoke_only is False
    assert constants.NATIVE_CAPTURE_DIMENSIONS == (4056, 3040)
    assert constants.CAPTURE_DIMENSIONS["block"] == (4056, 3040)
    assert constants.CAPTURE_DIMENSIONS["slide"] == (2028, 1520)


def test_imx219_profile_retargets_still_sizes():
    profile = apply_camera_profile("imx219")
    assert profile.name == "imx219"
    assert profile.smoke_only is True
    assert constants.NATIVE_CAPTURE_DIMENSIONS == (3280, 2464)
    assert constants.CAPTURE_DIMENSIONS["block"] == (3280, 2464)
    assert constants.CAPTURE_DIMENSIONS["slide"] == (1640, 1232)
    bw, bh = constants.CAPTURE_DIMENSIONS["block"]
    sw, sh = constants.CAPTURE_DIMENSIONS["slide"]
    assert sw * bh == sh * bw


@pytest.mark.parametrize("alias", ["v2.1", "v21", "camera-v2"])
def test_imx219_aliases(alias):
    assert normalize_camera_profile_name(alias) == "imx219"
    profile = apply_camera_profile(alias)
    assert profile.native_dimensions == (3280, 2464)


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown camera profile"):
        apply_camera_profile("imx999")


def test_validate_still_follows_active_profile(tmp_path):
    import cv2
    import numpy as np

    apply_camera_profile("imx219")
    path = tmp_path / "block.png"
    cv2.imwrite(str(path), np.zeros((2464, 3280, 3), dtype=np.uint8))
    CaptureStore._validate_still(path, "block")

    wrong = tmp_path / "hq.png"
    cv2.imwrite(str(wrong), np.zeros((3040, 4056, 3), dtype=np.uint8))
    with pytest.raises(PublicationError):
        CaptureStore._validate_still(wrong, "block")
