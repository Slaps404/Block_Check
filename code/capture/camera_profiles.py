"""Named capture-size profiles for temporary sensor swaps.

Production remains the IMX477 HQ contract in ``constants``. The ``imx219``
profile (Camera Module v2.1) is a smoke-test overlay: it retargets still
request/validation sizes only. Exposures, colour gains, framing, and CV
thresholds stay HQ-tuned and are not trustworthy on v2.1 without retune.
"""
from __future__ import annotations

from dataclasses import dataclass

import constants


@dataclass(frozen=True)
class CameraProfile:
    name: str
    native_dimensions: tuple[int, int]
    slide_dimensions: tuple[int, int]
    smoke_only: bool = False


DEFAULT_CAMERA_PROFILE = "imx477"

CAMERA_PROFILES: dict[str, CameraProfile] = {
    "imx477": CameraProfile(
        name="imx477",
        native_dimensions=(4056, 3040),
        slide_dimensions=(2028, 1520),
    ),
    "imx219": CameraProfile(
        name="imx219",
        native_dimensions=(3280, 2464),
        slide_dimensions=(1640, 1232),
        smoke_only=True,
    ),
}

CAMERA_PROFILE_ALIASES: dict[str, str] = {
    "hq": "imx477",
    "v2.1": "imx219",
    "v21": "imx219",
    "camera-v2": "imx219",
}

_active_profile_name = DEFAULT_CAMERA_PROFILE


def normalize_camera_profile_name(name: str) -> str:
    key = name.strip().lower()
    if key in CAMERA_PROFILES:
        return key
    if key in CAMERA_PROFILE_ALIASES:
        return CAMERA_PROFILE_ALIASES[key]
    known = ", ".join(sorted({*CAMERA_PROFILES, *CAMERA_PROFILE_ALIASES}))
    raise ValueError(f"unknown camera profile {name!r}; expected one of: {known}")


def active_camera_profile_name() -> str:
    return _active_profile_name


def apply_camera_profile(name: str = DEFAULT_CAMERA_PROFILE) -> CameraProfile:
    """Mutate shared ``CAPTURE_DIMENSIONS`` / native size to the named profile."""
    global _active_profile_name
    profile_name = normalize_camera_profile_name(name)
    profile = CAMERA_PROFILES[profile_name]
    constants.NATIVE_CAPTURE_DIMENSIONS = profile.native_dimensions
    constants.CAPTURE_DIMENSIONS["block"] = profile.native_dimensions
    constants.CAPTURE_DIMENSIONS["slide"] = profile.slide_dimensions
    _active_profile_name = profile_name
    return profile


def reset_camera_profile() -> CameraProfile:
    """Restore the production IMX477 sizes (for tests / teardown)."""
    return apply_camera_profile(DEFAULT_CAMERA_PROFILE)
