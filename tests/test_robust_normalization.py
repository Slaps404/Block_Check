"""Behavior tests for diagnostic-only robust radial normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
TOOLS = ROOT / "tools" / "scoring_diagnostics"
for path in (CODE, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verify.locked_alignment import radial_normalize_mask  # noqa: E402
from robust_normalization import normalize_mask  # noqa: E402


def _two_lobes(*, residual: bool) -> np.ndarray:
    mask = np.zeros((640, 640), dtype=np.uint8)
    cv2.ellipse(mask, (235, 340), (78, 52), 20, 0, 360, 255, -1)
    cv2.ellipse(mask, (405, 325), (72, 50), -15, 0, 360, 255, -1)
    if residual:
        cv2.circle(mask, (600, 70), 12, 255, -1)
    return mask


def test_max_mode_reproduces_the_previous_production_normalizer():
    mask = _two_lobes(residual=True)

    assert np.array_equal(
        normalize_mask(mask, "max"),
        radial_normalize_mask(mask, mode="max"),
    )


def test_rms_scale_is_less_sensitive_to_remote_small_component_than_max():
    clean = _two_lobes(residual=False)
    residual = _two_lobes(residual=True)

    max_area_ratio = normalize_mask(residual, "max").sum() / normalize_mask(clean, "max").sum()
    rms_area_ratio = normalize_mask(residual, "rms").sum() / normalize_mask(clean, "rms").sum()

    assert rms_area_ratio > max_area_ratio + 0.25


def test_every_robust_mode_keeps_a_remote_component_as_evidence():
    mask = _two_lobes(residual=True)

    for mode in ("rms", "percentile_98", "power_4"):
        normalized = normalize_mask(mask, mode)
        count, _, _, _ = cv2.connectedComponentsWithStats(normalized, 8)
        assert count - 1 == 3, mode
