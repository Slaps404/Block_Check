"""Behavior tests for slide-image tissue overlay compositing."""

from __future__ import annotations

import cv2
import numpy as np

from verify import slide_image_overlay as overlay_mod
from verify.slide_image_overlay import (
    DEFAULT_SLIDE_OVERLAY_OPACITY,
    build_slide_image_overlay,
)


def _circle_mask(size: int, center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _solid_bgr(size: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _aligned_pair(size: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = (size // 2, size // 2)
    radius = size // 5
    block_mask = _circle_mask(size, center, radius)
    slide_mask = _circle_mask(size, center, radius)
    block_bgr = _solid_bgr(size, (40, 40, 200))
    slide_bgr = _solid_bgr(size, (40, 200, 40))
    return block_bgr, slide_bgr, block_mask, slide_mask


def test_outside_slide_mask_matches_block_on_cassette_crop(monkeypatch):
    # Force a known cassette crop so Otsu on flat test images cannot interfere.
    monkeypatch.setattr(
        overlay_mod, "find_cassette_bbox", lambda _gray: (40, 40, 240, 240)
    )
    block_bgr, slide_bgr, block_mask, slide_mask = _aligned_pair()
    overlay = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=False,
        opacity=0.75,
    )
    block_crop = block_bgr[40:280, 40:280]
    assert overlay.shape == block_crop.shape
    # Corners of the cassette crop are outside the circular tissue.
    for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        np.testing.assert_array_equal(overlay[y, x], block_crop[y, x])


def test_inside_mask_blends_slide_over_block(monkeypatch):
    monkeypatch.setattr(
        overlay_mod, "find_cassette_bbox", lambda _gray: (0, 0, 320, 320)
    )
    block_bgr, slide_bgr, block_mask, slide_mask = _aligned_pair()
    cy = cx = 160

    full_slide = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=False,
        opacity=1.0,
    )
    np.testing.assert_array_equal(full_slide[cy, cx], slide_bgr[0, 0])

    half_blend = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=False,
        opacity=0.5,
    )
    expected = np.rint(
        block_bgr[0, 0].astype(np.float32) * 0.5
        + slide_bgr[0, 0].astype(np.float32) * 0.5
    ).astype(np.uint8)
    np.testing.assert_allclose(half_blend[cy, cx], expected, atol=1)


def test_flip_moves_distinctive_slide_mark(monkeypatch):
    size = 256
    monkeypatch.setattr(
        overlay_mod, "find_cassette_bbox", lambda _gray: (0, 0, size, size)
    )
    center = (size // 2, size // 2)
    radius = size // 6
    block_mask = _circle_mask(size, center, radius)
    slide_mask = _circle_mask(size, center, radius)
    block_bgr = _solid_bgr(size, (30, 30, 30))
    slide_bgr = _solid_bgr(size, (30, 30, 30))
    mark = (center[0] - 18, center[1] - 18)
    cv2.circle(slide_bgr, mark, 6, (0, 0, 255), -1)

    unflipped = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=False,
        opacity=1.0,
    )
    flipped = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=True,
        opacity=1.0,
    )

    assert not np.array_equal(unflipped, flipped)

    def _red_centroid(image: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(image[:, :, 2] > 200)
        assert xs.size > 0
        return float(xs.mean()), float(ys.mean())

    unflipped_xy = _red_centroid(unflipped)
    flipped_xy = _red_centroid(flipped)
    assert abs(unflipped_xy[0] - flipped_xy[0]) > 8.0


def test_output_is_full_cassette_crop_not_tissue_bbox(monkeypatch):
    size = 400
    monkeypatch.setattr(
        overlay_mod, "find_cassette_bbox", lambda _gray: (20, 30, 300, 250)
    )
    block_bgr, slide_bgr, block_mask, slide_mask = _aligned_pair(size=size)
    overlay = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=15.0,
        best_flip=False,
        opacity=DEFAULT_SLIDE_OVERLAY_OPACITY,
    )
    assert overlay.shape == (250, 300, 3)
    assert overlay.dtype == np.uint8


def test_native_warp_preserves_slide_checker_texture(monkeypatch):
    """Slide RGB must not be crushed through ALIGN_SIZE before blending."""
    size = 512
    monkeypatch.setattr(
        overlay_mod, "find_cassette_bbox", lambda _gray: (0, 0, size, size)
    )
    center = (size // 2, size // 2)
    radius = size // 5
    block_mask = _circle_mask(size, center, radius)
    slide_mask = _circle_mask(size, center, radius)
    block_bgr = _solid_bgr(size, (80, 80, 80))
    slide_bgr = np.zeros((size, size, 3), dtype=np.uint8)
    # Fine checkerboard only inside the slide mask region.
    for y in range(size):
        for x in range(size):
            if slide_mask[y, x] == 0:
                continue
            tone = 220 if ((x // 4) + (y // 4)) % 2 == 0 else 40
            slide_bgr[y, x] = (tone, tone, tone)

    overlay = build_slide_image_overlay(
        block_bgr,
        slide_bgr,
        block_mask,
        slide_mask,
        best_angle=0.0,
        best_flip=False,
        opacity=1.0,
    )
    ys, xs = np.nonzero(slide_mask > 0)
    # Sample a patch near the center of tissue in the overlay.
    cy, cx = int(ys.mean()), int(xs.mean())
    patch = overlay[cy - 20: cy + 20, cx - 20: cx + 20, 0]
    # Native warp keeps high local contrast; 256-roundtrip would smear toward gray.
    assert float(patch.std()) > 50.0
