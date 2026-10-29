import cv2
import numpy as np

from session import preparation as prep
from verify import segmentation as seg
from verify.scale import scale_area_min, scale_odd_length


def _synthetic_slide(width: int, height: int) -> np.ndarray:
    """A pink H&E-ish blob on a white field at the requested size.

    The blob carries deterministic per-pixel noise (seeded on (width, height)
    so repeat calls with the same size are reproducible) instead of a flat
    fill: SLIDE_TEXTURE_FLOOR requires local variance, and a uniform color
    patch has none, so segment_tissue would otherwise return an all-zero
    mask regardless of resolution.
    """
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    r = max(6, width // 12)
    rng = np.random.default_rng(seed=(width, height))
    block = np.full((2 * r, 2 * r, 3), (150, 120, 210), dtype=np.int16)  # BGR pinkish
    block = np.clip(block + rng.normal(0, 6.0, block.shape), 0, 255).astype(np.uint8)
    img[cy - r:cy + r, cx - r:cx + r] = block
    return img


def test_full_resolution_mask_unchanged_by_default():
    img = _synthetic_slide(4056, 3040)
    baseline = seg.segment_tissue(img, "slide")
    explicit = seg.segment_tissue(img, "slide", pixel_scale=1.0)
    assert np.array_equal(baseline, explicit)


def test_scaled_min_area_uses_experiment_rounding(monkeypatch):
    captured = {}
    real_postprocess = seg._postprocess

    def spy(mask, *, min_area, role, close_ksize=5, area_reference=None, pixel_scale=1.0):
        captured["min_area"] = min_area
        captured["close_length"] = scale_odd_length(close_ksize, pixel_scale)
        captured["pixel_scale"] = pixel_scale
        return real_postprocess(
            mask, min_area=min_area, role=role, close_ksize=close_ksize,
            area_reference=area_reference, pixel_scale=pixel_scale,
        )

    monkeypatch.setattr(seg, "_postprocess", spy)
    img = _synthetic_slide(2028, 1520)
    seg.segment_tissue(img, "slide", pixel_scale=0.5)
    assert captured["min_area"] == scale_area_min(seg.MIN_SLIDE_COMPONENT_AREA, 0.5)
    assert captured["close_length"] == scale_odd_length(seg.SLIDE_CLOSE_KERNEL, 0.5)
    # the component filter must see the scale too (slide_017 dot regression)
    assert captured["pixel_scale"] == 0.5


def test_slide_absolute_component_floor_scales_with_resolution():
    """A fragment above the scaled 450 px floor survives at half resolution.

    Mirrors slide_017: at 2028 px width its real esophagus dot is ~405 px,
    the full-resolution equivalent of ~1600 px.
    """
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (69, 69), 255, -1)  # 3600 px largest
    cv2.rectangle(mask, (80, 80), (99, 99), 255, -1)  # 400 px ~= 1600 px native
    out = seg._component_filter(mask, min_area=1, role="slide", pixel_scale=0.5)
    assert out[90, 90] == 255


def test_prepare_scales_opposite_tag_cap(monkeypatch):
    captured = {}
    real = prep._remove_opposite_tag_artifacts

    def spy(mask, label_rect, *, pixel_scale=1.0):
        captured["pixel_scale"] = pixel_scale
        return real(mask, label_rect, pixel_scale=pixel_scale)

    monkeypatch.setattr(prep, "_remove_opposite_tag_artifacts", spy)
    img = _synthetic_slide(2028, 1520)
    prep.prepare_specimen_from_image(img, "slide")
    assert captured["pixel_scale"] == 0.5


def test_half_downscale_preserves_mask_within_guardrails():
    full = _synthetic_slide(4056, 3040)
    half = cv2.resize(full, (2028, 1520), interpolation=cv2.INTER_AREA)

    m_full = prep.prepare_specimen_from_image(full, "slide").mask
    m_half = prep.prepare_specimen_from_image(half, "slide").mask

    frac_full = m_full.mean() / 255.0
    frac_half = m_half.mean() / 255.0
    assert frac_full > 0, "full-res mask is empty; drift check below would pass vacuously"
    assert abs(frac_full - frac_half) / max(frac_full, 1e-6) <= 0.05
