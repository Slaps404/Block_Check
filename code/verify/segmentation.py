"""Role-aware color segmentation for v2 claimed-pair verification.

Spatial ROI masking lives in preparation.py / block_roi_mask.py.
role="slide" → H&E; role="block" → brown/yellow.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from constants import (
    BLOCK_AREA_MAX_FRAC,
    BLOCK_BROWN_HUE_MAX,
    BLOCK_BROWN_HUE_MIN,
    BLOCK_BROWN_VAL_MAX,
    BLOCK_RTREES_MODEL_PATH,
    BLOCK_RTREES_SIDECAR_PATH,
    BLOCK_SEGMENTER,
    BLOCK_QUPATH_MODEL_PATH,
    BLOCK_DARK_TISSUE_VAL_MAX,
    BLOCK_DARK_YELLOW_HUE_MAX,
    BLOCK_DARK_YELLOW_HUE_MIN,
    BLOCK_DARK_YELLOW_SAT_MIN,
    BLOCK_DARK_YELLOW_VAL_MAX,
    BLOCK_HALO_GROW_BLUE_HUE_MAX,
    BLOCK_HALO_GROW_BLUE_LAB_B_MIN,
    BLOCK_HALO_GROW_HUE_MAX,
    BLOCK_HALO_GROW_LAB_B_MIN,
    BLOCK_HALO_GROW_SAT_MIN,
    BLOCK_LAB_A_MIN,
    BLOCK_LAB_B_MIN,
    BLOCK_LAB_L_MAX,
    BLOCK_MIN_COMPONENT_REL_AREA,
    BLOCK_SAT_MIN,
    BLUE_PARAFFIN_SEED_DILATE_PX,
    BLUE_PARAFFIN_WAX_LAB_B_MAX,
    BLUE_PARAFFIN_WAX_SAT_MIN,
    DUST_COMPONENT_AREA,
    MIN_AREA_FRACTION,
    MIN_BLOCK_COMPONENT_AREA,
    MIN_SLIDE_COMPONENT_AREA,
    SLIDE_CLOSE_KERNEL,
    SLIDE_IRIDESCENCE_FILTER_ENABLED,
    SLIDE_IRIDESCENCE_GREEN_MARGIN,
    SLIDE_IRIDESCENCE_NEAR_GREEN_FRAC,
    SLIDE_IRIDESCENCE_REACH_PX,
    SLIDE_IRIDESCENCE_STAIN_GUARD,
    SLIDE_LAB_A_MIN,
    SLIDE_MIN_COMPONENT_REL_AREA,
    SLIDE_OTSU_T_MAX,
    SLIDE_OTSU_T_MIN,
    SLIDE_SAT_FLOOR,
    SLIDE_SCORE_CEILING,
    SLIDE_SCORE_FLOOR,
    SLIDE_TEXTURE_FLOOR,
    SLIDE_TEXTURE_WINDOW,
    SLIDE_VAL_MAX,
)
from verify.qupath_rtrees import (
    RtreesSegmenter,
    load_qupath_rtrees_segmenter,
    load_rtrees_segmenter,
)
from verify.scale import scale_area_min, scale_odd_length, scale_reach


def segment_tissue(
    bgr_image: np.ndarray,
    role: str,
    *,
    slide_close_ksize: int | None = None,
    block_window: tuple[int, int, int, int] | None = None,
    block_area_reference: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Binary uint8 mask (0/255) for the given role.

    block_window: precomputed cassette-window bbox (x, y, w, h) from
        block_roi_mask.find_cassette_window, or None. Scopes the dark-tissue
        absorbance gate to the paraffin window so dense (near-black) tissue is
        captured without sweeping in the dark cassette walls. None disables the
        dark gate (chroma-only). Has no effect when role is "slide".
    block_area_reference: full-frame pixel count for ``BLOCK_AREA_MAX_FRAC``
        when segmenting on a pre-cropped block image. Defaults to crop size.
    """
    if role == "slide":
        return _stain_score_masked_otsu(
            bgr_image, slide_close_ksize=slide_close_ksize, pixel_scale=pixel_scale
        )
    if role == "block":
        if BLOCK_SEGMENTER == "qupath":
            return _postprocess_block_classifier_mask(
                _load_block_qupath(BLOCK_QUPATH_MODEL_PATH).predict(bgr_image),
                area_reference=block_area_reference,
                pixel_scale=pixel_scale,
            )
        if BLOCK_SEGMENTER == "rtrees":
            return _load_block_rtrees(
                BLOCK_RTREES_MODEL_PATH, BLOCK_RTREES_SIDECAR_PATH
            ).predict(bgr_image)
        if BLOCK_SEGMENTER != "classical":
            raise ValueError(f"unknown block segmenter: {BLOCK_SEGMENTER!r}")
        return _brown_hsv_block_mask(
            bgr_image,
            window=block_window,
            area_reference=block_area_reference,
            pixel_scale=pixel_scale,
        )
    raise ValueError(f"unknown role for segmentation: {role!r}")


def active_segmentation_backend(role: str) -> str:
    """Return the configured backend name recorded with prepared specimens."""
    if role == "block":
        return BLOCK_SEGMENTER
    if role == "slide":
        return "classical"
    raise ValueError(f"unknown role for segmentation: {role!r}")


@lru_cache(maxsize=1)
def _load_block_rtrees(model_path: object, sidecar_path: object) -> RtreesSegmenter:
    """Load the fixed block artifact once per process."""
    return load_rtrees_segmenter(str(model_path), str(sidecar_path))


@lru_cache(maxsize=1)
def _load_block_qupath(model_path: object) -> RtreesSegmenter:
    """Load the fixed native QuPath artifact once per process."""
    return load_qupath_rtrees_segmenter(str(model_path))


def _stain_score_masked_otsu(
    bgr_image: np.ndarray,
    *,
    slide_close_ksize: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """H&E stain score + hue/sat/texture/Lab gates (no label ROI)."""
    b, g, r = bgr_image[:, :, 0], bgr_image[:, :, 1], bgr_image[:, :, 2]
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    score = np.clip(
        np.maximum(r.astype(np.int16), b.astype(np.int16)) - g.astype(np.int16)
        + 2 * np.maximum(A.astype(np.int16) - 128, 0),
        0, 255,
    ).astype(np.uint8)

    t_val, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(SLIDE_OTSU_T_MIN, min(int(t_val), SLIDE_OTSU_T_MAX))

    gf = g.astype(np.float32)
    tw = scale_odd_length(SLIDE_TEXTURE_WINDOW, pixel_scale)
    win = (tw, tw)
    mean = cv2.boxFilter(gf, -1, win)
    mean_sq = cv2.boxFilter(gf * gf, -1, win)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    hue_gate = (h <= 12) | (h >= 145) | ((h >= 118) & (h <= 168))
    mask = (
        hue_gate &
        (s >= SLIDE_SAT_FLOOR) &
        (texture >= SLIDE_TEXTURE_FLOOR) &
        (score >= max(SLIDE_SCORE_FLOOR, min(t, SLIDE_SCORE_CEILING))) &
        (A >= SLIDE_LAB_A_MIN) &
        (v <= SLIDE_VAL_MAX)
    ).astype(np.uint8) * 255

    h_img, w_img = mask.shape
    min_area = max(
        scale_area_min(MIN_SLIDE_COMPONENT_AREA, pixel_scale),
        int(h_img * w_img * MIN_AREA_FRACTION),
    )
    effective_ksize = (
        slide_close_ksize
        if slide_close_ksize is not None
        else scale_odd_length(SLIDE_CLOSE_KERNEL, pixel_scale)
    )
    result = _postprocess(
        mask,
        min_area=min_area,
        role="slide",
        close_ksize=effective_ksize,
    )
    if SLIDE_IRIDESCENCE_FILTER_ENABLED:
        # `score` is the per-pixel H&E stain map (same formula the guard uses).
        result = _remove_green_iridescence(
            result, bgr_image, score, pixel_scale=pixel_scale
        )
    return result


def _brown_hsv_block_mask(
    bgr_image: np.ndarray,
    *,
    window: tuple[int, int, int, int] | None = None,
    area_reference: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Brown/yellow HSV + Lab gate, plus a window-scoped dark-absorbance gate.

    Dense tissue absorbs nearly all transilluminated light, so its core is
    near-black and carries no chroma -- the brown/yellow gates miss it. The
    dark gate recovers it: inside the cassette window, any pixel darker than
    BLOCK_DARK_TISSUE_VAL_MAX is tissue. Scoping to `window` keeps the dark
    cassette walls out. No cassette ROI clip here; preparation clips after.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    brown_hsv = (
        (h >= BLOCK_BROWN_HUE_MIN) & (h <= BLOCK_BROWN_HUE_MAX)
        & (s >= BLOCK_SAT_MIN) & (v <= BLOCK_BROWN_VAL_MAX)
    )
    lab_brown = (A >= BLOCK_LAB_A_MIN) & (B >= BLOCK_LAB_B_MIN) & (L <= BLOCK_LAB_L_MAX)
    dark_yellow = (
        (h >= BLOCK_DARK_YELLOW_HUE_MIN) & (h <= BLOCK_DARK_YELLOW_HUE_MAX)
        & (s >= BLOCK_DARK_YELLOW_SAT_MIN) & (v <= BLOCK_DARK_YELLOW_VAL_MAX)
    )

    dark_tissue = np.zeros(v.shape, dtype=bool)
    if window is not None:
        x, y, w, h_win = window
        inside = np.zeros(v.shape, dtype=bool)
        inside[y:y + h_win, x:x + w] = True
        dark_tissue = inside & (v <= BLOCK_DARK_TISSUE_VAL_MAX)

    mask = (brown_hsv | (lab_brown & dark_yellow) | dark_tissue).astype(np.uint8) * 255

    if window is not None:
        mask = _grow_tissue_halo(mask, h, s, B, window, pixel_scale=pixel_scale)

    h_img, w_img = mask.shape
    min_area = max(
        scale_area_min(MIN_BLOCK_COMPONENT_AREA, pixel_scale),
        int(h_img * w_img * MIN_AREA_FRACTION),
    )
    return _postprocess(
        mask,
        min_area=min_area,
        role="block",
        area_reference=area_reference,
        pixel_scale=pixel_scale,
    )


def _grow_tissue_halo(
    seed_mask: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    B: np.ndarray,
    window: tuple[int, int, int, int],
    *,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Grow the confident tissue mask into its translucent olive halo.

    Tissue is bright yellow only at its dense core; its thinner edges form a
    translucent olive/yellow-green halo that blends continuously into the green
    wax, so the chroma gates (hue <= 45) drop it. This is a hysteresis grow:
    `seed_mask` is the strong threshold (confident tissue), and the "weak" band
    is window-scoped pixels that are yellower than pure wax (hue below the wax
    floor, high Lab-b, moderate saturation). Weak pixels are kept only where they
    form a connected component that touches a seed. A wax-only block has no seed,
    so nothing grows. Purely additive: every seed pixel is preserved, so this
    cannot conflict with or erase any other gate's output.
    """
    x, y, w, h_win = window
    inside = np.zeros(seed_mask.shape, dtype=bool)
    inside[y:y + h_win, x:x + w] = True
    seed = seed_mask > 0

    # On blue-paraffin blocks the olive rim sits below the default Lab-b floor and
    # runs to greener hues; relax the gate there only (blue wax is chromatically
    # opposite the tissue, so no wax leak). Purple/pale blocks keep the strict gate.
    if _is_blue_paraffin_wax(seed, s, B, inside, pixel_scale=pixel_scale):
        hue_max, lab_b_min = BLOCK_HALO_GROW_BLUE_HUE_MAX, BLOCK_HALO_GROW_BLUE_LAB_B_MIN
    else:
        hue_max, lab_b_min = BLOCK_HALO_GROW_HUE_MAX, BLOCK_HALO_GROW_LAB_B_MIN

    weak = (
        inside
        & (h <= hue_max)
        & (B >= lab_b_min)
        & (s >= BLOCK_HALO_GROW_SAT_MIN)
    )
    union = (weak | seed).astype(np.uint8)
    num, labels = cv2.connectedComponents(union, connectivity=8)
    seed_labels = set(np.unique(labels[seed]).tolist())
    seed_labels.discard(0)
    if not seed_labels:
        return seed_mask
    keep = np.isin(labels, list(seed_labels))
    return (keep.astype(np.uint8)) * 255


def _is_blue_paraffin_wax(
    seed: np.ndarray,
    s: np.ndarray,
    B: np.ndarray,
    inside: np.ndarray,
    *,
    pixel_scale: float = 1.0,
) -> bool:
    """True when the block's paraffin wax reads blue (high sat AND low Lab-b).

    The wax region is the cassette-window interior minus the tissue seed dilated
    by BLUE_PARAFFIN_SEED_DILATE_PX (so tissue and its halo do not bias the
    medians). Blue paraffin is uniquely high-saturation and low-Lab-b; across the
    41-block library this fires on only the blue-wax (brain) blocks with a ~19-
    point margin on each axis. Purple wax (low sat) and pale wax (low sat) fail
    the saturation clause; yellow/orange wax fails the Lab-b clause. When the
    window has almost no wax (tissue fills it), it is treated as not-blue.
    """
    kernel = np.ones((3, 3), np.uint8)
    grown_seed = cv2.dilate(
        seed.astype(np.uint8),
        kernel,
        iterations=scale_reach(BLUE_PARAFFIN_SEED_DILATE_PX, pixel_scale),
    ) > 0
    wax = inside & ~grown_seed
    if int(wax.sum()) < scale_area_min(1000, pixel_scale):
        return False
    return (
        float(np.median(s[wax])) >= BLUE_PARAFFIN_WAX_SAT_MIN
        and float(np.median(B[wax])) <= BLUE_PARAFFIN_WAX_LAB_B_MAX
    )


def _slide_relative_area_floor(areas: list[int]) -> int:
    """Minimum kept area from the largest fragment on this slide.

    See SLIDE_MIN_COMPONENT_REL_AREA in constants.py (esophagus / sparse-tissue
    caution for future editors).
    """
    if not areas:
        return 0
    return max(1, int(max(areas) * SLIDE_MIN_COMPONENT_REL_AREA))


def _block_relative_area_floor(areas: list[int]) -> int:
    """Minimum kept area from the largest fragment on this block.

    Mirrors _slide_relative_area_floor but reads the decoupled
    BLOCK_MIN_COMPONENT_REL_AREA. At 0.0 the relative floor is disabled and only
    the absolute floor + block shape gates apply (see constants.py note).
    """
    if not areas or BLOCK_MIN_COMPONENT_REL_AREA <= 0.0:
        return 0
    return max(1, int(max(areas) * BLOCK_MIN_COMPONENT_REL_AREA))


def _component_filter(
    mask: np.ndarray,
    *,
    min_area: int,
    role: str = "unknown",
    area_reference: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Drop dust; slide keeps fragments >= rel floor of largest blob."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    slide_candidates: list[tuple[int, int]] = []
    block_candidates: list[tuple[int, int]] = []
    block_area_denom = area_reference if area_reference is not None else mask.size

    for i in range(1, num):
        x, y, w, h, area = [int(stats[i, j]) for j in range(5)]
        if area < scale_area_min(DUST_COMPONENT_AREA, pixel_scale):
            continue
        if role == "block":
            if area > block_area_denom * BLOCK_AREA_MAX_FRAC:
                continue
            if area >= min_area:
                block_candidates.append((i, area))
        else:
            # Curved, narrow tissue has low bounding-box fill even when it has
            # substantial area. Slides rely on the absolute and relative area
            # floors below rather than this geometry-dependent proxy.
            slide_candidates.append((i, area))

    if role == "block":
        # Symmetric with the slide relative floor (issue #78): drop speckle below
        # SLIDE_MIN_COMPONENT_REL_AREA (1%) of the largest block component so
        # blocks stop carrying tiny components their slide lacks. Shape gates
        # (fill / aspect / area-max) above are block-specific and kept.
        floor = max(
            min_area,
            _block_relative_area_floor([area for _, area in block_candidates]),
        )
        for i, area in block_candidates:
            if area >= floor:
                clean[labels == i] = 255

    if role == "slide":
        rel_floor = _slide_relative_area_floor([area for _, area in slide_candidates])
        if slide_candidates:
            largest_area = max(area for _, area in slide_candidates)
            abs_floor = (
                MIN_SLIDE_COMPONENT_AREA
                if largest_area >= MIN_SLIDE_COMPONENT_AREA
                else min_area
            )
        else:
            abs_floor = min_area
        floor = max(rel_floor, abs_floor)
        for i, area in slide_candidates:
            if area >= floor:
                clean[labels == i] = 255

    return clean


def _remove_green_iridescence(
    mask: np.ndarray,
    bgr_image: np.ndarray,
    stain: np.ndarray,
    *,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Drop coverslip/mountant rainbow iridescence from a slide mask.

    A kept connected component is removed only when BOTH hold:
      * >= SLIDE_IRIDESCENCE_NEAR_GREEN_FRAC of its pixels lie within
        SLIDE_IRIDESCENCE_REACH_PX of a green-dominant pixel, and
      * its own median stain score < SLIDE_IRIDESCENCE_STAIN_GUARD.

    Green-dominant pixels (g > r+margin and g > b+margin) cannot be H&E stain,
    so they mark rainbow iridescence. The stain guard spares real tissue that
    merely sits near the band: iridescent fringe is pale, real tissue is
    strongly stained. `stain` is the per-pixel H&E stain map (uint8).
    """
    b, g, r = (
        bgr_image[:, :, 0].astype(np.int16),
        bgr_image[:, :, 1].astype(np.int16),
        bgr_image[:, :, 2].astype(np.int16),
    )
    green_dom = (
        (g > r + SLIDE_IRIDESCENCE_GREEN_MARGIN)
        & (g > b + SLIDE_IRIDESCENCE_GREEN_MARGIN)
    ).astype(np.uint8)
    reach = scale_reach(SLIDE_IRIDESCENCE_REACH_PX, pixel_scale)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    near_green = cv2.dilate(green_dom, se) > 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    for i in range(1, num):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        near_frac = (comp & near_green).sum() / max(area, 1)
        if near_frac < SLIDE_IRIDESCENCE_NEAR_GREEN_FRAC:
            continue
        if int(np.median(stain[comp])) >= SLIDE_IRIDESCENCE_STAIN_GUARD:
            continue  # strongly stained => real tissue, keep
        out[comp] = 0
    return out


def _component_area_floor(
    stats: np.ndarray,
    num_components: int,
    min_area: int,
    role: str,
) -> int:
    """Return the role-aware component area floor for post-threshold cleanup."""
    if role != "slide" or num_components <= 1:
        return min_area
    largest_area = max(
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, num_components)
    )
    relative_floor = int(largest_area * SLIDE_MIN_COMPONENT_REL_AREA)
    return max(min_area, relative_floor)


def clean_tissue_components(
    mask: np.ndarray, role: str, *, pixel_scale: float = 1.0
) -> np.ndarray:
    """Apply role-aware connected-component cleanup without morphology."""
    h_img, w_img = mask.shape
    if role == "slide":
        min_area = max(MIN_SLIDE_COMPONENT_AREA, int(h_img * w_img * MIN_AREA_FRACTION))
        return _component_filter(
            mask,
            min_area=min_area,
            role=role,
        )
    if role == "block":
        min_area = max(
            scale_area_min(MIN_BLOCK_COMPONENT_AREA, pixel_scale),
            int(h_img * w_img * MIN_AREA_FRACTION),
        )
        return _component_filter(
            mask, min_area=min_area, role=role, pixel_scale=pixel_scale
        )
    raise ValueError(f"unknown role for component cleanup: {role!r}")


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by redrawing external contours solid."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out


def _postprocess(
    mask: np.ndarray,
    *,
    min_area: int,
    role: str,
    close_ksize: int = 5,
    area_reference: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Open, ellipse close, component filter, hole fill."""
    open_length = scale_odd_length(3, pixel_scale)
    close_length = scale_odd_length(close_ksize, pixel_scale)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_length, open_length))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_length, close_length)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    mask = _component_filter(
        mask,
        min_area=min_area,
        role=role,
        area_reference=area_reference,
        pixel_scale=pixel_scale,
    )
    return _fill_holes(mask)


def _postprocess_block_classifier_mask(
    mask: np.ndarray,
    *,
    area_reference: int | None = None,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Morphologically clean a classifier mask without classical shape gates."""
    del area_reference
    open_length = scale_odd_length(3, pixel_scale)
    close_length = scale_odd_length(5, pixel_scale)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_length, open_length)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_length, close_length)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return _fill_holes(mask)
