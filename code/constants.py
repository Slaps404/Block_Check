"""Central tunable constants for the v2 claimed-pair pipeline.

Add new MVP tuning knobs here (not inline in consumer modules).
Log changes in docs/mvp_tuning_log.md.

Pipeline order: preparation (label mask / ROI crop → segment → clip) → gates → scorer.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Preparation  (preparation.py)
# ---------------------------------------------------------------------------

MIN_CONTOUR_AREA: int = 100
MAX_ASPECT_RATIO: float = 10.0
MAX_MASK_COVERAGE: float = 0.95
# Outer band zeroed on slide masks; see mvp_tuning_log 002.
SLIDE_BORDER_INSET_FRAC: float = 0.07
# Opposite-tag artifact dots removed by spatial CC filter after segmentation.
SLIDE_OPPOSITE_TAG_ARTIFACT_MAX_AREA: int = 2500
SLIDE_OPPOSITE_TAG_SIDE_START_FRAC: float = 0.75
SLIDE_OPPOSITE_TAG_BAND_PAD_FRAC: float = 0.04

# ---------------------------------------------------------------------------
# 2. Slide boundary  (slide_boundary.py — pre-seg crop in preparation.py)
# ---------------------------------------------------------------------------

SLIDE_CROP_TAG_INSET_FRAC: float = 0.30   # tag end, fraction of slide length
SLIDE_CROP_FAR_INSET_FRAC: float = 0.10   # opposite end, fraction of slide length
SLIDE_CROP_WIDTH_INSET_FRAC: float = 0.02  # both short edges, fraction of width

# ---------------------------------------------------------------------------
# 3. Gates  (gates.py)
# ---------------------------------------------------------------------------

MIN_MASK_COVERAGE_FRAC: float = 0.00015
PAIR_SLIVER_COVERAGE_FRAC: float = 0.005
SLIVER_ASPECT_RATIO: float = 25.0

# ---------------------------------------------------------------------------
# 4. Block ROI  (block_roi_mask.py)
# ---------------------------------------------------------------------------

BLOCK_CLOSE_KERNEL: int = 25
BLOCK_INSET_FRAC: float = 0.13
# CC inside-fraction clip threshold; see tuning iteration 031.
BLOCK_CLIP_MIN_INSIDE_FRAC: float = 0.5
# Pre-seg crop buffer beyond cassette window; see tuning iteration 031.
BLOCK_SEG_BUFFER_FRAC: float = 0.20

# ---------------------------------------------------------------------------
# 5. Segmentation — block  (segmentation.py, role="block")
# ---------------------------------------------------------------------------

# Keep the deployed classical path active; QuPath RTrees remains available for
# explicit diagnostics until it has separate promotion evidence.
BLOCK_SEGMENTER: str = "classical"
_REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_QUPATH_MODEL_PATH = (
    _REPO_ROOT / "models" / "QuPath" / "block_tissue_v001_g-wg-gm_s1-2-8_t25.json"
)
BLOCK_RTREES_MODEL_PATH = _REPO_ROOT / "models" / "retrained" / "block_rtrees.yml.gz"
BLOCK_RTREES_SIDECAR_PATH = _REPO_ROOT / "models" / "retrained" / "block_rtrees.recipe.json"

# Brown / yellow chroma gate
BLOCK_BROWN_HUE_MIN: int = 5
BLOCK_BROWN_HUE_MAX: int = 45
BLOCK_BROWN_VAL_MAX: int = 245
BLOCK_SAT_MIN: int = 20
BLOCK_LAB_A_MIN: int = 124
BLOCK_LAB_B_MIN: int = 132
BLOCK_LAB_L_MAX: int = 238
BLOCK_DARK_YELLOW_HUE_MIN: int = 8
BLOCK_DARK_YELLOW_HUE_MAX: int = 55
BLOCK_DARK_YELLOW_SAT_MIN: int = 14
BLOCK_DARK_YELLOW_VAL_MAX: int = 230

# Near-black dense tissue (HSV value); see docs/mvp_tuning_log/028.
BLOCK_DARK_TISSUE_VAL_MAX: int = 40

# Seed-anchored halo grow past hue-45 cutoff; see docs/mvp_tuning_log/029.
BLOCK_HALO_GROW_HUE_MAX: int = 60
BLOCK_HALO_GROW_LAB_B_MIN: int = 170
BLOCK_HALO_GROW_SAT_MIN: int = 60

# Relaxed halo gate for blue-paraffin blocks only; see docs/mvp_tuning_log § blue-paraffin halo.
BLOCK_HALO_GROW_BLUE_HUE_MAX: int = 95
BLOCK_HALO_GROW_BLUE_LAB_B_MIN: int = 115

# Blue-paraffin wax detector thresholds; see docs/mvp_tuning_log § blue-paraffin halo.
BLUE_PARAFFIN_WAX_SAT_MIN: int = 165
BLUE_PARAFFIN_WAX_LAB_B_MAX: int = 100
BLUE_PARAFFIN_SEED_DILATE_PX: int = 30

# Topology-preserving growth predicates; see tuning iteration 037.
BLOCK_GROW_SELECTIVE_HUE_MAX: int = 12
BLOCK_GROW_SELECTIVE_SAT_MIN: int = 100
BLOCK_GROW_SELECTIVE_VALUE_MAX: int = 100
BLOCK_GROW_LOOSE_VALUE_MAX: int = 60

# Component size gates
BLOCK_AREA_MAX_FRAC: float = 0.04
MIN_BLOCK_COMPONENT_AREA: int = 500
# Relative speck floor vs largest block blob; see docs/mvp_tuning_log/036.
BLOCK_MIN_COMPONENT_REL_AREA: float = 0.003

# ---------------------------------------------------------------------------
# 6. Segmentation — shared  (segmentation.py, both roles)
# ---------------------------------------------------------------------------

DUST_COMPONENT_AREA: int = 18
MIN_AREA_FRACTION: float = 0.000008

# ---------------------------------------------------------------------------
# 7. Segmentation — slide  (segmentation.py, role="slide")
# ---------------------------------------------------------------------------

SEGMENTATION_REFERENCE_WIDTH: int = 4056  # px width the slide spatial controls were tuned at

# Component size gates
MIN_SLIDE_COMPONENT_AREA: int = 450   # absolute px floor; see mvp_tuning_log/027
# Curved, narrow tissue has low bounding-box fill despite substantial area.
# Keep the dust guard while retaining the live 53910454 esophagus strands.
SLIDE_FILL_MIN: float = 0.03
# Relative speck floor; see docs/mvp_tuning_log/024_slide_relative_component_floor.md.
SLIDE_MIN_COMPONENT_REL_AREA: float = 0.01
SLIDE_SMALL_FILL_MIN: float = 0.18
SLIDE_CLOSE_KERNEL: int = 5

# H&E stain gate (iterations 010, 017)
SLIDE_SAT_FLOOR: int = 12
SLIDE_TEXTURE_WINDOW: int = 9
SLIDE_TEXTURE_FLOOR: float = 1.5
SLIDE_OTSU_T_MIN: int = 10
SLIDE_OTSU_T_MAX: int = 45
SLIDE_SCORE_FLOOR: int = 12
SLIDE_SCORE_CEILING: int = 14
SLIDE_LAB_A_MIN: int = 130
SLIDE_VAL_MAX: int = 252

# Coverslip rainbow / green-iridescence filter (iteration 026)
SLIDE_IRIDESCENCE_FILTER_ENABLED: bool = True
SLIDE_IRIDESCENCE_GREEN_MARGIN: int = 3       # g exceeds r and b by this much
SLIDE_IRIDESCENCE_REACH_PX: int = 15          # dilation radius around green px
SLIDE_IRIDESCENCE_NEAR_GREEN_FRAC: float = 0.35
SLIDE_IRIDESCENCE_STAIN_GUARD: int = 40       # keep if median stain >= this

# ---------------------------------------------------------------------------
# 8. Slide label mask  (slide_label_mask.py)
# ---------------------------------------------------------------------------

BLACK_BG_THRESH: int = 40
AREA_MIN_FRAC: float = 0.015
AREA_MAX_FRAC: float = 0.20  # landscape Pi labels can reach ~9.5% of frame
RTY_MIN: float = 0.70
SPAN_MIN: float = 0.15
SPAN_MAX: float = 0.55
CY_BAND: float = 0.45  # label center must be in outer vertical band
CX_BAND: float = 0.45  # label center must be in outer horizontal band
MARGIN_SCALE: float = 1.25
BLUE_LOW: int = 100
LABEL_COMBINED_MAX_FRAC: float = 0.40
LABEL_YELLOW_G_MIN: int = 150
LABEL_YELLOW_R_MIN: int = 150
LABEL_MORPH_FRAC: float = 0.018

# ---------------------------------------------------------------------------
# 9. Scorer  (scorer.py, locked_alignment.py)
# ---------------------------------------------------------------------------

# RMS-grid dense/sparse router threshold; see tuning iterations 033 and 048
SHAPE_ROUTER_SIZE_THRESHOLD: float = 0.0194
# PASS / REVIEW decision (uncalibrated placeholder, do not use for production yet)
PASS_THRESHOLD: float = 0.85
# Open-retrieval work-order margin: min lead top block needs over runner-up
# to count as a clear win rather than an ambiguous near-miss; see ADR-0009.
MATCH_MARGIN: float = 0.02

# Point-layout cost matrix blending weights (rough estimates, not tuned —
# likely overfit to current dataset; revisit with a larger labeled set).
POINT_LAYOUT_AREA_WEIGHT: float = 0.25
POINT_LAYOUT_SHAPE_WEIGHT: float = 0.10

# ---------------------------------------------------------------------------
# 10. Live capture profiling (--profile, #168)
# ---------------------------------------------------------------------------

# Total-capture-ms threshold above which the profile console block marks a
# capture [SLOW] (uncalibrated placeholder, no field data yet -- see #168).
PROFILE_SLOW_CAPTURE_MS: int = 3000

# Native Pi sensor/capture contract. This is deliberately separate from the
# role-specific stored-capture policy below. (width, height)
NATIVE_CAPTURE_DIMENSIONS: tuple[int, int] = (4056, 3040)

# Per-role capture/store dimensions (ADR 0013). Blocks stay at the native
# sensor dimensions. Slides use the qualified half-resolution policy. This is
# not the native sensor contract. (width, height)
CAPTURE_DIMENSIONS: dict[str, tuple[int, int]] = {
    "block": NATIVE_CAPTURE_DIMENSIONS,
    "slide": (2028, 1520),
}

# ---------------------------------------------------------------------------
# 11. Live capture motion sampling (`motion` console command, #169)
# ---------------------------------------------------------------------------

# Duration the `motion` command polls sampled motion scores from the running
# background camera loop before summarizing them.
MOTION_SAMPLE_WINDOW_S: float = 3.0

# ---------------------------------------------------------------------------
# 12. Live capture settling and telemetry
# ---------------------------------------------------------------------------

# Consecutive preview classifications required before a presence, absence, or
# motion change is treated as real. At the Pi's 10 fps preview rate this is
# roughly 200 ms; tune only from profiled hardware evidence (issue #211).
SETTLING_CONFIRMATION_FRAMES: int = 2

# Periodic flush cadence for `MotionCurveWriter`'s Pi-local motion_curve.csv.
# Bounds a mid-session power loss to at most this many seconds of buffered
# per-frame motion rows, on top of whatever explicit state-change/capture
# flushes already happened.
MOTION_CURVE_FLUSH_INTERVAL_S: float = 2.0
