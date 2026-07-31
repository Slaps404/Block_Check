from verify.scale import (
    block_pixel_scale_for,
    pixel_scale_for,
    scale_area_max,
    scale_area_min,
    scale_odd_length,
    scale_reach,
)


def test_scale_is_identity_at_full_resolution():
    assert pixel_scale_for(4056) == 1.0
    assert scale_odd_length(5, 1.0) == 5
    assert scale_reach(15, 1.0) == 15
    assert scale_area_min(450, 1.0) == 450
    assert scale_area_max(2500, 1.0) == 2500


def test_half_resolution_matches_experiment_rounding():
    s = pixel_scale_for(2028)
    assert s == 0.5
    # lengths: round(v*s) then force odd, min 1
    assert scale_odd_length(5, s) == 3       # round(2.5)=2 -> +1
    assert scale_odd_length(9, s) == 5       # round(4.5)=4 -> +1
    # reach: round(v*s), not forced odd
    assert scale_reach(15, s) == 8           # round(7.5)=8
    # areas: ceil(v*s^2) for floors, floor(v*s^2) for caps
    assert scale_area_min(450, s) == 113     # ceil(112.5)
    assert scale_area_min(18, s) == 5        # ceil(4.5)
    assert scale_area_max(2500, s) == 625    # floor(625.0)


def test_lengths_never_drop_below_one():
    assert scale_odd_length(1, 0.25) == 1
    assert scale_area_min(1, 0.25) == 1


def test_scale_is_clamped_below_half_resolution():
    assert pixel_scale_for(600) == 0.5
    assert pixel_scale_for(800) == 0.5
    assert pixel_scale_for(200) == 0.5
    assert pixel_scale_for(9999) == 1.0


def test_block_scale_supports_four_times_downscaled_capture():
    """Block spatial controls retain their physical size at 4x downscale."""
    scale = block_pixel_scale_for(4056 // 4)

    assert scale == 0.25
    assert scale_odd_length(25, scale) == 7
    assert scale_reach(30, scale) == 8
    assert scale_area_min(500, scale) == 32
    assert scale_area_min(18, scale) == 2
