"""Grid deskew: load a saved homography config and warp a still with it."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import pi_session as run_pi_session
from deskew import DeskewConfigError, apply_deskew, load_deskew_config


def _config(**overrides):
    config = {
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "output_size": [60, 40],
        "source_size": [120, 100],
        "px_per_mm": 50.0,
        "grid": "20x20 @ 0.8mm",
        "matched": 400,
        "expected": 400,
        "residual_px": {"median": 0.5, "p95": 1.2, "max": 2.0},
        "source_image": "board.png",
        "fitted_at": "2026-10-13T00:00:00Z",
    }
    config.update(overrides)
    return config


def test_load_missing_config_points_to_grid_deskew_skill(tmp_path):
    with pytest.raises(DeskewConfigError, match="grid-deskew"):
        load_deskew_config(tmp_path / "nope.json")


def test_load_rejects_non_3x3_homography(tmp_path):
    path = tmp_path / "deskew.json"
    path.write_text(json.dumps(_config(homography=[[1, 0], [0, 1]])))
    with pytest.raises(DeskewConfigError, match="3x3"):
        load_deskew_config(path)


def test_apply_warps_to_output_size(tmp_path):
    path = tmp_path / "deskew.json"
    # Shift right by 10 px: source pixel (x, y) lands at (x + 10, y).
    path.write_text(json.dumps(_config(homography=[[1, 0, 10], [0, 1, 0], [0, 0, 1]])))
    config = load_deskew_config(path)
    image = np.zeros((100, 120), dtype=np.uint8)
    image[5, 5] = 255

    out = apply_deskew(image, config)

    assert out.shape == (40, 60)
    assert out[5, 15] == 255
    assert config.px_per_mm == 50.0


def test_apply_refuses_wrong_source_size(tmp_path):
    path = tmp_path / "deskew.json"
    path.write_text(json.dumps(_config()))
    config = load_deskew_config(path)
    with pytest.raises(DeskewConfigError, match="120x100"):
        apply_deskew(np.zeros((50, 60), dtype=np.uint8), config)


def test_build_arg_parser_deskew_config_defaults_to_pi_local_path():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    )
    assert args.deskew_config.endswith("deskew_grid.json")
    override = parser.parse_args(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1",
         "--deskew-config", "/tmp/fit.json"]
    )
    assert override.deskew_config == "/tmp/fit.json"
