"""Grid deskew: warp a still with a homography fitted from a known dot grid."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class DeskewConfigError(RuntimeError):
    """The saved deskew config is missing, malformed, or wrong for this still."""


@dataclass(frozen=True)
class DeskewConfig:
    homography: np.ndarray
    output_size: tuple[int, int]
    source_size: tuple[int, int]
    px_per_mm: float
    raw: dict


def load_deskew_config(path: str | Path) -> DeskewConfig:
    path = Path(path)
    if not path.exists():
        raise DeskewConfigError(
            f"No deskew config at {path}. Fit one from a grid photo with the "
            "grid-deskew skill, then copy it to this path."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        homography = np.asarray(raw["homography"], dtype=np.float64)
        output_size = tuple(int(v) for v in raw["output_size"])
        source_size = tuple(int(v) for v in raw["source_size"])
        px_per_mm = float(raw["px_per_mm"])
    except (ValueError, KeyError, TypeError) as exc:
        raise DeskewConfigError(f"Bad deskew config {path}: {exc}") from exc
    if homography.shape != (3, 3):
        raise DeskewConfigError(f"Deskew homography must be 3x3, got {homography.shape}")
    if len(output_size) != 2 or len(source_size) != 2:
        raise DeskewConfigError("output_size and source_size must be [width, height]")
    return DeskewConfig(homography, output_size, source_size, px_per_mm, raw)


def apply_deskew(image: np.ndarray, config: DeskewConfig) -> np.ndarray:
    height, width = image.shape[:2]
    if (width, height) != config.source_size:
        fitted_w, fitted_h = config.source_size
        raise DeskewConfigError(
            f"Still is {width}x{height} but the deskew config was fitted at "
            f"{fitted_w}x{fitted_h}; refit at this resolution."
        )
    return cv2.warpPerspective(image, config.homography, config.output_size)


def deskew_png(source: str | Path, dest: str | Path, config: DeskewConfig) -> None:
    """Read a still from disk, warp it, and write the result."""
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read still {source}")
    if not cv2.imwrite(str(dest), apply_deskew(image, config)):
        raise RuntimeError(f"could not write {dest}")
