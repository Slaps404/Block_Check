"""Focus aid: a sharpness score for the center of a camera frame."""
from __future__ import annotations

import cv2
import numpy as np


def center_roi_sharpness(frame: np.ndarray, roi_fraction: float = 0.2) -> float:
    """Variance of the Laplacian over a centered box; higher means sharper."""
    if not 0 < roi_fraction <= 1:
        raise ValueError("roi_fraction must be in (0, 1]")
    height, width = frame.shape[:2]
    roi_h = max(1, round(height * roi_fraction))
    roi_w = max(1, round(width * roi_fraction))
    top = (height - roi_h) // 2
    left = (width - roi_w) // 2
    roi = frame[top:top + roi_h, left:left + roi_w]
    if roi.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if roi.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        roi = cv2.cvtColor(roi, code)
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())
