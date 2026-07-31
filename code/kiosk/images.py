"""OpenCV JPEG encode helpers shared by kiosk still routes (#137 / #139)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_STILL_MAX_LONG_EDGE = 1920
DEFAULT_JPEG_QUALITY = 85
CLAIM_THUMB_MAX_LONG_EDGE = 160
CLAIM_DISPLAY_MAX_LONG_EDGE = 960


def downscale_for_display(
    image: np.ndarray, *, max_long_edge: int = DEFAULT_STILL_MAX_LONG_EDGE
) -> np.ndarray:
    """Shrink a BGR still so its longest edge fits the kiosk stage."""
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_jpeg(
    image: np.ndarray, *, quality: int = DEFAULT_JPEG_QUALITY
) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("JPEG encode failed")
    return encoded.tobytes()


def encode_downscaled_jpeg(
    image: np.ndarray,
    *,
    max_long_edge: int,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Downscale then JPEG-encode a BGR still for claim-result evidence."""
    scaled = downscale_for_display(image, max_long_edge=max_long_edge)
    return encode_image_jpeg(scaled, quality=quality)


def encode_still_jpeg(
    path: str | Path,
    *,
    max_long_edge: int = DEFAULT_STILL_MAX_LONG_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Read a published PNG still and return a downscaled JPEG for display."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read still: {path}")
    scaled = downscale_for_display(image, max_long_edge=max_long_edge)
    return encode_image_jpeg(scaled, quality=quality)


def encode_preview_jpeg(
    frame: np.ndarray,
    *,
    max_long_edge: int = DEFAULT_STILL_MAX_LONG_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Encode a live preview frame for kiosk display (MJPEG-ready seam)."""
    scaled = downscale_for_display(frame, max_long_edge=max_long_edge)
    return encode_image_jpeg(scaled, quality=quality)
