"""Regression tests for the active segmentation contact-sheet tool."""

import cv2
import numpy as np
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_seg_contact_sheets import make_normalized_component_view


def test_normalized_panel_uses_production_256_grid_and_preserves_component_count():
    mask = np.zeros((300, 500), dtype=np.uint8)
    cv2.circle(mask, (100, 150), 20, 255, -1)
    cv2.rectangle(mask, (350, 120), (390, 180), 255, -1)

    panel, count = make_normalized_component_view(mask)

    assert panel.shape == (256, 256, 3)
    assert count == 2
    assert np.count_nonzero(panel) > 0
