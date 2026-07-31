from pathlib import Path

import numpy as np
import pytest

from tools.surface_reference.reference_artifact import (
    ReferenceArtifactError,
    build_reference_rgba,
    write_reference_png,
)


def test_transparent_background() -> None:
    rgba = build_reference_rgba(np.array([[False, True]]), (0, 255, 255))

    assert rgba.tolist() == [[[0, 255, 255, 0], [0, 255, 255, 255]]]


def test_empty_mask_creates_no_file(tmp_path: Path) -> None:
    output = tmp_path / "ref.png"

    with pytest.raises(ReferenceArtifactError, match="empty tissue mask"):
        write_reference_png(np.zeros((2, 2), bool), output)

    assert not output.exists()
