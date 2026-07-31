from pathlib import Path

import pytest

from tools.surface_reference.pairing import PairingError, resolve_paired_slide


def test_resolves_current_pair(tmp_path: Path) -> None:
    block = tmp_path / "block_041_brain_NAIVE_01_HE.png"
    slide = tmp_path / "slide_041_brain_NAIVE_01_HE.png"
    block.touch()
    slide.touch()

    assert resolve_paired_slide(block) == slide


def test_rejects_ambiguous_legacy_pair(tmp_path: Path) -> None:
    block = tmp_path / "set_05_block_lung_HE.jpg"
    block.touch()
    (tmp_path / "set_05_slide_lung_HE.jpg").touch()
    (tmp_path / "set_05_slide_lung_PAS.jpg").touch()

    with pytest.raises(PairingError, match="2 matching slides"):
        resolve_paired_slide(block)


def test_resolves_capture_claim_pair(tmp_path: Path) -> None:
    block = tmp_path / "capture_000132_block_53524010_20260718T025929Z.png"
    slide = tmp_path / "slide_53524010_20260721T185701Z_a1b2c3d4e5f6.png"
    block.touch()
    slide.touch()

    assert resolve_paired_slide(block) == slide
