"""Parse capture filename conventions shared by capture tools and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DatasetFilenameMetadata:
    """Metadata encoded by a legacy or compact dataset filename."""

    style: str
    role: str
    set_number: str
    capture_index: str | None
    tissue: str
    stain: str
    genotype: str
    workorder: str
    slide_no: str


_LEGACY_RE = re.compile(
    r"^set_(?P<set>\d+)_(?P<role>block_silhouette|block|slide)_"
    r"(?:capture_(?P<capture>\d+)_)?"
    r"(?P<tissue>[^_]+)_(?P<stain>[^_]+)_(?P<genotype>[^_]+)"
    r"(?:_(?P<workorder>[^.]+))?\.[^.]+$",
    re.IGNORECASE,
)
_COMPACT_RE = re.compile(
    r"^(?P<role>block|slide)_(?P<set>\d+)_"
    r"(?:capture_(?P<capture>\d+)_)?"
    r"(?P<tissue>[^_]+)_(?P<genotype>[^_]+)_"
    r"(?P<slide_no>[^_]+)_(?P<stain>[^.]+)\.png$",
    re.IGNORECASE,
)


def parse_dataset_filename(path: str | Path) -> DatasetFilenameMetadata | None:
    """Parse a dataset filename while preserving optional capture numbering.

    Triple capture files use ``capture_N`` immediately after the shared set
    number, for example ``block_001_capture_02_lung_N2_01_HE.png``. The token
    is optional, so existing filenames retain their current interpretation.
    """
    name = Path(path).name
    match = _COMPACT_RE.fullmatch(name)
    if match:
        return DatasetFilenameMetadata(
            style="compact",
            role=match.group("role").lower(),
            set_number=match.group("set"),
            capture_index=match.group("capture"),
            tissue=match.group("tissue").lower(),
            stain=match.group("stain").upper(),
            genotype=match.group("genotype"),
            workorder="",
            slide_no=match.group("slide_no"),
        )

    match = _LEGACY_RE.fullmatch(name)
    if match:
        return DatasetFilenameMetadata(
            style="legacy",
            role=match.group("role").lower(),
            set_number=match.group("set"),
            capture_index=match.group("capture"),
            tissue=match.group("tissue").lower(),
            stain=match.group("stain").upper(),
            genotype=match.group("genotype"),
            workorder=match.group("workorder") or "",
            slide_no="",
        )

    return None


def capture_tag(index: int, *, width: int = 2) -> str:
    """Return the safe filename token used for one capture in a sequence."""
    if index < 1:
        raise ValueError("capture index must be positive")
    return f"capture_{index:0{width}d}"
