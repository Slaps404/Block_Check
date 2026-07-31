"""Validated, provenance-preserving input manifest for retrieval experiments."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class ManifestValidationError(ValueError):
    """Raised when a retrieval experiment manifest cannot be interpreted safely."""


@dataclass(frozen=True)
class Specimen:
    specimen_id: str
    role: str
    path: str
    work_order: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class ExcludedSlide:
    slide_id: str
    work_order: str
    reason: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class WorkOrder:
    work_order: str
    block_ids: tuple[str, ...]
    slide_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalManifest:
    source_path: str
    source_hash: str
    blocks: tuple[Specimen, ...]
    slides: tuple[Specimen, ...]
    exclusions: tuple[ExcludedSlide, ...]
    work_orders: Mapping[str, WorkOrder]


_REQUIRED = ("block_id", "block_path", "slide_id", "slide_path")
_CURATION_REQUIRED = (
    "row_id", "claim_id", "set_id", "label_source", "inclusion_status",
    "capture_profile", "capture_status",
)
_WORK_ORDER_COLUMNS = ("work_order", "workorder", "work_order_id")


def _text(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(row: Mapping[str, str]) -> Mapping[str, str]:
    omitted = {*_REQUIRED, *_WORK_ORDER_COLUMNS, "exclusion_reason"}
    return MappingProxyType({key: str(value) for key, value in row.items()
                             if key not in omitted and value not in (None, "")})


def _block_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    """Keep only identity/provenance fields that describe the shared block."""
    return MappingProxyType({
        key: value for key, value in metadata.items()
        if key.startswith("block_")
    })


def load_retrieval_manifest(path: str | Path) -> RetrievalManifest:
    """Load one-row-per-slide input, deduplicating repeated block definitions.

    Excluded rows are retained as provenance but deliberately never enter the
    scoring corpus. Paths are resolved relative to the manifest location.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise ManifestValidationError(f"manifest does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestValidationError("manifest has no header")
        required = (*_REQUIRED, *_CURATION_REQUIRED)
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ManifestValidationError(
                f"manifest missing required columns: {', '.join(missing)}"
            )
        if not any(name in reader.fieldnames for name in _WORK_ORDER_COLUMNS):
            raise ManifestValidationError("manifest missing required work_order column")
        rows = list(reader)

    blocks: dict[str, Specimen] = {}
    slides: dict[str, Specimen] = {}
    slide_paths: set[str] = set()
    block_ids_by_order: dict[str, set[str]] = {}
    exclusions: list[ExcludedSlide] = []
    for index, row in enumerate(rows, start=2):
        work_order = _text(row, *_WORK_ORDER_COLUMNS)
        if not work_order:
            raise ManifestValidationError(f"row {index}: missing work_order")
        for field in _CURATION_REQUIRED:
            if not _text(row, field):
                raise ManifestValidationError(f"row {index}: missing curation field {field}")
        block_id, slide_id = _text(row, "block_id"), _text(row, "slide_id")
        if not block_id or not slide_id:
            raise ManifestValidationError(
                f"row {index}: block_id and slide_id are required"
            )
        reason = _text(row, "exclusion_reason")
        inclusion = _text(row, "inclusion_status").lower()
        if inclusion not in {"included", "excluded"}:
            raise ManifestValidationError(
                f"row {index}: inclusion_status must be included or excluded"
            )
        if inclusion == "excluded" and not reason:
            raise ManifestValidationError(
                f"row {index}: excluded row requires exclusion_reason"
            )
        if inclusion == "included" and reason:
            raise ManifestValidationError(
                f"row {index}: included row cannot have exclusion_reason"
            )
        meta = _metadata(row)
        if inclusion == "excluded":
            exclusions.append(ExcludedSlide(slide_id, work_order, reason, meta))
            continue
        block_path = (source.parent / _text(row, "block_path")).resolve()
        slide_path = (source.parent / _text(row, "slide_path")).resolve()
        if not block_path.is_file() or not slide_path.is_file():
            raise ManifestValidationError(f"row {index}: specimen path does not exist")
        bkey, skey = block_id, slide_id
        block = Specimen(
            block_id, "block", str(block_path), work_order, _block_metadata(meta)
        )
        slide_metadata = MappingProxyType({**meta, "claim_block_id": block_id})
        slide = Specimen(
            slide_id, "slide", str(slide_path), work_order, slide_metadata
        )
        previous_block = blocks.get(bkey)
        if previous_block and (
            previous_block.path != block.path
            or dict(previous_block.metadata) != dict(block.metadata)
        ):
            raise ManifestValidationError(
                f"row {index}: conflicting block definition for {block_id}"
            )
        previous_slide = slides.get(skey)
        if previous_slide:
            raise ManifestValidationError(f"row {index}: duplicate slide {slide_id}")
        if str(slide_path) in slide_paths:
            raise ManifestValidationError(f"row {index}: duplicate slide path {slide_path}")
        blocks.setdefault(bkey, block)
        slides[skey] = slide
        slide_paths.add(str(slide_path))
        block_ids_by_order.setdefault(work_order, set()).add(block_id)

    ordered_blocks = tuple(sorted(
        blocks.values(), key=lambda item: (item.work_order, item.specimen_id)
    ))
    ordered_slides = tuple(sorted(
        slides.values(), key=lambda item: (item.work_order, item.specimen_id)
    ))
    order_names = sorted(
        {item.work_order for item in ordered_blocks}
        | {item.work_order for item in ordered_slides}
    )
    work_orders = MappingProxyType({
        name: WorkOrder(
            name,
            tuple(sorted(block_ids_by_order.get(name, ()))),
            tuple(item.specimen_id for item in ordered_slides
                  if item.work_order == name),
        )
        for name in order_names
    })
    return RetrievalManifest(
        str(source), _digest(source), ordered_blocks, ordered_slides,
        tuple(exclusions), work_orders,
    )
