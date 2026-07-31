"""Production-parity all-pairs diagnostics.

Diagnostics reuse production normalization, locked alignment, routing, and gates.
They add comparison-only metrics and labels; they never make a verdict.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

from diagnostic_metrics import (
    BASELINE_METRIC_NAMES,
    NEW_METRIC_NAMES,
    build_specimen_metric_cache_from_locked,
    score_locked_metrics,
)
from verify.gates import run_quality_gates
from verify.locked_alignment import align_normalized_masks
from session.preparation import PreparedSpecimen, prepare_specimen
from verify.scorer import (
    _component_features,
    build_locked_score_cache,
    LockedScoreCache,
    score_routed_caches,
)
from robust_normalization import normalize_mask

DIAGNOSTIC_LABEL_COLUMN = "diagnostic_label"
PRODUCTION_COLUMNS = (
    "score", "selected_metric", "best_angle", "best_flip",
    "align_soft_iou", "mask_iou", "block_occupied_fraction",
    "slide_occupied_fraction", "router_size_signal",
)
DIAGNOSTIC_METRIC_COLUMNS = tuple(
    name for name in BASELINE_METRIC_NAMES if name != "mask_iou"
) + NEW_METRIC_NAMES
DIAGNOSTIC_COLUMNS = (
    DIAGNOSTIC_LABEL_COLUMN,
    "block_set", "slide_set", "block_tissue_raw", "slide_tissue_raw",
    "tissue_bucket", "block_stain", "slide_stain", "block_genotype",
    "slide_genotype", "block_workorder", "slide_workorder",
    "block_path", "slide_path", "gate_passed", "gate_stage", "gate_reason",
    *PRODUCTION_COLUMNS, *DIAGNOSTIC_METRIC_COLUMNS,
    "true_vs_best_wrong_margin", "notes",
)


@dataclass(frozen=True)
class DiagnosticPairRecord:
    diagnostic_label: str
    block_set: str
    slide_set: str
    block_path: str
    slide_path: str
    gate_passed: bool
    gate_stage: str
    gate_reason: str
    score: float | None
    selected_metric: str
    best_angle: float | None
    best_flip: bool | None
    align_soft_iou: float | None
    mask_iou: float | None
    true_vs_best_wrong_margin: float | None
    columns: dict[str, object]


_SET_RE = re.compile(r"set_(\d+)_", re.IGNORECASE)
_TISSUE_RE = re.compile(r"set_\d+_(?:block_silhouette|block|slide)_([^_]+)_", re.IGNORECASE)
_META_RE = re.compile(
    r"set_(?P<set>\d+)_(?P<role>block_silhouette|block|slide)_"
    r"(?P<tissue>[^_]+)_(?P<stain>[^_]+)_(?P<genotype>[^_]+)_(?P<workorder>[^.]+)",
    re.IGNORECASE,
)
_V3_META_RE = re.compile(
    r"(?P<role>block|slide)_(?P<set>\d+)_(?P<tissue>[^_]+)_"
    r"(?P<genotype>[^_]+)_(?P<slide_no>[^_]+)_(?P<stain>[^.]+)",
    re.IGNORECASE,
)


def _canonical_tissue(tissue: str | None) -> str | None:
    return tissue.lower().strip() if tissue else None


def _extract_set_id(path: str | Path) -> str | None:
    match = _SET_RE.search(Path(path).name)
    return match.group(1) if match else None


def _extract_tissue(path: str | Path) -> str | None:
    match = _TISSUE_RE.search(Path(path).name)
    return match.group(1).lower() if match else None


def _extract_metadata(path: str | Path) -> dict[str, str]:
    name = Path(path).name
    match = _META_RE.search(name)
    if match:
        return {
            "set": match.group("set"), "tissue_raw": match.group("tissue").lower(),
            "tissue_bucket": match.group("tissue").lower(),
            "stain": match.group("stain").upper(), "genotype": match.group("genotype"),
            "workorder": match.group("workorder"),
        }
    match = _V3_META_RE.search(name)
    if match:
        return {
            "set": match.group("set"), "tissue_raw": match.group("tissue").lower(),
            "tissue_bucket": match.group("tissue").lower(),
            "stain": match.group("stain").upper(), "genotype": match.group("genotype"),
            "workorder": "",
        }
    tissue = _extract_tissue(path) or ""
    return {
        "set": _extract_set_id(path) or "", "tissue_raw": tissue,
        "tissue_bucket": tissue, "stain": "", "genotype": "", "workorder": "",
    }


def _metadata_for_path(
    path: str | Path,
    path_metadata: dict[str, tuple[str | None, str | None]] | None,
) -> dict[str, str]:
    meta = _extract_metadata(path)
    if path_metadata:
        supplied = path_metadata.get(Path(path).as_posix()) or path_metadata.get(str(path))
        if supplied:
            set_id, tissue = supplied
            meta["set"] = set_id or meta["set"]
            if tissue:
                meta["tissue_raw"] = tissue.lower()
                meta["tissue_bucket"] = tissue.lower()
    return meta


def _diagnostic_label_from_row(
    block_path: str | Path, slide_path: str | Path,
    block_set_id: str | None, slide_set_id: str | None,
    block_tissue: str | None, slide_tissue: str | None,
) -> str:
    block_set_id = block_set_id or _extract_set_id(block_path)
    slide_set_id = slide_set_id or _extract_set_id(slide_path)
    return "true_pair" if block_set_id and block_set_id == slide_set_id else "wrong_pair"


def _format(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _record_from_row(row: dict[str, object]) -> DiagnosticPairRecord:
    return DiagnosticPairRecord(
        diagnostic_label=str(row[DIAGNOSTIC_LABEL_COLUMN]),
        block_set=str(row["block_set"]),
        slide_set=str(row["slide_set"]),
        block_path=str(row["block_path"]),
        slide_path=str(row["slide_path"]),
        gate_passed=bool(_bool_or_none(row["gate_passed"])),
        gate_stage=str(row["gate_stage"]),
        gate_reason=str(row["gate_reason"]),
        score=_float_or_none(row["score"]),
        selected_metric=str(row["selected_metric"]),
        best_angle=_float_or_none(row["best_angle"]),
        best_flip=_bool_or_none(row["best_flip"]),
        align_soft_iou=_float_or_none(row["align_soft_iou"]),
        mask_iou=_float_or_none(row["mask_iou"]),
        true_vs_best_wrong_margin=_float_or_none(
            row["true_vs_best_wrong_margin"]
        ),
        columns=dict(row),
    )


def _csv_value(column: str, value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        digits = 1 if column == "best_angle" else 4
        return _format(value, digits)
    return str(value)


def _columns_for_record(record: DiagnosticPairRecord) -> dict[str, object]:
    row = {column: "" for column in DIAGNOSTIC_COLUMNS}
    row.update(record.columns)
    row.update({
        DIAGNOSTIC_LABEL_COLUMN: record.diagnostic_label,
        "block_set": record.block_set,
        "slide_set": record.slide_set,
        "block_path": record.block_path,
        "slide_path": record.slide_path,
        "gate_passed": record.gate_passed,
        "gate_stage": record.gate_stage,
        "gate_reason": record.gate_reason,
        "score": record.score if record.score is not None else "",
        "selected_metric": record.selected_metric,
        "best_angle": (
            record.best_angle if record.best_angle is not None else ""
        ),
        "best_flip": record.best_flip if record.best_flip is not None else "",
        "align_soft_iou": (
            record.align_soft_iou
            if record.align_soft_iou is not None else ""
        ),
        "mask_iou": record.mask_iou if record.mask_iou is not None else "",
        "true_vs_best_wrong_margin": (
            record.true_vs_best_wrong_margin
            if record.true_vs_best_wrong_margin is not None else ""
        ),
    })
    return row


def _build_caches(
    prepared: dict[str, object],
    *,
    normalization_mode: str = "rms",
) -> dict[str, object]:
    caches: dict[str, object] = {}
    for key, value in prepared.items():
        if not isinstance(value, PreparedSpecimen):
            continue
        if normalization_mode == "rms":
            caches[key] = build_locked_score_cache(value)
            continue
        normalized = normalize_mask(value.mask, normalization_mode)
        caches[key] = LockedScoreCache(
            normalized,
            _component_features(normalized),
        )
    return caches


def _row(
    bpath: str | Path, spath: str | Path,
    block: object, slide: object,
    block_meta: dict[str, str], slide_meta: dict[str, str],
    block_locked: dict[str, object], slide_locked: dict[str, object],
) -> dict[str, str]:
    gate = run_quality_gates(block, slide)
    label = _diagnostic_label_from_row(
        bpath, spath, block_meta["set"], slide_meta["set"],
        block_meta["tissue_bucket"], slide_meta["tissue_bucket"],
    )
    row: dict[str, object] = {column: "" for column in DIAGNOSTIC_COLUMNS}
    block_tissue = block_meta["tissue_bucket"]
    slide_tissue = slide_meta["tissue_bucket"]
    row.update({
        DIAGNOSTIC_LABEL_COLUMN: label,
        "block_set": block_meta["set"], "slide_set": slide_meta["set"],
        "block_tissue_raw": block_meta["tissue_raw"],
        "slide_tissue_raw": slide_meta["tissue_raw"],
        "tissue_bucket": block_tissue if block_tissue == slide_tissue else "",
        "block_stain": block_meta["stain"], "slide_stain": slide_meta["stain"],
        "block_genotype": block_meta["genotype"],
        "slide_genotype": slide_meta["genotype"],
        "block_workorder": block_meta["workorder"], "slide_workorder": slide_meta["workorder"],
        "block_path": str(bpath), "slide_path": str(spath),
        "gate_passed": gate.passed, "gate_stage": gate.stage,
        "gate_reason": gate.reason,
        "notes": "diagnostic only - not a production claim decision",
    })
    if not isinstance(block, PreparedSpecimen) or not isinstance(slide, PreparedSpecimen):
        return row

    bcache = block_locked[str(bpath)]
    scache = slide_locked[str(spath)]
    result = score_routed_caches(bcache, scache)
    row.update({
        "score": result.score if gate.passed else "",
        "selected_metric": result.selected_metric,
        "best_angle": result.best_angle,
        "best_flip": result.best_flip,
        "align_soft_iou": result.align_soft_iou,
        "mask_iou": result.mask_iou,
        "block_occupied_fraction": result.block_occupied_fraction,
        "slide_occupied_fraction": result.slide_occupied_fraction,
        "router_size_signal": result.router_size_signal,
    })
    alignment = align_normalized_masks(bcache.normalized_mask, scache.normalized_mask)
    bmetrics = build_specimen_metric_cache_from_locked(bcache)
    smetrics = build_specimen_metric_cache_from_locked(scache)
    metrics = score_locked_metrics(
        bmetrics, smetrics, alignment.aligned_slide_mask,
        block_soft_blur=None,
    )
    row.update(metrics)
    return row


def _assign_near_misses_and_margins(rows: list[dict[str, object]]) -> None:
    by_block: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_block.setdefault(row["block_path"], []).append(row)
    for block_rows in by_block.values():
        wrong = [
            row for row in block_rows
            if row[DIAGNOSTIC_LABEL_COLUMN] == "wrong_pair"
            and row["score"] not in ("", None)
        ]
        best_wrong = max(wrong, key=lambda r: float(r["score"]), default=None)
        true = next(
            (row for row in block_rows if row[DIAGNOSTIC_LABEL_COLUMN] == "true_pair"),
            None,
        )
        if best_wrong:
            best_wrong[DIAGNOSTIC_LABEL_COLUMN] = "near_miss"
        if true and true["score"] not in ("", None) and best_wrong:
            margin = float(true["score"]) - float(best_wrong["score"])
            true["true_vs_best_wrong_margin"] = margin
            best_wrong["true_vs_best_wrong_margin"] = margin


def _write_rows(rows: list[dict[str, object]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: _csv_value(column, row[column])
                for column in DIAGNOSTIC_COLUMNS
            })


def write_diagnostic_csv(
    records: Sequence[DiagnosticPairRecord],
    output_path: str | Path,
) -> None:
    _write_rows([_columns_for_record(record) for record in records], output_path)


def collect_all_pair_records(
    block_paths: Sequence[str | Path], slide_paths: Sequence[str | Path],
    path_metadata: dict[str, tuple[str | None, str | None]] | None = None,
    normalization_mode: str = "rms",
    **_compatibility_options,
) -> tuple[DiagnosticPairRecord, ...]:
    prepared_blocks = {str(p): prepare_specimen(p, role="block") for p in block_paths}
    prepared_slides = {str(p): prepare_specimen(p, role="slide") for p in slide_paths}
    block_locked = _build_caches(
        prepared_blocks,
        normalization_mode=normalization_mode,
    )
    slide_locked = _build_caches(
        prepared_slides,
        normalization_mode=normalization_mode,
    )
    rows = [
        _row(
            b, s, prepared_blocks[str(b)], prepared_slides[str(s)],
            _metadata_for_path(b, path_metadata), _metadata_for_path(s, path_metadata),
            block_locked, slide_locked,
        )
        for b in block_paths for s in slide_paths
    ]
    _assign_near_misses_and_margins(rows)
    return tuple(_record_from_row(row) for row in rows)


def collect_selected_pair_records(
    pairs: Sequence[tuple[str | Path, str | Path]],
    path_metadata: dict[str, tuple[str | None, str | None]] | None = None,
    normalization_mode: str = "rms",
    assign_near_misses: bool = False,
    **_compatibility_options,
) -> tuple[DiagnosticPairRecord, ...]:
    blocks = {str(b): b for b, _ in pairs}
    slides = {str(s): s for _, s in pairs}
    prepared_blocks = {key: prepare_specimen(path, role="block") for key, path in blocks.items()}
    prepared_slides = {key: prepare_specimen(path, role="slide") for key, path in slides.items()}
    block_locked = _build_caches(
        prepared_blocks,
        normalization_mode=normalization_mode,
    )
    slide_locked = _build_caches(
        prepared_slides,
        normalization_mode=normalization_mode,
    )
    rows = [
        _row(
            b, s, prepared_blocks[str(b)], prepared_slides[str(s)],
            _metadata_for_path(b, path_metadata), _metadata_for_path(s, path_metadata),
            block_locked, slide_locked,
        )
        for b, s in pairs
    ]
    if assign_near_misses:
        _assign_near_misses_and_margins(rows)
    return tuple(_record_from_row(row) for row in rows)


def run_all_pairs_diagnostic(
    block_paths: Sequence[str | Path], slide_paths: Sequence[str | Path],
    output_path: str | Path,
    path_metadata: dict[str, tuple[str | None, str | None]] | None = None,
    normalization_mode: str = "rms",
    **_compatibility_options,
) -> None:
    write_diagnostic_csv(
        collect_all_pair_records(
            block_paths,
            slide_paths,
            path_metadata=path_metadata,
            normalization_mode=normalization_mode,
            **_compatibility_options,
        ),
        output_path,
    )


def run_selected_pairs_diagnostic(
    pairs: Sequence[tuple[str | Path, str | Path]], output_path: str | Path,
    path_metadata: dict[str, tuple[str | None, str | None]] | None = None,
    normalization_mode: str = "rms",
    **_compatibility_options,
) -> None:
    write_diagnostic_csv(
        collect_selected_pair_records(
            pairs,
            path_metadata=path_metadata,
            normalization_mode=normalization_mode,
            assign_near_misses=True,
            **_compatibility_options,
        ),
        output_path,
    )
