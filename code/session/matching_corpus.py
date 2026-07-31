from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PAIRS_CSV_COLUMNS = (
    "pair_id",
    "session_number",
    "work_order",
    "block_id",
    "slide_capture_id",
    "pair_source",
    "is_match",
    "classical_score",
    "rank_for_block",
    "metric",
    "scored_at",
    "block_mask_path",
    "slide_mask_path",
)

SPECIMENS_CSV_COLUMNS = (
    "specimen_key",
    "session_number",
    "role",
    "ref_id",
    "mask_path",
    "capture_path",
)


@dataclass(frozen=True)
class TruePairRef:
    session_number: int
    work_order: str
    block_id: str
    slide_capture_id: str


@dataclass(frozen=True)
class CandidateRef:
    session_number: int
    work_order: str
    block_id: str
    slide_capture_id: str


@dataclass(frozen=True)
class ScoredPair:
    pair_id: str
    block_id: str
    is_match: bool
    score: float


def make_pair_id(session_number: int, block_id: str, slide_capture_id: str) -> str:
    return f"{session_number}|{block_id}|{slide_capture_id}"


def expand_same_work_order_candidates(
    true_pairs: list[TruePairRef],
) -> list[CandidateRef]:
    by_wo: dict[tuple[int, str], list[TruePairRef]] = {}
    for pair in true_pairs:
        by_wo.setdefault((pair.session_number, pair.work_order), []).append(pair)
    out: list[CandidateRef] = []
    for (session_number, work_order), group in by_wo.items():
        for block_row in group:
            for slide_row in group:
                if block_row.block_id == slide_row.block_id:
                    continue
                out.append(
                    CandidateRef(
                        session_number,
                        work_order,
                        block_row.block_id,
                        slide_row.slide_capture_id,
                    )
                )
    return out


def promote_near_misses(rows: list[ScoredPair], *, margin: float) -> set[str]:
    wrong_by_block: dict[str, list[ScoredPair]] = {}
    for row in rows:
        if row.is_match:
            continue
        wrong_by_block.setdefault(row.block_id, []).append(row)
    promoted: set[str] = set()
    for block_rows in wrong_by_block.values():
        best = max(block_rows, key=lambda r: r.score)
        for row in block_rows:
            if best.score - row.score <= margin:
                promoted.add(row.pair_id)
    return promoted


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _mask_dest_name(specimen_key: str) -> str:
    return specimen_key.replace("|", "_") + ".png"


def _rewrite_paths_for_copy(
    specimens: list[dict[str, object]],
    pairs: list[dict[str, object]],
    path_map: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rewritten_specimens: list[dict[str, object]] = []
    for row in specimens:
        updated = dict(row)
        src = str(updated.get("mask_path") or "")
        if src in path_map:
            updated["mask_path"] = path_map[src]
        rewritten_specimens.append(updated)

    rewritten_pairs: list[dict[str, object]] = []
    for row in pairs:
        updated = dict(row)
        for column in ("block_mask_path", "slide_mask_path"):
            src = str(updated.get(column) or "")
            if src in path_map:
                updated[column] = path_map[src]
        rewritten_pairs.append(updated)
    return rewritten_specimens, rewritten_pairs


def _build_freeze_readme(
    *,
    live_root: Path | None,
    session_number: int | None,
    work_order: str | None,
    pair_count: int,
    specimen_count: int,
    copy_masks: bool,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Matching corpus freeze",
        "",
        f"Created: {created_at}",
        f"Live root: {live_root or ''}",
        f"Session: {session_number if session_number is not None else ''}",
        f"Work order: {work_order or 'all'}",
        f"Pairs: {pair_count}",
        f"Specimens: {specimen_count}",
        f"Copy masks: {'yes' if copy_masks else 'no'}",
        "",
        "## Files",
        "",
        "- `pairs.csv` pair rows with resolved mask paths",
        "- `specimens.csv` unique block and slide specimens referenced by pairs",
    ]
    if copy_masks:
        lines.append("- `masks/` copied comparable mask PNGs referenced by the CSVs")
    return "\n".join(lines) + "\n"


def write_freeze_snapshot(
    output_dir: Path,
    *,
    pairs: list[dict[str, object]],
    specimens: list[dict[str, object]],
    copy_masks: bool = False,
    live_root: Path | None = None,
    session_number: int | None = None,
    work_order: str | None = None,
) -> None:
    """Write a read-only training snapshot (CSV tables + README)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    working_specimens = [dict(row) for row in specimens]
    working_pairs = [dict(row) for row in pairs]

    if copy_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        path_map: dict[str, str] = {}
        for spec in working_specimens:
            src = str(spec.get("mask_path") or "")
            if not src or src in path_map:
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue
            dest_name = _mask_dest_name(str(spec["specimen_key"]))
            dest_path = masks_dir / dest_name
            shutil.copy2(src_path, dest_path)
            path_map[src] = f"masks/{dest_name}"
        working_specimens, working_pairs = _rewrite_paths_for_copy(
            working_specimens, working_pairs, path_map,
        )

    pairs_path = output_dir / "pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PAIRS_CSV_COLUMNS, extrasaction="ignore",
        )
        writer.writeheader()
        for row in working_pairs:
            writer.writerow({col: _csv_cell(row.get(col)) for col in PAIRS_CSV_COLUMNS})

    specimens_path = output_dir / "specimens.csv"
    with specimens_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SPECIMENS_CSV_COLUMNS, extrasaction="ignore",
        )
        writer.writeheader()
        for row in working_specimens:
            writer.writerow(
                {col: _csv_cell(row.get(col)) for col in SPECIMENS_CSV_COLUMNS}
            )

    readme = _build_freeze_readme(
        live_root=live_root,
        session_number=session_number,
        work_order=work_order,
        pair_count=len(working_pairs),
        specimen_count=len(working_specimens),
        copy_masks=copy_masks,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
