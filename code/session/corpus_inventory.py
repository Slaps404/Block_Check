from __future__ import annotations

import csv
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


INVENTORY_CSV_COLUMNS = (
    "session_number",
    "session_mode",
    "work_order_id",
    "work_order",
    "block_id",
    "block_capture_id",
    "block_capture_path",
    "slide_capture_id",
    "slide_num",
    "stain",
    "payload_format",
    "slide_capture_path",
    "slide_job_state",
    "issues",
)


@dataclass(frozen=True)
class CorpusInventoryRow:
    session_number: int
    session_mode: str
    work_order_id: int | None
    work_order: str
    block_id: str
    block_capture_id: str
    block_capture_path: str
    slide_capture_id: str
    slide_num: str
    stain: str
    payload_format: str
    slide_capture_path: str
    slide_job_state: str
    issues: tuple[str, ...] = ()

    def as_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = ";".join(self.issues)
        return row


@dataclass(frozen=True)
class CorpusInventorySummary:
    sessions: int
    work_order_brackets: int
    named_work_orders: int
    blocks: int
    slides: int
    complete_claimed_pairs: int
    rows_with_issues: int


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    resolved = database_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _path_issue(path_text: str, *, role: str) -> str | None:
    if not path_text:
        return f"missing_{role}_capture_path"
    if not Path(path_text).is_file():
        return f"missing_{role}_image_file"
    return None


def _duplicate_claim_keys(rows: Iterable[CorpusInventoryRow]) -> set[tuple[object, ...]]:
    counts: dict[tuple[object, ...], int] = {}
    for row in rows:
        if not row.slide_capture_id or row.slide_job_state == "superseded":
            continue
        if not row.block_id or not row.slide_num:
            continue
        key = (
            row.session_number,
            row.work_order_id,
            row.block_id,
            row.slide_num,
            row.stain,
        )
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def collect_corpus_inventory(
    database_path: Path,
    *,
    session_number: int | None = None,
) -> list[CorpusInventoryRow]:
    """Read block/claimed-slide relationships without modifying the live DB."""
    session_filter = "" if session_number is None else "WHERE se.session_number=?"
    params: tuple[object, ...] = () if session_number is None else (session_number,)
    query = f"""
        SELECT
            se.session_number,
            se.session_mode,
            s.work_order_id,
            COALESCE(sc.work_order, '') AS work_order,
            s.block_id,
            COALESCE(s.capture_id, '') AS block_capture_id,
            COALESCE(s.capture_path, '') AS block_capture_path,
            COALESCE(sc.capture_id, '') AS slide_capture_id,
            COALESCE(sc.slide_num, '') AS slide_num,
            COALESCE(sc.stain, '') AS stain,
            COALESCE(sc.payload_format, '') AS payload_format,
            COALESCE(sc.capture_path, '') AS slide_capture_path,
            COALESCE(sc.job_state, '') AS slide_job_state,
            0 AS orphan_slide
        FROM sessions AS se
        JOIN sets AS s ON s.session_number=se.session_number
        LEFT JOIN slide_captures AS sc
          ON sc.session_number=s.session_number
         AND sc.success=1
         AND sc.block_id=s.block_id
         AND (
              sc.work_order_id=s.work_order_id
              OR (sc.work_order_id IS NULL AND s.work_order_id IS NULL)
         )
        {session_filter}

        UNION ALL

        SELECT
            se.session_number,
            se.session_mode,
            sc.work_order_id,
            COALESCE(sc.work_order, '') AS work_order,
            COALESCE(sc.block_id, '') AS block_id,
            '' AS block_capture_id,
            '' AS block_capture_path,
            sc.capture_id AS slide_capture_id,
            COALESCE(sc.slide_num, '') AS slide_num,
            COALESCE(sc.stain, '') AS stain,
            COALESCE(sc.payload_format, '') AS payload_format,
            sc.capture_path AS slide_capture_path,
            COALESCE(sc.job_state, '') AS slide_job_state,
            1 AS orphan_slide
        FROM sessions AS se
        JOIN slide_captures AS sc ON sc.session_number=se.session_number
        WHERE sc.success=1
          AND NOT EXISTS (
              SELECT 1 FROM sets AS s
              WHERE s.session_number=sc.session_number
                AND s.block_id=sc.block_id
                AND (
                     s.work_order_id=sc.work_order_id
                     OR (s.work_order_id IS NULL AND sc.work_order_id IS NULL)
                )
          )
          {'' if session_number is None else 'AND se.session_number=?'}
        ORDER BY 1, 3, 5, 9, 8
    """
    union_params = params if session_number is None else params + params
    with _read_only_connection(database_path) as connection:
        raw_rows = connection.execute(query, union_params).fetchall()

    rows: list[CorpusInventoryRow] = []
    orphan_flags: list[bool] = []
    for raw in raw_rows:
        row = CorpusInventoryRow(
            session_number=int(raw["session_number"]),
            session_mode=str(raw["session_mode"] or "normal"),
            work_order_id=(
                int(raw["work_order_id"])
                if raw["work_order_id"] is not None
                else None
            ),
            work_order=str(raw["work_order"]),
            block_id=str(raw["block_id"]),
            block_capture_id=str(raw["block_capture_id"]),
            block_capture_path=str(raw["block_capture_path"]),
            slide_capture_id=str(raw["slide_capture_id"]),
            slide_num=str(raw["slide_num"]),
            stain=str(raw["stain"]),
            payload_format=str(raw["payload_format"]),
            slide_capture_path=str(raw["slide_capture_path"]),
            slide_job_state=str(raw["slide_job_state"]),
        )
        rows.append(row)
        orphan_flags.append(bool(raw["orphan_slide"]))

    duplicate_keys = _duplicate_claim_keys(rows)
    block_locations: dict[str, set[tuple[int, int | None]]] = {}
    for row in rows:
        if row.block_capture_id and row.block_id:
            block_locations.setdefault(row.block_id, set()).add(
                (row.session_number, row.work_order_id)
            )

    audited: list[CorpusInventoryRow] = []
    for row, orphan_slide in zip(rows, orphan_flags):
        issues: list[str] = []
        block_issue = _path_issue(row.block_capture_path, role="block")
        if block_issue:
            issues.append(block_issue)
        if row.slide_capture_id:
            slide_issue = _path_issue(row.slide_capture_path, role="slide")
            if slide_issue:
                issues.append(slide_issue)
        else:
            issues.append("no_claimed_slide")
        if orphan_slide:
            issues.append(
                "unclaimed_slide" if not row.block_id else "claimed_block_not_in_work_order"
            )
        if row.slide_job_state == "superseded":
            issues.append("superseded_slide_capture")
        duplicate_key = (
            row.session_number,
            row.work_order_id,
            row.block_id,
            row.slide_num,
            row.stain,
        )
        if duplicate_key in duplicate_keys:
            issues.append("duplicate_slide_claim")
        if row.block_id and len(block_locations.get(row.block_id, ())) > 1:
            issues.append("repeated_block_id")
        audited.append(replace(row, issues=tuple(issues)))
    return audited


def summarize_corpus_inventory(
    rows: Iterable[CorpusInventoryRow],
) -> CorpusInventorySummary:
    materialized = list(rows)
    sessions = {row.session_number for row in materialized}
    work_order_brackets = {
        row.work_order_id
        for row in materialized
        if row.work_order_id is not None
    }
    named_work_orders = {row.work_order for row in materialized if row.work_order}
    blocks = {
        (row.session_number, row.work_order_id, row.block_id)
        for row in materialized
        if row.block_capture_id
    }
    slides = {row.slide_capture_id for row in materialized if row.slide_capture_id}
    complete_pairs = sum(
        1
        for row in materialized
        if row.block_capture_id
        and row.slide_capture_id
        and Path(row.block_capture_path).is_file()
        and Path(row.slide_capture_path).is_file()
        and row.slide_job_state != "superseded"
    )
    return CorpusInventorySummary(
        sessions=len(sessions),
        work_order_brackets=len(work_order_brackets),
        named_work_orders=len(named_work_orders),
        blocks=len(blocks),
        slides=len(slides),
        complete_claimed_pairs=complete_pairs,
        rows_with_issues=sum(bool(row.issues) for row in materialized),
    )


def write_corpus_inventory_csv(
    output_path: Path, rows: Iterable[CorpusInventoryRow],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
