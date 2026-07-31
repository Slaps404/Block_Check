from __future__ import annotations

import sqlite3
from pathlib import Path

from session.corpus_inventory import (
    collect_corpus_inventory,
    summarize_corpus_inventory,
    write_corpus_inventory_csv,
)


def _make_inventory_database(tmp_path: Path) -> tuple[Path, Path, Path]:
    database = tmp_path / "sessions.sqlite3"
    block_image = tmp_path / "block.png"
    slide_image = tmp_path / "slide.png"
    block_image.write_bytes(b"block")
    slide_image.write_bytes(b"slide")
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                session_number INTEGER PRIMARY KEY,
                session_mode TEXT NOT NULL
            );
            CREATE TABLE sets (
                session_number INTEGER NOT NULL,
                block_id TEXT NOT NULL,
                work_order_id INTEGER,
                capture_id TEXT,
                capture_path TEXT
            );
            CREATE TABLE slide_captures (
                capture_id TEXT PRIMARY KEY,
                session_number INTEGER NOT NULL,
                success INTEGER NOT NULL,
                work_order_id INTEGER,
                work_order TEXT,
                block_id TEXT,
                slide_num TEXT,
                stain TEXT,
                payload_format TEXT,
                capture_path TEXT,
                job_state TEXT
            );
            """
        )
        connection.execute("INSERT INTO sessions VALUES (1, 'hybrid')")
        connection.executemany(
            "INSERT INTO sets VALUES (?, ?, ?, ?, ?)",
            (
                (1, "B1", 10, "block-1", str(block_image)),
                (1, "B2", 10, "block-2", str(tmp_path / "missing-block.png")),
            ),
        )
        connection.executemany(
            "INSERT INTO slide_captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    "slide-1", 1, 1, 10, "WO1", "B1", "01", "HE", "current",
                    str(slide_image), "complete",
                ),
                (
                    "slide-2", 1, 1, 10, "WO1", "B1", "01", "HE", "current",
                    str(tmp_path / "missing-slide.png"), "complete",
                ),
                (
                    "slide-orphan", 1, 1, 10, "WO1", "B3", "01", "HE", "current",
                    str(slide_image), "complete",
                ),
            ),
        )
    return database, block_image, slide_image


def test_inventory_links_claims_and_flags_missing_or_duplicate_images(tmp_path):
    database, _block_image, _slide_image = _make_inventory_database(tmp_path)

    rows = collect_corpus_inventory(database)

    assert len(rows) == 4
    b1_rows = [row for row in rows if row.block_id == "B1"]
    assert {row.slide_capture_id for row in b1_rows} == {"slide-1", "slide-2"}
    assert all("duplicate_slide_claim" in row.issues for row in b1_rows)
    missing_slide = next(row for row in rows if row.slide_capture_id == "slide-2")
    assert "missing_slide_image_file" in missing_slide.issues
    b2 = next(row for row in rows if row.block_id == "B2")
    assert b2.slide_capture_id == ""
    assert set(b2.issues) == {"missing_block_image_file", "no_claimed_slide"}
    orphan = next(row for row in rows if row.slide_capture_id == "slide-orphan")
    assert "claimed_block_not_in_work_order" in orphan.issues

    summary = summarize_corpus_inventory(rows)
    assert summary.sessions == 1
    assert summary.work_order_brackets == 1
    assert summary.named_work_orders == 1
    assert summary.blocks == 2
    assert summary.slides == 3
    assert summary.complete_claimed_pairs == 1
    assert summary.rows_with_issues == 4


def test_inventory_can_filter_session_and_write_csv(tmp_path):
    database, _block_image, _slide_image = _make_inventory_database(tmp_path)
    rows = collect_corpus_inventory(database, session_number=1)
    output = tmp_path / "inventory.csv"

    write_corpus_inventory_csv(output, rows)

    text = output.read_text(encoding="utf-8")
    assert "block_capture_path" in text
    assert "duplicate_slide_claim" in text
