"""Tests for tools/prune_empty_sessions.py (empty live-session cleanup)."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from session.workflow import ProcessingStore

# Load by path (same pattern as test_feed_captures) so tools/ is not on sys.path.
_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prune_empty_sessions.py"
_spec = importlib.util.spec_from_file_location("prune_empty_sessions", _SCRIPT)
assert _spec is not None and _spec.loader is not None
prune = importlib.util.module_from_spec(_spec)
sys.modules["prune_empty_sessions"] = prune
_spec.loader.exec_module(prune)

STARTED = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def _store(tmp_path: Path) -> ProcessingStore:
    return ProcessingStore(tmp_path / "live", recover_jobs=False)


def test_find_empty_sessions_skips_sessions_with_sets(tmp_path: Path):
    store = _store(tmp_path)
    empty = store.start_session(started_at=STARTED)
    used = store.start_session(started_at=STARTED + timedelta(seconds=1))
    store.scan_block(used.number, "51151378", request_id="scan-1")

    found = prune.find_empty_sessions(store.root)
    numbers = [item.number for item in found]
    assert empty.number in numbers
    assert used.number not in numbers


def test_dry_run_does_not_delete(tmp_path: Path):
    store = _store(tmp_path)
    session = store.start_session(started_at=STARTED)
    assert session.directory.is_dir()

    candidates = prune.prune_empty_sessions(store.root, apply=False)
    assert [c.number for c in candidates] == [session.number]
    assert session.directory.is_dir()
    with sqlite3.connect(store.database) as db:
        row = db.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_number=?",
            (session.number,),
        ).fetchone()[0]
    assert row == 1


def test_apply_deletes_folder_and_sqlite_row(tmp_path: Path):
    store = _store(tmp_path)
    session = store.start_session(started_at=STARTED)
    path = session.directory
    number = session.number

    deleted = prune.prune_empty_sessions(store.root, apply=True)
    assert [c.number for c in deleted] == [number]
    assert not path.exists()
    with sqlite3.connect(store.database) as db:
        row = db.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_number=?", (number,)
        ).fetchone()[0]
    assert row == 0


def test_keep_protects_session(tmp_path: Path):
    store = _store(tmp_path)
    a = store.start_session(started_at=STARTED)
    b = store.start_session(started_at=STARTED + timedelta(seconds=1))

    found = prune.find_empty_sessions(store.root, keep=frozenset({a.number}))
    assert [item.number for item in found] == [b.number]


def test_min_age_skips_young_sessions(tmp_path: Path):
    store = _store(tmp_path)
    young = store.start_session(started_at=STARTED)
    old = store.start_session(started_at=STARTED - timedelta(hours=1))

    found = prune.find_empty_sessions(
        store.root,
        min_age_seconds=600,
        now=STARTED + timedelta(seconds=30),
    )
    assert [item.number for item in found] == [old.number]
    assert young.number not in {item.number for item in found}


def test_capture_artifacts_block_prune_even_without_sets(tmp_path: Path):
    store = _store(tmp_path)
    session = store.start_session(started_at=STARTED)
    captures = session.directory / "captures"
    captures.mkdir()
    (captures / "capture_000001.png").write_bytes(b"not-empty")

    found = prune.find_empty_sessions(store.root)
    assert found == []


def test_orphan_folder_without_db_row_is_pruned(tmp_path: Path):
    store = _store(tmp_path)
    store.start_session(started_at=STARTED)  # ensure DB exists
    orphan = store.root / "session_000099_20260709T000000Z"
    orphan.mkdir()
    (orphan / "session.json").write_text("{}", encoding="utf-8")

    found = prune.find_empty_sessions(store.root)
    assert any(item.number == 99 and item.directory == orphan for item in found)

    prune.prune_empty_sessions(store.root, apply=True)
    assert not orphan.exists()
