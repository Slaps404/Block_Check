"""Store schema and sync upserts for matching_pairs."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from session.matching_corpus import PAIRS_CSV_COLUMNS, write_freeze_snapshot
from slide.qr import DecodeCandidate, select_slide_identity
from session.workflow import ProcessingStore
from tests.test_session_workflow import (  # noqa: F401
    STARTED_AT,
    FastPreprocessor,
    _capture,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
    lightweight_qc_artifacts,
)

WORK_ORDER = "12080"
BLOCK_A = "51151378"
BLOCK_B = "51151379"


def _slide_result(block_id: str, work_order: str = WORK_ORDER):
    return select_slide_identity((
        DecodeCandidate(
            "zxing", "QRCode", "raw", f"{work_order}_{block_id}_01_HE",
        ),
    ))


def _two_block_work_order_fixture(store: ProcessingStore, session, tmp_path: Path):
    store.start_work_order(session.number)
    _evaluable_block(store, session, tmp_path, block_id=BLOCK_A)
    _evaluable_block(store, session, tmp_path, block_id=BLOCK_B)
    _drain_to_slides(store, session)
    capture_a = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT,
        result=_slide_result(BLOCK_A),
        duration_ms=10.0,
    )
    capture_b = store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide_b.png", 121),
        captured_at=STARTED_AT,
        result=_slide_result(BLOCK_B),
        duration_ms=10.0,
    )
    assert store.get_set(session.number, BLOCK_A)["mask_path"] is not None
    assert store.get_set(session.number, BLOCK_B)["mask_path"] is not None
    return capture_a, capture_b


def test_initialize_creates_matching_pairs_table(tmp_path):
    store = ProcessingStore(tmp_path / "processing", recover_jobs=False)
    with store._connect() as db:
        columns = {
            row["name"] for row in db.execute("PRAGMA table_info(matching_pairs)")
        }
    assert columns == {
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
    }


def test_sync_upserts_true_pairs_and_candidates(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        recover_jobs=False,
    )
    session = store.start_session(started_at=STARTED_AT)
    _two_block_work_order_fixture(store, session, tmp_path)

    synced = store.sync_matching_pairs_for_work_order(session.number, WORK_ORDER)
    assert synced == 4

    rows = store.list_matching_pairs(session.number)
    assert len(rows) == 4
    sources = {row["pair_source"] for row in rows}
    assert sources == {"true_pair", "candidate"}
    true_rows = [row for row in rows if row["pair_source"] == "true_pair"]
    candidate_rows = [row for row in rows if row["pair_source"] == "candidate"]
    assert len(true_rows) == 2
    assert len(candidate_rows) == 2
    assert {row["is_match"] for row in true_rows} == {1}
    assert {row["is_match"] for row in candidate_rows} == {0}

    synced_again = store.sync_matching_pairs_for_work_order(session.number, WORK_ORDER)
    assert synced_again == 4
    assert len(store.list_matching_pairs(session.number)) == 4


def _load_freeze_cli():
    cli_path = (
        Path(__file__).resolve().parent.parent
        / "tools"
        / "matching_corpus"
        / "freeze_matching_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("freeze_matching_corpus_cli", cli_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_slide_masks(session, capture_ids: tuple[str, ...]) -> None:
    artifact_dir = session.directory / "slide_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    for capture_id in capture_ids:
        cv2.imwrite(str(artifact_dir / f"{capture_id}_mask.png"), mask)


def _load_sync_cli():
    cli_path = (
        Path(__file__).resolve().parent.parent
        / "tools"
        / "matching_corpus"
        / "sync_matching_pairs.py"
    )
    spec = importlib.util.spec_from_file_location("sync_matching_pairs_cli", cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_matching_corpus_clis_default_live_root_to_outputs_live_session():
    sync_cli = _load_sync_cli()
    assert sync_cli.DEFAULT_LIVE_ROOT == sync_cli.ROOT / "outputs" / "live_session"

    for name in ("score_matching_pairs.py", "freeze_matching_corpus.py"):
        cli_path = (
            Path(__file__).resolve().parent.parent / "tools" / "matching_corpus" / name
        )
        spec = importlib.util.spec_from_file_location(f"{name}_mod", cli_path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        assert cli.DEFAULT_LIVE_ROOT == cli.ROOT / "outputs" / "live_session"


def test_write_matching_pair_score_and_promote_near_misses(tmp_path):
    store = ProcessingStore(tmp_path / "processing", recover_jobs=False)
    session = store.start_session(started_at=STARTED_AT)
    scored_at = "2026-07-27T10:00:00+00:00"

    true_id = store.upsert_matching_pair(
        session_number=session.number,
        work_order=WORK_ORDER,
        block_id=BLOCK_A,
        slide_capture_id="slide-a",
        pair_source="true_pair",
        is_match=1,
    )
    best_wrong = store.upsert_matching_pair(
        session_number=session.number,
        work_order=WORK_ORDER,
        block_id=BLOCK_A,
        slide_capture_id="slide-b",
        pair_source="candidate",
        is_match=0,
    )
    near_wrong = store.upsert_matching_pair(
        session_number=session.number,
        work_order=WORK_ORDER,
        block_id=BLOCK_A,
        slide_capture_id="slide-c",
        pair_source="candidate",
        is_match=0,
    )
    far_wrong = store.upsert_matching_pair(
        session_number=session.number,
        work_order=WORK_ORDER,
        block_id=BLOCK_A,
        slide_capture_id="slide-d",
        pair_source="candidate",
        is_match=0,
    )

    store.write_matching_pair_score(
        true_id,
        classical_score=0.95,
        rank_for_block=None,
        metric="mask_iou",
        scored_at=scored_at,
    )
    store.write_matching_pair_score(
        best_wrong,
        classical_score=0.80,
        rank_for_block=1,
        metric="mask_iou",
        scored_at=scored_at,
    )
    store.write_matching_pair_score(
        near_wrong,
        classical_score=0.77,
        rank_for_block=2,
        metric="mask_iou",
        scored_at=scored_at,
    )
    store.write_matching_pair_score(
        far_wrong,
        classical_score=0.50,
        rank_for_block=3,
        metric="mask_iou",
        scored_at=scored_at,
    )

    promoted = store.promote_matching_near_misses(
        session.number, WORK_ORDER, margin=0.05,
    )
    assert promoted == 2

    rows = {row["pair_id"]: row for row in store.list_matching_pairs(session.number)}
    assert rows[true_id]["pair_source"] == "true_pair"
    assert rows[best_wrong]["pair_source"] == "near_miss"
    assert rows[near_wrong]["pair_source"] == "near_miss"
    assert rows[far_wrong]["pair_source"] == "candidate"
    assert rows[best_wrong]["classical_score"] == 0.80


def test_write_freeze_snapshot_contains_true_pair_with_existing_mask(tmp_path):
    store_root = tmp_path / "processing"
    store = ProcessingStore(
        store_root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        recover_jobs=False,
    )
    session = store.start_session(started_at=STARTED_AT)
    capture_a, capture_b = _two_block_work_order_fixture(store, session, tmp_path)
    _write_slide_masks(session, (capture_a, capture_b))
    store.sync_matching_pairs_for_work_order(session.number, WORK_ORDER)

    freeze_cli = _load_freeze_cli()
    pairs, specimens = freeze_cli.collect_freeze_data(
        store, session.number, WORK_ORDER,
    )
    freeze_dir = tmp_path / "freeze"
    write_freeze_snapshot(
        freeze_dir,
        pairs=pairs,
        specimens=specimens,
        live_root=store_root,
        session_number=session.number,
        work_order=WORK_ORDER,
    )

    pairs_csv = freeze_dir / "pairs.csv"
    specimens_csv = freeze_dir / "specimens.csv"
    readme = freeze_dir / "README.md"
    assert pairs_csv.is_file()
    assert specimens_csv.is_file()
    assert readme.is_file()

    with pairs_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PAIRS_CSV_COLUMNS)
        rows = list(reader)
    true_rows = [row for row in rows if row["pair_source"] == "true_pair"]
    assert true_rows
    true_row = true_rows[0]
    assert Path(true_row["block_mask_path"]).is_file()
    assert Path(true_row["slide_mask_path"]).is_file()


def test_sync_matching_pairs_cli_main(tmp_path, capsys):
    store_root = tmp_path / "processing"
    store = ProcessingStore(
        store_root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        recover_jobs=False,
    )
    session = store.start_session(started_at=STARTED_AT)
    _two_block_work_order_fixture(store, session, tmp_path)

    main = _load_sync_cli().main
    exit_code = main([
        "--live-root", str(store_root),
        "--work-order", WORK_ORDER,
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert (
        f"synced 4 matching_pairs rows for session={session.number} wo={WORK_ORDER}"
        in captured.out
    )
