"""Export a read-only matching corpus snapshot from sessions.sqlite3."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from session.matching_corpus import write_freeze_snapshot  # noqa: E402
from session.processing_store import ProcessingStore  # noqa: E402

DEFAULT_LIVE_ROOT = ROOT / "outputs" / "live_session"


def _slide_mask_path(
    store: ProcessingStore, session_number: int, slide_capture_id: str,
) -> Path:
    session = store._session_identity(session_number)
    return session.directory / "slide_artifacts" / f"{slide_capture_id}_mask.png"


def collect_freeze_data(
    store: ProcessingStore,
    session_number: int,
    work_order: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = list(store.list_matching_pairs(session_number))
    if work_order is not None:
        rows = [row for row in rows if str(row["work_order"]) == work_order]

    pairs_out: list[dict[str, object]] = []
    specimens_by_key: dict[str, dict[str, object]] = {}

    for row in rows:
        session_num = int(row["session_number"])
        block_id = str(row["block_id"])
        slide_capture_id = str(row["slide_capture_id"])
        block_set = store.get_set(session_num, block_id)
        block_mask = block_set.get("mask_path")
        slide_mask = _slide_mask_path(store, session_num, slide_capture_id)
        slide_row = store.get_slide_capture(session_num, slide_capture_id)

        pairs_out.append({
            "pair_id": row["pair_id"],
            "session_number": session_num,
            "work_order": str(row["work_order"]),
            "block_id": block_id,
            "slide_capture_id": slide_capture_id,
            "pair_source": str(row["pair_source"]),
            "is_match": row["is_match"],
            "classical_score": row["classical_score"],
            "rank_for_block": row["rank_for_block"],
            "metric": row["metric"],
            "scored_at": row["scored_at"],
            "block_mask_path": str(block_mask) if block_mask else "",
            "slide_mask_path": str(slide_mask),
        })

        block_key = f"{session_num}|block|{block_id}"
        if block_key not in specimens_by_key and block_mask:
            specimens_by_key[block_key] = {
                "specimen_key": block_key,
                "session_number": session_num,
                "role": "block",
                "ref_id": block_id,
                "mask_path": str(block_mask),
                "capture_path": str(block_set.get("capture_path") or ""),
            }

        slide_key = f"{session_num}|slide|{slide_capture_id}"
        if slide_key not in specimens_by_key:
            specimens_by_key[slide_key] = {
                "specimen_key": slide_key,
                "session_number": session_num,
                "role": "slide",
                "ref_id": slide_capture_id,
                "mask_path": str(slide_mask),
                "capture_path": str(slide_row.get("capture_path") or ""),
            }

    return pairs_out, list(specimens_by_key.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-root",
        type=Path,
        default=DEFAULT_LIVE_ROOT,
        help=f"processing root (default: {DEFAULT_LIVE_ROOT})",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--work-order", default=None)
    parser.add_argument(
        "--copy-masks",
        action="store_true",
        help="copy referenced masks into output/masks/ and rewrite CSV paths",
    )
    args = parser.parse_args(argv)
    store = ProcessingStore(args.live_root, recover_jobs=False)
    session = (
        args.session if args.session is not None else store.latest_session_number()
    )
    pairs, specimens = collect_freeze_data(store, session, args.work_order)
    write_freeze_snapshot(
        args.output,
        pairs=pairs,
        specimens=specimens,
        copy_masks=args.copy_masks,
        live_root=args.live_root,
        session_number=session,
        work_order=args.work_order,
    )
    print(
        f"froze {len(pairs)} pairs and {len(specimens)} specimens "
        f"to {args.output} (session={session})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
