"""Score unscored matching_pairs rows with deferred classical writeback."""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from constants import MATCH_MARGIN  # noqa: E402
from session.preparation import PreparedSpecimen, PreparationFailure  # noqa: E402
from session.processing_store import ProcessingStore  # noqa: E402
from verify.scorer import score_pair_result_routed  # noqa: E402

DEFAULT_LIVE_ROOT = ROOT / "outputs" / "live_session"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slide_mask_path(store: ProcessingStore, session_number: int, slide_capture_id: str) -> Path:
    session = store._session_identity(session_number)
    return session.directory / "slide_artifacts" / f"{slide_capture_id}_mask.png"


def _load_block_specimen(
    store: ProcessingStore, session_number: int, block_id: str,
) -> PreparedSpecimen | PreparationFailure:
    row = store.get_set(session_number, block_id)
    mask_path = row.get("mask_path")
    if not mask_path:
        return PreparationFailure(role="block", reason="block mask_path is missing")
    return store._load_block_result(row)


def _load_slide_specimen(mask_path: Path) -> PreparedSpecimen | PreparationFailure:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return PreparationFailure(
            role="slide",
            reason=f"could not read slide comparable mask: {mask_path}",
        )
    return PreparedSpecimen(role="slide", mask=mask, roi_ok=True)


def _selected_pair_sources(include_true_pairs: bool) -> tuple[str, ...]:
    if include_true_pairs:
        return ("true_pair", "candidate", "near_miss")
    return ("candidate", "near_miss")


def _pairs_to_score(
    store: ProcessingStore,
    session_number: int,
    work_order: str,
    *,
    include_true_pairs: bool,
) -> list[dict[str, object]]:
    allowed = set(_selected_pair_sources(include_true_pairs))
    return [
        row for row in store.list_matching_pairs(session_number, unscored_only=True)
        if row["work_order"] == work_order and row["pair_source"] in allowed
    ]


def _assign_ranks_for_work_order(
    store: ProcessingStore, session_number: int, work_order: str,
) -> int:
    wrong_rows = [
        row for row in store.list_matching_pairs(session_number)
        if row["work_order"] == work_order
        and row["is_match"] == 0
        and row["classical_score"] is not None
    ]
    by_block: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in wrong_rows:
        by_block[str(row["block_id"])].append(row)
    updated = 0
    for block_rows in by_block.values():
        ranked = sorted(
            block_rows,
            key=lambda row: float(row["classical_score"]),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            store.write_matching_pair_score(
                str(row["pair_id"]),
                classical_score=float(row["classical_score"]),
                rank_for_block=rank,
                metric=str(row["metric"]),
                scored_at=str(row["scored_at"]),
            )
            updated += 1
    return updated


def score_work_order(
    store: ProcessingStore,
    session_number: int,
    work_order: str,
    *,
    include_true_pairs: bool = False,
    margin: float = MATCH_MARGIN,
) -> tuple[int, int, int]:
    """Score unscored pairs for one work order; return scored, skipped, promoted."""
    pairs = _pairs_to_score(
        store, session_number, work_order, include_true_pairs=include_true_pairs,
    )
    scored_at = _utc_now_iso()
    scored = 0
    skipped = 0
    for row in pairs:
        pair_id = str(row["pair_id"])
        block_id = str(row["block_id"])
        slide_capture_id = str(row["slide_capture_id"])
        block_result = _load_block_specimen(store, session_number, block_id)
        if isinstance(block_result, PreparationFailure):
            warnings.warn(
                f"skipping {pair_id}: block mask unavailable ({block_result.reason})",
                stacklevel=2,
            )
            skipped += 1
            continue
        slide_mask_path = _slide_mask_path(store, session_number, slide_capture_id)
        slide_result = _load_slide_specimen(slide_mask_path)
        if isinstance(slide_result, PreparationFailure):
            warnings.warn(
                f"skipping {pair_id}: slide mask unavailable ({slide_result.reason})",
                stacklevel=2,
            )
            skipped += 1
            continue
        result = score_pair_result_routed(block_result, slide_result, item_id=pair_id)
        store.write_matching_pair_score(
            pair_id,
            classical_score=result.score,
            rank_for_block=None,
            metric=result.selected_metric,
            scored_at=scored_at,
        )
        scored += 1
    _assign_ranks_for_work_order(store, session_number, work_order)
    promoted = store.promote_matching_near_misses(
        session_number, work_order, margin=margin,
    )
    return scored, skipped, promoted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-root",
        type=Path,
        default=DEFAULT_LIVE_ROOT,
        help=f"processing root (default: {DEFAULT_LIVE_ROOT})",
    )
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--work-order", required=True)
    parser.add_argument(
        "--include-true-pairs",
        action="store_true",
        help="also score true_pair rows with null classical_score",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=MATCH_MARGIN,
        help=f"near-miss promotion band (default: {MATCH_MARGIN}, MATCH_MARGIN)",
    )
    args = parser.parse_args(argv)
    store = ProcessingStore(args.live_root, recover_jobs=False)
    session = (
        args.session if args.session is not None else store.latest_session_number()
    )
    scored, skipped, promoted = score_work_order(
        store,
        session,
        args.work_order,
        include_true_pairs=args.include_true_pairs,
        margin=args.margin,
    )
    print(
        "scored "
        f"{scored} matching_pairs rows for session={session} wo={args.work_order} "
        f"(skipped={skipped}, promoted_near_miss={promoted})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
