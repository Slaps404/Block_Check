"""Inventory saved block/slide images in sessions.sqlite3 without scoring."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from session.corpus_inventory import (  # noqa: E402
    collect_corpus_inventory,
    summarize_corpus_inventory,
    write_corpus_inventory_csv,
)

DEFAULT_LIVE_ROOT = ROOT / "outputs" / "live_session"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-root",
        type=Path,
        default=DEFAULT_LIVE_ROOT,
        help=f"processing root (default: {DEFAULT_LIVE_ROOT})",
    )
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    database_path = args.live_root / "sessions.sqlite3"
    rows = collect_corpus_inventory(database_path, session_number=args.session)
    summary = summarize_corpus_inventory(rows)
    print(
        f"sessions={summary.sessions} "
        f"work_order_brackets={summary.work_order_brackets} "
        f"named_work_orders={summary.named_work_orders} "
        f"blocks={summary.blocks} slides={summary.slides} "
        f"complete_claimed_pairs={summary.complete_claimed_pairs} "
        f"rows_with_issues={summary.rows_with_issues}"
    )
    if args.output is not None:
        write_corpus_inventory_csv(args.output, rows)
        print(f"wrote {len(rows)} inventory rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
