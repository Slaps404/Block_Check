"""Backfill matching_pairs from claimed slides in a processing root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from session.processing_store import ProcessingStore  # noqa: E402

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
    parser.add_argument("--work-order", required=True)
    args = parser.parse_args(argv)
    store = ProcessingStore(args.live_root, recover_jobs=False)
    session = args.session if args.session is not None else store.latest_session_number()
    n = store.sync_matching_pairs_for_work_order(session, args.work_order)
    print(f"synced {n} matching_pairs rows for session={session} wo={args.work_order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
