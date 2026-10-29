"""Run the processing-computer capture receiver on the dedicated Ethernet link.

The receiver (``LoopbackCaptureReceiver``) defaults to loopback for tests. In a
live session it must bind to the processing computer's dedicated static address
so the Pi can reach it, and *only* that interface. Binding to a concrete host
IP is what enforces "only": the socket then listens on
that address alone, not on Wi-Fi or any other interface.

Usage (processing computer, PowerShell):
    .\\venv\\Scripts\\python.exe tools\\run_receiver.py --root outputs\\live_session ^
        --start-session

    # laptop-only shakeout, no Pi wired yet:
    .\\venv\\Scripts\\python.exe tools\\run_receiver.py --root outputs\\live_session ^
        --host 127.0.0.1 --start-session

    # restart after a crash/kill, same session identity:
    .\\venv\\Scripts\\python.exe tools\\run_receiver.py --root outputs\\live_session --resume 1

Pass exactly one of --start-session (new session) or --resume N (existing
session) so uploads from tools/feed_captures.py have a session to attach to;
the assigned/resumed session number is printed prominently at boot. Point
the Pi's HttpCaptureClient at the printed URL. Stop with Ctrl+C.
"""
from datetime import datetime, timezone
import argparse
import sys
import threading

from session.session_mode import (  # noqa: E402
    HYBRID_FLAG,
    HYBRID_SHADOW_FLAG,
    OPEN_RETRIEVAL_FLAG,
    SessionModeConflictError,
    resolve_session_mode,
)
from session.workflow import LoopbackCaptureReceiver, ProcessingStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Durable processing-computer session directory (SQLite + artifacts).",
    )
    parser.add_argument(
        "--host",
        default="192.168.50.1",
        help="Interface to bind. Default is the dedicated Ethernet address.",
    )
    parser.add_argument(
        "--port", type=int, default=8077, help="TCP port for the capture receiver."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Background block-preprocessing workers. Sweep this on the target "
        "computer to find the safe count (issue #101 AC#5).",
    )
    parser.add_argument(
        "--start-session",
        action="store_true",
        help="Start a new session at boot so uploads have something to attach to.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        metavar="N",
        help="Resume an existing session by number, for restart tests (issue #101 AC#3/#4).",
    )
    parser.add_argument(
        OPEN_RETRIEVAL_FLAG,
        action="store_true",
        help="Start a new session in open-retrieval mode (issue #147-#151). "
        f"Only meaningful with --start-session. Mutually exclusive with "
        f"{HYBRID_FLAG}/{HYBRID_SHADOW_FLAG}.",
    )
    parser.add_argument(
        HYBRID_FLAG,
        action="store_true",
        help="Start a new session in Hybrid mode (#269: durable "
        "sessions.session_mode, real work-order bracket). Only meaningful "
        f"with --start-session. Mutually exclusive with "
        f"{OPEN_RETRIEVAL_FLAG}/{HYBRID_SHADOW_FLAG}.",
    )
    parser.add_argument(
        HYBRID_SHADOW_FLAG,
        action="store_true",
        help="Start a new session in Hybrid Shadow mode. Only meaningful "
        f"with --start-session. Mutually exclusive with "
        f"{OPEN_RETRIEVAL_FLAG}/{HYBRID_FLAG}.",
    )
    args = parser.parse_args()

    if args.host in ("0.0.0.0", ""):
        print(
            "REFUSING to bind all interfaces: this exposes specimen captures beyond "
            "the isolated link. Pass the dedicated Ethernet IP explicitly.",
            file=sys.stderr,
        )
        return 2

    if args.start_session and args.resume is not None:
        print("Pass only one of --start-session or --resume.", file=sys.stderr)
        return 2

    # This is the only place the session mode reaches the durable
    # sessions.session_mode column: run_pi_session.py only resumes an
    # already-created session. Resolve/reject the mode before any store or
    # camera side effect, the same way run_pi_session.py does.
    try:
        session_mode = resolve_session_mode(
            open_retrieval=args.open_retrieval,
            hybrid=args.hybrid,
            hybrid_shadow=args.hybrid_shadow,
        )
    except SessionModeConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    store = ProcessingStore(args.root, workers=args.workers)

    session = None
    if args.start_session:
        session = store.start_session(
            started_at=datetime.now(timezone.utc), session_mode=session_mode.value
        )
    elif args.resume is not None:
        session = store.resume_session(args.resume)

    with LoopbackCaptureReceiver(store, host=args.host, port=args.port) as receiver:
        print(f"Capture receiver bound to {receiver.url}  (workers={args.workers})")
        if session is not None:
            print("=" * 60)
            print(f"SESSION {session.number}  (started_at={session.started_at.isoformat()})")
            print("=" * 60)
        print("Point the Pi HttpCaptureClient at this URL. Ctrl+C to stop.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down receiver...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
