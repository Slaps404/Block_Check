"""Inbound HTTP receiver and JSON-RPC dispatch for session workflow (#201 slice 3)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from threading import Thread
from typing import TYPE_CHECKING, Callable
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from slide.qr import SlideQRResult
import store.wire as store_wire
from session.atomic_io import atomic_bytes as _atomic_bytes
from session.outbox_transport import default_debug_snap_dir, save_debug_snap

if TYPE_CHECKING:
    from session.workflow import ProcessingStore


# Explicit, getattr-free whitelist of the ProcessingStore surface the
# /rpc route may invoke. Mirrors the shape of session_console._COMMANDS:
# a plain dict of name -> lambda is the SOLE authority for what a remote
# caller can trigger, so a typo'd or malicious method name can never reach
# a real (and possibly private/dunder) attribute via getattr/eval. Every
# lambda has the shape (store, session_number, args) -> result, where
# `args` is the JSON body's decoded ``args`` list. `receive_capture` and
# `snapshot` are deliberately absent: they keep their existing explicit
# `/captures` and `/status` routes and are not reachable via /rpc.
#
# This dict, `_RPC_ARITY` below, and the public `RemoteProcessingStore` proxy
# surface (code/store/remote.py) are hand-maintained in three places; keeping
# them in sync is asserted by
# test_architecture_boundaries.test_rpc_whitelist_arity_and_proxy_surface_stay_in_sync.
_RPC_METHODS: dict[
    str, Callable[["ProcessingStore", int, list, "str | None"], object]
] = {
    "start_session": lambda store, n, args, request_id: store.start_session(
        started_at=datetime.fromisoformat(args[0]),
        session_mode=(args[1] if len(args) > 1 else "normal"),
        request_id=request_id,
    ),
    "resume_session": lambda store, n, args, request_id: store.resume_session(n),
    "scan_block": lambda store, n, args, request_id: store.scan_block(
        n, args[0], request_id=request_id
    ),
    "awaiting_capture_blocks": (
        lambda store, n, args, request_id: store.awaiting_capture_blocks(n)
    ),
    "unscan_block": lambda store, n, args, request_id: store.unscan_block(
        n, args[0], request_id=request_id
    ),
    "begin_block_drain": lambda store, n, args, request_id: store.begin_block_drain(n),
    "try_enter_slides": lambda store, n, args, request_id: store.try_enter_slides(n),
    "freeze_hybrid_pool": lambda store, n, args, request_id: store.freeze_hybrid_pool(
        n, descriptor_names=args[0],
        candidate_configuration=(args[1] if len(args) > 1 else None), request_id=request_id
    ),
    "reconcile_session_metadata": (
        lambda store, n, args, request_id: store.reconcile_session_metadata(n)
    ),
    "begin_finalization": lambda store, n, args, request_id: store.begin_finalization(n),
    "prepare_finalization": (
        lambda store, n, args, request_id: store.prepare_finalization(n)
    ),
    "complete_finalization": (
        lambda store, n, args, request_id: store.complete_finalization(n)
    ),
    "wait_for_jobs": lambda store, n, args, request_id: store.wait_for_jobs(),
    "get_set": lambda store, n, args, request_id: store.get_set(n, args[0]),
    "precheck_slide_scan": (
        lambda store, n, args, request_id: store.precheck_slide_scan(n, args[0])
    ),
    "record_slide_capture": lambda store, n, args, request_id: store.record_slide_capture(
        n,
        args[0],
        captured_at=datetime.fromisoformat(args[1]),
        result=store_wire.decode(SlideQRResult, args[2]),
        duration_ms=args[3],
        request_id=request_id,
        # #256: optional 5th positional arg -- the durable Hybrid job
        # scheduling-order key #255 added to the store method but never
        # threaded across this whitelist. Absent (every pre-#256 caller)
        # keeps today's behavior: an ordinary FIFO job.
        priority=(args[4] if len(args) > 4 else None),
    ),
    "retry_hybrid_slide": (
        lambda store, n, args, request_id: store.retry_hybrid_slide(
            n, args[0], request_id=request_id
        )
    ),
    "recapture_hybrid_slide": (
        lambda store, n, args, request_id: store.recapture_hybrid_slide(
            n,
            args[0],
            args[1],
            captured_at=datetime.fromisoformat(args[2]),
            result=store_wire.decode(SlideQRResult, args[3]),
            duration_ms=args[4],
            request_id=request_id,
        )
    ),
    "resolve_claim": lambda store, n, args, request_id: store.resolve_claim(
        n, args[0], args[1], args[2], request_id=request_id
    ),
    "slide_captures": lambda store, n, args, request_id: store.slide_captures(n),
    "slide_recovery_state": (
        lambda store, n, args, request_id: store.slide_recovery_state(n)
    ),
    "skip_unreadable_slide": (
        lambda store, n, args, request_id: store.skip_unreadable_slide(
            n, request_id=request_id
        )
    ),
    "mark_waiting_for_slide": (
        lambda store, n, args, request_id: store.mark_waiting_for_slide(
            n, request_id=request_id
        )
    ),
    "active_warnings": lambda store, n, args, request_id: store.active_warnings(n),
    "summarize": lambda store, n, args, request_id: store.summarize(store.resume_session(n)),
    "dismiss_block": lambda store, n, args, request_id: store.dismiss_block(
        n, args[0], reason=args[1], request_id=request_id
    ),
    "block_readiness": lambda store, n, args, request_id: store.block_readiness(n, args[0]),
    "events": lambda store, n, args, request_id: store.events(n),
    "record_event": lambda store, n, args, request_id: store.record_event(
        n, args[0], args[1], request_id=request_id
    ),
    "record_finalization_error": (
        lambda store, n, args, request_id: store.record_finalization_error(
            n, args[0], reconcile=(args[1] if len(args) > 1 else True)
        )
    ),
    "record_profile_capture": (
        lambda store, n, args, request_id: store.record_profile_capture(
            n, args[0], args[1], request_id=request_id
        )
    ),
    "record_slide_benchmark": (
        lambda store, n, args, request_id: store.record_slide_benchmark(
            n, args[0], args[1]
        )
    ),
    "start_work_order": lambda store, n, args, request_id: store.start_work_order(n),
    "finish_work_order": lambda store, n, args, request_id: store.finish_work_order(
        n, start_job=(args[0] if args else True), request_id=request_id
    ),
    "open_work_order_id": lambda store, n, args, request_id: store.open_work_order_id(n),
    "has_work_orders": lambda store, n, args, request_id: store.has_work_orders(n),
    "list_results_ready_work_orders": (
        lambda store, n, args, request_id: store.list_results_ready_work_orders(n)
    ),
    "list_hybrid_results": (
        lambda store, n, args, request_id: store.list_hybrid_results(n)
    ),
    "list_retrieval_results": (
        lambda store, n, args, request_id: store.list_retrieval_results(n)
    ),
    "list_hybrid_profile_rows": (
        lambda store, n, args, request_id: store.list_hybrid_profile_rows(n)
    ),
}

# Accepted positional-argument counts for each public RPC method.  Validate at
# the envelope boundary so malformed requests become stable JSON 400 responses
# instead of leaking IndexError out of a request-handler thread.
_RPC_ARITY: dict[str, tuple[int, int]] = {
    "start_session": (1, 2), "resume_session": (0, 0), "scan_block": (1, 1),
    "awaiting_capture_blocks": (0, 0), "unscan_block": (1, 1),
    "begin_block_drain": (0, 0), "try_enter_slides": (0, 0),
    "freeze_hybrid_pool": (1, 2),
    "reconcile_session_metadata": (0, 0), "begin_finalization": (0, 0),
    "prepare_finalization": (0, 0), "complete_finalization": (0, 0),
    "wait_for_jobs": (0, 0), "get_set": (1, 1),
    "precheck_slide_scan": (1, 1),
    "record_slide_capture": (4, 5), "resolve_claim": (3, 3),
    "retry_hybrid_slide": (1, 1), "recapture_hybrid_slide": (5, 5),
    "slide_captures": (0, 0), "slide_recovery_state": (0, 0),
    "skip_unreadable_slide": (0, 0), "mark_waiting_for_slide": (0, 0),
    "active_warnings": (0, 0), "summarize": (0, 0),
    "dismiss_block": (2, 2), "block_readiness": (1, 1), "events": (0, 0),
    "record_event": (2, 2), "record_finalization_error": (1, 2),
    "record_profile_capture": (2, 2), "record_slide_benchmark": (2, 2),
    "start_work_order": (0, 0), "finish_work_order": (0, 1),
    "list_results_ready_work_orders": (0, 0), "open_work_order_id": (0, 0),
    "has_work_orders": (0, 0), "list_hybrid_results": (0, 0),
    "list_retrieval_results": (0, 0),
    "list_hybrid_profile_rows": (0, 0),
}


# Methods whose result is a tuple of plain dicts (SQLite rows already
# ``dict()``-ed). These must always serialize as a *bare* JSON list -- even
# when empty -- because the client decodes them with ``tuple(json.loads(...))``,
# not ``store_wire.loads_list``. The generic ``_serialize_rpc_result`` cannot
# tell an empty dict-tuple from an empty dataclass-tuple by value alone, so it
# would otherwise route an empty result down the dataclass-envelope path and
# hand the client a shape it cannot read (the #149 empty-results-table case).
_BARE_DICT_TUPLE_METHODS = frozenset({
    "slide_captures",
    "list_results_ready_work_orders",
    "list_hybrid_results",
    "list_retrieval_results",
    "list_hybrid_profile_rows",
})


def _serialize_rpc_result(result: object) -> str:
    """Encode one store return value for the /rpc response body.

    Dataclasses and homogeneous dataclass sequences round-trip through the
    self-describing store_wire envelope so the caller can `store_wire.loads`
    them back into typed instances. `get_set`/`slide_captures` return plain
    dicts (SQLite rows already ``dict()``-ed) with no dataclass to name, so
    those pass through as bare JSON instead.
    """
    if result is None:
        return json.dumps(None)
    if is_dataclass(result) and not isinstance(result, type):
        return store_wire.dumps(result)
    if isinstance(result, (tuple, list)):
        if any(isinstance(item, dict) for item in result):
            return json.dumps(store_wire.passthrough_dict_tuple(result))
        return store_wire.dumps_list(result)
    if isinstance(result, dict):
        return json.dumps(store_wire.passthrough_dict(result))
    if isinstance(result, (bool, int, float, str)):
        return json.dumps(result)
    raise TypeError(f"cannot serialize rpc result of type {type(result)!r}")


def _dispatch_rpc(
    store: "ProcessingStore", session_number: int, handler: BaseHTTPRequestHandler
) -> bytes:
    """Decode one JSON-RPC envelope, dispatch via the getattr-free whitelist
    above, and return the response body. Also sets the response status on
    `handler`, so the caller only has to write the shared header tail.

    The 400 vs 500 split is deliberate: 400 means the *request* is wrong
    (bad JSON, unknown method, a store call rejecting its own arguments) and
    retrying the identical request will never help. 500 means the
    whitelisted store call hit an infrastructure failure (disk I/O, a
    broken runtime invariant) where an identical retry might succeed.

    #269 FIX2b: `sqlite3.Error` (e.g. `OperationalError`) is caught here
    too. Before this, neither this 500 handler nor the 400 handler above
    caught it, so a SQL-layer defect (a legacy-schema migration gap, or any
    future one) escaped this method entirely -- the stdlib request-handler
    thread prints a bare traceback and drops the connection with no HTTP
    response at all, which no client-side retry/timeout logic can recover
    from.
    """
    try:
        size = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(size)
        envelope = json.loads(raw)
        method = envelope["method"]
        if not isinstance(method, str):
            raise TypeError("method must be a string")
        args = envelope.get("args", [])
        if not isinstance(args, list):
            raise ValueError("args must be a list")
        # The client idempotency key (ADR 0002): optional, forwarded verbatim
        # into the four mutating methods whose ledger this activates. Absent
        # entirely (older client, or a read-only method), it stays None and
        # every store call below behaves exactly as it did before the ledger.
        request_id = envelope.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        try:
            rpc_method = _RPC_METHODS[method]
        except KeyError:
            raise ValueError(f"unknown method: {method}") from None
        minimum, maximum = _RPC_ARITY[method]
        if not minimum <= len(args) <= maximum:
            expected = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
            noun = "argument" if minimum == maximum == 1 else "arguments"
            raise ValueError(
                f"{method} expects {expected} {noun}; received {len(args)}"
            )
        result = rpc_method(store, session_number, args, request_id)
        if method in _BARE_DICT_TUPLE_METHODS:
            payload = json.dumps(
                store_wire.passthrough_dict_tuple(result)
            ).encode("utf-8")
        else:
            payload = _serialize_rpc_result(result).encode("utf-8")
        handler.send_response(200)
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        handler.send_response(400)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        handler.send_response(500)
        return json.dumps({"error": str(exc)}).encode("utf-8")


def _dispatch_slide_upload(
    store: "ProcessingStore", session_number: int, handler: BaseHTTPRequestHandler
) -> bytes:
    """Receive framed metadata + PNG bytes without trusting a Pi-local path."""
    temporary: Path | None = None
    try:
        size = int(handler.headers.get("Content-Length", "0"))
        framed = handler.rfile.read(size)
        if len(framed) < 9:
            raise ValueError("slide upload body is truncated")
        metadata_size = int.from_bytes(framed[:8], "big")
        if metadata_size <= 0 or metadata_size > len(framed) - 8:
            raise ValueError("slide upload metadata length is invalid")
        metadata = json.loads(framed[8:8 + metadata_size].decode("utf-8"))
        image = framed[8 + metadata_size:]
        session = store.resume_session(session_number)
        incoming = session.directory / ".incoming_slides"
        incoming.mkdir(exist_ok=True)
        temporary = incoming / f"{uuid4().hex}.png"
        _atomic_bytes(temporary, image)
        capture_id = store.record_slide_capture(
            session_number,
            temporary,
            captured_at=datetime.fromisoformat(metadata["captured_at"]),
            result=store_wire.decode(SlideQRResult, metadata["result"]),
            duration_ms=metadata["duration_ms"],
            request_id=metadata.get("request_id"),
            source_token=metadata.get("source_token"),
            # #256: thread the durable Hybrid scheduling-order key across the
            # wire. Absent for every ordinary slide upload (every pre-#256
            # client, and every real client today -- nothing yet sends a
            # non-None priority over this route), so this preserves today's
            # ordinary-FIFO behavior unchanged.
            priority=metadata.get("priority"),
            # #258: thread the `--profile` gate across the wire, mirroring
            # `priority` immediately above. Absent (every pre-#258 client)
            # defaults to `False`, so profiling stays opt-in by construction
            # -- nothing is collected or persisted unless a caller
            # explicitly sends `"profile": true`.
            profile=bool(metadata.get("profile", False)),
        )
        handler.send_response(200)
        return json.dumps(capture_id).encode("utf-8")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        handler.send_response(400)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    except (OSError, RuntimeError) as exc:
        handler.send_response(500)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dispatch_slide_recapture(
    store: "ProcessingStore", session_number: int, handler: BaseHTTPRequestHandler
) -> bytes:
    """#256: receive one recapture's framed metadata + PNG bytes, mirroring
    `_dispatch_slide_upload`'s shape exactly (same 8-byte length-prefixed
    metadata envelope) but dispatching to `store.recapture_hybrid_slide`
    instead of `store.record_slide_capture`, and returning the
    `RecaptureOutcome` wire envelope instead of a bare capture-id string.
    A separate route (rather than overloading `/slides`) keeps the ordinary
    slide-upload contract and this one's different response shape from
    ever being confused with each other.
    """
    temporary: Path | None = None
    try:
        size = int(handler.headers.get("Content-Length", "0"))
        framed = handler.rfile.read(size)
        if len(framed) < 9:
            raise ValueError("recapture upload body is truncated")
        metadata_size = int.from_bytes(framed[:8], "big")
        if metadata_size <= 0 or metadata_size > len(framed) - 8:
            raise ValueError("recapture upload metadata length is invalid")
        metadata = json.loads(framed[8:8 + metadata_size].decode("utf-8"))
        image = framed[8 + metadata_size:]
        session = store.resume_session(session_number)
        incoming = session.directory / ".incoming_slides"
        incoming.mkdir(exist_ok=True)
        temporary = incoming / f"{uuid4().hex}.png"
        _atomic_bytes(temporary, image)
        outcome = store.recapture_hybrid_slide(
            session_number,
            metadata["superseded_capture_id"],
            temporary,
            captured_at=datetime.fromisoformat(metadata["captured_at"]),
            result=store_wire.decode(SlideQRResult, metadata["result"]),
            duration_ms=metadata["duration_ms"],
            request_id=metadata.get("request_id"),
            source_token=metadata.get("source_token"),
        )
        handler.send_response(200)
        return store_wire.dumps(outcome).encode("utf-8")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        handler.send_response(400)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    except (OSError, RuntimeError) as exc:
        handler.send_response(500)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class LoopbackCaptureReceiver:
    """Replaceable HTTP receiver adapter, bindable to loopback for tests."""

    def __init__(
        self,
        store: ProcessingStore,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        debug_snap_dir: Path | None = None,
        open_debug_snaps: bool = True,
    ):
        self.store = store
        self.debug_snap_dir = (
            Path(debug_snap_dir) if debug_snap_dir is not None else default_debug_snap_dir()
        )
        self.open_debug_snaps = open_debug_snaps
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib interface
                try:
                    parsed = urlsplit(self.path)
                    parts = parsed.path.strip("/").split("/")
                    if len(parts) != 3 or parts[0] != "sessions":
                        raise ValueError("unknown receiver path")
                    session_number = int(parts[1])
                    if parts[2] == "status":
                        session = outer.store.resume_session(session_number)
                        snapshot = outer.store.snapshot(session)
                        payload = json.dumps(asdict(snapshot), default=str).encode("utf-8")
                        content_type = "application/json"
                        self.send_response(200)
                    elif parts[2] == "evidence":
                        artifact_path = parse_qs(parsed.query).get("path", [""])[0]
                        payload = outer.store.results_evidence_bytes(
                            session_number, artifact_path
                        ) if artifact_path else None
                        if payload is None:
                            payload = b""
                            self.send_response(404)
                        else:
                            self.send_response(200)
                        content_type = "image/jpeg"
                    else:
                        raise ValueError("unknown receiver path")
                except (TypeError, ValueError) as exc:
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    content_type = "application/json"
                    self.send_response(400)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802 - stdlib interface
                try:
                    parts = self.path.strip("/").split("/")
                    if parts == ["debug", "snap"]:
                        size = int(self.headers.get("Content-Length", "0"))
                        if size <= 0:
                            raise ValueError("debug snap body is empty")
                        saved = save_debug_snap(
                            self.rfile.read(size),
                            dest_dir=outer.debug_snap_dir,
                            open_image=outer.open_debug_snaps,
                        )
                        payload = json.dumps({"path": str(saved)}).encode("utf-8")
                        self.send_response(200)
                    elif len(parts) != 3 or parts[0] != "sessions":
                        raise ValueError("unknown receiver path")
                    elif parts[2] == "captures":
                        size = int(self.headers.get("Content-Length", "0"))
                        receipt = outer.store.receive_capture(
                            int(parts[1]),
                            capture_id=self.headers["X-Capture-Id"],
                            block_id=self.headers["X-Block-Id"],
                            checksum=self.headers["X-Checksum-Sha256"],
                            body=self.rfile.read(size),
                            recapture=self.headers.get("X-Block-Recapture") == "true",
                            profile=self.headers.get("X-Profile") == "true",
                        )
                        payload = json.dumps(asdict(receipt)).encode("utf-8")
                        self.send_response(200)
                    elif parts[2] == "rpc":
                        # _dispatch_rpc sets its own response status (200/400/500)
                        # and returns the body; only the header tail is shared.
                        payload = _dispatch_rpc(outer.store, int(parts[1]), self)
                    elif parts[2] == "slides":
                        payload = _dispatch_slide_upload(
                            outer.store, int(parts[1]), self
                        )
                    elif parts[2] == "recapture-slide":
                        payload = _dispatch_slide_recapture(
                            outer.store, int(parts[1]), self
                        )
                    elif parts[2] == "profile-curve":
                        size = int(self.headers.get("Content-Length", "0"))
                        body = self.rfile.read(size)
                        identity = outer.store._session_identity(int(parts[1]))
                        destination = identity.directory / "motion_curve.csv"
                        _atomic_bytes(destination, body)
                        payload = json.dumps(
                            {"path": str(destination)}
                        ).encode("utf-8")
                        self.send_response(200)
                    elif parts[2] == "profile-config":
                        size = int(self.headers.get("Content-Length", "0"))
                        body = self.rfile.read(size)
                        identity = outer.store._session_identity(int(parts[1]))
                        destination = identity.directory / "profile_config.json"
                        _atomic_bytes(destination, body)
                        payload = json.dumps(
                            {"path": str(destination)}
                        ).encode("utf-8")
                        self.send_response(200)
                    else:
                        raise ValueError("unknown receiver path")
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "LoopbackCaptureReceiver":
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)
