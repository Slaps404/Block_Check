"""Inbound HTTP receiver and JSON-RPC dispatch for session workflow."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from slide.qr import SlideQRResult
import transport.wire as store_wire
from transport.rpc_table import (
    RPC_ARITY as _RPC_ARITY,
    RPC_METHODS as _RPC_METHODS,
    BARE_DICT_TUPLE_METHODS as _BARE_DICT_TUPLE_METHODS,
)
from session.atomic_io import atomic_bytes as _atomic_bytes
from session.outbox_transport import default_debug_snap_dir, save_debug_snap

if TYPE_CHECKING:
    from session.workflow import ProcessingStore


# `_RPC_METHODS` (the getattr-free dispatch whitelist -- a plain dict of
# name -> callable is the SOLE authority for what a remote caller can
# trigger, so a typo'd or malicious method name can never reach a real (and
# possibly private/dunder) attribute via getattr/eval) and `_RPC_ARITY` (the
# accepted positional-argument count per method, validated at the envelope
# boundary so malformed requests become stable JSON 400 responses instead of
# leaking IndexError out of a request-handler thread) are DERIVED from
# the single declarative table in `transport.rpc_table` -- see that module's
# docstring for the full design. `receive_capture` and `snapshot` are
# deliberately absent from the table: they keep their explicit
# `/captures` and `/status` routes and are not reachable via /rpc.
#
# `_BARE_DICT_TUPLE_METHODS` -- methods whose result is a tuple of plain
# dicts (SQLite rows already ``dict()``-ed), which must always serialize as
# a *bare* JSON list even when empty, because the client decodes them with
# ``tuple(json.loads(...))`` rather than ``store_wire.loads_list`` -- is
# likewise derived, from each op's ``bare_dict_tuple`` flag.


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

    `sqlite3.Error` (e.g. `OperationalError`) is caught here too:
    uncaught, a SQL-layer defect (such as a schema migration gap) would
    escape this method, and the stdlib request-handler thread would drop
    the connection with no HTTP response, which no client-side
    retry/timeout logic can recover from.
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
        # the store calls below skip the ledger.
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


def _receive_framed_upload(handler: BaseHTTPRequestHandler, noun: str):
    """Parse one 8-byte-length-prefixed metadata frame plus trailing image bytes.

    `noun` only shapes the error text, so a truncated `/slides` body and a
    truncated `/recapture-slide` body still report distinguishable errors.
    """
    size = int(handler.headers.get("Content-Length", "0"))
    framed = handler.rfile.read(size)
    if len(framed) < 9:
        raise ValueError(f"{noun} body is truncated")
    metadata_size = int.from_bytes(framed[:8], "big")
    if metadata_size <= 0 or metadata_size > len(framed) - 8:
        raise ValueError(f"{noun} metadata length is invalid")
    metadata = json.loads(framed[8:8 + metadata_size].decode("utf-8"))
    return metadata, framed[8 + metadata_size:]


def _dispatch_framed_upload(
    store: "ProcessingStore",
    session_number: int,
    handler: BaseHTTPRequestHandler,
    noun: str,
    call,
) -> bytes:
    """Shared receive path for both framed PNG upload routes.

    Never trusts a Pi-local path: the image is written to a session-scoped
    temporary file and removed in `finally`, whatever the store call does.
    `call(metadata, temporary)` returns the already-encoded response body, which
    is what distinguishes the two routes (a bare capture-id string on `/slides`,
    a `RecaptureOutcome` envelope on `/recapture-slide`) and is why they must
    stay separate routes rather than one overloaded one.
    """
    temporary: Path | None = None
    try:
        metadata, image = _receive_framed_upload(handler, noun)
        session = store.resume_session(session_number)
        incoming = session.directory / ".incoming_slides"
        incoming.mkdir(exist_ok=True)
        temporary = incoming / f"{uuid4().hex}.png"
        _atomic_bytes(temporary, image)
        body = call(metadata, temporary)
        handler.send_response(200)
        return body
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        handler.send_response(400)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    except (OSError, RuntimeError) as exc:
        handler.send_response(500)
        return json.dumps({"error": str(exc)}).encode("utf-8")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dispatch_slide_upload(
    store: "ProcessingStore", session_number: int, handler: BaseHTTPRequestHandler
) -> bytes:
    """Receive framed metadata + PNG bytes without trusting a Pi-local path."""

    def _call(metadata, temporary):
        capture_id = store.record_slide_capture(
            session_number,
            temporary,
            captured_at=datetime.fromisoformat(metadata["captured_at"]),
            result=store_wire.decode(SlideQRResult, metadata["result"]),
            duration_ms=metadata["duration_ms"],
            request_id=metadata.get("request_id"),
            source_token=metadata.get("source_token"),
            # Thread the durable Hybrid scheduling-order key across the
            # wire. Absent for ordinary slide uploads (no client currently
            # sends a non-None priority over this route), which keeps
            # ordinary FIFO order.
            priority=metadata.get("priority"),
            # Thread the `--profile` gate across the wire, mirroring
            # `priority` immediately above. Absent (older clients)
            # defaults to `False`, so profiling stays opt-in by construction
            # -- nothing is collected or persisted unless a caller
            # explicitly sends `"profile": true`.
            profile=bool(metadata.get("profile", False)),
        )
        return json.dumps(capture_id).encode("utf-8")

    return _dispatch_framed_upload(
        store, session_number, handler, "slide upload", _call
    )


def _dispatch_slide_recapture(
    store: "ProcessingStore", session_number: int, handler: BaseHTTPRequestHandler
) -> bytes:
    """Receive one recapture's framed metadata + PNG bytes.

    Shares `_dispatch_framed_upload`'s envelope with `/slides` but dispatches to
    `store.recapture_hybrid_slide` and returns the `RecaptureOutcome` wire
    envelope instead of a bare capture-id string.
    """

    def _call(metadata, temporary):
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
        return store_wire.dumps(outcome).encode("utf-8")

    return _dispatch_framed_upload(
        store, session_number, handler, "recapture upload", _call
    )


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
                        open_image = outer.open_debug_snaps
                        open_header = self.headers.get("X-Debug-Snap-Open")
                        if open_header is not None:
                            if open_header.lower() == "true":
                                open_image = True
                            elif open_header.lower() == "false":
                                open_image = False
                            else:
                                raise ValueError(
                                    "X-Debug-Snap-Open must be true or false"
                                )
                        saved = save_debug_snap(
                            self.rfile.read(size),
                            dest_dir=outer.debug_snap_dir,
                            open_image=open_image,
                            label=self.headers.get("X-Debug-Snap-Label"),
                            note=unquote(self.headers.get("X-Debug-Snap-Note", "")),
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
        # Short poll so __exit__ returns fast (stdlib default waits 0.5 s).
        self._thread = Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05}, daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)
