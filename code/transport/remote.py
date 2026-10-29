"""HTTP proxy that stands in for ``ProcessingStore`` across the wire.

``RemoteProcessingStore`` mirrors the ``/rpc``-whitelisted surface used by
``SessionWorkflow`` (see ``_RPC_METHODS`` there), so the workflow runs
identically with the in-process store or this deployed proxy. The deliberate
lifecycle exception is ``resume_session(None)``: latest-session discovery stays
processing-computer-local, and deployed callers pass the explicit session
number printed by ``run_receiver.py``. Every mutating call, and every read, goes over
``POST /sessions/{n}/rpc`` except ``snapshot`` (``GET /sessions/{n}/status``,
mirroring ``HttpCaptureClient.status``).

Two-level error model:

* ``StoreError`` -- the server answered with HTTP 400. That means the
  *request itself* is wrong (bad args, wrong phase, unknown session/block).
  Deterministic: retrying the identical request will never help, so it is
  raised immediately, without retry.
* ``TransportError`` -- the request never got a clean answer at all
  (connection refused/reset, timeout, ``URLError``, or HTTP 500 -- an
  infrastructure failure on the server side). This is the only exception
  the retry loop below acts on.

The request_id contract (ADR 0002, the load-bearing rule): one uuid4 hex is
minted per *logical* proxy call, in ``_rpc`` below, BEFORE the retry loop
starts, and that same id rides on every attempt. A fresh id per attempt
would silently defeat the server's ``request_ledger``: the server
dedupes on exact ``request_id`` match, so a retry that changed its id would
be executed twice instead of replayed once.
"""
from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import transport.wire as store_wire
from transport.rpc_table import PROXY_OPS, invoke_proxy, validate_rpc_payload
from session.workflow import (
    RecaptureOutcome,
    SlideQRResult,
    SessionIdentity,
    SessionSummary,
    WorkflowSnapshot,
)


class StoreError(ValueError):
    """The server rejected the request itself (HTTP 400). Do not retry."""


class TransportError(RuntimeError):
    """The request did not get a clean answer (connection failure or HTTP 500).

    Safe -- and expected -- to retry with the SAME request_id.
    """


def _error_message(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")
    if isinstance(payload, dict) and "error" in payload:
        return str(payload["error"])
    return body.decode("utf-8", errors="replace")


class UrlTransport:
    """Default transport: real HTTP via ``urllib``.

    Kept as a small, replaceable seam (like ``HttpCaptureClient`` already is)
    so tests can wrap or fake it -- e.g. to drop a response after the real
    server has already durably applied the mutation, or to count/deny calls.
    """

    def __init__(self, timeout: float = 10):
        self.timeout = timeout

    def post(self, url: str, payload: bytes) -> bytes:
        request = Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._send(request)

    def post_binary(self, url: str, payload: bytes) -> bytes:
        request = Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        return self._send(request)

    def get(self, url: str) -> bytes:
        return self._send(Request(url))

    def _send(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            message = _error_message(exc.read())
            if exc.code == 400:
                raise StoreError(message) from exc
            raise TransportError(f"HTTP {exc.code}: {message}") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise TransportError(str(exc)) from exc


class RemoteProcessingStore:
    """Transparent ``ProcessingStore``-shaped proxy over ``/rpc`` and ``/status``.

    ``resume_session`` requires an explicit number; unlike the local store it
    does not discover the latest session across the machine boundary.

    Two methods on the real ``ProcessingStore`` -- ``start_session`` (no
    session exists yet) and ``wait_for_jobs`` (waits across every session's
    executor, not one session's) -- do not take a session number. The server
    route is still ``/sessions/{n}/rpc`` and its dispatch table ignores ``n``
    for exactly those two methods (see ``_RPC_METHODS`` in
    ``session_workflow.py``), so this proxy sends a placeholder ``0`` for
    them and keeps their public signature identical to the real store's.
    """

    _PLACEHOLDER_SESSION = 0

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10,
        max_attempts: int = 3,
        backoff: float = 0.05,
        transport: "UrlTransport | object | None" = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff = backoff
        self._transport = transport if transport is not None else UrlTransport(timeout)

    # ------------------------------------------------------------------
    # transport plumbing
    # ------------------------------------------------------------------

    def _rpc(
        self,
        session_number: int,
        method: str,
        args: Sequence[object],
        *,
        request_id: str | None = None,
    ) -> bytes:
        """POST one JSON-RPC envelope, retrying only on ``TransportError``.

        The request_id is minted ONCE, right here, before the retry loop --
        never inside it -- so every attempt (including retries after a
        dropped response) carries the identical id the server's ledger keys
        on.
        """
        stable_request_id = request_id if request_id is not None else uuid.uuid4().hex
        payload = json.dumps({
            "method": method,
            "args": list(args),
            "request_id": stable_request_id,
        }).encode("utf-8")
        url = f"{self.base_url}/sessions/{session_number}/rpc"
        attempt = 0
        while True:
            attempt += 1
            try:
                body = self._transport.post(url, payload)
                try:
                    validate_rpc_payload(method, body)
                except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                    raise TransportError(
                        "HTTP 200 response was not valid UTF-8 JSON"
                    ) from exc
                return body
            except TransportError:
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)

    def _get(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                body = self._transport.get(url)
                try:
                    json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise TransportError(
                        "HTTP 200 response was not valid UTF-8 JSON"
                    ) from exc
                return body
            except TransportError:
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)

    def _get_binary_optional(self, path: str) -> bytes | None:
        """Fetch an optional binary artifact without applying JSON validation."""
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._transport.get(url)
            except TransportError as exc:
                # The receiver deliberately uses 404 for an absent or invalid
                # session artifact.  It is not a transport outage and should
                # become the kiosk's ordinary missing-image response.
                if str(exc).startswith("HTTP 404:"):
                    return None
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def resume_session(self, session_number: int | None = None) -> SessionIdentity:
        if session_number is None:
            raise ValueError(
                "RemoteProcessingStore.resume_session requires an explicit "
                "session_number: the /rpc route binds the session from the "
                "URL path, so there is no way to ask the server to resume "
                "'whatever the latest session is' over the wire."
            )
        body = self._rpc(session_number, "resume_session", [])
        return store_wire.loads(body.decode("utf-8"))

    # ------------------------------------------------------------------
    # slides / claims
    # ------------------------------------------------------------------

    def _post_framed_upload(self, url, metadata, image, decode, invalid_message):
        """Post one framed metadata-plus-PNG upload, with the shared retry policy.

        The frame is an 8-byte big-endian metadata length, then the UTF-8 JSON
        metadata, then the raw image bytes. `decode` both validates and decodes
        the response; it runs inside the retry loop, so a malformed HTTP 200
        body is retried exactly like a transport failure rather than being
        returned to the caller.
        """
        payload = json.dumps(metadata).encode("utf-8")
        framed = len(payload).to_bytes(8, "big") + payload + image
        attempt = 0
        while True:
            attempt += 1
            try:
                post_binary = getattr(self._transport, "post_binary", None)
                if post_binary is None:
                    raise TransportError("transport does not support binary slide upload")
                body = post_binary(url, framed)
                try:
                    return decode(body)
                except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                    raise TransportError(invalid_message) from exc
            except TransportError:
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)

    def record_slide_capture(
        self,
        session_number: int,
        source: str | Path,
        *,
        captured_at: datetime,
        result: SlideQRResult,
        duration_ms: float,
        request_id: str | None = None,
        priority: int | None = None,
        profile: bool = False,
    ) -> str:
        stable_request_id = request_id if request_id is not None else uuid.uuid4().hex
        image = Path(source).read_bytes()
        metadata = {
            "captured_at": store_wire.encode(captured_at),
            "result": store_wire.encode(result),
            "duration_ms": duration_ms,
            "request_id": stable_request_id,
            # Durable Hybrid scheduling-order key. `None` (ordinary callers)
            # keeps ordinary FIFO ordering on the server.
            "priority": priority,
            # `--profile` gate. `False` (the default) keeps the upload
            # unprofiled.
            "profile": profile,
            "source_token": hashlib.sha256(image).hexdigest(),
        }
        return self._post_framed_upload(
            f"{self.base_url}/sessions/{session_number}/slides",
            metadata,
            image,
            lambda body: json.loads(body.decode("utf-8")),
            "HTTP 200 response was not valid UTF-8 JSON",
        )

    def retry_hybrid_slide(
        self, session_number: int, capture_id: str, *, request_id: str | None = None,
    ) -> bool:
        body = self._rpc(
            session_number, "retry_hybrid_slide", [capture_id], request_id=request_id,
        )
        return bool(json.loads(body.decode("utf-8")))

    def recapture_hybrid_slide(
        self,
        session_number: int,
        superseded_capture_id: str,
        source: str | Path,
        *,
        captured_at: datetime,
        result: SlideQRResult,
        duration_ms: float,
        request_id: str | None = None,
        source_token: str | None = None,
    ) -> RecaptureOutcome:
        """Binary-upload counterpart of `record_slide_capture` for an
        accepted recapture: same framed-metadata-plus-PNG-bytes shape, but
        posted to the dedicated `/recapture-slide` route (never `/slides`,
        so the two response shapes -- a bare capture-id string here vs. a
        `RecaptureOutcome` envelope -- can never be confused) and decoded as
        a `RecaptureOutcome` instead of a bare capture id.
        """
        stable_request_id = request_id if request_id is not None else uuid.uuid4().hex
        image = Path(source).read_bytes()
        metadata = {
            "superseded_capture_id": superseded_capture_id,
            "captured_at": store_wire.encode(captured_at),
            "result": store_wire.encode(result),
            "duration_ms": duration_ms,
            "request_id": stable_request_id,
            "source_token": source_token or hashlib.sha256(image).hexdigest(),
        }
        return self._post_framed_upload(
            f"{self.base_url}/sessions/{session_number}/recapture-slide",
            metadata,
            image,
            lambda body: store_wire.loads_as(RecaptureOutcome, body.decode("utf-8")),
            "HTTP 200 response was not a valid RecaptureOutcome",
        )

    # ------------------------------------------------------------------
    # summary / snapshot
    # ------------------------------------------------------------------

    def summarize(self, session: SessionIdentity) -> SessionSummary:
        body = self._rpc(session.number, "summarize", [])
        return store_wire.loads(body.decode("utf-8"))

    def snapshot(self, session: SessionIdentity) -> WorkflowSnapshot:
        body = self._get(f"/sessions/{session.number}/status")
        return store_wire.loads_as(WorkflowSnapshot, body.decode("utf-8"))

    def results_evidence_bytes(
        self, session_number: int, artifact_path: str | Path
    ) -> bytes | None:
        """Fetch a session-scoped evidence JPEG from the processing computer.

        Hand-written: this speaks the dedicated binary `/evidence` route, not
        the plain-JSON `/rpc` envelope the table synthesizes.
        """
        from urllib.parse import urlencode

        query = urlencode({"path": str(artifact_path)})
        return self._get_binary_optional(
            f"/sessions/{session_number}/evidence?{query}"
        )

    def __getattr__(self, name: str):
        """Synthesize the generic proxy methods from the one declarative table.

        Python calls this only after normal attribute lookup fails, so the
        hand-written proxies above (and every real attribute) still win. Every
        other remote operation is built from its `RpcOp`, which is also what
        derives `RPC_METHODS`/`RPC_ARITY` on the server, so the three
        surfaces cannot drift apart.
        """
        op = PROXY_OPS.get(name)
        if op is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        def _call(*args: object, request_id: str | None = None, **kwargs: object):
            return invoke_proxy(self, op, args, kwargs, request_id)

        _call.__name__ = name
        _call.__qualname__ = f"{type(self).__name__}.{name}"
        return _call

    # ------------------------------------------------------------------
    # deliberately excluded
    # ------------------------------------------------------------------
    #
    # `receive_capture` is NOT implemented here. It already has a dedicated,
    # binary-body `/captures` route and an existing HTTP client
    # (`session_workflow.HttpCaptureClient.upload`) that speaks it with the
    # right headers (`X-Capture-Id`, `X-Block-Id`, `X-Checksum-Sha256`,
    # `X-Captured-At`, `X-Block-Recapture`). Re-implementing it here would
    # either duplicate that logic verbatim or thinly wrap it for no benefit;
    # callers that need `receive_capture` should keep using
    # `HttpCaptureClient` directly, exactly as `PiOutbox.replay` already does.
