"""HTTP proxy that stands in for ``ProcessingStore`` across the wire (#115).

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
would silently defeat the server's ``request_ledger`` (#114): the server
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
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import store.wire as store_wire
from session.workflow import (
    BlockReadiness,
    ClaimOutcome,
    FailedBlockWarning,
    HybridPoolFreezeResult,
    RecaptureOutcome,
    ScanOutcome,
    SessionIdentity,
    SessionSummary,
    SlideQRResult,
    WorkflowEvent,
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


_WIRE_RESULT_TYPES = {
    "start_session": SessionIdentity,
    "resume_session": SessionIdentity,
    "scan_block": ScanOutcome,
    "block_readiness": BlockReadiness,
    "resolve_claim": ClaimOutcome,
    "summarize": SessionSummary,
    "freeze_hybrid_pool": HybridPoolFreezeResult,
}
_WIRE_LIST_TYPES = {
    "active_warnings": FailedBlockWarning,
    "events": WorkflowEvent,
}
_JSON_RESULT_TYPES = {
    "try_enter_slides": bool,
    "precheck_slide_scan": bool,
    "retry_hybrid_slide": bool,
    "get_set": dict,
    "slide_captures": list,
    "slide_recovery_state": str,
    "prepare_finalization": bool,
    "complete_finalization": bool,
    "start_work_order": int,
    "finish_work_order": int,
    "list_results_ready_work_orders": list,
    "list_hybrid_results": list,
    "list_retrieval_results": list,
    "list_hybrid_profile_rows": list,
    "open_work_order_id": (int, type(None)),
}


def _validate_rpc_payload(method: str, body: bytes) -> None:
    """Validate successful response shape while still inside the retry loop."""
    text = body.decode("utf-8")
    if method in _WIRE_RESULT_TYPES:
        store_wire.loads_as(_WIRE_RESULT_TYPES[method], text)
    elif method in _WIRE_LIST_TYPES:
        store_wire.loads_list(_WIRE_LIST_TYPES[method], text)
    else:
        value = json.loads(text)
        expected = _JSON_RESULT_TYPES.get(method)
        if expected is not None and not isinstance(value, expected):
            expected_names = (
                "/".join(t.__name__ for t in expected)
                if isinstance(expected, tuple) else expected.__name__
            )
            raise TypeError(
                f"{method} response must be {expected_names}, "
                f"got {type(value).__name__}"
            )


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
                    _validate_rpc_payload(method, body)
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

    def start_session(
        self, *, started_at: datetime, session_mode: str = "normal"
    ) -> SessionIdentity:
        body = self._rpc(
            self._PLACEHOLDER_SESSION, "start_session",
            [store_wire.encode(started_at), session_mode],
        )
        return store_wire.loads(body.decode("utf-8"))

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

    def reconcile_session_metadata(self, session_number: int) -> None:
        self._rpc(session_number, "reconcile_session_metadata", [])

    # ------------------------------------------------------------------
    # blocks
    # ------------------------------------------------------------------

    def scan_block(
        self, session_number: int, block_id: str, *, request_id: str | None = None
    ) -> ScanOutcome:
        body = self._rpc(
            session_number, "scan_block", [block_id], request_id=request_id
        )
        return store_wire.loads(body.decode("utf-8"))

    def awaiting_capture_blocks(self, session_number: int) -> tuple[str, ...]:
        body = self._rpc(session_number, "awaiting_capture_blocks", [])
        return store_wire.loads_list(None, body.decode("utf-8"))

    def unscan_block(self, session_number: int, block_id: str) -> bool:
        body = self._rpc(session_number, "unscan_block", [block_id])
        return bool(json.loads(body.decode("utf-8")))

    def begin_block_drain(self, session_number: int) -> None:
        self._rpc(session_number, "begin_block_drain", [])

    def try_enter_slides(self, session_number: int) -> bool:
        body = self._rpc(session_number, "try_enter_slides", [])
        return bool(json.loads(body.decode("utf-8")))

    def freeze_hybrid_pool(
        self,
        session_number: int,
        *,
        descriptor_names: Sequence[str],
        candidate_configuration: Mapping[str, object] | None = None,
        request_id: str | None = None,
    ) -> HybridPoolFreezeResult:
        body = self._rpc(
            session_number, "freeze_hybrid_pool",
            [list(descriptor_names), candidate_configuration],
            request_id=request_id,
        )
        return store_wire.loads_as(HybridPoolFreezeResult, body.decode("utf-8"))

    def get_set(self, session_number: int, block_id: str) -> dict[str, object]:
        body = self._rpc(session_number, "get_set", [block_id])
        return json.loads(body.decode("utf-8"))

    def precheck_slide_scan(self, session_number: int, block_id: str) -> bool:
        body = self._rpc(session_number, "precheck_slide_scan", [block_id])
        return bool(json.loads(body.decode("utf-8")))

    def active_warnings(self, session_number: int) -> tuple[FailedBlockWarning, ...]:
        body = self._rpc(session_number, "active_warnings", [])
        return store_wire.loads_list(FailedBlockWarning, body.decode("utf-8"))

    def dismiss_block(self, session_number: int, block_id: str, *, reason: str) -> None:
        self._rpc(session_number, "dismiss_block", [block_id, reason])

    def block_readiness(self, session_number: int, block_id: str) -> BlockReadiness:
        body = self._rpc(session_number, "block_readiness", [block_id])
        return store_wire.loads(body.decode("utf-8"))

    # ------------------------------------------------------------------
    # slides / claims
    # ------------------------------------------------------------------

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
        metadata = json.dumps({
            "captured_at": store_wire.encode(captured_at),
            "result": store_wire.encode(result),
            "duration_ms": duration_ms,
            "request_id": stable_request_id,
            # #256: thread the durable Hybrid scheduling-order key (#255)
            # across the wire. `None` (every ordinary caller today) is
            # dropped by `_dispatch_slide_upload`'s `metadata.get("priority")`
            # read, preserving today's ordinary-FIFO behavior unchanged.
            "priority": priority,
            # #258: thread the `--profile` gate across the wire, mirroring
            # `priority` immediately above. `False` (the default, every
            # pre-#258 caller) preserves today's unprofiled behavior
            # unchanged.
            "profile": profile,
        }).encode("utf-8")
        image = Path(source).read_bytes()
        metadata_payload = json.loads(metadata.decode("utf-8"))
        metadata_payload["source_token"] = hashlib.sha256(image).hexdigest()
        metadata = json.dumps(metadata_payload).encode("utf-8")
        payload = len(metadata).to_bytes(8, "big") + metadata + image
        url = f"{self.base_url}/sessions/{session_number}/slides"
        attempt = 0
        while True:
            attempt += 1
            try:
                post_binary = getattr(self._transport, "post_binary", None)
                if post_binary is None:
                    raise TransportError("transport does not support binary slide upload")
                body = post_binary(url, payload)
                try:
                    json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise TransportError(
                        "HTTP 200 response was not valid UTF-8 JSON"
                    ) from exc
                break
            except TransportError:
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)
        return json.loads(body.decode("utf-8"))

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
        """#256: binary-upload counterpart of `record_slide_capture` for an
        accepted recapture -- same framed-metadata-plus-PNG-bytes shape, but
        posted to the dedicated `/recapture-slide` route (never `/slides`,
        so the two response shapes -- a bare capture-id string here vs. a
        `RecaptureOutcome` envelope -- can never be confused) and decoded as
        a `RecaptureOutcome` instead of a bare capture id.
        """
        stable_request_id = request_id if request_id is not None else uuid.uuid4().hex
        image = Path(source).read_bytes()
        metadata_payload = {
            "superseded_capture_id": superseded_capture_id,
            "captured_at": store_wire.encode(captured_at),
            "result": store_wire.encode(result),
            "duration_ms": duration_ms,
            "request_id": stable_request_id,
            "source_token": source_token or hashlib.sha256(image).hexdigest(),
        }
        metadata = json.dumps(metadata_payload).encode("utf-8")
        payload = len(metadata).to_bytes(8, "big") + metadata + image
        url = f"{self.base_url}/sessions/{session_number}/recapture-slide"
        attempt = 0
        while True:
            attempt += 1
            try:
                post_binary = getattr(self._transport, "post_binary", None)
                if post_binary is None:
                    raise TransportError("transport does not support binary slide upload")
                body = post_binary(url, payload)
                try:
                    store_wire.loads_as(RecaptureOutcome, body.decode("utf-8"))
                except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                    raise TransportError(
                        "HTTP 200 response was not a valid RecaptureOutcome"
                    ) from exc
                break
            except TransportError:
                if attempt >= self.max_attempts:
                    raise
                if self.backoff:
                    sleep(self.backoff)
        return store_wire.loads_as(RecaptureOutcome, body.decode("utf-8"))

    def resolve_claim(
        self,
        session_number: int,
        block_id: str,
        slide_capture_id: str,
        slide_path: str | Path,
        *,
        request_id: str | None = None,
    ) -> ClaimOutcome:
        args = [block_id, slide_capture_id, str(slide_path)]
        body = self._rpc(
            session_number, "resolve_claim", args, request_id=request_id
        )
        return store_wire.loads(body.decode("utf-8"))

    def slide_captures(self, session_number: int) -> tuple[dict[str, object], ...]:
        body = self._rpc(session_number, "slide_captures", [])
        return tuple(json.loads(body.decode("utf-8")))

    def slide_recovery_state(self, session_number: int) -> str:
        body = self._rpc(session_number, "slide_recovery_state", [])
        return json.loads(body.decode("utf-8"))

    def skip_unreadable_slide(self, session_number: int) -> None:
        self._rpc(session_number, "skip_unreadable_slide", [])

    def mark_waiting_for_slide(self, session_number: int) -> None:
        self._rpc(session_number, "mark_waiting_for_slide", [])

    # ------------------------------------------------------------------
    # finalization
    # ------------------------------------------------------------------

    def begin_finalization(self, session_number: int) -> None:
        self._rpc(session_number, "begin_finalization", [])

    def prepare_finalization(self, session_number: int) -> bool:
        body = self._rpc(session_number, "prepare_finalization", [])
        return bool(json.loads(body.decode("utf-8")))

    def complete_finalization(self, session_number: int) -> bool:
        body = self._rpc(session_number, "complete_finalization", [])
        return bool(json.loads(body.decode("utf-8")))

    def record_finalization_error(
        self, session_number: int, message: str, *, reconcile: bool = True
    ) -> None:
        self._rpc(session_number, "record_finalization_error", [message, reconcile])

    def record_profile_capture(
        self, session_number: int, capture_id: str, fields: Mapping[str, object]
    ) -> None:
        self._rpc(
            session_number, "record_profile_capture", [capture_id, dict(fields)]
        )

    def record_slide_benchmark(
        self, session_number: int, capture_id: str, fields: Mapping[str, object]
    ) -> None:
        self._rpc(
            session_number, "record_slide_benchmark", [capture_id, dict(fields)]
        )

    def wait_for_jobs(self) -> None:
        self._rpc(self._PLACEHOLDER_SESSION, "wait_for_jobs", [])

    # ------------------------------------------------------------------
    # work orders (#149/#150/#153)
    # ------------------------------------------------------------------
    #
    # The work-order lifecycle (start/finish bracket) and the results reader
    # the kiosk table consumes. On the Pi ``SessionWorkflow.store`` is this
    # proxy, so these must forward over ``/rpc`` to the real ``ProcessingStore``
    # on the PC (ADR 0002). ``finish_work_order`` dispatches its N^2 scoring job
    # on the *server's* executor -- ``start_job`` rides the wire so the PC, not
    # the Pi, owns that background work.

    def start_work_order(self, session_number: int) -> int:
        body = self._rpc(session_number, "start_work_order", [])
        return int(json.loads(body.decode("utf-8")))

    def finish_work_order(self, session_number: int, *, start_job: bool = True) -> int:
        body = self._rpc(session_number, "finish_work_order", [start_job])
        return int(json.loads(body.decode("utf-8")))

    def list_results_ready_work_orders(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        body = self._rpc(session_number, "list_results_ready_work_orders", [])
        return tuple(json.loads(body.decode("utf-8")))

    def list_hybrid_results(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        body = self._rpc(session_number, "list_hybrid_results", [])
        return tuple(json.loads(body.decode("utf-8")))

    def list_retrieval_results(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        body = self._rpc(session_number, "list_retrieval_results", [])
        return tuple(json.loads(body.decode("utf-8")))

    def list_hybrid_profile_rows(
        self, session_number: int
    ) -> tuple[dict[str, object], ...]:
        body = self._rpc(session_number, "list_hybrid_profile_rows", [])
        return tuple(json.loads(body.decode("utf-8")))

    def results_evidence_bytes(
        self, session_number: int, artifact_path: str | Path
    ) -> bytes | None:
        """Fetch a session-scoped evidence JPEG from the processing computer."""
        from urllib.parse import urlencode

        query = urlencode({"path": str(artifact_path)})
        return self._get_binary_optional(
            f"/sessions/{session_number}/evidence?{query}"
        )

    def open_work_order_id(self, session_number: int) -> int | None:
        body = self._rpc(session_number, "open_work_order_id", [])
        value = json.loads(body.decode("utf-8"))
        return int(value) if value is not None else None

    def has_work_orders(self, session_number: int) -> bool:
        body = self._rpc(session_number, "has_work_orders", [])
        return bool(json.loads(body.decode("utf-8")))

    # ------------------------------------------------------------------
    # summary / events / snapshot
    # ------------------------------------------------------------------

    def summarize(self, session: SessionIdentity) -> SessionSummary:
        body = self._rpc(session.number, "summarize", [])
        return store_wire.loads(body.decode("utf-8"))

    def events(self, session_number: int) -> tuple[WorkflowEvent, ...]:
        body = self._rpc(session_number, "events", [])
        return store_wire.loads_list(WorkflowEvent, body.decode("utf-8"))

    def record_event(
        self,
        session_number: int,
        kind: str,
        message: str,
        *,
        request_id: str | None = None,
    ) -> None:
        self._rpc(
            session_number, "record_event", [kind, message], request_id=request_id
        )

    def snapshot(self, session: SessionIdentity) -> WorkflowSnapshot:
        body = self._get(f"/sessions/{session.number}/status")
        return store_wire.loads_as(WorkflowSnapshot, body.decode("utf-8"))

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
