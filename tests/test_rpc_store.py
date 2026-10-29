"""TDD coverage for the `POST /sessions/{n}/rpc` surface.

Exercises public `record_finalization_error`, the getattr-free `_RPC_METHODS`
whitelist inside `session_workflow._dispatch_rpc`, and the 400-vs-500 error
mapping. Does NOT cover request_id/idempotency/ledger behavior.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import pytest

import transport.wire as store_wire
from session.workflow import (
    LoopbackCaptureReceiver,
    ProcessingStore,
    WorkflowEvent,
)

STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


def _rpc(url: str, session_number: int, method: str, args=None) -> tuple[int, bytes]:
    """POST one JSON-RPC envelope; return (status_code, raw_body) either way."""
    request = Request(
        f"{url}/sessions/{session_number}/rpc",
        data=json.dumps({"method": method, "args": args or []}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urlopen(request, timeout=2)
        return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def test_record_finalization_error_reachable_via_rpc_and_durably_recorded(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "record_finalization_error",
            args=["synthetic finalization failure"],
        )

    assert status == 200
    assert json.loads(body) is None
    summary = store.summarize(session)
    assert summary.finalization_error == "synthetic finalization failure"


def test_evidence_endpoint_serves_only_its_session_claim_artifact_jpegs(tmp_path):
    """The Pi can read laptop-owned JPEGs, but cannot escape its session."""
    store = ProcessingStore(tmp_path / "processing")
    first = store.start_session(started_at=STARTED_AT)
    second = store.start_session(started_at=STARTED_AT + timedelta(seconds=1))
    artifacts = first.directory / "claim_artifacts"
    artifacts.mkdir()
    jpeg = artifacts / "cap-1_slide_thumb.jpg"
    jpeg.write_bytes(b"\xff\xd8evidence")

    with LoopbackCaptureReceiver(store) as receiver:
        response = urlopen(  # noqa: S310 -- loopback receiver under test
            receiver.url + f"/sessions/{first.number}/evidence?" + urlencode({"path": str(jpeg)}),
            timeout=2,
        )
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read() == b"\xff\xd8evidence"

        forbidden = (
            receiver.url + f"/sessions/{second.number}/evidence?"
            + urlencode({"path": str(jpeg)})
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(forbidden, timeout=2)  # noqa: S310 -- loopback receiver under test

    assert exc_info.value.code == 404


@pytest.mark.parametrize(
    "error",
    [
        # sqlite3.Error must not escape the handler and drop the connection.
        sqlite3.OperationalError("no such column: work_order_id"),
        RuntimeError("unexpected invariant failure"),
    ],
    ids=["sqlite", "runtime"],
)
def test_unexpected_store_failure_returns_500(tmp_path, monkeypatch, error):
    """500 (retryable) rather than 400 or a dropped connection, so the Pi
    proxy retries instead of giving up."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    def _boom(self, session_number, message, *, reconcile=True):
        raise error

    monkeypatch.setattr(ProcessingStore, "record_finalization_error", _boom)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "record_finalization_error",
            args=["trigger"],
        )

    assert status == 500
    assert json.loads(body) == {"error": str(error)}


@pytest.mark.parametrize(
    "method",
    [
        "no_such_method",
        # The dispatch is a dict lookup, never getattr(store, method).
        "__init__",
        # The old private name must not still be reachable under any guise.
        "_record_finalization_error",
    ],
)
def test_unlisted_method_is_a_400_with_named_error_and_never_runs(tmp_path, method):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, method, args=["should not run"],
        )

    assert status == 400
    assert json.loads(body) == {"error": f"unknown method: {method}"}
    assert store.summarize(session).finalization_error is None


@pytest.mark.parametrize(
    ("method", "args", "expected"),
    [
        # KeyError: unknown block.
        ("get_set", ["99999999"], "99999999"),
        # ValueError: only a failed block can be dismissed.
        ("dismiss_block", ["51151378", "operator dismissed"], "dismissed"),
        # ValueError: skip is only valid for an unreadable slide.
        ("skip_unreadable_slide", [], "unreadable slide"),
    ],
    ids=["key", "value", "skip"],
)
def test_domain_rejection_maps_to_400(tmp_path, method, args, expected):
    """400 = the request itself is wrong; an identical retry never helps."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, method, args=args)

    assert status == 400
    assert expected in json.loads(body)["error"]


def test_args_must_be_a_list(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        request = Request(
            f"{receiver.url}/sessions/{session.number}/rpc",
            data=json.dumps({"method": "events", "args": {"not": "a list"}}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urlopen(request, timeout=2)
            assert False, "non-list args should have been rejected"
        except HTTPError as exc:
            assert exc.code == 400


def test_missing_rpc_argument_returns_json_400_and_server_stays_alive(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "scan_block", args=[])
        next_status, next_body = _rpc(receiver.url, session.number, "events")

    assert status == 400
    assert json.loads(body) == {
        "error": "scan_block expects 1 argument; received 0"
    }
    assert next_status == 200
    assert isinstance(store_wire.loads_list(WorkflowEvent, next_body.decode()), tuple)


def test_omitted_optional_rpc_args_take_the_store_defaults(tmp_path):
    """Older-shaped calls keep working: start_session's session_mode
    and finish_work_order's start_job are additive optional args."""
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, 0, "start_session", args=[store_wire.encode(STARTED_AT)]
        )
        assert status == 200
        identity = store_wire.loads(body.decode("utf-8"))
        assert store._session_mode(identity.number) == "normal"

        _rpc(receiver.url, identity.number, "start_work_order")
        # No start_job arg: the server keeps start_job=True and dispatches scoring.
        status, body = _rpc(receiver.url, identity.number, "finish_work_order")

    assert status == 200
    assert isinstance(json.loads(body), int)
    store.wait_for_jobs()
