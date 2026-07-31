"""TDD coverage for the `POST /sessions/{n}/rpc` surface (issue #114 slice).

Exercises `record_finalization_error` now being public, the getattr-free
`_RPC_METHODS` whitelist inside `session_workflow._dispatch_rpc`, and the
400-vs-500 error mapping. Does NOT cover request_id/idempotency/ledger
behavior -- that is a later slice.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import cv2
import numpy as np
import pytest

import store.wire as store_wire
from session.workflow import (
    HybridPoolFreezeResult,
    LoopbackCaptureReceiver,
    ProcessingStore,
    ScanOutcome,
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


def test_record_finalization_error_is_public_not_private():
    assert hasattr(ProcessingStore, "record_finalization_error")
    assert not hasattr(ProcessingStore, "_record_finalization_error")


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


def test_scan_block_via_rpc_round_trips_scan_outcome(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "scan_block", args=["51151378"]
        )

    assert status == 200
    outcome = store_wire.loads_as(ScanOutcome, body.decode("utf-8"))
    assert outcome == ScanOutcome(True, "Accepted block 51151378")


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


def test_sqlite_operational_error_from_a_dispatched_method_returns_500(
    tmp_path, monkeypatch
):
    """#269 review BLOCKER finding 2b: neither `_dispatch_rpc`'s 400 handler
    (`ValueError`/`TypeError`/`KeyError`/`json.JSONDecodeError`) nor its old
    500 handler (`OSError`/`RuntimeError`) caught `sqlite3.OperationalError`
    -- a legacy-schema SQL defect (or any future one) raised inside a
    whitelisted store method used to escape the request-handler thread
    entirely: stdlib prints a bare traceback and drops the connection with
    NO HTTP response at all, which no client-side retry/timeout logic can
    recover from. Force a whitelisted method to raise it and assert the RPC
    surface now answers 500 instead."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    def _boom(self, session_number, message, *, reconcile=True):
        raise sqlite3.OperationalError("no such column: work_order_id")

    monkeypatch.setattr(ProcessingStore, "record_finalization_error", _boom)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "record_finalization_error",
            args=["trigger"],
        )

    assert status == 500
    assert "no such column" in json.loads(body)["error"]


def _fake_fingerprint_builder(mask):
    from verify.invariant_descriptors import DescriptorValue
    return {"fake_v1": DescriptorValue(vector=np.array([1.0]), construction_ns=1)}


def _fake_score_cache_builder(specimen):
    from verify.scorer import LockedScoreCache, _ComponentFeatures
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def _preprocessor(_capture_path):
    return np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


def _upload_block(store, session_number, block_id, value):
    assert store.scan_block(session_number, block_id).accepted
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    body = png.tobytes()
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


def test_freeze_hybrid_pool_via_rpc_round_trips_and_freezes(tmp_path, monkeypatch):
    """#250: freeze_hybrid_pool must be reachable over /rpc (the Pi never
    holds the block masks -- only the processing computer's ProcessingStore
    can build fingerprints/caches)."""
    def write_qc(capture, mask, destination):
        assert cv2.imwrite(str(destination), np.zeros((4, 4, 3), dtype=np.uint8))

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    store = ProcessingStore(
        tmp_path / "processing", preprocessor=_preprocessor,
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    _upload_block(store, session.number, "11111111", 10)
    _upload_block(store, session.number, "22222222", 20)
    store.wait_for_jobs()
    store.begin_block_drain(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "freeze_hybrid_pool",
            args=[["fake_v1"]],
        )

    assert status == 200
    result = store_wire.loads_as(HybridPoolFreezeResult, body.decode("utf-8"))
    assert result.frozen is True
    assert set(result.usable_block_ids) == {"11111111", "22222222"}
    assert store.hybrid_pool(work_order_id) is not None


def test_start_session_via_rpc_with_session_mode_persists_the_durable_mode(tmp_path):
    """#269: start_session's RPC arity grew from (1,1) to (1,2) to carry the
    optional session_mode string -- this proves the whole registry (the
    _RPC_METHODS lambda, _RPC_ARITY, and the persisted sessions.session_mode
    column) actually cooperates over the wire, not just that each side's
    tests agree with themselves."""
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, 0, "start_session",
            args=[store_wire.encode(STARTED_AT), "hybrid"],
        )

    assert status == 200
    identity = store_wire.loads(body.decode("utf-8"))
    assert store._session_mode(identity.number) == "hybrid"


def test_start_session_via_rpc_without_session_mode_arg_defaults_to_normal(tmp_path):
    """Older-shaped one-arg calls (arity minimum still 1) must keep working
    unchanged -- the new second argument is additive, not required."""
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, 0, "start_session", args=[store_wire.encode(STARTED_AT)]
        )

    assert status == 200
    identity = store_wire.loads(body.decode("utf-8"))
    assert store._session_mode(identity.number) == "normal"


def test_events_via_rpc_round_trips_through_codec(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "events")

    assert status == 200
    events = store_wire.loads_list(WorkflowEvent, body.decode("utf-8"))
    assert isinstance(events, tuple)
    assert any(event.kind == "block_scanned" for event in events)
    assert all(isinstance(event, WorkflowEvent) for event in events)


def test_block_readiness_via_rpc_round_trips_single_dataclass(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "block_readiness", args=["51151378"]
        )

    assert status == 200
    envelope = json.loads(body)
    assert envelope["type"] == "BlockReadiness"
    readiness = store_wire.loads(body.decode("utf-8"))
    assert readiness.evaluable is False
    assert readiness.review_reason


def test_unknown_method_is_a_400_with_named_error(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "no_such_method")

    assert status == 400
    assert json.loads(body) == {"error": "unknown method: no_such_method"}


def test_dunder_method_name_is_rejected_not_executed(tmp_path):
    """Proves the dispatch is a dict lookup, never getattr(store, method)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "__init__")

    assert status == 400
    assert json.loads(body) == {"error": "unknown method: __init__"}


def test_removed_private_method_name_is_rejected(tmp_path):
    """The old private name must not still be reachable under any guise."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "_record_finalization_error",
            args=["should not run"],
        )

    assert status == 400
    assert json.loads(body) == {"error": "unknown method: _record_finalization_error"}
    summary = store.summarize(session)
    assert summary.finalization_error is None


def test_domain_key_error_maps_to_400(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "get_set", args=["99999999"]
        )

    assert status == 400
    error = json.loads(body)["error"]
    assert "99999999" in error


def test_domain_value_error_maps_to_400(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        # dismiss_block requires a non-empty reason; also the block hasn't
        # failed yet, so this raises ValueError("only a failed block can be
        # dismissed") -- a domain rejection, not a malformed request.
        status, body = _rpc(
            receiver.url, session.number, "dismiss_block",
            args=["51151378", "operator dismissed"],
        )

    assert status == 400
    assert "dismissed" in json.loads(body)["error"]


def test_domain_runtime_error_maps_to_400_but_unexpected_runtime_stays_500(
    tmp_path, monkeypatch
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "skip_unreadable_slide")
        assert status == 400
        assert "unreadable slide" in json.loads(body)["error"]

        def crash(_session_number, *, request_id=None):
            raise RuntimeError("unexpected invariant failure")

        monkeypatch.setattr(store, "skip_unreadable_slide", crash)
        status, body = _rpc(receiver.url, session.number, "skip_unreadable_slide")

    assert status == 500
    assert json.loads(body) == {"error": "unexpected invariant failure"}


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


def test_work_order_lifecycle_reachable_via_rpc(tmp_path):
    """#149: start/finish_work_order must be dispatchable over /rpc so the Pi
    (RemoteProcessingStore) can drive the work-order bracket, not only an
    in-process store. `start_job=False` keeps the finish synchronous (no
    executor scoring job) so the test stays deterministic."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "start_work_order")
        assert status == 200
        work_order_id = json.loads(body)
        assert isinstance(work_order_id, int)

        status, body = _rpc(
            receiver.url, session.number, "finish_work_order", args=[False]
        )

    assert status == 200
    assert json.loads(body) == work_order_id
    assert store.get_work_order(session.number, work_order_id)["lifecycle_state"] == (
        "scoring"
    )


def test_finish_work_order_defaults_to_dispatching_the_scoring_job(tmp_path):
    """With no start_job arg the server keeps its default (start_job=True), so
    the order advances toward scoring exactly as the local store would."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        _rpc(receiver.url, session.number, "start_work_order")
        status, body = _rpc(receiver.url, session.number, "finish_work_order")

    assert status == 200
    assert isinstance(json.loads(body), int)
    store.wait_for_jobs()


def test_list_results_ready_work_orders_reachable_via_rpc_returns_list(tmp_path):
    """#150/#153: the kiosk results reader must round-trip a JSON list over
    /rpc. With nothing results-ready yet it is an empty list, not a 400."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(
            receiver.url, session.number, "list_results_ready_work_orders"
        )

    assert status == 200
    assert json.loads(body) == []


def test_has_work_orders_reachable_via_rpc(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        status, body = _rpc(receiver.url, session.number, "has_work_orders")

    assert status == 200
    assert json.loads(body) is False


def test_captures_route_is_unaffected_by_rpc_addition(tmp_path):
    """Guard against accidentally rerouting /captures through the rpc branch."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        request = Request(
            f"{receiver.url}/sessions/{session.number}/captures",
            data=b"not a real png",
            method="POST",
            headers={
                "X-Capture-Id": "capture-x",
                "X-Block-Id": "51151378",
                "X-Checksum-Sha256": "0" * 64,
            },
        )
        try:
            urlopen(request, timeout=2)
            assert False, "bad checksum should be rejected"
        except HTTPError as exc:
            assert exc.code == 400
            assert "checksum" in json.loads(exc.read())["error"]
