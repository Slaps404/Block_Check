"""TDD coverage for the remote store proxy (issue #115).

Exercises `RemoteProcessingStore` against a REAL `LoopbackCaptureReceiver` +
`ProcessingStore` (the same server pattern as `tests/test_rpc_store.py` and
`tests/test_idempotency_ledger.py`), so these tests prove the proxy and the
#114 server/ledger actually cooperate over the wire -- not just that each
side's mocks agree with each other.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from store.remote import RemoteProcessingStore, StoreError, TransportError, UrlTransport
from slide.qr import select_slide_identity
from session.workflow import (
    LoopbackCaptureReceiver, PiOutbox, ProcessingStore, ScanOutcome, WorkflowEvent,
)

STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# fakes: transports that observe/interfere with the wire, never the RPC logic
# --------------------------------------------------------------------------


class CountingTransport:
    """Wraps a real transport; records call count and every outgoing payload."""

    def __init__(self, inner):
        self.inner = inner
        self.post_calls = 0
        self.sent_payloads: list[bytes] = []

    def post(self, url, payload):
        self.post_calls += 1
        self.sent_payloads.append(payload)
        return self.inner.post(url, payload)

    def get(self, url):
        return self.inner.get(url)


class DropFirstResponseTransport:
    """Lets the real request through every time, but discards the FIRST
    response after the real server has already durably applied it -- this
    is the "response never arrived" half of a mid-command disconnect. Every
    later attempt behaves normally."""

    def __init__(self, inner):
        self.inner = inner
        self.post_calls = 0
        self.sent_payloads: list[bytes] = []

    def post(self, url, payload):
        self.post_calls += 1
        self.sent_payloads.append(payload)
        result = self.inner.post(url, payload)
        if self.post_calls == 1:
            raise TransportError("simulated: response dropped after the server applied it")
        return result

    def post_binary(self, url, payload):
        self.post_calls += 1
        self.sent_payloads.append(payload)
        result = self.inner.post_binary(url, payload)
        if self.post_calls == 1:
            raise TransportError("simulated: binary response dropped after apply")
        return result

    def get(self, url):
        return self.inner.get(url)


class AlwaysFailTransport:
    """Every call fails at the transport level; never reaches any server."""

    def __init__(self):
        self.post_calls = 0
        self.sent_payloads: list[bytes] = []

    def post(self, url, payload):
        self.post_calls += 1
        self.sent_payloads.append(payload)
        raise TransportError("simulated: connection refused")

    def get(self, url):
        raise TransportError("simulated: connection refused")


class RejectingTransport:
    """Simulates a domain rejection (HTTP 400) without touching any server."""

    def __init__(self):
        self.post_calls = 0

    def post(self, url, payload):
        self.post_calls += 1
        raise StoreError("simulated: domain rejection")

    def get(self, url):
        raise StoreError("simulated: domain rejection")


class ResponseSequenceTransport:
    """Returns successive HTTP-200 bodies without interpreting their content."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.post_calls = 0

    def post(self, url, payload):
        body = self.bodies[min(self.post_calls, len(self.bodies) - 1)]
        self.post_calls += 1
        return body

    def get(self, url):
        raise AssertionError("not used")


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_scan_block_happy_path_returns_rehydrated_scan_outcome(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        outcome = proxy.scan_block(session.number, "51151378")

    assert outcome == ScanOutcome(True, "Accepted block 51151378")
    assert isinstance(outcome, ScanOutcome)


def test_results_evidence_bytes_round_trips_binary_artifact_from_receiver(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    artifacts = session.directory / "claim_artifacts"
    artifacts.mkdir()
    jpeg = artifacts / "cap-1_slide_thumb.jpg"
    jpeg.write_bytes(b"\xff\xd8evidence")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url, backoff=0)
        assert proxy.results_evidence_bytes(session.number, jpeg) == b"\xff\xd8evidence"
        assert proxy.results_evidence_bytes(session.number, artifacts / "missing.jpg") is None


def test_block_capture_recovery_methods_match_local_store_over_rpc(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "22222222")
    store.scan_block(session.number, "11111111")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        assert proxy.awaiting_capture_blocks(session.number) == (
            "22222222", "11111111"
        )
        assert proxy.unscan_block(session.number, "22222222")
        assert proxy.awaiting_capture_blocks(session.number) == ("11111111",)


def test_events_read_only_call_round_trips_to_the_real_type(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        events = proxy.events(session.number)

    assert isinstance(events, tuple)
    assert all(isinstance(event, WorkflowEvent) for event in events)
    assert any(event.kind == "block_scanned" for event in events)


def test_slide_bytes_cross_wire_and_server_uses_only_processing_local_path(
    tmp_path, monkeypatch
):
    processing_root = tmp_path / "processing"
    pi_root = tmp_path / "pi"
    pi_root.mkdir()
    slide_path = pi_root / "slide.png"
    assert cv2.imwrite(
        str(slide_path), np.full((32, 48, 3), 120, dtype=np.uint8)
    )
    store = ProcessingStore(processing_root)
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)
    original = store.record_slide_capture
    server_sources: list[Path] = []

    def observe_server_source(session_number, source, **kwargs):
        source = Path(source).resolve()
        server_sources.append(source)
        assert source.is_relative_to(processing_root.resolve())
        assert source != slide_path.resolve()
        return original(session_number, source, **kwargs)

    monkeypatch.setattr(store, "record_slide_capture", observe_server_source)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        capture_id = proxy.record_slide_capture(
            session.number,
            slide_path,
            captured_at=STARTED_AT,
            result=select_slide_identity(()),
            duration_ms=5.0,
        )

    rows = store.slide_captures(session.number)
    assert len(server_sources) == 1
    assert len(rows) == 1
    assert rows[0]["capture_id"] == capture_id
    durable_path = Path(rows[0]["capture_path"])
    assert durable_path.is_relative_to(processing_root.resolve())
    assert durable_path.read_bytes() == slide_path.read_bytes()


def test_slide_upload_reuses_request_id_after_dropped_response(tmp_path):
    slide_path = tmp_path / "pi" / "slide.png"
    slide_path.parent.mkdir()
    assert cv2.imwrite(str(slide_path), np.full((32, 48, 3), 120, dtype=np.uint8))
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        transport = DropFirstResponseTransport(UrlTransport())
        proxy = RemoteProcessingStore(
            receiver.url, transport=transport, max_attempts=2, backoff=0
        )
        capture_id = proxy.record_slide_capture(
            session.number, slide_path, captured_at=STARTED_AT,
            result=select_slide_identity(()), duration_ms=5.0,
        )

    rows = store.slide_captures(session.number)
    assert transport.post_calls == 2
    assert transport.sent_payloads[0] == transport.sent_payloads[1]
    assert [row["capture_id"] for row in rows] == [capture_id]


def test_slide_outbox_survives_process_restart_and_replays_after_reconnect(tmp_path):
    slide_path = tmp_path / "slide.png"
    assert cv2.imwrite(str(slide_path), np.full((32, 48, 3), 120, dtype=np.uint8))
    outbox_root = tmp_path / "outbox"
    outbox = PiOutbox(outbox_root)
    published = outbox.publish_slide(
        slide_path, STARTED_AT, result=select_slide_identity(()), duration_ms=5.0
    )
    dead = RemoteProcessingStore(
        "http://127.0.0.1:1", max_attempts=1, backoff=0
    )
    assert outbox.replay_slides(1, dead) == ()

    restarted = PiOutbox(outbox_root)
    assert [item.capture_id for item in restarted.pending_slides()] == [
        published.capture_id
    ]
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        receipts = restarted.replay_slides(
            session.number, RemoteProcessingStore(receiver.url)
        )

    assert receipts == (published.capture_id,)
    assert restarted.pending_slides() == ()
    assert len(store.slide_captures(session.number)) == 1


def test_corrupt_slide_outbox_entry_remains_visible_and_blocking(tmp_path):
    slide_path = tmp_path / "slide.png"
    assert cv2.imwrite(str(slide_path), np.full((32, 48, 3), 120, dtype=np.uint8))
    outbox = PiOutbox(tmp_path / "outbox")
    published = outbox.publish_slide(
        slide_path, STARTED_AT, result=select_slide_identity(()), duration_ms=5.0
    )
    published.path.write_bytes(b"corrupt")

    assert outbox.pending_slides() == ()
    assert outbox.invalid_slide_entries() == (published.capture_id,)


def test_block_readiness_read_only_call_round_trips_to_the_real_type(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        readiness = proxy.block_readiness(session.number, "51151378")

    assert readiness.evaluable is False
    assert readiness.review_reason


def test_snapshot_round_trips_via_status_route(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        snapshot = proxy.snapshot(session)

    assert snapshot.session_number == session.number
    assert snapshot.phase == "blocks"


# --------------------------------------------------------------------------
# work-order lifecycle (#149/#150/#153): reachable through the proxy
# --------------------------------------------------------------------------


def test_work_order_start_and_finish_round_trip_over_rpc(tmp_path):
    """The crash reproduced on the Pi: SessionWorkflow.start_work_order ->
    store.start_work_order over the wire. Proves both start and finish are
    proxied (fixing only start would just move the crash to finish)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        work_order_id = proxy.start_work_order(session.number)
        assert isinstance(work_order_id, int)
        finished = proxy.finish_work_order(session.number, start_job=False)

    assert finished == work_order_id
    assert store.get_work_order(session.number, work_order_id)["lifecycle_state"] == (
        "scoring"
    )


def test_list_results_ready_work_orders_round_trips_empty_like_local_store(tmp_path):
    """Reachability + shape parity when nothing is results-ready yet: the proxy
    returns the same empty tuple the local store does, over /rpc."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        remote_rows = proxy.list_results_ready_work_orders(session.number)

    assert remote_rows == store.list_results_ready_work_orders(session.number) == ()


def test_open_work_order_id_round_trips_over_rpc_including_none(tmp_path):
    """#155: the boot-seed read helper must be proxied over /rpc -- a real Pi
    (behind RemoteProcessingStore, never a local ProcessingStore) calling it
    would otherwise AttributeError despite green unit tests on the store
    itself. Exercises both the None case (nothing open yet) and the open-id
    case, matching the local store exactly."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        assert proxy.open_work_order_id(session.number) is None

        work_order_id = proxy.start_work_order(session.number)
        assert proxy.open_work_order_id(session.number) == work_order_id

    assert store.open_work_order_id(session.number) == work_order_id


def test_has_work_orders_round_trips_over_rpc(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        assert proxy.has_work_orders(session.number) is False
        proxy.start_work_order(session.number)
        assert proxy.has_work_orders(session.number) is True


# --------------------------------------------------------------------------
# freeze_hybrid_pool (#250): reachable through the proxy, same RPC hazard
# class as the work-order methods above -- the Pi never holds block masks,
# so only the real ProcessingStore over /rpc can build fingerprints/caches.
# --------------------------------------------------------------------------


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


def test_freeze_hybrid_pool_round_trips_over_rpc(tmp_path, monkeypatch):
    def write_qc(capture, mask, destination):
        assert cv2.imwrite(str(destination), np.zeros((4, 4, 3), dtype=np.uint8))

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=lambda path: (
            np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}
        ),
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
        proxy = RemoteProcessingStore(receiver.url)
        result = proxy.freeze_hybrid_pool(session.number, descriptor_names=["fake_v1"])

    assert result.frozen is True
    assert set(result.usable_block_ids) == {"11111111", "22222222"}
    assert store.hybrid_pool(work_order_id) is not None


def test_start_session_with_session_mode_round_trips_over_rpc(tmp_path):
    """#269: RemoteProcessingStore.start_session gained a session_mode
    parameter threaded into the /rpc args list -- proves the proxy, the
    server lambda, and the durable sessions.session_mode column agree."""
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        identity = proxy.start_session(started_at=STARTED_AT, session_mode="hybrid_shadow")

    assert store._session_mode(identity.number) == "hybrid_shadow"


def test_start_session_default_session_mode_is_normal_over_rpc(tmp_path):
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        identity = proxy.start_session(started_at=STARTED_AT)

    assert store._session_mode(identity.number) == "normal"


def test_start_session_identity_carries_the_durable_session_mode(tmp_path):
    """Confirmed HIGH-severity fix: `SessionIdentity.session_mode` rides the
    SAME start_session/resume_session round trip proxied above -- not a new
    store method -- so `tools/run_pi_session.py::main` can compare the Pi's
    own resolved mode against the durable value without any new RPC."""
    store = ProcessingStore(tmp_path / "processing")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        identity = proxy.start_session(started_at=STARTED_AT, session_mode="hybrid")

    assert identity.session_mode == "hybrid"


def test_resume_session_identity_carries_the_durable_session_mode_over_rpc(tmp_path):
    """The startup-mismatch guard reads `resume_session`'s returned identity
    (the Pi never calls start_session itself -- see run_pi_session.py's
    docstring), so this exact round trip is what the guard depends on."""
    store = ProcessingStore(tmp_path / "processing")
    started = store.start_session(started_at=STARTED_AT, session_mode="hybrid_shadow")

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        resumed = proxy.resume_session(started.number)

    assert resumed.session_mode == "hybrid_shadow"


def test_resume_session_identity_defaults_to_normal_session_mode_over_rpc(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    started = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        resumed = proxy.resume_session(started.number)

    assert resumed.session_mode == "normal"


# --------------------------------------------------------------------------
# StoreError: HTTP 400, no retry
# --------------------------------------------------------------------------


def test_store_error_on_domain_rejection_is_not_retried(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    store.start_session(started_at=STARTED_AT)  # session 1 exists; 99999 does not

    with LoopbackCaptureReceiver(store) as receiver:
        transport = CountingTransport(UrlTransport())
        proxy = RemoteProcessingStore(receiver.url, transport=transport)

        with pytest.raises(StoreError):
            proxy.scan_block(99999, "51151378")

    # A domain error is deterministic -- retrying the identical request
    # would just fail the same way again, so the proxy must not retry it.
    assert transport.post_calls == 1


def test_store_error_from_fake_transport_is_propagated_without_retry():
    proxy = RemoteProcessingStore(
        "http://example.invalid", transport=RejectingTransport()
    )

    with pytest.raises(StoreError):
        proxy.scan_block(1, "51151378")

    assert proxy._transport.post_calls == 1


# --------------------------------------------------------------------------
# THE load-bearing test: induced mid-command disconnect
# --------------------------------------------------------------------------


def test_mid_command_disconnect_retries_with_same_request_id_and_replays_ledger(
    tmp_path,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        transport = DropFirstResponseTransport(UrlTransport())
        proxy = RemoteProcessingStore(receiver.url, transport=transport, backoff=0)

        outcome = proxy.scan_block(session.number, "51151378")

    # (a) the final result equals the ORIGINAL outcome, not a second,
    # different outcome (e.g. "already scanned").
    assert outcome == ScanOutcome(True, "Accepted block 51151378")

    # The transport really was interrupted once and retried once.
    assert transport.post_calls == 2

    # Stable-id proof: the retry carried the EXACT SAME request_id as the
    # first, dropped attempt -- a fresh id per attempt would have defeated
    # the server's ledger and caused a genuine duplicate.
    first_id = json.loads(transport.sent_payloads[0])["request_id"]
    second_id = json.loads(transport.sent_payloads[1])["request_id"]
    assert first_id == second_id

    # (b) no loss and no duplicate: exactly one durable effect on the real
    # server store, despite the client-visible retry.
    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("block_scanned") == 1
    # `sets` has PRIMARY KEY (session_number, block_id); a genuine second
    # insert (duplicate execution rather than a ledger replay) would have
    # raised sqlite3.IntegrityError inside scan_block itself, so reaching
    # this point with the row intact already proves there was only one.
    row = store.get_set(session.number, "51151378")
    assert row["block_id"] == "51151378"


def test_mid_command_disconnect_retries_finish_work_order_exactly_once(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)

    with LoopbackCaptureReceiver(store) as receiver:
        transport = DropFirstResponseTransport(UrlTransport())
        proxy = RemoteProcessingStore(receiver.url, transport=transport, backoff=0)

        work_order_id = proxy.finish_work_order(session.number, start_job=False)

    assert transport.post_calls == 2
    first_id = json.loads(transport.sent_payloads[0])["request_id"]
    second_id = json.loads(transport.sent_payloads[1])["request_id"]
    assert first_id == second_id

    # Exactly one lifecycle transition took place, despite the retry.
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "scoring"
    assert store.open_work_order_id(session.number) is None


def test_mid_command_disconnect_retries_dismiss_block_exactly_once(tmp_path):
    class FailingPreprocessor:
        def __call__(self, capture_path):
            raise ValueError("cassette window is not evaluable")

    store = ProcessingStore(
        tmp_path / "processing", preprocessor=FailingPreprocessor()
    )
    session = store.start_session(started_at=STARTED_AT)
    store.scan_block(session.number, "51151378")
    capture_source = tmp_path / "51151378_block.png"
    assert cv2.imwrite(
        str(capture_source), np.full((3040, 4056, 3), 80, dtype=np.uint8)
    )
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        capture_source, "51151378", STARTED_AT,
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    assert store.get_set(session.number, "51151378")["preprocessing_status"] == "failed"

    with LoopbackCaptureReceiver(store) as receiver:
        transport = DropFirstResponseTransport(UrlTransport())
        proxy = RemoteProcessingStore(receiver.url, transport=transport, backoff=0)

        proxy.dismiss_block(session.number, "51151378", reason="operator confirmed")

    assert transport.post_calls == 2
    first_id = json.loads(transport.sent_payloads[0])["request_id"]
    second_id = json.loads(transport.sent_payloads[1])["request_id"]
    assert first_id == second_id

    kinds = [event.kind for event in store.events(session.number)]
    assert kinds.count("block_dismissed") == 1
    row = store.get_set(session.number, "51151378")
    assert row["preprocessing_status"] == "unusable"


def test_stable_request_id_across_bounded_transport_failures():
    """Isolated (no real server) proof that retries never mint a new id."""
    transport = AlwaysFailTransport()
    proxy = RemoteProcessingStore(
        "http://example.invalid", transport=transport, max_attempts=3, backoff=0
    )

    with pytest.raises(TransportError):
        proxy.scan_block(1, "51151378")

    ids = {json.loads(payload)["request_id"] for payload in transport.sent_payloads}
    assert len(transport.sent_payloads) == 3
    assert len(ids) == 1


# --------------------------------------------------------------------------
# TransportError exhaustion
# --------------------------------------------------------------------------


def test_transport_error_propagates_after_bounded_retries_are_exhausted():
    transport = AlwaysFailTransport()
    proxy = RemoteProcessingStore(
        "http://example.invalid", transport=transport, max_attempts=4, backoff=0
    )

    with pytest.raises(TransportError):
        proxy.scan_block(1, "51151378")

    assert transport.post_calls == 4


@pytest.mark.parametrize("malformed", [b"not-json", b"\xff", b"[]"])
def test_malformed_http_200_body_is_transport_error_under_retry_policy(malformed):
    transport = ResponseSequenceTransport(malformed, malformed)
    proxy = RemoteProcessingStore(
        "http://example.invalid", transport=transport, max_attempts=2, backoff=0
    )

    with pytest.raises(TransportError):
        proxy.scan_block(1, "51151378")

    assert transport.post_calls == 2
