"""Walking-skeleton transport shell for the kiosk (#121 / 119a).

Proves the Pi-local relay boots over real HTTP on localhost, serves ONE
extracted screen to Chromium with the ``<x-dc>``/``deck-stage.js`` deck runtime
stripped, exposes the read path as JSON, and routes a POSTed tap through the
write path. Uses a real ``ThreadingHTTPServer`` + ``urllib`` (same "drive a real
server" spirit as ``test_remote_store``), not mocks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from kiosk.inspection import project_inspection
from kiosk.relay import KioskRelay
from kiosk.server import KioskServer
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    ScanOutcome,
    SessionWorkflow,
)

# #153: reuse test_session_workflow's real-workflow scoring fixtures/helpers
# rather than re-deriving a second work-order lifecycle harness. Imported at
# module level (not locally inside the test) so pytest can see
# ``lightweight_qc_artifacts`` as a requestable fixture -- autouse fixtures
# don't cross module boundaries, so the test here must ask for it explicitly.
from tests.test_session_workflow import (  # noqa: F401 -- fixture import
    STARTED_AT as WF_STARTED_AT,
    FastPreprocessor,
    StubContactSheetRenderer,
    StubWorkOrderScorer,
    ToggleTransport,
    _capture as wf_capture,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
    _valid_slide_result,
    lightweight_qc_artifacts,
)

STARTED_AT = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


class _FakeHandle:
    def __init__(self):
        self.calls: list[tuple] = []
        self._scanned: list[str] = []

    def scan_block(self, block_id):
        self.calls.append(("scan_block", block_id))
        self._scanned.append(block_id)
        return ScanOutcome(True, f"Accepted block {block_id}")

    def snapshot(self):
        return _Snapshot(self._scanned[-1] if self._scanned else None)

    def summarize(self):
        return _Summary(len(self._scanned))

    def events(self):
        return ()


class _Snapshot:
    def __init__(self, latest):
        self.phase = "blocks"
        self.session_number = 1042
        self.latest_block_id = latest
        self.latest_block_status = "captured"


class _Summary:
    def __init__(self, count):
        self.session_number = 1042
        self.started_at = STARTED_AT
        self.sets_processed = count
        self.pass_count = count
        self.review_count = 0
        self.blocks_captured = count


def _server(handle=None):
    return KioskServer(KioskRelay(handle or _FakeHandle()), host="127.0.0.1", port=0)


def _get(url):
    with urlopen(url, timeout=5) as resp:  # noqa: S310 (localhost only)
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def _post_json(url, payload):
    body = json.dumps(payload).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:  # noqa: S310
        return resp.status, json.loads(resp.read())


def test_relay_boots_and_serves_one_screen_without_the_deck_runtime():
    server = _server()
    server.start()
    try:
        status, content_type, body = _get(server.url + "/")
    finally:
        server.stop()

    html = body.decode()
    assert status == 200
    assert "text/html" in content_type
    assert "Scan Block" in html  # the extracted screen 06
    # deck runtime must be gone -- screen selection is event-driven, not deck-nav.
    assert "<x-dc" not in html
    assert "deck-stage" not in html
    assert "support.js" not in html


def test_served_client_sends_the_view_results_guard_on_state_polls():
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert "view_results_guard: false" in html
    assert 'p.set("view_results_guard"' in html


def test_served_client_renders_projected_result_rows_and_supports_go_back():
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert 'id="results-table"' in html
    assert "function renderResultsTable(rows)" in html
    assert "renderResultsTable(state.results_rows || [])" in html

    # #257 expanded this from a one-line switch case into a multi-line block
    # (GO BACK on view_results_guard now also resumes capture, but only when
    # THIS client is the one that paused it). Pin the surviving behavior
    # instead of exact formatting: every "back" tap must still unconditionally
    # clear its guard flag, and the new resume_capture dispatch must stay
    # gated on the view_results_guard target specifically -- otherwise Open
    # Retrieval's between_orders GO BACK (and #256's recapture_guard GO BACK)
    # would wrongly start dispatching resume_capture too.
    back_case = html[html.index('case "back":'):html.index('case "engage":')]
    assert "setFlag(b.target, false)" in back_case
    assert 'b.target === "view_results_guard"' in back_case
    assert 'command("resume_capture")' in back_case


def test_served_client_maps_all_four_result_states_to_distinct_colors():
    # #248: the browser table mirrors the Python projection's four
    # operator-visible states -- ERROR amber, REVIEW red, PASS green,
    # PENDING gray -- via one explicit lookup table.
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert (
        'const RT_STATE_COLOR = '
        '{ ERROR: "rt-error", REVIEW: "rt-review", PASS: "rt-pass", '
        'PENDING: "rt-pending" };'
    ) in html
    assert "--rt-error:" in html
    assert "--rt-pending:" in html


def test_served_client_orders_results_error_review_pass_pending():
    # #248: ordering changes from REVIEW-first to ERROR, REVIEW, PASS,
    # PENDING so a system failure sorts ahead of a REVIEW match failure.
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert (
        'const RT_STATE_RANK = { ERROR: 0, REVIEW: 1, PASS: 2, PENDING: 3 };'
    ) in html


def test_served_client_handles_unrecognized_state_explicitly_not_as_review():
    # #248: replace the silent red fallback. The old code defaulted a
    # falsy/unrecognized verdict's pill text to "REVIEW"; that must be gone,
    # and lookup must fall through to a distinct "rt-unknown" class instead
    # of "rt-review".
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert (
        'function rtStateClass(verdict) '
        '{ return RT_STATE_COLOR[verdict] || "rt-unknown"; }'
    ) in html
    assert 'icon + (verdict || "REVIEW")' not in html


def test_served_client_repaint_fingerprint_includes_row_state():
    # #248: the fingerprint (`sig`) must include r.verdict, or a row
    # transitioning from PENDING to a final state (PASS/REVIEW/ERROR) will
    # not repaint on the next poll.
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert "JSON.stringify(rows.map(r =>" in html
    assert (
        "[r.work_order, r.block_id, r.verdict, r.capture_id, r.claim_reason,"
    ) in html


def test_served_client_auto_opens_first_row_needing_attention_not_only_review():
    # #248: ERROR now outranks REVIEW, so the "one open at a time" auto-open
    # rule must consider both instead of REVIEW alone.
    server = _server()
    server.start()
    try:
        html = _get(server.url + "/")[2].decode()
    finally:
        server.stop()

    assert (
        'function rtNeedsAttention(verdict) '
        '{ return verdict === "ERROR" || verdict === "REVIEW"; }'
    ) in html
    assert "firstAttentionOpened" in html


def test_state_endpoint_exposes_the_read_path_as_json():
    handle = _FakeHandle()
    handle.scan_block("51151378")
    server = KioskServer(KioskRelay(handle), host="127.0.0.1", port=0)
    server.start()
    try:
        status, content_type, body = _get(server.url + "/state")
    finally:
        server.stop()

    assert status == 200
    assert "application/json" in content_type
    state = json.loads(body)
    assert state["online"] is True
    assert state["captured"] == 1
    assert state["latest_block_id"] == "51151378"


def test_state_endpoint_forwards_the_view_results_guard():
    server = _server()
    server.start()
    try:
        state = json.loads(
            _get(server.url + "/state?view_results_guard=1")[2]
        )
    finally:
        server.stop()

    assert state["view_results_guard"] is True


def test_command_endpoint_routes_a_tap_through_the_write_path():
    handle = _FakeHandle()
    server = KioskServer(KioskRelay(handle), host="127.0.0.1", port=0)
    server.start()
    try:
        status, result = _post_json(
            server.url + "/command", {"name": "scan_block", "args": ["51151378"]}
        )
    finally:
        server.stop()

    assert status == 200
    assert result["ok"] is True
    assert handle.calls == [("scan_block", "51151378")]


def test_end_to_end_tap_over_http_mutates_the_real_workflow(tmp_path):
    """The whole 119a thread through the REAL stack: browser POST -> HTTP
    server -> relay -> dispatch -> SessionWorkflow -> remote store, then the
    read path shows the mutation. No fakes below the HTTP boundary."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        from store.remote import RemoteProcessingStore

        workflow = SessionWorkflow(
            session=session,
            store=RemoteProcessingStore(receiver.url),
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(tmp_path / "framing.json"),
        )
        server = KioskServer(KioskRelay(workflow), host="127.0.0.1", port=0)
        server.start()
        try:
            before = json.loads(_get(server.url + "/state")[2])
            status, result = _post_json(
                server.url + "/command",
                {"name": "scan_block", "args": ["51151378"]},
            )
            after = json.loads(_get(server.url + "/state")[2])
        finally:
            server.stop()

    assert before["captured"] == 0
    assert status == 200 and result["ok"] is True
    # Read path reflects the write: a block_scanned event now tails the log.
    # (snapshot().latest_block_id only fills at CAPTURE time, not scan time --
    # a distinction 119b's screen-07 router must respect.)
    assert after["event_count"] > before["event_count"]
    assert after["last_event"]["block_id"] == "51151378"
    # Durable mutation landed exactly once on the real store.
    assert store.awaiting_capture_blocks(session.number) == ("51151378",)


def test_catalog_endpoint_serves_the_static_screen_catalog():
    """GET /catalog is pure static data: 200 + JSON, the frozen screen map,
    and the no-store cache header every route uses. No session needed."""
    server = _server()
    server.start()
    try:
        status, content_type, body = _get(server.url + "/catalog")
    finally:
        server.stop()

    assert status == 200
    assert "application/json" in content_type
    catalog = json.loads(body)
    screens = catalog["screens"]
    # Exactly the router-returned ids plus the client-overlay id 18.
    assert set(screens.keys()) == {
        "01", "02", "04", "05", "06", "07", "08", "09", "10", "12", "13",
        "14", "15", "16", "17", "18", "19", "20", "21", "processing",
        "hold_still", "capture_review", "results_table", "first_work_order",
        "between_orders", "block_scan_work_order", "slide_capture_work_order",
        "hybrid_attention",
    }
    # A representative entry carries the render fields the frontend reads.
    scan = screens["06"]
    assert scan["headline"] == "Scan Block"
    assert scan["buttons"][0]["action"] == "guard"


def test_command_endpoint_returns_benign_ok_false_when_verb_raises_runtimeerror():
    """A double-tapped CALIBRATE NOW moves off AWAITING_BASELINE_CONFIRMATION
    within the poll gap, so the second ``confirm_empty`` raises RuntimeError.
    The seam must surface a benign {ok: False}, never an HTTP 500."""

    class _RaisingHandle(_FakeHandle):
        def confirm_empty(self):
            raise RuntimeError("baseline already confirmed")

    server = KioskServer(KioskRelay(_RaisingHandle()), host="127.0.0.1", port=0)
    server.start()
    try:
        # _post_json raises on a non-2xx (e.g. 500); a clean return proves 200.
        status, result = _post_json(
            server.url + "/command", {"name": "confirm_empty", "args": []}
        )
    finally:
        server.stop()

    assert status == 200
    assert result["ok"] is False


def test_command_endpoint_rejects_an_unknown_verb_with_400():
    handle = _FakeHandle()
    server = KioskServer(KioskRelay(handle), host="127.0.0.1", port=0)
    server.start()
    try:
        req = Request(
            server.url + "/command",
            data=json.dumps({"name": "delete_everything", "args": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status = None
        try:
            urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:  # HTTPError is a subclass; capture its code
            status = getattr(exc, "code", None)
    finally:
        server.stop()

    assert status == 400
    assert handle.calls == []


def test_review_still_returns_404_when_no_pending_capture():
    server = _server()
    server.start()
    try:
        req = Request(server.url + "/review-still", method="GET")
        status = None
        try:
            urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:
            status = getattr(exc, "code", None)
    finally:
        server.stop()

    assert status == 404


def test_preview_frame_returns_503_placeholder_when_no_frame():
    server = _server()
    server.start()
    try:
        req = Request(server.url + "/preview-frame", method="GET")
        status = None
        body = b""
        content_type = ""
        try:
            with urlopen(req, timeout=5) as resp:  # noqa: S310
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read()
        except Exception as exc:
            status = getattr(exc, "code", None)
            if hasattr(exc, "read"):
                body = exc.read()
            if hasattr(exc, "headers"):
                content_type = exc.headers.get("Content-Type", "")
    finally:
        server.stop()

    assert status == 503
    assert "image/jpeg" in content_type
    assert body[:2] == b"\xff\xd8"


def test_preview_frame_returns_live_jpeg_from_runtime():
    import cv2
    import numpy as np

    from kiosk.images import encode_preview_jpeg

    frame = np.full((120, 160, 3), 42, dtype=np.uint8)
    payload = encode_preview_jpeg(frame)

    class _PreviewHandle(_FakeHandle):
        def latest_preview_jpeg(self):
            return payload

    server = KioskServer(KioskRelay(_PreviewHandle()), host="127.0.0.1", port=0)
    server.start()
    try:
        status, content_type, body = _get(server.url + "/preview-frame")
    finally:
        server.stop()

    assert status == 200
    assert "image/jpeg" in content_type
    assert body == payload
    decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


def test_review_still_returns_downscaled_jpeg(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "capture_1_block_20260709T120000Z.png"
    cv2.imwrite(str(path), np.zeros((3040, 4056), dtype=np.uint8))

    class _StillHandle(_FakeHandle):
        def review_still_jpeg(self):
            from kiosk.images import encode_still_jpeg
            return encode_still_jpeg(path)

    server = KioskServer(KioskRelay(_StillHandle()), host="127.0.0.1", port=0)
    server.start()
    try:
        status, content_type, body = _get(server.url + "/review-still?id=capture_1")
    finally:
        server.stop()

    assert status == 200
    assert "image/jpeg" in content_type
    assert body[:2] == b"\xff\xd8"


def test_get_inspection_sheet_returns_png_bytes():
    """#151: GET /inspection-sheet?path=... surfaces the rendered contact
    sheet for an expanded REVIEW row, mirroring /review-still's shape."""
    payload = b"\x89PNGfakepng"

    class _SheetHandle(_FakeHandle):
        def inspection_sheet_bytes(self, path):
            self.calls.append(("inspection_sheet_bytes", path))
            return payload

    server = KioskServer(KioskRelay(_SheetHandle()), host="127.0.0.1", port=0)
    server.start()
    try:
        status, content_type, body = _get(
            server.url + "/inspection-sheet?path=capture_9__51151378.png"
        )
    finally:
        server.stop()

    assert status == 200
    assert "image/png" in content_type
    assert body == payload


def test_get_inspection_sheet_returns_404_when_missing():
    server = _server()
    server.start()
    try:
        req = Request(
            server.url + "/inspection-sheet?path=missing.png", method="GET"
        )
        status = None
        try:
            urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:
            status = getattr(exc, "code", None)
    finally:
        server.stop()

    assert status == 404


def test_get_results_evidence_returns_jpeg_bytes():
    payload = b"\xff\xd8fakejpeg"

    class _EvidenceHandle(_FakeHandle):
        def results_evidence_bytes(self, path):
            self.calls.append(("results_evidence_bytes", path))
            return payload

    server = KioskServer(
        KioskRelay(_EvidenceHandle()), host="127.0.0.1", port=0
    )
    server.start()
    try:
        status, content_type, body = _get(
            server.url + "/results-evidence?path="
            + "%2Fsessions%2F1%2Fclaim_artifacts%2Fcap-1_block_thumb.jpg"
        )
    finally:
        server.stop()

    assert status == 200
    assert "image/jpeg" in content_type
    assert body == payload


def test_get_results_evidence_reads_laptop_artifact_through_remote_store(tmp_path):
    """The Pi kiosk gets bytes from the receiver, never from its local disk."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    artifacts = session.directory / "claim_artifacts"
    artifacts.mkdir()
    jpeg = artifacts / "cap-1_slide_thumb.jpg"
    jpeg.write_bytes(b"\xff\xd8remote-evidence")

    with LoopbackCaptureReceiver(store) as receiver:
        from store.remote import RemoteProcessingStore

        workflow = SessionWorkflow(
            session=session,
            store=RemoteProcessingStore(receiver.url, backoff=0),
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(tmp_path / "framing.json"),
        )
        server = KioskServer(KioskRelay(workflow), host="127.0.0.1", port=0)
        server.start()
        try:
            status, content_type, body = _get(
                server.url + "/results-evidence?path=" + quote(str(jpeg))
            )
        finally:
            server.stop()

    assert status == 200
    assert "image/jpeg" in content_type
    assert body == b"\xff\xd8remote-evidence"


def test_get_results_evidence_returns_404_when_missing():
    server = _server()
    server.start()
    try:
        req = Request(
            server.url + "/results-evidence?path=missing.jpg", method="GET"
        )
        status = None
        try:
            urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:
            status = getattr(exc, "code", None)
    finally:
        server.stop()

    assert status == 404


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_get_inspection_sheet_returns_the_real_contact_sheet_for_a_scored_review_row(
    tmp_path,
):
    """#153: expanding a REVIEW row must reach a REAL rendered contact sheet
    through the full server -> relay -> ``SessionWorkflow.
    inspection_sheet_bytes`` path -- not only the pure ``kiosk.inspection``
    projection layer. Drives a real work-order lifecycle to a
    claim-disagreement REVIEW verdict (mirrors ``test_session_workflow``'s
    contact-sheet tests), then fetches the sheet over real HTTP.
    ``lightweight_qc_artifacts`` is requested explicitly (see the module-level
    import comment above) since autouse fixtures don't cross module
    boundaries."""
    from urllib.parse import quote

    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    renderer = StubContactSheetRenderer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
        contact_sheet_renderer=renderer,
    )
    session = store.start_session(started_at=WF_STARTED_AT)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_b = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
    )

    workflow.start_work_order()
    # Claimed block is block_b, but block_a scores highest -> disagreement.
    slide_id = store.record_slide_capture(
        session.number, wf_capture(tmp_path / "slide.png", 120),
        captured_at=WF_STARTED_AT, result=_valid_slide_result(block_b), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_a: 0.9, block_b: 0.2}
    workflow.finish_work_order()
    store.wait_for_jobs()

    row = next(
        r for r in workflow.list_results_ready_work_orders()
        if r["capture_id"] == slide_id
    )
    assert row["verdict"] == "REVIEW"
    expected_descriptors = project_inspection(row)

    server = KioskServer(KioskRelay(workflow), host="127.0.0.1", port=0)
    server.start()
    try:
        status, content_type, body = _get(
            server.url + "/inspection?capture_id=" + quote(slide_id, safe="")
        )
        assert status == 200
        assert "application/json" in content_type
        descriptors = json.loads(body)
        assert descriptors == expected_descriptors

        # Chain a real /inspection-sheet fetch of a returned descriptor path
        # to prove the descriptor endpoint feeds the byte-serving route.
        top_match = next(d for d in descriptors if d["role"] == "TOP MATCH")
        sheet_path = f"{row['contact_sheet_dir']}/{top_match['unique_id']}.png"
        assert Path(sheet_path).is_file(), "the renderer must have written the real sheet"
        status, content_type, body = _get(
            server.url + "/inspection-sheet?path=" + quote(sheet_path, safe="")
        )
    finally:
        server.stop()

    assert status == 200
    assert "image/png" in content_type
    assert body == Path(sheet_path).read_bytes()
