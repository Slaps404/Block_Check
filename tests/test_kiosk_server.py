"""Walking-skeleton transport shell for the kiosk.

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
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from kiosk.inspection import project_inspection
from kiosk.relay import KioskRelay
from kiosk.screens import CATALOG
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

# Reuse test_session_workflow's real-workflow scoring fixtures/helpers
# rather than re-deriving a second work-order lifecycle harness. Imported at
# module level (not locally inside the test) so pytest can see
# ``lightweight_qc_artifacts`` as a requestable fixture -- autouse fixtures
# don't cross module boundaries, so the test here must ask for it explicitly.
from store_harness import (  # noqa: F401
    WORKFLOW_STARTED_AT as WF_STARTED_AT,
    FastPreprocessor,
    StubContactSheetRenderer,
    StubWorkOrderScorer,
    ToggleTransport,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
)
from store_harness import (  # noqa: F401
    full_res_capture as wf_capture,
    lightweight_qc_artifacts,
    valid_slide_result as _valid_slide_result,
    lightweight_slide_qc,
)


@pytest.fixture(autouse=True)
def _lightweight_qc(lightweight_qc_artifacts, lightweight_slide_qc):  # noqa: F811
    """Autouse shims do not cross module boundaries, so opt this
    module into both shared harness QC fixtures."""

STARTED_AT = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)
_INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "code" / "kiosk" / "static" / "index.html"
)


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


def _fetch(handle, path):
    """GET ``path`` from a fresh server; HTTP errors return their status too."""
    server = _server(handle)
    server.start()
    try:
        return _get(server.url + path)
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()
    finally:
        server.stop()


def _post_json(url, payload):
    body = json.dumps(payload).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:  # noqa: S310
        return resp.status, json.loads(resp.read())


def test_snap_page_serves_space_to_capture_ui():
    status, content_type, body = _fetch(None, "/snap")
    assert status == 200
    assert content_type.startswith("text/html")
    html = body.decode()
    assert "/preview-frame" in html
    assert '"snap_quiet"' in html
    assert '"deskew_quiet"' in html
    assert 'id="note"' in html


def test_relay_boots_and_serves_one_screen_without_the_deck_runtime():
    status, content_type, body = _fetch(None, "/")

    html = body.decode()
    assert status == 200
    assert "text/html" in content_type
    assert "Scan Block" in html  # the extracted screen 06
    # deck runtime must be gone -- screen selection is event-driven, not deck-nav.
    assert "<x-dc" not in html
    assert "deck-stage" not in html
    assert "support.js" not in html


# Client-side contracts pinned textually (no JS test harness in this repo).
@pytest.mark.parametrize(
    "present, absent",
    [
        # view_results_guard rides every /state poll.
        (["view_results_guard: false", 'p.set("view_results_guard"'], []),
        # A slow /state response must not create an unbounded poll backlog.
        (
            ["let statePollInFlight = false;", "let statePollQueued = false;",
             "if (statePollInFlight)", "setInterval(refresh, STATE_POLL_MS)"],
            ["setInterval(refresh, 750)"],
        ),
        # The results table renders the projected rows.
        (
            ['id="results-table"', "function renderResultsTable(rows)",
             "renderResultsTable(state.results_rows || [])"],
            [],
        ),
        # CSS vars backing the four-state colors (lookup table itself is
        # pinned against Python in test_kiosk_results_table).
        (["--rt-error:", "--rt-pending:"], []),
        # An unrecognized verdict falls through to rt-unknown, not REVIEW.
        (
            ['function rtStateClass(verdict) '
             '{ return RT_STATE_COLOR[verdict] || "rt-unknown"; }'],
            ['icon + (verdict || "REVIEW")'],
        ),
        # The repaint fingerprint includes r.verdict so PENDING repaints.
        (
            ["JSON.stringify(rows.map(r =>",
             "[r.work_order, r.block_id, r.verdict, r.capture_id, r.claim_reason,"],
            [],
        ),
        # Auto-open considers ERROR as well as REVIEW.
        (
            ['function rtNeedsAttention(verdict) '
             '{ return verdict === "ERROR" || verdict === "REVIEW"; }',
             "firstAttentionOpened"],
            [],
        ),
    ],
    ids=["view_results_guard", "poll_coalescing", "results_table",
         "state_css_vars", "unknown_state", "repaint_fingerprint", "auto_open"],
)
def test_index_html_client_contract(present, absent):
    html = _INDEX_HTML.read_text(encoding="utf-8")
    for snippet in present:
        assert snippet in html
    for snippet in absent:
        assert snippet not in html


def test_state_endpoint_exposes_the_read_path_as_json():
    handle = _FakeHandle()
    handle.scan_block("51151378")
    status, content_type, body = _fetch(handle, "/state")

    assert status == 200
    assert "application/json" in content_type
    state = json.loads(body)
    assert state["online"] is True
    assert state["captured"] == 1
    assert state["latest_block_id"] == "51151378"


def test_state_endpoint_forwards_the_view_results_guard():
    state = json.loads(_fetch(None, "/state?view_results_guard=1")[2])

    assert state["view_results_guard"] is True


def test_end_to_end_tap_over_http_mutates_the_real_workflow(tmp_path):
    """The whole thread through the REAL stack: browser POST -> HTTP
    server -> relay -> dispatch -> SessionWorkflow -> remote store, then the
    read path shows the mutation. No fakes below the HTTP boundary."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        from transport.remote import RemoteProcessingStore

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
    # a distinction the screen-07 router must respect.)
    assert after["event_count"] > before["event_count"]
    assert after["last_event"]["block_id"] == "51151378"
    # Durable mutation landed exactly once on the real store.
    assert store.awaiting_capture_blocks(session.number) == ("51151378",)



def test_catalog_endpoint_serves_the_static_screen_catalog():
    """GET /catalog is pure static data: 200 + JSON equal to the production
    screen CATALOG. No session needed."""
    status, content_type, body = _fetch(None, "/catalog")

    assert status == 200
    assert "application/json" in content_type
    assert json.loads(body) == json.loads(json.dumps(CATALOG))


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
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)  # noqa: S310
    finally:
        server.stop()

    assert exc_info.value.code == 400
    assert handle.calls == []


@pytest.mark.parametrize(
    "path",
    ["/review-still", "/inspection-sheet?path=missing.png",
     "/results-evidence?path=missing.jpg"],
)
def test_image_route_returns_404_when_the_handle_has_nothing(path):
    # A bare handle lacks the byte method, so the relay degrades to None.
    assert _fetch(None, path)[0] == 404


def test_preview_frame_returns_503_placeholder_when_no_frame():
    status, content_type, body = _fetch(None, "/preview-frame")

    assert status == 503
    assert "image/jpeg" in content_type
    assert body[:2] == b"\xff\xd8"


def _preview_payload(tmp_path):
    import numpy as np

    from kiosk.images import encode_preview_jpeg

    return encode_preview_jpeg(np.full((120, 160, 3), 42, dtype=np.uint8))


def _still_payload(tmp_path):
    import cv2
    import numpy as np

    from kiosk.images import encode_still_jpeg

    path = tmp_path / "capture_1_block_20260709T120000Z.png"
    cv2.imwrite(str(path), np.zeros((3040, 4056), dtype=np.uint8))
    return encode_still_jpeg(path)


@pytest.mark.parametrize(
    "method, route, content_type, make_payload, forwarded",
    [
        ("latest_preview_jpeg", "/preview-frame", "image/jpeg",
         _preview_payload, ()),
        ("review_still_jpeg", "/review-still?id=capture_1", "image/jpeg",
         _still_payload, ()),
        # Expanded REVIEW row's contact sheet.
        ("inspection_sheet_bytes", "/inspection-sheet?path=capture_9__51151378.png",
         "image/png", lambda _: b"\x89PNGfakepng", ("capture_9__51151378.png",)),
        ("results_evidence_bytes",
         "/results-evidence?path=%2Fsessions%2F1%2Fclaim_artifacts%2Fcap-1_block_thumb.jpg",
         "image/jpeg", lambda _: b"\xff\xd8fakejpeg",
         ("/sessions/1/claim_artifacts/cap-1_block_thumb.jpg",)),
    ],
    ids=["preview_frame", "review_still", "inspection_sheet", "results_evidence"],
)
def test_image_route_serves_the_handle_bytes(
    tmp_path, method, route, content_type, make_payload, forwarded
):
    payload = make_payload(tmp_path)
    handle = _FakeHandle()

    def _bytes(*args):
        handle.calls.append((method, *args))
        return payload

    setattr(handle, method, _bytes)

    status, served_type, body = _fetch(handle, route)

    assert status == 200
    assert content_type in served_type
    assert body == payload
    # The relay forwards the query path verbatim to the handle.
    assert handle.calls == [(method, *forwarded)]
    if content_type == "image/jpeg" and make_payload in (_preview_payload, _still_payload):
        import cv2
        import numpy as np

        assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) is not None


def test_get_results_evidence_reads_laptop_artifact_through_remote_store(tmp_path):
    """The Pi kiosk gets bytes from the receiver, never from its local disk."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    artifacts = session.directory / "claim_artifacts"
    artifacts.mkdir()
    jpeg = artifacts / "cap-1_slide_thumb.jpg"
    jpeg.write_bytes(b"\xff\xd8remote-evidence")

    with LoopbackCaptureReceiver(store) as receiver:
        from transport.remote import RemoteProcessingStore

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


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_get_inspection_sheet_returns_the_real_contact_sheet_for_a_scored_review_row(
    tmp_path,
):
    """Expanding a REVIEW row must reach a REAL rendered contact sheet
    through the full server -> relay -> ``SessionWorkflow.
    inspection_sheet_bytes`` path -- not only the pure ``kiosk.inspection``
    projection layer. Drives a real work-order lifecycle to a
    claim-disagreement REVIEW verdict, then fetches the sheet over real HTTP.
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
