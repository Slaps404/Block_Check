"""Walking-skeleton contract for the Pi-local kiosk relay.

The relay is a SECOND renderer of the SAME in-process ``SessionWorkflow`` beside
``session_console`` (ADR 0004). It owns no phase/recovery logic: it READS via the
workflow's pure observe API (``snapshot``/``summarize``/``events``) and WRITES only
through the existing ``session_console.dispatch`` verbs + the ADR-0002 ``request_id``
ledger. These tests lock its three foundation seams.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

import pytest

from kiosk.relay import KioskRelay
from kiosk.results_table import project_results_table
from transport.remote import TransportError
from session.session_mode import SessionMode
from session.workflow import (
    PiOutbox,
    ProcessingStore,
    ScanOutcome,
    SessionWorkflow,
)

# Reuse test_session_workflow's real-workflow scoring fixtures/helpers
# rather than re-deriving a second work-order lifecycle harness. Imported at
# module level (not locally inside the test) so pytest can see
# ``lightweight_qc_artifacts`` as a requestable fixture -- autouse fixtures
# don't cross module boundaries, so tests here must ask for it explicitly.
from store_harness import (  # noqa: F401
    WORKFLOW_STARTED_AT as WF_STARTED_AT,
    FastPreprocessor,
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


# --------------------------------------------------------------------------
# fakes: a non-networked handle that records how the relay drives it
# --------------------------------------------------------------------------


class _FakeHandle:
    """Duck-types the workflow surface the relay touches, recording calls.

    Mirrors ``test_session_console._FakeWorkflow`` -- the relay must route
    through ``dispatch`` exactly like the console, adding no logic of its own.
    """

    def __init__(self, *, scanned=("51151378",), phase="blocks"):
        self.calls: list[tuple] = []
        self._phase = phase
        self._scanned = list(scanned)

    def scan_block(self, block_id):
        self.calls.append(("scan_block", block_id))
        self._scanned.append(block_id)
        return ScanOutcome(True, f"Accepted block {block_id}")

    # --- pure reads the relay renders -------------------------------------
    def snapshot(self):
        return _Snapshot(self._phase, self._scanned[-1] if self._scanned else None)

    def summarize(self):
        return _Summary(len(self._scanned))

    def events(self):
        return tuple(_Event(b) for b in self._scanned)


class _Snapshot:
    def __init__(self, phase, latest_block_id):
        self.phase = phase
        self.session_number = 1042
        self.latest_block_id = latest_block_id
        self.latest_block_status = "captured"


class _Summary:
    def __init__(self, count):
        self.session_number = 1042
        self.started_at = STARTED_AT
        self.sets_processed = count
        self.pass_count = count
        self.review_count = 0
        self.blocks_captured = count


class _Event:
    def __init__(self, block_id):
        self.kind = "block_scanned"
        self.session_number = 1042
        self.phase = "blocks"
        self.message = f"Accepted block {block_id}"
        self.block_id = block_id


class _RecordingLock:
    """A lock double that records acquire/release ordering."""

    def __init__(self):
        self.events: list[str] = []
        self._lock = Lock()

    def __enter__(self):
        self.events.append("acquire")
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        self.events.append("release")
        return False


# --------------------------------------------------------------------------
# read path
# --------------------------------------------------------------------------


def test_state_projects_the_live_observe_api_into_json_safe_values():
    relay = KioskRelay(_FakeHandle(scanned=("51151378",)))

    state = relay.state()

    assert state["online"] is True
    assert state["phase"] == "blocks"
    assert state["session_number"] == 1042
    assert state["latest_block_id"] == "51151378"
    assert state["captured"] == 1  # "N captured" status-bar signal
    # JSON-safe: no dataclasses/objects leak through.
    import json

    json.dumps(state)


class _BounceEvent:
    """A `hybrid_pool_freeze_insufficient_blocks` event, as `ProcessingStore.
    _bounce_hybrid_pool_to_blocks` emits it."""

    def __init__(self, message):
        self.kind = "hybrid_pool_freeze_insufficient_blocks"
        self.session_number = 1042
        self.phase = "blocks"
        self.message = message
        self.block_id = None


class _FakeHandleWithEvents(_FakeHandle):
    def __init__(self, *, events, **kwargs):
        super().__init__(**kwargs)
        self._fixed_events = events

    def events(self):
        return tuple(self._fixed_events)


def test_hybrid_pool_bounce_message_surfaces_only_while_it_is_the_latest_event():
    """The bounce reason must reach the operator through its
    OWN dedicated field, distinct from `last_event` (which is consumed purely
    as a duplicate-scan dedupe key, never rendered as text)."""
    message = (
        "Only 1 usable block(s); Hybrid requires at least 2. Capture more "
        "blocks, then Finish Blocks again."
    )
    handle = _FakeHandleWithEvents(events=[_BounceEvent(message)])

    state = KioskRelay(handle).state()

    assert state["hybrid_pool_bounce_message"] == message
    # last_event itself is untouched/still usable for its existing purpose.
    assert state["last_event"]["kind"] == "hybrid_pool_freeze_insufficient_blocks"


def test_hybrid_pool_bounce_message_clears_once_a_newer_event_supersedes_it():
    """A later, different-kind event (e.g. a fresh block scan) naturally
    clears the notice -- it is derived from the latest event, not latched."""
    message = "Only 1 usable block(s); Hybrid requires at least 2."
    handle = _FakeHandleWithEvents(
        events=[_BounceEvent(message), _Event("22222222")]
    )

    state = KioskRelay(handle).state()

    assert state["hybrid_pool_bounce_message"] is None


def test_state_reports_offline_instead_of_raising_when_link_drops():
    class _Offline:
        def snapshot(self):
            raise TransportError("processing computer offline")

        def summarize(self):
            raise TransportError("processing computer offline")

        def events(self):
            raise TransportError("processing computer offline")

    state = KioskRelay(_Offline()).state()

    assert state["online"] is False  # seam the 119c banner formalizes


# --------------------------------------------------------------------------
# write path -- routes through dispatch, owns no logic
# --------------------------------------------------------------------------


def test_command_routes_a_tap_through_the_existing_dispatch_verb():
    handle = _FakeHandle()
    relay = KioskRelay(handle)

    outcome = relay.command("scan_block", "51151378")

    assert isinstance(outcome, ScanOutcome) and outcome.accepted
    assert handle.calls == [("scan_block", "51151378")]


def test_command_rejects_unknown_verbs_without_touching_the_workflow():
    handle = _FakeHandle()
    relay = KioskRelay(handle)

    with pytest.raises(ValueError, match="unknown command"):
        relay.command("delete_everything")

    assert handle.calls == []


def test_command_serializes_writes_on_the_relays_own_lock():
    handle = _FakeHandle()
    lock = _RecordingLock()
    relay = KioskRelay(handle, lock=lock)

    relay.command("scan_block", "51151378")

    # Each write brackets the dispatch in exactly one acquire/release.
    assert lock.events == ["acquire", "release"]


# --------------------------------------------------------------------------
# idempotency seam -- a tap really does ride the request_id ledger
# --------------------------------------------------------------------------


class _Ev:
    def __init__(self, kind, *, capture_id=None, block_id=None, message=""):
        self.kind = kind
        self.phase = "slides"
        self.capture_id = capture_id
        self.block_id = block_id
        self.message = message


class _RuntimeFake:
    """Duck-types the PiCaptureRuntime surface the relay reads: adds
    capture_status() + skip_unreadable_slide() + a configurable event tail."""

    def __init__(self, *, capture_state, capture_mode, phase="slides", events=()):
        self.calls: list[tuple] = []
        self._cs = capture_state
        self._cm = capture_mode
        self._phase = phase
        self._events = tuple(events)
        # Pause/resume pure state, mirroring SessionWorkflow's own
        # capture_paused attribute (relay.state() reads this via a plain
        # degrading getattr, never a callable).
        self.capture_paused = False

    def snapshot(self):
        s = _Snapshot(self._phase, None)
        s.phase = self._phase
        return s

    def summarize(self):
        return _Summary(0)

    def events(self):
        return self._events

    def capture_status(self):
        return {"capture_state": self._cs, "capture_mode": self._cm}

    def skip_unreadable_slide(self):
        self.calls.append(("skip_unreadable_slide",))

    def next_capture(self):
        self.calls.append(("next_capture",))

    def pause_capture(self):
        self.calls.append(("pause_capture",))
        self.capture_paused = True

    def resume_capture(self):
        self.calls.append(("resume_capture",))
        self.capture_paused = False


def test_state_exposes_capture_layer_signals_and_routes_to_a_fine_screen():
    handle = _RuntimeFake(capture_state="WAITING_FOR_SCAN", capture_mode="block",
                          phase="blocks")
    state = KioskRelay(handle).state({"engaged": True})

    assert state["capture_state"] == "WAITING_FOR_SCAN"
    assert state["capture_mode"] == "block"
    assert state["screen"] == "06"  # engaged block idle -> Scan Block


def test_state_id_matched_verdict_paints_pass_but_stale_one_does_not_bleed():
    passed = _RuntimeFake(
        capture_state="WAITING_FOR_REMOVAL", capture_mode="slide",
        events=(_Ev("slide_captured", capture_id="cap-7"),
                _Ev("claim_pass", capture_id="cap-7", message="PASS: ok")),
    )
    state = KioskRelay(passed).state({"engaged": True})
    assert state["latest_verdict"]["kind"] == "claim_pass"
    assert state["current_capture_id"] == "cap-7"
    assert state["screen"] == "15"

    # A newer slide capture advances current_capture_id; the old PASS no longer
    # id-matches, so the router falls back to Verifying (no green bleed).
    stale = _RuntimeFake(
        capture_state="WAITING_FOR_REMOVAL", capture_mode="slide",
        events=(_Ev("slide_captured", capture_id="cap-7"),
                _Ev("claim_pass", capture_id="cap-7", message="PASS: ok"),
                _Ev("slide_captured", capture_id="cap-8")),
    )
    assert KioskRelay(stale).state({"engaged": True})["screen"] == "14"


@pytest.mark.parametrize(
    "verbs, expected_calls, expected_paused",
    [
        (["skip_slide"], [("skip_unreadable_slide",)], False),
        (["next_capture"], [("next_capture",)], False),
        (["pause_capture"], [("pause_capture",)], True),
        (["resume_capture"], [("resume_capture",)], False),
        # The whole pause/resume cycle records ONLY the two verbs --
        # the fake has no job methods, so any cancel/drain/wait would raise.
        (["pause_capture", "resume_capture"],
         [("pause_capture",), ("resume_capture",)], False),
    ],
)
def test_command_routes_thin_verbs_through_to_the_handle(
    verbs, expected_calls, expected_paused
):
    handle = _RuntimeFake(capture_state="EMPTY", capture_mode="slide")
    handle.capture_paused = verbs == ["resume_capture"]
    relay = KioskRelay(handle)

    for verb in verbs:
        relay.command(verb)

    assert handle.calls == expected_calls
    assert handle.capture_paused is expected_paused


# --------------------------------------------------------------------------
# View Results during an open Hybrid slide-capture work order --
# entry pauses automatic capture only; Go Back resumes the same bracket.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hybrid_mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_hybrid_open_slide_capture_work_order_reaches_results_table_and_pause_is_visible(
    hybrid_mode,
):
    """Results is reachable from an OPEN Hybrid
    slide-capture work order (not just between orders), and the pause state
    the VIEW RESULTS verb sets is visible in the same state projection."""
    handle = _ResultsFake(
        capture_state="EMPTY", capture_mode="slide", phase="slides",
        work_orders=(11,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PENDING",
             "claim_reason": "", "claim_score": None, "work_order_id": 11,
             "work_order": "12080"},
        ],
    )
    handle.session_mode = hybrid_mode
    handle.open_retrieval = False
    handle.work_order_open = True

    before = KioskRelay(handle).state({"view_results_guard": False})
    assert before["screen"] == "slide_capture_work_order"
    assert before["capture_paused"] is False

    handle.pause_capture()
    after = KioskRelay(handle).state({"view_results_guard": True})
    assert after["screen"] == "results_table"
    assert after["capture_paused"] is True
    # Nothing about the bracket/phase changed -- pausing capture is not
    # destructive (requirement: never finish the work order / end the
    # session / clear the bracket).
    assert handle.work_order_open is True

    handle.resume_capture()
    back = KioskRelay(handle).state({"view_results_guard": False})
    assert back["screen"] == "slide_capture_work_order"
    assert back["capture_paused"] is False
    assert handle.work_order_open is True


# --------------------------------------------------------------------------
# WS-C: `remaining` (len(missing_slides)) + forwarded `pending_block_id`
# --------------------------------------------------------------------------


class _RemainingSummary:
    """A summary carrying the ``missing_slides`` tuple the real SessionSummary
    has (the walking-skeleton ``_Summary`` above omits it)."""

    def __init__(self, missing):
        self.session_number = 1042
        self.started_at = STARTED_AT
        self.sets_processed = 0
        self.pass_count = 0
        self.review_count = 0
        self.missing_slides = tuple(missing)


class _RemainingFake:
    """Slide-phase handle exposing missing_slides + a capture_status that
    latches a scanned-but-not-captured block id (screens 07/18)."""

    def __init__(self, *, missing=(), pending_block_id=None, pending_slide_id=None):
        self._missing = missing
        self._pending = pending_block_id
        self._pending_slide = pending_slide_id

    def snapshot(self):
        s = _Snapshot("slides", None)
        return s

    def summarize(self):
        return _RemainingSummary(self._missing)

    def events(self):
        return ()

    def capture_status(self):
        status = {"capture_state": "EMPTY", "capture_mode": "slide"}
        if self._pending is not None:
            status["pending_block_id"] = self._pending
        if self._pending_slide is not None:
            status["pending_slide_id"] = self._pending_slide
        return status


def test_state_projects_remaining_and_forwards_pending_block_id():
    handle = _RemainingFake(missing=("b1", "b2", "b3"), pending_block_id="51151378")

    state = KioskRelay(handle).state({"engaged": True})

    # remaining = number of slides ready-and-awaiting capture (len missing_slides)
    assert state["remaining"] == 3
    # pending_block_id (dropped before WS-C) now surfaces from capture_status
    assert state["pending_block_id"] == "51151378"


def test_state_forwards_pending_slide_id():
    # The scanned-slide display id surfaces from capture_status so screen 12 can
    # confirm the scan registered before the slide is placed.
    handle = _RemainingFake(pending_slide_id="51137181")

    state = KioskRelay(handle).state({"engaged": True})

    assert state["pending_slide_id"] == "51137181"


def test_state_projects_open_retrieval_work_order_flags_from_handle_attributes():
    """When the handle (a real ``PiCaptureRuntime``) exposes these as
    plain attributes, the relay forwards them verbatim -- this is the signal
    the router's between_orders/block_scan_work_order gate reads."""
    handle = _FakeHandle()
    handle.open_retrieval = True
    handle.has_work_orders = lambda: True

    state = KioskRelay(handle).state()

    assert state["has_work_orders"] is True
    assert state["screen"] == "between_orders"

    handle.work_order_open = True
    state = KioskRelay(handle).state()

    assert state["open_retrieval"] is True
    assert state["work_order_open"] is True


def test_state_degrades_every_optional_projection_on_a_bare_handle():
    """A bare handle exposes no capture_status pending ids, missing_slides,
    capture_paused, open_retrieval/work_order_open/has_work_orders,
    session_mode, results_status or open_work_order_id: each must degrade to
    its documented default, never raise."""
    state = KioskRelay(_FakeHandle()).state()

    assert state["remaining"] == 0
    assert state["pending_block_id"] is None
    assert state["pending_slide_id"] is None
    assert state["capture_paused"] is False
    assert state["open_retrieval"] is False
    assert state["work_order_open"] is False
    assert state["has_work_orders"] is False
    # "normal" as a plain JSON-safe string, not a SessionMode enum or None.
    assert state["mode"] == "normal"
    assert isinstance(state["mode"], str)
    assert state["attention"] is None
    assert state["results_ready_work_orders"] == ()
    assert state["results_rows"] == []


# --------------------------------------------------------------------------
# The third, non-interrupting attention-banner layer
# --------------------------------------------------------------------------


class _FakeAttentionHandle(_FakeHandle):
    """Duck-types just the three signals `KioskRelay._attention_banner`
    reads (`results_status`, `session_mode`, `open_work_order_id`) on top
    of `_FakeHandle`'s snapshot/summarize/events shape, so `relay.state()`
    still succeeds end-to-end without a real store or camera."""

    def __init__(self, *, rows=(), session_mode=SessionMode.HYBRID,
                 work_order_open=False, open_work_order_id=None, **kwargs):
        super().__init__(**kwargs)
        self.session_mode = session_mode
        self.work_order_open = work_order_open
        self._open_work_order_id = open_work_order_id
        self._rows = rows

    def results_status(self):
        return {"work_orders": (), "rows": list(self._rows)}

    def open_work_order_id(self):
        return self._open_work_order_id


def _error_row(**overrides):
    base = dict(
        capture_id="cap-1", block_id="11111111", work_order_id=5,
        verdict="ERROR", claim_score=None, claim_reason=None,
    )
    base.update(overrides)
    return base


def test_attention_banner_surfaces_for_a_hybrid_error_row():
    handle = _FakeAttentionHandle(rows=(_error_row(),))

    state = KioskRelay(handle).state()

    assert state["attention"] == {
        "capture_id": "cap-1", "block_id": "11111111", "work_order_id": 5,
        "message": "Slide needs attention: block 11111111",
        "can_recapture": True,
    }
    # The banner never changes which screen is active by itself.
    assert state["screen"] != "hybrid_attention"


@pytest.mark.parametrize("mode", [SessionMode.NORMAL, SessionMode.OPEN_RETRIEVAL])
def test_attention_banner_absent_for_normal_and_open_retrieval(mode):
    """Non-vacuous control: must fail if the explicit mode gate in
    `KioskRelay._attention_banner` is dropped -- an ERROR row is supplied
    here precisely so "no banner" is proven by the gate, not merely by the
    absence of ERROR-shaped data."""
    handle = _FakeAttentionHandle(rows=(_error_row(),), session_mode=mode)
    state = KioskRelay(handle).state()
    assert state["attention"] is None


@pytest.mark.parametrize(
    "mode, expected",
    [
        (SessionMode.NORMAL, "normal"),
        (SessionMode.OPEN_RETRIEVAL, "open_retrieval"),
        (SessionMode.HYBRID, "hybrid"),
        (SessionMode.HYBRID_SHADOW, "hybrid_shadow"),
    ],
)
def test_state_mode_projects_each_session_mode_as_its_explicit_string_value(
    mode, expected
):
    handle = _FakeHandle()
    handle.session_mode = mode

    state = KioskRelay(handle).state()

    assert state["mode"] == expected
    # kiosk serves this over HTTP -- must be the plain string value, never
    # the SessionMode member itself (which json.dumps cannot serialize).
    assert isinstance(state["mode"], str)
    import json

    json.dumps(state)


def test_state_mode_degrades_to_normal_not_a_silent_guess_on_a_bogus_attribute():
    """ADR-0016: missing/unexpected receiver-owned state must fail to the
    documented default, never silently guess from whatever happens to be
    present. ``KioskRelay._mode_value`` enforces this with an explicit
    ``isinstance(mode, SessionMode)`` check, not a truthy/attribute-exists
    check -- so a handle carrying a plain string (not a real SessionMode
    member, e.g. a stale/hand-rolled value) still degrades to "normal"
    rather than being coerced into that string. Regression for a future
    refactor collapsing the isinstance check into a bare ``getattr(...,
    "normal")`` default, which would silently guess instead of failing
    safe."""
    handle = _FakeHandle()
    handle.session_mode = "hybrid"  # NOT a SessionMode instance

    state = KioskRelay(handle).state()

    assert state["mode"] == "normal"


def test_state_open_bracket_surfaces_effective_engaged_and_routes_to_calibrating():
    # With an open work order but the client latch off, relay.state()
    # must (a) route to Calibrating (05) and (b) surface effective_engaged=True
    # so the client's maybeAutoCalibrate fires confirm_empty. Regression for the
    # open-retrieval "Calibrating… forever" hang.
    handle = _RuntimeFake(
        capture_state="AWAITING_BASELINE_CONFIRMATION", capture_mode="block",
        phase="blocks",
    )
    handle.open_retrieval = True
    handle.work_order_open = True

    state = KioskRelay(handle).state({"engaged": False})

    assert state["screen"] == "05"
    assert state["effective_engaged"] is True
    # the raw echo stays honest (client sent engaged=False)
    assert state["engaged"] is False


def test_state_projects_pending_capture_id_for_review_gate():
    class _ReviewFake(_RuntimeFake):
        def pending_capture_status(self):
            return {"capture_id": "capture_3_block_20260709T120000Z"}

    handle = _ReviewFake(
        capture_state="AWAITING_ACCEPT", capture_mode="block", phase="blocks"
    )
    state = KioskRelay(handle).state({"engaged": True})

    assert state["pending_capture_id"] == "capture_3_block_20260709T120000Z"
    assert state["screen"] == "capture_review"


# --------------------------------------------------------------------------
# Results table: degrading projection of results-ready work orders
# --------------------------------------------------------------------------


class _ResultsFake(_RuntimeFake):
    """Adds ``results_status()`` -- the same degrading getattr+callable shape
    as ``capture_status``/``pending_capture_status`` -- carrying every
    results-ready work order's per-slide verdict rows in the session."""

    def __init__(self, *, work_orders, rows, **kwargs):
        super().__init__(**kwargs)
        self._work_orders = work_orders
        self._rows = rows

    def results_status(self):
        return {"work_orders": self._work_orders, "rows": self._rows}


def test_state_projects_rows_from_every_results_ready_work_order_in_the_session():
    handle = _ResultsFake(
        capture_state="WAITING_FOR_SCAN", capture_mode="block", phase="blocks",
        work_orders=(3, 4),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PASS",
             "claim_reason": "", "claim_score": 0.91, "work_order_id": 3,
             "work_order": "12080"},
            {"capture_id": "cap-2", "block_id": "b2", "verdict": "REVIEW",
             "claim_reason": "ambiguous near-miss", "claim_score": 0.40,
             "work_order_id": 4, "work_order": "12094"},
        ],
    )

    state = KioskRelay(handle).state({"engaged": True})

    assert state["results_ready_work_orders"] == (3, 4)
    # relay.state() hands the screen project_results_table's
    # sorted/colored output (REVIEW-first), not a raw store passthrough.
    assert [row["capture_id"] for row in state["results_rows"]] == ["cap-2", "cap-1"]
    # The human work-order number is the results screen's grouping key --
    # it must survive end to end from results_status() rows into results_rows.
    assert {row["capture_id"]: row["work_order"] for row in state["results_rows"]} == {
        "cap-1": "12080", "cap-2": "12094",
    }


@pytest.mark.usefixtures("lightweight_qc_artifacts")
def test_relay_state_keeps_scored_work_order_on_hub_until_view_results(tmp_path):
    """Once results are ready, keep the preview hub visible until the operator
    taps VIEW RESULTS; then hand the results screen ``project_results_table``'s
    sorted/colored
    output (REVIEW first, ``color``/``expand_target`` attached), not raw
    store rows. Drives a REAL ``SessionWorkflow`` end to end through the
    work-order lifecycle; nothing here is mocked. ``lightweight_qc_artifacts``
    is ``test_session_workflow``'s own autouse fixture -- it must be
    requested explicitly here since autouse fixtures don't cross module
    boundaries -- swapping the real (large) QC-panel rendering for a tiny
    stub so ``FastPreprocessor``'s 8x8 stand-in mask doesn't shape-mismatch
    against the full-size synthetic block capture."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=WF_STARTED_AT)
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_2 = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
    )

    workflow.start_work_order()
    slide_a = store.record_slide_capture(
        session.number, wf_capture(tmp_path / "slide_a.png", 120),
        captured_at=WF_STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9, block_2: 0.1}
    workflow.finish_work_order()
    store.wait_for_jobs()

    workflow.start_work_order()
    block_3 = _evaluable_block(store, session, tmp_path, block_id="73737373")
    block_4 = _evaluable_block(store, session, tmp_path, block_id="84848484")
    _drain_to_slides(store, session)
    slide_b = store.record_slide_capture(
        session.number, wf_capture(tmp_path / "slide_b.png", 121),
        captured_at=WF_STARTED_AT, result=_valid_slide_result(block_3), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_b] = {block_3: 0.05, block_4: 0.02}
    workflow.finish_work_order()
    store.wait_for_jobs()

    # Ready data alone must not steal focus from the between-orders preview.
    workflow.open_retrieval = True
    workflow.work_order_open = False
    state = KioskRelay(workflow).state({"engaged": True})
    assert state["screen"] == "between_orders"

    state = KioskRelay(workflow).state(
        {"engaged": True, "view_results_guard": True}
    )
    assert state["screen"] == "results_table"
    raw_rows = workflow.results_status()["rows"]
    assert state["results_rows"] == project_results_table(raw_rows)
    # Projected render fields, not a raw passthrough. The 0.03 lead is a
    # PASS under the current 0.02 work-order margin.
    assert state["results_rows"][0]["verdict"] == "PASS"
    assert {"color", "expand_target"} <= state["results_rows"][0].keys()
    assert {"block_thumb", "slide_thumb", "block_display", "slide_display",
            "overlay_display"} <= state["results_rows"][0]["evidence"].keys()
    # The SELECT emits the human work-order number (from the slide's
    # "12080_<block>_01_HE" barcode) all the way through to the results screen,
    # where it is the section grouping key. Proven here on a REAL workflow.
    assert {row["work_order"] for row in state["results_rows"]} == {"12080"}
