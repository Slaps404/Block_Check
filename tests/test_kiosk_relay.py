"""Walking-skeleton contract for the Pi-local kiosk relay (#121 / 119a).

The relay is a SECOND renderer of the SAME in-process ``SessionWorkflow`` beside
``session_console`` (ADR 0004). It owns no phase/recovery logic: it READS via the
workflow's pure observe API (``snapshot``/``summarize``/``events``) and WRITES only
through the existing ``session_console.dispatch`` verbs + the ADR-0002 ``request_id``
ledger. These tests lock the three foundation seams 119b/119c build on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

import pytest

from kiosk.relay import KioskRelay
from store.remote import (
    RemoteProcessingStore,
    TransportError,
    UrlTransport,
)
from session.session_mode import SessionMode
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
# don't cross module boundaries, so tests here must ask for it explicitly.
from tests.test_session_workflow import (  # noqa: F401 -- fixture import
    STARTED_AT as WF_STARTED_AT,
    FastPreprocessor,
    StubWorkOrderScorer,
    ToggleTransport,
    _capture as wf_capture,
    _drain_to_slides,
    _evaluable_block,
    _identical_mask_slide_preprocessor,
    _valid_slide_result,
    _FakeResultsRowsStore,
    _results_workflow,
    lightweight_qc_artifacts,
)
from kiosk.results_table import project_results_table
from session.profile_report import PROFILE_STAGE_ORDER, ProfileRow

STARTED_AT = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


class DropFirstResponseTransport:
    """Lets the real request through but discards the FIRST response after the
    server durably applied it -- the "response never arrived" half of a
    mid-command disconnect (mirrors the fake in ``test_remote_store``)."""

    def __init__(self, inner):
        self.inner = inner
        self.post_calls = 0
        self.sent_payloads: list[bytes] = []

    def post(self, url, payload):
        self.post_calls += 1
        self.sent_payloads.append(payload)
        result = self.inner.post(url, payload)
        if self.post_calls == 1:
            raise TransportError("simulated: response dropped after apply")
        return result

    def get(self, url):
        return self.inner.get(url)


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


def test_state_updates_when_the_workflow_state_changes():
    handle = _FakeHandle(scanned=("51151378",))
    relay = KioskRelay(handle)

    before = relay.state()["captured"]
    handle.scan_block("22222222")
    after = relay.state()

    assert after["captured"] == before + 1
    assert after["latest_block_id"] == "22222222"


class _BounceEvent:
    """A `hybrid_pool_freeze_insufficient_blocks` event, as `ProcessingStore.
    _bounce_hybrid_pool_to_blocks` emits it (#250 review F1)."""

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
    """#250 review F1: the bounce reason must reach the operator through its
    OWN dedicated field, distinct from `last_event` (which is consumed purely
    as a duplicate-scan dedupe key, never rendered as text today)."""
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
    """Duck-types the PiCaptureRuntime surface the 119b relay reads: adds
    capture_status() + skip_unreadable_slide() + a configurable event tail."""

    def __init__(self, *, capture_state, capture_mode, phase="slides", events=()):
        self.calls: list[tuple] = []
        self._cs = capture_state
        self._cm = capture_mode
        self._phase = phase
        self._events = tuple(events)
        # #257: pause/resume pure state, mirroring SessionWorkflow's own
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


def test_state_boot_chooser_when_client_not_engaged():
    handle = _RuntimeFake(capture_state="WAITING_FOR_SCAN", capture_mode="block",
                          phase="blocks")
    # No engaged flag -> boot chooser. captured==0 (default summary) -> 01 Start.
    assert KioskRelay(handle).state()["screen"] == "01"


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


@pytest.mark.parametrize("kind", ("claim_pass", "claim_review"))
def test_hybrid_open_work_order_keeps_neutral_slide_removal_after_background_verdict(kind):
    handle = _RuntimeFake(
        capture_state="WAITING_FOR_REMOVAL", capture_mode="slide",
        events=(_Ev("slide_captured", capture_id="cap-7"),
                _Ev(kind, capture_id="cap-7", message="background result")),
    )
    handle.session_mode = SessionMode.HYBRID
    handle.work_order_open = True

    state = KioskRelay(handle).state({"engaged": True})

    assert state["latest_verdict"]["kind"] == kind
    assert state["screen"] == "hybrid_slide_queued"


def test_finish_blocks_guard_flag_routes_to_confirm_screen():
    handle = _RuntimeFake(capture_state="WAITING_FOR_SCAN", capture_mode="block",
                          phase="blocks")
    state = KioskRelay(handle).state({"engaged": True, "finish_blocks_guard": True})
    assert state["screen"] == "10"


def test_command_skip_slide_routes_through_the_thin_verb():
    handle = _RuntimeFake(capture_state="REPOSITION_SLIDE", capture_mode="slide")
    KioskRelay(handle).command("skip_slide")
    assert handle.calls == [("skip_unreadable_slide",)]


# --------------------------------------------------------------------------
# #257: View Results during an open Hybrid slide-capture work order --
# entry pauses automatic capture only; Go Back resumes the same bracket.
# --------------------------------------------------------------------------


def test_command_pause_capture_routes_through_the_thin_verb():
    handle = _RuntimeFake(capture_state="EMPTY", capture_mode="slide")
    KioskRelay(handle).command("pause_capture")
    assert handle.calls == [("pause_capture",)]
    assert handle.capture_paused is True


def test_command_resume_capture_routes_through_the_thin_verb():
    handle = _RuntimeFake(capture_state="EMPTY", capture_mode="slide")
    handle.capture_paused = True
    KioskRelay(handle).command("resume_capture")
    assert handle.calls == [("resume_capture",)]
    assert handle.capture_paused is False


def test_command_pause_then_resume_never_touches_jobs():
    """Non-vacuous #257 assertion: the ONLY calls recorded on the handle for
    the whole pause/view/resume cycle are the two pause/resume verbs
    themselves -- nothing that could cancel, drain, or wait on a background
    job (the fake has no such method at all, so any attempt to call one
    would raise, not pass silently)."""
    handle = _RuntimeFake(capture_state="EMPTY", capture_mode="slide")
    relay = KioskRelay(handle)

    relay.command("pause_capture")
    relay.command("resume_capture")

    assert handle.calls == [("pause_capture",), ("resume_capture",)]


def test_state_projects_capture_paused_flag_from_handle():
    handle = _RuntimeFake(capture_state="EMPTY", capture_mode="slide")
    handle.capture_paused = True

    state = KioskRelay(handle).state({"engaged": True})

    assert state["capture_paused"] is True


def test_state_capture_paused_degrades_false_on_a_bare_handle():
    # A bare handle (_FakeHandle) exposes no capture_paused attribute at all
    # -- same degrading getattr shape as open_retrieval/work_order_open.
    state = KioskRelay(_FakeHandle()).state()

    assert state["capture_paused"] is False


@pytest.mark.parametrize("hybrid_mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_hybrid_open_slide_capture_work_order_reaches_results_table_and_pause_is_visible(
    hybrid_mode,
):
    """#257 core criterion: Results is reachable from an OPEN Hybrid
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


def test_state_remaining_and_pending_block_id_degrade_none_safely():
    # A bare handle: its summary has no missing_slides and it exposes no
    # capture_status pending id -- both new fields must degrade, not raise.
    state = KioskRelay(_FakeHandle()).state()

    assert state["remaining"] == 0
    assert state["pending_block_id"] is None
    assert state["pending_slide_id"] is None


def test_state_defaults_open_retrieval_and_work_order_open_false_on_bare_handle():
    """#155: a bare ``SessionWorkflow``/``_FakeHandle`` test double exposes
    neither attribute -- must degrade to False, not raise, exactly like the
    existing ``capture_status``/``results_status`` getattr seams."""
    state = KioskRelay(_FakeHandle()).state()

    assert state["open_retrieval"] is False
    assert state["work_order_open"] is False
    assert state["has_work_orders"] is False


def test_state_projects_durable_work_order_history_from_handle():
    handle = _FakeHandle()
    handle.open_retrieval = True
    handle.has_work_orders = lambda: True

    state = KioskRelay(handle).state()

    assert state["has_work_orders"] is True
    assert state["screen"] == "between_orders"


def test_state_projects_open_retrieval_and_work_order_open_from_handle_attributes():
    """#155: when the handle (a real ``PiCaptureRuntime``) exposes both as
    plain attributes, the relay forwards them verbatim -- this is the signal
    the router's between_orders/block_scan_work_order gate reads."""
    handle = _FakeHandle()
    handle.open_retrieval = True
    handle.work_order_open = True

    state = KioskRelay(handle).state()

    assert state["open_retrieval"] is True
    assert state["work_order_open"] is True


# --------------------------------------------------------------------------
# session mode (#247): "Kiosk relay state carries the mode as one explicit
# value" -- so the router can branch on it later without new attribute
# lookups. Previously only exercised transitively via PiCaptureRuntime tests
# in test_hybrid_launch.py; these pin the relay-level projection itself.
# --------------------------------------------------------------------------


def test_state_mode_defaults_to_normal_on_a_bare_handle():
    """A bare handle (``_FakeHandle``) has no ``.session_mode`` attribute at
    all -- same degrading getattr shape as ``open_retrieval``/
    ``work_order_open`` above. Must degrade to "normal", not raise or
    leak ``None``."""
    state = KioskRelay(_FakeHandle()).state()

    assert state["mode"] == "normal"
    assert isinstance(state["mode"], str)  # JSON-safe: not a SessionMode enum


# --------------------------------------------------------------------------
# #256: the third, non-interrupting attention-banner layer
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


def test_attention_banner_absent_without_any_error_row():
    handle = _FakeAttentionHandle(rows=())
    assert KioskRelay(handle).state()["attention"] is None


def test_attention_banner_waits_while_another_work_order_actively_captures():
    handle = _FakeAttentionHandle(
        rows=(_error_row(work_order_id=5),),
        work_order_open=True, open_work_order_id=9,
    )
    state = KioskRelay(handle).state()
    assert state["attention"]["can_recapture"] is False


def test_attention_banner_does_not_wait_for_its_own_open_work_order():
    handle = _FakeAttentionHandle(
        rows=(_error_row(work_order_id=5),),
        work_order_open=True, open_work_order_id=5,
    )
    state = KioskRelay(handle).state()
    assert state["attention"]["can_recapture"] is True


@pytest.mark.parametrize("mode", [SessionMode.NORMAL, SessionMode.OPEN_RETRIEVAL])
def test_attention_banner_absent_for_normal_and_open_retrieval(mode):
    """Non-vacuous control (#256): must fail if the explicit mode gate in
    `KioskRelay._attention_banner` is dropped -- an ERROR row is supplied
    here precisely so "no banner" is proven by the gate, not merely by the
    absence of ERROR-shaped data."""
    handle = _FakeAttentionHandle(rows=(_error_row(),), session_mode=mode)
    state = KioskRelay(handle).state()
    assert state["attention"] is None


def test_attention_banner_degrades_to_none_on_a_bare_handle():
    """Blast-radius guard: a handle with no `session_mode`/`results_status`/
    `open_work_order_id` at all must degrade to no banner, never raise."""
    state = KioskRelay(_FakeHandle()).state()
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


def test_state_open_retrieval_stays_false_under_hybrid_and_hybrid_shadow_mode():
    """#247: Hybrid and Hybrid Shadow boot into the exact same capture flow
    as normal verification this slice, so the real ``PiCaptureRuntime`` only
    ever derives ``self.open_retrieval = self.session_mode is
    SessionMode.OPEN_RETRIEVAL`` (tools/run_pi_session.py:179) -- meaning
    open_retrieval is always False for these two modes. The relay's "mode"
    and "open_retrieval" keys are independently-degrading projections
    (``_mode_value`` vs. plain ``getattr(handle, "open_retrieval", False)``
    at kiosk/relay.py:123/133); this pins the relay side of that invariant
    so a future refactor that derives open_retrieval FROM mode inside the
    relay can't silently flip router/screen gating for Hybrid/Hybrid
    Shadow. (open_retrieval is left unset on the handle here -- it degrades
    to False via getattr exactly as it does on the real runtime.)"""
    for mode in (SessionMode.HYBRID, SessionMode.HYBRID_SHADOW):
        handle = _FakeHandle()
        handle.session_mode = mode

        state = KioskRelay(handle).state()

        assert state["mode"] == mode.value
        assert state["open_retrieval"] is False


def test_state_open_bracket_surfaces_effective_engaged_and_routes_to_calibrating():
    # #159: with an open work order but the client latch off, relay.state()
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


def test_tap_scan_block_rides_the_request_id_ledger_once(tmp_path):
    """A dropped response mid-scan must NOT double-apply: the relay's tap goes
    through scan_block, whose request_id lets the server replay, not re-run."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        transport = DropFirstResponseTransport(UrlTransport())
        proxy = RemoteProcessingStore(
            receiver.url, transport=transport, max_attempts=2, backoff=0
        )
        workflow = SessionWorkflow(
            session=session,
            store=proxy,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "framing.json"
            ),
        )
        relay = KioskRelay(workflow)

        outcome = relay.command("scan_block", "51151378")

    assert outcome == ScanOutcome(True, "Accepted block 51151378")
    # First attempt applied server-side, its response was dropped, so _rpc
    # immediately reissued the SAME request_id (identical bytes) -- the server
    # replayed the stored response instead of scanning again.
    assert transport.post_calls >= 2  # the drop forced at least one reissue
    assert transport.sent_payloads[0] == transport.sent_payloads[1]
    # The load-bearing invariant: exactly one block scanned, no double-apply.
    assert store.awaiting_capture_blocks(session.number) == ("51151378",)


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


def test_review_still_jpeg_is_read_only_projection():
    payload = b"\xff\xd8fakejpeg"

    class _StillFake(_RuntimeFake):
        def review_still_jpeg(self):
            return payload

    fake = _StillFake(capture_state="AWAITING_ACCEPT", capture_mode="block")
    assert KioskRelay(fake).review_still_jpeg() == payload


def test_latest_preview_jpeg_is_read_only_projection():
    payload = b"\xff\xd8preview"

    class _PreviewFake(_RuntimeFake):
        def latest_preview_jpeg(self):
            return payload

    fake = _PreviewFake(capture_state="WAITING_FOR_SCAN", capture_mode="block")
    assert KioskRelay(fake).latest_preview_jpeg() == payload


def test_relay_inspection_sheet_reads_from_handle_when_present():
    """#151: expanding a REVIEW row fetches a rendered contact-sheet PNG
    through the same read-only, degrading getattr+callable seam as
    ``review_still_jpeg``/``latest_preview_jpeg``."""
    payload = b"\x89PNGfakepng"

    class _SheetFake(_RuntimeFake):
        def inspection_sheet_bytes(self, path):
            self.calls.append(("inspection_sheet_bytes", path))
            return payload

    fake = _SheetFake(capture_state="AWAITING_ACCEPT", capture_mode="block")
    result = KioskRelay(fake).inspection_sheet_bytes("capture_9__51151378.png")

    assert result == payload
    assert ("inspection_sheet_bytes", "capture_9__51151378.png") in fake.calls


def test_relay_inspection_sheet_returns_none_on_a_bare_handle_without_the_method():
    fake = _RuntimeFake(capture_state="AWAITING_ACCEPT", capture_mode="block")
    assert KioskRelay(fake).inspection_sheet_bytes("missing.png") is None


def test_relay_results_evidence_reads_from_handle_when_present():
    payload = b"\xff\xd8fakejpeg"

    class _EvidenceFake(_RuntimeFake):
        def results_evidence_bytes(self, path):
            self.calls.append(("results_evidence_bytes", path))
            return payload

    fake = _EvidenceFake(capture_state="AWAITING_ACCEPT", capture_mode="block")
    result = KioskRelay(fake).results_evidence_bytes(
        "/sessions/1/claim_artifacts/cap-1_block_thumb.jpg"
    )

    assert result == payload
    assert (
        "results_evidence_bytes",
        "/sessions/1/claim_artifacts/cap-1_block_thumb.jpg",
    ) in fake.calls


def test_relay_results_evidence_returns_none_on_a_bare_handle_without_the_method():
    fake = _RuntimeFake(capture_state="AWAITING_ACCEPT", capture_mode="block")
    assert KioskRelay(fake).results_evidence_bytes("missing.jpg") is None


def test_state_results_rows_carry_evidence_refs_from_handle():
    evidence = {
        "block_thumb": "/artifacts/cap-1_block_thumb.jpg",
        "slide_thumb": "/artifacts/cap-1_slide_thumb.jpg",
        "block_display": "/artifacts/cap-1_block_display.jpg",
        "slide_display": "/artifacts/cap-1_slide_display.jpg",
        "overlay_display": "/artifacts/cap-1_overlay_display.jpg",
    }
    handle = _ResultsFake(
        capture_state="WAITING_FOR_SCAN", capture_mode="block", phase="blocks",
        work_orders=(3,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PASS",
             "claim_reason": "", "claim_score": 0.91, "work_order_id": 3,
             "work_order": "12080", "evidence": evidence},
            {"capture_id": "cap-2", "block_id": "b2", "verdict": "REVIEW",
             "claim_reason": "low", "claim_score": 0.40, "work_order_id": 3,
             "work_order": "12080", "evidence": {
                 **evidence,
                 "overlay_display": "/artifacts/cap-2_overlay_display.jpg",
             }},
        ],
    )

    state = KioskRelay(handle).state({"engaged": True, "view_results_guard": True})

    by_capture = {row["capture_id"]: row for row in state["results_rows"]}
    assert by_capture["cap-1"]["evidence"]["block_thumb"] == evidence["block_thumb"]
    assert by_capture["cap-2"]["evidence"]["overlay_display"] is not None


# --------------------------------------------------------------------------
# results table (#150): degrading projection of results-ready work orders
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
    # #153: relay.state() now hands the screen project_results_table's
    # sorted/colored output (REVIEW-first), not a raw store passthrough.
    assert [row["capture_id"] for row in state["results_rows"]] == ["cap-2", "cap-1"]
    # #232: the human work-order number is the results screen's grouping key --
    # it must survive end to end from results_status() rows into results_rows.
    assert {row["capture_id"]: row["work_order"] for row in state["results_rows"]} == {
        "cap-1": "12080", "cap-2": "12094",
    }


def test_state_results_projection_degrades_none_safely_on_a_bare_handle():
    # A bare SessionWorkflow-shaped handle exposes no results_status() at all.
    state = KioskRelay(_FakeHandle()).state()

    assert state["results_ready_work_orders"] == ()
    assert state["results_rows"] == []


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
    # #232: the SELECT emits the human work-order number (from the slide's
    # "12080_<block>_01_HE" barcode) all the way through to the results screen,
    # where it is the section grouping key. Proven here on a REAL workflow.
    assert {row["work_order"] for row in state["results_rows"]} == {"12080"}


# --------------------------------------------------------------------------
# Hybrid results (#252): Results must be reachable -- and rendered -- while
# scoring is still in flight, not only after the whole work order resolves.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hybrid_mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_relay_state_reaches_results_table_for_hybrid_session_with_only_pending_rows(
    hybrid_mode,
):
    """#252 core criterion: Finish Slides (-> finish_work_order, closing the
    bracket) returns immediately and Results must open right away even
    though nothing has scored yet. Mirrors the closed-bracket state
    ``finish_work_order`` leaves (``work_order_open`` False, ``has_work_
    orders`` True) -- the same shape ``test_hybrid_mode_between_orders_
    escapes_to_results_table`` in test_kiosk_router.py pins at the pure
    router level, proven here end to end through the relay's real
    ``results_status`` -> ``results_ready_work_orders`` projection."""
    handle = _ResultsFake(
        capture_state="EMPTY", capture_mode="slide", phase="slides",
        work_orders=(11,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PENDING",
             "claim_reason": "", "claim_score": None, "work_order_id": 11,
             "work_order": "12080"},
            {"capture_id": "cap-2", "block_id": "b2", "verdict": "PENDING",
             "claim_reason": "", "claim_score": None, "work_order_id": 11,
             "work_order": "12080"},
        ],
    )
    handle.session_mode = hybrid_mode
    handle.open_retrieval = False
    handle.work_order_open = False
    handle.has_work_orders = True

    state = KioskRelay(handle).state({"view_results_guard": True})

    assert state["screen"] == "results_table"
    assert [row["verdict"] for row in state["results_rows"]] == ["PENDING", "PENDING"]


def test_relay_state_renders_pending_and_error_rows_with_their_treatment():
    """Rendered-output requirement: a PENDING row and an ERROR row must each
    actually carry their gray/amber treatment in ``results_rows`` -- not just
    reach the ``results_table`` screen id."""
    handle = _ResultsFake(
        capture_state="EMPTY", capture_mode="slide", phase="slides",
        work_orders=(11,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PENDING",
             "claim_reason": "", "claim_score": None, "work_order_id": 11,
             "work_order": "12080"},
            {"capture_id": "cap-2", "block_id": "b2", "verdict": "ERROR",
             "claim_reason": "artifact write failed", "claim_score": None,
             "work_order_id": 11, "work_order": "12080"},
        ],
    )
    handle.session_mode = SessionMode.HYBRID
    handle.open_retrieval = False
    handle.work_order_open = False
    handle.has_work_orders = True

    state = KioskRelay(handle).state({"view_results_guard": True})

    assert state["screen"] == "results_table"
    by_capture = {row["capture_id"]: row for row in state["results_rows"]}
    assert by_capture["cap-1"]["verdict"] == "PENDING"
    assert by_capture["cap-1"]["color"] == "gray"
    assert by_capture["cap-2"]["verdict"] == "ERROR"
    assert by_capture["cap-2"]["color"] == "amber"
    # ERROR outranks PENDING in project_results_table's stable sort.
    assert [row["capture_id"] for row in state["results_rows"]] == ["cap-2", "cap-1"]


def test_relay_state_resolves_a_pending_row_to_pass_in_place_same_capture_id():
    """A row PENDING in one poll and PASS in the next must resolve in place
    under the SAME capture id, not disappear/reappear as a new row -- proving
    the projection is keyed on capture_id, not on verdict."""
    handle = _ResultsFake(
        capture_state="EMPTY", capture_mode="slide", phase="slides",
        work_orders=(11,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PENDING",
             "claim_reason": "", "claim_score": None, "work_order_id": 11,
             "work_order": "12080"},
        ],
    )
    handle.session_mode = SessionMode.HYBRID
    handle.open_retrieval = False
    handle.work_order_open = False
    handle.has_work_orders = True

    before = KioskRelay(handle).state({"view_results_guard": True})
    assert before["screen"] == "results_table"
    assert before["results_rows"][0]["capture_id"] == "cap-1"
    assert before["results_rows"][0]["verdict"] == "PENDING"
    assert before["results_rows"][0]["color"] == "gray"

    # The store's live projection moves on: same capture id, resolved verdict.
    handle._rows = [
        {"capture_id": "cap-1", "block_id": "b1", "verdict": "PASS",
         "claim_reason": "", "claim_score": 0.92, "work_order_id": 11,
         "work_order": "12080"},
    ]

    after = KioskRelay(handle).state({"view_results_guard": True})
    assert after["screen"] == "results_table"
    assert after["results_rows"][0]["capture_id"] == "cap-1"
    assert after["results_rows"][0]["verdict"] == "PASS"
    assert after["results_rows"][0]["color"] == "green"


class _RealResultsStatusHandle:
    """Wraps a REAL ``SessionWorkflow.results_status()`` -- exercising its
    actual #252 mode-routing/degrade logic against a fake store -- behind the
    minimal duck-typed surface ``KioskRelay.state()`` needs for everything
    else (``snapshot``/``summarize``/``events``/``session_mode``). Building a
    full real workflow with a real outbox/transport/camera just to prove the
    poll survives a raising store would be scope creep unrelated to what this
    test is about."""

    def __init__(self, workflow, *, session_mode):
        self._workflow = workflow
        self.session_mode = session_mode

    def snapshot(self):
        return _Snapshot("slides", None)

    def summarize(self):
        return _Summary(0)

    def events(self):
        return ()

    def results_status(self):
        return self._workflow.results_status()


@pytest.mark.parametrize("hybrid_mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_relay_state_survives_list_hybrid_results_raising_on_a_real_workflow(
    tmp_path, hybrid_mode,
):
    """Blast-radius guard, end to end: a REAL ``SessionWorkflow.results_
    status()`` (not a duck-typed fake of the method itself) whose store's
    ``list_hybrid_results`` raises must still let ``KioskRelay.state()``
    complete -- ``relay.state()`` has no try/except of its own around
    ``results_status()``, so the degrade must happen entirely inside
    ``SessionWorkflow.results_status``, exactly where the kiosk's poll loop
    calls it."""
    store = _FakeResultsRowsStore(hybrid_error=RuntimeError("boom"))
    workflow = _results_workflow(tmp_path, session_mode=hybrid_mode, store=store)
    handle = _RealResultsStatusHandle(workflow, session_mode=hybrid_mode)

    state = KioskRelay(handle).state({"view_results_guard": True})

    assert state["online"] is True
    assert state["results_rows"] == []
    assert state["results_ready_work_orders"] == ()
    assert store.hybrid_calls == 1


# --------------------------------------------------------------------------
# #258: --profile-only Hybrid queue/timing display. `KioskRelay._profile_
# fields` gates purely on `getattr(handle, "profile", False)` -- an
# independent, explicit second gate on top of `hybrid_profile_status`'s own
# session-mode gate (pinned separately in test_session_workflow.py).
# --------------------------------------------------------------------------


class _ProfileFake(_RuntimeFake):
    """Duck-types just the two signals ``KioskRelay._profile_fields`` reads
    (``.profile`` and ``hybrid_profile_status``) on top of ``_RuntimeFake``'s
    snapshot/summarize/events/capture_status shape."""

    def __init__(self, *, profile, rows=(), queue_count=0, raises=False, **kwargs):
        super().__init__(**kwargs)
        self.profile = profile
        self._rows = rows
        self._queue_count = queue_count
        self._raises = raises

    def hybrid_profile_status(self, *, now_ns):
        if self._raises:
            raise RuntimeError("boom")
        return {"queue_count": self._queue_count, "rows": self._rows}


def _pending_row(
    *, capture_id: str = "cap-1", block_id: str = "11111111",
    stage: str = "heuristic_selection", elapsed_ms: int = 1500,
) -> ProfileRow:
    return ProfileRow(
        capture_id=capture_id, block_id=block_id, state="PENDING",
        stage=stage, elapsed_ms=elapsed_ms, total_ms=None,
        stage_ms={}, shadow=False,
    )


def _finished_row(
    *, capture_id: str = "cap-2", block_id: str = "22222222",
    state: str = "PASS", total_ms: int = 4200, shadow: bool = False,
) -> ProfileRow:
    return ProfileRow(
        capture_id=capture_id, block_id=block_id, state=state,
        stage=None, elapsed_ms=None, total_ms=total_ms,
        stage_ms={
            "queue_wait": 100, "preparation": 200,
            "heuristic_selection": 300, "accurate_scoring": 3500,
            "artifact_write": 100,
        },
        shadow=shadow,
    )


def test_state_profile_shows_queue_count_pending_stage_and_elapsed():
    """#258 criterion 1: --profile in Hybrid shows the queue count, a
    pending row's current stage, and its elapsed time -- asserted on the
    RENDERED ``state["profile"]`` content, not just a boolean flag."""
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=True, rows=(_pending_row(),), queue_count=1,
    )
    handle.session_mode = SessionMode.HYBRID

    state = KioskRelay(handle).state({"engaged": True})

    assert state["profile"]["queue_count"] == 1
    row = state["profile"]["rows"][0]
    assert row["state"] == "PENDING"
    assert row["stage"] == "heuristic_selection"
    assert row["elapsed_ms"] == 1500


def test_state_profile_finished_row_total_and_breakdown_is_expandable():
    """#258 criterion 2/3: a finished row's total time plus a breakdown
    covering all five profiled stages -- the data an expandable UI element
    reveals. Folded into the existing rendered-row payload (no new screen:
    see report) rather than a second endpoint."""
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=True, rows=(_finished_row(),), queue_count=0,
    )
    handle.session_mode = SessionMode.HYBRID

    state = KioskRelay(handle).state({"engaged": True})

    row = state["profile"]["rows"][0]
    assert row["total_ms"] == 4200
    assert set(row["stage_breakdown"]) == set(PROFILE_STAGE_ORDER)
    assert row["stage_breakdown"]["accurate_scoring"] == 3500


def test_state_omits_the_profile_key_entirely_without_the_profile_flag():
    """#258 hard constraint: without --profile, no queue count, stage, or
    timing field appears ANYWHERE in the rendered operator view. Checked
    across the WHOLE serialized state, not just `state["profile"]` being
    absent, so a leak through any other key would still be caught."""
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=False, rows=(_pending_row(), _finished_row()), queue_count=1,
    )
    handle.session_mode = SessionMode.HYBRID

    state = KioskRelay(handle).state({"engaged": True})

    assert "profile" not in state
    import json

    rendered = json.dumps(state)
    for leaked in ("queue_count", "elapsed_ms", "stage_breakdown", "heuristic_selection"):
        assert leaked not in rendered


def test_state_profile_works_with_no_debug_flag_set():
    """#258 constraint: --profile must work independent of any visual-debug
    surface. Non-vacuous: this fake sets no debug-shaped attribute at all
    (there is none on the real handle either), so if profile display were
    ever made to also require one, `getattr(handle, "debug", False)` would
    default False and this assertion would fail."""
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=True, rows=(_pending_row(),), queue_count=1,
    )
    handle.session_mode = SessionMode.HYBRID
    assert not hasattr(handle, "debug")

    state = KioskRelay(handle).state({"engaged": True})

    assert "profile" in state


def test_state_profile_shadow_row_is_visibly_tagged():
    """#258: Hybrid Shadow's complete-pool cost must be distinguishable from
    real Hybrid timing on screen."""
    shadow_row = _finished_row(capture_id="cap-9", block_id="99999999", shadow=True)
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=True, rows=(shadow_row,), queue_count=0,
    )
    handle.session_mode = SessionMode.HYBRID_SHADOW

    state = KioskRelay(handle).state({"engaged": True})

    row = state["profile"]["rows"][0]
    assert row["shadow"] is True
    assert "Hybrid Shadow" in row["shadow_note"]


def test_state_profile_degrades_to_absent_when_hybrid_profile_status_raises():
    """Blast-radius guard (#258): a raising ``hybrid_profile_status`` must
    not reach ``_camera_loop``'s bare ``except Exception`` -- the poll
    survives, degrading to no profile fields at all."""
    handle = _ProfileFake(
        capture_state="EMPTY", capture_mode="slide",
        profile=True, raises=True,
    )
    handle.session_mode = SessionMode.HYBRID

    state = KioskRelay(handle).state({"engaged": True})

    assert state["online"] is True
    assert "profile" not in state


def test_relay_state_normal_mode_results_gate_is_unchanged():
    """Regression control (#252): a NORMAL-mode session's results gate must
    route byte-identically to before -- results_ready_work_orders sourced
    from list_results_ready_work_orders, never list_hybrid_results."""
    handle = _ResultsFake(
        capture_state="WAITING_FOR_SCAN", capture_mode="block", phase="blocks",
        work_orders=(3,),
        rows=[
            {"capture_id": "cap-1", "block_id": "b1", "verdict": "PASS",
             "claim_reason": "", "claim_score": 0.91, "work_order_id": 3,
             "work_order": "12080"},
        ],
    )
    handle.session_mode = SessionMode.NORMAL

    state = KioskRelay(handle).state({"engaged": True, "view_results_guard": True})

    assert state["mode"] == "normal"
    assert state["screen"] == "results_table"
    assert [row["capture_id"] for row in state["results_rows"]] == ["cap-1"]
