"""The kiosk read/write core -- HTTP-free and thread-aware (#121 / 119a).

``KioskRelay`` is the non-owning seam between the touchscreen and the live
session. It holds the SAME handle the console uses (in production a
``PiCaptureRuntime``; in tests a bare ``SessionWorkflow``) and never constructs
one, so there is no second source of truth.

Two seams, deliberately asymmetric on locking:

* **Read** (:meth:`state`) projects the workflow's pure observe API
  (``snapshot``/``summarize``/``events``) into JSON-safe values. It is
  lock-free on purpose: these are independent reads, and holding the write
  lock across a network RPC would stall taps. A dropped store link surfaces as
  ``online: False`` rather than an exception -- the seam 119c's banner builds on.
* **Write** (:meth:`command`) routes a tap through the existing
  ``session_console.dispatch`` verbs under the relay's OWN lock, serializing
  concurrent browser taps. It must not share the runtime's lock: camera verbs
  (e.g. ``scan_block``) re-acquire that non-reentrant lock inside the call, so
  sharing it would self-deadlock. Idempotency for retried taps is already
  provided by the ADR-0002 ``request_id`` ledger underneath ``scan_block``.
"""
from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, ContextManager, Mapping, cast

from kiosk.attention import project_attention_banner
from kiosk.inspection import InspectionDescriptor, project_inspection
from kiosk.results_table import project_results_table
from kiosk.router import effective_engaged, select_screen
from store.remote import TransportError
from session.console import dispatch
from session.profile_report import profile_screen_fields
from session.session_mode import SessionMode

log = logging.getLogger(__name__)

_VERDICT_KINDS = ("claim_pass", "claim_review")


class KioskRelay:
    """Second, non-owning renderer of a live ``SessionWorkflow`` handle."""

    def __init__(self, handle: Any, *, lock: ContextManager | None = None):
        # ``handle`` is the shared runtime/workflow -- the relay never makes one.
        self._handle = handle
        self._lock: ContextManager = lock if lock is not None else Lock()

    # -- read path ---------------------------------------------------------
    def state(self, ui: Mapping | None = None) -> dict[str, Any]:
        """A JSON-safe projection of the live session, including the routed
        screen id (:func:`kiosk.router.select_screen`).

        ``ui`` carries the client-owned flags routing needs but the workflow
        cannot know: ``engaged`` (boot latch) and the two finish-guard flags.
        Never raises on a dropped processing-computer link: returns
        ``{"online": False, ...}`` so the screen can keep rendering.
        """
        ui = ui or {}
        try:
            snapshot = self._handle.snapshot()
            summary = self._handle.summarize()
            events = self._handle.events()
        except TransportError:
            return {"online": False}

        capture = self._capture_status()
        pending_capture = self._pending_capture_status()
        results = self._results_status()
        latest_verdict, current_capture_id = _scan_events(events)
        last = events[-1] if events else None
        results_ready_work_orders = tuple(results.get("work_orders") or ()) if results else ()
        projected_rows = project_results_table(results.get("rows") or []) if results else []
        work_order_open = bool(getattr(self._handle, "work_order_open", False))
        state = {
            "online": True,
            "session_number": snapshot.session_number,
            "session_present": True,  # a session is always resumed in-process
            "phase": snapshot.phase,
            "resumable": snapshot.phase != "finalized",
            "started_at": summary.started_at.isoformat(),
            "latest_block_id": snapshot.latest_block_id,
            "latest_block_status": snapshot.latest_block_status,
            # scanned-but-not-yet-captured block id (screens 07/18). Forwarded
            # from capture_status (dropped pre-119c); None on a bare workflow.
            "pending_block_id": capture.get("pending_block_id") if capture else None,
            # scanned slide id for the screen 12 positive scan indicator
            # ("SCANNED: <id> - place slide"); None until a slide is scanned.
            "pending_slide_id": capture.get("pending_slide_id") if capture else None,
            # "N captured" status-bar signal on screen 06.
            "captured": summary.sets_processed,
            # #188: any block ever captured this session, regardless of verdict
            # status -- the boot chooser's "is there real work to resume" signal.
            # getattr keeps it None-safe for bare-summary test fakes (degradation
            # invariant, same pattern as missing_slides/remaining below).
            "blocks_captured": getattr(summary, "blocks_captured", 0),
            # slides ready-and-awaiting capture (screen 12 "N remaining").
            # missing_slides = blocks preprocessed but not yet verdicted; the
            # locked WS-C decision fixes remaining = len(missing_slides). getattr
            # keeps it None-safe for bare-summary test fakes (degradation invariant).
            "remaining": len(getattr(summary, "missing_slides", ()) or ()),
            "pass_count": summary.pass_count,
            "review_count": summary.review_count,
            "event_count": len(events),
            "last_event": _event_dict(last) if last is not None else None,
            # #250 review F1: a dedicated, renderable field for "why did
            # Finish Blocks bounce back to block capture" -- deliberately NOT
            # overloading `last_event` (that field is consumed purely as a
            # dedupe key for the duplicate-scan flash today, never rendered
            # as text). Surfaces only while it IS the most recent event, so
            # any later event (e.g. a fresh block scan) naturally clears it.
            "hybrid_pool_bounce_message": (
                last.message if last is not None
                and last.kind == "hybrid_pool_freeze_insufficient_blocks"
                else None
            ),
            # fine-grained capture-layer signals (None on a bare workflow)
            "capture_state": capture["capture_state"] if capture else None,
            "capture_mode": capture["capture_mode"] if capture else None,
            "latest_verdict": latest_verdict,
            "current_capture_id": current_capture_id,
            "pending_capture_id": (
                pending_capture.get("capture_id") if pending_capture else None
            ),
            # #257: server-side pause state (View Results while a
            # slide-capture work order is open). Read-only projection, same
            # degrading getattr shape as open_retrieval/work_order_open --
            # never a router input, just visibility for the operator/tests.
            "capture_paused": bool(getattr(self._handle, "capture_paused", False)),
            # client-owned UI flags, echoed so routing stays a pure function
            "engaged": bool(ui.get("engaged", False)),
            "finish_blocks_guard": bool(ui.get("finish_blocks_guard", False)),
            "finish_slides_guard": bool(ui.get("finish_slides_guard", False)),
            # Navigation is operator-owned: ready data enables the destination,
            # but results open only after VIEW RESULTS is tapped.
            "view_results_guard": bool(ui.get("view_results_guard", False)),
            # #256: client-owned nav flag for the attention correction-flow
            # screen, same guard+data pattern as finish_blocks_guard/
            # view_results_guard above.
            "recapture_guard": bool(ui.get("recapture_guard", False)),
            # results table (#150/#252): every results-ready work order's
            # per-slide verdict rows this session (NORMAL/OPEN_RETRIEVAL), or
            # every Hybrid/Hybrid Shadow slide capture across every work
            # order, scored or not (`SessionWorkflow.results_status` picks
            # the source by session mode). None on a bare workflow -> () / [].
            "results_ready_work_orders": results_ready_work_orders,
            "results_rows": projected_rows,
            # #155: plain attribute reads (not the getattr+callable method
            # shape above) -- PiCaptureRuntime exposes these as booleans, not
            # callables. Degrades to False on a bare SessionWorkflow handle.
            "open_retrieval": bool(getattr(self._handle, "open_retrieval", False)),
            "work_order_open": work_order_open,
            "has_work_orders": self._has_work_orders(),
            # #256: the third, non-interrupting attention layer -- a pure
            # projection over the SAME rows already fetched above, degrading
            # to None (no banner) rather than raising (see
            # `_attention_banner`'s own docstring for the blast-radius
            # guard). None for NORMAL/OPEN_RETRIEVAL and whenever no Hybrid
            # slide currently carries an ERROR verdict.
            "attention": self._attention_banner(
                projected_rows, work_order_open=work_order_open,
            ),
            # #247: the single explicit session scoring mode, JSON-safe as
            # its string value ("normal"/"open_retrieval"/"hybrid"/
            # "hybrid_shadow"). Degrades to "normal" on a bare workflow
            # handle (no `.session_mode` attribute), same getattr-default
            # shape as `open_retrieval` above. The router does not branch on
            # this key yet -- it is exposed now so it can without new
            # attribute lookups once Hybrid behavior lands.
            "mode": self._mode_value(),
        }
        state["effective_engaged"] = effective_engaged(state)
        state["screen"] = select_screen(state)
        # #258: the queue-count/stage/timing affordance is added to `state`
        # ONLY while `--profile` is on -- never present as an always-there
        # key degrading to None/False like everything above. That is the
        # enforcement for "without --profile, no queue count, stage, or
        # timing field appears anywhere in the operator view": the key is
        # simply absent from the JSON the touchscreen renders, not merely
        # null.
        profile_fields = self._profile_fields()
        if profile_fields is not None:
            state["profile"] = profile_fields
        return state

    def _capture_status(self) -> dict[str, str] | None:
        """Read the capture-layer state/mode if the handle exposes it (the Pi
        runtime does; a bare ``SessionWorkflow`` in tests does not)."""
        reader = getattr(self._handle, "capture_status", None)
        return cast("dict[str, str]", reader()) if callable(reader) else None

    def _pending_capture_status(self) -> dict[str, str] | None:
        """Read-only projection of the held still for the review gate."""
        reader = getattr(self._handle, "pending_capture_status", None)
        if not callable(reader):
            return None
        status = reader()
        return cast("dict[str, str]", status) if status else None

    def _results_status(self) -> dict[str, Any] | None:
        """Read the results-ready work orders + verdict rows if the handle
        exposes them (same degrading getattr+callable shape as
        ``capture_status``); ``None`` on a bare ``SessionWorkflow`` handle."""
        reader = getattr(self._handle, "results_status", None)
        return cast("dict[str, Any]", reader()) if callable(reader) else None

    def _has_work_orders(self) -> bool:
        """Read durable work-order history; False on older/bare handles."""
        reader = getattr(self._handle, "has_work_orders", None)
        return bool(reader()) if callable(reader) else bool(reader)

    def _open_work_order_id(self) -> int | None:
        """The currently open work order's id, or None; degrades to None on
        a handle without the method (mirrors ``_has_work_orders``), or on
        one whose method returns something other than an int."""
        reader = getattr(self._handle, "open_work_order_id", None)
        if not callable(reader):
            return None
        value = reader()
        return value if isinstance(value, int) else None

    def _attention_banner(
        self, rows: list[dict[str, Any]], *, work_order_open: bool,
    ) -> dict[str, Any] | None:
        """#256: the third, non-interrupting attention layer.

        HYBRID/HYBRID_SHADOW only -- an explicit second gate, not merely an
        accident of what data happens to exist (NORMAL/OPEN_RETRIEVAL rows
        never carry an ERROR verdict either way). Wrapped in its own
        try/except so a defect here degrades to "no banner" rather than
        reaching `_camera_loop`'s bare `except Exception` and killing the
        camera loop -- the same blast-radius contract `list_hybrid_results`
        and this class's own `TransportError` guard already keep.
        """
        if self._mode_value() not in (
            SessionMode.HYBRID.value, SessionMode.HYBRID_SHADOW.value,
        ):
            return None
        try:
            return project_attention_banner(
                rows, work_order_open=work_order_open,
                open_work_order_id=self._open_work_order_id(),
            )
        except Exception:
            log.warning(
                "KioskRelay._attention_banner raised; degrading to no banner",
                exc_info=True,
            )
            return None

    def _profile_fields(self) -> dict[str, Any] | None:
        """#258: the touchscreen half of the shared ``--profile`` Hybrid
        queue/timing display -- ``None`` (never rendered) unless the LAUNCH
        flag itself is on.

        Checks ``getattr(self._handle, "profile", False)`` directly, rather
        than trusting ``hybrid_profile_status`` (or the store's own
        ``profile_enabled`` row filter underneath it) to keep the affordance
        hidden by itself -- ``.profile`` is a plain attribute
        ``RunPiSession`` sets from its own ``--profile`` CLI flag, degrading
        to ``False`` on a bare ``SessionWorkflow`` test handle exactly like
        ``open_retrieval``/``work_order_open`` above. This is the SAME
        pattern ``format_profile_console`` (console) reads via its own
        caller's ``self.profile`` check -- two independent readers of one
        boolean, not two decisions.

        Wrapped in its own try/except (same shape as ``_attention_banner``):
        this runs on the kiosk poll path, so any unexpected failure
        (including one ``hybrid_profile_status`` itself failed to degrade)
        must still leave the poll -- and the camera loop underneath it --
        alive rather than raising up to ``_camera_loop``'s bare
        ``except Exception``.
        """
        if not bool(getattr(self._handle, "profile", False)):
            return None
        reader = getattr(self._handle, "hybrid_profile_status", None)
        if not callable(reader):
            return None
        try:
            status = reader(now_ns=time.time_ns())
            return profile_screen_fields(
                status["rows"], queue_count=status["queue_count"]
            )
        except Exception:
            log.warning(
                "KioskRelay._profile_fields raised; degrading to no profile "
                "fields",
                exc_info=True,
            )
            return None

    def _mode_value(self) -> str:
        """JSON-safe session-mode string; degrades to "normal" when the
        handle has no `.session_mode` attribute (bare `SessionWorkflow` test
        doubles, or an older handle predating #247)."""
        mode = getattr(self._handle, "session_mode", None)
        return mode.value if isinstance(mode, SessionMode) else SessionMode.NORMAL.value

    def review_still_jpeg(self) -> bytes | None:
        """Read-only JPEG bytes for ``GET /review-still`` (None when none)."""
        reader = getattr(self._handle, "review_still_jpeg", None)
        return reader() if callable(reader) else None

    def latest_preview_jpeg(self) -> bytes | None:
        """Read-only JPEG bytes for ``GET /preview-frame`` (None when none)."""
        reader = getattr(self._handle, "latest_preview_jpeg", None)
        return reader() if callable(reader) else None

    def inspection_descriptors(self, capture_id: str) -> list[InspectionDescriptor] | None:
        """Ordered contact-sheet descriptors (#153) for one results-ready
        slide, via ``kiosk.inspection.project_inspection``. Looks the row up
        by ``capture_id`` from the same live ``results_status()`` rows the
        results table reads -- so the client never has to replicate the
        ``{contact_sheet_dir}/{capture_id}__{block_id}.png`` path convention
        itself, only fetch each returned descriptor's ``path`` via
        ``GET /inspection-sheet``. ``None`` when the handle has no results
        data, or no row matches ``capture_id``."""
        results = self._results_status()
        if not results:
            return None
        row = next(
            (r for r in results.get("rows") or [] if r.get("capture_id") == capture_id),
            None,
        )
        if row is None:
            return None
        return project_inspection(row)

    def inspection_sheet_bytes(self, path: str) -> bytes | None:
        """Read-only contact-sheet PNG bytes for ``GET /inspection-sheet``
        (#151): same degrading getattr+callable seam as
        ``review_still_jpeg``/``latest_preview_jpeg``. None on a bare handle
        without the method, or when the handle has nothing at ``path``."""
        reader = getattr(self._handle, "inspection_sheet_bytes", None)
        return reader(path) if callable(reader) else None

    def results_evidence_bytes(self, path: str) -> bytes | None:
        """Read-only claim-artifact JPEG bytes for ``GET /results-evidence``
        (#236): same degrading getattr+callable seam as
        ``inspection_sheet_bytes``."""
        reader = getattr(self._handle, "results_evidence_bytes", None)
        return reader(path) if callable(reader) else None

    # -- write path --------------------------------------------------------
    def command(self, name: str, *args: str) -> object:
        """Serialize one tap onto the relay lock and route it through
        ``dispatch`` -- the same command surface the console uses. Unknown
        verbs raise ``ValueError`` from ``dispatch`` (nothing is mutated)."""
        with self._lock:
            return dispatch(self._handle, name, *args)


def _scan_events(events) -> tuple[dict[str, Any] | None, str | None]:
    """From the cumulative event tail, derive the latest slide verdict and the
    id of the capture currently on the platen.

    ``current_capture_id`` = the newest event carrying a capture id (a fresh
    slide capture advances it), so a stale prior-slide PASS/REVIEW can never
    id-match the new slide and bleed its colour onto it (router R13-R15).
    """
    latest_verdict: dict[str, Any] | None = None
    current_capture_id: str | None = None
    for event in events:
        capture_id = getattr(event, "capture_id", None)
        if capture_id:
            current_capture_id = capture_id
        if event.kind in _VERDICT_KINDS:
            latest_verdict = {
                "kind": event.kind,
                "capture_id": capture_id,
                "block_id": getattr(event, "block_id", None),
                "reason": event.message,
            }
    return latest_verdict, current_capture_id


def _event_dict(event: Any) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "phase": event.phase,
        "message": event.message,
        "block_id": getattr(event, "block_id", None),
    }
