"""TDD coverage for the #256 follow-up gap: the operator route that ARMS a
running Hybrid session so the NEXT physically captured slide routes through
`ProcessingStore.recapture_hybrid_slide` (supersession) instead of the
ordinary `record_slide_capture` path.

`tests/test_hybrid_slide_attention_recapture.py` already proves
`recapture_hybrid_slide`'s own CAS/identity-match contract directly against
the store; this file does NOT re-test that. It proves the piece that did not
exist before: a real `SessionWorkflow` (the SAME `capture_slide` entry point
`tools/run_pi_session.py::PiCaptureRuntime._consume_capture` calls for every
physically captured slide) can be armed for one specific attention item, and
that arming actually changes which store method the next capture reaches --
end to end through the real `PiOutbox` publish/replay durability layer, not
a mocked-out stand-in for it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from session.console import COMMAND_HELP, _COMMANDS, dispatch
from session.processing_store import ProcessingStore
from session.session_mode import SessionMode
from session.workflow import PiOutbox, SessionWorkflow
from tests.test_hybrid_slide_queue import (
    _freeze_hybrid_session,
    _make_store,
    _valid_slide_result,
    _write_slide_png,
)
from tests.test_session_workflow import FakePhaseCamera, ToggleTransport

STARTED_AT = datetime(2026, 7, 29, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """Mirrors tests/test_hybrid_slide_queue.py's fixture of the same name:
    the real QC panel renderer assumes the mask matches the capture's
    dimensions, which these tests' small synthetic masks deliberately do
    not. Autouse fixtures do not cross module boundaries, so this file
    needs its own copy."""

    def write_qc(capture, mask, destination):
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


def _hybrid_workflow(tmp_path, store, session_number) -> SessionWorkflow:
    """A real SessionWorkflow wired to the SAME production PiOutbox/store
    seam `PiCaptureRuntime` uses, reattached (not freshly started) to a
    Hybrid session `_freeze_hybrid_session` already drove into 'slides'
    phase at the store layer."""
    workflow = SessionWorkflow(
        session=store.resume_session(session_number),
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        session_mode=SessionMode.HYBRID,
    )
    workflow.prepare_empty_backlight("slide")
    return workflow


def _seed_error_row(tmp_path, store, session_number, block_id, *, value) -> str:
    """Simulate a prior Hybrid Processing Error the same way
    test_hybrid_slide_attention_recapture.py does: a durable capture forced
    into job_state='error' without ever completing a real scoring pass."""
    capture_id = store.record_slide_capture(
        session_number,
        _write_slide_png(tmp_path / f"slide_error_{value}.png", value=value),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id),
        duration_ms=5.0, start_job=False,
    )
    store._set_slide_job_state(capture_id, "error")
    return capture_id


# ---------------------------------------------------------------------------
# 1. Arming is gated on the SAME attention-banner eligibility policy, called
#    (not duplicated).
# ---------------------------------------------------------------------------


def test_arm_rejects_a_capture_id_that_is_not_the_current_attention_item(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    workflow = _hybrid_workflow(tmp_path, store, session_number)

    with pytest.raises(ValueError):
        workflow.arm_hybrid_recapture("does-not-exist")
    assert workflow._pending_recapture_capture_id is None


def test_arm_succeeds_for_the_current_attention_item(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)

    workflow.arm_hybrid_recapture(old_capture_id)

    assert workflow._pending_recapture_capture_id == old_capture_id


# ---------------------------------------------------------------------------
# 2. The core gap: an armed capture_slide call routes through
#    recapture_hybrid_slide (matching identity -> supersession) instead of
#    record_slide_capture.
# ---------------------------------------------------------------------------


def test_armed_capture_slide_supersedes_the_error_row(tmp_path):
    store = _make_store(tmp_path)
    session_number, work_order_id = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)
    workflow.arm_hybrid_recapture(old_capture_id)

    result = workflow.capture_slide(
        _write_slide_png(tmp_path / "recapture.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        scanned_payload="12080_11111111_01_HE",
    )
    assert result.success is True
    # The arm is consumed immediately by this one capture, regardless of
    # whether the eventual store outcome is a match.
    assert workflow._pending_recapture_capture_id is None

    store.wait_for_jobs()
    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "superseded"

    rows = store.slide_captures(session_number)
    assert len(rows) == 2
    new_row = next(row for row in rows if row["capture_id"] != old_capture_id)
    assert new_row["job_state"] == "complete"
    assert new_row["work_order_id"] == work_order_id


# ---------------------------------------------------------------------------
# 3. Identity mismatch: must NOT supersede, must NOT be silently dropped
#    (a durable event records the rejection), and the arm still clears.
# ---------------------------------------------------------------------------


def test_armed_capture_slide_with_mismatched_identity_does_not_supersede(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)
    workflow.arm_hybrid_recapture(old_capture_id)
    before_rows = len(store.slide_captures(session_number))

    result = workflow.capture_slide(
        _write_slide_png(tmp_path / "mismatch.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        scanned_payload="12080_22222222_01_HE",  # DIFFERENT block than armed
    )

    assert result.success is True
    assert result.block_id == "22222222"
    # Cleared even though the store rejected the recapture -- "clears after
    # one use, success or identity mismatch" (never stuck armed).
    assert workflow._pending_recapture_capture_id is None

    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "error"  # untouched, never superseded
    # No new row either (mirrors the store-level contract already proven in
    # test_hybrid_slide_attention_recapture.py): nothing is silently
    # half-applied.
    assert len(store.slide_captures(session_number)) == before_rows

    kinds = [event.kind for event in workflow.events()]
    assert "hybrid_recapture_rejected" in kinds


# ---------------------------------------------------------------------------
# 4. Arm clears after ONE use: a later, unrelated slide is never hijacked
#    into a stale recapture.
# ---------------------------------------------------------------------------


def test_arm_clears_after_one_capture_so_a_later_slide_is_ordinary(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)
    workflow.arm_hybrid_recapture(old_capture_id)

    # First capture consumes the arm (matching identity -> supersedes).
    workflow.capture_slide(
        _write_slide_png(tmp_path / "first.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        scanned_payload="12080_11111111_01_HE",
    )
    store.wait_for_jobs()
    rows_after_first = {row["capture_id"] for row in store.slide_captures(session_number)}

    # A second, unrelated slide (a different block) must go through the
    # ORDINARY path -- never re-consumed as another recapture.
    workflow.capture_slide(
        _write_slide_png(tmp_path / "second.png", value=140),
        captured_at=STARTED_AT + timedelta(seconds=10),
        scanned_payload="12080_22222222_01_HE",
    )
    store.wait_for_jobs()

    rows = store.slide_captures(session_number)
    new_ids = {row["capture_id"] for row in rows} - rows_after_first
    assert len(new_ids) == 1
    (new_capture_id,) = new_ids
    new_row = store.get_slide_capture(session_number, new_capture_id)
    assert new_row["job_state"] == "complete"
    assert new_row["block_id"] == "22222222"


# ---------------------------------------------------------------------------
# 5. Disarm/cancel path.
# ---------------------------------------------------------------------------


def test_disarm_clears_the_pending_arm(tmp_path):
    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)
    workflow.arm_hybrid_recapture(old_capture_id)

    workflow.disarm_hybrid_recapture()
    assert workflow._pending_recapture_capture_id is None

    # The next capture for the SAME claimed block now goes through the
    # ORDINARY path -- proving disarm took effect, not merely resetting a
    # value nothing reads: if it had NOT taken effect, this capture would
    # still supersede old_capture_id.
    workflow.capture_slide(
        _write_slide_png(tmp_path / "after_disarm.png", value=90),
        captured_at=STARTED_AT + timedelta(seconds=5),
        scanned_payload="12080_11111111_01_HE",
    )
    store.wait_for_jobs()

    old_row = store.get_slide_capture(session_number, old_capture_id)
    assert old_row["job_state"] == "error"  # never superseded
    assert len(store.slide_captures(session_number)) == 2  # old row + one new row


# ---------------------------------------------------------------------------
# 6. Operator route: arm/disarm are reachable through the SAME shared
#    console/kiosk dispatch table every other verb uses, not a side channel.
# ---------------------------------------------------------------------------


def test_arm_and_disarm_are_wired_into_the_shared_command_table(tmp_path):
    assert "arm_hybrid_recapture" in _COMMANDS
    assert "disarm_hybrid_recapture" in _COMMANDS
    assert set(COMMAND_HELP) == set(_COMMANDS)

    store = _make_store(tmp_path)
    session_number, _ = _freeze_hybrid_session(
        store, "hybrid", ("11111111", "22222222"),
    )
    old_capture_id = _seed_error_row(tmp_path, store, session_number, "11111111", value=50)
    workflow = _hybrid_workflow(tmp_path, store, session_number)

    dispatch(workflow, "arm_hybrid_recapture", old_capture_id)
    assert workflow._pending_recapture_capture_id == old_capture_id

    dispatch(workflow, "disarm_hybrid_recapture")
    assert workflow._pending_recapture_capture_id is None
