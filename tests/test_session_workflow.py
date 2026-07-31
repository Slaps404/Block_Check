"""Highest-level acceptance test for the fake-hardware session workflow,
covering startup through summary and explicit finalization (#93-#100)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from threading import Event
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import contact_sheet
import numpy as np
import pytest

from camera_calibration import (
    ActivatedCameraMode,
    CalibrationQuality,
    LockedCameraControls,
    PhaseCameraCalibration,
)
import session.workflow as session_workflow_module

from capture_runtime import CaptureController
from capture_session import CaptureSession, CaptureState, SessionConfig
from capture_storage import CaptureRecord, CaptureStore
from session.pipeline import decide_claim
from session.preparation import PreparationFailure, PreparedSpecimen
from session.session_mode import SessionMode
from session.workflow_types import SessionIdentity
from slide.qr import DecodeCandidate, select_slide_identity
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
    UploadReceipt,
    default_work_order_scorer,
)


STARTED_AT = datetime(2026, 7, 2, 18, 5, 6, tzinfo=timezone.utc)


def test_restore_pending_block_rearms_block_capture_but_rejects_slide_mode():
    block_session = CaptureSession(mode="block")
    block_session.restore_pending_block("51151378")

    assert block_session.pending_block_id == "51151378"
    assert block_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION

    slide_session = CaptureSession(mode="slide")
    with pytest.raises(RuntimeError, match="block mode"):
        slide_session.restore_pending_block("51151378")


class FakePhaseCamera:
    def __init__(self):
        self.actions = []

    def activate_mode(self, mode):
        self.actions.append(("activate", mode))
        calibration = PhaseCameraCalibration(
            mode=mode,
            controls=LockedCameraControls(1, 1.0, (1.0, 1.0)),
            quality=CalibrationQuality(
                True, 1, 0, 0.0, 0.0, 0.0, 0.0, 220.0, 0.0, 0.0
            ),
            metadata_samples=(),
        )
        return ActivatedCameraMode(calibration, f"{mode}-baseline-{len(self.actions)}")


def _prepare_block_backlight(workflow: SessionWorkflow) -> None:
    workflow.prepare_empty_backlight("block")


def _prepare_slide_backlight(workflow: SessionWorkflow) -> None:
    workflow.prepare_empty_backlight("slide")


class FakeStillCamera:
    def capture_still(self, path, *, settings, size):
        image = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)

    def resume_preview(self):
        pass


class ToggleTransport:
    def __init__(self, store):
        self.store = store
        self.connected = False

    def status(self, session_number):
        if not self.connected:
            raise ConnectionRefusedError("offline")
        return {"session_number": session_number}

    def upload(self, session_number, capture):
        if not self.connected:
            raise ConnectionRefusedError("offline")
        return self.store.receive_capture(
            session_number,
            capture_id=capture.capture_id,
            block_id=capture.block_id,
            checksum=capture.checksum,
            body=capture.path.read_bytes(),
        )


class FastPreprocessor:
    def __call__(self, capture_path):
        assert capture_path.is_file()
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}


def test_session_orders_block_drain_and_slide_modes_with_fresh_baselines(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    camera = FakePhaseCamera()
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=camera,
    )

    assert workflow.snapshot().phase == "blocks"
    assert camera.actions == []  # deferred — no ctor activate
    with pytest.raises(RuntimeError, match="no block"):
        workflow.calibration_for("block")

    workflow.prepare_empty_backlight("block")
    assert camera.actions == [("activate", "block")]
    assert (tmp_path / "outbox" / "camera_calibration_block.json").is_file()
    block_calibration = workflow.calibration_for("block")
    block_baseline = workflow.baseline_for("block")

    drained = workflow.finish_blocks()
    assert drained.phase == "slides"
    # slide activate also deferred until prepare_empty_backlight
    assert camera.actions == [("activate", "block")]
    with pytest.raises(RuntimeError, match="slide camera calibration is not active"):
        workflow.require_slide_mode()

    workflow.prepare_empty_backlight("slide")
    assert camera.actions[-1] == ("activate", "slide")
    assert workflow.calibration_for("slide") != block_calibration
    assert workflow.baseline_for("slide") != block_baseline
    with pytest.raises(RuntimeError, match="no block calibration is active"):
        workflow.calibration_for("block")
    workflow.require_slide_mode()

    slide_event = [
        event for event in workflow.events() if event.kind == "slide_mode_entered"
    ][-1]
    assert slide_event.message == (
        "Slide camera calibrated, controls locked, and "
        "empty-backlight baseline collected"
    )


def test_failed_slide_activation_invalidates_block_camera_state(tmp_path):
    class FailingSlideCamera(FakePhaseCamera):
        failed = False

        def activate_mode(self, mode):
            if mode == "slide" and not self.failed:
                self.failed = True
                raise RuntimeError("calibration failed")
            return super().activate_mode(mode)

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    camera = FailingSlideCamera()
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=camera,
    )

    workflow.prepare_empty_backlight("block")
    workflow.finish_blocks()

    with pytest.raises(RuntimeError, match="calibration failed"):
        workflow.prepare_empty_backlight("slide")

    with pytest.raises(RuntimeError, match="no block calibration is active"):
        workflow.calibration_for("block")
    with pytest.raises(RuntimeError, match="slide camera calibration is not active"):
        workflow.require_slide_mode()
    assert "slide_mode_entered" not in [event.kind for event in workflow.events()]

    workflow.prepare_empty_backlight("slide")
    workflow.require_slide_mode()
    assert workflow.calibration_for("slide").mode == "slide"


def test_slide_capture_persists_validated_identity_and_full_decode_audit(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    decoded = select_slide_identity((
        DecodeCandidate("zxing", "QRCode", "raw", "malformed"),
        DecodeCandidate(
            "zxing", "DataMatrix", "label+raw", "12080_51137181_01_HE"
        ),
    ))
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: decoded,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    source = _capture(tmp_path / "slide.png", 120)
    result = workflow.consume_slide_capture(CaptureRecord(
        counter=1,
        path=source,
        role="slide",
        captured_at=STARTED_AT,
    ))

    assert result.success is True
    rows = store.slide_captures(session.number)
    assert len(rows) == 1
    assert Path(rows[0]["capture_path"]).is_file()
    assert rows[0]["block_id"] == "51137181"
    assert rows[0]["symbology"] == "DataMatrix"
    attempts = rows[0]["attempts"]
    assert [attempt["payload"] for attempt in attempts] == [
        "malformed", "12080_51137181_01_HE"
    ]
    kinds = [event.kind for event in workflow.events()]
    assert "slide_identity_validated" in kinds
    # The decoded block was never scanned in this session, so the immediate
    # claimed-pair lookup fails closed to REVIEW without visual verification.
    assert kinds[-1] == "claim_review"
    assert rows[0]["verdict"] == "REVIEW"
    assert rows[0]["claim_stage"] == "identity_lookup"


def test_scanned_payload_overrides_slide_decoder_and_persists_identity(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    def unexpected_decoder(_image):
        raise AssertionError("slide decoder must not run for scanned payload")

    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=unexpected_decoder,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    result = workflow.capture_slide(
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        scanned_payload="12080_51137181_01_HE",
    )

    assert result.success is True
    assert result.block_id == "51137181"
    assert result.engine == "scanner"
    assert result.preprocessing == "scanner"
    rows = store.slide_captures(session.number)
    assert len(rows) == 1


def test_invalid_scanned_payload_fails_without_calling_slide_decoder(
    tmp_path, monkeypatch
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    def unexpected_decoder(_image):
        raise AssertionError("slide decoder must not run for scanned payload")

    monkeypatch.setattr(session_workflow_module, "sleep", lambda _seconds: None)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=unexpected_decoder,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    result = workflow.capture_slide(
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        scanned_payload="not-a-slide-identity",
    )

    assert result.success is False
    assert result.attempts[0].engine == "scanner"
    assert len(store.slide_captures(session.number)) == 1


def test_failed_slide_decode_waits_for_quiet_window_before_prompt(
    tmp_path, monkeypatch
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity(())
    clock = iter((10.0, 10.2, 11.5))
    slept = []
    monkeypatch.setattr(session_workflow_module, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(session_workflow_module, "sleep", slept.append)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    workflow.capture_slide(
        _capture(tmp_path / "unreadable.png", 120), captured_at=STARTED_AT
    )

    assert slept == [pytest.approx(1.3)]
    row = store.slide_captures(session.number)[0]
    assert row["duration_ms"] == pytest.approx(1500.0)
    event = workflow.events()[-1]
    assert event.kind == "slide_identity_failed"
    assert event.message == "Reposition slide"


def test_session_workflow_defaults_to_real_clock(tmp_path):
    """#171: omitting `clock=` must not raise, and production still measures
    real (>=0) per-stage slide durations via the default wall clock, mirroring
    `CaptureController(clock=...)` (capture_runtime.py)."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    result = workflow.capture_slide(
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        scanned_payload="12080_51137181_01_HE",
    )

    assert result.success is True
    timings = workflow.last_slide_stage_timings
    assert timings.decode_ms >= 0
    assert timings.outbox_ms >= 0
    assert timings.send_ms >= 0


def test_capture_slide_stamps_exact_decode_outbox_send_durations_against_scripted_clock(
    tmp_path,
):
    """#171: decode, outbox-write, and send-to-main-computer are each measured
    off an injectable `clock=` seam (identical prior art to
    `CaptureController(clock=...)`), against exact scripted durations rather
    than a loose `>= 0` check.

    Scripted clock schedule (each call returns the next value):
      1. capture_slide entry (`started`)
      2. decode finished (before DECODE_BUDGET_SECONDS padding)   -> +10ms
      3. outbox-write (`outbox.publish_slide`) start
      4. outbox-write finished                                    -> +35ms
      5. send-to-main-computer (`outbox.replay_slides`) start
      6. send-to-main-computer finished                           -> +53ms
      7. (any further/final clock read, e.g. the existing decode
         audit `duration_ms` end-of-call reading)
    """
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    schedule = iter([
        0.000,
        0.010,
        0.010,
        0.045,
        0.045,
        0.098,
        0.098,
    ])

    def scripted_clock() -> float:
        return next(schedule)

    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        clock=scripted_clock,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    result = workflow.capture_slide(
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        scanned_payload="12080_51137181_01_HE",
    )

    assert result.success is True
    timings = workflow.last_slide_stage_timings
    assert timings.decode_ms == pytest.approx(10.0)
    assert timings.outbox_ms == pytest.approx(35.0)
    assert timings.send_ms == pytest.approx(53.0)


def test_unreadable_slide_reposition_state_is_restored_after_restart(
    tmp_path, monkeypatch
):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity(())
    clock = iter((10.0, 10.2, 11.5))
    monkeypatch.setattr(session_workflow_module, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(session_workflow_module, "sleep", lambda seconds: None)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    workflow.capture_slide(
        _capture(tmp_path / "unreadable.png", 120), captured_at=STARTED_AT
    )

    restarted_store = ProcessingStore(root)
    restarted = SessionWorkflow(
        session=restarted_store.resume_session(session.number),
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(restarted_store),
        camera=FakePhaseCamera(),
    )
    _prepare_slide_backlight(restarted)
    capture_session = CaptureSession(SessionConfig(baseline_frames=1), mode="slide")
    restarted.restore_slide_capture_session(capture_session)

    assert capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION
    capture_session.confirm_empty()
    capture_session.accept_frame(
        np.full((80, 120, 3), 180, dtype=np.uint8), now=0.0
    )
    assert capture_session.state is CaptureState.REPOSITION_SLIDE
    assert restarted.events()[-1].kind == "slide_reposition_required"
    assert restarted.events()[-1].message == "Reposition slide"


def test_recovered_unreadable_slide_can_be_skipped_before_baseline(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)
    store.record_slide_capture(
        session.number,
        _capture(tmp_path / "unreadable.png", 120),
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=1500.0,
    )
    restarted_store = ProcessingStore(root)
    restarted = SessionWorkflow(
        session=restarted_store.resume_session(session.number),
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(restarted_store),
        camera=FakePhaseCamera(),
    )
    _prepare_slide_backlight(restarted)
    capture_session = CaptureSession(SessionConfig(baseline_frames=1), mode="slide")
    restarted.restore_slide_capture_session(capture_session)

    restarted.skip_unreadable_slide(capture_session)

    assert restarted_store.slide_recovery_state(session.number) == (
        "waiting_for_removal"
    )
    capture_session.confirm_empty()
    capture_session.accept_frame(
        np.full((80, 120, 3), 180, dtype=np.uint8), now=0.0
    )
    assert capture_session.state is CaptureState.WAITING_FOR_REMOVAL


def test_removing_unreadable_slide_clears_durable_recovery_state(
    tmp_path, monkeypatch
):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity(())
    monkeypatch.setattr(session_workflow_module, "DECODE_BUDGET_SECONDS", 0.0)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    workflow.capture_slide(
        _capture(tmp_path / "unreadable.png", 120), captured_at=STARTED_AT
    )
    capture_session = CaptureSession(
        SessionConfig(baseline_frames=1, removal_duration=0.5), mode="slide"
    )
    capture_session.confirm_empty()
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    capture_session.accept_frame(empty, now=0.0)
    workflow.restore_slide_capture_session(capture_session)

    capture_session.accept_frame(empty, now=1.0)
    capture_session.accept_frame(empty, now=1.5)

    assert capture_session.state is CaptureState.EMPTY
    assert store.slide_recovery_state(session.number) == "waiting"

    restarted_store = ProcessingStore(root)
    restarted = SessionWorkflow(
        session=restarted_store.resume_session(session.number),
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(restarted_store),
        camera=FakePhaseCamera(),
    )
    restarted_capture_session = CaptureSession(
        SessionConfig(baseline_frames=1), mode="slide"
    )
    _prepare_slide_backlight(restarted)
    restarted.restore_slide_capture_session(restarted_capture_session)
    assert restarted_capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION


def test_removal_survives_recovery_state_persistence_error(
    tmp_path, monkeypatch
):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity(())
    monkeypatch.setattr(session_workflow_module, "DECODE_BUDGET_SECONDS", 0.0)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    workflow.capture_slide(
        _capture(tmp_path / "unreadable.png", 120), captured_at=STARTED_AT
    )
    capture_session = CaptureSession(
        SessionConfig(baseline_frames=1, removal_duration=0.5), mode="slide"
    )
    capture_session.confirm_empty()
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    capture_session.accept_frame(empty, now=0.0)
    workflow.restore_slide_capture_session(capture_session)
    monkeypatch.setattr(
        store,
        "mark_waiting_for_slide",
        lambda session_number: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    capture_session.accept_frame(empty, now=1.0)
    capture_session.accept_frame(empty, now=1.5)

    assert capture_session.state is CaptureState.REPOSITION_SLIDE
    assert store.slide_recovery_state(session.number) == "reposition"


def test_skipped_slide_remains_disarmed_when_removal_persistence_fails(
    tmp_path, monkeypatch
):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)
    store.record_slide_capture(
        session.number,
        _capture(tmp_path / "unreadable.png", 120),
        captured_at=STARTED_AT,
        result=select_slide_identity(()),
        duration_ms=1500.0,
    )
    capture_session = CaptureSession(
        SessionConfig(baseline_frames=1, removal_duration=0.5), mode="slide"
    )
    capture_session.confirm_empty()
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    capture_session.accept_frame(empty, now=0.0)
    workflow.restore_slide_capture_session(capture_session)
    workflow.skip_unreadable_slide(capture_session)
    monkeypatch.setattr(
        store,
        "mark_waiting_for_slide",
        lambda session_number: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    capture_session.accept_frame(empty, now=1.0)
    capture_session.accept_frame(empty, now=1.5)

    assert capture_session.state is CaptureState.WAITING_FOR_REMOVAL
    assert not capture_session.unreadable_slide_can_be_skipped
    assert store.slide_recovery_state(session.number) == "waiting_for_removal"


def test_workflow_recaptures_once_after_reposition_and_skip_retains_audits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(session_workflow_module, "DECODE_BUDGET_SECONDS", 0.0)
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity((DecodeCandidate(
        "zxing", "QRCode", "raw", "malformed"
    ),))
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)
    capture_session = CaptureSession(
        SessionConfig(baseline_frames=1, stable_duration=1.0), mode="slide"
    )
    capture_session.confirm_empty()
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    slide = empty.copy()
    moved_slide = empty.copy()
    moved_again = empty.copy()
    cv2.rectangle(slide, (30, 25), (90, 55), (45, 45, 45), -1)
    cv2.rectangle(moved_slide, (40, 25), (100, 55), (45, 45, 45), -1)
    cv2.rectangle(moved_again, (50, 25), (110, 55), (45, 45, 45), -1)
    capture_session.accept_frame(empty, now=0.0)
    controller = CaptureController(
        session=capture_session,
        camera=FakeStillCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        capture_consumer=workflow.consume_slide_capture,
    )
    workflow.restore_slide_capture_session(capture_session)

    controller.handle_frame(slide, now=1.0, captured_at=STARTED_AT)
    controller.handle_frame(slide, now=2.0, captured_at=STARTED_AT)
    controller.handle_frame(slide, now=3.0, captured_at=STARTED_AT)
    controller.handle_frame(
        moved_slide, now=4.0, captured_at=STARTED_AT + timedelta(seconds=1)
    )
    controller.handle_frame(
        moved_again, now=5.0, captured_at=STARTED_AT + timedelta(seconds=1)
    )
    controller.handle_frame(
        moved_again, now=6.0, captured_at=STARTED_AT + timedelta(seconds=2)
    )
    workflow.skip_unreadable_slide(capture_session)

    captures = store.slide_captures(session.number)
    assert len(captures) == 2
    assert all(capture["success"] == 0 for capture in captures)
    assert all(capture["reason"] for capture in captures)
    assert all(capture["duration_ms"] >= 0 for capture in captures)
    assert all(len(capture["attempts"]) == 1 for capture in captures)
    assert capture_session.state is CaptureState.WAITING_FOR_REMOVAL
    assert store.slide_recovery_state(session.number) == "waiting_for_removal"
    assert workflow.snapshot().latest_block_id is None

    restarted_store = ProcessingStore(tmp_path / "processing")
    restarted = SessionWorkflow(
        session=restarted_store.resume_session(session.number),
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(restarted_store),
        camera=FakePhaseCamera(),
    )
    restarted_capture_session = CaptureSession(
        SessionConfig(baseline_frames=1), mode="slide"
    )
    _prepare_slide_backlight(restarted)
    restarted.restore_slide_capture_session(restarted_capture_session)

    assert (
        restarted_capture_session.state
        is CaptureState.AWAITING_BASELINE_CONFIRMATION
    )
    restarted_capture_session.confirm_empty()
    restarted_capture_session.accept_frame(empty, now=0.0)
    assert restarted_capture_session.state is CaptureState.WAITING_FOR_REMOVAL


def test_settled_slide_capture_automatically_emits_validated_identity(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    decoded = select_slide_identity((DecodeCandidate(
        "zxing", "QRCode", "raw", "12080_51137181_01_HE"
    ),))
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        camera=FakePhaseCamera(),
        slide_decoder=lambda image: decoded,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)
    capture_session = CaptureSession(
        SessionConfig(baseline_frames=1), mode="slide"
    )
    capture_session.confirm_empty()
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    slide = empty.copy()
    cv2.rectangle(slide, (30, 25), (90, 55), (45, 45, 45), -1)
    capture_session.accept_frame(empty, now=0.0)
    controller = CaptureController(
        session=capture_session,
        camera=FakeStillCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        capture_consumer=workflow.consume_slide_capture,
    )
    workflow.restore_slide_capture_session(capture_session)

    controller.handle_frame(slide, now=1.0, captured_at=STARTED_AT)
    controller.handle_frame(slide, now=2.0, captured_at=STARTED_AT)
    controller.handle_frame(slide, now=3.0, captured_at=STARTED_AT)

    rows = store.slide_captures(session.number)
    assert len(rows) == 1
    assert rows[0]["block_id"] == "51137181"
    kinds = [event.kind for event in workflow.events()]
    assert "slide_identity_validated" in kinds
    # No block was scanned this session, so the claim lookup fails closed.
    assert kinds[-1] == "claim_review"


def test_finish_blocks_is_resumable_across_disconnect_and_reports_drain_counts(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
        camera=FakePhaseCamera(),
    )
    assert workflow.scan_block("51151378").accepted
    outbox.publish_block(
        _capture(tmp_path / "block.png", 80), "51151378", STARTED_AT
    )

    draining = workflow.finish_blocks()

    assert draining.phase == "draining_blocks"
    assert draining.pending_transfers == 1
    assert draining.preprocessing_pending == 0
    assert not workflow.scan_block("87654321").accepted

    # A process restart reconstructs the durable draining phase instead of
    # defaulting to blocks or entering slides early.
    workflow = SessionWorkflow(
        session=store.resume_session(session.number),
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport,
        camera=FakePhaseCamera(),
    )
    assert workflow.snapshot().phase == "draining_blocks"

    transport.connected = True
    workflow.poll_status()
    store.wait_for_jobs()
    finished = workflow.poll_drain()

    assert finished.phase == "slides"
    assert finished.pending_transfers == 0
    kinds = [event.kind for event in workflow.events()]
    assert "block_drain_started" in kinds
    assert "block_drain_progress" in kinds
    assert "slide_mode_entered" not in kinds
    workflow.prepare_empty_backlight("slide")
    assert "slide_mode_entered" in [event.kind for event in workflow.events()]


def test_failed_block_auto_unusable_on_drain_enters_slides(tmp_path):
    """#187: a failed segment must not stall Finish Blocks on Processing…"""

    def fail(_capture_path):
        raise ValueError("specimen could not be segmented")

    store = ProcessingStore(tmp_path / "processing", preprocessor=fail)
    session = store.start_session(started_at=STARTED_AT)
    transport = ToggleTransport(store)
    transport.connected = True
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport,
        camera=FakePhaseCamera(),
    )
    workflow.capture_block(
        "51151378", _capture(tmp_path / "failed.png", 80), captured_at=STARTED_AT
    )
    store.wait_for_jobs()
    assert store.get_set(session.number, "51151378")["preprocessing_status"] == "failed"
    assert len(workflow.active_warnings()) == 1

    assert workflow.finish_blocks().phase == "slides"
    row = store.get_set(session.number, "51151378")
    assert row["preprocessing_status"] == "unusable"
    assert row["dismissed_at"] is not None
    assert "specimen could not be segmented" in row["unusable_reason"]
    assert workflow.active_warnings() == ()
    assert workflow.snapshot().unresolved_blocks == 0
    assert any(
        event.kind == "block_dismissed" and event.block_id == "51151378"
        for event in workflow.events()
    )


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


def test_finish_blocks_in_hybrid_mode_freezes_pool_with_two_usable_blocks(tmp_path):
    """#250: Finish Blocks in HYBRID mode freezes the Hybrid Candidate Pool
    instead of unconditionally entering slides."""
    store = ProcessingStore(
        tmp_path / "processing", preprocessor=FastPreprocessor(),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT)
    transport = ToggleTransport(store)
    transport.connected = True
    workflow = SessionWorkflow(
        session=session, store=store, outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport, camera=FakePhaseCamera(),
        session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    # #269: Hybrid now requires the same real work-order bracket Open
    # Retrieval already has -- freeze_hybrid_pool resolves the open
    # (capturing) work order internally and keys the pool on it.
    work_order_id = workflow.start_work_order()
    workflow.capture_block(
        "51151378", _capture(tmp_path / "b1.png", 10), captured_at=STARTED_AT
    )
    workflow.capture_block(
        "62262489", _capture(tmp_path / "b2.png", 20), captured_at=STARTED_AT
    )
    store.wait_for_jobs()

    finished = workflow.finish_blocks()

    assert finished.phase == "slides"
    pool = store.hybrid_pool(work_order_id)
    assert pool is not None
    assert set(pool.block_ids) == {"51151378", "62262489"}
    assert pool.descriptor_names == ("fake_v1",)


def test_finish_blocks_in_hybrid_mode_with_one_block_returns_to_blocks(tmp_path):
    """#250: fewer than two usable blocks does not freeze, does not abandon,
    and leaves the work order usable (back to the blocks phase)."""
    store = ProcessingStore(
        tmp_path / "processing", preprocessor=FastPreprocessor(),
        fingerprint_builder=_fake_fingerprint_builder,
        score_cache_builder=_fake_score_cache_builder,
    )
    session = store.start_session(started_at=STARTED_AT)
    transport = ToggleTransport(store)
    transport.connected = True
    workflow = SessionWorkflow(
        session=session, store=store, outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport, camera=FakePhaseCamera(),
        session_mode=SessionMode.HYBRID,
        hybrid_descriptor_names=("fake_v1",),
    )
    work_order_id = workflow.start_work_order()
    workflow.capture_block(
        "51151378", _capture(tmp_path / "b1.png", 10), captured_at=STARTED_AT
    )
    store.wait_for_jobs()

    finished = workflow.finish_blocks()

    assert finished.phase == "blocks"
    assert store.hybrid_pool(work_order_id) is None
    # The work order remains usable: the operator can capture another block
    # and click Finish Blocks again.
    assert workflow.scan_block("62262489").accepted


@pytest.mark.parametrize("session_mode", [SessionMode.NORMAL, SessionMode.OPEN_RETRIEVAL])
def test_finish_blocks_in_normal_and_open_retrieval_modes_never_touch_hybrid_pool(
    tmp_path, session_mode
):
    """#250 acceptance criterion: NORMAL/OPEN_RETRIEVAL Finish Blocks behavior
    is unchanged -- a single captured block still enters slides directly
    (no Hybrid >=2 usable-block requirement), and no Hybrid Candidate Pool is
    ever created."""
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    transport = ToggleTransport(store)
    transport.connected = True
    workflow = SessionWorkflow(
        session=session, store=store, outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport, camera=FakePhaseCamera(), session_mode=session_mode,
    )
    workflow.capture_block(
        "51151378", _capture(tmp_path / "b1.png", 10), captured_at=STARTED_AT
    )
    store.wait_for_jobs()

    finished = workflow.finish_blocks()

    assert finished.phase == "slides"
    # #269 FIX5b: `hybrid_pool`'s parameter is `work_order_id`, not
    # `session_number` -- passing `session.number` here is a semantically
    # wrong negative control that only ever passed vacuously (`hybrid_pools`
    # is empty for NORMAL/OPEN_RETRIEVAL regardless of which id is queried).
    # Assert the real invariant directly against the table instead.
    with store._connect() as db:
        count = db.execute("SELECT COUNT(*) AS n FROM hybrid_pools").fetchone()["n"]
    assert count == 0


def test_framing_calibration_is_durable_and_explicitly_replaced(tmp_path):
    path = tmp_path / "framing-calibration.json"
    calibration = FramingCalibrationStore(path)
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store),
        framing_calibration=calibration,
    )
    assert workflow.view_framing_calibration() is None

    workflow.approve_framing_calibration(
        "alignment-v1.png", approved_at=STARTED_AT
    )
    assert FramingCalibrationStore(path).view().image_path == Path("alignment-v1.png")

    workflow.recalibrate_framing(
        "alignment-v2.png", approved_at=STARTED_AT
    )
    assert FramingCalibrationStore(path).view().image_path == Path("alignment-v2.png")
    assert [event.kind for event in workflow.events()][-2:] == [
        "framing_calibration_approved",
        "framing_recalibrated",
    ]


_CAPTURE_PNGS: dict[int, bytes] = {}


def _capture(path: Path, value: int) -> Path:
    encoded = _CAPTURE_PNGS.get(value)
    if encoded is None:
        image = np.full((3040, 4056, 3), value, dtype=np.uint8)
        success, png = cv2.imencode(".png", image)
        assert success
        encoded = png.tobytes()
        _CAPTURE_PNGS[value] = encoded
    path.write_bytes(encoded)
    return path


_REAL_WRITE_QC = ProcessingStore._write_qc


@pytest.fixture(autouse=True)
def lightweight_qc_artifacts(monkeypatch):
    """Keep workflow tests focused on artifact lifecycle, not large rendering."""
    def write_qc(capture, mask, destination):
        assert capture.is_file()
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        assert capture.is_file()
        assert reason
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_slide_qc(image, mask, destination):
        assert image.ndim == 3
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )
    monkeypatch.setattr(
        ProcessingStore, "_write_slide_qc", staticmethod(write_slide_qc)
    )


def test_real_qc_writer_creates_full_resolution_three_panel(tmp_path):
    capture = _capture(tmp_path / "capture.png", 80)
    mask = np.full((3040, 4056), 255, dtype=np.uint8)
    destination = tmp_path / "qc.png"

    _REAL_WRITE_QC(capture, mask, destination)

    panel = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape == (3040, 4056 * 3, 3)


class ControllablePreprocessor:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def __call__(self, capture_path: Path):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5), "test did not release preprocessing"
        assert capture_path.is_file()
        return np.full((8, 8), 255, dtype=np.uint8), {
            "method": "controllable-test"
        }


def test_block_runs_from_durable_outbox_through_loopback_preprocessing(tmp_path):
    preprocessor = ControllablePreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "pi-outbox"),
            transport=HttpCaptureClient(receiver.url),
        )
        first = workflow.capture_block(
            "51151378", _capture(tmp_path / "first.png", 80), captured_at=STARTED_AT
        )

        assert first.acknowledged
        assert first.capture_id.startswith("capture_000001_block_51151378_")
        assert preprocessor.started.wait(timeout=2)

        # Preprocessing is still blocked, but the operator-facing workflow accepts
        # another scan immediately.
        assert workflow.scan_block("87654321").accepted

        preprocessor.release.set()
        workflow.wait_for_block_jobs()

    metadata = session.directory / "session.json"
    assert metadata.is_file()
    assert '"session_number": 1' in metadata.read_text(encoding="utf-8")
    assert '"started_at": "2026-07-02T18:05:06+00:00"' in metadata.read_text(
        encoding="utf-8"
    )

    block = store.get_set(session.number, "51151378")
    assert block["capture_id"] == first.capture_id
    assert block["preprocessing_status"] == "complete"
    assert Path(block["mask_path"]).is_file()
    assert cv2.imread(block["mask_path"], cv2.IMREAD_UNCHANGED).dtype == np.uint8
    assert Path(block["qc_path"]).is_file()
    assert preprocessor.calls == 1

    event_kinds = [event.kind for event in workflow.events()]
    assert {"session_started", "upload_acknowledged", "preprocessing_started",
            "preprocessing_complete"} <= set(event_kinds)
    latest = workflow.snapshot()
    assert latest.session_number == 1
    assert latest.phase == "blocks"
    assert latest.latest_block_id == "51151378"
    assert latest.preprocessing_pending == 0

    restarted = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    later = restarted.start_session(
        started_at=datetime(2026, 7, 2, 19, 0, tzinfo=timezone.utc)
    )
    assert later.number == 2


def test_receiver_rejects_bad_checksum_without_acknowledging_or_queueing(tmp_path):
    preprocessor = ControllablePreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted

    with LoopbackCaptureReceiver(store) as receiver:
        request = Request(
            f"{receiver.url}/sessions/{session.number}/captures",
            data=b"incomplete image",
            method="POST",
            headers={
                "X-Capture-Id": "capture-bad",
                "X-Block-Id": "51151378",
                "X-Checksum-Sha256": "0" * 64,
            },
        )
        try:
            urlopen(request, timeout=2)
            assert False, "bad checksum was acknowledged"
        except HTTPError as exc:
            assert exc.code == 400

    row = store.get_set(session.number, "51151378")
    assert row["capture_id"] is None
    assert row["preprocessing_status"] == "awaiting_capture"
    assert preprocessor.calls == 0


def test_repeated_capture_id_returns_receipt_without_duplicate_job(tmp_path):
    class ImmediatePreprocessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, capture_path):
            self.calls += 1
            return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}

    preprocessor = ImmediatePreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    capture = outbox.publish_block(
        _capture(tmp_path / "block.png", 70), "51151378", STARTED_AT
    )
    assert store.scan_block(session.number, "51151378").accepted

    with LoopbackCaptureReceiver(store) as receiver:
        client = HttpCaptureClient(receiver.url)
        first = client.upload(session.number, capture)
        second = client.upload(session.number, capture)
        store.wait_for_jobs()

    assert first == second
    assert preprocessor.calls == 1


def test_pi_restart_replays_pending_captures_in_capture_order(tmp_path):
    class RecoveringTransport:
        def __init__(self):
            self.available = False
            self.uploaded = []

        def status(self, session_number):
            if not self.available:
                raise ConnectionRefusedError("processing computer is offline")
            return {"session_number": session_number, "phase": "blocks"}

        def upload(self, session_number, capture):
            self.uploaded.append(capture.capture_id)
            return UploadReceipt(capture.capture_id, True, capture.checksum)

    outbox_path = tmp_path / "pi-outbox"
    transport = RecoveringTransport()
    first_outbox = PiOutbox(outbox_path)
    first = first_outbox.publish_block(
        _capture(tmp_path / "first.png", 60), "51151378", STARTED_AT
    )
    second = first_outbox.publish_block(
        _capture(tmp_path / "second.png", 70), "87654321", STARTED_AT
    )

    assert first_outbox.replay(1, transport) == ()
    restarted = PiOutbox(outbox_path)
    assert [entry.capture_id for entry in restarted.pending()] == [
        first.capture_id,
        second.capture_id,
    ]

    transport.available = True
    receipts = restarted.replay(1, transport)

    assert [receipt.capture_id for receipt in receipts] == [
        first.capture_id,
        second.capture_id,
    ]
    assert transport.uploaded == [first.capture_id, second.capture_id]
    assert restarted.pending() == ()
    assert [entry.state for entry in restarted.entries()] == [
        "acknowledged",
        "acknowledged",
    ]
    # Even a later stale status poll has no authority to regress durable acks.
    transport.uploaded.clear()
    assert PiOutbox(outbox_path).replay(1, transport) == ()
    assert transport.uploaded == []
    assert first.path.is_file() and second.path.is_file()


def test_capture_continues_while_receiver_is_unreachable(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:1", timeout=0.1),
    )

    assert workflow.capture_block(
        "51151378", _capture(tmp_path / "one.png", 50), captured_at=STARTED_AT
    ) is None
    assert workflow.capture_block(
        "87654321", _capture(tmp_path / "two.png", 60), captured_at=STARTED_AT
    ) is None

    assert len(workflow.outbox.pending()) == 2
    assert all(entry.path.is_file() for entry in workflow.outbox.pending())


def test_receiver_restart_resumes_same_session_and_queued_job(tmp_path):
    class ImmediatePreprocessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, capture_path):
            self.calls += 1
            return np.full((8, 8), 255, dtype=np.uint8), {"recovered": True}

    root = tmp_path / "processing"
    first_store = ProcessingStore(root, recover_jobs=False)
    session = first_store.start_session(started_at=STARTED_AT)
    assert first_store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "block.png", 80), "51151378", STARTED_AT
    )
    # Model a crash after durable receipt/queue commit but before worker start.
    first_store.receive_capture(
        session.number,
        capture_id=capture.capture_id,
        block_id=capture.block_id,
        checksum=capture.checksum,
        body=capture.path.read_bytes(),
        start_job=False,
    )

    preprocessor = ImmediatePreprocessor()
    restarted = ProcessingStore(root, preprocessor=preprocessor)
    resumed = restarted.resume_session()
    restarted.wait_for_jobs()

    assert resumed.number == session.number
    assert resumed.started_at == session.started_at
    assert restarted.get_set(session.number, "51151378")["preprocessing_status"] == "complete"
    assert preprocessor.calls == 1


def test_lost_ack_retry_uses_original_receipt_without_duplicate_work(tmp_path):
    class ImmediatePreprocessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, capture_path):
            self.calls += 1
            return np.full((8, 8), 255, dtype=np.uint8), {}

    class LoseFirstAcknowledgement:
        def __init__(self, store):
            self.store = store
            self.lose_ack = True

        def status(self, session_number):
            return {"session_number": session_number}

        def upload(self, session_number, capture):
            receipt = self.store.receive_capture(
                session_number,
                capture_id=capture.capture_id,
                block_id=capture.block_id,
                checksum=capture.checksum,
                body=capture.path.read_bytes(),
            )
            if self.lose_ack:
                self.lose_ack = False
                raise ConnectionResetError("acknowledgement was lost")
            return receipt

    preprocessor = ImmediatePreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    outbox = PiOutbox(tmp_path / "outbox")
    capture = outbox.publish_block(
        _capture(tmp_path / "block.png", 90), "51151378", STARTED_AT
    )
    transport = LoseFirstAcknowledgement(store)

    assert outbox.replay(session.number, transport) == ()
    receipts = outbox.replay(session.number, transport)
    store.wait_for_jobs()

    assert receipts == (UploadReceipt(capture.capture_id, True, capture.checksum),)
    assert outbox.pending() == ()
    assert preprocessor.calls == 1


def test_restart_recovers_capture_published_before_outbox_metadata(tmp_path):
    directory = tmp_path / "outbox"
    record = CaptureStore(directory).publish(  # simulate a process stop at the seam
        _capture(tmp_path / "block.png", 65),
        "block",
        block_id="51151378",
        captured_at=STARTED_AT,
    )
    assert not record.path.with_suffix(".json").exists()

    recovered = PiOutbox(directory).pending()

    assert len(recovered) == 1
    assert recovered[0].capture_id == record.path.stem
    assert recovered[0].path.read_bytes() == record.path.read_bytes()


def test_duplicate_is_not_acknowledged_if_durable_receiver_copy_is_missing(tmp_path):
    store = ProcessingStore(tmp_path / "processing", recover_jobs=False)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "block.png", 80), "51151378", STARTED_AT
    )
    receipt = store.receive_capture(
        session.number,
        capture_id=capture.capture_id,
        block_id=capture.block_id,
        checksum=capture.checksum,
        body=capture.path.read_bytes(),
        start_job=False,
    )
    Path(store.get_set(session.number, "51151378")["capture_path"]).unlink()

    try:
        store.receive_capture(
            session.number,
            capture_id=capture.capture_id,
            block_id=capture.block_id,
            checksum=capture.checksum,
            body=capture.path.read_bytes(),
        )
        assert False, "missing durable storage was acknowledged"
    except ValueError as exc:
        assert "not durably stored" in str(exc)
    assert receipt.acknowledged


def test_duplicate_scan_warns_without_replacing_or_adding_a_set(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    assert store.scan_block(session.number, "51151378").accepted
    duplicate = store.scan_block(session.number, "51151378")

    assert not duplicate.accepted
    assert duplicate.message == "Block already scanned"
    assert store.get_set(session.number, "51151378")["capture_id"] is None
    duplicate_events = [
        event for event in store.events(session.number)
        if event.kind == "duplicate_block_scan"
    ]
    assert len(duplicate_events) == 1
    assert duplicate_events[0].message == "Block already scanned"


def test_awaiting_capture_blocks_are_ordered_and_unscan_only_removes_uncaptured(
    tmp_path,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "22222222").accepted
    assert store.scan_block(session.number, "11111111").accepted

    assert store.awaiting_capture_blocks(session.number) == (
        "22222222", "11111111"
    )
    assert store.unscan_block(session.number, "22222222")
    assert store.awaiting_capture_blocks(session.number) == ("11111111",)

    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "captured.png", 80), "11111111", STARTED_AT
    )
    store.receive_capture(
        session.number,
        capture_id=capture.capture_id,
        block_id=capture.block_id,
        checksum=capture.checksum,
        body=capture.path.read_bytes(),
    )

    assert not store.unscan_block(session.number, "11111111")
    assert store.get_set(session.number, "11111111")["capture_id"] == capture.capture_id


def test_preprocessing_failure_preserves_evidence_and_does_not_stop_capture(tmp_path):
    class FailingPreprocessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, capture_path):
            self.calls += 1
            raise ValueError("cassette window is not evaluable")

    preprocessor = FailingPreprocessor()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "failed.png", 80), "51151378", STARTED_AT
    )

    store.receive_capture(
        session.number,
        capture_id=capture.capture_id,
        block_id=capture.block_id,
        checksum=capture.checksum,
        body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()

    failed = store.get_set(session.number, "51151378")
    assert failed["preprocessing_status"] == "failed"
    assert failed["failure_reason"] == "cassette window is not evaluable"
    assert failed["mask_path"] is None
    assert Path(failed["qc_path"]).is_file()
    warnings = store.active_warnings(session.number)
    assert len(warnings) == 1
    assert warnings[0].block_id == "51151378"
    assert warnings[0].can_recapture and warnings[0].can_dismiss
    readiness = store.block_readiness(session.number, "51151378")
    assert not readiness.evaluable
    assert readiness.review_reason == "cassette window is not evaluable"
    assert store.scan_block(session.number, "87654321").accepted
    assert any(event.kind == "failed_block_warning" for event in store.events(session.number))
    restarted = ProcessingStore(
        tmp_path / "processing", preprocessor=preprocessor, recover_jobs=False
    )
    assert restarted.active_warnings(session.number)[0].block_id == "51151378"


def test_successful_recapture_replaces_failed_input_once_and_clears_warning(tmp_path):
    class FailThenSucceed:
        def __init__(self):
            self.calls = 0

        def __call__(self, capture_path):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("first capture is blurred")
            assert capture_path.is_file()
            return np.full((8, 8), 255, dtype=np.uint8), {"attempt": self.calls}

    preprocessor = FailThenSucceed()
    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    with LoopbackCaptureReceiver(store) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
        )
        first = workflow.capture_block(
            "51151378", _capture(tmp_path / "first.png", 50), captured_at=STARTED_AT
        )
        workflow.wait_for_block_jobs()
        assert len(workflow.active_warnings()) == 1
        try:
            workflow.recapture_block(
                "87654321", _capture(tmp_path / "wrong.png", 70),
                captured_at=STARTED_AT,
            )
            assert False, "recapture was accepted for a block that did not fail"
        except ValueError as exc:
            assert "intended failed block" in str(exc)

        second = workflow.recapture_block(
            "51151378", _capture(tmp_path / "second.png", 90), captured_at=STARTED_AT
        )
        workflow.wait_for_block_jobs()

    row = store.get_set(session.number, "51151378")
    assert first.capture_id != second.capture_id
    assert row["capture_id"] == second.capture_id
    assert row["preprocessing_status"] == "complete"
    assert row["failure_reason"] is None
    assert preprocessor.calls == 2
    assert workflow.active_warnings() == ()
    assert len([event for event in workflow.events() if event.kind == "block_recaptured"]) == 1


def test_dismissal_is_auditable_and_failed_block_stays_fail_closed(tmp_path):
    def fail(capture_path):
        raise ValueError("no tissue found")

    store = ProcessingStore(tmp_path / "processing", preprocessor=fail)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "failed.png", 50), "51151378", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()

    store.dismiss_block(session.number, "51151378", reason="operator confirmed unusable")

    row = store.get_set(session.number, "51151378")
    assert row["preprocessing_status"] == "unusable"
    assert row["unusable_reason"] == "operator confirmed unusable"
    assert row["dismissed_at"] is not None
    assert row["mask_path"] is None
    assert store.active_warnings(session.number) == ()
    readiness = store.block_readiness(session.number, "51151378")
    assert not readiness.evaluable
    assert readiness.review_reason == "operator confirmed unusable"


def test_empty_preprocessing_result_is_non_evaluable_without_a_fake_mask(tmp_path):
    def empty(capture_path):
        assert capture_path.is_file()
        return np.zeros((8, 8), dtype=np.uint8), {}

    store = ProcessingStore(tmp_path / "processing", preprocessor=empty)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "empty.png", 50), "51151378", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()

    row = store.get_set(session.number, "51151378")
    assert row["preprocessing_status"] == "failed"
    assert row["failure_reason"] == "preprocessor returned an empty comparable mask"
    assert row["mask_path"] is None


def _identical_mask_slide_preprocessor(_img):
    """Deterministic evaluable slide result that mirrors ``FastPreprocessor``."""
    return PreparedSpecimen(
        role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
    )


def _evaluable_block(store, session, tmp_path, block_id="51151378"):
    assert store.scan_block(session.number, block_id).accepted
    capture = PiOutbox(tmp_path / "outbox_for_test").publish_block(
        _capture(tmp_path / f"{block_id}_block.png", 80), block_id, STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    return block_id


def test_evaluable_claim_reuses_block_mask_and_persists_pass_verdict(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert outcome.verdict == "PASS"
    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"
    assert row["score"] == pytest.approx(1.0)
    assert row["decision_stage"] == "scoring"
    assert row["slide_capture_id"] == "slide_capture_1"
    assert row["decided_at"] is not None

    export = (store.root / f"session_{session.number:06d}_{STARTED_AT:%Y%m%dT%H%M%SZ}"
              / "decisions.csv").read_text(encoding="utf-8")
    assert block_id in export
    assert "PASS" in export

    kinds = [event.kind for event in store.events(session.number)]
    assert kinds[-1] == "claim_pass"


def test_absent_block_id_reviews_without_scoring(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    outcome = store.resolve_claim(
        session.number, "99999999", "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert outcome.verdict == "REVIEW"
    assert outcome.stage == "identity_lookup"
    with pytest.raises(KeyError):
        store.get_set(session.number, "99999999")


def test_dismissed_block_reviews_claim_without_invoking_scorer(tmp_path):
    def fail(_capture_path):
        raise ValueError("no tissue found")

    calls = {"count": 0}

    def poisoned_slide_preprocessor(_img):
        calls["count"] += 1
        raise AssertionError("scorer must not run for a dismissed block")

    store = ProcessingStore(
        tmp_path / "processing", preprocessor=fail,
        slide_preprocessor=poisoned_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "failed.png", 50), "51151378", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    store.dismiss_block(session.number, "51151378", reason="operator confirmed unusable")

    outcome = store.resolve_claim(
        session.number, "51151378", "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert outcome.verdict == "REVIEW"
    assert outcome.stage == "block_unusable"
    assert outcome.reason == "operator confirmed unusable"
    assert calls["count"] == 0
    row = store.get_set(session.number, "51151378")
    assert row["verdict"] == "REVIEW"


def test_evaluable_claim_invokes_slide_preprocessor_exactly_once(tmp_path):
    """Complements
    `test_dismissed_block_reviews_claim_without_invoking_scorer`: pins the
    OTHER side of af526f7's hoisted `_prepare_slide_for_artifacts` call so a
    future change cannot "fix" the not-evaluable leak by making
    `resolve_claim` stop preparing slides altogether. An evaluable block
    must still invoke `self.slide_preprocessor` -- exactly once, never
    zero, never twice -- on the ordinary scoring path.
    """
    calls = {"count": 0}

    def counting_slide_preprocessor(_img):
        calls["count"] += 1
        return _identical_mask_slide_preprocessor(_img)

    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=counting_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert outcome.verdict == "PASS"
    assert calls["count"] == 1


def test_second_completed_claim_is_rejected_as_already_processed(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    first = store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide_1.png", 120),
    )
    assert first.accepted

    second = store.resolve_claim(
        session.number, block_id, "slide_capture_2",
        _capture(tmp_path / "slide_2.png", 121),
    )

    assert not second.accepted
    assert second.message == "Slide already processed"
    row = store.get_set(session.number, block_id)
    assert row["slide_capture_id"] == "slide_capture_1"
    assert row["verdict"] == "PASS"
    kinds = [event.kind for event in store.events(session.number)]
    assert kinds[-1] == "slide_already_processed"


def test_precheck_slide_scan_flags_an_already_verdicted_block(tmp_path):
    # Scan-time mirror of the post-capture "already processed" guard above:
    # a handheld re-scan of a block that already carries a durable verdict is
    # rejected before anything is stashed, and emits the kiosk-flash event.
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide_1.png", 120),
    )
    assert store.get_set(session.number, block_id)["verdict"] is not None

    accepted = store.precheck_slide_scan(session.number, block_id)

    assert accepted is False
    kinds = [event.kind for event in store.events(session.number)]
    assert kinds[-1] == "duplicate_slide_scan"


def test_precheck_slide_scan_allows_unverdicted_and_unknown_blocks(tmp_path):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    # A scanned block that has not been verdicted yet is fine to stash, and a
    # block id absent from the session inventory is not a duplicate either.
    assert store.precheck_slide_scan(session.number, block_id) is True
    assert store.precheck_slide_scan(session.number, "99999999") is True

    kinds = [event.kind for event in store.events(session.number)]
    assert "duplicate_slide_scan" not in kinds


def test_crashing_slide_preprocessor_fails_closed_to_review(tmp_path):
    def crashing_slide_preprocessor(_img):
        raise ValueError("segmentation blew up on a degenerate crop")

    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=crashing_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert outcome.verdict == "REVIEW"
    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "REVIEW"
    assert "segmentation blew up" in row["decision_reason"]


def test_qc_write_failure_leaves_verdict_uncommitted_for_retry(tmp_path, monkeypatch):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)

    def failing_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ProcessingStore, "_write_claim_qc", staticmethod(failing_write))

    with pytest.raises(OSError):
        store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)

    # The verdict must not be committed when QC evidence could not be
    # written, so the claim remains retriable instead of permanently
    # orphaned behind the "already processed" duplicate guard.
    assert store.get_set(session.number, block_id)["verdict"] is None

    monkeypatch.undo()
    outcome = store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)
    assert outcome.accepted
    assert outcome.verdict == "PASS"


def test_contact_sheet_false_write_leaves_verdict_uncommitted(tmp_path, monkeypatch):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)
    monkeypatch.setattr(contact_sheet.cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(OSError, match="contact sheet"):
        store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)

    assert store.get_set(session.number, block_id)["verdict"] is None


def test_decisions_export_failure_is_retried_for_same_slide(tmp_path, monkeypatch):
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)
    original_refresh = store._refresh_decisions_export
    attempts = 0

    def fail_once(session_identity):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated decisions export failure")
        return original_refresh(session_identity)

    monkeypatch.setattr(store, "_refresh_decisions_export", fail_once)
    with pytest.raises(OSError, match="decisions export failure"):
        store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)

    retried = store.resolve_claim(
        session.number, block_id, "slide_capture_1", slide_path
    )

    assert retried.accepted
    assert retried.verdict == "PASS"
    assert "PASS" in (session.directory / "decisions.csv").read_text("utf-8")
    assert attempts == 2


def test_restart_repairs_export_after_verdict_commit(tmp_path, monkeypatch):
    root = tmp_path / "processing"
    store = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    slide_path = _capture(tmp_path / "slide.png", 120)

    def fail_export(_session_identity):
        raise OSError("simulated decisions export failure")

    monkeypatch.setattr(store, "_refresh_decisions_export", fail_export)
    with pytest.raises(OSError, match="decisions export failure"):
        store.resolve_claim(session.number, block_id, "slide_capture_1", slide_path)
    assert store.get_set(session.number, block_id)["verdict"] == "PASS"
    assert not (session.directory / "decisions.csv").exists()

    ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )

    export = (session.directory / "decisions.csv").read_text("utf-8")
    assert block_id in export
    assert "PASS" in export


def _valid_slide_result(block_id):
    return select_slide_identity((
        DecodeCandidate(
            "zxing", "QRCode", "raw", f"12080_{block_id}_01_HE"
        ),
    ))


def test_pi_outbox_slide_filename_includes_decoded_block_id(tmp_path):
    source = _capture(tmp_path / "slide.png", 120)
    result = _valid_slide_result("51151378")

    capture = PiOutbox(tmp_path / "outbox").publish_slide(
        source, STARTED_AT, result=result, duration_ms=10.0,
    )

    checksum = session_workflow_module._sha256(source)
    expected = f"slide_51151378_20260702T180506Z_{checksum[:12]}"
    assert capture.capture_id == expected
    assert capture.path.name == f"{expected}.png"


def test_processing_store_slide_filename_includes_decoded_block_id(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)
    source = _capture(tmp_path / "slide.png", 120)

    capture_id = store.record_slide_capture(
        session.number, source, captured_at=STARTED_AT,
        result=_valid_slide_result("51151378"), duration_ms=10.0,
    )

    checksum = session_workflow_module._sha256(source)
    expected = f"slide_51151378_20260702T180506Z_{checksum[:12]}"
    assert capture_id == expected
    assert Path(store.get_slide_capture(session.number, capture_id)["capture_path"]).name == (
        f"{expected}.png"
    )


@pytest.mark.parametrize("publisher", ("outbox", "store"))
def test_unresolved_slide_filename_does_not_claim_a_block_id(tmp_path, publisher):
    source = _capture(tmp_path / "slide.png", 120)
    result = select_slide_identity(())
    checksum = session_workflow_module._sha256(source)
    expected = f"slide_unresolved_20260702T180506Z_{checksum[:12]}"

    if publisher == "outbox":
        capture_id = PiOutbox(tmp_path / "outbox").publish_slide(
            source, STARTED_AT, result=result, duration_ms=10.0,
        ).capture_id
    else:
        store = ProcessingStore(tmp_path / "processing")
        session = store.start_session(started_at=STARTED_AT)
        store.begin_block_drain(session.number)
        assert store.try_enter_slides(session.number)
        capture_id = store.record_slide_capture(
            session.number, source, captured_at=STARTED_AT,
            result=result, duration_ms=10.0,
        )

    assert capture_id == expected


def test_missing_block_review_is_in_decisions_export(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    store.record_slide_capture(
        session.number,
        _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT,
        result=_valid_slide_result("99999999"),
        duration_ms=10.0,
    )

    export = (session.directory / "decisions.csv").read_text("utf-8")
    assert "99999999" in export
    assert "REVIEW" in export
    assert "block id not found" in export


def test_restart_recovers_valid_slide_committed_before_verdict(tmp_path, monkeypatch):
    root = tmp_path / "processing"
    store = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)

    def crash_before_verdict(*_args, **_kwargs):
        raise RuntimeError("simulated crash before verdict")

    monkeypatch.setattr(store, "resolve_claim", crash_before_verdict)
    with pytest.raises(RuntimeError, match="crash before verdict"):
        store.record_slide_capture(
            session.number,
            _capture(tmp_path / "slide.png", 120),
            captured_at=STARTED_AT,
            result=_valid_slide_result(block_id),
            duration_ms=10.0,
        )
    assert store.get_set(session.number, block_id)["verdict"] is None

    restarted = ProcessingStore(
        root,
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )

    assert restarted.get_set(session.number, block_id)["verdict"] == "PASS"
    assert "PASS" in (session.directory / "decisions.csv").read_text("utf-8")


def test_summary_reports_processed_pass_review_counts_from_durable_state(tmp_path):
    def preprocessor(capture_path):
        if "62262489" in capture_path.name:
            raise ValueError("no tissue found")
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}

    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=preprocessor,
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    passed_block = _evaluable_block(store, session, tmp_path, block_id="51151378")
    store.resolve_claim(
        session.number, passed_block, "slide_capture_pass",
        _capture(tmp_path / "slide_pass.png", 120),
    )
    reviewed_block = "62262489"
    assert store.scan_block(session.number, reviewed_block).accepted
    failed_capture = PiOutbox(tmp_path / "outbox_reviewed").publish_block(
        _capture(tmp_path / "62262489_block.png", 60), reviewed_block, STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=failed_capture.capture_id,
        block_id=failed_capture.block_id, checksum=failed_capture.checksum,
        body=failed_capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    store.dismiss_block(session.number, reviewed_block, reason="operator confirmed unusable")
    store.resolve_claim(
        session.number, reviewed_block, "slide_capture_review",
        _capture(tmp_path / "slide_review.png", 121),
    )

    summary = store.summarize(session)

    assert summary.session_number == session.number
    assert summary.started_at == STARTED_AT
    assert summary.sets_processed == 2
    assert summary.pass_count == 1
    assert summary.review_count == 1


def test_summary_exposes_pending_and_failed_block_detail(tmp_path):
    held = Event()
    release = Event()

    def preprocessor(capture_path):
        if "22222222" in capture_path.name:
            raise ValueError("no tissue found")
        held.set()
        assert release.wait(timeout=5), "test did not release preprocessing"
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}

    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    # A block whose preprocessing failed and awaits recapture/dismissal.
    assert store.scan_block(session.number, "22222222").accepted
    failed_capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "failed.png", 50), "22222222", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=failed_capture.capture_id,
        block_id=failed_capture.block_id, checksum=failed_capture.checksum,
        body=failed_capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    # A block still mid-preprocessing: pending work, not yet decided.
    assert store.scan_block(session.number, "11111111").accepted
    pending_capture = PiOutbox(tmp_path / "outbox2").publish_block(
        _capture(tmp_path / "pending.png", 51), "11111111", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=pending_capture.capture_id,
        block_id=pending_capture.block_id, checksum=pending_capture.checksum,
        body=pending_capture.path.read_bytes(),
    )
    assert held.wait(timeout=5)

    summary = store.summarize(session)
    release.set()
    store.wait_for_jobs()

    assert summary.sets_processed == 0
    assert summary.pass_count == 0
    assert summary.review_count == 0
    assert summary.pending_blocks == ("11111111",)
    assert [warning.block_id for warning in summary.block_failures] == ["22222222"]


def test_summary_exposes_missing_slides_once_blocks_are_ready(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    summary = store.summarize(session)

    assert summary.missing_slides == (block_id,)


def test_summary_exposes_blocks_captured_even_when_nothing_scored(tmp_path):
    # #188: sets_processed only counts verdicted pairs, so a session with real
    # blocks captured and zero slides scored yet must still surface a nonzero
    # "there is work here" signal via blocks_captured.
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "11111111").accepted
    assert store.scan_block(session.number, "22222222").accepted

    summary = store.summarize(session)

    assert summary.sets_processed == 0
    assert summary.blocks_captured == 2


def test_summary_excludes_dismissed_blocks_from_blocks_captured(tmp_path):
    # #188: a dismissed (resolved-unusable) block is not work left to resume --
    # an all-dismissed session must not misreport blocks_captured > 0.
    def preprocessor(capture_path):
        raise ValueError("no tissue found")

    store = ProcessingStore(tmp_path / "processing", preprocessor=preprocessor)
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "22222222").accepted
    failed_capture = PiOutbox(tmp_path / "outbox").publish_block(
        _capture(tmp_path / "failed.png", 50), "22222222", STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=failed_capture.capture_id,
        block_id=failed_capture.block_id, checksum=failed_capture.checksum,
        body=failed_capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    store.dismiss_block(session.number, "22222222", reason="operator confirmed unusable")

    summary = store.summarize(session)

    assert summary.blocks_captured == 0


def test_summary_exposes_skipped_slide_decodes(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    failed = select_slide_identity(())
    workflow = SessionWorkflow(
        session=session, store=store, outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(store), camera=FakePhaseCamera(),
        slide_decoder=lambda image: failed,
    )
    workflow.finish_blocks()
    _prepare_slide_backlight(workflow)

    workflow.capture_slide(
        _capture(tmp_path / "unreadable.png", 120), captured_at=STARTED_AT
    )

    summary = store.summarize(session)

    assert len(summary.skipped_decodes) == 1


def test_workflow_summary_includes_outbox_pending_uploads(tmp_path):
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox,
        transport=ToggleTransport(store), camera=FakePhaseCamera(),
    )
    assert workflow.scan_block("51151378").accepted
    outbox.publish_block(_capture(tmp_path / "block.png", 80), "51151378", STARTED_AT)

    summary = workflow.summarize()

    assert summary.pending_uploads == ("capture_000001_block_51151378_20260702T180506Z",)


def _session_ready_for_slides(tmp_path, *, preprocessor=None, slide_preprocessor=None):
    """Start a session with no blocks and drain it straight into slide mode."""
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=preprocessor or FastPreprocessor(),
        slide_preprocessor=slide_preprocessor or _identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
        camera=FakePhaseCamera(),
    )
    drained = workflow.finish_blocks()
    assert drained.phase == "slides"
    return store, session, outbox, transport, workflow


def test_end_session_requires_explicit_confirmation(tmp_path):
    _, _, _, _, workflow = _session_ready_for_slides(tmp_path)

    with pytest.raises(ValueError, match="confirmation"):
        workflow.end_session(confirm=False)

    assert workflow.snapshot().phase == "slides"


def test_end_session_blocks_further_slide_capture(tmp_path):
    _, _, _, _, workflow = _session_ready_for_slides(tmp_path)

    workflow.end_session(confirm=True)

    with pytest.raises(RuntimeError, match="slide actions require slide mode"):
        workflow.capture_slide(
            _capture(tmp_path / "late_slide.png", 90), captured_at=STARTED_AT
        )


def test_full_lifecycle_finalizes_with_verified_exports_and_pi_cleanup(tmp_path):
    block_id = "51151378"
    store2 = ProcessingStore(
        tmp_path / "processing2", preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session2 = store2.start_session(started_at=STARTED_AT)
    outbox2 = PiOutbox(tmp_path / "outbox2")
    transport2 = ToggleTransport(store2)
    workflow2 = SessionWorkflow(
        session=session2, store=store2, outbox=outbox2, transport=transport2,
        camera=FakePhaseCamera(),
    )
    transport2.connected = True
    receipt = workflow2.capture_block(
        block_id, _capture(tmp_path / "block.png", 80), captured_at=STARTED_AT
    )
    assert receipt is not None and receipt.acknowledged
    store2.wait_for_jobs()
    assert workflow2.finish_blocks().phase == "slides"

    outcome_slide = _capture(tmp_path / "slide.png", 120)
    claim = store2.resolve_claim(
        session2.number, block_id, "slide_capture_1", outcome_slide
    )
    assert claim.verdict == "PASS"

    summary = workflow2.summarize()
    assert summary.sets_processed == 1
    assert summary.pass_count == 1

    final = workflow2.end_session(confirm=True)

    assert final.phase == "finalized"
    session_dir = session2.directory
    manifest = (session_dir / "manifest.csv").read_text(encoding="utf-8")
    assert block_id in manifest and "PASS" in manifest
    decisions = (session_dir / "decisions.csv").read_text(encoding="utf-8")
    assert block_id in decisions and "PASS" in decisions
    session_meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_meta["phase"] == "finalized"
    assert session_meta["finalized_at"]
    # The one durable Pi capture was acknowledged, so it must be gone after
    # finalization confirms the processing computer's copy is safe.
    assert outbox2.entries() == ()
    kinds = [event.kind for event in workflow2.events()]
    assert "session_finalized" in kinds


def test_finalization_survives_restart_and_completes_once_resolved(tmp_path):
    block_id = "51151378"
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
        camera=FakePhaseCamera(),
    )
    transport.connected = True
    workflow.capture_block(
        block_id, _capture(tmp_path / "block.png", 80), captured_at=STARTED_AT
    )
    store.wait_for_jobs()
    assert workflow.finish_blocks().phase == "slides"
    row = store.get_set(session.number, block_id)
    original_bytes = Path(row["capture_path"]).read_bytes()
    Path(row["capture_path"]).write_bytes(b"corrupted before the first finalize attempt")

    interrupted = workflow.end_session(confirm=True)
    assert interrupted.phase == "finalizing"

    # A restart reconstructs the still-unfinalized session instead of losing
    # track of the fact that finalization was already requested.
    restarted = SessionWorkflow(
        session=store.resume_session(session.number),
        store=store, outbox=PiOutbox(tmp_path / "outbox"), transport=transport,
        camera=FakePhaseCamera(),
    )
    assert restarted.snapshot().phase == "finalizing"

    # The underlying storage issue is fixed, so the next poll can complete.
    Path(row["capture_path"]).write_bytes(original_bytes)
    finished = restarted.poll_finalization()

    assert finished.phase == "finalized"


def test_finalization_rejects_corrupted_capture_and_stays_resumable(tmp_path):
    block_id = "51151378"
    # Build one durable, acknowledged block capture the normal way, then
    # corrupt the durably stored file to simulate storage-layer bit rot.
    store2 = ProcessingStore(tmp_path / "processing_corrupt", preprocessor=FastPreprocessor())
    session2 = store2.start_session(started_at=STARTED_AT)
    outbox2 = PiOutbox(tmp_path / "outbox_corrupt")
    transport2 = ToggleTransport(store2)
    workflow2 = SessionWorkflow(
        session=session2, store=store2, outbox=outbox2, transport=transport2,
        camera=FakePhaseCamera(),
    )
    transport2.connected = True
    workflow2.capture_block(
        block_id, _capture(tmp_path / "corrupt_block.png", 80), captured_at=STARTED_AT
    )
    store2.wait_for_jobs()
    assert workflow2.finish_blocks().phase == "slides"
    row = store2.get_set(session2.number, block_id)
    Path(row["capture_path"]).write_bytes(b"not a real capture anymore")

    result = workflow2.end_session(confirm=True)

    assert result.phase == "finalizing"
    kinds = [event.kind for event in workflow2.events()]
    assert "finalization_verification_failed" in kinds
    assert outbox2.entries() != ()  # nothing was deleted on a failed verification


def test_export_failure_keeps_finalization_recoverable_before_pi_cleanup(
    tmp_path, monkeypatch
):
    block_id = "51151378"
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
        camera=FakePhaseCamera(),
    )
    transport.connected = True
    workflow.capture_block(
        block_id, _capture(tmp_path / "block.png", 80), captured_at=STARTED_AT
    )
    store.wait_for_jobs()
    assert workflow.finish_blocks().phase == "slides"

    original_export = store._refresh_manifest_export
    attempts = 0

    def fail_once(session_identity):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated export failure")
        original_export(session_identity)

    monkeypatch.setattr(store, "_refresh_manifest_export", fail_once)

    with pytest.raises(OSError, match="simulated export failure"):
        workflow.end_session(confirm=True)

    assert workflow.snapshot().phase == "finalizing"
    assert outbox.entries() != ()

    finished = workflow.poll_finalization()

    assert finished.phase == "finalized"
    assert outbox.entries() == ()
    assert attempts == 2


def test_finalization_failure_reason_is_durable_across_restart(tmp_path):
    block_id = "51151378"
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
        camera=FakePhaseCamera(),
    )
    transport.connected = True
    workflow.capture_block(
        block_id, _capture(tmp_path / "block.png", 80), captured_at=STARTED_AT
    )
    store.wait_for_jobs()
    assert workflow.finish_blocks().phase == "slides"
    row = store.get_set(session.number, block_id)
    Path(row["capture_path"]).write_bytes(b"not a real capture anymore")

    workflow.end_session(confirm=True)
    original_error = store.summarize(session).finalization_error
    assert original_error is not None
    assert str(row["capture_path"]) in original_error

    # A process restart must not lose the reason a prior attempt failed; it
    # is recovery information, not just an in-memory event.
    session_meta = json.loads(
        (session.directory / "session.json").read_text(encoding="utf-8")
    )
    assert session_meta["last_finalization_error"] == original_error

    restarted_store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    resumed_summary = restarted_store.summarize(
        restarted_store.resume_session(session.number)
    )
    assert resumed_summary.finalization_error == original_error


def test_outbox_delete_acknowledged_leaves_pending_entries_untouched(tmp_path):
    outbox = PiOutbox(tmp_path / "outbox")
    acked = outbox.publish_block(
        _capture(tmp_path / "acked.png", 80), "51151378", STARTED_AT
    )
    pending = outbox.publish_block(
        _capture(tmp_path / "pending.png", 81), "22222222", STARTED_AT
    )
    outbox.acknowledge(UploadReceipt(acked.capture_id, True, acked.checksum))

    deleted = outbox.delete_acknowledged()

    assert deleted == (acked.capture_id,)
    remaining = {entry.capture_id for entry in outbox.entries()}
    assert remaining == {pending.capture_id}
    assert not acked.path.exists()
    assert pending.path.exists()

    # Idempotent: re-running after the files are already gone changes nothing.
    assert outbox.delete_acknowledged() == ()


def test_existing_session_database_is_migrated_for_finalization(tmp_path):
    root = tmp_path / "processing"
    root.mkdir()
    database = root / "sessions.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            """CREATE TABLE sessions (
               session_number INTEGER PRIMARY KEY AUTOINCREMENT,
               started_at TEXT NOT NULL UNIQUE,
               phase TEXT NOT NULL,
               slide_recovery_state TEXT NOT NULL DEFAULT 'waiting'
               )"""
        )
        db.execute(
            "INSERT INTO sessions(started_at, phase) VALUES (?, 'slides')",
            (STARTED_AT.isoformat(),),
        )
    session_dir = root / "session_000001_20260702T180506Z"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps({
            "session_number": 1,
            "started_at": STARTED_AT.isoformat(),
            "phase": "slides",
        }),
        encoding="utf-8",
    )

    store = ProcessingStore(root)
    session = store.resume_session(1)

    assert store.summarize(session).finalization_error is None
    store.begin_finalization(1)
    assert store.prepare_finalization(1) is True
    store.complete_finalization(1)
    assert store.snapshot(session).phase == "finalized"


def _rewrite_outbox_metadata(outbox, capture, update):
    metadata_path = outbox.directory / f"{capture.capture_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(update)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


@pytest.mark.parametrize(
    "update",
    (
        {"capture_id": "capture_999999_block_51151378_20260702T180506Z"},
        {"block_id": "22222222"},
        {"path": "capture_999999_block_51151378_20260702T180506Z.png"},
    ),
)
def test_outbox_rejects_mismatched_capture_metadata(tmp_path, update):
    outbox = PiOutbox(tmp_path / "outbox")
    capture = outbox.publish_block(
        _capture(tmp_path / "capture.png", 80), "51151378", STARTED_AT
    )
    _rewrite_outbox_metadata(outbox, capture, update)

    assert outbox.entries() == ()
    assert outbox.invalid_entries() == (capture.capture_id,)
    assert outbox.delete_acknowledged() == ()
    assert capture.path.exists()


def test_outbox_traversal_metadata_cannot_delete_outside_file(tmp_path):
    outbox = PiOutbox(tmp_path / "outbox")
    capture = outbox.publish_block(
        _capture(tmp_path / "capture.png", 80), "51151378", STARTED_AT
    )
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"must survive")
    metadata_path = outbox.directory / f"{capture.capture_id}.json"
    metadata_path.write_text(
        json.dumps({
            "capture_id": capture.capture_id,
            "block_id": "51151378",
            "checksum": session_workflow_module._sha256(victim),
            "captured_at": STARTED_AT.isoformat(),
            "path": "../victim.bin",
            "state": "acknowledged",
        }),
        encoding="utf-8",
    )

    assert outbox.entries() == ()
    assert outbox.invalid_entries() == (capture.capture_id,)
    assert outbox.delete_acknowledged() == ()
    assert victim.read_bytes() == b"must survive"


def test_malformed_outbox_metadata_blocks_finalization_and_remains_visible(tmp_path):
    _, _, outbox, _, workflow = _session_ready_for_slides(tmp_path)
    malformed = outbox.directory / "capture_broken.json"
    malformed.write_text("{not-json", encoding="utf-8")

    result = workflow.end_session(confirm=True)

    assert result.phase == "finalizing"
    assert result.pending_transfers == 1
    assert workflow.summarize().pending_uploads == ("capture_broken",)
    assert malformed.exists()


def test_cleanup_pending_resumes_acknowledged_cleanup_after_restart(tmp_path):
    store, session, outbox, transport, workflow = _session_ready_for_slides(tmp_path)
    capture = outbox.publish_block(
        _capture(tmp_path / "orphan.png", 80), "51151378", STARTED_AT
    )
    outbox.acknowledge(UploadReceipt(capture.capture_id, True, capture.checksum))
    store.begin_finalization(session.number)
    assert store.prepare_finalization(session.number) is True
    assert store.snapshot(session).phase == "cleanup_pending"

    restarted_store = ProcessingStore(tmp_path / "processing")
    resumed_session = restarted_store.resume_session(session.number)
    repaired_on_resume = json.loads(
        (session.directory / "session.json").read_text("utf-8")
    )
    assert repaired_on_resume["phase"] == "cleanup_pending"
    restarted = SessionWorkflow(
        session=resumed_session,
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=transport,
        camera=FakePhaseCamera(),
    )
    result = restarted.poll_finalization(replay=False)

    assert result.phase == "finalized"
    assert not capture.path.exists()


def test_deleted_cleanup_is_idempotently_resumed_before_final_commit(
    tmp_path, monkeypatch
):
    store, session, outbox, _, workflow = _session_ready_for_slides(tmp_path)
    capture = outbox.publish_block(
        _capture(tmp_path / "orphan.png", 80), "51151378", STARTED_AT
    )
    outbox.acknowledge(UploadReceipt(capture.capture_id, True, capture.checksum))
    original_complete = store.complete_finalization
    attempts = 0

    def fail_once(session_number):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated final phase commit failure")
        return original_complete(session_number)

    monkeypatch.setattr(store, "complete_finalization", fail_once)

    with pytest.raises(OSError, match="final phase commit failure"):
        workflow.end_session(confirm=True)

    assert workflow.snapshot().phase == "cleanup_pending"
    assert not capture.path.exists()
    assert store.summarize(session).finalization_error is not None

    result = workflow.poll_finalization(replay=False)

    assert result.phase == "finalized"
    assert attempts == 2


def test_cleanup_failure_is_durable_and_restartable(tmp_path, monkeypatch):
    store, session, outbox, _, workflow = _session_ready_for_slides(tmp_path)
    capture = outbox.publish_block(
        _capture(tmp_path / "orphan.png", 80), "51151378", STARTED_AT
    )
    outbox.acknowledge(UploadReceipt(capture.capture_id, True, capture.checksum))
    original_delete = outbox.delete_acknowledged

    def fail_cleanup():
        raise OSError("simulated Pi cleanup failure")

    monkeypatch.setattr(outbox, "delete_acknowledged", fail_cleanup)
    with pytest.raises(OSError, match="Pi cleanup failure"):
        workflow.end_session(confirm=True)

    assert workflow.snapshot().phase == "cleanup_pending"
    error = store.summarize(session).finalization_error
    assert error is not None and "Pi cleanup failure" in error
    metadata = json.loads((session.directory / "session.json").read_text("utf-8"))
    assert metadata["last_finalization_error"] == error

    monkeypatch.setattr(outbox, "delete_acknowledged", original_delete)
    assert workflow.poll_finalization(replay=False).phase == "finalized"


def test_metadata_reconciles_from_sqlite_after_transition_crash(tmp_path, monkeypatch):
    store, session, _, _, workflow = _session_ready_for_slides(tmp_path)
    original_reconcile = store.reconcile_session_metadata
    calls = 0

    def fail_cleanup_pending_sync(session_number):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated metadata sync crash")
        return original_reconcile(session_number)

    monkeypatch.setattr(store, "reconcile_session_metadata", fail_cleanup_pending_sync)

    with pytest.raises(OSError, match="metadata sync crash"):
        workflow.end_session(confirm=True)

    assert store.snapshot(session).phase == "cleanup_pending"
    metadata = json.loads((session.directory / "session.json").read_text("utf-8"))
    assert metadata["phase"] == "finalizing"
    assert metadata["phase"] != "finalized"

    restarted_store = ProcessingStore(tmp_path / "processing")
    resumed_session = restarted_store.resume_session(session.number)
    repaired_on_resume = json.loads(
        (session.directory / "session.json").read_text("utf-8")
    )
    assert repaired_on_resume["phase"] == "cleanup_pending"
    restarted = SessionWorkflow(
        session=resumed_session,
        store=restarted_store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=ToggleTransport(restarted_store),
        camera=FakePhaseCamera(),
    )

    assert restarted.poll_finalization(replay=False).phase == "finalized"
    repaired = json.loads((session.directory / "session.json").read_text("utf-8"))
    assert repaired["phase"] == "finalized"


def test_export_and_metadata_failures_are_recorded_durably(tmp_path, monkeypatch):
    store, session, _, _, workflow = _session_ready_for_slides(tmp_path)

    def fail_export(session_identity):
        raise OSError("simulated export failure")

    monkeypatch.setattr(store, "_refresh_manifest_export", fail_export)
    with pytest.raises(OSError, match="export failure"):
        workflow.end_session(confirm=True)
    assert "export failure" in store.summarize(session).finalization_error

    monkeypatch.undo()
    original_reconcile = store.reconcile_session_metadata

    def fail_metadata(session_number):
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(store, "reconcile_session_metadata", fail_metadata)
    with pytest.raises(OSError, match="metadata failure"):
        workflow.poll_finalization(replay=False)
    with store._connect() as db:
        error = db.execute(
            "SELECT last_finalization_error FROM sessions WHERE session_number=?",
            (session.number,),
        ).fetchone()["last_finalization_error"]
    assert "metadata failure" in error
    monkeypatch.setattr(store, "reconcile_session_metadata", original_reconcile)


# ---------------------------------------------------------------------------
# Work-order-scoped N^2 retrieval mode (#149, ADR 0009).
#
# A work order is the operator's explicit start/finish capture bracket. Every
# block scan and slide capture stamped inside the bracket belongs to that work
# order; finishing it dispatches an async all-pairs scoring job so the kiosk
# stays responsive while a prior order's results are still computing.
# ---------------------------------------------------------------------------


class StubWorkOrderScorer:
    """Deterministic, synchronous stand-in for the production N^2 scorer.

    Ignores the (block_results, slide_results) mappings it is handed and
    instead returns whatever ``scores_by_slide`` has been populated with for
    each slide capture id -- tests populate this dict *after* construction,
    once the real capture id is known, but *before* ``finish_work_order`` is
    called.
    """

    def __init__(self):
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}
        self.calls = 0

    def __call__(self, block_results, slide_results):
        self.calls += 1
        return {
            capture_id: dict(self.scores_by_slide.get(capture_id, {}))
            for capture_id in slide_results
        }


class GatedWorkOrderScorer:
    """Blocks inside the scoring call until the test releases it.

    Proves that ``finish_work_order`` returns before scoring completes (the
    job runs on the executor) and that a second work order can be started
    while the first one is still scoring.
    """

    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}

    def __call__(self, block_results, slide_results):
        self.started.set()
        assert self.release.wait(timeout=5), "test did not release scoring"
        return {
            capture_id: dict(self.scores_by_slide.get(capture_id, {}))
            for capture_id in slide_results
        }


def _drain_to_slides(store, session):
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)


def test_work_order_lifecycle_survives_restart(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")

    assert store.get_set(session.number, block_id)["work_order_id"] == work_order_id
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "capturing"

    restarted = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )

    restarted_wo = restarted.get_work_order(session.number, work_order_id)
    assert restarted_wo["lifecycle_state"] == "capturing"
    restarted_row = restarted.get_set(session.number, block_id)
    assert restarted_row["work_order_id"] == work_order_id


def test_recover_work_orders_reenqueues_scoring_after_restart(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer, recover_jobs=False,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    capture_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[capture_id] = {block_id: 0.95}

    # Model a crash after the finalized->scoring transition commits but
    # before the executor actually starts the job.
    finished_id = store.finish_work_order(session.number, start_job=False)
    assert finished_id == work_order_id
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "scoring"

    restarted_scorer = StubWorkOrderScorer()
    restarted_scorer.scores_by_slide[capture_id] = {block_id: 0.95}
    restarted = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=restarted_scorer,
    )
    restarted.wait_for_jobs()

    wo_after = restarted.get_work_order(session.number, work_order_id)
    assert wo_after["lifecycle_state"] == "results_ready"
    assert restarted_scorer.calls == 1
    assert Path(wo_after["verdict_csv_path"]).is_file()


@pytest.mark.parametrize("session_mode", [SessionMode.NORMAL])
def test_finish_work_order_scoring_transition_unaffected_by_session_mode_gate(
    tmp_path, session_mode
):
    """#269 acceptance criterion / negative control: NORMAL remains unchanged
    by the new `sessions.session_mode` gate on
    `finish_work_order`'s N-by-N scoring transition -- the
    finalized->scoring commit and job submission still happen exactly as
    before, and a crash between them is still recovered by
    `_recover_work_orders`. Mirrors
    `test_recover_work_orders_reenqueues_scoring_after_restart` above,
    for the non-retrieval mode."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer, recover_jobs=False,
    )
    session = store.start_session(
        started_at=STARTED_AT, session_mode=session_mode.value
    )
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    capture_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[capture_id] = {block_id: 0.95}

    # Model a crash after the finalized->scoring transition commits but
    # before the executor actually starts the job.
    finished_id = store.finish_work_order(session.number, start_job=False)
    assert finished_id == work_order_id
    wo = store.get_work_order(session.number, work_order_id)
    assert wo["lifecycle_state"] == "scoring"

    restarted_scorer = StubWorkOrderScorer()
    restarted_scorer.scores_by_slide[capture_id] = {block_id: 0.95}
    restarted = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=restarted_scorer,
    )
    restarted.wait_for_jobs()

    wo_after = restarted.get_work_order(session.number, work_order_id)
    assert wo_after["lifecycle_state"] == "results_ready"
    assert restarted_scorer.calls == 1


def test_session_mode_fails_closed_for_an_unknown_session(tmp_path):
    """#269 review LOW finding 5d: `_session_mode` used to default to
    'normal' for an unknown session number. That is a safety-gate
    failure-OPEN bug: `finish_work_order` is the sole caller, and uses this
    value to decide whether a work order may run the full N-by-N scoring
    path -- silently guessing "not Hybrid" for a session it cannot find is
    exactly backwards. It must fail closed (raise) instead."""
    store = ProcessingStore(tmp_path / "processing")

    with pytest.raises(ValueError, match="unknown session"):
        store._session_mode(999999)


@pytest.mark.parametrize("session_mode", [SessionMode.NORMAL, SessionMode.OPEN_RETRIEVAL])
def test_session_mode_still_correct_for_every_real_session_mode(tmp_path, session_mode):
    """#269 review 5d follow-up: the fail-closed change must not disturb the
    one real caller (`finish_work_order`, exercised end-to-end elsewhere in
    this file) for any legitimate, existing session."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode.value)

    assert store._session_mode(session.number) == session_mode.value


def test_start_and_finish_work_order_stamp_captures_between_the_bracket(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)

    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )

    assert store.get_set(session.number, block_id)["work_order_id"] == work_order_id
    assert (
        store.get_slide_capture(session.number, slide_id)["work_order_id"]
        == work_order_id
    )

    kinds = [event.kind for event in store.events(session.number)]
    assert "work_order_started" in kinds


def test_finishing_one_work_order_lets_a_new_one_start_and_capture_independently(
    tmp_path,
):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)

    work_order_a = store.start_work_order(session.number)
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9}
    store.finish_work_order(session.number)

    work_order_b = store.start_work_order(session.number)
    assert work_order_b != work_order_a
    block_2 = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    slide_b = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_b.png", 121),
        captured_at=STARTED_AT, result=_valid_slide_result(block_2), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_b] = {block_2: 0.9}
    store.finish_work_order(session.number)
    store.wait_for_jobs()

    assert (
        store.get_slide_capture(session.number, slide_a)["work_order_id"]
        == work_order_a
    )
    assert (
        store.get_slide_capture(session.number, slide_b)["work_order_id"]
        == work_order_b
    )
    assert store.get_work_order(session.number, work_order_a)["lifecycle_state"] == (
        "results_ready"
    )
    assert store.get_work_order(session.number, work_order_b)["lifecycle_state"] == (
        "results_ready"
    )


def test_starting_the_next_work_order_returns_the_session_to_block_capture(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)

    first = store.start_work_order(session.number)
    _drain_to_slides(store, session)
    store.finish_work_order(session.number)
    assert store.snapshot(session).phase == "slides"

    second = store.start_work_order(session.number)

    assert second != first
    assert store.snapshot(session).phase == "blocks"


def test_start_work_order_is_idempotent_returns_existing_capturing_row(tmp_path):
    """#155: a double-tap on START WORK ORDER (retry, or a Pi restart landing
    mid-request) must not orphan an extra capturing row -- the second call
    reuses the SAME work order id the first call opened, mirroring the
    SELECT-before-INSERT guard `finish_work_order` already applies."""
    root = tmp_path / "processing"
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)

    first = store.start_work_order(session.number)
    second = store.start_work_order(session.number)

    assert second == first
    rows = [
        row for row in store.events(session.number) if row.kind == "work_order_started"
    ]
    assert len(rows) == 1  # the second call did not re-emit the started event


def test_open_work_order_id_reflects_capturing_state_then_none_after_finish(tmp_path):
    """#155: the boot-seed read helper -- returns the open bracket's id while
    one is `capturing`, and None once it has been finished (or before any
    work order has ever started)."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)

    assert store.open_work_order_id(session.number) is None

    work_order_id = store.start_work_order(session.number)
    assert store.open_work_order_id(session.number) == work_order_id

    store.finish_work_order(session.number)
    assert store.open_work_order_id(session.number) is None


def test_captures_under_an_open_work_order_have_no_verdict_until_scored(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    capture_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )

    assert store.get_set(session.number, block_id)["verdict"] is None
    assert store.get_slide_capture(session.number, capture_id)["verdict"] is None

    scorer.scores_by_slide[capture_id] = {block_id: 0.95}
    store.finish_work_order(session.number)
    store.wait_for_jobs()

    assert store.get_set(session.number, block_id)["verdict"] == "PASS"


def test_finish_work_order_dispatches_async_job_and_a_new_order_can_start_while_scoring(
    tmp_path,
):
    root = tmp_path / "processing"
    scorer = GatedWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_a = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_id: 0.9}

    finished_id = store.finish_work_order(session.number)
    assert finished_id == work_order_a
    assert scorer.started.wait(timeout=2), "finish_work_order must not block the kiosk"

    # A new work order can start and accept a capture while the first order is
    # still scoring in the background.
    work_order_b = store.start_work_order(session.number)
    assert work_order_b != work_order_a
    assert (
        store.get_work_order(session.number, work_order_a)["lifecycle_state"]
        == "scoring"
    )

    scorer.release.set()
    store.wait_for_jobs()

    assert (
        store.get_work_order(session.number, work_order_a)["lifecycle_state"]
        == "results_ready"
    )


def test_finish_work_order_persists_verdict_csv_with_ranked_candidates(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_b = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_a), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_a: 0.92, block_b: 0.40}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    wo = store.get_work_order(session.number, work_order_id)
    csv_text = Path(wo["verdict_csv_path"]).read_text(encoding="utf-8")
    assert slide_id in csv_text
    assert block_a in csv_text
    assert "PASS" in csv_text
    assert "Claimed block clearly scored highest." in csv_text

    row = store.get_set(session.number, block_a)
    assert row["verdict"] == "PASS"
    assert row["score"] == pytest.approx(0.92)

    kinds = [event.kind for event in store.events(session.number)]
    assert "work_order_results_ready" in kinds


def test_default_work_order_scorer_builds_candidate_scores_from_production_scorer():
    block_results = {
        "51151378": PreparedSpecimen(
            role="block", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
        "62626262": PreparedSpecimen(
            role="block", mask=np.zeros((8, 8), dtype=np.uint8), roi_ok=True,
        ),
    }
    slide_results = {
        "slide_capture_1": PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
        ),
    }

    scores = default_work_order_scorer(block_results, slide_results).scores

    assert scores["slide_capture_1"]["51151378"] == pytest.approx(1.0)
    other = scores["slide_capture_1"]["62626262"]
    assert other is None or other < scores["slide_capture_1"]["51151378"]


def _synthetic_specimen(role, seed):
    """A small deterministic mask so each item's cache differs from the others."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    size = 10 + (seed % 5) * 6
    top_left = (seed * 3) % 20
    cv2.rectangle(
        mask, (top_left, top_left), (top_left + size, top_left + size), 255, -1,
    )
    return PreparedSpecimen(role=role, mask=mask, roi_ok=True)


def test_work_order_scorer_hoists_normalization_cache_per_item_not_per_pair():
    """ADR 0011 / issue #158: `default_work_order_scorer` must build
    `build_locked_score_cache` once per item (M+K), not once per pair (2*M*K),
    and the per-pair scores it produces must be unchanged (within 1e-9) from
    the old per-pair-rebuilding path (`decide_claim` with no `scorer=`
    override, i.e. `score_pair_result_routed`)."""
    block_results = {
        f"block_{i}": _synthetic_specimen("block", i) for i in range(3)
    }
    slide_results = {
        f"slide_{i}": _synthetic_specimen("slide", i + 10) for i in range(4)
    }
    num_blocks = len(block_results)
    num_slides = len(slide_results)

    import session.processing_store as processing_store_module

    real_build_cache = processing_store_module.build_locked_score_cache
    call_counts = {"default": 0, "cached": 0}

    def counting_build_default(specimen):
        call_counts["default"] += 1
        return real_build_cache(specimen)

    def counting_build_cached(specimen):
        call_counts["cached"] += 1
        return real_build_cache(specimen)

    # (a) default path: the old per-pair loop body, `decide_claim` with no
    # `scorer=` override so it falls through to `score_pair_result_routed`,
    # which rebuilds both caches on every single pair.
    with patch("verify.scorer.build_locked_score_cache", side_effect=counting_build_default):
        default_scores: dict[str, dict[str, float | None]] = {}
        for slide_id, slide_result in slide_results.items():
            row: dict[str, float | None] = {}
            for block_id, block_result in block_results.items():
                decision = decide_claim(
                    f"{slide_id}:{block_id}", block_result, slide_result,
                )
                row[block_id] = decision.score
            default_scores[slide_id] = row

    # (b) new cache-injected path via the production entry point.
    with patch(
        "session.processing_store.build_locked_score_cache",
        side_effect=counting_build_cached,
    ):
        cached_scores = default_work_order_scorer(block_results, slide_results).scores

    assert call_counts["default"] == 2 * num_blocks * num_slides
    assert call_counts["cached"] == num_blocks + num_slides

    for slide_id in slide_results:
        for block_id in block_results:
            assert cached_scores[slide_id][block_id] == pytest.approx(
                default_scores[slide_id][block_id], abs=1e-9
            )


def test_work_order_scorer_skips_cache_and_scorer_for_failed_block():
    valid_slide = _synthetic_specimen("slide", 10)
    block_results = {
        "failed_block": PreparationFailure(role="block", reason="segmentation failed"),
    }
    slide_results = {"slide_1": valid_slide}
    cached_specimens = []

    def record_cache_build(specimen):
        cached_specimens.append(specimen)
        return object()

    with patch(
        "session.processing_store.build_locked_score_cache", side_effect=record_cache_build
    ), patch(
        "session.processing_store.score_routed_caches",
        side_effect=AssertionError("failed pair must not reach cache lookup"),
    ):
        scores = default_work_order_scorer(block_results, slide_results).scores

    assert scores == {"slide_1": {"failed_block": None}}
    assert cached_specimens == [valid_slide]


def test_work_order_scorer_skips_cache_and_scorer_for_failed_slide():
    valid_block = _synthetic_specimen("block", 1)
    block_results = {"block_1": valid_block}
    slide_results = {
        "failed_slide": PreparationFailure(role="slide", reason="label detection failed"),
    }
    cached_specimens = []

    def record_cache_build(specimen):
        cached_specimens.append(specimen)
        return object()

    with patch(
        "session.processing_store.build_locked_score_cache", side_effect=record_cache_build
    ), patch(
        "session.processing_store.score_routed_caches",
        side_effect=AssertionError("failed pair must not reach cache lookup"),
    ):
        scores = default_work_order_scorer(block_results, slide_results).scores

    assert scores == {"failed_slide": {"block_1": None}}
    assert cached_specimens == [valid_block]


def test_finish_work_order_single_block_order_falls_back_to_threshold(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_id: 0.90}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"
    wo = store.get_work_order(session.number, work_order_id)
    csv_text = Path(wo["verdict_csv_path"]).read_text(encoding="utf-8")
    assert "unverified" in csv_text


def test_finish_work_order_claimed_block_absent_from_order_reviews(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result("99999999"), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_id: 0.90}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    slide_row = store.get_slide_capture(session.number, slide_id)
    assert slide_row["verdict"] == "REVIEW"
    wo = store.get_work_order(session.number, work_order_id)
    csv_text = Path(wo["verdict_csv_path"]).read_text(encoding="utf-8")
    assert "Claimed block 99999999 not in this order." in csv_text


def test_finish_work_order_claimed_pair_gate_failure_reviews_fail_closed(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_id: None}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "REVIEW"
    wo = store.get_work_order(session.number, work_order_id)
    csv_text = Path(wo["verdict_csv_path"]).read_text(encoding="utf-8")
    assert "Preparation failed." in csv_text


def test_processing_store_accepts_injectable_work_order_scorer_kwarg(tmp_path):
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    store.start_work_order(session.number)
    block_id = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_id), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_id: 0.9}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    assert scorer.calls == 1
    row = store.get_set(session.number, block_id)
    assert row["verdict"] == "PASS"


def test_workflow_start_and_finish_work_order_delegate_to_the_store(tmp_path):
    store, session, _outbox, _transport, workflow = _session_ready_for_slides(tmp_path)

    work_order_id = workflow.start_work_order()
    assert store.get_work_order(session.number, work_order_id)["lifecycle_state"] == (
        "capturing"
    )

    finished_id = workflow.finish_work_order()
    assert finished_id == work_order_id


def test_results_status_returns_work_orders_and_rows_from_list_results_ready_work_orders(
    tmp_path,
):
    """#153: the kiosk's live-results seam. ``results_status()`` must wrap
    the already-existing ``list_results_ready_work_orders`` into the
    ``{"work_orders": (...), "rows": [...]}`` shape ``test_kiosk_relay``'s
    ``_ResultsFake.results_status()`` test double exercises against
    ``KioskRelay._results_status()``'s degrading getattr+callable read --
    proving the REAL ``SessionWorkflow`` handle resolves the same seam
    production reads, with zero mocking of the underlying store."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    outbox = PiOutbox(tmp_path / "outbox")
    transport = ToggleTransport(store)
    workflow = SessionWorkflow(
        session=session, store=store, outbox=outbox, transport=transport,
    )

    work_order_a = workflow.start_work_order()
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    decoy_1 = _evaluable_block(store, session, tmp_path, block_id="12121212")
    _drain_to_slides(store, session)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9, decoy_1: 0.1}
    workflow.finish_work_order()
    store.wait_for_jobs()

    work_order_b = workflow.start_work_order()
    block_2 = _evaluable_block(store, session, tmp_path, block_id="62626262")
    decoy_2 = _evaluable_block(store, session, tmp_path, block_id="73737373")
    _drain_to_slides(store, session)
    slide_b = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_b.png", 121),
        captured_at=STARTED_AT, result=_valid_slide_result(block_2), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_b] = {block_2: 0.05, decoy_2: 0.02}
    workflow.finish_work_order()
    store.wait_for_jobs()

    status = workflow.results_status()

    # Distinct work orders, sorted -- not the raw per-row duplication.
    assert status["work_orders"] == tuple(sorted({work_order_a, work_order_b}))
    by_capture = {row["capture_id"]: row for row in status["rows"]}
    assert set(by_capture) == {slide_a, slide_b}
    assert by_capture[slide_a]["verdict"] == "PASS"
    assert by_capture[slide_a]["work_order_id"] == work_order_a
    assert by_capture[slide_b]["verdict"] == "PASS"
    assert by_capture[slide_b]["work_order_id"] == work_order_b


class _FakeResultsRowsStore:
    """Minimal store double for ``results_status()``'s #252 mode-routing
    seam: only the handful of methods ``SessionWorkflow.__init__``/
    ``results_status`` touch, so these tests don't need a full block/slide
    capture lifecycle through a real ``ProcessingStore`` (which has no
    ``list_hybrid_results`` yet -- that lands separately) just to prove
    which store method gets called for which session mode."""

    def __init__(
        self, *, ready_rows=(), hybrid_rows=(), retrieval_rows=(),
        hybrid_error=None, retrieval_error=None,
        profile_rows=(), profile_error=None,
    ):
        self.ready_calls = 0
        self.hybrid_calls = 0
        self.retrieval_calls = 0
        # #257: recorded, never asserted-away -- a real assertion that
        # pause_capture/resume_capture never wait on or cancel jobs.
        self.wait_for_jobs_calls = 0
        # #258: recorded so hybrid_profile_status's own mode-gate/degrade
        # tests can prove list_hybrid_profile_rows was (or was not) reached,
        # mirroring hybrid_calls above.
        self.profile_calls = 0
        self._ready_rows = tuple(ready_rows)
        self._hybrid_rows = tuple(hybrid_rows)
        self._retrieval_rows = tuple(retrieval_rows)
        self._hybrid_error = hybrid_error
        self._retrieval_error = retrieval_error
        self._profile_rows = tuple(profile_rows)
        self._profile_error = profile_error

    def slide_recovery_state(self, session_number):
        return None

    def list_results_ready_work_orders(self, session_number):
        self.ready_calls += 1
        return self._ready_rows

    def list_hybrid_results(self, session_number):
        self.hybrid_calls += 1
        if self._hybrid_error is not None:
            raise self._hybrid_error
        return self._hybrid_rows

    def list_retrieval_results(self, session_number):
        self.retrieval_calls += 1
        if self._retrieval_error is not None:
            raise self._retrieval_error
        return self._retrieval_rows

    def list_hybrid_profile_rows(self, session_number):
        self.profile_calls += 1
        if self._profile_error is not None:
            raise self._profile_error
        return self._profile_rows

    def wait_for_jobs(self):
        self.wait_for_jobs_calls += 1


def _results_workflow(tmp_path, *, session_mode, store):
    """A ``SessionWorkflow`` wired to a fake store, for #252 results_status()
    mode-routing tests. ``framing_calibration`` is passed explicitly (rather
    than defaulted from ``store.root``) since the fake store has no ``root``.
    ``outbox``/``transport`` are never touched by ``results_status()`` or
    ``SessionWorkflow.__init__`` -- real, harmless instances stand in (a real
    ``PiOutbox`` directory and an ``HttpCaptureClient`` whose base_url is
    never dialed) so this stays type-correct rather than passing ``None``
    against their non-Optional parameter types."""
    identity = SessionIdentity(
        number=99,
        started_at=STARTED_AT,
        directory=tmp_path,
        session_mode=session_mode.value,
    )
    return SessionWorkflow(
        session=identity,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://unused.invalid"),
        framing_calibration=FramingCalibrationStore(tmp_path / "framing.json"),
        session_mode=session_mode,
    )


def test_pause_capture_and_resume_capture_toggle_pure_state_only(tmp_path):
    """#257: pause_capture/resume_capture must be pure in-process flags -- no
    store I/O, no job wait/cancel -- so background scoring and queued jobs
    are completely unaffected while Results is open. `wait_for_jobs_calls`
    is a real, non-vacuous assertion (not merely "no exception raised")."""
    store = _FakeResultsRowsStore()
    workflow = _results_workflow(tmp_path, session_mode=SessionMode.HYBRID, store=store)

    assert workflow.capture_paused is False

    workflow.pause_capture()
    assert workflow.capture_paused is True

    workflow.resume_capture()
    assert workflow.capture_paused is False

    # Neither call touched the store at all: no results reads, no job wait.
    assert store.ready_calls == 0
    assert store.hybrid_calls == 0
    assert store.wait_for_jobs_calls == 0


@pytest.mark.parametrize("mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_results_status_sources_hybrid_rows_from_list_hybrid_results(tmp_path, mode):
    """#252: HYBRID/HYBRID_SHADOW must read the dedicated Hybrid projection,
    not the NORMAL/OPEN_RETRIEVAL batch-atomic reveal -- and must do so from
    ``self.session_mode`` (the durable, resume-carried value), never from
    whether rows/artifacts happen to exist."""
    hybrid_rows = (
        {"capture_id": "cap-1", "block_id": "b1", "verdict": "PENDING",
         "claim_reason": "", "claim_score": None, "work_order_id": 5,
         "work_order": "12080"},
        {"capture_id": "cap-2", "block_id": "b2", "verdict": "PASS",
         "claim_reason": "", "claim_score": 0.9, "work_order_id": 5,
         "work_order": "12080"},
    )
    store = _FakeResultsRowsStore(hybrid_rows=hybrid_rows)
    workflow = _results_workflow(tmp_path, session_mode=mode, store=store)

    status = workflow.results_status()

    assert store.hybrid_calls == 1
    assert store.ready_calls == 0
    assert status["work_orders"] == (5,)
    by_capture = {row["capture_id"]: row for row in status["rows"]}
    assert by_capture["cap-1"]["verdict"] == "PENDING"
    assert by_capture["cap-2"]["verdict"] == "PASS"


def test_results_status_open_retrieval_sources_live_retrieval_rows(tmp_path):
    retrieval_rows = (
        {"capture_id": "cap-open", "block_id": "b1", "verdict": "PENDING",
         "claim_reason": "", "claim_score": None, "work_order_id": 9,
         "work_order": "12090"},
    )
    store = _FakeResultsRowsStore(retrieval_rows=retrieval_rows)
    workflow = _results_workflow(
        tmp_path, session_mode=SessionMode.OPEN_RETRIEVAL, store=store
    )

    status = workflow.results_status()

    assert store.retrieval_calls == 1
    assert store.ready_calls == 0
    assert store.hybrid_calls == 0
    assert status["rows"][0]["verdict"] == "PENDING"


def test_results_status_normal_keeps_results_ready_projection(tmp_path):
    """Normal mode remains batch-atomic and never reads retrieval jobs."""
    ready_rows = (
        {"capture_id": "cap-3", "block_id": "b3", "verdict": "PASS",
         "claim_reason": "", "claim_score": 0.9, "work_order_id": 9,
         "work_order": "12090"},
    )
    store = _FakeResultsRowsStore(ready_rows=ready_rows)
    workflow = _results_workflow(tmp_path, session_mode=SessionMode.NORMAL, store=store)

    status = workflow.results_status()

    assert store.ready_calls == 1
    assert store.hybrid_calls == 0
    assert store.retrieval_calls == 0
    assert [row["capture_id"] for row in status["rows"]] == ["cap-3"]


@pytest.mark.parametrize("mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_results_status_degrades_to_empty_rows_when_list_hybrid_results_raises(
    tmp_path, mode,
):
    """Blast-radius guard (#252): ``results_status()`` runs on the kiosk's
    poll path, which reaches ``_camera_loop``'s bare ``except Exception`` if
    anything escapes. Even though the store contract promises
    ``list_hybrid_results`` never raises on its own poll path, this proves
    the degrade holds if it ever does anyway."""
    store = _FakeResultsRowsStore(hybrid_error=RuntimeError("boom"))
    workflow = _results_workflow(tmp_path, session_mode=mode, store=store)

    status = workflow.results_status()

    assert status == {"work_orders": (), "rows": []}
    assert store.hybrid_calls == 1


# --------------------------------------------------------------------------
# #258: hybrid_profile_status -- the ONE shared source both the kiosk relay
# and the console read for --profile Hybrid queue/timing display. Mirrors
# results_status's own mode-gate + broad-except tests immediately above.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_hybrid_profile_status_projects_pending_and_finished_rows(tmp_path, mode):
    """A PENDING row surfaces its current stage + elapsed time (computed from
    the controlled ``now_ns`` passed in, never a real clock read); a
    finished row surfaces its total time plus the full five-stage
    breakdown; the queue count counts only the PENDING row."""
    raw_rows = (
        {"capture_id": "cap-1", "block_id": "11111111", "job_state": "scoring",
         "verdict": None, "stage": "heuristic_selection",
         "queued_ns": 4_000_000_000, "total_ms": None, "stage_ms_json": None,
         "shadow": 0},
        {"capture_id": "cap-2", "block_id": "22222222", "job_state": "complete",
         "verdict": "PASS", "stage": None, "queued_ns": 0, "total_ms": 4200,
         "stage_ms_json": json.dumps({
             "queue_wait": 100, "preparation": 200,
             "heuristic_selection": 300, "accurate_scoring": 3500,
             "artifact_write": 100,
         }),
         "shadow": 0},
    )
    store = _FakeResultsRowsStore(profile_rows=raw_rows)
    workflow = _results_workflow(tmp_path, session_mode=mode, store=store)

    status = workflow.hybrid_profile_status(now_ns=5_000_000_000)

    assert store.profile_calls == 1
    assert status["queue_count"] == 1
    by_capture = {row.capture_id: row for row in status["rows"]}
    pending = by_capture["cap-1"]
    assert pending.state == "PENDING"
    assert pending.stage == "heuristic_selection"
    assert pending.elapsed_ms == 1000  # (5_000_000_000 - 4_000_000_000) ns -> ms
    finished = by_capture["cap-2"]
    assert finished.state == "PASS"
    assert finished.total_ms == 4200
    assert set(finished.stage_ms) == {
        "queue_wait", "preparation", "heuristic_selection",
        "accurate_scoring", "artifact_write",
    }


@pytest.mark.parametrize("mode", [SessionMode.NORMAL, SessionMode.OPEN_RETRIEVAL])
def test_hybrid_profile_status_normal_and_open_retrieval_never_call_the_store(
    tmp_path, mode,
):
    """Non-vacuous control (#258): must fail if the explicit mode gate in
    ``hybrid_profile_status`` is dropped -- rows are supplied here precisely
    so "empty queue" is proven by the gate, not merely by the store having
    nothing to return."""
    raw_rows = (
        {"capture_id": "cap-1", "block_id": "11111111", "job_state": "scoring",
         "verdict": None, "stage": "preparation", "queued_ns": 0,
         "total_ms": None, "stage_ms_json": None, "shadow": 0},
    )
    store = _FakeResultsRowsStore(profile_rows=raw_rows)
    workflow = _results_workflow(tmp_path, session_mode=mode, store=store)

    status = workflow.hybrid_profile_status(now_ns=1_000_000_000)

    assert status == {"queue_count": 0, "rows": ()}
    assert store.profile_calls == 0


@pytest.mark.parametrize("mode", [SessionMode.HYBRID, SessionMode.HYBRID_SHADOW])
def test_hybrid_profile_status_degrades_to_empty_queue_when_the_store_raises(
    tmp_path, mode,
):
    """Blast-radius guard (#258): this runs on the same kiosk poll path as
    ``results_status``/``list_hybrid_results`` -- even though
    ``list_hybrid_profile_rows`` promises never to raise on its own poll
    path, this proves the degrade holds if it ever does anyway."""
    store = _FakeResultsRowsStore(profile_error=RuntimeError("boom"))
    workflow = _results_workflow(tmp_path, session_mode=mode, store=store)

    status = workflow.hybrid_profile_status(now_ns=1_000_000_000)

    assert status == {"queue_count": 0, "rows": ()}
    assert store.profile_calls == 1


def test_hybrid_profile_status_shadow_row_carries_the_shadow_flag(tmp_path):
    """Hybrid Shadow's complete-pool cost must be distinguishable from
    pruned real Hybrid timing -- ``shadow=True`` rides straight through from
    the store's ``profile_shadow`` column."""
    raw_rows = (
        {"capture_id": "cap-9", "block_id": "99999999", "job_state": "complete",
         "verdict": "PASS", "stage": None, "queued_ns": 0, "total_ms": 999,
         "stage_ms_json": None, "shadow": 1},
    )
    store = _FakeResultsRowsStore(profile_rows=raw_rows)
    workflow = _results_workflow(
        tmp_path, session_mode=SessionMode.HYBRID_SHADOW, store=store
    )

    status = workflow.hybrid_profile_status(now_ns=1_000_000_000)

    assert status["rows"][0].shadow is True


def test_console_and_screen_render_the_same_numbers_from_one_shared_source(tmp_path):
    """#258: proves console (``format_profile_console``) and touchscreen
    (``profile_screen_fields``) can never disagree, because both read the
    exact same ``ProfileRow`` tuple ``hybrid_profile_status`` returns -- one
    shared formatter, not two independently maintained renderers."""
    from session.profile_report import format_profile_console, profile_screen_fields

    raw_rows = (
        {"capture_id": "cap-1", "block_id": "11111111", "job_state": "scoring",
         "verdict": None, "stage": "accurate_scoring",
         "queued_ns": 3_000_000_000, "total_ms": None, "stage_ms_json": None,
         "shadow": 0},
    )
    store = _FakeResultsRowsStore(profile_rows=raw_rows)
    workflow = _results_workflow(
        tmp_path, session_mode=SessionMode.HYBRID, store=store
    )

    status = workflow.hybrid_profile_status(now_ns=5_000_000_000)
    screen = profile_screen_fields(status["rows"], queue_count=status["queue_count"])
    console = format_profile_console(status["rows"], queue_count=status["queue_count"])

    elapsed_ms = screen["rows"][0]["elapsed_ms"]
    assert elapsed_ms == 2000
    assert f"elapsed={elapsed_ms}ms" in console
    assert f"queue={status['queue_count']}" in console
    assert screen["queue_count"] == status["queue_count"] == 1


def test_list_results_ready_work_orders_returns_per_slide_verdict_rows(tmp_path):
    """#150: the kiosk results table's data source -- every slide-capture row
    from every ``results_ready`` work order in the session, carrying the
    verdict/claim_score/claim_reason/block_id/capture_id columns #149's
    ``_finalize_claim`` already populates. No new schema, no CSV re-parsing."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)

    work_order_a = store.start_work_order(session.number)
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    decoy_1 = _evaluable_block(store, session, tmp_path, block_id="12121212")
    _drain_to_slides(store, session)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9, decoy_1: 0.1}
    store.finish_work_order(session.number)
    store.wait_for_jobs()

    work_order_b = store.start_work_order(session.number)
    block_2 = _evaluable_block(store, session, tmp_path, block_id="62626262")
    decoy_2 = _evaluable_block(store, session, tmp_path, block_id="73737373")
    _drain_to_slides(store, session)
    slide_b = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_b.png", 121),
        captured_at=STARTED_AT, result=_valid_slide_result(block_2), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_b] = {block_2: 0.05, decoy_2: 0.02}
    store.finish_work_order(session.number)
    store.wait_for_jobs()

    assert store.get_work_order(session.number, work_order_a)["lifecycle_state"] == (
        "results_ready"
    )
    assert store.get_work_order(session.number, work_order_b)["lifecycle_state"] == (
        "results_ready"
    )

    rows = store.list_results_ready_work_orders(session.number)
    by_capture = {row["capture_id"]: row for row in rows}

    # Every results-ready work order in the session is represented -- not a
    # single-order picker -- so any order finished this session is viewable.
    assert set(by_capture) == {slide_a, slide_b}
    assert by_capture[slide_a]["block_id"] == block_1
    assert by_capture[slide_a]["work_order_id"] == work_order_a
    assert by_capture[slide_a]["lab_work_order"] == "12080"
    assert by_capture[slide_a]["work_order"] == "12080"  # legacy alias
    assert by_capture[slide_a]["verdict"] == "PASS"
    assert by_capture[slide_b]["block_id"] == block_2
    assert by_capture[slide_b]["work_order_id"] == work_order_b
    assert by_capture[slide_b]["verdict"] == "PASS"
    assert by_capture[slide_b]["claim_reason"] == "Claimed block clearly scored highest."
    assert by_capture[slide_b]["claim_score"] == pytest.approx(0.05)


def test_list_results_ready_work_orders_proxy_matches_local_store_over_rpc(tmp_path):
    """#149/#150/#153: the kiosk reads results through RemoteProcessingStore
    over /rpc, so the proxied per-slide verdict rows must equal the local
    store's rows -- not silently degrade to empty (the AttributeError gap that
    made results unreachable from the Pi)."""
    from store.remote import RemoteProcessingStore

    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_2 = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)

    store.start_work_order(session.number)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9, block_2: 0.1}
    store.finish_work_order(session.number)
    store.wait_for_jobs()

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        remote_rows = proxy.list_results_ready_work_orders(session.number)

    assert remote_rows == store.list_results_ready_work_orders(session.number)
    assert {row["capture_id"] for row in remote_rows} == {slide_a}
    assert remote_rows[0]["verdict"] == "PASS"


def test_list_results_ready_work_orders_excludes_orders_still_capturing_or_scoring(
    tmp_path,
):
    """A work order that has not reached ``results_ready`` yet must not leak
    unfinished/half-scored rows into the results table."""
    root = tmp_path / "processing"
    scorer = GatedWorkOrderScorer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_1 = _evaluable_block(store, session, tmp_path, block_id="51151378")
    _drain_to_slides(store, session)

    store.start_work_order(session.number)
    slide_a = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide_a.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_1), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_a] = {block_1: 0.9}
    store.finish_work_order(session.number)
    try:
        assert scorer.started.wait(timeout=5), "scoring job never started"

        rows = store.list_results_ready_work_orders(session.number)
        assert rows == ()
    finally:
        scorer.release.set()
        store.wait_for_jobs()


# ---------------------------------------------------------------------------
# #151: contact-sheet rendering for flagged pairs on REVIEW verdicts.
#
# ``ProcessingStore`` accepts an injectable ``contact_sheet_renderer`` (same
# seam shape as ``work_order_scorer``), and ``_score_work_order`` uses it to
# render one sheet per flagged pair (top match, and the claim when it
# differs) into a durable directory persisted onto
# ``work_orders.contact_sheet_dir``.
# ---------------------------------------------------------------------------


class StubContactSheetRenderer:
    """Records every call instead of touching cv2; writes a tiny placeholder
    file at ``output_path`` so existence assertions have something real to
    check, mirroring ``StubWorkOrderScorer``'s "record + fake the effect"
    shape."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        output_path = kwargs.get("output_path")
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"stub-contact-sheet")


def test_processing_store_accepts_injectable_contact_sheet_renderer_kwarg(tmp_path):
    """A stub renderer injected at construction must be called instead of
    the real cv2-backed ``write_contact_sheet`` free function."""
    renderer = StubContactSheetRenderer()
    store = ProcessingStore(
        tmp_path / "processing",
        preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        contact_sheet_renderer=renderer,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _evaluable_block(store, session, tmp_path)

    outcome = store.resolve_claim(
        session.number, block_id, "slide_capture_1",
        _capture(tmp_path / "slide.png", 120),
    )

    assert outcome.accepted
    assert renderer.calls, "the injected renderer must be called, not the free function"


def test_score_work_order_writes_a_sheet_for_both_top_match_and_claim_when_they_differ(
    tmp_path,
):
    """On a claim-disagreement REVIEW, a sheet must exist for the top match
    AND the claimed block -- two distinct files, not one."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    renderer = StubContactSheetRenderer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
        contact_sheet_renderer=renderer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_b = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    # Claimed block is block_b, but block_a scores highest -> disagreement.
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_b), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_a: 0.9, block_b: 0.2}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    row = store.get_set(session.number, block_b)
    assert row["verdict"] == "REVIEW"

    wo = store.get_work_order(session.number, work_order_id)
    sheets_dir = Path(wo["contact_sheet_dir"])
    top_match_sheet = sheets_dir / f"{slide_id}__{block_a}.png"
    claim_sheet = sheets_dir / f"{slide_id}__{block_b}.png"
    assert top_match_sheet.is_file()
    assert claim_sheet.is_file()
    assert top_match_sheet != claim_sheet

    # AC3: each rendered sheet must self-identify which physical block it
    # shows -- the block id must reach the header via role_label, not just
    # live in the filename.
    role_labels_by_block = {
        call["output_path"].name: call["role_label"]
        for call in renderer.calls
        if call.get("role_label") is not None
    }
    assert block_a in role_labels_by_block[top_match_sheet.name]
    assert block_b in role_labels_by_block[claim_sheet.name]


def test_finish_work_order_persists_contact_sheets_alongside_verdict_csv(tmp_path):
    """The contact-sheet directory must be persisted onto
    ``work_orders.contact_sheet_dir``, sibling to the verdict CSV's
    directory, and contain the rendered PNG(s) for audit."""
    root = tmp_path / "processing"
    scorer = StubWorkOrderScorer()
    renderer = StubContactSheetRenderer()
    store = ProcessingStore(
        root, preprocessor=FastPreprocessor(),
        slide_preprocessor=_identical_mask_slide_preprocessor,
        work_order_scorer=scorer,
        contact_sheet_renderer=renderer,
    )
    session = store.start_session(started_at=STARTED_AT)
    work_order_id = store.start_work_order(session.number)
    block_a = _evaluable_block(store, session, tmp_path, block_id="51151378")
    block_b = _evaluable_block(store, session, tmp_path, block_id="62626262")
    _drain_to_slides(store, session)
    slide_id = store.record_slide_capture(
        session.number, _capture(tmp_path / "slide.png", 120),
        captured_at=STARTED_AT, result=_valid_slide_result(block_b), duration_ms=10.0,
    )
    scorer.scores_by_slide[slide_id] = {block_a: 0.9, block_b: 0.2}

    store.finish_work_order(session.number)
    store.wait_for_jobs()

    wo = store.get_work_order(session.number, work_order_id)
    assert wo["contact_sheet_dir"], "contact_sheet_dir must be populated on results_ready"
    sheets_dir = Path(wo["contact_sheet_dir"])
    assert sheets_dir.is_dir()
    png_files = list(sheets_dir.glob("*.png"))
    assert len(png_files) >= 1

    verdict_csv_dir = Path(wo["verdict_csv_path"]).parent
    assert sheets_dir.parent == verdict_csv_dir, (
        "contact sheets must live sibling to the verdict CSV's directory"
    )


# ---------------------------------------------------------------------------
# --profile (#168): ProcessingStore.record_profile_capture + RPC whitelist
# ---------------------------------------------------------------------------

_PROFILE_FIELDS = {
    "camera_capture_ms": 100,
    "publish_ms": 20,
    "consumer_ms": 30,
    "session_accept_ms": 5,
    "total_capture_ms": 155,
    "final_file_size_bytes": 123456,
    "capture_mode": "block",
}


def _profile_summary_csv(root: Path, session_number: int) -> Path:
    matches = list(root.glob(f"session_{session_number:06d}_*/profile_summary.csv"))
    assert matches, f"no profile_summary.csv under {root} for session {session_number}"
    return matches[0]


def test_record_profile_capture_appends_row_with_one_column_per_stage(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)

    store.record_profile_capture(session.number, "capture_000001", _PROFILE_FIELDS)

    csv_path = _profile_summary_csv(root, session.number)
    text = csv_path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for key in _PROFILE_FIELDS:
        assert key in header
    assert "capture_id" in header
    for value in ("100", "20", "30", "5", "155"):
        assert value in text


def test_record_profile_capture_joins_by_capture_identity(tmp_path):
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)

    store.record_profile_capture(session.number, "capture_000001", _PROFILE_FIELDS)
    store.record_profile_capture(
        session.number,
        "capture_000002",
        {**_PROFILE_FIELDS, "total_capture_ms": 999},
    )

    csv_path = _profile_summary_csv(root, session.number)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    data_lines = lines[1:]
    assert len(data_lines) == 2
    assert any("capture_000001" in line for line in data_lines)
    assert any("capture_000002" in line and "999" in line for line in data_lines)


def test_record_profile_capture_includes_consumer_split_columns(tmp_path):
    """#171: the per-capture summary table gains decode/outbox/send columns
    once the consumer stage reports its sub-durations."""
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    fields = {
        **_PROFILE_FIELDS,
        "consumer_decode_ms": 12,
        "consumer_outbox_ms": 7,
        "consumer_send_ms": 41,
    }

    store.record_profile_capture(session.number, "capture_000001", fields)

    csv_path = _profile_summary_csv(root, session.number)
    text = csv_path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for key in ("consumer_decode_ms", "consumer_outbox_ms", "consumer_send_ms"):
        assert key in header
    for value in ("12", "7", "41"):
        assert value in text


def test_record_profile_capture_includes_settling_columns(tmp_path):
    """#172: the per-capture summary table gains settling duration/reset/
    peak-motion columns once the capture carries a settling summary."""
    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)
    fields = {
        **_PROFILE_FIELDS,
        "settling_duration_ms": 850,
        "settling_resets": 2,
        "settling_max_motion": 0.031,
    }

    store.record_profile_capture(session.number, "capture_000001", fields)

    csv_path = _profile_summary_csv(root, session.number)
    text = csv_path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for key in ("settling_duration_ms", "settling_resets", "settling_max_motion"):
        assert key in header
    for value in ("850", "2", "0.031"):
        assert value in text


def test_remote_processing_store_record_profile_capture_reaches_rpc_whitelist(tmp_path):
    """#168, per memory `pi-store-remote-rpc`: a new ProcessingStore method the
    Pi calls must reach both `_RPC_METHODS` and the `RemoteProcessingStore`
    proxy, or the Pi gets a live AttributeError despite green in-process
    tests."""
    from store.remote import RemoteProcessingStore

    root = tmp_path / "processing"
    store = ProcessingStore(root)
    session = store.start_session(started_at=STARTED_AT)

    with LoopbackCaptureReceiver(store) as receiver:
        proxy = RemoteProcessingStore(receiver.url)
        proxy.record_profile_capture(session.number, "capture_000001", _PROFILE_FIELDS)

    csv_path = _profile_summary_csv(root, session.number)
    assert "capture_000001" in csv_path.read_text(encoding="utf-8")
