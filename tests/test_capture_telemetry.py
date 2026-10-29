"""Status/event and calibration telemetry tests."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from capture_runtime import CaptureController, camera_settings_for
from capture_session import CaptureSession, CaptureState, FrameResult, SessionConfig
from capture_storage import CaptureStore
from capture_telemetry import (
    MotionCurveWriter,
    SettlingObserver,
    SettlingSummary,
    StatusReporter,
    TelemetryWriter,
)
from constants import MOTION_CURVE_FLUSH_INTERVAL_S


def _empty_frame() -> np.ndarray:
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _calibrate_block(session: CaptureSession) -> None:
    session.confirm_empty()
    for index in range(session.config.baseline_frames):
        session.accept_frame(_empty_frame(), now=index * 0.1)
    assert session.state is CaptureState.WAITING_FOR_SCAN


def test_normal_status_is_one_hz_but_events_are_immediate():
    messages = []
    session = CaptureSession(SessionConfig(baseline_frames=1), mode="block")
    _calibrate_block(session)
    reporter = StatusReporter(messages.append, interval=1.0)

    reporter.publish(now=0.0, session=session, frame=FrameResult())
    initial_count = len(messages)
    reporter.publish(now=0.2, session=session, frame=FrameResult())
    assert len(messages) == initial_count

    session.submit_scan("bad")
    reporter.publish(now=0.3, session=session, frame=FrameResult())
    assert len(messages) == initial_count + 1
    assert "rejected" in messages[-1].lower()

    reporter.publish(now=1.0, session=session, frame=FrameResult())
    assert len(messages) == initial_count + 2
    assert "WAITING_FOR_SCAN" in messages[-1]


def test_baseline_scan_retry_error_and_saved_events_are_ui_independent(tmp_path):
    session = CaptureSession(SessionConfig(baseline_frames=1), mode="block")
    kinds = [event.kind for event in session.drain_events()]
    assert "baseline_instruction" in kinds

    _calibrate_block(session)
    kinds = [event.kind for event in session.drain_events()]
    assert "baseline_ready" in kinds
    assert "scan_required" in kinds

    session.submit_scan("51151378")
    kinds = [event.kind for event in session.drain_events()]
    assert "scan_accepted" in kinds


def test_per_frame_csv_contains_calibration_contract(tmp_path):
    path = tmp_path / "telemetry.csv"
    session = CaptureSession(SessionConfig(baseline_frames=1), mode="block")
    _calibrate_block(session)
    session.submit_scan("51151378")
    frame = FrameResult(
        presence_score=0.12,
        motion_score=0.03,
        stable_elapsed=0.4,
    )

    with TelemetryWriter(path) as writer:
        writer.record(
            timestamp="2026-07-01T19:30:45Z",
            session=session,
            frame=frame,
            settings=camera_settings_for("block"),
            capture_path="captures/capture_000001_block.png",
            event="settling",
        )

    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["timestamp"] == "2026-07-01T19:30:45Z"
    assert row["mode"] == "block"
    assert row["state"] == session.state.name
    assert row["presence_score"] == "0.12"
    assert row["motion_score"] == "0.03"
    assert row["stable_elapsed"] == "0.4"
    assert row["exposure_us"] == "33333"
    assert row["pending_block_id"] == "51151378"
    assert row["capture_path"].endswith("capture_000001_block.png")
    assert row["event"] == "settling"


def test_capture_metadata_is_complete_for_both_roles(tmp_path):
    required = {
        "roi",
        "presence_threshold",
        "motion_threshold",
        "baseline_frames",
        "stable_duration",
        "removal_duration",
        "exposure_us",
        "analogue_gain",
        "colour_gains",
        "contrast",
        "sharpness",
        "denoise",
        "still_dimensions",
    }

    class UnusedCamera:
        pass

    for role in ("block", "slide"):
        session = CaptureSession(mode=role)
        controller = CaptureController(
            session=session,
            camera=UnusedCamera(),
            store=CaptureStore(tmp_path / role),
            working_dir=tmp_path / f"work-{role}",
        )
        assert required <= controller._capture_metadata().keys()


# ---------------------------------------------------------------------------
# Settling observer: re-derives settling wait / reset-count / peak motion
# from the same (now, state, FrameResult) triple StatusReporter.publish
# consumes, rather than reading CaptureSession's private _stable_since.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "captured_at", "expected"),
    [
        # Quiet entry; one two-frame motion run is one reset.
        (
            [(0.0, 0.01), (0.3, 0.05), (0.5, 0.04), (1.1, 0.01)],
            1.5,
            (1500, 1, 0.05),
        ),
        # Entry frame motion (the presence transition) is not a reset; two
        # separate two-frame runs are two resets.
        (
            [(0.0, 0.5), (0.2, 0.03), (0.5, 0.04), (0.7, 0.0), (0.9, 0.03), (1.1, 0.04)],
            1.7,
            (1700, 2, 0.5),
        ),
    ],
    ids=["quiet-entry", "flicker-entry"],
)
def test_settling_observer_computes_wait_resets_and_max_motion(
    frames, captured_at, expected
):
    observer = SettlingObserver(threshold=0.02)

    for now, motion in frames:
        assert observer.observe(
            now=now, state=CaptureState.SETTLING,
            frame=FrameResult(motion_score=motion),
        ) is None
    summary = observer.observe(
        now=captured_at,
        state=CaptureState.CAPTURE_REQUESTED,
        frame=FrameResult(motion_score=0.0, capture_requested=True),
    )

    assert summary == SettlingSummary(*expected)


def test_settling_observer_discards_an_abandoned_settling_window_that_never_captures():
    """A SETTLING window abandoned before CAPTURE_REQUESTED is discarded."""

    observer = SettlingObserver(threshold=0.02)

    observer.observe(
        now=0.0, state=CaptureState.SETTLING, frame=FrameResult(motion_score=0.01)
    )
    observer.observe(
        now=0.4, state=CaptureState.SETTLING, frame=FrameResult(motion_score=0.03)
    )
    # Specimen removed before stabilizing -- back to EMPTY, no capture.
    abandoned = observer.observe(
        now=0.6, state=CaptureState.EMPTY, frame=FrameResult()
    )

    assert abandoned is None

    # A fresh settling window afterwards starts its own reset count from
    # zero -- the abandoned window's reset must not leak forward.
    observer.observe(
        now=1.0, state=CaptureState.SETTLING, frame=FrameResult(motion_score=0.01)
    )
    summary = observer.observe(
        now=2.0,
        state=CaptureState.CAPTURE_REQUESTED,
        frame=FrameResult(motion_score=0.01, capture_requested=True),
    )

    assert summary.resets == 0
    assert summary.duration_ms == 1000


# ---------------------------------------------------------------------------
# Motion curve writer: buffers per-frame rows in memory, flushed on an
# injectable clock's periodic interval plus explicit event-driven flush()
# calls. It must NOT flush per record, to avoid per-frame I/O.
# ---------------------------------------------------------------------------


def test_motion_curve_writer_buffers_rows_and_flushes_only_on_state_change_capture_or_periodic_interval(tmp_path):
    # The production interval bounds mid-session data loss to a few seconds.
    assert 0 < MOTION_CURVE_FLUSH_INTERVAL_S <= 5
    path = tmp_path / "motion_curve.csv"
    now = {"t": 0.0}
    writer = MotionCurveWriter(path, interval=3.0, clock=lambda: now["t"])

    with writer:
        writer.record(
            timestamp="t0", state="SETTLING", presence_score=0.11,
            motion_score=0.01,
            stable_elapsed=0.0, event="",
        )
        now["t"] = 1.0
        writer.record(
            timestamp="t1", state="SETTLING", presence_score=0.12,
            motion_score=0.02,
            stable_elapsed=0.5, event="",
        )
        now["t"] = 2.0
        writer.record(
            timestamp="t2", state="SETTLING", presence_score=0.13,
            motion_score=0.01,
            stable_elapsed=1.0, event="",
        )

        with path.open(newline="", encoding="utf-8") as stream:
            buffered_rows = list(csv.DictReader(stream))
        assert buffered_rows == []  # periodic interval (3.0s) not yet reached

        writer.flush()  # explicit event-driven flush (state change / capture)
        with path.open(newline="", encoding="utf-8") as stream:
            flushed_rows = list(csv.DictReader(stream))
        assert len(flushed_rows) == 3
        assert flushed_rows[0]["timestamp"] == "t0"
        assert flushed_rows[0]["presence_score"] == "0.11"

        now["t"] = 5.0  # 3.0s since the explicit flush -- interval elapsed
        writer.record(
            timestamp="t3", state="CAPTURE_REQUESTED", presence_score=0.14,
            motion_score=0.0,
            stable_elapsed=1.0, event="capture_requested",
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows_after_interval = list(csv.DictReader(stream))
        assert len(rows_after_interval) == 4
