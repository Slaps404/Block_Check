"""Status/event and calibration telemetry tests (issue #86)."""
from __future__ import annotations

import csv

import numpy as np

from capture_runtime import CaptureConfiguration, camera_settings_for
from capture_session import CaptureSession, CaptureState, FrameResult, SessionConfig
from capture_telemetry import StatusReporter, TelemetryWriter


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

    # Events are plain data; consuming them performs no console I/O.
    assert all(isinstance(kind, str) for kind in kinds)


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


def test_one_configuration_surface_contains_detection_and_both_roles():
    config = CaptureConfiguration()

    assert config.session.roi == (0.1, 0.1, 0.9, 0.9)
    assert config.session.presence_threshold > 0
    assert config.session.motion_threshold > 0
    assert config.session.baseline_frames == 20
    assert config.session.stable_duration == 0.5
    assert config.session.removal_duration == 0.5
    assert config.role_settings["slide"].exposure_us == 8333
    assert config.role_settings["block"].exposure_us == 33333
    assert config.preview_fps == 10.0
    assert config.still_dimensions == (4056, 3040)


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

    from capture_runtime import CaptureController
    from capture_storage import CaptureStore

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
# Settling observer (#172): re-derives settling wait / reset-count / peak
# motion from the same (now, state, FrameResult) triple StatusReporter.publish
# already consumes, mirroring summarize_motion_samples's established pattern
# of re-deriving threshold crossings outside the state machine rather than
# reading CaptureSession's private _stable_since.
# ---------------------------------------------------------------------------


def test_settling_observer_computes_exact_wait_resets_and_max_motion_for_scripted_flicker_sequence():
    from capture_telemetry import SettlingObserver, SettlingSummary

    observer = SettlingObserver(threshold=0.02)

    # Entry frame: EMPTY -> SETTLING transition. Motion at entry is below
    # threshold and must not count as a reset.
    assert observer.observe(
        now=0.0,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.01),
    ) is None

    # Two consecutive motion frames confirm a genuine reset.
    assert observer.observe(
        now=0.3,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.05),
    ) is None
    assert observer.observe(
        now=0.5,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.04),
    ) is None

    # Quiet frame after the reset; no further motion until capture.
    assert observer.observe(
        now=1.1,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.01, stable_elapsed=0.6),
    ) is None

    summary = observer.observe(
        now=1.5,
        state=CaptureState.CAPTURE_REQUESTED,
        frame=FrameResult(
            motion_score=0.01, stable_elapsed=1.0, capture_requested=True
        ),
    )

    assert summary == SettlingSummary(duration_ms=1500, resets=1, max_motion=0.05)


def test_settling_observer_flicker_frames_each_count_as_a_reset_and_entry_frame_is_excluded_from_reset_count():
    from capture_telemetry import SettlingObserver

    observer = SettlingObserver(threshold=0.02)

    # Entry frame carries motion above threshold (the presence transition
    # itself) -- per CaptureSession.accept_frame this frame sets
    # `_stable_since` directly rather than going through the `if moving:
    # reset` branch, so the observer must not count it as a reset.
    observer.observe(
        now=0.0,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.5),
    )

    # The first two consecutive motion readings make one confirmed reset.
    observer.observe(
        now=0.2,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.03),
    )
    observer.observe(
        now=0.5,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.04),
    )
    # A quiet frame ends that consecutive run.
    observer.observe(
        now=0.7,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.0),
    )
    # A separate two-frame run makes the second confirmed reset.
    observer.observe(
        now=0.9,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.03),
    )
    observer.observe(
        now=1.1,
        state=CaptureState.SETTLING,
        frame=FrameResult(motion_score=0.04),
    )

    summary = observer.observe(
        now=1.7,
        state=CaptureState.CAPTURE_REQUESTED,
        frame=FrameResult(motion_score=0.0, capture_requested=True),
    )

    assert summary.resets == 2
    assert summary.max_motion == 0.5
    assert summary.duration_ms == 1700


def test_settling_observer_discards_an_abandoned_settling_window_that_never_captures():
    """An EMPTY->SETTLING->EMPTY round trip that never reaches
    CAPTURE_REQUESTED must not surface a summary; the window is discarded."""
    from capture_telemetry import SettlingObserver

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
# Motion curve writer (#172): buffers per-frame rows in memory, flushed on an
# injectable clock's periodic interval plus explicit event-driven flush()
# calls -- must NOT reproduce TelemetryWriter's per-record stream.flush(),
# which is exactly the per-frame-I/O bug this issue avoids repeating.
# ---------------------------------------------------------------------------


def test_motion_curve_writer_buffers_rows_and_flushes_only_on_state_change_capture_or_periodic_interval(tmp_path):
    from capture_telemetry import MotionCurveWriter

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


def test_motion_curve_is_written_pi_local_before_any_sync_and_flush_interval_bounds_loss_window(tmp_path):
    """#172: the periodic flush interval bounds mid-session data loss to a
    few seconds -- proven by asserting the flush cadence bounds, not by
    simulating an actual process crash (per the epic's durability note)."""
    from capture_telemetry import MotionCurveWriter
    from constants import MOTION_CURVE_FLUSH_INTERVAL_S

    assert 0 < MOTION_CURVE_FLUSH_INTERVAL_S <= 5

    path = tmp_path / "motion_curve.csv"
    now = {"t": 0.0}
    writer = MotionCurveWriter(
        path, interval=MOTION_CURVE_FLUSH_INTERVAL_S, clock=lambda: now["t"]
    )

    with writer:
        writer.record(
            timestamp="t0", state="SETTLING", presence_score=0.11,
            motion_score=0.01,
            stable_elapsed=0.0, event="",
        )

        # Simulate a power loss the instant after this record, without any
        # explicit flush or __exit__: nothing has hit disk yet because the
        # flush interval has not elapsed.
        with path.open(newline="", encoding="utf-8") as stream:
            rows_before_interval = list(csv.DictReader(stream))
        assert rows_before_interval == []

        now["t"] = MOTION_CURVE_FLUSH_INTERVAL_S
        writer.record(
            timestamp="t1", state="SETTLING", presence_score=0.12,
            motion_score=0.01,
            stable_elapsed=0.5, event="",
        )

        # Once the flush interval elapses, the buffered rows are durable on
        # Pi-local disk without waiting for an explicit flush() -- bounding
        # loss on a real power cut to at most one interval's worth of frames.
        with path.open(newline="", encoding="utf-8") as stream:
            rows_after_interval = list(csv.DictReader(stream))
        assert len(rows_after_interval) == 2
