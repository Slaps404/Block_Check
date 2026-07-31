"""End-to-end --profile integration test (issue #173).

Every profiling slice (#168 flag+bundle, #169 motion command, #171 consumer
clock seam, #172 settling observer + motion curve sync) already has unit/pure
coverage in isolation, but nothing drives the REAL `PiCaptureRuntime` loop
end-to-end with `profile=True` the way
`test_pi_capture_runtime_drives_controller_and_publishes_block` does for
block mode -- existing slide-mode tests bypass the loop by calling
`runtime._consume_capture(...)` directly. This module closes that gap: it
builds a real `ProcessingStore` + `LoopbackCaptureReceiver` +
`SessionWorkflow` (with an injected scripted `clock`) and a real
`PiCaptureRuntime(profile=True, action_logger=...)`, then drives
`runtime.start(background=False)` + `runtime.process_frame(...)` through a
scripted EMPTY -> SETTLING (with a deliberate flicker frame, so the settling
reset count is >= 1) -> CAPTURE_REQUESTED sequence for one normal slide
capture and one deliberately-marked-[SLOW] slide capture, then
`runtime.end_session(confirm=True)` inside the receiver's `with` block so the
motion-curve flush + HTTP upload lands `motion_curve.csv` into the bundle.

No wall-clock comparison ever occurs between "two machines": the only clocks
involved are the injected `SessionWorkflow._clock` (a scripted queue) and the
local `now` floats this test hands to `process_frame` directly -- the join
between the Pi-side profile_summary.csv rows and the uploaded motion curve is
by capture identity (`result.path.stem`), not by timestamp equality.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.append(str(_TOOLS_DIR))

from action_logger import ActionLogger  # noqa: E402
from capture_runtime import CAPTURE_STAGE_TIMING_KEYS, SETTLING_STAGE_KEYS  # noqa: E402
from constants import SETTLING_CONFIRMATION_FRAMES  # noqa: E402
from capture_session import SessionConfig  # noqa: E402
from session.workflow import (  # noqa: E402
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
)

import run_pi_session  # noqa: E402


STARTED_AT = datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc)


class _FakeRealCamera:
    """Mirrors `test_pi_session.py`'s `_FakeRealCamera`: a real-shaped PNG
    still (matching `CaptureSession._still_acceptable`'s 4056x3040 check)."""

    def __init__(self):
        self.started = 0
        self.closed = 0

    def start_preview(self, **_kwargs):
        self.started += 1

    def preview_frame(self):
        return np.full((80, 120, 3), 180, dtype=np.uint8)

    def capture_still(self, path, **_kwargs):
        assert cv2.imwrite(str(path), np.full((3040, 4056), 100, dtype=np.uint8))

    def resume_preview(self):
        pass

    def close(self):
        self.closed += 1


class _FakePhaseCamera:
    """Phase-activation camera for `SessionWorkflow` (not the still/preview
    camera). Mirrors `test_pi_session.py`'s `_FakePhaseCamera`."""

    def activate_mode(self, mode: str):
        from camera_calibration import (
            ActivatedCameraMode,
            CalibrationQuality,
            LockedCameraControls,
            PhaseCameraCalibration,
        )

        calibration = PhaseCameraCalibration(
            mode=mode,
            controls=LockedCameraControls(1, 1.0, (1.0, 1.0)),
            quality=CalibrationQuality(
                True, 1, 0, 0.0, 0.0, 0.0, 0.0, 220.0, 0.0, 0.0
            ),
            metadata_samples=(),
        )
        baseline = np.full((480, 640, 3), 220, dtype=np.uint8)
        return ActivatedCameraMode(calibration, baseline)


def _scripted_clock(values):
    """A `Callable[[], float]` that pops one scripted value per call.

    Backs `SessionWorkflow(clock=...)` so `capture_slide`'s
    decode/outbox/send sub-durations are deterministic -- the ONLY clock
    besides the local `now` floats this test hands to `process_frame`
    directly. No real wall-clock sleep and no cross-machine comparison.
    """
    queue = list(values)

    def _clock() -> float:
        return queue.pop(0)

    return _clock


def _build_workflow_and_runtime(
    tmp_path, store, session, receiver_url, *, clock, session_config=None
):
    """Builds a real slide-phase `SessionWorkflow` + `PiCaptureRuntime(profile=True)`.

    Mirrors `test_pi_session.py`'s `_build_workflow`, plus: (1) an injected
    scripted `clock` for deterministic consumer-split timings, (2) a real
    `ActionLogger` (mandatory -- `_attach_capture_logging` is a no-op without
    one, so `profile=True` alone would silently produce neither the console
    block nor the summary row), (3) `profile=True`, and (4) draining an empty
    session into the slides phase + calibrating the slide phase camera before
    construction, exactly like `test_scanned_slide_capture_never_attempts_camera_qr_decode`.
    """
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient(receiver_url),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
        camera=_FakePhaseCamera(),
        clock=clock,
    )
    workflow.finish_blocks()
    workflow.prepare_empty_backlight("slide")

    action_logger = ActionLogger(tmp_path / "actions.log", session_number=session.number)
    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeRealCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=session_config
        or run_pi_session.SessionConfig(
            baseline_frames=1, stable_duration=1.0, removal_duration=0.0
        ),
        action_logger=action_logger,
        profile=True,
    )
    assert run_pi_session.CaptureMode.SLIDE is runtime.mode
    return workflow, runtime


def _drive_one_capture(runtime, payload, *, base_now, with_flicker):
    """Drives one settled slide capture from EMPTY, returning the console text.

    `with_flicker=True` inserts two consecutive distinct-motion readings after
    SETTLING entry, so `SettlingObserver` counts one confirmed reset. Returns
    the printed profile block for the capture.

    Each capture gets its own `captured_at` (derived from `base_now`) --
    `PiOutbox.publish_slide`'s request-ledger id is derived from the
    timestamp, so reusing one across two distinct captures collides in the
    idempotency ledger (`ValueError: request_id was already used with
    different request arguments`).

    `flicker`'s rectangle sits in a different region with a different
    (bright) fill than `specimen_a`'s (dark) rectangle: normalizing by mean
    intensity makes a spatially UNIFORM frame indistinguishable from any
    other uniform frame (including the uniform baseline) regardless of its
    absolute brightness, so the flicker frame needs its own distinct texture
    to register both presence (vs. the uniform baseline) and motion (vs.
    `specimen_a`).
    """
    empty = np.full((80, 120, 3), 180, dtype=np.uint8)
    specimen_a = empty.copy()
    cv2.rectangle(specimen_a, (30, 20), (90, 60), (40, 40, 40), -1)
    flicker = empty.copy()
    cv2.rectangle(flicker, (60, 30), (115, 79), (250, 250, 250), -1)

    runtime.scan_qr(payload)
    captured_at = STARTED_AT + timedelta(seconds=base_now)

    t = base_now
    runtime.process_frame(specimen_a, now=t, captured_at=captured_at)
    t += 1.0
    runtime.process_frame(specimen_a, now=t, captured_at=captured_at)  # SETTLING entry
    if with_flicker:
        t += 1.0
        runtime.process_frame(flicker, now=t, captured_at=captured_at)  # candidate
        t += 1.0
        runtime.process_frame(specimen_a, now=t, captured_at=captured_at)  # reset
    t += 1.0
    runtime.process_frame(specimen_a, now=t, captured_at=captured_at)  # settles -> capture

    # Removal: two consecutive "not present" frames (removal_duration=0.0)
    # return the slide-mode session to EMPTY, ready for the next capture.
    t += 1.0
    runtime.process_frame(empty, now=t, captured_at=captured_at)
    t += 1.0
    runtime.process_frame(empty, now=t, captured_at=captured_at)


def test_profile_session_drives_a_normal_and_a_slow_slide_capture_end_to_end(
    tmp_path, monkeypatch, capsys,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock(
        [
            # Capture 1: decode 10ms, outbox 20ms, send 30ms.
            0.000, 0.010, 0.010, 0.030, 0.030, 0.060,
            # Capture 2: decode 5ms, outbox 5ms, send 5ms.
            0.000, 0.005, 0.005, 0.010, 0.010, 0.015,
        ]
    )

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)

        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )

        # Capture 2 is forced [SLOW] regardless of real elapsed wall-clock
        # time: total_capture_ms is a real time.perf_counter() wrap inside
        # capture_runtime.py that cannot be injected, so the threshold is
        # lowered below any possible elapsed time instead of sleeping.
        monkeypatch.setattr(run_pi_session, "PROFILE_SLOW_CAPTURE_MS", -1)
        _drive_one_capture(
            runtime, "12080_51137182_01_HE", base_now=10.0, with_flicker=False
        )

        runtime.end_session(confirm=True)

    console = capsys.readouterr().out
    assert console.count("Profile") >= 2
    assert "[SLOW]" in console


def test_profile_summary_csv_has_one_row_per_capture_with_every_stage_column_joined_by_capture_id(
    tmp_path, monkeypatch,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock(
        [
            0.000, 0.010, 0.010, 0.030, 0.030, 0.060,
            0.000, 0.005, 0.005, 0.010, 0.010, 0.015,
        ]
    )

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)

        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )
        monkeypatch.setattr(run_pi_session, "PROFILE_SLOW_CAPTURE_MS", -1)
        _drive_one_capture(
            runtime, "12080_51137182_01_HE", base_now=10.0, with_flicker=False
        )

        runtime.end_session(confirm=True)

    session_dir = next(store.root.glob(f"session_{session.number:06d}_*"))
    summary_path = session_dir / "profile_summary.csv"
    assert summary_path.is_file()

    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    required_columns = {
        "capture_id",
        "camera_capture_ms",
        "publish_ms",
        "consumer_ms",
        "consumer_decode_ms",
        "consumer_outbox_ms",
        "consumer_send_ms",
        "session_accept_ms",
        "total_capture_ms",
        "settling_duration_ms",
        "settling_resets",
        "settling_max_motion",
    }
    assert required_columns <= set(rows[0].keys())
    assert set(CAPTURE_STAGE_TIMING_KEYS) | set(SETTLING_STAGE_KEYS) | {"capture_id"} == set(
        rows[0].keys()
    )

    # Joined by capture identity: every row's capture_id is a distinct,
    # non-empty stem (the published capture file's UUID stem) -- internal
    # consistency, not a cross-reference against store.get_set (slide
    # captures have no block-id business key at all).
    capture_ids = [row["capture_id"] for row in rows]
    assert all(capture_ids)
    assert len(set(capture_ids)) == 2

    # Capture 1 had confirmed motion -> at least one settling reset.
    assert int(rows[0]["settling_resets"]) >= 1
    # Capture 2's every stage column is populated too (not just capture 1's).
    for row in rows:
        for key in required_columns:
            assert row[key] not in (None, ""), f"missing {key} in row {row}"


def test_slide_benchmark_combines_pi_and_pc_stages_by_final_slide_id(
    tmp_path, monkeypatch,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock([0.000, 0.010, 0.010, 0.030, 0.030, 0.060])

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)
        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )
        runtime.end_session(confirm=True)

    session_dir = next(store.root.glob(f"session_{session.number:06d}_*"))
    with (session_dir / "slide_benchmark.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["capture_id"].startswith("slide_51137181_")
    for key in (
        "settling_ms", "camera_capture_ms", "publish_ms", "qr_decode_ms",
        "outbox_write_ms", "transfer_wait_ms", "receive_persist_ms",
        "qc_render_ms", "verdict_commit_export_ms", "full_total_ms",
    ):
        assert row[key] != "", f"missing {key}: {row}"


def test_motion_curve_csv_is_synced_into_the_bundle_at_session_end(tmp_path, monkeypatch):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock(
        [
            0.000, 0.010, 0.010, 0.030, 0.030, 0.060,
            0.000, 0.005, 0.005, 0.010, 0.010, 0.015,
        ]
    )

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)

        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )
        monkeypatch.setattr(run_pi_session, "PROFILE_SLOW_CAPTURE_MS", -1)
        _drive_one_capture(
            runtime, "12080_51137182_01_HE", base_now=10.0, with_flicker=False
        )

        runtime.end_session(confirm=True)

    session_dir = next(store.root.glob(f"session_{session.number:06d}_*"))
    curve_path = session_dir / "motion_curve.csv"
    assert curve_path.is_file()

    with curve_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Non-empty: header plus at least one buffered per-frame row synced in.
    assert len(rows) > 0
    assert "presence_score" in rows[0]
    assert all(row["presence_score"] != "" for row in rows)


def test_profile_config_records_effective_settling_controls_in_the_bundle(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock([0.000, 0.010, 0.010, 0.030, 0.030, 0.060])
    configured = SessionConfig(
        baseline_frames=1,
        stable_duration=0.75,
        removal_duration=0.25,
        motion_threshold=0.034,
        presence_threshold=0.056,
    )

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path,
            store,
            session,
            receiver.url,
            clock=clock,
            session_config=configured,
        )
        runtime.start(background=False)
        assert runtime.profile_config_path is not None
        assert runtime.profile_config_path.is_file()

        runtime.end_session(confirm=True)

    session_dir = next(store.root.glob(f"session_{session.number:06d}_*"))
    config_path = session_dir / "profile_config.json"
    assert config_path.is_file()
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "stable_duration": 0.75,
        "removal_duration": 0.25,
        "motion_threshold": 0.034,
        "presence_threshold": 0.056,
        "SETTLING_CONFIRMATION_FRAMES": SETTLING_CONFIRMATION_FRAMES,
    }


def test_console_output_contains_consumer_split_settling_row_and_slow_marker(
    tmp_path, monkeypatch, capsys,
):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock(
        [
            0.000, 0.010, 0.010, 0.030, 0.030, 0.060,
            0.000, 0.005, 0.005, 0.010, 0.010, 0.015,
        ]
    )

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)

        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )
        capsys.readouterr()  # discard capture 1's console output for isolation

        monkeypatch.setattr(run_pi_session, "PROFILE_SLOW_CAPTURE_MS", -1)
        _drive_one_capture(
            runtime, "12080_51137182_01_HE", base_now=10.0, with_flicker=False
        )

        runtime.end_session(confirm=True)

    console = capsys.readouterr().out
    # Capture 2's block: consumer split, [SLOW] marker (capture 2 is forced
    # slow). The settling row requires a flicker (capture 1 only), so it is
    # asserted on capture 1's block below instead.
    assert "consumer" in console
    assert "decode" in console and "outbox" in console and "send" in console
    assert "[SLOW]" in console


def test_profile_console_settling_row_appears_for_a_flickered_capture(
    tmp_path, capsys,
):
    """The settling roll-up row (`settling: <ms>ms (resets=..., max_motion=...)`)
    only renders when all three `SETTLING_STAGE_KEYS` are present -- which only
    happens for a capture that actually passed through an observed SETTLING
    window (capture 1's deliberate flicker), documented separately from the
    consumer-split/[SLOW] assertions above so each capture's distinct
    contribution is pinned down on its own."""
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock([0.000, 0.010, 0.010, 0.030, 0.030, 0.060])

    with LoopbackCaptureReceiver(store) as receiver:
        workflow, runtime = _build_workflow_and_runtime(
            tmp_path, store, session, receiver.url, clock=clock
        )
        runtime.start(background=False)

        _drive_one_capture(
            runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
        )

        runtime.end_session(confirm=True)

    console = capsys.readouterr().out
    assert "settling:" in console
    assert "resets=" in console
    assert "max_motion=" in console


def test_profile_join_uses_capture_identity_not_wall_clock():
    """Documents (rather than newly asserts) the no-cross-machine-clock
    property: `SessionWorkflow` takes exactly one injectable `clock`
    (`code/session/workflow.py`'s `SessionWorkflow.__init__(..., clock=...)`),
    and `PiCaptureRuntime.process_frame`/`CaptureController._capture` only
    ever consume the local `now`/`time.perf_counter()` values this test
    supplies directly -- profile_summary.csv rows are joined to the uploaded
    motion_curve.csv by `capture_id` (`result.path.stem`), never by comparing
    timestamps produced on two different machines. See
    `test_profile_summary_csv_has_one_row_per_capture_with_every_stage_column_joined_by_capture_id`
    and `test_motion_curve_csv_is_synced_into_the_bundle_at_session_end` above,
    which together are the only two clock-touching assertions in this module.
    """
    import inspect

    signature = inspect.signature(SessionWorkflow.__init__)
    assert "clock" in signature.parameters
