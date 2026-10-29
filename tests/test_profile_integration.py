"""End-to-end --profile integration test.

Drives the REAL `PiCaptureRuntime` loop with `profile=True` against a real
`ProcessingStore` + `LoopbackCaptureReceiver` + `SessionWorkflow` (with an
injected scripted `clock`). Other slide-mode tests bypass the loop by calling
`runtime._consume_capture(...)` directly. Runs one normal slide capture (with
a flicker frame, so the settling reset count is >= 1) and one forced-[SLOW]
capture, then `runtime.end_session(confirm=True)` inside the receiver's `with`
block so the motion-curve upload lands `motion_curve.csv` in the bundle.

No wall-clock comparison ever occurs between "two machines": the only clocks
involved are the injected `SessionWorkflow._clock` (a scripted queue) and the
local `now` floats this test hands to `process_frame` directly -- the join
between the Pi-side profile_summary.csv rows and the uploaded motion curve is
by capture identity (`result.path.stem`), not by timestamp equality.
"""
from __future__ import annotations

import contextlib
import csv
import io
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
from session.workflow import (  # noqa: E402
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
)

from app import pi_session as run_pi_session  # noqa: E402
from store_harness import _identical_mask_slide_preprocessor  # noqa: E402


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

    Uses an injected scripted `clock` for deterministic consumer-split
    timings and a real `ActionLogger` (mandatory: `_attach_capture_logging` is
    a no-op without one, so `profile=True` alone would silently produce
    neither the console block nor the summary row). The empty session is
    drained into the slides phase and the slide camera calibrated before
    construction.
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


def test_profile_session_drives_a_normal_and_a_slow_slide_capture_end_to_end(tmp_path):
    """One profiled session: capture 1 (flickered, so it has a settling row)
    then capture 2 (forced [SLOW]). Profiling is dev-only instrumentation, so
    this single test is the whole --profile end-to-end guard.

    Capture 2 is forced [SLOW] by lowering the threshold below any possible
    elapsed time, since total_capture_ms is a real perf_counter() wrap.
    """
    store = ProcessingStore(
        tmp_path / "processing",
        slide_preprocessor=_identical_mask_slide_preprocessor,
    )
    session = store.start_session(started_at=STARTED_AT)
    clock = _scripted_clock(
        [
            # Capture 1: decode 10ms, outbox 20ms, send 30ms.
            0.000, 0.010, 0.010, 0.030, 0.030, 0.060,
            # Capture 2: decode 5ms, outbox 5ms, send 5ms.
            0.000, 0.005, 0.005, 0.010, 0.010, 0.015,
        ]
    )
    first_console, second_console = io.StringIO(), io.StringIO()

    with pytest.MonkeyPatch.context() as monkeypatch:
        with LoopbackCaptureReceiver(store) as receiver:
            _workflow, runtime = _build_workflow_and_runtime(
                tmp_path, store, session, receiver.url, clock=clock
            )
            runtime.start(background=False)

            with contextlib.redirect_stdout(first_console):
                _drive_one_capture(
                    runtime, "12080_51137181_01_HE", base_now=1.0, with_flicker=True
                )
            monkeypatch.setattr(run_pi_session, "PROFILE_SLOW_CAPTURE_MS", -1)
            with contextlib.redirect_stdout(second_console):
                _drive_one_capture(
                    runtime, "12080_51137182_01_HE", base_now=10.0, with_flicker=False
                )
                runtime.end_session(confirm=True)

    session_dir = next(store.root.glob(f"session_{session.number:06d}_*"))
    first, second = first_console.getvalue(), second_console.getvalue()

    # Console: a profile block per capture, settling row for the flickered
    # one, consumer split and [SLOW] marker for the forced-slow one.
    assert (first + second).count("Profile") >= 2
    assert "settling:" in first and "resets=" in first
    assert "decode" in second and "outbox" in second and "send" in second
    assert "[SLOW]" in second

    # Summary CSV: one fully-populated row per capture, joined by capture id.
    with (session_dir / "profile_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert set(rows[0].keys()) == (
        set(CAPTURE_STAGE_TIMING_KEYS) | set(SETTLING_STAGE_KEYS) | {"capture_id"}
    )
    assert len({row["capture_id"] for row in rows if row["capture_id"]}) == 2
    assert int(rows[0]["settling_resets"]) >= 1
    required = (
        "camera_capture_ms", "publish_ms", "consumer_ms", "consumer_decode_ms",
        "consumer_outbox_ms", "consumer_send_ms", "session_accept_ms",
        "total_capture_ms", "settling_duration_ms", "settling_resets",
        "settling_max_motion",
    )
    for row in rows:
        for key in required:
            assert row[key] not in (None, ""), f"missing {key} in row {row}"

    # Motion curve is flushed and uploaded into the bundle at session end.
    with (session_dir / "motion_curve.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        curve = list(csv.DictReader(stream))
    assert curve and all(row["presence_score"] != "" for row in curve)
