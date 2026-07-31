"""UI-independent status formatting and per-preview-frame CSV telemetry."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from capture_runtime import CameraSettings
from capture_session import CaptureSession, CaptureState, FrameResult
from constants import SETTLING_CONFIRMATION_FRAMES


TELEMETRY_COLUMNS = (
    "timestamp",
    "mode",
    "state",
    "presence_score",
    "motion_score",
    "stable_elapsed",
    "exposure_us",
    "pending_block_id",
    "capture_path",
    "event",
)


class StatusReporter:
    def __init__(self, sink: Callable[[str], None], *, interval: float = 1.0):
        self._sink = sink
        self._interval = interval
        self._last_status_at: float | None = None

    def publish(
        self, *, now: float, session: CaptureSession, frame: FrameResult
    ) -> None:
        for event in session.drain_events():
            self._sink(f"[{event.state.name}] {event.message}")

        if (
            self._last_status_at is None
            or now - self._last_status_at >= self._interval
        ):
            self._sink(
                f"[{session.state.name}] presence={frame.presence_score:.4f} "
                f"motion={frame.motion_score:.4f} "
                f"stable={frame.stable_elapsed:.2f}s"
            )
            self._last_status_at = now


class TelemetryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = None
        self._writer = None

    def __enter__(self) -> "TelemetryWriter":
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=TELEMETRY_COLUMNS)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is not None:
            self._stream.close()

    def record(
        self,
        *,
        timestamp: str,
        session: CaptureSession,
        frame: FrameResult,
        settings: CameraSettings,
        capture_path: str = "",
        event: str = "",
    ) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("TelemetryWriter must be used as a context manager")
        self._writer.writerow(
            {
                "timestamp": timestamp,
                "mode": session.mode.value,
                "state": session.state.name,
                "presence_score": frame.presence_score,
                "motion_score": frame.motion_score,
                "stable_elapsed": frame.stable_elapsed,
                "exposure_us": settings.exposure_us,
                "pending_block_id": session.pending_block_id or "",
                "capture_path": capture_path,
                "event": event,
            }
        )
        self._stream.flush()


# ---------------------------------------------------------------------------
# Settling observer (#172): re-derives the settling wait / reset count / peak
# motion from the same (now, state, FrameResult) triple StatusReporter.publish
# already consumes, mirroring summarize_motion_samples's established pattern
# of re-deriving threshold crossings outside the state machine rather than
# reading CaptureSession's private _stable_since.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlingSummary:
    """Roll-up of one EMPTY->SETTLING->CAPTURE_REQUESTED window."""

    duration_ms: int
    resets: int
    max_motion: float


class SettlingObserver:
    """Watch (now, state, frame) triples; summarize on capture_requested.

    The window opens on the first frame observed in SETTLING (the presence
    transition itself never counts as a reset, mirroring
    `CaptureSession.accept_frame`'s own `if moving: self._stable_since = now`
    branch, which only fires on frames observed while already SETTLING).
    Any frame in any other state discards an in-progress window silently --
    an abandoned settling window that never reaches capture never surfaces a
    summary and never leaks its reset count into the next window.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._entry_now: float | None = None
        self._resets = 0
        self._max_motion = 0.0
        self._consecutive_motion_frames = 0

    def observe(
        self, *, now: float, state: CaptureState, frame: FrameResult
    ) -> SettlingSummary | None:
        if frame.capture_requested:
            if self._entry_now is None:
                return None
            summary = SettlingSummary(
                duration_ms=int(round((now - self._entry_now) * 1000)),
                resets=self._resets,
                max_motion=self._max_motion,
            )
            self._reset_window()
            return summary

        if state is CaptureState.SETTLING:
            if self._entry_now is None:
                self._entry_now = now
                self._resets = 0
                self._max_motion = frame.motion_score
            else:
                if frame.motion_score >= self.threshold:
                    self._consecutive_motion_frames += 1
                    if (
                        self._consecutive_motion_frames
                        == SETTLING_CONFIRMATION_FRAMES
                    ):
                        self._resets += 1
                else:
                    self._consecutive_motion_frames = 0
                self._max_motion = max(self._max_motion, frame.motion_score)
            return None

        # Any other state (e.g. back to EMPTY) abandons an in-progress window.
        self._reset_window()
        return None

    def _reset_window(self) -> None:
        self._entry_now = None
        self._resets = 0
        self._max_motion = 0.0
        self._consecutive_motion_frames = 0


# ---------------------------------------------------------------------------
# Motion curve writer (#172): buffers per-frame rows in memory, flushed on an
# injectable clock's periodic interval plus explicit event-driven flush()
# calls -- must NOT reproduce TelemetryWriter's per-record stream.flush(),
# which is exactly the per-frame-I/O bug this issue avoids repeating.
# ---------------------------------------------------------------------------

MOTION_CURVE_COLUMNS = (
    "timestamp",
    "state",
    "presence_score",
    "motion_score",
    "stable_elapsed",
    "event",
)


class MotionCurveWriter:
    """Buffered CSV writer for the raw per-frame motion curve.

    Rows accumulate in memory via `record()`. They only hit disk on an
    explicit `flush()` call (state-change / capture events) or once the
    injectable `clock` shows `interval` seconds have elapsed since the last
    flush -- this bounds a Pi power loss to at most one interval's worth of
    unflushed frames without paying TelemetryWriter's per-record I/O cost.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._interval = interval
        self._clock = clock
        self._stream = None
        self._writer = None
        self._buffer: list[dict] = []
        self._last_flush_at: float = 0.0

    def __enter__(self) -> "MotionCurveWriter":
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=MOTION_CURVE_COLUMNS)
        self._writer.writeheader()
        self._stream.flush()
        self._last_flush_at = self._clock()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is not None:
            self.flush()
            self._stream.close()

    def record(
        self,
        *,
        timestamp: str,
        state: str,
        presence_score: float,
        motion_score: float,
        stable_elapsed: float,
        event: str = "",
    ) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("MotionCurveWriter must be used as a context manager")
        self._buffer.append(
            {
                "timestamp": timestamp,
                "state": state,
                "presence_score": presence_score,
                "motion_score": motion_score,
                "stable_elapsed": stable_elapsed,
                "event": event,
            }
        )
        if self._clock() - self._last_flush_at >= self._interval:
            self.flush()

    def flush(self) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("MotionCurveWriter must be used as a context manager")
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._buffer.clear()
            self._stream.flush()
        self._last_flush_at = self._clock()
