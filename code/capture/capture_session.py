"""Hardware-independent automatic specimen capture state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from capture_storage import ValidatedStill
from constants import CAPTURE_DIMENSIONS, SETTLING_CONFIRMATION_FRAMES


class CaptureState(Enum):
    WAITING_FOR_SCAN = auto()
    AWAITING_BASELINE_CONFIRMATION = auto()
    BUILDING_BASELINE = auto()
    CALIBRATION_FAILED = auto()
    EMPTY = auto()
    SETTLING = auto()
    CAPTURE_REQUESTED = auto()
    AWAITING_ACCEPT = auto()
    WAITING_FOR_REMOVAL = auto()
    REPOSITION_SLIDE = auto()
    CAPTURE_ERROR = auto()


class CaptureMode(str, Enum):
    BLOCK = "block"
    SLIDE = "slide"


@dataclass(frozen=True)
class SessionConfig:
    baseline_frames: int = 20
    roi: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)
    presence_threshold: float = 0.04
    motion_threshold: float = 0.02
    stable_duration: float = 0.5
    removal_duration: float = 0.5


@dataclass(frozen=True)
class CaptureResult:
    ok: bool
    path: Path | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validated: ValidatedStill | None = None

    @classmethod
    def success(
        cls,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        validated: ValidatedStill | None = None,
    ) -> "CaptureResult":
        return cls(
            ok=True,
            path=Path(path),
            metadata=dict(metadata or {}),
            validated=validated,
        )

    @classmethod
    def failure(
        cls, error: str, *, metadata: Mapping[str, Any] | None = None
    ) -> "CaptureResult":
        return cls(ok=False, error=error, metadata=dict(metadata or {}))


@dataclass(frozen=True)
class FrameResult:
    presence_score: float = 0.0
    motion_score: float = 0.0
    stable_elapsed: float = 0.0
    capture_requested: bool = False


@dataclass(frozen=True)
class MotionSample:
    """Summary of sampled `motion_score` readings over a fixed window.

    Built by `summarize_motion_samples`; rendered by
    `session.console.render_motion_sample`. See the `motion` console
    command (#169).
    """

    min_score: float
    mean_score: float
    max_score: float
    threshold_crossings: int
    sample_count: int


def summarize_motion_samples(
    scores: "list[float] | tuple[float, ...]", *, threshold: float
) -> MotionSample:
    """Summarize a sequence of sampled motion scores.

    A score counts as a threshold crossing when it is `>= threshold`,
    mirroring `CaptureSession.accept_frame`'s own
    `moving = motion_score >= self.config.motion_threshold` convention.
    """
    scores = tuple(scores)
    return MotionSample(
        min_score=min(scores),
        mean_score=sum(scores) / len(scores),
        max_score=max(scores),
        threshold_crossings=sum(1 for score in scores if score >= threshold),
        sample_count=len(scores),
    )


@dataclass(frozen=True)
class ScanResult:
    accepted: bool
    message: str


@dataclass(frozen=True)
class SuccessfulCapture:
    path: Path
    role: CaptureMode
    block_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    message: str
    state: CaptureState
    path: Path | None = None


class CaptureSession:
    def __init__(
        self,
        config: SessionConfig | None = None,
        *,
        mode: CaptureMode | str = CaptureMode.SLIDE,
        review_captures: bool = False,
    ):
        self.config = config or SessionConfig()
        self.mode = CaptureMode(mode)
        self.review_captures = review_captures
        # Both modes calibrate on an empty backlight before any specimen work.
        # Block mode then waits for a scan; slide mode goes straight to EMPTY.
        self.state = CaptureState.AWAITING_BASELINE_CONFIRMATION
        self.baseline = None
        self.pending_block_id: str | None = None
        self.last_capture: SuccessfulCapture | None = None
        self._events: list[SessionEvent] = []
        self._event_listeners: list[Callable[[SessionEvent], None]] = []

        self._baseline_frames: list[np.ndarray] = []
        self._previous_normalized: np.ndarray | None = None
        self._stable_since: float | None = None
        self._absence_since: float | None = None
        self._present_frames = 0
        self._absent_frames = 0
        self._motion_frames = 0
        self._reposition_movement_seen = False
        self._restored_state: CaptureState | None = None
        self._emit(
            "baseline_instruction",
            "Leave the backlight empty, then confirm baseline creation",
        )

    def confirm_empty(self) -> None:
        if self.baseline is not None:
            return
        if self.state is not CaptureState.AWAITING_BASELINE_CONFIRMATION:
            raise RuntimeError("session is not ready to build an empty baseline")
        self.baseline = None
        self._baseline_frames.clear()
        self._previous_normalized = None
        self.state = CaptureState.BUILDING_BASELINE
        self._emit("baseline_started", "Collecting empty-backlight baseline")

    def begin_empty_backlight_setup(self) -> None:
        if self.state not in (
            CaptureState.AWAITING_BASELINE_CONFIRMATION,
            CaptureState.CALIBRATION_FAILED,
        ):
            raise RuntimeError(
                "empty-backlight setup only from awaiting confirmation or calibration failure"
            )
        self.baseline = None
        self._baseline_frames.clear()
        self._previous_normalized = None
        self.state = CaptureState.BUILDING_BASELINE
        self._emit("baseline_started", "Collecting empty-backlight baseline")

    def mark_calibration_failed(self) -> None:
        self.baseline = None
        self._baseline_frames.clear()
        self._previous_normalized = None
        self.state = CaptureState.CALIBRATION_FAILED
        self._emit("calibration_failed", "Empty-backlight setup failed")

    def install_locked_baseline(self, frame: np.ndarray) -> None:
        """Install the empty-field baseline collected under locked phase controls."""
        self.baseline = self._roi_gray(frame).astype(np.float32)
        self._baseline_frames.clear()
        self._previous_normalized = self._normalize(self.baseline)
        restored_state = self._restored_state
        self._restored_state = None
        if restored_state is not None:
            self.state = restored_state
        elif self.mode is CaptureMode.BLOCK and self.pending_block_id is None:
            self.state = CaptureState.WAITING_FOR_SCAN
        else:
            self.state = CaptureState.EMPTY
        self._emit("baseline_ready", "Locked empty-backlight baseline is ready")

    def accept_frame(self, frame: np.ndarray, *, now: float) -> FrameResult:
        if self.state in (
            CaptureState.WAITING_FOR_SCAN,
            CaptureState.CALIBRATION_FAILED,
        ):
            return FrameResult()
        gray = self._roi_gray(frame)

        if self.state is CaptureState.AWAITING_BASELINE_CONFIRMATION:
            return FrameResult()

        if self.state is CaptureState.AWAITING_ACCEPT:
            return FrameResult()

        if self.state is CaptureState.BUILDING_BASELINE:
            self._baseline_frames.append(gray)
            if len(self._baseline_frames) >= self.config.baseline_frames:
                self.baseline = np.mean(self._baseline_frames, axis=0)
                self._baseline_frames.clear()
                self._previous_normalized = self._normalize(self.baseline)
                restored_state = self._restored_state
                self._restored_state = None
                if restored_state is not None:
                    self.state = restored_state
                elif (
                    self.mode is CaptureMode.BLOCK
                    and self.pending_block_id is None
                ):
                    self.state = CaptureState.WAITING_FOR_SCAN
                else:
                    self.state = CaptureState.EMPTY
                self._emit("baseline_ready", "Empty-backlight baseline is ready")
                if self.state is CaptureState.WAITING_FOR_SCAN:
                    self._emit(
                        "scan_required", "Scan an eight-digit block ID"
                    )
                if self.state is CaptureState.REPOSITION_SLIDE:
                    self._emit("slide_reposition_required", "Reposition slide")
            return FrameResult()

        normalized = self._normalize(gray)
        presence_score = self._difference(
            normalized, self._normalize(self.baseline)
        )
        motion_score = 0.0
        if self._previous_normalized is not None:
            current_blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
            previous_blurred = cv2.GaussianBlur(
                self._previous_normalized, (5, 5), 0
            )
            motion_score = self._difference(current_blurred, previous_blurred)
        self._previous_normalized = normalized

        present = presence_score >= self.config.presence_threshold
        moving = motion_score >= self.config.motion_threshold
        requested = False
        stable_elapsed = 0.0

        if self.state is CaptureState.EMPTY:
            if present:
                if self._present_frames == 0:
                    # Preserve the existing one-second settling clock from the
                    # first present frame, while requiring a second frame before
                    # committing the UI/state transition.
                    self._stable_since = now
                self._present_frames += 1
                if self._present_frames >= SETTLING_CONFIRMATION_FRAMES:
                    self.state = CaptureState.SETTLING
                    self._present_frames = 0
                    stable_elapsed = max(0.0, now - self._stable_since)
                    if self.config.stable_duration <= 0:
                        self.state = CaptureState.CAPTURE_REQUESTED
                        requested = True
                        self._emit(
                            "capture_requested", "Specimen is stable; capture requested"
                        )
            else:
                self._present_frames = 0
                self._stable_since = None

        elif self.state is CaptureState.SETTLING:
            if not present:
                self._absent_frames += 1
                self._motion_frames = 0
                if self._absent_frames >= SETTLING_CONFIRMATION_FRAMES:
                    self.state = CaptureState.EMPTY
                    self._absent_frames = 0
                    self._stable_since = None
            else:
                self._absent_frames = 0
                if moving:
                    self._motion_frames += 1
                    # Once movement is confirmed, keep the clock pinned to the
                    # latest moving frame until the specimen is quiet again.
                    if self._motion_frames >= SETTLING_CONFIRMATION_FRAMES:
                        self._stable_since = now
                else:
                    self._motion_frames = 0
                if self._stable_since is None:
                    self._stable_since = now
                stable_elapsed = max(0.0, now - self._stable_since)
                if (
                    not moving
                    and stable_elapsed >= self.config.stable_duration
                ):
                    self.state = CaptureState.CAPTURE_REQUESTED
                    requested = True
                    self._emit("capture_requested", "Specimen is stable; capture requested")

        elif self.state is CaptureState.WAITING_FOR_REMOVAL:
            if present:
                self._absence_since = None
            elif self._absence_since is None:
                self._absence_since = now
            elif now - self._absence_since >= self.config.removal_duration:
                self.state = (
                    CaptureState.WAITING_FOR_SCAN
                    if self.mode is CaptureMode.BLOCK
                    else CaptureState.EMPTY
                )
                self._absence_since = None
                self._stable_since = None
                self._present_frames = 0
                self._absent_frames = 0
                self._motion_frames = 0
                self._emit("removal_confirmed", "Specimen removal confirmed")
                if self.state is CaptureState.WAITING_FOR_SCAN:
                    self._emit("scan_required", "Scan the next eight-digit block ID")

        elif self.state is CaptureState.REPOSITION_SLIDE:
            if not present:
                if self._absence_since is None:
                    self._absence_since = now
                elif now - self._absence_since >= self.config.removal_duration:
                    self.state = CaptureState.EMPTY
                    self._absence_since = None
                    self._stable_since = None
                    self._motion_frames = 0
                    self._reposition_movement_seen = False
                    self._emit("removal_confirmed", "Waiting for slide")
            else:
                self._absence_since = None
                if moving:
                    self._motion_frames += 1
                    if self._motion_frames >= SETTLING_CONFIRMATION_FRAMES:
                        self._reposition_movement_seen = True
                        self._stable_since = now
                else:
                    self._motion_frames = 0
                    if self._reposition_movement_seen:
                        if self._stable_since is None:
                            self._stable_since = now
                        stable_elapsed = max(0.0, now - self._stable_since)
                        if stable_elapsed >= self.config.stable_duration:
                            self.state = CaptureState.CAPTURE_REQUESTED
                            requested = True
                            self._emit(
                                "capture_requested",
                                "Repositioned slide is stable; recapture requested",
                            )

        return FrameResult(
            presence_score=presence_score,
            motion_score=motion_score,
            stable_elapsed=stable_elapsed,
            capture_requested=requested,
        )

    def accept_capture_result(self, result: CaptureResult) -> None:
        if self.state is not CaptureState.CAPTURE_REQUESTED:
            raise RuntimeError("capture result received without a request")
        if not result.ok or not self._still_acceptable(result, self.mode.value):
            self.state = CaptureState.CAPTURE_ERROR
            self._emit(
                "capture_error",
                result.error or "Capture validation failed; Retry is available",
            )
            return
        block_id = self.pending_block_id if self.mode is CaptureMode.BLOCK else None
        published_block_id = result.metadata.get("block_id")
        if published_block_id is not None and published_block_id != block_id:
            self.state = CaptureState.CAPTURE_ERROR
            self._emit("capture_error", "Published block ID did not match pending scan")
            return
        self.last_capture = SuccessfulCapture(
            path=result.path,
            role=self.mode,
            block_id=block_id,
            metadata=dict(result.metadata),
        )
        if self.review_captures:
            self.state = CaptureState.AWAITING_ACCEPT
            self._absence_since = None
            self._emit("capture_saved", f"Saved {result.path.name}", result.path)
            return
        if self.mode is CaptureMode.BLOCK:
            self.pending_block_id = None
        self.state = CaptureState.WAITING_FOR_REMOVAL
        self._absence_since = None
        self._emit("capture_saved", f"Saved {result.path.name}", result.path)

    def accept_capture(self) -> SuccessfulCapture:
        if self.state is not CaptureState.AWAITING_ACCEPT:
            raise RuntimeError(
                "accept is only valid while a capture is awaiting operator review"
            )
        if self.last_capture is None:
            raise RuntimeError("no pending capture is held for review")
        if self.mode is CaptureMode.BLOCK:
            self.pending_block_id = None
        self.state = CaptureState.WAITING_FOR_REMOVAL
        self._absence_since = None
        self._emit("capture_accepted", f"Accepted {self.last_capture.path.name}")
        return self.last_capture

    def mark_slide_unreadable(self) -> None:
        """Require movement before another automatic capture of this slide."""
        if self.mode is not CaptureMode.SLIDE:
            raise RuntimeError("unreadable-slide recovery is only valid in slide mode")
        if self.state is not CaptureState.WAITING_FOR_REMOVAL:
            raise RuntimeError("unreadable-slide recovery requires a saved capture")
        self.state = CaptureState.REPOSITION_SLIDE
        self._absence_since = None
        self._stable_since = None
        self._reposition_movement_seen = False
        self._emit("slide_reposition_required", "Reposition slide")

    def restore_pending_block(self, block_id: str) -> None:
        """Restore an accepted-but-uncaptured block after process restart."""
        if self.mode is not CaptureMode.BLOCK:
            raise RuntimeError("pending-block recovery is only valid in block mode")
        self.pending_block_id = block_id
        self.state = CaptureState.AWAITING_BASELINE_CONFIRMATION
        self._emit(
            "baseline_instruction",
            "Leave the backlight empty, then confirm baseline creation",
        )

    def restore_unreadable_slide(self) -> None:
        """Restore fail-closed reposition behavior after a process restart."""
        if self.mode is not CaptureMode.SLIDE:
            raise RuntimeError("unreadable-slide recovery is only valid in slide mode")
        if self.baseline is None:
            self._restored_state = CaptureState.REPOSITION_SLIDE
            return
        self.state = CaptureState.REPOSITION_SLIDE
        self._absence_since = None
        self._stable_since = None
        self._reposition_movement_seen = False
        self._emit("slide_reposition_required", "Reposition slide")

    def restore_waiting_for_removal(self) -> None:
        """Keep a captured or skipped slide disarmed across a restart."""
        if self.mode is not CaptureMode.SLIDE:
            raise RuntimeError("slide recovery is only valid in slide mode")
        if self.baseline is None:
            self._restored_state = CaptureState.WAITING_FOR_REMOVAL
            return
        self.state = CaptureState.WAITING_FOR_REMOVAL
        self._absence_since = None
        self._stable_since = None
        self._reposition_movement_seen = False

    def skip_unreadable_slide(self) -> None:
        """Accept the retained unidentified captures and advance after removal."""
        if not self.unreadable_slide_can_be_skipped:
            raise RuntimeError("Skip is only valid for an unreadable slide")
        if self._restored_state is CaptureState.REPOSITION_SLIDE:
            self._restored_state = CaptureState.WAITING_FOR_REMOVAL
        else:
            self.state = CaptureState.WAITING_FOR_REMOVAL
        self._absence_since = None
        self._stable_since = None
        self._reposition_movement_seen = False
        self._emit("unreadable_slide_skipped", "Unidentified slide skipped")

    @property
    def unreadable_slide_can_be_skipped(self) -> bool:
        return (
            self.state is CaptureState.REPOSITION_SLIDE
            or self._restored_state is CaptureState.REPOSITION_SLIDE
        )

    def submit_scan(self, value: str) -> ScanResult:
        if self.mode is CaptureMode.SLIDE:
            result = ScanResult(False, "slide mode does not accept block scans")
            self._emit("scan_rejected", result.message)
            return result
        if not (len(value) == 8 and value.isascii() and value.isdigit()):
            result = ScanResult(
                False, "block ID must contain exactly eight numeric digits"
            )
            self._emit("scan_rejected", f"Scan rejected: {result.message}")
            return result
        if self.pending_block_id is not None:
            result = ScanResult(False, "a block ID is already pending")
            self._emit("scan_rejected", f"Scan rejected: {result.message}")
            return result
        if self.state is not CaptureState.WAITING_FOR_SCAN:
            result = ScanResult(False, "session is not waiting for a scan")
            self._emit("scan_rejected", f"Scan rejected: {result.message}")
            return result
        self.pending_block_id = value
        self.state = (
            CaptureState.EMPTY
            if self.baseline is not None
            else CaptureState.AWAITING_BASELINE_CONFIRMATION
        )
        self._emit("scan_accepted", f"Accepted block ID {value}")
        if self.baseline is None:
            self._emit(
                "baseline_instruction",
                "Leave the backlight empty, then confirm baseline creation",
            )
        return ScanResult(True, f"accepted block ID {value}")

    def retry_capture(self) -> FrameResult:
        if self.state is CaptureState.AWAITING_ACCEPT:
            self.last_capture = None
            self.state = CaptureState.CAPTURE_REQUESTED
            self._emit("capture_retaken", "Discarded held capture; retrying")
            return FrameResult(capture_requested=True)
        if self.state not in (
            CaptureState.CAPTURE_ERROR,
            CaptureState.REPOSITION_SLIDE,
        ):
            raise RuntimeError(
                "Retry is only valid after a capture error or unreadable-slide "
                "reposition"
            )
        self.state = CaptureState.CAPTURE_REQUESTED
        self._emit("retry", "Retrying capture for the same specimen")
        return FrameResult(capture_requested=True)

    def drain_events(self) -> tuple[SessionEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def add_event_listener(self, listener: Callable[[SessionEvent], None]) -> None:
        """Observe state events without consuming presentation-facing events."""
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)

    def _emit(self, kind: str, message: str, path: Path | None = None) -> None:
        event = SessionEvent(kind, message, self.state, path)
        self._events.append(event)
        for listener in self._event_listeners:
            listener(event)

    def _roi_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("preview frame must not be empty")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 3):
            raise ValueError(
                "preview frame must be 2-D grayscale or 3-channel colour; "
                f"got shape {frame.shape}"
            )
        if not np.issubdtype(frame.dtype, np.number):
            raise ValueError(
                f"preview frame must be numeric; got dtype {frame.dtype}"
            )
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = self.config.roi
        left, right = round(x0 * width), round(x1 * width)
        top, bottom = round(y0 * height), round(y1 * height)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("ROI must describe a non-empty part of the frame")
        roi = frame[top:bottom, left:right]
        if roi.ndim == 3:
            return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return roi.astype(np.float32)

    @staticmethod
    def _normalize(gray: np.ndarray) -> np.ndarray:
        mean = float(np.mean(gray))
        if mean <= 1e-6:
            return np.zeros_like(gray, dtype=np.float32)
        return gray.astype(np.float32) / mean

    @staticmethod
    def _difference(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            second = cv2.resize(
                second,
                (first.shape[1], first.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return float(np.mean(np.abs(first - second)))

    @staticmethod
    def _still_acceptable(result: CaptureResult, role: str) -> bool:
        path = result.path
        if path is None or path.suffix.lower() != ".png" or not path.is_file():
            return False
        dims = CAPTURE_DIMENSIONS.get(role)
        if dims is None:
            return False
        exp_w, exp_h = dims
        validated = result.validated
        if validated is not None:
            return (
                validated.width == exp_w
                and validated.height == exp_h
                and validated.format == ".png"
            )
        return CaptureSession._valid_still(path, role)

    @staticmethod
    def _valid_still(path: Path | None, role: str) -> bool:
        if path is None or path.suffix.lower() != ".png" or not path.is_file():
            return False
        dims = CAPTURE_DIMENSIONS.get(role)
        if dims is None:
            return False
        exp_w, exp_h = dims
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        valid = image is not None and image.shape[:2] == (exp_h, exp_w)
        del image
        return valid
