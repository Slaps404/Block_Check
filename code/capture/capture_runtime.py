"""Orchestration between the capture state machine, camera, and storage."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np

from capture_session import (
    CaptureMode,
    CaptureResult,
    CaptureSession,
    CaptureState,
    SessionConfig,
    SuccessfulCapture,
)
from capture_storage import CaptureRecord, CaptureStore, PublicationError, ValidatedStill
from constants import CAPTURE_DIMENSIONS, NATIVE_CAPTURE_DIMENSIONS


PREVIEW_SIZE = (640, 480)
PREVIEW_FPS = 10.0

# Keys written into SuccessfulCapture.metadata by CaptureController._capture and
# forwarded onto ActionLogger capture_saved lines (Decision #4).
CAPTURE_STAGE_TIMING_KEYS = (
    "camera_capture_ms",
    "publish_ms",
    "consumer_ms",
    "consumer_decode_ms",
    "consumer_outbox_ms",
    "consumer_send_ms",
    "session_accept_ms",
    "total_capture_ms",
    "final_file_size_bytes",
    "capture_mode",
)

# Slide consumer sub-durations (#171: decode / outbox-write /
# send-to-main-computer), duck-typed off the consumer result exactly like the
# existing `getattr(consumer_result, "success", True)` read. Absent when the
# consumer result carries no decode stage (e.g. block-mode's `UploadReceipt`).
_CONSUMER_SPLIT_ATTRS: tuple[tuple[str, str], ...] = (
    ("consumer_decode_ms", "decode_ms"),
    ("consumer_outbox_ms", "outbox_ms"),
    ("consumer_send_ms", "send_ms"),
)

# Settling roll-up columns (#172), a sibling of CAPTURE_STAGE_TIMING_KEYS.
# Populated by run_pi_session.PiCaptureRuntime from a SettlingObserver
# summary and merged into the same stage_fields dict the timing keys ride
# in on; absent for block-mode captures and any capture that never passed
# through SETTLING (e.g. a recapture-after-error path).
SETTLING_STAGE_KEYS = (
    "settling_duration_ms",
    "settling_resets",
    "settling_max_motion",
)


@dataclass(frozen=True)
class CameraSettings:
    exposure_us: int
    contrast: float
    sharpness: float
    # libcamera Minimal is rpicam's `--denoise cdn_off`: spatial denoise only.
    denoise: str = "Minimal"
    analogue_gain: float = 1.0
    colour_gains: tuple[float, float] = (3.08, 1.492)
    ae_enabled: bool = False
    awb_enabled: bool = False


ROLE_CAMERA_SETTINGS = {
    CaptureMode.SLIDE: CameraSettings(
        exposure_us=8333,
        contrast=1.8,
        sharpness=1.5,
    ),
    CaptureMode.BLOCK: CameraSettings(
        exposure_us=33333,
        contrast=1.4,
        sharpness=1.6,
    ),
}


@dataclass(frozen=True)
class CaptureConfiguration:
    session: SessionConfig = field(default_factory=SessionConfig)
    role_settings: dict[str, CameraSettings] = field(
        default_factory=lambda: {
            mode.value: settings for mode, settings in ROLE_CAMERA_SETTINGS.items()
        }
    )
    preview_size: tuple[int, int] = PREVIEW_SIZE
    preview_fps: float = PREVIEW_FPS
    still_dimensions: tuple[int, int] = NATIVE_CAPTURE_DIMENSIONS


def camera_settings_for(mode: CaptureMode | str) -> CameraSettings:
    return ROLE_CAMERA_SETTINGS[CaptureMode(mode)]


class CameraAdapter(Protocol):
    def start_preview(
        self, *, settings: CameraSettings, size: tuple[int, int], fps: float
    ) -> None: ...

    def preview_frame(self) -> np.ndarray: ...

    def capture_still(
        self, path: Path, *, settings: CameraSettings, size: tuple[int, int]
    ) -> None: ...

    def resume_preview(self) -> None: ...

    def close(self) -> None: ...


class CaptureController:
    """Drive core actions without putting state transitions in the camera."""

    def __init__(
        self,
        *,
        session: CaptureSession,
        camera: CameraAdapter,
        store: CaptureStore,
        working_dir: str | Path,
        configuration: CaptureConfiguration | None = None,
        capture_consumer: Callable[[CaptureRecord], object] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.session = session
        self.camera = camera
        self.store = store
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.configuration = configuration or CaptureConfiguration(
            session=session.config
        )
        self.capture_consumer = capture_consumer
        self._clock = clock or time.perf_counter
        self.settings = self.configuration.role_settings[session.mode.value]
        self.last_error: str | None = None
        self.last_frame_result = None
        self.pending_record: CaptureRecord | None = None
        self.last_failure_timings: dict[str, int | str] | None = None

    def start(self) -> None:
        self.camera.start_preview(
            settings=self.settings,
            size=self.configuration.preview_size,
            fps=self.configuration.preview_fps,
        )

    def handle_frame(
        self, frame: np.ndarray, *, now: float, captured_at: datetime
    ):
        result = self.session.accept_frame(frame, now=now)
        self.last_frame_result = result
        if not result.capture_requested:
            return None
        return self._capture(captured_at)

    def retry(self, *, captured_at: datetime):
        if self.session.state is CaptureState.AWAITING_ACCEPT:
            self._discard_pending_record()
        self.session.retry_capture()
        return self._capture(captured_at)

    def accept_capture(self):
        if self.pending_record is None:
            raise RuntimeError("no pending capture is held for review")
        record = self.pending_record
        consumer_result = None
        if self.capture_consumer is not None:
            consumer_result = self.capture_consumer(record)
        held = self.session.accept_capture()
        self.pending_record = None
        if (
            self.session.mode is CaptureMode.SLIDE
            and getattr(consumer_result, "success", True) is False
        ):
            self.session.mark_slide_unreadable()
        return held

    def close(self) -> None:
        self.camera.close()

    def _capture(self, captured_at: datetime):
        pending_path = self.working_dir / f"pending-{uuid4().hex}.png"
        timings: dict[str, int | str] = {"capture_mode": self.session.mode.value}
        total_start = self._clock()
        camera_start = self._clock()
        try:
            self.camera.capture_still(
                pending_path,
                settings=self.settings,
                # Tier-3 (#186 follow-up): request each role's still at its
                # own configured resolution (slides shoot native half-res,
                # blocks stay full sensor res) instead of always capturing
                # full-res and downscaling at publish. Falls back to the
                # flat still_dimensions default if the role is unknown.
                size=CAPTURE_DIMENSIONS.get(
                    self.session.mode.value, self.configuration.still_dimensions
                ),
            )
            timings["camera_capture_ms"] = self._elapsed_ms(camera_start)
            publish_start = self._clock()
            record = self.store.publish(
                pending_path,
                self.session.mode.value,
                block_id=self.session.pending_block_id,
                captured_at=captured_at,
                metadata=self._capture_metadata(),
            )
            timings["publish_ms"] = self._elapsed_ms(publish_start)
            timings["final_file_size_bytes"] = record.path.stat().st_size
            if self.session.review_captures:
                self.pending_record = record
                metadata = dict(record.metadata)
                metadata.update(
                    counter=record.counter,
                    role=record.role,
                    block_id=record.block_id,
                )
                metadata.update(timings)
                self._accept_with_timing(
                    record.path, metadata, total_start, validated=record.validated
                )
                self.last_error = None
                self.last_failure_timings = None
                return self.session.last_capture
            consumer_result = None
            if self.capture_consumer is not None:
                consumer_start = self._clock()
                consumer_result = self.capture_consumer(record)
                timings["consumer_ms"] = self._elapsed_ms(consumer_start)
            for metadata_key, attr in _CONSUMER_SPLIT_ATTRS:
                value = getattr(consumer_result, attr, None)
                if value is not None:
                    timings[metadata_key] = int(round(value))
            capture_id = getattr(consumer_result, "capture_id", None)
            if capture_id:
                timings["slide_capture_id"] = str(capture_id)
            metadata = dict(record.metadata)
            metadata.update(
                counter=record.counter,
                role=record.role,
                block_id=record.block_id,
            )
            metadata.update(timings)
            self._accept_with_timing(
                record.path, metadata, total_start, validated=record.validated
            )
            if (
                self.session.mode is CaptureMode.SLIDE
                and getattr(consumer_result, "success", True) is False
            ):
                self.session.mark_slide_unreadable()
            self.last_error = None
            self.last_failure_timings = None
            return self.session.last_capture
        except (OSError, PublicationError, ValueError) as exc:
            self.last_error = str(exc)
            timings.setdefault("camera_capture_ms", self._elapsed_ms(camera_start))
            timings["total_capture_ms"] = self._elapsed_ms(total_start)
            self.last_failure_timings = dict(timings)
            self.session.accept_capture_result(
                CaptureResult.failure(str(exc), metadata=timings)
            )
            return None
        finally:
            pending_path.unlink(missing_ok=True)
            self.camera.resume_preview()

    def _elapsed_ms(self, start: float) -> int:
        return int(round((self._clock() - start) * 1000))

    def _accept_with_timing(
        self,
        path: Path,
        metadata: dict,
        total_start: float,
        *,
        validated: ValidatedStill | None = None,
    ) -> None:
        accept_start = self._clock()
        self.session.accept_capture_result(
            CaptureResult.success(path, metadata=metadata, validated=validated)
        )
        session_accept_ms = self._elapsed_ms(accept_start)
        total_capture_ms = self._elapsed_ms(total_start)
        last = self.session.last_capture
        if last is None or last.path != path:
            return
        annotated = dict(last.metadata)
        annotated["session_accept_ms"] = session_accept_ms
        annotated["total_capture_ms"] = total_capture_ms
        self.session.last_capture = SuccessfulCapture(
            path=last.path,
            role=last.role,
            block_id=last.block_id,
            metadata=annotated,
        )

    def _capture_metadata(self) -> dict:
        x0, y0, x1, y1 = self.session.config.roi
        preview_width, preview_height = self.configuration.preview_size
        calibration_metadata = getattr(self.camera, "calibration_metadata", lambda: {})()
        return {
            **asdict(self.settings),
            **calibration_metadata,
            "preview_size": self.configuration.preview_size,
            "preview_fps": self.configuration.preview_fps,
            "still_dimensions": self.configuration.still_dimensions,
            "roi": self.session.config.roi,
            "roi_pixels": (
                round(x0 * preview_width),
                round(y0 * preview_height),
                round(x1 * preview_width),
                round(y1 * preview_height),
            ),
            "presence_threshold": self.session.config.presence_threshold,
            "motion_threshold": self.session.config.motion_threshold,
            "baseline_frames": self.session.config.baseline_frames,
            "stable_duration": self.session.config.stable_duration,
            "removal_duration": self.session.config.removal_duration,
        }

    def _discard_pending_record(self) -> None:
        if self.pending_record is None:
            return
        self.pending_record.path.unlink(missing_ok=True)
        self.pending_record = None
