"""Orchestration between the capture state machine, camera, and storage."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol
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
from constants import (
    CAPTURE_DIMENSIONS,
    NATIVE_CAPTURE_DIMENSIONS,
    TRIPLE_CAPTURE_EXPOSURES_US,
    TRIPLE_CAPTURE_VARIANTS,
)
from illumination import ILLUMINATION_METADATA_KEYS, Illumination, IlluminationError
from session.atomic_io import atomic_json as _atomic_json, sha256 as _sha256


PREVIEW_SIZE = (640, 480)
PREVIEW_FPS = 10.0

# Keys written into SuccessfulCapture.metadata by CaptureController._capture and
# forwarded onto ActionLogger capture_saved lines.
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
    "triple_capture_enabled",
    "triple_capture_group_id",
    "triple_capture_manifest",
    "triple_capture_sequence",
    "triple_capture_exposures_us",
    *ILLUMINATION_METADATA_KEYS,
)

# Slide consumer sub-durations (decode / outbox-write / send-to-main-computer),
# duck-typed off the consumer result. Absent when the consumer result carries
# no decode stage (e.g. block-mode's `UploadReceipt`).
_CONSUMER_SPLIT_ATTRS: tuple[tuple[str, str], ...] = (
    ("consumer_decode_ms", "decode_ms"),
    ("consumer_outbox_ms", "outbox_ms"),
    ("consumer_send_ms", "send_ms"),
)

# Settling roll-up columns, merged into the same stage_fields as
# CAPTURE_STAGE_TIMING_KEYS from a SettlingObserver summary. Absent for
# block-mode captures and any capture that never passed through SETTLING
# (e.g. a recapture-after-error path).
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
        # The backlight is PWM-driven. 20 ms spans two PWM cycles, which
        # averages out rolling-shutter bands (visible at ~8.3 ms) without
        # washing out the slide field.
        exposure_us=20000,
        contrast=1.8,
        sharpness=1.5,
    ),
    CaptureMode.BLOCK: CameraSettings(
        exposure_us=TRIPLE_CAPTURE_EXPOSURES_US["coax"],
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


def camera_settings_for_variant(
    mode: CaptureMode | str,
    variant: str,
    *,
    base_settings: CameraSettings | None = None,
) -> CameraSettings:
    """Return camera settings for one optional triple-capture light variant."""
    mode_value = CaptureMode(mode)
    settings = base_settings or camera_settings_for(mode_value)
    if mode_value is not CaptureMode.BLOCK:
        return settings
    try:
        exposure_us = TRIPLE_CAPTURE_EXPOSURES_US[variant]
    except KeyError as exc:
        raise ValueError(f"unknown triple-capture variant: {variant}") from exc
    if variant == "coax":
        # Coax is the base block role; its exposure is the live block setting.
        return settings
    return replace(settings, exposure_us=exposure_us)


class CameraAdapter(Protocol):
    def start_preview(
        self, *, settings: CameraSettings, size: tuple[int, int], fps: float
    ) -> None: ...

    def preview_frame(self) -> np.ndarray: ...

    def set_preview_exposure(self, exposure_us: int) -> None: ...

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
        sleep: Callable[[float], None] | None = None,
        illumination: Illumination | None = None,
        triple_capture: bool = False,
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
        self._sleep = sleep
        self.illumination = illumination
        self.triple_capture = triple_capture
        self.settings = self.configuration.role_settings[session.mode.value]
        self.last_error: str | None = None
        self.last_frame_result = None
        self.pending_record: CaptureRecord | None = None
        self.last_failure_timings: dict[str, object] | None = None

    def start(self) -> None:
        self.camera.start_preview(
            settings=self.settings,
            size=self.configuration.preview_size,
            fps=self.configuration.preview_fps,
        )

    def settings_for_variant(self, variant: str) -> CameraSettings:
        """Use calibrated settings for a block triple-capture variant."""
        return camera_settings_for_variant(
            self.session.mode,
            variant,
            base_settings=self.settings,
        )

    def set_preview_variant(self, variant: str) -> None:
        """Set the live preview exposure for one illumination variant."""
        setter = getattr(self.camera, "set_preview_exposure", None)
        if callable(setter):
            setter(self.settings_for_variant(variant).exposure_us)

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

    def capture_now(self, *, captured_at: datetime):
        """Operator override: request a capture without waiting to settle."""
        self.session.capture_now()
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
        timings: dict[str, object] = {"capture_mode": self.session.mode.value}
        total_start = self._clock()
        triple_paths: dict[str, Path] | None = None
        triple_manifest_path: Path | None = None
        triple_capture_published = False
        try:
            triple_enabled = (
                self.triple_capture and self.session.mode is CaptureMode.BLOCK
            )
            if triple_enabled:
                triple_group_id = uuid4().hex
                triple_dir = self.working_dir.parent / "triple_captures"
                triple_dir.mkdir(parents=True, exist_ok=True)
                triple_paths = {
                    "coax": pending_path,
                    "backlight": (
                        triple_dir / f"triple_{triple_group_id}_backlight.png"
                    ),
                    "neutral_oblique": (
                        triple_dir
                        / f"triple_{triple_group_id}_neutral_oblique.png"
                    ),
                }
                triple_manifest_path = (
                    triple_dir / f"triple_{triple_group_id}.json"
                )
                timings.update(
                    {
                        "triple_capture_enabled": True,
                        "triple_capture_group_id": triple_group_id,
                        "triple_capture_manifest": str(
                            triple_manifest_path.resolve()
                        ),
                        "triple_capture_sequence": ",".join(
                            TRIPLE_CAPTURE_VARIANTS
                        ),
                        "triple_capture_exposures_us": ",".join(
                            f"{variant}={self.settings_for_variant(variant).exposure_us}"
                            for variant in TRIPLE_CAPTURE_VARIANTS
                        ),
                    }
                )

            def capture_still(
                path: Path = pending_path,
                *,
                settings: CameraSettings | None = None,
            ) -> None:
                camera_start = self._clock()
                try:
                    if self.session.led_only and (
                        self.illumination is None
                        or not self.illumination.status().physical_output
                    ):
                        raise IlluminationError("LED is off before block capture")
                    self.camera.capture_still(
                        path,
                        settings=self.settings if settings is None else settings,
                        # Request each role's still at its own resolution
                        # (slides native half-res, blocks full sensor res)
                        # rather than downscaling at publish. Unknown roles
                        # fall back to still_dimensions.
                        size=CAPTURE_DIMENSIONS.get(
                            self.session.mode.value, self.configuration.still_dimensions
                        ),
                    )
                    if self.session.led_only and not self.illumination.status().physical_output:
                        raise IlluminationError("LED switched off during block capture")
                finally:
                    timings["camera_capture_ms"] = self._elapsed_ms(camera_start)

            if triple_enabled and self.illumination is None:
                raise IlluminationError(
                    "triple capture requires illumination control"
                )
            if self.illumination is not None and self.session.mode is CaptureMode.BLOCK:
                timings.update(self.illumination.capture_metadata())
                if triple_enabled:
                    if not callable(
                        getattr(self.illumination, "capture_triple", None)
                    ):
                        raise IlluminationError(
                            "triple capture illumination control is unavailable"
                        )
                    assert triple_paths is not None
                    _, _, _, illumination_metadata = (
                        self.illumination.capture_triple(
                            lambda: capture_still(
                                triple_paths["coax"],
                                settings=self.settings_for_variant("coax"),
                            ),
                            lambda: capture_still(
                                triple_paths["backlight"],
                                settings=self.settings_for_variant("backlight"),
                            ),
                            lambda: capture_still(
                                triple_paths["neutral_oblique"],
                                settings=self.settings_for_variant(
                                    "neutral_oblique"
                                ),
                            ),
                            sleep=self._sleep,
                            clock=self._clock,
                        )
                    )
                else:
                    _, illumination_metadata = self.illumination.capture_block(
                        capture_still,
                        sleep=self._sleep,
                        clock=self._clock,
                    )
                timings.update(illumination_metadata)
            else:
                if (
                    self.illumination is not None
                    and self.session.mode is CaptureMode.SLIDE
                ):
                    self.illumination.prepare_slide_capture()
                    capture_still(settings=self.settings_for_variant("coax"))
                    timings.update(self.illumination.capture_metadata())
                else:
                    capture_still()
            if triple_enabled:
                assert triple_paths is not None
                assert triple_manifest_path is not None
                self._write_triple_manifest(
                    triple_manifest_path,
                    triple_paths,
                    block_id=self.session.pending_block_id,
                    captured_at=captured_at,
                    group_id=str(timings["triple_capture_group_id"]),
                )
            timings.setdefault("camera_capture_ms", 0)
            publish_start = self._clock()
            capture_metadata = self._capture_metadata()
            capture_metadata.update(timings)
            record = self.store.publish(
                pending_path,
                self.session.mode.value,
                block_id=self.session.pending_block_id,
                captured_at=captured_at,
                metadata=capture_metadata,
            )
            triple_capture_published = triple_enabled
            if triple_enabled:
                self._mark_triple_manifest_published(
                    triple_manifest_path, record.path
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
        except (OSError, PublicationError, ValueError, IlluminationError) as exc:
            self.last_error = str(exc)
            timings.setdefault("camera_capture_ms", 0)
            timings["total_capture_ms"] = self._elapsed_ms(total_start)
            self.last_failure_timings = dict(timings)
            self.session.accept_capture_result(
                CaptureResult.failure(str(exc), metadata=timings)
            )
            return None
        finally:
            if triple_paths is not None and not triple_capture_published:
                self._cleanup_triple_files(
                    triple_paths.values(), triple_manifest_path
                )
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
            "placement_detection": (
                "dark_background_difference" if self.session.led_only
                else "normalized_difference"
            ),
            **({
                "led_noise_floor": self.session._noise_floor,
                "led_presence_fraction": self.session.config.led_presence_fraction,
                "led_motion_fraction": self.session.config.led_motion_fraction,
            } if self.session.led_only else {}),
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
        self._cleanup_triple_manifest(self.pending_record.metadata)
        self.pending_record.path.unlink(missing_ok=True)
        self.pending_record = None

    def _write_triple_manifest(
        self,
        manifest_path: Path,
        paths: Mapping[str, Path],
        *,
        block_id: str | None,
        captured_at: datetime,
        group_id: str,
    ) -> None:
        variants = {
            name: {
                "path": str(path.resolve()),
                "checksum": _sha256(path),
                "state": "captured",
            }
            for name, path in paths.items()
        }
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "capture_group_id": group_id,
                "block_id": block_id,
                "captured_at": captured_at.isoformat(),
                "canonical_variant": "coax",
                "sequence": list(TRIPLE_CAPTURE_VARIANTS),
                "variants": variants,
            },
        )

    def _mark_triple_manifest_published(
        self, manifest_path: Path | None, published_path: Path
    ) -> None:
        if manifest_path is None:
            return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            variants = payload.get("variants", {})
            coax = variants.get("coax") if isinstance(variants, Mapping) else None
            if isinstance(coax, dict):
                coax["path"] = str(published_path.resolve())
                coax["checksum"] = _sha256(published_path)
                coax["state"] = "published"
            _atomic_json(manifest_path, payload)
        except (OSError, TypeError, ValueError, KeyError):
            # Canonical verifier publication must not fail because an
            # auxiliary manifest annotation could not be written. The raw
            # sidecars remain on disk for later collection.
            return

    def _cleanup_triple_files(
        self, paths: Iterable[Path], manifest_path: Path | None
    ) -> None:
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        if manifest_path is not None:
            try:
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_triple_manifest(self, metadata: Mapping[str, object]) -> None:
        raw_manifest_path = metadata.get("triple_capture_manifest")
        if not isinstance(raw_manifest_path, str):
            return
        manifest_path = Path(raw_manifest_path)
        triple_dir = (self.working_dir.parent / "triple_captures").resolve()
        if manifest_path.resolve().parent != triple_dir:
            return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            variants = payload.get("variants", {})
            if isinstance(variants, Mapping):
                for variant in variants.values():
                    if not isinstance(variant, Mapping):
                        continue
                    raw_path = variant.get("path")
                    if isinstance(raw_path, str):
                        path = Path(raw_path)
                        if path.resolve().parent == triple_dir:
                            path.unlink(missing_ok=True)
        except (OSError, TypeError, ValueError):
            pass
        try:
            manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
