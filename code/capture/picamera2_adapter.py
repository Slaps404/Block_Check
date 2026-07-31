"""Raspberry Pi camera adapter; Pi-only imports stay inside the class."""
from __future__ import annotations

from pathlib import Path
import threading
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from camera_calibration import (
    ActivatedCameraMode,
    CalibrationQuality,
    CameraCalibrationError,
    LockedCameraControls,
    PhaseCameraCalibration,
)
from capture_runtime import (
    PREVIEW_FPS,
    PREVIEW_SIZE,
    CameraSettings,
    camera_settings_for,
)
from occupancy import detect_occupancy


LOCK_FLUSH_FRAMES = 4
LOCK_VERIFY_FRAMES = 4
BASELINE_FRAMES = 10
CONTROL_RELATIVE_TOLERANCE = 0.02


class Picamera2Adapter:
    def __init__(self):
        from libcamera import controls
        from picamera2 import Picamera2  # Pi-only dependency

        self._camera = Picamera2()
        self._preview_config = None
        # rpicam's `--denoise cdn_off`: retain spatial denoise while disabling
        # colour and temporal denoise.  HighQuality would re-enable Pi 5 TDN.
        self._denoise_minimal = controls.draft.NoiseReductionModeEnum.Minimal
        self._started = False
        self._camera_lock = threading.RLock()
        self._locked_by_mode: dict[str, LockedCameraControls] = {}
        self._active_mode: str | None = None

    def start_preview(self, *, settings, size, fps):
        with self._lock():
            return self._start_preview_locked(settings=settings, size=size, fps=fps)

    def _start_preview_locked(self, *, settings, size, fps):
        if self._started:
            self._camera.stop()
            self._started = False
        sensor_size = tuple(self._camera.sensor_resolution)
        full_field_modes = [
            mode
            for mode in self._camera.sensor_modes
            if tuple(mode.get("crop_limits", ())) == (0, 0, *sensor_size)
        ]
        sensor_mode = min(
            full_field_modes,
            key=lambda mode: mode["size"][0] * mode["size"][1],
            default=None,
        )
        if sensor_mode is None:
            raise RuntimeError(
                f"no sensor mode covers the full sensor area {sensor_size}"
            )
        self._preview_config = self._camera.create_preview_configuration(
            main={"size": size, "format": "RGB888"},
            # Without an explicit mode, Picamera2 may satisfy a small main
            # stream from a center crop. Choose the smallest full-field mode,
            # then keep the processed array at the inexpensive 640x480 size.
            sensor={
                "output_size": sensor_mode["size"],
                "bit_depth": sensor_mode["bit_depth"],
            },
            controls={
                **self._effective_controls(settings),
                "FrameRate": fps,
            },
            buffer_count=4,
        )
        self._camera.configure(self._preview_config)
        self._camera.start()
        self._started = True

    def preview_frame(self):
        with self._lock():
            request = self._camera.capture_request()
            try:
                return request.make_array("main").copy()
            finally:
                request.release()

    def capture_still(self, path: Path, *, settings, size):
        with self._lock():
            controls = self._effective_controls(settings)
            still_config = self._camera.create_still_configuration(
                main={"size": size, "format": "RGB888"},
                controls=controls,
                buffer_count=2,
            )
            self._camera.switch_mode_and_capture_file(
                still_config, str(path), format="png"
            )
            if self._active_mode is not None:
                self._camera.set_controls(self._locked_controls_dict())

    def resume_preview(self):
        """No-op: switch_mode_and_capture_file already restores and restarts the
        preview configuration before returning (per the Picamera2 manual), so
        the camera is back in preview mode by the time capture_still returns.
        Kept to satisfy the CameraAdapter contract and let fake adapters assert
        preview-to-still-to-preview ordering. Do not add a second switch here."""

    def close(self):
        with self._lock():
            if self._started:
                self._camera.stop()
                self._started = False
            self._camera.close()

    def _lock_role_controls(self, mode: str, *, settings: CameraSettings):
        """Apply and verify the fixed controls used to tune the CV pipeline."""
        if mode not in {"block", "slide"}:
            raise ValueError(f"unknown camera mode: {mode}")
        with self._lock():
            locked = LockedCameraControls(
                exposure_time_us=settings.exposure_us,
                analogue_gain=settings.analogue_gain,
                colour_gains=settings.colour_gains,
            )
            try:
                self._camera.set_controls(self._controls(settings))
                self._discard_metadata(LOCK_FLUSH_FRAMES)
                verification = tuple(
                    self._camera.capture_metadata()
                    for _ in range(LOCK_VERIFY_FRAMES)
                )
                self._verify_lock(mode, locked, verification)
            except CameraCalibrationError:
                self._locked_by_mode.pop(mode, None)
                if self._active_mode == mode:
                    self._active_mode = None
                raise
            except Exception as exc:
                self._locked_by_mode.pop(mode, None)
                if self._active_mode == mode:
                    self._active_mode = None
                raise CameraCalibrationError(mode, str(exc), {}) from exc

            self._locked_by_mode[mode] = locked
            self._active_mode = mode
            quality = CalibrationQuality(
                stable=True,
                sample_count=len(verification),
                settling_frames=LOCK_FLUSH_FRAMES,
                exposure_cv=self._cv(verification, "ExposureTime"),
                gain_cv=self._cv(verification, "AnalogueGain"),
                red_gain_cv=self._colour_cv(verification, 0),
                blue_gain_cv=self._colour_cv(verification, 1),
                background_luma_median=0.0,
                clipped_high_fraction=0.0,
                clipped_low_fraction=0.0,
            )
            return PhaseCameraCalibration(
                mode=mode,
                controls=locked,
                quality=quality,
                metadata_samples=verification,
            )

    def activate_mode(self, mode: str) -> ActivatedCameraMode:
        """Configure fixed controls, verify them, then collect the baseline."""
        settings = camera_settings_for(mode)
        with self._lock():
            # A fresh role calibration must start from that role's settings.
            # Otherwise _effective_controls can merge the previous role's lock
            # (for example, block exposure) into the new slide preview config.
            self._active_mode = None
            self._start_preview_locked(
                settings=settings, size=PREVIEW_SIZE, fps=PREVIEW_FPS
            )
            calibration = self._lock_role_controls(mode, settings=settings)
            frames = [
                self.preview_frame().astype(np.float32)
                for _ in range(BASELINE_FRAMES)
            ]
            stack = np.stack(frames)
            baseline = np.mean(stack, axis=0)
            luma = np.mean(stack, axis=-1) if stack.ndim == 4 else stack
            height, width = luma.shape[1:]
            roi = luma[:, height // 5:height * 4 // 5, width // 5:width * 4 // 5]
            luma_median = float(np.median(roi))
            clipped_high = float(np.mean(roi >= 254))
            clipped_low = float(np.mean(roi <= 1))
            occupancy = detect_occupancy(baseline)
            if occupancy.occupied:
                self._invalidate_mode(mode)
                raise CameraCalibrationError(
                    mode, "capture area occupied",
                    {
                        "chromatic_fraction": occupancy.chromatic_fraction,
                        "occupancy_sat_min": occupancy.sat_min,
                        "occupancy_area_frac_max": occupancy.area_frac_max,
                    },
                )
            calibration = replace(
                calibration,
                quality=replace(
                    calibration.quality,
                    background_luma_median=luma_median,
                    clipped_high_fraction=clipped_high,
                    clipped_low_fraction=clipped_low,
                ),
                calibrated_at=datetime.now(timezone.utc),
                calibration_id=uuid4().hex,
            )
            return ActivatedCameraMode(calibration=calibration, baseline=baseline)

    def calibration_metadata(self) -> dict[str, object]:
        """Small capture-side fingerprint of the controls active on every stream."""
        with self._lock():
            controls = self._locked_by_mode.get(self._active_mode)
            if controls is None or self._active_mode is None:
                return {}
            return {
                "camera_mode": self._active_mode,
                "exposure_time_us": controls.exposure_time_us,
                "analogue_gain": controls.analogue_gain,
                "colour_gains": controls.colour_gains,
                "frame_duration_us": controls.frame_duration_us,
            }

    def _invalidate_mode(self, mode):
        self._locked_by_mode.pop(mode, None)
        if self._active_mode == mode:
            self._active_mode = None

    def _verify_lock(self, mode, locked, samples):
        expected = {
            "ExposureTime": locked.exposure_time_us,
            "AnalogueGain": locked.analogue_gain,
            "ColourGains": locked.colour_gains,
        }
        for sample in samples:
            for key, wanted in expected.items():
                if key not in sample:
                    raise CameraCalibrationError(
                        mode, f"lock verification missing {key}", {"sample": sample}
                    )
                actual = sample[key]
                pairs = (
                    zip(actual, wanted)
                    if key == "ColourGains"
                    else ((actual, wanted),)
                )
                outside_tolerance = any(
                    abs(float(got) - float(target))
                    > max(abs(float(target)), 1.0) * CONTROL_RELATIVE_TOLERANCE
                    for got, target in pairs
                )
                if outside_tolerance:
                    raise CameraCalibrationError(
                        mode,
                        f"locked {key} did not take effect",
                        {"expected": wanted, "actual": actual},
                    )

    def _discard_metadata(self, count):
        for _ in range(count):
            self._camera.capture_metadata()

    @staticmethod
    def _cv(samples, key):
        values = np.asarray([s[key] for s in samples], dtype=float)
        return float(np.std(values) / max(abs(float(np.mean(values))), 1e-9))

    @staticmethod
    def _colour_cv(samples, index):
        values = np.asarray([s["ColourGains"][index] for s in samples], dtype=float)
        return float(np.std(values) / max(abs(float(np.mean(values))), 1e-9))

    def _locked_controls_dict(self, controls=None):
        controls = controls or self._locked_by_mode.get(self._active_mode)
        if controls is None:
            return None
        values = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": controls.exposure_time_us,
            "AnalogueGain": controls.analogue_gain,
            "ColourGains": controls.colour_gains,
        }
        if controls.frame_duration_us is not None:
            values["FrameDurationLimits"] = (
                controls.frame_duration_us,
                controls.frame_duration_us,
            )
        return values

    def _lock(self):
        if not hasattr(self, "_camera_lock"):
            self._camera_lock = threading.RLock()
        if not hasattr(self, "_locked_by_mode"):
            self._locked_by_mode = {}
        if not hasattr(self, "_active_mode"):
            self._active_mode = None
        return self._camera_lock

    def _effective_controls(self, settings: CameraSettings) -> dict:
        """Apply a lock only when it belongs to the requested camera role."""
        role_controls = self._controls(settings)
        if (
            self._active_mode is None
            or settings != camera_settings_for(self._active_mode)
        ):
            return role_controls
        return {
            **role_controls,
            **(self._locked_controls_dict() or {}),
        }

    def _controls(self, settings: CameraSettings) -> dict:
        return {
            "ExposureTime": settings.exposure_us,
            "Contrast": settings.contrast,
            "Sharpness": settings.sharpness,
            "NoiseReductionMode": self._denoise_minimal,
            "AnalogueGain": settings.analogue_gain,
            "ColourGains": settings.colour_gains,
            "AeEnable": settings.ae_enabled,
            "AwbEnable": settings.awb_enabled,
        }
