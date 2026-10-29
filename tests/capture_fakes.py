"""Shared camera and illumination fakes for capture-side tests."""
from __future__ import annotations

from camera_calibration import (
    ActivatedCameraMode,
    CalibrationQuality,
    LockedCameraControls,
    PhaseCameraCalibration,
)


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


class _FakeOutput:
    pin = 17
    is_on = False

    def on(self):
        self.is_on = True

    def off(self):
        self.is_on = False

    def close(self):
        self.off()


class _FakeBacklight:
    pin = 27

    def __init__(self):
        self.value = 0.0

    def set_brightness(self, value):
        self.value = value

    def off(self):
        self.value = 0.0

    def close(self):
        self.off()
