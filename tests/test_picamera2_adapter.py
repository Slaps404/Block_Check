from __future__ import annotations

import threading

import numpy as np
import pytest

from camera_calibration import CameraCalibrationError
from capture_runtime import camera_settings_for
from picamera2_adapter import BASELINE_FRAMES, Picamera2Adapter


class _Request:
    def __init__(self, camera):
        self.camera = camera

    def make_array(self, _stream):
        self.camera.calls.append("baseline_frame")
        return np.full((2, 3, 3), 220, dtype=np.uint8)

    def release(self):
        pass


class _FakeCamera:
    sensor_resolution = (4056, 3040)
    sensor_modes = [
        {
            "size": (2028, 1520),
            "bit_depth": 12,
            "crop_limits": (0, 0, 4056, 3040),
        }
    ]

    def __init__(self, *, verification_exposure=None):
        self.calls = []
        self.started = False
        self.auto = False
        self.locked = False
        self.verification_exposure = verification_exposure
        self.applied_exposure = 12000
        self.applied_gain = 1.25
        self.applied_colour_gains = (2.0, 1.5)

    def create_preview_configuration(self, **kwargs):
        return kwargs

    def create_still_configuration(self, **kwargs):
        self.calls.append(("still_configuration", kwargs))
        return kwargs

    def switch_mode_and_capture_file(self, config, path, *, format):
        self.calls.append(("capture_file", config, path, format))

    def configure(self, _config):
        self.calls.append("configure")

    def start(self):
        self.started = True
        self.calls.append("start")

    def stop(self):
        self.started = False

    def set_controls(self, controls):
        self.calls.append(("controls", dict(controls)))
        if controls.get("AeEnable") is True:
            self.auto = True
            self.locked = False
        if controls.get("AeEnable") is False and "ExposureTime" in controls:
            self.auto = False
            self.locked = True
            self.applied_exposure = controls["ExposureTime"]
            self.applied_gain = controls["AnalogueGain"]
            self.applied_colour_gains = controls["ColourGains"]

    def capture_metadata(self):
        self.calls.append("metadata")
        exposure = (
            self.verification_exposure
            if self.locked and self.verification_exposure is not None
            else self.applied_exposure if self.locked else 12000
        )
        return {
            "ExposureTime": exposure,
            "AnalogueGain": self.applied_gain if self.locked else 1.25,
            "ColourGains": self.applied_colour_gains if self.locked else (2.0, 1.5),
            "FrameDuration": 33333,
        }

    def capture_request(self):
        return _Request(self)

    def capture_array(self, _stream):
        self.calls.append("tuning_frame")
        return np.full((2, 3, 3), 220, dtype=np.uint8)


def _adapter(camera):
    adapter = Picamera2Adapter.__new__(Picamera2Adapter)
    adapter._camera = camera
    adapter._preview_config = None
    adapter._denoise_minimal = "minimal"
    adapter._started = False
    adapter._camera_lock = threading.RLock()
    adapter._locked_by_mode = {}
    adapter._active_mode = None
    return adapter


class _OccupiedRequest(_Request):
    def make_array(self, _stream):
        self.camera.calls.append("baseline_frame")
        # Bright AND chromatic: saturation (255-170)/255*255=85 > 40,
        # so occupancy remains the fail-closed safety gate.
        frame = np.empty((2, 3, 3), dtype=np.uint8)
        frame[..., 0] = 255  # R
        frame[..., 1] = 170  # G
        frame[..., 2] = 255  # B
        return frame


class _OccupiedCamera(_FakeCamera):
    def capture_request(self):
        return _OccupiedRequest(self)


class _ChromaticPatchRequest(_Request):
    def make_array(self, _stream):
        self.camera.calls.append("baseline_frame")
        frame = np.full(
            (10, 10, 3), self.camera.background_level, dtype=np.uint8
        )
        frame[0, :6] = self.camera.chromatic_pixel
        return frame


class _ChromaticPatchCamera(_FakeCamera):
    def __init__(self, *, background_level, chromatic_pixel):
        super().__init__()
        self.background_level = background_level
        self.chromatic_pixel = chromatic_pixel

    def capture_request(self):
        return _ChromaticPatchRequest(self)


def test_activate_mode_fails_closed_when_capture_area_is_occupied():
    camera = _OccupiedCamera()
    adapter = _adapter(camera)

    with pytest.raises(CameraCalibrationError, match="capture area occupied"):
        adapter.activate_mode("block")

    assert "block" not in adapter._locked_by_mode
    assert adapter._active_mode is None


def test_dim_slide_chromatic_occupancy_rejects_and_clears_active_lock():
    camera = _ChromaticPatchCamera(
        background_level=156,
        chromatic_pixel=(156, 40, 156),
    )
    adapter = _adapter(camera)

    with pytest.raises(CameraCalibrationError, match="capture area occupied") as exc:
        adapter.activate_mode("slide")

    assert exc.value.diagnostics["chromatic_fraction"] == pytest.approx(0.06)
    assert "slide" not in adapter._locked_by_mode
    assert adapter._active_mode is None


def test_clipped_block_chromatic_occupancy_rejects_and_clears_active_lock():
    camera = _ChromaticPatchCamera(
        background_level=255,
        chromatic_pixel=(255, 0, 255),
    )
    adapter = _adapter(camera)

    with pytest.raises(CameraCalibrationError, match="capture area occupied") as exc:
        adapter.activate_mode("block")

    assert exc.value.diagnostics["chromatic_fraction"] == pytest.approx(0.06)
    assert "block" not in adapter._locked_by_mode
    assert adapter._active_mode is None


def test_unoccupied_dim_slide_activates_and_records_luma_quality():
    class _UnderbrightRequest(_Request):
        def make_array(self, _stream):
            self.camera.calls.append("baseline_frame")
            return np.full((2, 3, 3), 156, dtype=np.uint8)

    class _UnderbrightCamera(_FakeCamera):
        def capture_request(self):
            return _UnderbrightRequest(self)

    activated = _adapter(_UnderbrightCamera()).activate_mode("slide")

    assert activated.calibration.quality.background_luma_median == 156.0
    assert activated.calibration.quality.clipped_high_fraction == 0.0
    assert activated.calibration.quality.clipped_low_fraction == 0.0
    settings = camera_settings_for("slide")
    assert activated.calibration.controls.exposure_time_us == settings.exposure_us
    assert activated.calibration.controls.analogue_gain == settings.analogue_gain


def test_unoccupied_clipped_block_activates_and_records_clipping_quality():
    class _OverbrightRequest(_Request):
        def make_array(self, _stream):
            self.camera.calls.append("baseline_frame")
            return np.full((2, 3, 3), 255, dtype=np.uint8)

    class _OverbrightCamera(_FakeCamera):
        def capture_request(self):
            return _OverbrightRequest(self)

    activated = _adapter(_OverbrightCamera()).activate_mode("block")

    assert activated.calibration.quality.background_luma_median == 255.0
    assert activated.calibration.quality.clipped_high_fraction == 1.0
    assert activated.calibration.quality.clipped_low_fraction == 0.0
    settings = camera_settings_for("block")
    assert activated.calibration.controls.exposure_time_us == settings.exposure_us
    assert activated.calibration.controls.analogue_gain == settings.analogue_gain


def test_activate_mode_locks_role_controls_before_flushing_and_baseline():
    camera = _FakeCamera()

    activated = _adapter(camera).activate_mode("block")

    control_updates = [call[1] for call in camera.calls if isinstance(call, tuple)]
    assert not any(controls.get("AeEnable") is True for controls in control_updates)
    assert not any(controls.get("AwbEnable") is True for controls in control_updates)
    lock_index = next(
        i
        for i, controls in enumerate(control_updates)
        if controls.get("AeEnable") is False and "ExposureTime" in controls
    )
    assert control_updates[lock_index]["AwbEnable"] is False
    assert control_updates[lock_index]["NoiseReductionMode"] == "minimal"
    settings = camera_settings_for("block")
    assert control_updates[lock_index]["ExposureTime"] == settings.exposure_us
    first_baseline = camera.calls.index("baseline_frame")
    last_control = max(i for i, call in enumerate(camera.calls) if isinstance(call, tuple))
    assert camera.calls[last_control + 1:first_baseline].count("metadata") >= 4
    assert camera.calls.count("baseline_frame") == BASELINE_FRAMES
    assert activated.calibration.controls.exposure_time_us == settings.exposure_us
    assert "tuning_frame" not in camera.calls
    assert np.all(activated.baseline == 220)


def test_slide_activation_locks_fixed_role_controls_without_enabling_auto():
    camera = _FakeCamera()

    activated = _adapter(camera).activate_mode("slide")

    settings = camera_settings_for("slide")
    control_updates = [
        call[1]
        for call in camera.calls
        if isinstance(call, tuple) and call[0] == "controls"
    ]
    assert not any(controls.get("AeEnable") is True for controls in control_updates)
    assert not any(controls.get("AwbEnable") is True for controls in control_updates)
    assert activated.calibration.controls.exposure_time_us == settings.exposure_us
    assert activated.calibration.controls.analogue_gain == settings.analogue_gain
    assert activated.calibration.controls.colour_gains == settings.colour_gains


def test_activate_mode_fails_closed_without_collecting_baseline_when_verification_fails():
    camera = _FakeCamera(verification_exposure=20000)
    adapter = _adapter(camera)

    with pytest.raises(CameraCalibrationError, match="did not take effect"):
        adapter.activate_mode("slide")

    assert "baseline_frame" not in camera.calls
    assert "slide" not in adapter._locked_by_mode
    assert adapter._active_mode is None


def test_preview_and_still_share_a_reentrant_camera_lock():
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.activate_mode("block")

    assert isinstance(adapter._camera_lock, type(threading.RLock()))
    assert adapter.preview_frame().shape == (2, 3, 3)


@pytest.mark.parametrize("mode", ["block", "slide"])
def test_locked_still_capture_keeps_minimal_denoise(mode):
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.activate_mode(mode)

    adapter.capture_still(
        "still.png",
        settings=camera_settings_for(mode),
        size=(4056, 3040),
    )

    still_call = next(
        call for call in camera.calls
        if isinstance(call, tuple) and call[0] == "still_configuration"
    )
    assert still_call[1]["controls"]["NoiseReductionMode"] == "minimal"
    assert still_call[1]["controls"]["ExposureTime"] == camera_settings_for(mode).exposure_us


def test_restarting_preview_preserves_the_active_locked_controls():
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.activate_mode("block")

    adapter.start_preview(
        settings=camera_settings_for("block"), size=(640, 480), fps=10.0
    )

    controls = adapter._preview_config["controls"]
    assert controls["NoiseReductionMode"] == "minimal"
    assert controls["AeEnable"] is False
    assert controls["AwbEnable"] is False
    assert controls["ExposureTime"] == camera_settings_for("block").exposure_us


def test_activating_slide_mode_starts_preview_with_slide_role_controls():
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.activate_mode("block")

    adapter.activate_mode("slide")

    controls = adapter._preview_config["controls"]
    slide_settings = camera_settings_for("slide")
    assert controls["ExposureTime"] == slide_settings.exposure_us
    assert controls["AnalogueGain"] == slide_settings.analogue_gain
    assert controls["ColourGains"] == slide_settings.colour_gains
    assert controls["Contrast"] == slide_settings.contrast
    assert controls["Sharpness"] == slide_settings.sharpness
    assert controls["NoiseReductionMode"] == "minimal"


def test_starting_slide_preview_does_not_inherit_active_block_lock():
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.activate_mode("block")
    slide_settings = camera_settings_for("slide")

    adapter.start_preview(settings=slide_settings, size=(640, 480), fps=10.0)

    controls = adapter._preview_config["controls"]
    assert controls["ExposureTime"] == slide_settings.exposure_us
    assert controls["AnalogueGain"] == slide_settings.analogue_gain
    assert controls["ColourGains"] == slide_settings.colour_gains
    assert "FrameDurationLimits" not in controls
