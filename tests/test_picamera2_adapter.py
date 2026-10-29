from __future__ import annotations

import importlib
import sys
import threading

import numpy as np
import pytest

from camera_calibration import CameraCalibrationError
from constants import CALIBRATION_MIN_BACKGROUND_LUMA
from capture_runtime import camera_settings_for
from occupancy import detect_occupancy
from picamera2_adapter import BASELINE_FRAMES, Picamera2Adapter


class _Request:
    def __init__(self, camera):
        self.camera = camera

    def make_array(self, _stream):
        self.camera.calls.append("baseline_frame")
        frame = self.camera.frame
        return frame(self.camera) if callable(frame) else frame.copy()

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

    def __init__(self, *, verification_exposure=None, frame=None):
        self.calls = []
        self.started = False
        self.auto = False
        self.locked = False
        self.frames = 0
        self.frame = (
            np.full((2, 3, 3), 220, dtype=np.uint8) if frame is None else frame
        )
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
        assert not self.started
        self.calls.append("configure")

    def start(self):
        assert not self.started
        self.started = True
        self.calls.append("start")

    def stop(self):
        assert self.started
        self.started = False
        self.calls.append("stop")

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


def _camera(frame, **kwargs):
    """Fake camera whose baseline frames are `frame` (array or fn(camera))."""
    return _FakeCamera(frame=frame, **kwargs)


def _uniform(level, shape=(2, 3, 3)):
    return np.full(shape, level, dtype=np.uint8)


def _chromatic_patch(background_level, chromatic_pixel):
    frame = _uniform(background_level, (10, 10, 3))
    frame[0, :6] = chromatic_pixel
    return frame


def _occupied_frame():
    # Bright AND chromatic: saturation 85 > 40, so occupancy stays the gate.
    frame = np.empty((2, 3, 3), dtype=np.uint8)
    frame[..., 0] = 255  # R
    frame[..., 1] = 170  # G
    frame[..., 2] = 255  # B
    return frame


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


def test_importing_pi_adapter_on_windows_does_not_import_picamera2():
    importlib.import_module("picamera2_adapter")

    assert "picamera2" not in sys.modules


def test_led_only_accepts_dark_block_background_but_not_dark_slide():
    adapter = _adapter(_camera(_uniform(2, (20, 30, 3))))
    adapter.led_only_blocks = True
    activated = adapter.activate_mode("block")
    assert activated.baseline.mean() == 2
    assert adapter.block_presence_noise_floor == 8
    assert adapter.calibration_metadata()["placement_illumination"] == "led_only"
    with pytest.raises(CameraCalibrationError, match="too dark"):
        adapter.activate_mode("slide")


def test_led_only_rejects_unstable_empty_background():
    def flicker(camera):
        camera.frames += 1
        return _uniform((camera.frames % 2) * 100, (20, 30, 3))

    adapter = _adapter(_camera(flicker))
    adapter.led_only_blocks = True
    with pytest.raises(CameraCalibrationError, match="unstable"):
        adapter.activate_mode("block")
    assert adapter._active_mode is None


def test_occupied_looking_field_still_calibrates_as_the_empty_baseline():
    # A tilted camera sees a colored table; whatever is seen becomes "empty".
    adapter = _adapter(_camera(_occupied_frame()))

    activated = adapter.activate_mode("block")

    assert adapter._active_mode == "block"
    assert detect_occupancy(activated.baseline).occupied
    assert float(np.mean(activated.baseline[..., 1])) == pytest.approx(170.0)


@pytest.mark.parametrize(
    ("mode", "background_level", "chromatic_pixel"),
    [("slide", 156, (156, 40, 156)), ("block", 255, (255, 0, 255))],
    ids=["dim-slide", "clipped-block"],
)
def test_chromatic_patch_does_not_block_calibration(
    mode, background_level, chromatic_pixel
):
    adapter = _adapter(_camera(_chromatic_patch(background_level, chromatic_pixel)))

    adapter.activate_mode(mode)

    assert adapter._active_mode == mode


@pytest.mark.parametrize(
    ("mode", "level", "clipped_high"),
    [("slide", 156, 0.0), ("block", 255, 1.0)],
    ids=["dim-slide", "clipped-block"],
)
def test_unoccupied_field_activates_and_records_luma_quality(mode, level, clipped_high):
    activated = _adapter(_camera(_uniform(level))).activate_mode(mode)

    quality = activated.calibration.quality
    assert quality.background_luma_median == float(level)
    assert quality.clipped_high_fraction == clipped_high
    assert quality.clipped_low_fraction == 0.0
    settings = camera_settings_for(mode)
    assert activated.calibration.controls.exposure_time_us == settings.exposure_us
    assert activated.calibration.controls.analogue_gain == settings.analogue_gain


@pytest.mark.parametrize("level", [2, 0])
def test_activate_mode_rejects_a_field_too_dark_to_detect_anything(level):
    # A dark baseline would lock but never see a block arrive.
    adapter = _adapter(_camera(_uniform(level)))

    with pytest.raises(CameraCalibrationError, match="capture area too dark") as exc:
        adapter.activate_mode("block")

    assert exc.value.diagnostics["background_luma_median"] == float(level)
    assert (
        exc.value.diagnostics["minimum_background_luma"]
        == CALIBRATION_MIN_BACKGROUND_LUMA
    )
    assert "block" not in adapter._locked_by_mode
    assert adapter._active_mode is None


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
    # Derived so the sentinel can never collide with a retuned slide exposure.
    camera = _FakeCamera(verification_exposure=camera_settings_for("slide").exposure_us * 2)
    adapter = _adapter(camera)

    with pytest.raises(CameraCalibrationError, match="did not take effect"):
        adapter.activate_mode("slide")

    assert "baseline_frame" not in camera.calls
    assert "slide" not in adapter._locked_by_mode
    assert adapter._active_mode is None


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


def test_restarting_preview_stops_before_reconfiguring():
    camera = _FakeCamera()
    adapter = _adapter(camera)

    adapter.start_preview(settings=camera_settings_for("block"), size=(640, 480), fps=10)
    adapter.start_preview(settings=camera_settings_for("slide"), size=(640, 480), fps=10)

    # The fake asserts configure/start never run while the camera is started.
    assert camera.calls.count("start") == 2
    assert camera.calls.index("stop") < len(camera.calls) - 1


def test_preview_uses_smallest_full_sensor_field_of_view_mode():
    camera = _FakeCamera()
    camera.sensor_modes = [
        {"size": (1332, 990), "bit_depth": 10, "crop_limits": (696, 528, 2664, 1980)},
        {"size": (2028, 1520), "bit_depth": 12, "crop_limits": (0, 0, 4056, 3040)},
        {"size": (4056, 3040), "bit_depth": 12, "crop_limits": (0, 0, 4056, 3040)},
    ]
    adapter = _adapter(camera)

    adapter.start_preview(settings=camera_settings_for("block"), size=(640, 480), fps=10)

    assert adapter._preview_config["main"]["size"] == (640, 480)
    assert adapter._preview_config["sensor"] == {
        "output_size": (2028, 1520),
        "bit_depth": 12,
    }


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


def test_set_preview_exposure_updates_running_preview_without_restarting_camera():
    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.start_preview(
        settings=camera_settings_for("block"), size=(640, 480), fps=10.0
    )

    adapter.set_preview_exposure(150000)

    assert camera.calls[-1] == ("controls", {"ExposureTime": 150000})
    assert adapter._preview_config["controls"]["ExposureTime"] == 150000
    assert "configure" not in camera.calls[3:]


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
