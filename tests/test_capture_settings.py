"""Live-editable capture settings: store, runtime wiring, console verbs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import pi_session as run_pi_session
from capture_runtime import ROLE_CAMERA_SETTINGS, camera_settings_for
from capture_session import CaptureMode, CaptureState, SessionConfig
from capture_settings import CaptureSettings
from illumination import IlluminationController
from session.console import COMMAND_ARITY, COMMAND_HELP, _COMMANDS, dispatch


NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Store: registry, validation, persistence
# --------------------------------------------------------------------------


def test_defaults_come_from_role_camera_settings_and_session_config():
    settings = CaptureSettings()
    block = ROLE_CAMERA_SETTINGS[CaptureMode.BLOCK]
    slide = ROLE_CAMERA_SETTINGS[CaptureMode.SLIDE]

    assert settings.value("block.exposure_us") == block.exposure_us
    assert settings.value("slide.exposure_us") == slide.exposure_us
    assert settings.value("block.colour_gain_red") == block.colour_gains[0]
    assert settings.value("slide.colour_gain_blue") == slide.colour_gains[1]
    assert settings.value("slide.sharpness") == slide.sharpness
    assert settings.value("detection.presence_threshold") == (
        SessionConfig().presence_threshold
    )
    assert settings.value("detection.removal_duration") == (
        SessionConfig().removal_duration
    )
    assert settings.value("lights.backlight_level") == 50
    assert settings.value("lights.led_auto") is True
    assert settings.camera_settings("block") == block


def test_snapshot_rows_are_json_friendly_and_complete():
    rows = CaptureSettings().snapshot()

    json.dumps(rows)
    by_name = {row["name"]: row for row in rows}
    assert {"block.exposure_us", "slide.analogue_gain", "detection.motion_threshold",
            "lights.neutral_level", "lights.led_auto"} <= set(by_name)
    row = by_name["block.exposure_us"]
    assert row == {
        "name": "block.exposure_us", "group": "block", "applies": "camera",
        "type": "int", "value": row["default"], "default": row["default"],
        "min": 100, "max": 200000, "step": 100, "changed": False,
    }
    assert by_name["lights.led_auto"]["type"] == "bool"
    assert by_name["detection.stable_duration"]["applies"] == "detection"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("block.exposure_us", "50", "block.exposure_us must be between 100 and 200000"),
        ("block.exposure_us", 300000, "between 100 and 200000"),
        ("block.exposure_us", "fast", "block.exposure_us must be a whole number"),
        ("block.exposure_us", "1.5", "must be a whole number"),
        ("slide.analogue_gain", "0.5", "slide.analogue_gain must be between 1 and 16"),
        ("detection.motion_threshold", "abc", "must be a number"),
        ("detection.motion_threshold", True, "must be a number"),
        ("lights.led_auto", "maybe", "lights.led_auto must be on or off"),
        ("lights.backlight_level", "101", "between 0 and 100"),
        ("block.iso", "100", "unknown setting: block.iso"),
    ],
)
def test_validation_rejects_bad_names_types_and_ranges(name, value, message):
    with pytest.raises(ValueError, match=message):
        CaptureSettings().validate(name, value)


def test_validation_coerces_console_strings():
    settings = CaptureSettings()

    assert settings.validate("block.exposure_us", "20000") == 20000
    assert settings.validate("slide.contrast", "2") == 2.0
    assert settings.validate("lights.led_auto", "off") is False
    assert settings.validate("lights.led_auto", "ON") is True


def test_save_and_load_round_trip_stores_only_overrides(tmp_path):
    path = tmp_path / "capture_settings.json"
    settings = CaptureSettings(path)

    settings.set("block.exposure_us", "20000")
    settings.set("detection.motion_threshold", 0.05)
    settings.set("slide.sharpness", settings.value("slide.sharpness"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overrides"] == {
        "block.exposure_us": 20000, "detection.motion_threshold": 0.05,
    }
    reloaded = CaptureSettings(path)
    assert reloaded.value("block.exposure_us") == 20000
    assert reloaded.camera_settings("block").exposure_us == 20000
    assert reloaded.camera_settings("slide") == ROLE_CAMERA_SETTINGS[CaptureMode.SLIDE]
    assert reloaded.session_config(SessionConfig()).motion_threshold == 0.05


@pytest.mark.parametrize(
    "body", ["{not json", "[]", '{"overrides": 7}'], ids=["garbled", "list", "bad-overrides"]
)
def test_corrupt_file_falls_back_to_defaults_with_a_warning(tmp_path, body):
    path = tmp_path / "capture_settings.json"
    path.write_text(body, encoding="utf-8")
    warnings: list[str] = []

    settings = CaptureSettings(path, warn=warnings.append)

    assert settings.overrides() == {}
    assert warnings and "using defaults" in warnings[0]


def test_one_bad_saved_value_is_skipped_and_the_rest_load(tmp_path):
    path = tmp_path / "capture_settings.json"
    path.write_text(json.dumps({"schema_version": 1, "overrides": {
        "block.exposure_us": 999999, "block.contrast": 2.0, "nope": 1,
    }}), encoding="utf-8")
    warnings: list[str] = []

    settings = CaptureSettings(path, warn=warnings.append)

    assert settings.overrides() == {"block.contrast": 2.0}
    assert len(warnings) == 2


def test_reset_restores_defaults_and_deletes_the_file(tmp_path):
    path = tmp_path / "capture_settings.json"
    settings = CaptureSettings(path)
    settings.set("block.exposure_us", 20000)

    settings.reset()

    assert settings.overrides() == {}
    assert settings.value("block.exposure_us") == camera_settings_for("block").exposure_us
    assert not path.exists()


def test_detection_defaults_follow_the_launch_config():
    base = SessionConfig(stable_duration=0.0, baseline_frames=1)
    settings = CaptureSettings(session_defaults=base)

    assert settings.value("detection.stable_duration") == 0.0
    assert settings.session_config(base) == base


# --------------------------------------------------------------------------
# Camera adapter: role settings seam
# --------------------------------------------------------------------------


def test_triple_variants_keep_their_exposures_on_top_of_base_settings():
    from capture_runtime import camera_settings_for_variant
    from dataclasses import replace

    base = replace(camera_settings_for("block"), exposure_us=20000, analogue_gain=2.0)

    coax = camera_settings_for_variant("block", "coax", base_settings=base)
    backlight = camera_settings_for_variant("block", "backlight", base_settings=base)

    assert coax == base
    assert backlight.exposure_us == 40000
    assert backlight.analogue_gain == 2.0


def test_picamera_adapter_activates_with_runtime_role_settings():
    from dataclasses import replace
    from test_picamera2_adapter import _FakeCamera, _adapter

    camera = _FakeCamera()
    adapter = _adapter(camera)
    adapter.role_settings = {
        "block": replace(camera_settings_for("block"), exposure_us=20000),
    }

    activated = adapter.activate_mode("block")

    assert activated.calibration.controls.exposure_time_us == 20000


# --------------------------------------------------------------------------
# Runtime wiring
# --------------------------------------------------------------------------


class _Workflow:
    """Minimal block-phase workflow that counts calibrations."""

    capture_paused = False

    def __init__(self):
        self.calibrations: list[str] = []

    def snapshot(self):
        return SimpleNamespace(phase="blocks")

    def awaiting_capture_blocks(self):
        return ()

    def prepare_empty_backlight(self, mode):
        self.calibrations.append(mode)

    def baseline_for(self, _mode):
        if not self.calibrations:
            raise RuntimeError("no baseline")
        return np.full((20, 20, 3), 200, dtype=np.uint8)

    def scan_block(self, _block_id):
        return SimpleNamespace(accepted=True)

    def restore_slide_capture_session(self, _capture_session):
        pass


class _Camera:
    def close(self):
        pass


def _runtime(tmp_path, **kwargs):
    kwargs.setdefault("capture_root", tmp_path / "captures")
    kwargs.setdefault("capture_settings_path", tmp_path / "capture_settings.json")
    workflow = kwargs.pop("workflow", None) or _Workflow()
    camera = kwargs.pop("camera", None) or _Camera()
    return run_pi_session.PiCaptureRuntime(workflow, camera, **kwargs)


def _on_scan_block(runtime):
    runtime.confirm_empty()
    assert runtime.capture_session.state is CaptureState.WAITING_FOR_SCAN


def test_detection_change_reaches_the_live_capture_session(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    runtime.scan_block("12345678")

    result = runtime.set_setting("detection.motion_threshold", "0.05")

    assert runtime.capture_session.config.motion_threshold == 0.05
    assert runtime.session_config.motion_threshold == 0.05
    assert result["setting"]["value"] == 0.05
    assert result["recalibrated"] is False
    runtime._build_controller(CaptureMode.SLIDE)
    assert runtime.capture_session.config.motion_threshold == 0.05


def test_camera_change_on_scan_block_recalibrates_and_persists(tmp_path):
    camera = _Camera()
    runtime = _runtime(tmp_path, camera=camera)
    _on_scan_block(runtime)

    result = runtime.set_setting("block.exposure_us", "20000")

    assert runtime.workflow.calibrations == ["block", "block"]
    assert result["recalibrated"] is True
    assert "recalibrating" in result["message"]
    assert camera.role_settings["block"].exposure_us == 20000
    assert runtime.controller.settings.exposure_us == 20000
    assert CaptureSettings(tmp_path / "capture_settings.json").value(
        "block.exposure_us"
    ) == 20000


def test_camera_change_is_refused_and_not_persisted_off_scan_block(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    runtime.scan_block("12345678")
    assert runtime.capture_session.state is CaptureState.EMPTY

    with pytest.raises(RuntimeError, match="only change on Scan Block"):
        runtime.set_setting("block.exposure_us", 20000)

    assert runtime.capture_settings.value("block.exposure_us") != 20000
    assert not (tmp_path / "capture_settings.json").exists()
    assert runtime.workflow.calibrations == ["block"]


def test_camera_change_before_first_calibration_waits_for_confirm_empty(tmp_path):
    runtime = _runtime(tmp_path)
    assert runtime.capture_session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION

    result = runtime.set_setting("block.contrast", 2.0)

    assert result["recalibrated"] is False
    assert runtime.workflow.calibrations == []
    assert runtime.controller.settings.contrast == 2.0


def test_other_role_camera_change_applies_at_its_next_calibration(tmp_path):
    camera = _Camera()
    runtime = _runtime(tmp_path, camera=camera)
    _on_scan_block(runtime)
    runtime.scan_block("12345678")

    result = runtime.set_setting("slide.exposure_us", 25000)

    assert result["recalibrated"] is False
    assert camera.role_settings["slide"].exposure_us == 25000
    assert runtime.workflow.calibrations == ["block"]


def test_saved_settings_are_applied_at_startup(tmp_path):
    path = tmp_path / "capture_settings.json"
    CaptureSettings(path).set("block.sharpness", 3.0)
    CaptureSettings(path).set("detection.stable_duration", 1.5)
    camera = _Camera()

    runtime = _runtime(tmp_path, camera=camera)

    assert camera.role_settings["block"].sharpness == 3.0
    assert runtime.controller.settings.sharpness == 3.0
    assert runtime.capture_session.config.stable_duration == 1.5


class _FakeOutput:
    is_on = False

    def on(self):
        self.is_on = True

    def off(self):
        self.is_on = False

    def close(self):
        pass


class _FakeBacklight:
    pin = 27

    def __init__(self):
        self.value = 0.0

    def set_brightness(self, value):
        self.value = value

    def off(self):
        self.value = 0.0

    def close(self):
        pass


def test_light_settings_apply_immediately(tmp_path):
    backlight = _FakeBacklight()
    light = IlluminationController(_FakeOutput(), backlight=backlight)
    runtime = _runtime(tmp_path, illumination=light)

    runtime.set_setting("lights.backlight_level", 30)
    runtime.set_setting("lights.led_auto", "off")

    assert backlight.value == pytest.approx(0.30)
    assert light.mode == "auto_off"
    assert runtime.capture_settings.value("lights.backlight_level") == 30


def test_led_auto_is_refused_under_triple_capture(tmp_path):
    light = IlluminationController(_FakeOutput(), backlight=_FakeBacklight())
    runtime = _runtime(tmp_path, illumination=light, triple_capture=True)

    with pytest.raises(RuntimeError, match="triple capture"):
        runtime.set_setting("lights.led_auto", "off")

    assert light.mode == "auto"


def test_reset_settings_restores_defaults_and_recalibrates_on_scan_block(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    runtime.set_setting("block.exposure_us", 20000)
    runtime.set_setting("detection.motion_threshold", 0.05)

    result = runtime.reset_settings()

    assert runtime.capture_settings.overrides() == {}
    assert runtime.capture_session.config.motion_threshold == SessionConfig().motion_threshold
    assert runtime.controller.settings.exposure_us == camera_settings_for("block").exposure_us
    assert result["recalibrated"] is True
    assert not (tmp_path / "capture_settings.json").exists()


def test_settings_status_marks_every_row_editable_on_scan_block(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)

    status = runtime.settings_status()

    json.dumps(status)
    assert status["active_role"] == "block"
    assert all(row["editable"] for row in status["settings"])
    assert all(row["locked_reason"] is None for row in status["settings"])


def test_settings_status_locks_active_role_camera_rows_off_scan_block(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    runtime.scan_block("12345678")

    rows = {row["name"]: row for row in runtime.settings_status()["settings"]}

    # Same message set_setting raises, so the UI and the refusal cannot drift.
    with pytest.raises(RuntimeError) as refusal:
        runtime.set_setting("block.exposure_us", 20000)
    assert rows["block.exposure_us"]["editable"] is False
    assert rows["block.exposure_us"]["locked_reason"] == str(refusal.value)
    assert rows["slide.exposure_us"]["editable"] is True
    assert rows["detection.motion_threshold"]["editable"] is True


def test_settings_status_locks_led_auto_under_triple_capture(tmp_path):
    light = IlluminationController(_FakeOutput(), backlight=_FakeBacklight())
    runtime = _runtime(tmp_path, illumination=light, triple_capture=True)

    rows = {row["name"]: row for row in runtime.settings_status()["settings"]}

    assert rows["lights.led_auto"]["editable"] is False
    assert "triple capture" in rows["lights.led_auto"]["locked_reason"]
    assert rows["lights.backlight_level"]["editable"] is True


def test_reset_settings_is_refused_when_camera_reset_needs_scan_block(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    runtime.set_setting("block.exposure_us", 20000)
    runtime.scan_block("12345678")

    with pytest.raises(RuntimeError, match="Scan Block"):
        runtime.reset_settings()

    assert runtime.capture_settings.value("block.exposure_us") == 20000


class _StillCamera:
    def __init__(self):
        self.still_settings = []

    def start_preview(self, **_kwargs):
        pass

    def capture_still(self, path, *, settings, size):
        self.still_settings.append(settings)
        cv2.imwrite(str(path), np.zeros((size[1], size[0]), dtype=np.uint8))

    def resume_preview(self):
        pass

    def close(self):
        pass


def _empty_frame():
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _specimen_frame():
    frame = _empty_frame()
    cv2.rectangle(frame, (30, 25), (90, 55), (45, 45, 45), -1)
    return frame


def test_capture_saved_line_records_the_effective_settings(tmp_path):
    from action_logger import ActionLogger
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        LoopbackCaptureReceiver,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=NOW)
    log_path = tmp_path / "actions.log"
    camera = _StillCamera()
    with LoopbackCaptureReceiver(store) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(tmp_path / "framing.json"),
        )
        runtime = _runtime(
            tmp_path,
            workflow=workflow,
            camera=camera,
            session_config=SessionConfig(baseline_frames=1, stable_duration=0.0),
            review_captures=True,
            action_logger=ActionLogger(
                log_path, session_number=session.number, print_sink=lambda _l: None
            ),
        )
        runtime.start(background=False)
        runtime.confirm_empty()
        runtime.set_setting("block.exposure_us", 20000)
        runtime.process_frame(_empty_frame(), now=0.0, captured_at=NOW)
        runtime.scan_block("51151378")
        runtime.process_frame(_specimen_frame(), now=1.0, captured_at=NOW)
        runtime.process_frame(_specimen_frame(), now=2.0, captured_at=NOW)
        runtime.close()

    assert camera.still_settings[-1].exposure_us == 20000
    saved = [
        line for line in log_path.read_text(encoding="utf-8").splitlines()
        if "event=capture_saved" in line
    ]
    assert saved
    field = next(t for t in saved[0].split("  ") if t.startswith("capture_settings="))
    recorded = json.loads(field.split("=", 1)[1])
    assert recorded["block.exposure_us"] == 20000
    assert "detection.motion_threshold" in recorded
    assert "lights.led_auto" in recorded
    assert not any(name.startswith("slide.") for name in recorded)


# --------------------------------------------------------------------------
# Console verbs
# --------------------------------------------------------------------------


def test_settings_verbs_exist_in_both_tables_with_arity():
    assert {"settings", "set", "reset_settings"} <= set(_COMMANDS)
    assert set(COMMAND_HELP) == set(_COMMANDS)
    assert COMMAND_ARITY["set"] == 2
    assert COMMAND_ARITY["settings"] == 0


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args)) or name


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("settings", ("settings_snapshot", ())),
        ("set block.exposure_us 20000", ("set_setting", ("block.exposure_us", "20000"))),
        ("reset_settings", ("reset_settings", ())),
    ],
)
def test_settings_verbs_dispatch_to_the_runtime(line, expected):
    runtime = _Recorder()
    command, *args = line.split()

    dispatch(runtime, command, *args)

    assert runtime.calls == [expected]


def test_run_one_command_renders_settings_and_rejections(tmp_path):
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)

    listing = run_pi_session.run_one_command(runtime, "settings")
    changed = run_pi_session.run_one_command(runtime, "set block.exposure_us 20000")
    rejected = run_pi_session.run_one_command(runtime, "set block.exposure_us 5")

    assert "block.exposure_us" in listing and "100..200000" in listing
    assert "recalibrating" in changed
    assert rejected == "Rejected: block.exposure_us must be between 100 and 200000"
    relisted = run_pi_session.run_one_command(runtime, "settings")
    line = next(row for row in relisted.splitlines() if "block.exposure_us" in row)
    assert "20000" in line and "*" in line
