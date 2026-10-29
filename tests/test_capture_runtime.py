"""Camera/runtime contract tests that never import Picamera2."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from capture_runtime import (
    CaptureController,
    camera_settings_for,
    camera_settings_for_variant,
)
from capture_session import CaptureSession, CaptureState, SessionConfig
from capture_storage import CaptureStore
from constants import CAPTURE_DIMENSIONS, NATIVE_CAPTURE_DIMENSIONS
from illumination import IlluminationController


NOW = datetime(2026, 7, 1, 19, 30, 45, tzinfo=timezone.utc)


class SteppingClock:
    """Monotonic clock that advances by a fixed delta each call."""

    def __init__(self, start: float = 1000.0, step: float = 0.010):
        self._t = start
        self._step = step

    def __call__(self) -> float:
        value = self._t
        self._t += self._step
        return value


class FakeCamera:
    def __init__(self):
        self.calls = []

    def start_preview(self, *, settings, size, fps):
        self.calls.append(("preview", settings, size, fps))

    def set_preview_exposure(self, exposure_us):
        self.calls.append(("preview_exposure", exposure_us))

    def capture_still(self, path, *, settings, size):
        self.calls.append(("still", settings, size))
        assert cv2.imwrite(str(path), np.zeros((size[1], size[0]), dtype=np.uint8))

    def resume_preview(self):
        self.calls.append(("resume",))

    def close(self):
        self.calls.append(("close",))


class FailOnceCamera(FakeCamera):
    def capture_still(self, path, *, settings, size):
        if not any(call[0] == "still" for call in self.calls):
            self.calls.append(("still", settings, size))
            raise OSError("camera disconnected")
        super().capture_still(path, settings=settings, size=size)


class EventCamera(FakeCamera):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def capture_still(self, path, *, settings, size):
        self.events.append("camera")
        super().capture_still(path, settings=settings, size=size)


class EventOutput:
    def __init__(self, events):
        self.events = events
        self.is_on = False

    def on(self):
        self.events.append("on")
        self.is_on = True

    def off(self):
        self.events.append("off")
        self.is_on = False

    def close(self):
        self.events.append("close")


class FakeBacklight:
    pin = 27

    def __init__(self):
        self.value = 0.0

    def set_brightness(self, value):
        self.value = value

    def off(self):
        self.value = 0.0

    def close(self):
        pass


def _controller(session, tmp_path, *, camera=None, **kw):
    kw.setdefault("working_dir", tmp_path / "pending")
    return CaptureController(
        session=session,
        camera=camera if camera is not None else FakeCamera(),
        store=CaptureStore(tmp_path / "published"),
        **kw,
    )


def _empty() -> np.ndarray:
    return np.full((80, 120, 3), 180, dtype=np.uint8)


def _slide() -> np.ndarray:
    frame = _empty()
    cv2.rectangle(frame, (30, 25), (90, 55), (45, 45, 45), -1)
    return frame


def _ready_slide_session() -> CaptureSession:
    session = CaptureSession(SessionConfig(baseline_frames=2), mode="slide")
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    return session


def _ready_block_session() -> CaptureSession:
    session = CaptureSession(SessionConfig(baseline_frames=2), mode="block")
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    assert session.state is CaptureState.WAITING_FOR_SCAN
    scan = session.submit_scan("51151378")
    assert scan.accepted
    return session


def _trigger(controller, *, captured_at=NOW):
    controller.handle_frame(_slide(), now=3.0, captured_at=captured_at)
    controller.handle_frame(_slide(), now=3.1, captured_at=captured_at)
    return controller.handle_frame(_slide(), now=4.0, captured_at=captured_at)


@pytest.mark.parametrize("timeout_during_still", [False, True])
def test_led_only_capture_keeps_canonical_png_and_rejects_light_timeout(
    tmp_path, monkeypatch, timeout_during_still,
):
    monkeypatch.setitem(CAPTURE_DIMENSIONS, "block", (100, 80))
    empty = np.full((80, 100, 3), 2, dtype=np.uint8)
    block = empty.copy()
    block[25:55, 35:65] = 90
    session = CaptureSession(mode="block", led_only=True)
    session.install_locked_baseline(empty)
    for now, frame in ((0, block), (0.1, block), (0.7, block),
                       (0.8, empty), (1.4, empty)):
        session.accept_frame(frame, now=now)
    assert session.submit_scan("12345678").accepted
    light_on = True

    class Camera(FakeCamera):
        def capture_still(self, *args, **kwargs):
            nonlocal light_on
            super().capture_still(*args, **kwargs)
            if timeout_during_still:
                light_on = False

    light = SimpleNamespace(
        status=lambda: SimpleNamespace(physical_output=light_on),
        capture_metadata=lambda: {},
        capture_block=lambda capture, **_: (capture(), {}),
    )
    consumed = []
    controller = _controller(
        session, tmp_path, camera=Camera(), illumination=light,
        capture_consumer=lambda record: consumed.append(record),
    )
    record = controller.capture_now(captured_at=NOW)
    if timeout_during_still:
        assert record is None
        assert session.state is CaptureState.CAPTURE_ERROR
        assert not consumed
        assert not list((tmp_path / "published").glob("*.png"))
    else:
        assert record is not None
        assert record.block_id == "12345678"
        assert record.metadata["placement_detection"] == "dark_background_difference"
        assert cv2.imread(str(record.path)).shape[:2] == (80, 100)
        assert session.state is CaptureState.WAITING_FOR_REMOVAL


def test_capture_records_stage_timing_metadata(tmp_path):
    controller = _controller(
        _ready_slide_session(), tmp_path,
        clock=SteppingClock(start=100.0, step=0.025),
    )
    controller.start()
    record = _trigger(controller)
    assert record is not None
    meta = record.metadata
    for key in (
        "camera_capture_ms",
        "publish_ms",
        "session_accept_ms",
        "total_capture_ms",
        "final_file_size_bytes",
        "capture_mode",
    ):
        assert key in meta
    assert meta["capture_mode"] == "slide"
    assert meta["final_file_size_bytes"] > 0
    assert meta["camera_capture_ms"] >= 0
    assert meta["publish_ms"] >= 0
    assert meta["total_capture_ms"] >= meta["camera_capture_ms"]
    assert "consumer_ms" not in meta


def test_capture_records_consumer_decode_outbox_send_ms_when_present(tmp_path):
    """Slide consumer sub-durations become int-ms metadata keys."""
    def consumer(_record):
        return SimpleNamespace(
            success=True, decode_ms=12.4, outbox_ms=7.6, send_ms=41.2
        )

    controller = _controller(
        _ready_slide_session(), tmp_path, capture_consumer=consumer,
        clock=SteppingClock(start=100.0, step=0.025),
    )
    controller.start()
    record = _trigger(controller)

    assert record is not None
    assert record.metadata["consumer_decode_ms"] == 12
    assert record.metadata["consumer_outbox_ms"] == 8
    assert record.metadata["consumer_send_ms"] == 41


def test_capture_omits_consumer_split_keys_when_consumer_result_lacks_them(tmp_path):
    """A plain consumer result only gets the `consumer_ms` key."""
    controller = _controller(
        _ready_slide_session(), tmp_path,
        capture_consumer=lambda _record: SimpleNamespace(success=True),
        clock=SteppingClock(start=100.0, step=0.025),
    )
    controller.start()
    record = _trigger(controller)

    assert record is not None
    assert record.metadata["consumer_ms"] >= 0
    assert "consumer_decode_ms" not in record.metadata
    assert "consumer_outbox_ms" not in record.metadata
    assert "consumer_send_ms" not in record.metadata


def test_failed_capture_still_exposes_partial_stage_timings(tmp_path):
    session = _ready_slide_session()
    captured_results = []
    original_accept = session.accept_capture_result

    def spy_accept(result):
        captured_results.append(result)
        return original_accept(result)

    session.accept_capture_result = spy_accept

    controller = _controller(
        session, tmp_path, camera=FailOnceCamera(),
        clock=SteppingClock(start=100.0, step=0.025),
    )
    controller.start()
    assert _trigger(controller) is None
    assert len(captured_results) == 1
    result = captured_results[0]
    assert not result.ok
    meta = result.metadata
    assert meta["capture_mode"] == "slide"
    assert meta["total_capture_ms"] >= 0
    assert meta["camera_capture_ms"] >= 0
    assert controller.last_failure_timings is not None
    assert controller.last_failure_timings["capture_mode"] == "slide"
    assert "camera_capture_ms" in controller.last_failure_timings
    assert "total_capture_ms" in controller.last_failure_timings


def test_controller_uses_low_resolution_preview_at_ten_fps(tmp_path):
    camera = FakeCamera()
    controller = _controller(_ready_slide_session(), tmp_path, camera=camera)

    controller.start()

    assert camera.calls == [
        ("preview", camera_settings_for("slide"), (640, 480), 10.0)
    ]


def test_capture_orders_preview_still_publish_resume(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session()
    controller = _controller(session, tmp_path, camera=camera, working_dir=tmp_path)
    controller.start()

    record = _trigger(controller)

    assert [call[0] for call in camera.calls] == ["preview", "still", "resume"]
    # Slides are captured natively at their half-res dimensions.
    assert camera.calls[1][2] == (2028, 1520)
    assert record is session.last_capture
    assert record.path.name == "capture_000001_slide_20260701T193045Z.png"
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert not list(tmp_path.glob("pending-*.png"))


@pytest.mark.native_resolution
def test_block_capture_still_requests_full_sensor_resolution(tmp_path):
    camera = FakeCamera()
    session = _ready_block_session()
    controller = _controller(session, tmp_path, camera=camera)
    controller.start()

    record = _trigger(controller)

    assert [call[0] for call in camera.calls] == ["preview", "still", "resume"]
    assert camera.calls[1][2] == NATIVE_CAPTURE_DIMENSIONS
    assert record is session.last_capture


def test_controller_set_preview_variant_uses_variant_exposure(tmp_path):
    camera = FakeCamera()
    controller = _controller(
        CaptureSession(SessionConfig(), mode="block"), tmp_path, camera=camera
    )

    controller.set_preview_variant("neutral_oblique")

    assert camera.calls == [("preview_exposure", 150000)]

    with pytest.raises(ValueError, match="unknown triple-capture variant"):
        camera_settings_for_variant("block", "unknown")


def test_block_capture_uses_auto_illumination_bracket_and_records_metadata(tmp_path):
    events: list[object] = []
    light = IlluminationController(
        EventOutput(events), sleep=lambda seconds: events.append(("sleep", seconds))
    )
    events.clear()
    camera = EventCamera(events)
    controller = _controller(
        _ready_block_session(), tmp_path, camera=camera, illumination=light
    )

    record = _trigger(controller)

    assert record is not None
    assert events == ["on", ("sleep", 0.010), "camera", ("sleep", 0.005), "off"]
    assert record.metadata["illumination_auto_mode"] is True
    assert record.metadata["illumination_pin"] == 17
    assert record.metadata["illumination_lead_ms"] == 10
    assert record.metadata["illumination_tail_ms"] == 5
    assert record.metadata["illumination_led_on_duration_ms"] is not None
    assert [call[1].exposure_us for call in camera.calls if call[0] == "still"] == [
        33333
    ]


def test_triple_block_capture_keeps_sidecars_and_canonical_coax_image(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(CAPTURE_DIMENSIONS, "block", (120, 80))
    light = IlluminationController(
        EventOutput([]),
        backlight=FakeBacklight(),
        neutral_oblique=EventOutput([]),
        sleep=lambda _seconds: None,
    )
    camera = FakeCamera()
    controller = _controller(
        _ready_block_session(), tmp_path, camera=camera,
        illumination=light, triple_capture=True,
    )

    record = _trigger(controller)

    assert record is not None
    assert [call[0] for call in camera.calls] == [
        "still",
        "still",
        "still",
        "resume",
    ]
    assert record.metadata["triple_capture_enabled"] is True
    assert record.metadata["triple_capture_sequence"] == (
        "coax,backlight,neutral_oblique"
    )
    assert record.metadata["triple_capture_exposures_us"] == (
        "coax=33333,backlight=40000,neutral_oblique=150000"
    )
    assert [call[1].exposure_us for call in camera.calls if call[0] == "still"] == [
        33333,
        40000,
        150000,
    ]
    assert record.metadata["illumination_capture_mode"] == "triple"

    manifests = list((tmp_path / "triple_captures").glob("triple_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["sequence"] == ["coax", "backlight", "neutral_oblique"]
    assert manifest["block_id"] == "51151378"
    assert Path(manifest["variants"]["coax"]["path"]) == record.path.resolve()
    assert Path(manifest["variants"]["coax"]["path"]).is_file()
    assert Path(manifest["variants"]["backlight"]["path"]).is_file()
    assert Path(manifest["variants"]["neutral_oblique"]["path"]).is_file()


def test_slide_capture_preserves_manual_led_off_override(tmp_path):
    events: list[str] = []
    timers = []
    light = IlluminationController(
        EventOutput(events),
        timer_factory=lambda delay, callback: SimpleNamespace(
            daemon=False,
            start=lambda: None,
            cancel=lambda: timers.append("cancel"),
            callback=callback,
        ),
    )
    light.led_on()
    events.clear()
    controller = _controller(
        _ready_slide_session(), tmp_path, camera=EventCamera(events),
        illumination=light,
    )

    record = _trigger(controller)

    assert record is not None
    assert events == ["off", "camera"]
    assert timers == ["cancel"]
    assert light.auto_mode is False
    assert light.logical_output is False
    assert record.metadata["illumination_auto_mode"] is False
    assert record.metadata["illumination_led_on_duration_ms"] is None


def test_auto_slide_capture_uses_gpio_backlight_and_slide_exposure(tmp_path):
    events: list[str] = []
    backlight = FakeBacklight()
    light = IlluminationController(EventOutput(events), backlight=backlight)
    light.backlight_off()
    events.clear()
    controller = _controller(
        _ready_slide_session(), tmp_path, camera=EventCamera(events),
        illumination=light,
    )

    record = _trigger(controller)

    assert record is not None
    assert events == ["off", "camera"]
    assert light.auto_mode is True
    assert light.logical_output is False
    assert backlight.value == 0.5
    assert record.metadata["illumination_led_on_duration_ms"] is None
    assert [
        call[1].exposure_us
        for call in controller.camera.calls
        if call[0] == "still"
    ] == [20000]


def test_gpio_failure_enters_existing_capture_error_and_forces_led_off(tmp_path):
    class FailingOutput:
        def __init__(self):
            self.calls = []
            self.is_on = False

        def on(self):
            self.calls.append("on")
            raise OSError("GPIO unavailable")

        def off(self):
            self.calls.append("off")
            self.is_on = False

        def close(self):
            self.calls.append("close")

    output = FailingOutput()
    light = IlluminationController(output)
    output.calls.clear()
    session = _ready_block_session()
    controller = _controller(session, tmp_path, illumination=light)

    assert _trigger(controller) is None

    assert session.state is CaptureState.CAPTURE_ERROR
    assert controller.last_error is not None
    assert "GPIO unavailable" in controller.last_error
    assert output.calls == ["on", "off"]


def test_published_slide_is_forwarded_once_to_capture_consumer(tmp_path):
    session = _ready_slide_session()
    consumed = []
    controller = _controller(session, tmp_path, capture_consumer=consumed.append)

    _trigger(controller)
    controller.handle_frame(_slide(), now=5.0, captured_at=NOW)

    assert len(consumed) == 1
    assert consumed[0].role == "slide"
    assert consumed[0].path == session.last_capture.path
    assert consumed[0].captured_at == NOW


def test_failed_identity_decode_enters_reposition_after_capture_is_retained(tmp_path):
    session = _ready_slide_session()
    consumed = []

    def fail_decode(record):
        consumed.append(record)
        return SimpleNamespace(success=False)

    controller = _controller(session, tmp_path, capture_consumer=fail_decode)
    _trigger(controller)

    assert len(consumed) == 1
    assert consumed[0].path.is_file()
    assert session.state is CaptureState.REPOSITION_SLIDE
    assert session.drain_events()[-1].message == "Reposition slide"


def test_capture_error_resumes_preview_and_can_retry_same_specimen(tmp_path):
    camera = FailOnceCamera()
    session = _ready_slide_session()
    controller = _controller(session, tmp_path, camera=camera)
    controller.start()
    assert _trigger(controller) is None
    assert session.state is CaptureState.CAPTURE_ERROR
    assert camera.calls[-1] == ("resume",)

    record = controller.retry(captured_at=NOW)

    assert record.path.is_file()
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert [call[0] for call in camera.calls].count("still") == 2
    assert [call[0] for call in camera.calls].count("resume") == 2


def test_capture_now_shoots_immediately_through_the_normal_path(tmp_path):
    # CAPTURE NOW is a trigger override only: same capture/validate/publish path.
    session = _ready_slide_session()
    camera = FakeCamera()
    controller = _controller(session, tmp_path, camera=camera)
    controller.start()

    record = controller.capture_now(captured_at=NOW)

    assert record.path.is_file()
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert [call[0] for call in camera.calls].count("still") == 1
