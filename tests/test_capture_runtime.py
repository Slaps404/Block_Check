"""Camera/runtime contract tests that never import Picamera2 (issue #85)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import cv2
import numpy as np

from capture_runtime import (
    CameraSettings,
    CaptureController,
    camera_settings_for,
)
from capture_session import CaptureSession, CaptureState, SessionConfig
from capture_storage import CaptureStore
from constants import NATIVE_CAPTURE_DIMENSIONS
from picamera2_adapter import Picamera2Adapter


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

    def capture_still(self, path, *, settings, size):
        self.calls.append(("still", settings, size))
        assert cv2.imwrite(str(path), np.zeros((size[1], size[0]), dtype=np.uint8))

    def resume_preview(self):
        self.calls.append(("resume",))

    def close(self):
        self.calls.append(("close",))


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


def _trigger_slide_capture(controller, *, captured_at=NOW):
    controller.handle_frame(_slide(), now=3.0, captured_at=captured_at)
    controller.handle_frame(_slide(), now=3.1, captured_at=captured_at)
    return controller.handle_frame(_slide(), now=4.0, captured_at=captured_at)


def test_capture_records_stage_timing_metadata(tmp_path):
    session = _ready_slide_session()
    camera = FakeCamera()
    store = CaptureStore(tmp_path / "published")
    clock = SteppingClock(start=100.0, step=0.025)
    controller = CaptureController(
        session=session,
        camera=camera,
        store=store,
        working_dir=tmp_path / "pending",
        clock=clock,
    )
    controller.start()
    record = _trigger_slide_capture(controller)
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


def test_capture_records_consumer_ms_when_consumer_runs(tmp_path):
    session = _ready_slide_session()
    clock = SteppingClock(start=100.0, step=0.025)

    def consumer(_record):
        return SimpleNamespace(success=True)

    controller = CaptureController(
        session=session,
        camera=FakeCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        capture_consumer=consumer,
        clock=clock,
    )
    controller.start()
    record = _trigger_slide_capture(controller)
    assert record is not None
    assert "consumer_ms" in record.metadata
    assert record.metadata["consumer_ms"] >= 0


def test_capture_records_consumer_decode_outbox_send_ms_when_present(tmp_path):
    """#171: when the slide consumer result carries decode/outbox/send
    sub-durations (mirrors `tools/run_pi_session.py`'s `_consume_capture`
    slide branch), CaptureController forwards them as three new int-ms
    metadata keys, duck-typed exactly like the existing `success` read."""
    session = _ready_slide_session()
    clock = SteppingClock(start=100.0, step=0.025)

    def consumer(_record):
        return SimpleNamespace(
            success=True, decode_ms=12.4, outbox_ms=7.6, send_ms=41.2
        )

    controller = CaptureController(
        session=session,
        camera=FakeCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        capture_consumer=consumer,
        clock=clock,
    )
    controller.start()
    record = _trigger_slide_capture(controller)

    assert record is not None
    assert record.metadata["consumer_decode_ms"] == 12
    assert record.metadata["consumer_outbox_ms"] == 8
    assert record.metadata["consumer_send_ms"] == 41


def test_capture_omits_consumer_split_keys_when_consumer_result_lacks_them(tmp_path):
    """#171: a plain consumer result (e.g. block-mode's `UploadReceipt`, which
    has no decode stage) must not grow bogus split-timing keys -- only the
    existing `consumer_ms` key is present."""
    session = _ready_slide_session()
    clock = SteppingClock(start=100.0, step=0.025)

    def consumer(_record):
        return SimpleNamespace(success=True)

    controller = CaptureController(
        session=session,
        camera=FakeCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        capture_consumer=consumer,
        clock=clock,
    )
    controller.start()
    record = _trigger_slide_capture(controller)

    assert record is not None
    assert "consumer_ms" in record.metadata
    assert "consumer_decode_ms" not in record.metadata
    assert "consumer_outbox_ms" not in record.metadata
    assert "consumer_send_ms" not in record.metadata


def test_failed_capture_still_exposes_partial_stage_timings(tmp_path):
    class FailOnceCamera(FakeCamera):
        def capture_still(self, path, *, settings, size):
            self.calls.append(("still", settings, size))
            if sum(call[0] == "still" for call in self.calls) == 1:
                raise OSError("camera disconnected")
            assert cv2.imwrite(
                str(path), np.zeros((size[1], size[0]), dtype=np.uint8)
            )

    session = _ready_slide_session()
    captured_results = []
    original_accept = session.accept_capture_result

    def spy_accept(result):
        captured_results.append(result)
        return original_accept(result)

    session.accept_capture_result = spy_accept

    camera = FailOnceCamera()
    clock = SteppingClock(start=100.0, step=0.025)
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path / "pending",
        clock=clock,
    )
    controller.start()
    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    assert controller.handle_frame(_slide(), now=4.0, captured_at=NOW) is None
    assert len(captured_results) == 1
    result = captured_results[0]
    assert not result.ok
    meta = result.metadata
    assert "total_capture_ms" in meta
    assert "capture_mode" in meta
    assert meta["capture_mode"] == "slide"
    assert meta["total_capture_ms"] >= 0
    assert "camera_capture_ms" in meta
    assert meta["camera_capture_ms"] >= 0
    assert controller.last_failure_timings is not None
    assert controller.last_failure_timings["capture_mode"] == "slide"
    assert "camera_capture_ms" in controller.last_failure_timings
    assert "total_capture_ms" in controller.last_failure_timings


def test_action_logger_capture_failed_can_include_stage_timings(tmp_path):
    """Mirror run_pi_session._attach_capture_logging failure forwarding."""
    from action_logger import ActionLogger
    from capture_runtime import CAPTURE_STAGE_TIMING_KEYS

    path = tmp_path / "actions.log"
    logger = ActionLogger(path, session_number=1, print_sink=lambda _line: None)
    failure_meta = {
        "camera_capture_ms": 12,
        "total_capture_ms": 40,
        "capture_mode": "slide",
    }
    stage_fields = {
        key: failure_meta[key]
        for key in CAPTURE_STAGE_TIMING_KEYS
        if key in failure_meta
    }
    logger.log("capture_failed", elapsed_ms=55, **stage_fields)
    line = path.read_text(encoding="utf-8").strip()
    assert "event=capture_failed" in line
    assert "elapsed_ms=55" in line
    assert "camera_capture_ms=12" in line
    assert "total_capture_ms=40" in line
    assert "capture_mode=slide" in line


def test_locked_role_camera_settings_are_centralized():
    slide = camera_settings_for("slide")
    block = camera_settings_for("block")

    assert slide == CameraSettings(
        exposure_us=8333,
        contrast=1.8,
        sharpness=1.5,
        denoise="Minimal",
        analogue_gain=1.0,
        colour_gains=(3.08, 1.492),
        ae_enabled=False,
        awb_enabled=False,
    )
    assert block.exposure_us == 33333
    assert block.contrast == 1.4
    assert block.sharpness == 1.6
    assert block.analogue_gain == slide.analogue_gain
    assert block.colour_gains == slide.colour_gains


def test_picamera_adapter_stops_before_reconfiguring_running_preview():
    class FakePicamera:
        sensor_resolution = (4056, 3040)
        sensor_modes = [
            {
                "size": (2028, 1520),
                "bit_depth": 12,
                "crop_limits": (0, 0, 4056, 3040),
            }
        ]

        def __init__(self):
            self.calls = []
            self.started = False

        def create_preview_configuration(self, **kwargs):
            return kwargs

        def configure(self, config):
            assert not self.started
            self.calls.append("configure")

        def set_controls(self, _controls):
            self.calls.append("controls")

        def start(self):
            assert not self.started
            self.started = True
            self.calls.append("start")

        def stop(self):
            assert self.started
            self.started = False
            self.calls.append("stop")

    adapter = Picamera2Adapter.__new__(Picamera2Adapter)
    adapter._camera = FakePicamera()
    adapter._preview_config = None
    adapter._denoise_minimal = "minimal"
    adapter._started = False

    adapter.start_preview(settings=camera_settings_for("block"), size=(640, 480), fps=10)
    adapter.start_preview(settings=camera_settings_for("slide"), size=(640, 480), fps=10)

    assert adapter._camera.calls.count("start") == 2
    assert adapter._camera.calls.index("stop") < len(adapter._camera.calls) - 1


def test_picamera_adapter_preview_uses_full_sensor_field_of_view():
    class FakePicamera:
        sensor_resolution = (4056, 3040)
        sensor_modes = [
            {
                "size": (1332, 990),
                "bit_depth": 10,
                "crop_limits": (696, 528, 2664, 1980),
            },
            {
                "size": (2028, 1520),
                "bit_depth": 12,
                "crop_limits": (0, 0, 4056, 3040),
            },
            {
                "size": (4056, 3040),
                "bit_depth": 12,
                "crop_limits": (0, 0, 4056, 3040),
            },
        ]

        def __init__(self):
            self.preview_kwargs = None

        def create_preview_configuration(self, **kwargs):
            self.preview_kwargs = kwargs
            return kwargs

        def configure(self, _config):
            pass

        def start(self):
            pass

    adapter = Picamera2Adapter.__new__(Picamera2Adapter)
    adapter._camera = FakePicamera()
    adapter._preview_config = None
    adapter._denoise_minimal = "minimal"
    adapter._started = False

    adapter.start_preview(
        settings=camera_settings_for("block"), size=(640, 480), fps=10
    )

    assert adapter._camera.preview_kwargs["main"]["size"] == (640, 480)
    assert adapter._camera.preview_kwargs["sensor"] == {
        "output_size": (2028, 1520),
        "bit_depth": 12,
    }


def test_controller_uses_low_resolution_preview_at_ten_fps(tmp_path):
    camera = FakeCamera()
    controller = CaptureController(
        session=_ready_slide_session(),
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
    )

    controller.start()

    assert camera.calls == [
        ("preview", camera_settings_for("slide"), (640, 480), 10.0)
    ]


def test_capture_orders_preview_still_publish_resume(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
    )
    controller.start()

    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    record = controller.handle_frame(_slide(), now=4.0, captured_at=NOW)

    assert [call[0] for call in camera.calls] == ["preview", "still", "resume"]
    # Tier-3 (#186 follow-up): slides are captured natively at their
    # configured half-res dimensions -- no publish-time downscale needed.
    assert camera.calls[1][2] == (2028, 1520)
    assert record is session.last_capture
    assert record.path.name == "capture_000001_slide_20260701T193045Z.png"
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert not list(tmp_path.glob("pending-*.png"))


def _ready_block_session() -> CaptureSession:
    session = CaptureSession(SessionConfig(baseline_frames=2), mode="block")
    session.confirm_empty()
    session.accept_frame(_empty(), now=0.0)
    session.accept_frame(_empty(), now=0.1)
    assert session.state is CaptureState.WAITING_FOR_SCAN
    scan = session.submit_scan("51151378")
    assert scan.accepted
    return session


def test_block_capture_still_requests_full_sensor_resolution(tmp_path):
    camera = FakeCamera()
    session = _ready_block_session()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
    )
    controller.start()

    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    record = controller.handle_frame(_slide(), now=4.0, captured_at=NOW)

    assert [call[0] for call in camera.calls] == ["preview", "still", "resume"]
    assert camera.calls[1][2] == NATIVE_CAPTURE_DIMENSIONS
    assert record is session.last_capture


def test_published_slide_is_forwarded_once_to_capture_consumer(tmp_path):
    camera = FakeCamera()
    session = _ready_slide_session()
    consumed = []
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=consumed.append,
    )

    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    controller.handle_frame(_slide(), now=4.0, captured_at=NOW)
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

    controller = CaptureController(
        session=session,
        camera=FakeCamera(),
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
        capture_consumer=fail_decode,
    )
    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    controller.handle_frame(_slide(), now=4.0, captured_at=NOW)

    assert len(consumed) == 1
    assert consumed[0].path.is_file()
    assert session.state is CaptureState.REPOSITION_SLIDE
    assert session.drain_events()[-1].message == "Reposition slide"


def test_capture_error_resumes_preview_and_can_retry_same_specimen(tmp_path):
    class FailOnceCamera(FakeCamera):
        def capture_still(self, path, *, settings, size):
            self.calls.append(("still", settings, size))
            if sum(call[0] == "still" for call in self.calls) == 1:
                raise OSError("camera disconnected")
            assert cv2.imwrite(
                str(path), np.zeros((size[1], size[0]), dtype=np.uint8)
            )

    camera = FailOnceCamera()
    session = _ready_slide_session()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(tmp_path / "published"),
        working_dir=tmp_path,
    )
    controller.start()
    controller.handle_frame(_slide(), now=3.0, captured_at=NOW)
    controller.handle_frame(_slide(), now=3.1, captured_at=NOW)
    assert controller.handle_frame(_slide(), now=4.0, captured_at=NOW) is None
    assert session.state is CaptureState.CAPTURE_ERROR
    assert camera.calls[-1] == ("resume",)

    record = controller.retry(captured_at=NOW)

    assert record.path.is_file()
    assert session.state is CaptureState.WAITING_FOR_REMOVAL
    assert [call[0] for call in camera.calls].count("still") == 2
    assert [call[0] for call in camera.calls].count("resume") == 2
