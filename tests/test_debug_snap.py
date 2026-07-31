"""Debug snap: POST /debug/snap + HttpCaptureClient + PiCaptureRuntime.snap."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.append(str(_TOOLS_DIR))

from session.workflow import (  # noqa: E402
    FramingCalibrationStore,
    HttpCaptureClient,
    LoopbackCaptureReceiver,
    PiOutbox,
    ProcessingStore,
    SessionWorkflow,
    save_debug_snap,
)

import run_pi_session  # noqa: E402


STARTED_AT = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def test_save_debug_snap_writes_png_under_dest(tmp_path):
    body = b"\x89PNG\r\n\x1a\n" + b"fake"
    path = save_debug_snap(body, dest_dir=tmp_path / "pi_captures")
    assert path.parent == tmp_path / "pi_captures"
    assert path.name.startswith("snap_")
    assert path.suffix == ".png"
    assert path.read_bytes() == body


def test_save_debug_snap_open_image_calls_viewer(tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        "session.outbox_transport.open_saved_image",
        lambda path: opened.append(str(path)),
    )
    path = save_debug_snap(
        b"\x89PNG\r\n\x1a\n" + b"open-me",
        dest_dir=tmp_path / "pi_captures",
        open_image=True,
    )
    assert opened == [str(path)]


def test_receiver_debug_snap_route_saves_and_returns_path(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    dest = tmp_path / "Desktop" / "pi_captures"
    png = b"\x89PNG\r\n\x1a\n" + b"debug-body"

    with LoopbackCaptureReceiver(
        store, debug_snap_dir=dest, open_debug_snaps=False
    ) as receiver:
        request = Request(
            f"{receiver.url}/debug/snap",
            data=png,
            method="POST",
            headers={"Content-Type": "image/png"},
        )
        with urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8")

    assert '"path"' in payload
    saved = list(dest.glob("snap_*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == png


def test_http_client_debug_snap_round_trip(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    dest = tmp_path / "pi_captures"
    local = tmp_path / "pending.png"
    local.write_bytes(b"\x89PNG\r\n\x1a\n" + b"client")

    with LoopbackCaptureReceiver(
        store, debug_snap_dir=dest, open_debug_snaps=False
    ) as receiver:
        client = HttpCaptureClient(receiver.url)
        saved = client.debug_snap(local)

    assert Path(saved).exists()
    assert Path(saved).read_bytes() == local.read_bytes()


class _FakeCamera:
    def __init__(self):
        self.still_calls = 0
        self.resume_calls = 0

    def start_preview(self, **_kwargs):
        pass

    def preview_frame(self):
        return np.full((80, 120, 3), 180, dtype=np.uint8)

    def capture_still(self, path, **_kwargs):
        self.still_calls += 1
        assert cv2.imwrite(str(path), np.full((100, 120), 90, dtype=np.uint8))

    def resume_preview(self):
        self.resume_calls += 1

    def close(self):
        pass


def test_runtime_snap_uploads_deletes_pi_temp_and_resumes(tmp_path):
    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=STARTED_AT)
    dest = tmp_path / "pi_captures"
    camera = _FakeCamera()

    with LoopbackCaptureReceiver(
        store, debug_snap_dir=dest, open_debug_snaps=False
    ) as receiver:
        workflow = SessionWorkflow(
            session=session,
            store=store,
            outbox=PiOutbox(tmp_path / "outbox"),
            transport=HttpCaptureClient(receiver.url),
            framing_calibration=FramingCalibrationStore(
                tmp_path / "framing_calibration.json"
            ),
        )
        runtime = run_pi_session.PiCaptureRuntime(
            workflow, camera, capture_root=tmp_path / "captures"
        )
        runtime.start(background=False)
        message = runtime.snap()
        runtime.close()

    assert message.startswith("Saved:")
    laptop_path = Path(message.split("Saved:", 1)[1].strip())
    assert laptop_path.exists()
    assert laptop_path.parent == dest
    assert camera.still_calls == 1
    assert camera.resume_calls >= 1
    leftovers = list((tmp_path / "captures").rglob("debug-snap-*.png"))
    assert leftovers == []


def test_console_registers_snap_verb():
    from session.console import COMMAND_HELP, _COMMANDS

    assert "snap" in _COMMANDS
    assert "snap" in COMMAND_HELP
