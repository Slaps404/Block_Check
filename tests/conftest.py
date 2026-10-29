"""Pytest fixtures and import paths for the test suite."""

from __future__ import annotations

import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CODE_DIR = _REPO_ROOT / "code"
_TOOLS_DIR = _REPO_ROOT / "tools"
_TOOL_SUBDIRS = (
    "capture",
    "manifest",
    "identity",
    "scoring_diagnostics",
    "visual_audit",
    "diagnostics",
    "calibration",
)
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_CAPTURE_DIR = _CODE_DIR / "capture"
if _CAPTURE_DIR.is_dir() and str(_CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAPTURE_DIR))
for _subdir in _TOOL_SUBDIRS:
    _path = _TOOLS_DIR / _subdir
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.append(str(_path))


@pytest.fixture(autouse=True)
def _small_block_captures(request, monkeypatch):
    # Camera-size PNGs dominate decode time; opt out with native_resolution.
    if request.node.get_closest_marker("native_resolution"):
        return
    import constants

    monkeypatch.setitem(constants.CAPTURE_DIMENSIONS, "block", (406, 304))


@pytest.fixture(scope="session")
def dead_receiver_url():
    # A closed port takes ~2 s to refuse on Windows; an RST listener fails fast.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def reset_all():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()

    threading.Thread(target=reset_all, daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    listener.close()
