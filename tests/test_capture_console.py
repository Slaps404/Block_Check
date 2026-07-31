"""Console entry-point contract for explicit mode selection."""
from __future__ import annotations

import importlib
import sys

import pytest

from capture_console import build_parser, create_session
from capture_session import CaptureState


def test_console_requires_explicit_block_or_slide_mode():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    assert parser.parse_args(["--mode", "block"]).mode == "block"
    assert parser.parse_args(["--mode", "slide"]).mode == "slide"


def test_console_builds_the_same_shared_session_contract():
    assert (
        create_session("block").state
        is CaptureState.AWAITING_BASELINE_CONFIRMATION
    )
    assert (
        create_session("slide").state
        is CaptureState.AWAITING_BASELINE_CONFIRMATION
    )


def test_importing_pi_adapter_on_windows_does_not_import_picamera2():
    before = set(sys.modules)

    importlib.import_module("picamera2_adapter")

    assert "picamera2" not in set(sys.modules) - before
