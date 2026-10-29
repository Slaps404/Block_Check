"""Console entry-point contract for explicit mode selection."""
from __future__ import annotations

import pytest

from capture_console import build_parser, create_session, main
from capture_session import CaptureState


def test_console_requires_explicit_block_or_slide_mode():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    assert parser.parse_args(["--mode", "block"]).mode == "block"
    assert parser.parse_args(["--mode", "slide"]).mode == "slide"


def test_console_exposes_triple_capture_for_block_mode():
    args = build_parser().parse_args(["--mode", "block", "--triple-capture"])

    assert args.triple_capture is True


def test_console_rejects_triple_capture_in_slide_mode():
    with pytest.raises(SystemExit):
        main(["--mode", "slide", "--triple-capture"])


def test_console_builds_the_same_shared_session_contract():
    assert (
        create_session("block").state
        is CaptureState.AWAITING_BASELINE_CONFIRMATION
    )
    assert (
        create_session("slide").state
        is CaptureState.AWAITING_BASELINE_CONFIRMATION
    )
