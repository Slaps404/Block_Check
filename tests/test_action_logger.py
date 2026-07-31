"""Unit tests for the Pi workflow action logger (Track B)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import patch

from action_logger import ActionLogger

_ISO_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z  "
)


def test_log_appends_one_iso_utc_line_to_file(tmp_path):
    path = tmp_path / "nested" / "workflow_actions_session_8.log"
    logger = ActionLogger(path, session_number=8, print_sink=lambda _line: None)

    logger.log("slide_reposition_required", state="REPOSITION_SLIDE")

    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 1
    assert _ISO_UTC_RE.match(lines[0])
    assert "session=8" in lines[0]
    assert "event=slide_reposition_required" in lines[0]
    assert "state=REPOSITION_SLIDE" in lines[0]


def test_log_calls_print_sink_with_same_line(tmp_path):
    path = tmp_path / "actions.log"
    printed: list[str] = []
    logger = ActionLogger(path, session_number=3, print_sink=printed.append)

    logger.log("capture_saved", elapsed_ms=2770)

    assert len(printed) == 1
    assert printed[0] == path.read_text(encoding="utf-8").strip()


def test_log_verb_omits_event_token(tmp_path):
    path = tmp_path / "actions.log"
    logger = ActionLogger(path, session_number=8, print_sink=lambda _line: None)

    logger.log("", verb="retry_capture")

    line = path.read_text(encoding="utf-8").strip()
    assert "verb=retry_capture" in line
    assert "event=" not in line


def test_log_appends_multiple_lines(tmp_path):
    path = tmp_path / "actions.log"
    logger = ActionLogger(path, session_number=1, print_sink=lambda _line: None)

    logger.log("capture_start")
    logger.log("capture_saved", elapsed_ms=100)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all("session=1" in line for line in lines)


@patch("action_logger.datetime")
def test_timestamp_uses_utc_milliseconds(mock_datetime, tmp_path):
    fixed = datetime(2026, 7, 9, 22, 15, 3, 412000, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed
    path = tmp_path / "actions.log"
    logger = ActionLogger(path, session_number=8, print_sink=lambda _line: None)

    logger.log("test")

    line = path.read_text(encoding="utf-8").strip()
    assert line.startswith("2026-07-09T22:15:03.412Z  ")
