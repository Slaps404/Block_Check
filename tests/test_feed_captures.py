"""TDD coverage for the Pi-side feeder that publishes real captures into a
``PiOutbox`` and replays them to the processing-computer receiver.

Covers issue #101 handoff B (capture-slice hardware test): this proves the
publish -> replay -> ack path against a real ``LoopbackCaptureReceiver`` on
loopback, standing in for the Ethernet-connected processing computer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from session.workflow import LoopbackCaptureReceiver, PiOutbox, ProcessingStore

# Load tools/feed_captures.py by file path rather than adding the bare
# ``tools`` directory to sys.path: doing the latter would expose every
# tools/<subdir> (e.g. tools/manifest) as a top-level importable package and
# shadow code/session/manifest.py, breaking unrelated imports.
_FEED_CAPTURES_PATH = Path(__file__).resolve().parent.parent / "tools" / "feed_captures.py"
_spec = importlib.util.spec_from_file_location("feed_captures", _FEED_CAPTURES_PATH)
assert _spec is not None and _spec.loader is not None
feed_captures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feed_captures)


STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


class FastPreprocessor:
    """Stand-in preprocessor so tests never touch the real CV pipeline."""

    def __call__(self, capture_path: Path):
        assert capture_path.is_file()
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}


def _write_capture_png(path: Path, value: int) -> Path:
    image = np.full((3040, 4056, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_feed_captures_publishes_and_replays_to_the_real_receiver(tmp_path, capsys):
    store = ProcessingStore(tmp_path / "processing", preprocessor=FastPreprocessor())
    session = store.start_session(started_at=STARTED_AT)
    assert store.scan_block(session.number, "51151378").accepted

    png_path = _write_capture_png(tmp_path / "block.png", 80)
    outbox_dir = tmp_path / "pi-outbox"

    with LoopbackCaptureReceiver(store) as receiver:
        exit_code = feed_captures.main([
            "--outbox", str(outbox_dir),
            "--receiver-url", receiver.url,
            "--session", str(session.number),
            str(png_path), "51151378",
        ])
        store.wait_for_jobs()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "acked 1 / pending 0" in out

    outbox = PiOutbox(outbox_dir)
    assert outbox.pending() == ()
    acknowledged = [entry for entry in outbox.entries() if entry.state == "acknowledged"]
    assert len(acknowledged) == 1

    block = store.get_set(session.number, "51151378")
    assert block["capture_id"] == acknowledged[0].capture_id


def test_feed_captures_keeps_captures_pending_when_receiver_is_unreachable(
    tmp_path, capsys
):
    png_path = _write_capture_png(tmp_path / "block.png", 90)
    outbox_dir = tmp_path / "pi-outbox"

    exit_code = feed_captures.main([
        "--outbox", str(outbox_dir),
        "--receiver-url", "http://127.0.0.1:1",
        "--session", "1",
        str(png_path), "51151378",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "acked 0 / pending 1" in out
    assert PiOutbox(outbox_dir).pending()[0].block_id == "51151378"


def test_feed_captures_rejects_an_unpaired_trailing_argument(tmp_path):
    with pytest.raises(SystemExit):
        feed_captures.main([
            "--outbox", str(tmp_path / "outbox"),
            "--receiver-url", "http://127.0.0.1:1",
            "--session", "1",
            str(tmp_path / "only_one_arg.png"),
        ])
