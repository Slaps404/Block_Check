"""Durable publication tests for the shared capture counter."""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from capture_session import CaptureResult, CaptureSession
from capture_storage import CaptureStore, PublicationError, ValidatedStill
from constants import CAPTURE_DIMENSIONS, NATIVE_CAPTURE_DIMENSIONS

SLIDE_WIDTH, SLIDE_HEIGHT = CAPTURE_DIMENSIONS["slide"]


NOW = datetime(2026, 7, 1, 19, 30, 45, tzinfo=timezone.utc)


def _png(
    path: Path,
    *,
    width: int = NATIVE_CAPTURE_DIMENSIONS[0],
    height: int = NATIVE_CAPTURE_DIMENSIONS[1],
) -> Path:
    assert cv2.imwrite(str(path), np.zeros((height, width), dtype=np.uint8))
    return path


@pytest.mark.native_resolution
def test_counter_is_shared_across_roles_and_survives_restart(tmp_path):
    store = CaptureStore(tmp_path / "published")
    slide = store.publish(_png(tmp_path / "slide.png"), "slide", captured_at=NOW)

    restarted = CaptureStore(tmp_path / "published")
    block = restarted.publish(
        _png(tmp_path / "block.png"),
        "block",
        block_id="51151378",
        captured_at=NOW,
    )

    assert slide.counter == 1
    assert slide.path.name == "capture_000001_slide_20260701T193045Z.png"
    assert block.counter == 2
    assert block.path.name == (
        "capture_000002_block_51151378_20260701T193045Z.png"
    )


def test_preexisting_filename_collision_is_never_overwritten(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    collision = output / "capture_000001_slide_20260701T193045Z.png"
    collision.write_bytes(b"keep me")

    record = CaptureStore(output).publish(
        _png(tmp_path / "new.png"), "slide", captured_at=NOW
    )

    assert collision.read_bytes() == b"keep me"
    assert record.counter == 2
    assert record.path.exists()


@pytest.mark.parametrize(
    "source_factory",
    [
        lambda path: path,
        lambda path: (path.write_bytes(b"not a png"), path)[1],
        lambda path: _png(path, width=40, height=30),
    ],
    ids=["missing", "unreadable", "wrong-dimensions"],
)
def test_failed_publication_does_not_advance_counter(tmp_path, source_factory):
    source = tmp_path / "candidate.png"
    made = source_factory(source)
    if made is None:
        made = source
    store = CaptureStore(tmp_path / "published")

    with pytest.raises(PublicationError):
        store.publish(made, "slide", captured_at=NOW)

    good = store.publish(_png(tmp_path / "good.png"), "slide", captured_at=NOW)
    assert good.counter == 1


@pytest.mark.native_resolution
def test_success_exposes_final_path_and_retained_metadata(tmp_path):
    store = CaptureStore(tmp_path / "published")

    record = store.publish(
        _png(tmp_path / "block.png"),
        "block",
        block_id="51151378",
        captured_at=NOW,
        metadata={"exposure_us": 33333, "roi": (0.1, 0.1, 0.9, 0.9)},
    )

    assert record.path.is_file()
    assert record.role == "block"
    assert record.block_id == "51151378"
    assert record.metadata["exposure_us"] == 33333
    reopened = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
    assert reopened.shape[:2] == (3040, 4056)


@pytest.mark.native_resolution
def test_capture_tag_is_saved_in_filename_and_record_metadata(tmp_path):
    record = CaptureStore(tmp_path / "published").publish(
        _png(tmp_path / "block.png"),
        "block",
        block_id="51151378",
        captured_at=NOW,
        metadata={"capture_group_id": "group-1", "capture_index": 2},
        capture_tag="capture_02",
    )

    assert record.path.name == (
        "capture_000001_block_51151378_capture_02_20260701T193045Z.png"
    )
    assert record.metadata["capture_group_id"] == "group-1"
    assert record.metadata["capture_index"] == 2


def test_store_recovers_counter_if_process_died_after_file_publication(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    orphan = output / "capture_000007_slide_20260701T193045Z.png"
    _png(orphan)

    record = CaptureStore(output).publish(
        _png(tmp_path / "next.png"), "slide", captured_at=NOW
    )

    assert record.counter == 8


def test_pending_source_unlink_after_hard_link_leaves_final_intact(tmp_path):
    source = _png(tmp_path / "pending.png", width=SLIDE_WIDTH, height=SLIDE_HEIGHT)
    store = CaptureStore(tmp_path / "published")

    record = store.publish(source, "slide", captured_at=NOW)

    assert record.path.is_file()
    source.unlink()
    assert record.path.is_file()
    reopened = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
    assert reopened.shape[:2] == (SLIDE_HEIGHT, SLIDE_WIDTH)
    assert record.validated == ValidatedStill(
        width=SLIDE_WIDTH, height=SLIDE_HEIGHT, format=".png"
    )


def test_validate_still_rejects_unknown_role(tmp_path):
    path = _png(tmp_path / "capture_0001.png")
    with pytest.raises(PublicationError):
        CaptureStore._validate_still(path, "not_a_role")


def test_storage_validation_facts_are_accepted_by_session_for_same_role(tmp_path):
    path = _png(tmp_path / "capture_0001.png", width=SLIDE_WIDTH, height=SLIDE_HEIGHT)

    validated = CaptureStore._validate_still(path, "slide")
    result = CaptureResult.success(path, validated=validated)

    assert validated == ValidatedStill(
        width=SLIDE_WIDTH, height=SLIDE_HEIGHT, format=".png"
    )
    assert CaptureSession._still_acceptable(result, "slide")


def test_exdev_fallback_still_publishes_via_copy(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path / "published")
    real_link = os.link
    link_calls: list[tuple[Path, Path]] = []

    def link_with_exdev(src, dst):
        link_calls.append((Path(src), Path(dst)))
        if len(link_calls) == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_link(src, dst)

    monkeypatch.setattr(os, "link", link_with_exdev)

    record = store.publish(
        _png(tmp_path / "source.png", width=SLIDE_WIDTH, height=SLIDE_HEIGHT),
        "slide",
        captured_at=NOW,
    )

    assert record.path.is_file()
    assert len(link_calls) == 2
    assert link_calls[0][0] != link_calls[1][0]
    reopened = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
    assert reopened.shape[:2] == (SLIDE_HEIGHT, SLIDE_WIDTH)


def test_happy_path_publish_decodes_source_once(tmp_path, monkeypatch):
    decode_count = {"n": 0}
    original_imread = cv2.imread

    def counting_imread(path, flags):
        decode_count["n"] += 1
        return original_imread(path, flags)

    monkeypatch.setattr(cv2, "imread", counting_imread)
    store = CaptureStore(tmp_path / "published")
    store.publish(
        _png(tmp_path / "slide.png", width=SLIDE_WIDTH, height=SLIDE_HEIGHT),
        "slide",
        captured_at=NOW,
    )

    assert decode_count["n"] == 1
