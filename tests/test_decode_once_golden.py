"""Golden characterization test for ``ProcessingStore.resolve_claim`` (#185 PR1).

This is a byte-identical safety net for an upcoming "decode-once" perf
refactor. It replays ``resolve_claim`` end-to-end against real committed
images (``images/pi_images_v2``) using the REAL production preprocessors
(no stubs), and asserts a committed baseline snapshot in
``tests/golden/decode_once_baseline.json`` is reproduced exactly. It must
NOT change behavior -- only lock it in.

To regenerate the baseline from current behavior (e.g. after a deliberate,
reviewed change to CV output), run:

    $env:REGEN_DECODE_ONCE_GOLDEN=1
    .\\venv\\Scripts\\python.exe -m pytest tests/test_decode_once_golden.py -q
    Remove-Item Env:REGEN_DECODE_ONCE_GOLDEN

With the env var set, the test writes the freshly computed snapshot to the
baseline file instead of asserting against it, and still passes.

This repo has no pytest CI (only Claude-review GitHub Actions), so this is a
MANUAL local gate: whoever does the decode-once refactor (#185 PR2/PR3) must
run this test locally before AND after their change, on the same machine, to
confirm byte-identity. On Windows, a stale ``pytest-current`` symlink can
throw a ``PermissionError`` during teardown that is unrelated to the
assertions -- if that appears, rerun with a fresh ``--basetemp``.
"""
from __future__ import annotations

import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from contact_sheet import write_contact_sheet
from session.pipeline import ClaimDecision
from session.preparation import PreparationFailure, PreparedSpecimen
from session.workflow import PiOutbox, ProcessingStore

STARTED_AT = datetime(2026, 6, 23, 18, 0, 0, tzinfo=timezone.utc)
BLOCK_ID = "51137201"
SLIDE_CAPTURE_ID = "slide_capture_1"

_IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "pi_images_v2"
# set_01: block + its own (matched) slide, per pair_manifest.csv.
_BLOCK_IMAGE = _IMG_DIR / "capture_20260623_200438.jpg"
_SLIDE_MATCHED = _IMG_DIR / "capture_20260623_200415.jpg"
# set_02 slide claimed against the set_01 block: forces a mismatched claim.
_SLIDE_MISMATCHED = _IMG_DIR / "capture_20260623_202054.jpg"

_REQUIRED_IMAGES = (_BLOCK_IMAGE, _SLIDE_MATCHED, _SLIDE_MISMATCHED)
_IMAGES_PRESENT = all(path.is_file() for path in _REQUIRED_IMAGES)

_BASELINE_PATH = Path(__file__).resolve().parent / "golden" / "decode_once_baseline.json"
_REGEN = bool(os.environ.get("REGEN_DECODE_ONCE_GOLDEN"))
_RUNNING_IN_CI = bool(os.environ.get("CI"))

pytestmark = pytest.mark.skipif(
    not _IMAGES_PRESENT or _RUNNING_IN_CI,
    reason=(
        "pi_images_v2 golden-case images not found, or running in automated CI; "
        "this is a manual local gate only -- JPEG decode is not guaranteed "
        "bit-identical across libjpeg builds/platforms, so the baseline can "
        "only be trusted when replayed on the same machine that captured it"
    ),
)


class _RecordingContactSheetRenderer:
    """Captures each claim-QC render call, then delegates to the real writer.

    Mirrors the ``work_order_scorer``/``contact_sheet_renderer`` injection
    seam already used in ``tests/test_session_workflow.py``: the real QC PNG
    is genuinely written (nothing is faked), but the masks and decision that
    produced it are also stashed here so the test can hash them directly.
    """

    def __init__(self, real_renderer):
        self._real_renderer = real_renderer
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        self._real_renderer(*args, **kwargs)


def _ingest_real_block(store: ProcessingStore, session, tmp_path: Path) -> str:
    """Scan + capture a real committed block image through the normal flow.

    ``CaptureStore.publish`` requires a ``.png``-suffixed source path (it
    checks the filename, not the content), so the real JPEG bytes are copied
    verbatim to a ``.png``-named file first. The pixel content reaching the
    preprocessor is byte-identical to the committed image; only the on-disk
    extension differs, and ``cv2.imread`` decodes by content, not suffix.
    """
    assert store.scan_block(session.number, BLOCK_ID).accepted
    png_named_source = tmp_path / f"{BLOCK_ID}_source.png"
    png_named_source.write_bytes(_BLOCK_IMAGE.read_bytes())
    capture = PiOutbox(tmp_path / "outbox").publish_block(
        png_named_source, BLOCK_ID, STARTED_AT
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    return BLOCK_ID


def _mask_snapshot(result) -> dict[str, object]:
    if isinstance(result, PreparedSpecimen):
        mask: np.ndarray = result.mask
        return {
            "sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
        }
    assert isinstance(result, PreparationFailure)
    return {"preparation_failed": True, "reason": result.reason}


def _round(value):
    return round(value, 6) if isinstance(value, float) else value


def _decision_snapshot(decision: ClaimDecision) -> dict[str, object]:
    """Serialize every ``ClaimDecision`` field for the golden snapshot.

    ``block_path``/``slide_path`` are reduced to their basename: the block
    path is anchored under pytest's per-run ``tmp_path`` (never the same
    twice), so the full path would make the baseline unreproducible even
    though the underlying decision it names is identical.
    """
    snapshot = {}
    for field in fields(decision):
        value = getattr(decision, field.name)
        if field.name in ("block_path", "slide_path") and value:
            value = Path(value).name
        snapshot[field.name] = _round(value)
    return snapshot


def _decisions_csv_row(session_directory: Path, block_id: str) -> dict[str, str]:
    text = (session_directory / "decisions.csv").read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        if row["block_id"] == block_id:
            row.pop("decided_at", None)
            return row
    raise AssertionError(f"no decisions.csv row for block_id={block_id}")


def _qc_pixel_sha256(session_directory: Path) -> str:
    qc_path = session_directory / "claim_artifacts" / f"{SLIDE_CAPTURE_ID}_claim_qc.png"
    image = cv2.imread(str(qc_path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"QC PNG could not be reopened: {qc_path}"
    return hashlib.sha256(image.tobytes()).hexdigest()


def _build_case_snapshot(tmp_path: Path, case_dir_name: str, slide_path: Path) -> dict:
    recorder = _RecordingContactSheetRenderer(write_contact_sheet)
    store = ProcessingStore(
        tmp_path / case_dir_name, contact_sheet_renderer=recorder,
    )
    session = store.start_session(started_at=STARTED_AT)
    block_id = _ingest_real_block(store, session, tmp_path / case_dir_name)

    outcome = store.resolve_claim(
        session.number, block_id, SLIDE_CAPTURE_ID, slide_path,
    )
    assert outcome.accepted

    call = recorder.calls[-1]
    decision: ClaimDecision = call["decision"]
    snapshot = {
        "block_mask": _mask_snapshot(call["block_result"]),
        "slide_mask": _mask_snapshot(call["slide_result"]),
        "gates": _decision_snapshot(decision),
        "outcome": {
            "verdict": outcome.verdict,
            "score": _round(outcome.score),
            "stage": outcome.stage,
            "reason": outcome.reason,
        },
        "decisions_csv_row": _decisions_csv_row(session.directory, block_id),
        "qc_pixel_sha256": _qc_pixel_sha256(session.directory),
    }
    return snapshot


def _load_baseline() -> dict:
    if not _BASELINE_PATH.is_file():
        return {}
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _write_baseline(baseline: dict) -> None:
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_matches_baseline(case_name: str, snapshot: dict, expected: dict) -> None:
    for key, expected_value in expected.items():
        assert snapshot.get(key) == expected_value, (
            f"[{case_name}] golden artifact '{key}' drifted from baseline:\n"
            f"expected={expected_value!r}\nactual={snapshot.get(key)!r}"
        )
    extra_keys = set(snapshot) - set(expected)
    assert not extra_keys, f"[{case_name}] snapshot has undeclared keys: {extra_keys}"


def test_decode_once_golden_snapshot(tmp_path):
    snapshots = {
        "matched": _build_case_snapshot(tmp_path, "matched", _SLIDE_MATCHED),
        "mismatched": _build_case_snapshot(tmp_path, "mismatched", _SLIDE_MISMATCHED),
    }

    if _REGEN:
        _write_baseline(snapshots)
        return

    baseline = _load_baseline()
    assert baseline, (
        f"no baseline at {_BASELINE_PATH}; regenerate with "
        "REGEN_DECODE_ONCE_GOLDEN=1 first"
    )
    for case_name, snapshot in snapshots.items():
        assert case_name in baseline, f"baseline missing case '{case_name}'"
        _assert_matches_baseline(case_name, snapshot, baseline[case_name])

    # Sanity cross-check on the case design itself: the mismatched claim must
    # score no better than the matched one, and land in REVIEW.
    matched_score = snapshots["matched"]["outcome"]["score"]
    mismatched_score = snapshots["mismatched"]["outcome"]["score"]
    assert mismatched_score is None or matched_score is None or (
        mismatched_score <= matched_score
    )
    assert snapshots["mismatched"]["outcome"]["verdict"] == "REVIEW"
