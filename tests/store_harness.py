"""Shared store-only harness for the `ProcessingStore` test files.

Every one of them drives the same shape: a `ProcessingStore` with
injected preprocessors and score-cache builder, synthetic PNG captures, and
a frozen Hybrid Candidate Pool. No camera, no network, no real
segmentation. This module owns that plumbing so each file holds only the
behaviour it is pinning.

Most files inject their own preprocessor/fingerprint doubles and keep their
own `_make_store`; the fixed-mask variant shared by several Hybrid files
lives here. It also owns the real-`SessionWorkflow` scoring helpers that
several kiosk/work-order files reuse.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import constants
from session.pipeline import ClaimDecision
from session.preparation import PreparedResult, PreparedSpecimen
from session.processing_store import ProcessingStore
from session.workflow import (
    FramingCalibrationStore,
    HttpCaptureClient,
    PiOutbox,
    SessionWorkflow,
    WorkOrderScoringResult,
)
from session.workflow_types import SessionIdentity
from slide.qr import DecodeCandidate, select_slide_identity
from retrieval.invariant_descriptors import DescriptorValue
from verify.scorer import LockedScoreCache, _ComponentFeatures

STARTED_AT = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
# The real-workflow helpers below stamp captures with this one.
WORKFLOW_STARTED_AT = datetime(2026, 7, 2, 18, 5, 6, tzinfo=timezone.utc)
DESCRIPTOR_NAMES = ("fake_descriptor_v1",)


@pytest.fixture
def lightweight_qc_artifacts(monkeypatch):
    """Replace the QC panel renderers with fixed-size stand-ins.

    The real renderer assumes the mask matches the capture's dimensions,
    which these tests' small synthetic masks deliberately do not. The
    asserts keep the double honest about what the store actually hands it.
    Slide QC is a separate fixture: not every file that wants block QC
    stubbed wants the slide panel stubbed too.
    """
    def write_qc(capture, mask, destination):
        assert capture.is_file()
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    def write_failure_qc(capture, reason, destination):
        assert capture.is_file()
        assert reason
        panel = np.full((8, 8, 3), (0, 0, 180), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(ProcessingStore, "_write_qc", staticmethod(write_qc))
    monkeypatch.setattr(
        ProcessingStore, "_write_failure_qc", staticmethod(write_failure_qc)
    )


class FixedMaskPreprocessor:
    """Every block prepares to the SAME fully-filled mask, whatever the
    pixel content, so only block ids separate captures."""

    def __call__(self, capture_path: Path):
        return np.full((8, 8), 255, dtype=np.uint8), {"role": "block", "roi_ok": True}


class FixedSlidePreprocessor:
    """Every slide prepares to the SAME fully-filled specimen."""

    def __call__(self, image: np.ndarray) -> PreparedResult:
        return PreparedSpecimen(
            role="slide", mask=np.full((8, 8), 255, dtype=np.uint8),
            roi_ok=True, roi_reason="",
        )


class IdenticalFingerprintBuilder:
    """Every block gets the IDENTICAL fingerprint vector, so only distinct
    block ids separate them."""

    def __call__(self, mask):
        return {
            "fake_descriptor_v1": DescriptorValue(
                vector=np.array([1.0, 2.0, 3.0]), construction_ns=1
            )
        }


def score_cache_builder(specimen):
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def capture_png(value: int) -> bytes:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    return png.tobytes()


def write_slide_png(path: Path, value: int = 200) -> Path:
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def valid_slide_result(block_id: str):
    return select_slide_identity((
        DecodeCandidate("zxing", "QRCode", "raw", f"12080_{block_id}_01_HE"),
    ))


def drain(store: ProcessingStore, session_number: int) -> None:
    store.wait_for_jobs()
    store.begin_block_drain(session_number)


def capture_block(
    store: ProcessingStore, session_number: int, block_id: str, value: int = 0
) -> None:
    assert store.scan_block(session_number, block_id).accepted
    body = capture_png(value)
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


def freeze_hybrid_session(
    store: ProcessingStore, session_mode: str, block_ids: tuple[str, ...],
) -> tuple[int, int]:
    """Start a Hybrid session, capture/freeze ``block_ids`` into one pool.

    Returns (session_number, work_order_id).
    """
    session = store.start_session(started_at=STARTED_AT, session_mode=session_mode)
    work_order_id = store.start_work_order(session.number)
    for index, block_id in enumerate(block_ids):
        capture_block(store, session.number, block_id, index + 1)
    drain(store, session.number)
    freeze_result = store.freeze_hybrid_pool(
        session.number, descriptor_names=DESCRIPTOR_NAMES
    )
    assert freeze_result.frozen is True
    return session.number, work_order_id


@pytest.fixture
def lightweight_slide_qc(monkeypatch):
    """Stub the slide QC panel renderer, the slide-side counterpart to
    `lightweight_qc_artifacts`."""
    def write_slide_qc(image, mask, destination):
        assert image.ndim == 3
        assert mask.ndim == 2
        panel = np.full((8, 24, 3), (0, 128, 0), dtype=np.uint8)
        assert cv2.imwrite(str(destination), panel)

    monkeypatch.setattr(
        ProcessingStore, "_write_slide_qc", staticmethod(write_slide_qc)
    )


def fake_fingerprint_builder(mask):
    return {"fake_v1": DescriptorValue(vector=np.array([1.0]), construction_ns=1)}


def fake_score_cache_builder(specimen):
    return LockedScoreCache(
        normalized_mask=specimen.mask,
        component_features=_ComponentFeatures(
            points=np.zeros((0, 2)), areas=np.zeros(0), shapes=np.zeros((0, 3)),
        ),
    )


def upload_block(store, session_number, block_id, value):
    assert store.scan_block(session_number, block_id).accepted
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    success, png = cv2.imencode(".png", image)
    assert success
    body = png.tobytes()
    checksum = hashlib.sha256(body).hexdigest()
    store.receive_capture(
        session_number, capture_id=f"cap-{block_id}", block_id=block_id,
        checksum=checksum, body=body,
    )


_FULL_RES_PNGS: dict[tuple[int, int, int], bytes] = {}


def full_res_capture(path: Path, value: int) -> Path:
    """Write a block-capture-sized flat PNG, encoding each size/value once.

    The size is ``CAPTURE_DIMENSIONS["block"]``, which conftest shrinks
    unless a test is marked ``native_resolution``.
    """
    width, height = constants.CAPTURE_DIMENSIONS["block"]
    key = (value, width, height)
    encoded = _FULL_RES_PNGS.get(key)
    if encoded is None:
        image = np.full((height, width, 3), value, dtype=np.uint8)
        success, png = cv2.imencode(".png", image)
        assert success
        encoded = png.tobytes()
        _FULL_RES_PNGS[key] = encoded
    path.write_bytes(encoded)
    return path


def _make_store(tmp_path: Path, **kwargs) -> ProcessingStore:
    """Fixed-mask, identical-fingerprint store for the Hybrid queue files."""
    kwargs.setdefault("preprocessor", FixedMaskPreprocessor())
    kwargs.setdefault("slide_preprocessor", FixedSlidePreprocessor())
    kwargs.setdefault("fingerprint_builder", IdenticalFingerprintBuilder())
    kwargs.setdefault("score_cache_builder", score_cache_builder)
    return ProcessingStore(tmp_path / "processing", **kwargs)


# ---------------------------------------------------------------------------
# Real-SessionWorkflow scoring helpers.
# ---------------------------------------------------------------------------


class ToggleTransport:
    def __init__(self, store):
        self.store = store
        self.connected = False

    def status(self, session_number):
        if not self.connected:
            raise ConnectionRefusedError("offline")
        return {"session_number": session_number}

    def upload(self, session_number, capture):
        if not self.connected:
            raise ConnectionRefusedError("offline")
        return self.store.receive_capture(
            session_number,
            capture_id=capture.capture_id,
            block_id=capture.block_id,
            checksum=capture.checksum,
            body=capture.path.read_bytes(),
        )


class FastPreprocessor:
    def __call__(self, capture_path):
        assert capture_path.is_file()
        return np.full((8, 8), 255, dtype=np.uint8), {"method": "test"}


def _identical_mask_slide_preprocessor(_img):
    """Deterministic evaluable slide result that mirrors ``FastPreprocessor``."""
    return PreparedSpecimen(
        role="slide", mask=np.full((8, 8), 255, dtype=np.uint8), roi_ok=True,
    )


def _evaluable_block(store, session, tmp_path, block_id="51151378"):
    assert store.scan_block(session.number, block_id).accepted
    capture = PiOutbox(tmp_path / "outbox_for_test").publish_block(
        full_res_capture(tmp_path / f"{block_id}_block.png", 80),
        block_id, WORKFLOW_STARTED_AT,
    )
    store.receive_capture(
        session.number, capture_id=capture.capture_id, block_id=capture.block_id,
        checksum=capture.checksum, body=capture.path.read_bytes(),
    )
    store.wait_for_jobs()
    return block_id


class StubWorkOrderScorer:
    """Deterministic, synchronous stand-in for the production N^2 scorer.

    Ignores the (block_results, slide_results) mappings it is handed and
    instead returns whatever ``scores_by_slide`` has been populated with for
    each slide capture id -- tests populate this dict *after* construction,
    once the real capture id is known, but *before* ``finish_work_order`` is
    called.
    """

    def __init__(self):
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}
        self.calls = 0

    def __call__(self, block_results, slide_results):
        self.calls += 1
        return {
            capture_id: dict(self.scores_by_slide.get(capture_id, {}))
            for capture_id in slide_results
        }


class PoseAwareStubWorkOrderScorer:
    """Returns scores plus a fixed pose for integration tests."""

    def __init__(self, *, best_angle: float = 42.0, best_flip: bool = True):
        self.scores_by_slide: dict[str, dict[str, float | None]] = {}
        self.best_angle = best_angle
        self.best_flip = best_flip

    def __call__(self, block_results, slide_results):
        pair_decisions: dict[str, dict[str, ClaimDecision]] = {}
        scores: dict[str, dict[str, float | None]] = {}
        for capture_id in slide_results:
            row_scores = dict(self.scores_by_slide.get(capture_id, {}))
            scores[capture_id] = row_scores
            row_decisions: dict[str, ClaimDecision] = {}
            for block_id, block_score in row_scores.items():
                row_decisions[block_id] = ClaimDecision(
                    claim_id=block_id,
                    block_path="",
                    slide_path="",
                    verdict="REVIEW",
                    stage="scoring",
                    reason="stub",
                    score=block_score,
                    best_angle=self.best_angle,
                    best_flip=self.best_flip,
                    align_soft_iou=0.5,
                    mask_iou=0.4,
                )
            pair_decisions[capture_id] = row_decisions
        return WorkOrderScoringResult(scores=scores, pair_decisions=pair_decisions)


def _drain_to_slides(store, session):
    store.begin_block_drain(session.number)
    assert store.try_enter_slides(session.number)


class _FakeResultsRowsStore:
    """Minimal store double for ``results_status()``'s mode routing.

    Implements only the methods ``SessionWorkflow.__init__`` and
    ``results_status`` touch, and counts calls so tests can prove which
    store method a session mode reaches."""

    def __init__(
        self, *, ready_rows=(), hybrid_rows=(), retrieval_rows=(),
        hybrid_error=None, retrieval_error=None,
        profile_rows=(), profile_error=None,
    ):
        self.ready_calls = 0
        self.hybrid_calls = 0
        self.retrieval_calls = 0
        # Lets tests assert pause_capture/resume_capture never wait on or
        # cancel jobs.
        self.wait_for_jobs_calls = 0
        # Lets hybrid_profile_status tests prove list_hybrid_profile_rows
        # was (or was not) reached.
        self.profile_calls = 0
        self._ready_rows = tuple(ready_rows)
        self._hybrid_rows = tuple(hybrid_rows)
        self._retrieval_rows = tuple(retrieval_rows)
        self._hybrid_error = hybrid_error
        self._retrieval_error = retrieval_error
        self._profile_rows = tuple(profile_rows)
        self._profile_error = profile_error

    def slide_recovery_state(self, session_number):
        return None

    def list_results_ready_work_orders(self, session_number):
        self.ready_calls += 1
        return self._ready_rows

    def list_hybrid_results(self, session_number):
        self.hybrid_calls += 1
        if self._hybrid_error is not None:
            raise self._hybrid_error
        return self._hybrid_rows

    def list_retrieval_results(self, session_number):
        self.retrieval_calls += 1
        if self._retrieval_error is not None:
            raise self._retrieval_error
        return self._retrieval_rows

    def list_hybrid_profile_rows(self, session_number):
        self.profile_calls += 1
        if self._profile_error is not None:
            raise self._profile_error
        return self._profile_rows

    def wait_for_jobs(self):
        self.wait_for_jobs_calls += 1


def _results_workflow(tmp_path, *, session_mode, store):
    """A ``SessionWorkflow`` wired to a fake store for results_status()
    mode-routing tests.

    ``framing_calibration`` is explicit because the fake store has no
    ``root``. ``outbox``/``transport`` are never used here; real, harmless
    instances stand in so the call stays type-correct."""
    identity = SessionIdentity(
        number=99,
        started_at=WORKFLOW_STARTED_AT,
        directory=tmp_path,
        session_mode=session_mode.value,
    )
    return SessionWorkflow(
        session=identity,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://unused.invalid"),
        framing_calibration=FramingCalibrationStore(tmp_path / "framing.json"),
        session_mode=session_mode,
    )


class StubContactSheetRenderer:
    """Records every call instead of touching cv2; writes a tiny placeholder
    file at ``output_path`` so existence assertions have something real to
    check, mirroring ``StubWorkOrderScorer``'s "record + fake the effect"
    shape."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        output_path = kwargs.get("output_path")
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"stub-contact-sheet")
