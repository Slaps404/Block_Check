"""Production observer seam tests for the runtime harness."""

from __future__ import annotations

import numpy as np

from session.preparation import PreparedSpecimen
from session.pipeline import decide_claim
from runtime_observer import NullRuntimeObserver, observed
from verify.scorer import score_pair_result_routed


class RecordingObserver:
    def __init__(self) -> None:
        self.records: list[tuple[str, int, str]] = []

    def record(self, stage: str, elapsed_ns: int, item_id: str) -> None:
        self.records.append((stage, elapsed_ns, item_id))


def _specimen(role: str) -> PreparedSpecimen:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[300:700, 300:700] = 255
    return PreparedSpecimen(role=role, mask=mask, roi_ok=True, roi_reason="ok")


def test_null_observer_accepts_records_without_effect():
    observer = NullRuntimeObserver()

    observer.record("alignment_scoring", 10, "claim-1")


def test_observed_records_elapsed_time_and_item_id():
    observer = RecordingObserver()

    with observed(observer, "decode_load", "block-1"):
        pass

    assert observer.records[0][0] == "decode_load"
    assert observer.records[0][1] >= 0
    assert observer.records[0][2] == "block-1"


def test_default_observer_does_not_change_score():
    block = _specimen("block")
    slide = _specimen("slide")

    baseline = score_pair_result_routed(block, slide)
    candidate = score_pair_result_routed(block, slide, observer=None)

    assert candidate == baseline


def test_scorer_records_locked_cache_and_alignment_stages():
    observer = RecordingObserver()

    score_pair_result_routed(
        _specimen("block"),
        _specimen("slide"),
        observer=observer,
        item_id="claim-1",
    )

    assert {"locked_cache", "alignment_scoring"} <= {
        stage for stage, _, _ in observer.records
    }


def test_decide_claim_records_quality_gate_and_scoring_stages():
    observer = RecordingObserver()

    decide_claim(
        "claim-1",
        _specimen("block"),
        _specimen("slide"),
        observer=observer,
    )

    assert {"quality_gates", "locked_cache", "alignment_scoring"} <= {
        stage for stage, _, _ in observer.records
    }
