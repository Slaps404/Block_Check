"""Thin command/debug adapter contract: renders events, invokes actions,
owns no phase or recovery rules of its own (#100)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from capture_session import MotionSample
from session.console import (
    dispatch,
    render_events,
    render_motion_sample,
    render_profile_capture_block,
    render_results,
    render_summary,
)
from session.workflow import (
    FailedBlockWarning,
    SessionSummary,
    WorkflowEvent,
    format_profile_summary_row,
)


STARTED_AT = datetime(2026, 7, 2, 18, 5, 6, tzinfo=timezone.utc)


def _summary(**overrides):
    base = dict(
        session_number=1,
        started_at=STARTED_AT,
        sets_processed=3,
        pass_count=2,
        review_count=1,
        missing_slides=("11111111",),
        block_failures=(FailedBlockWarning("22222222", "no tissue found", None),),
        skipped_decodes=("slide_20260702T180506Z_abc123456789",),
        pending_blocks=("33333333",),
        pending_uploads=("capture_000001_block_44444444_20260702T180506Z",),
    )
    base.update(overrides)
    return SessionSummary(**base)


def test_render_summary_default_is_compact_without_detail_lists():
    rendered = render_summary(_summary())

    assert "Session 1" in rendered
    assert "Processed: 3" in rendered
    assert "PASS: 2" in rendered
    assert "REVIEW: 1" in rendered
    assert "11111111" not in rendered
    assert "22222222" not in rendered
    assert "33333333" not in rendered


def test_render_summary_debug_includes_detail_categories():
    rendered = render_summary(_summary(), debug=True)

    assert "11111111" in rendered  # missing slide
    assert "22222222" in rendered  # block failure
    assert "33333333" in rendered  # pending block
    assert "slide_20260702T180506Z_abc123456789" in rendered  # skipped decode
    assert "44444444" in rendered  # pending upload


def test_render_summary_debug_marks_empty_categories_explicitly():
    empty = SessionSummary(
        session_number=2, started_at=STARTED_AT, sets_processed=0,
        pass_count=0, review_count=0,
    )

    rendered = render_summary(empty, debug=True)

    assert "none" in rendered.lower()


def test_render_summary_debug_includes_finalization_error_or_none():
    failed = render_summary(
        _summary(finalization_error="cleanup failed: disk is read-only"), debug=True
    )
    healthy = render_summary(_summary(finalization_error=None), debug=True)

    assert "Finalization error: cleanup failed: disk is read-only" in failed
    assert "Finalization error: none" in healthy


def test_render_events_lists_phase_kind_and_message_per_line():
    events = (
        WorkflowEvent("session_started", 1, "blocks", "Session started"),
        WorkflowEvent("block_scanned", 1, "blocks", "Accepted block 51151378", "51151378"),
    )

    rendered = render_events(events)

    lines = rendered.splitlines()
    assert len(lines) == 2
    assert "session_started" in lines[0] and "Session started" in lines[0]
    assert "51151378" in lines[1] or "Accepted block 51151378" in lines[1]


class _FakeWorkflow:
    def __init__(self):
        self.calls = []

    def scan_block(self, block_id):
        self.calls.append(("scan_block", block_id))
        return "scanned"

    def finish_blocks(self):
        self.calls.append(("finish_blocks",))
        return "draining"

    def end_session(self, *, confirm):
        self.calls.append(("end_session", confirm))
        return "finalizing"

    def start_work_order(self):
        self.calls.append(("start_work_order",))
        return 7

    def finish_work_order(self):
        self.calls.append(("finish_work_order",))
        return 7

    def results_status(self):
        self.calls.append(("results_status",))
        return {"work_orders": (1,), "rows": []}

    def sample_motion(self):
        self.calls.append(("sample_motion",))
        return "Motion sample rendered"


def test_dispatch_invokes_the_named_workflow_action_without_extra_logic():
    workflow = _FakeWorkflow()

    result = dispatch(workflow, "scan_block", "51151378")

    assert result == "scanned"
    assert workflow.calls == [("scan_block", "51151378")]


def test_dispatch_passes_through_confirmation_for_end_session():
    workflow = _FakeWorkflow()

    dispatch(workflow, "end_session")

    assert workflow.calls == [("end_session", True)]


def test_dispatch_rejects_unknown_command_without_touching_the_workflow():
    workflow = _FakeWorkflow()

    with pytest.raises(ValueError, match="unknown command"):
        dispatch(workflow, "delete_everything")

    assert workflow.calls == []


def test_dispatch_invokes_start_and_finish_work_order():
    workflow = _FakeWorkflow()

    start_result = dispatch(workflow, "start_work_order")
    finish_result = dispatch(workflow, "finish_work_order")

    assert start_result == 7
    assert finish_result == 7
    assert workflow.calls == [("start_work_order",), ("finish_work_order",)]


def test_dispatch_invokes_results_status():
    workflow = _FakeWorkflow()

    result = dispatch(workflow, "results")

    assert result == {"work_orders": (1,), "rows": []}
    assert workflow.calls == [("results_status",)]


def test_dispatch_invokes_motion_sampling():
    """`motion` is a debug still-style command (like `snap`): the console
    dispatches it against the runtime handle (not `SessionWorkflow` phase
    logic), 0-arg like `summary`."""
    workflow = _FakeWorkflow()

    result = dispatch(workflow, "motion")

    assert result == "Motion sample rendered"
    assert workflow.calls == [("sample_motion",)]


def test_render_results_with_no_rows_shows_a_clear_sentinel():
    rendered = render_results({"work_orders": (), "rows": []})

    assert "No results yet" in rendered


def test_render_results_sorts_review_before_pass_and_includes_details():
    status = {
        "work_orders": (1,),
        "rows": [
            {
                "capture_id": "capture_000001",
                "block_id": "11111111",
                "verdict": "PASS",
                "claim_score": 0.42,
                "claim_reason": "matched top block",
                "work_order_id": 1,
                "top_block": "11111111",
                "contact_sheet_dir": None,
            },
            {
                "capture_id": "capture_000002",
                "block_id": "22222222",
                "verdict": "REVIEW",
                "claim_score": 0.11,
                "claim_reason": "low confidence match",
                "work_order_id": 1,
                "top_block": "33333333",
                "contact_sheet_dir": None,
            },
        ],
    }

    rendered = render_results(status)
    lines = rendered.splitlines()

    review_line_index = next(i for i, line in enumerate(lines) if "22222222" in line)
    pass_line_index = next(i for i, line in enumerate(lines) if "11111111" in line)
    assert review_line_index < pass_line_index
    assert "REVIEW" in lines[review_line_index]
    assert "low confidence match" in lines[review_line_index]
    assert "PASS" in lines[pass_line_index]
    assert "matched top block" in lines[pass_line_index]
    assert "Work orders: 1" in rendered


def test_render_results_renders_none_claim_score_as_dash():
    status = {
        "work_orders": (1,),
        "rows": [
            {
                "capture_id": "capture_000003",
                "block_id": "44444444",
                "verdict": "REVIEW",
                "claim_score": None,
                "claim_reason": "no candidate blocks",
                "work_order_id": 1,
                "top_block": None,
                "contact_sheet_dir": None,
            },
        ],
    }

    rendered = render_results(status)

    assert "None" not in rendered
    assert "-" in rendered


# ---------------------------------------------------------------------------
# --profile (#168): render_profile_capture_block + format_profile_summary_row
# ---------------------------------------------------------------------------

_PROFILE_FIELDS = {
    "camera_capture_ms": 100,
    "publish_ms": 20,
    "consumer_ms": 30,
    "session_accept_ms": 5,
    "total_capture_ms": 155,
    "final_file_size_bytes": 123456,
    "capture_mode": "block",
}


def test_render_profile_capture_block_formats_all_stages():
    rendered = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)

    assert "100" in rendered
    assert "20" in rendered
    assert "30" in rendered
    assert "5" in rendered
    assert "155" in rendered
    assert "camera" in rendered.lower()
    assert "publish" in rendered.lower()
    assert "consumer" in rendered.lower()
    assert "accept" in rendered.lower()
    assert "total" in rendered.lower()


def test_render_profile_capture_block_marks_slow_capture():
    rendered = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=100)

    assert "[SLOW]" in rendered


def test_render_profile_capture_block_omits_slow_marker_when_fast():
    rendered = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)

    assert "[SLOW]" not in rendered


def test_render_profile_capture_block_is_pure_no_io(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("render_profile_capture_block must not do I/O")

    monkeypatch.setattr("builtins.open", _boom)
    monkeypatch.setattr("builtins.print", _boom)

    first = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)
    second = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)

    assert isinstance(first, str)
    assert first == second


_PROFILE_FIELDS_WITH_CONSUMER_SPLIT = {
    **_PROFILE_FIELDS,
    "consumer_decode_ms": 12,
    "consumer_outbox_ms": 8,
    "consumer_send_ms": 41,
}


def test_render_profile_capture_block_shows_consumer_split_when_present():
    """#171: under --profile, the consumer line explodes into its
    decode/outbox/send sub-durations when the capture carries them."""
    rendered = render_profile_capture_block(
        _PROFILE_FIELDS_WITH_CONSUMER_SPLIT, slow_threshold_ms=3000
    )

    assert "consumer 30ms = decode 12ms + outbox 8ms + send 41ms" in rendered
    assert "consumer: 30ms" not in rendered


def test_render_profile_capture_block_falls_back_to_plain_consumer_line_when_split_absent():
    """#171 regression guard: block-mode captures (no decode stage) keep
    today's plain `consumer: Nms` line untouched."""
    rendered = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)

    assert "consumer: 30ms" in rendered
    assert "decode" not in rendered.lower()
    assert "outbox" not in rendered.lower()
    assert "send" not in rendered.lower()


_PROFILE_FIELDS_WITH_SETTLING = {
    **_PROFILE_FIELDS,
    "settling_duration_ms": 850,
    "settling_resets": 2,
    "settling_max_motion": 0.031,
}


def test_render_profile_capture_block_includes_settling_row_with_duration_resets_and_max_motion():
    """#172: under --profile, the console block gains a settling row showing
    how long the specimen took to stabilize, how many times motion reset the
    stable timer, and the peak motion score seen while settling."""
    rendered = render_profile_capture_block(
        _PROFILE_FIELDS_WITH_SETTLING, slow_threshold_ms=3000
    )

    assert "settling: 850ms (resets=2, max_motion=0.031)" in rendered


def test_render_profile_capture_block_omits_settling_row_when_absent():
    """Regression guard: captures with no settling summary (e.g. block-mode
    or a recapture-after-error path) keep today's output unchanged."""
    rendered = render_profile_capture_block(_PROFILE_FIELDS, slow_threshold_ms=3000)

    assert "settling" not in rendered.lower()


def test_format_profile_summary_row_is_pure_dict_to_row():
    first = format_profile_summary_row("capture_000001", _PROFILE_FIELDS)
    second = format_profile_summary_row("capture_000001", _PROFILE_FIELDS)

    assert first == second
    assert isinstance(first, dict)
    assert first["capture_id"] == "capture_000001"
    assert first["camera_capture_ms"] == 100
    assert first["publish_ms"] == 20
    assert first["consumer_ms"] == 30
    assert first["session_accept_ms"] == 5
    assert first["total_capture_ms"] == 155


# ---------------------------------------------------------------------------
# `motion` console command (#169): render_motion_sample
# ---------------------------------------------------------------------------


def test_render_motion_sample_formats_min_mean_max_crossings_state_and_hint():
    sample = MotionSample(
        min_score=0.0,
        mean_score=0.035,
        max_score=0.10,
        threshold_crossings=4,
        sample_count=6,
    )

    rendered = render_motion_sample(sample, state="SETTLING", threshold=0.02)

    assert "0.0" in rendered
    assert "0.035" in rendered
    assert "0.1" in rendered
    assert "4" in rendered
    assert "6" in rendered
    assert "SETTLING" in rendered
    assert "min" in rendered.lower()
    assert "mean" in rendered.lower()
    assert "max" in rendered.lower()


def test_render_motion_sample_hint_no_crossings():
    sample = MotionSample(
        min_score=0.0, mean_score=0.0, max_score=0.0, threshold_crossings=0,
        sample_count=5,
    )

    rendered = render_motion_sample(sample, state="EMPTY", threshold=0.02)

    assert "no motion above threshold" in rendered.lower()


def test_render_motion_sample_hint_intermittent_crossings():
    sample = MotionSample(
        min_score=0.0, mean_score=0.01, max_score=0.03, threshold_crossings=2,
        sample_count=5,
    )

    rendered = render_motion_sample(sample, state="EMPTY", threshold=0.02)

    assert "intermittent" in rendered.lower()


def test_render_motion_sample_hint_crossings_throughout():
    sample = MotionSample(
        min_score=0.02, mean_score=0.03, max_score=0.05, threshold_crossings=5,
        sample_count=5,
    )

    rendered = render_motion_sample(sample, state="EMPTY", threshold=0.02)

    assert "throughout the window" in rendered.lower()


def test_render_motion_sample_is_pure_no_io(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("render_motion_sample must not do I/O")

    monkeypatch.setattr("builtins.open", _boom)
    monkeypatch.setattr("builtins.print", _boom)
    sample = MotionSample(
        min_score=0.0, mean_score=0.01, max_score=0.02, threshold_crossings=1,
        sample_count=3,
    )

    first = render_motion_sample(sample, state="EMPTY", threshold=0.02)
    second = render_motion_sample(sample, state="EMPTY", threshold=0.02)

    assert isinstance(first, str)
    assert first == second
