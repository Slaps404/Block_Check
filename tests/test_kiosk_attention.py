"""#256: the pure Hybrid attention-banner projection (`kiosk.attention`).

Mirrors `test_kiosk_results_table.py`'s style: synthetic row dicts, no I/O,
no store, no camera. This is the seam `KioskRelay._attention_banner` wraps
in its own try/except; these tests exercise the pure function directly.
"""
from __future__ import annotations

from kiosk.attention import project_attention_banner


def _row(**overrides):
    base = dict(
        capture_id="cap-1", block_id="11111111", work_order_id=5,
        verdict="ERROR", claim_score=None, claim_reason=None,
    )
    base.update(overrides)
    return base


def test_no_banner_when_there_is_no_error_row():
    rows = [_row(verdict="PASS"), _row(verdict="PENDING", capture_id="cap-2")]
    assert project_attention_banner(
        rows, work_order_open=False, open_work_order_id=None,
    ) is None


def test_no_banner_for_empty_rows():
    assert project_attention_banner(
        [], work_order_open=False, open_work_order_id=None,
    ) is None


def test_banner_surfaces_the_first_error_row_and_offers_recapture():
    rows = [_row(verdict="PASS"), _row(verdict="ERROR")]
    banner = project_attention_banner(
        rows, work_order_open=False, open_work_order_id=None,
    )
    assert banner == {
        "capture_id": "cap-1",
        "block_id": "11111111",
        "work_order_id": 5,
        "message": "Slide needs attention: block 11111111",
        "can_recapture": True,
    }


def test_banner_can_recapture_when_no_work_order_is_open():
    rows = [_row()]
    banner = project_attention_banner(
        rows, work_order_open=False, open_work_order_id=None,
    )
    assert banner is not None
    assert banner["can_recapture"] is True


def test_banner_can_recapture_when_its_own_work_order_is_the_one_open():
    rows = [_row(work_order_id=5)]
    banner = project_attention_banner(
        rows, work_order_open=True, open_work_order_id=5,
    )
    assert banner is not None
    assert banner["can_recapture"] is True


def test_banner_waits_while_a_different_work_order_is_actively_capturing():
    """Acceptance criterion: the attention item WAITS for an available
    transition rather than interrupting -- non-vacuous, this is exactly the
    branch a dropped `work_order_open`/`open_work_order_id` check would
    silently always pass."""
    rows = [_row(work_order_id=5)]
    banner = project_attention_banner(
        rows, work_order_open=True, open_work_order_id=9,
    )
    assert banner is not None
    assert banner["can_recapture"] is False
    # The banner itself still surfaces -- "waits" means the recapture route
    # is gated, never that the notice disappears.
    assert banner["message"] == "Slide needs attention: block 11111111"


def test_banner_picks_the_first_error_row_stably():
    rows = [
        _row(capture_id="cap-a", block_id="11111111"),
        _row(capture_id="cap-b", block_id="22222222"),
    ]
    banner = project_attention_banner(
        rows, work_order_open=False, open_work_order_id=None,
    )
    assert banner is not None
    assert banner["capture_id"] == "cap-a"
