"""Thin command/debug adapter over `SessionWorkflow`.

Renders workflow state and dispatches named actions. Owns no phase or
recovery rules of its own; a future touchscreen replaces this module without
moving any workflow logic.
"""
from __future__ import annotations

from typing import Callable

from capture_session import MotionSample
from session.workflow import SessionSummary, SessionWorkflow, WorkflowEvent


def _joined(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


# Stage key -> printed label, in display order. Keys mirror
# `capture_runtime.CAPTURE_STAGE_TIMING_KEYS` minus the non-timing keys
# (`final_file_size_bytes`, `capture_mode`) that ride along in the same dict.
_PROFILE_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("camera_capture_ms", "camera"),
    ("publish_ms", "publish"),
    ("consumer_ms", "consumer"),
    ("session_accept_ms", "accept"),
    ("total_capture_ms", "total"),
)


def render_profile_capture_block(fields: dict, *, slow_threshold_ms: float) -> str:
    """Pure header + indented per-stage text block for one ``--profile`` capture.

    `fields` is the stage-timing subset of `SuccessfulCapture.metadata` (the
    same dict `_capture_with_log` already builds from
    `CAPTURE_STAGE_TIMING_KEYS`). No I/O -- caller prints the result.
    """
    total_ms = fields.get("total_capture_ms")
    slow_marker = (
        " [SLOW]" if total_ms is not None and total_ms > slow_threshold_ms else ""
    )
    lines = [f"Profile{slow_marker}:"]
    consumer_split_keys = ("consumer_decode_ms", "consumer_outbox_ms", "consumer_send_ms")
    has_consumer_split = all(key in fields for key in consumer_split_keys)
    for key, label in _PROFILE_STAGE_LABELS:
        if key == "consumer_ms" and has_consumer_split:
            lines.append(
                f"  consumer {fields['consumer_ms']}ms = "
                f"decode {fields['consumer_decode_ms']}ms + "
                f"outbox {fields['consumer_outbox_ms']}ms + "
                f"send {fields['consumer_send_ms']}ms"
            )
            continue
        if key in fields:
            lines.append(f"  {label}: {fields[key]}ms")
    settling_keys = ("settling_duration_ms", "settling_resets", "settling_max_motion")
    if all(key in fields for key in settling_keys):
        lines.append(
            f"  settling: {fields['settling_duration_ms']}ms "
            f"(resets={fields['settling_resets']}, "
            f"max_motion={fields['settling_max_motion']})"
        )
    return "\n".join(lines)


def render_summary(summary: SessionSummary, *, debug: bool = False) -> str:
    """Compact PASS/REVIEW counts by default; full detail categories in debug mode."""
    lines = [
        f"Session {summary.session_number} ({summary.started_at.isoformat()})",
        f"Processed: {summary.sets_processed}  "
        f"PASS: {summary.pass_count}  REVIEW: {summary.review_count}",
    ]
    if debug:
        lines.append(f"Missing slides: {_joined(summary.missing_slides)}")
        lines.append(
            "Block failures: "
            + _joined(tuple(w.block_id for w in summary.block_failures))
        )
        lines.append(f"Skipped decodes: {_joined(summary.skipped_decodes)}")
        lines.append(f"Pending blocks: {_joined(summary.pending_blocks)}")
        lines.append(f"Pending uploads: {_joined(summary.pending_uploads)}")
        lines.append(
            f"Finalization error: {summary.finalization_error or 'none'}"
        )
    return "\n".join(lines)


def render_events(events: tuple[WorkflowEvent, ...]) -> str:
    return "\n".join(
        f"[{event.phase}] {event.kind}: {event.message}" for event in events
    )


def render_results(status: dict) -> str:
    """Human-readable dump of `SessionWorkflow.results_status()` for the
    `pi>` prompt -- REVIEW rows first (mirrors `project_results_table`'s
    REVIEW-first ordering) so rows needing attention are on top. Pure and
    dependency-free: no I/O, just dict -> str."""
    rows = list(status.get("rows") or ())
    if not rows:
        return "No results yet -- a work order must finish scoring before verdicts appear."
    work_orders = tuple(str(wo) for wo in (status.get("work_orders") or ()))
    lines = [f"Work orders: {_joined(work_orders)}"]
    ordered = sorted(rows, key=lambda row: 0 if row.get("verdict") == "REVIEW" else 1)
    for row in ordered:
        score = row.get("claim_score")
        score_text = "-" if score is None else str(score)
        lines.append(
            f"work_order={row.get('work_order_id')}  "
            f"verdict={row.get('verdict')}  "
            f"block={row.get('block_id')}  "
            f"score={score_text}  "
            f"reason={row.get('claim_reason')}"
        )
    return "\n".join(lines)


def render_motion_sample(sample: MotionSample, *, state: str, threshold: float) -> str:
    """Pure min/mean/max/crossings/state block plus a one-line diagnosis hint
    for the `motion` console command (#169). No I/O -- caller prints the
    result."""
    if sample.threshold_crossings <= 0:
        hint = "No motion above threshold during the sample window"
    elif sample.threshold_crossings >= sample.sample_count:
        hint = "Motion above threshold throughout the window"
    else:
        hint = "Intermittent motion above threshold during the window"
    lines = [
        "Motion sample:",
        f"  state: {state}",
        f"  threshold: {threshold}",
        f"  min: {sample.min_score}",
        f"  mean: {sample.mean_score}",
        f"  max: {sample.max_score}",
        f"  crossings: {sample.threshold_crossings}/{sample.sample_count}",
        f"  hint: {hint}",
    ]
    return "\n".join(lines)


_COMMANDS: dict[str, Callable[[SessionWorkflow, tuple[str, ...]], object]] = {
    "scan_block": lambda workflow, args: workflow.scan_block(args[0]),
    # Handheld-scanner front door via PiCaptureRuntime.scan_qr (not
    # SessionWorkflow); the console dispatches against the runtime handle in
    # run_pi_session. Routes to scan_block in block mode, stashes the slide
    # identity in slide mode.
    "scan_qr": lambda workflow, args: workflow.scan_qr(args[0]),
    "confirm_empty": lambda workflow, args: workflow.confirm_empty(),
    "retry_capture": lambda workflow, args: workflow.retry_capture(),
    "accept_capture": lambda workflow, args: workflow.accept_capture(),
    "skip_slide": lambda workflow, args: workflow.skip_unreadable_slide(),
    "finish_blocks": lambda workflow, args: workflow.finish_blocks(),
    "poll_status": lambda workflow, args: workflow.poll_status(),
    "poll_drain": lambda workflow, args: workflow.poll_drain(),
    "summary": lambda workflow, args: workflow.summarize(),
    "events": lambda workflow, args: workflow.events(),
    "end_session": lambda workflow, args: workflow.end_session(confirm=True),
    "poll_finalization": lambda workflow, args: workflow.poll_finalization(),
    # Debug still via PiCaptureRuntime.snap (not SessionWorkflow); console
    # dispatches against the runtime handle in run_pi_session.
    "snap": lambda workflow, args: workflow.snap(),
    # Motion diagnostic via PiCaptureRuntime.sample_motion (not
    # SessionWorkflow); console dispatches against the runtime handle in
    # run_pi_session.
    "motion": lambda workflow, args: workflow.sample_motion(),
    "start_work_order": lambda workflow, args: workflow.start_work_order(),
    "finish_work_order": lambda workflow, args: workflow.finish_work_order(),
    "results": lambda workflow, args: workflow.results_status(),
    # #257: View Results while a slide-capture work order is open pauses
    # automatic capture only (no store/job side effects).
    "pause_capture": lambda workflow, args: workflow.pause_capture(),
    "resume_capture": lambda workflow, args: workflow.resume_capture(),
    # #256: operator-triggered retry of a Hybrid Processing Error, dispatched
    # from the attention banner's correction-flow screen with the errored
    # capture's id (client-supplied, mirrors scan_block/scan_qr's own single
    # client-supplied arg).
    "retry_hybrid_slide": lambda workflow, args: workflow.retry_hybrid_slide(args[0]),
    # #256 follow-up: arm the runtime so the NEXT captured slide supersedes
    # a Hybrid attention item, dispatched from the same correction-flow
    # screen's own new button, with the errored capture's id (client-
    # supplied, mirrors retry_hybrid_slide's own single client-supplied arg).
    "arm_hybrid_recapture": (
        lambda workflow, args: workflow.arm_hybrid_recapture(args[0])
    ),
    "disarm_hybrid_recapture": (
        lambda workflow, args: workflow.disarm_hybrid_recapture()
    ),
}

# One-line descriptions, keyed identically to `_COMMANDS` -- the single
# source of truth for which commands exist. `command_cheat_sheet()` renders
# these; a test asserts the key sets stay identical so this cannot drift.
COMMAND_HELP: dict[str, str] = {
    "scan_block": "Scan a block by its 8-digit id",
    "scan_qr": "Scanner input: block id in block mode, slide payload in slide mode",
    "confirm_empty": "Confirm the backlight is empty and build a baseline",
    "retry_capture": (
        "Retry the same specimen after a camera error or unreadable-slide "
        "reposition retake"
    ),
    "accept_capture": "Accept the held still and commit it to the session",
    "skip_slide": "Skip the current unreadable slide (durably)",
    "finish_blocks": "Close block intake and begin draining to slides",
    "poll_status": "One presentation-loop poll; also drains reconnects",
    "poll_drain": "Poll block-drain progress toward slide mode",
    "summary": "Show the session summary (processed/PASS/REVIEW)",
    "events": "Show the session event log",
    "end_session": "Confirm and begin finalization",
    "poll_finalization": "Poll finalization progress to completion",
    "snap": "Debug still -> laptop Desktop/pi_captures; delete Pi temp",
    "motion": "Sample motion for a few seconds and report min/mean/max",
    "start_work_order": "Open a work order: everything captured next belongs to it",
    "finish_work_order": "Close the open work order and score it in the background",
    "results": "Show scored work-order verdicts (PASS/REVIEW rows) for this session",
    "pause_capture": "Pause automatic capture only (View Results while a work order is open)",
    "resume_capture": "Resume automatic capture after Results is dismissed",
    "retry_hybrid_slide": "Retry a Hybrid Processing Error from its durable capture",
    "arm_hybrid_recapture": (
        "Arm the next captured slide to supersede a Hybrid attention item"
    ),
    "disarm_hybrid_recapture": "Cancel a pending Hybrid recapture arm",
}

# Argument hints shown next to the command name for commands that take args.
_COMMAND_ARGS: dict[str, str] = {
    "scan_block": "<id>",
    "scan_qr": "<payload>",
    "retry_hybrid_slide": "<capture_id>",
    "arm_hybrid_recapture": "<capture_id>",
}

COMMAND_ARITY: dict[str, int] = {
    name: (1 if name in _COMMAND_ARGS else 0) for name in _COMMANDS
}


def command_cheat_sheet() -> str:
    """Aligned `<command> [<args>]   <description>` list, one line per command."""
    usages = {
        name: f"{name} {_COMMAND_ARGS[name]}" if name in _COMMAND_ARGS else name
        for name in _COMMANDS
    }
    width = max(len(usage) for usage in usages.values())
    return "\n".join(
        f"{usages[name].ljust(width)}   {COMMAND_HELP[name]}" for name in _COMMANDS
    )


def dispatch(workflow: SessionWorkflow, command: str, *args: str) -> object:
    """Invoke one named workflow action; the adapter decides nothing itself."""
    try:
        handler = _COMMANDS[command]
    except KeyError:
        raise ValueError(f"unknown command: {command}") from None
    expected = COMMAND_ARITY[command]
    if len(args) != expected:
        raise ValueError(
            f"{command} expects {expected} argument(s); received {len(args)}"
        )
    return handler(workflow, args)
