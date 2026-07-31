"""Canonical public import surface for ``session.workflow`` (#201 slice 1).

``session.workflow`` remains the stable façade. Later extraction slices may
move implementations into sibling modules, but every name listed here must stay
importable via ``from session.workflow import <name>`` unless a deliberate,
versioned breaking change is made.
"""
from __future__ import annotations

# Production + test imports (repo AST scan, 2026-07-25).
_PUBLIC_NAMES: tuple[str, ...] = (
    "BlockReadiness",
    "ClaimOutcome",
    "FailedBlockWarning",
    "FramingCalibrationStore",
    "HttpCaptureClient",
    "HybridPoolFreezeResult",
    "LoopbackCaptureReceiver",
    "PiOutbox",
    "ProcessingStore",
    "RecaptureOutcome",
    "ScanOutcome",
    "SessionIdentity",
    "SessionSummary",
    "SessionWorkflow",
    "SlideQRResult",
    "UploadReceipt",
    "WorkOrderScoringResult",
    "WorkflowEvent",
    "WorkflowSnapshot",
    "default_debug_snap_dir",
    "default_work_order_scorer",
    "format_profile_summary_row",
    "open_saved_image",
    "save_debug_snap",
)

# Semi-public: RPC whitelist imported by architecture / ledger tests.
_COMPAT_NAMES: tuple[str, ...] = (
    "_RPC_METHODS",
    "build_slide_image_overlay",
)

PUBLIC_SYMBOLS: tuple[str, ...] = _PUBLIC_NAMES + _COMPAT_NAMES

__all__ = list(PUBLIC_SYMBOLS)
