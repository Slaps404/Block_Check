"""Characterization coverage for the ``session.workflow`` public import façade (#201).

Locks every symbol currently imported by production code and tests before
extraction slices move implementations out of ``workflow.py``.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from session import workflow as session_workflow
from session.workflow_public import PUBLIC_SYMBOLS


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _scan_session_workflow_imports() -> set[str]:
    """Collect every name imported via ``from session.workflow import ...``."""
    symbols: set[str] = set()
    for path in _REPO_ROOT.rglob("*.py"):
        if "venv" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "session.workflow":
                continue
            for alias in node.names:
                symbols.add(alias.name)
    return symbols


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_session_workflow_exports_public_symbol(name: str) -> None:
    assert hasattr(session_workflow, name), (
        f"{name!r} is documented in workflow_public.PUBLIC_SYMBOLS "
        "but missing from session.workflow"
    )


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_from_session_workflow_import_public_symbol(name: str) -> None:
    module = importlib.import_module("session.workflow")
    imported = getattr(importlib.import_module("session.workflow"), name)
    assert imported is getattr(module, name)


def test_session_workflow_all_matches_public_registry() -> None:
    assert tuple(session_workflow.__all__) == PUBLIC_SYMBOLS


def test_public_registry_covers_repo_imports() -> None:
    imported = _scan_session_workflow_imports()
    documented = set(PUBLIC_SYMBOLS)
    missing = imported - documented
    assert not missing, (
        "Repo imports from session.workflow that are not in PUBLIC_SYMBOLS: "
        f"{sorted(missing)}"
    )


def test_session_package_does_not_eagerly_reexport_workflow_symbols() -> None:
    """Package-level mirrors were removed: they cycle contact_sheet ↔ session."""
    import session

    for name in PUBLIC_SYMBOLS:
        assert not hasattr(session, name), (
            f"{name!r} must not live on the session package root; "
            "import from session.workflow instead"
        )


_WORKFLOW_TYPE_REEXPORTS: tuple[str, ...] = (
    "BlockReadiness",
    "ClaimOutcome",
    "FailedBlockWarning",
    "ScanOutcome",
    "SessionIdentity",
    "SessionSummary",
    "UploadReceipt",
    "WorkOrderScoringResult",
    "WorkflowEvent",
    "WorkflowSnapshot",
)


@pytest.mark.parametrize("name", _WORKFLOW_TYPE_REEXPORTS)
def test_workflow_types_reexport_identity(name: str) -> None:
    """Slice 2: moved types must be the same object via session.workflow."""
    from session import workflow_types

    assert getattr(session_workflow, name) is getattr(workflow_types, name)


_SLICE3_TRANSPORT_REEXPORTS: tuple[str, ...] = (
    "PiOutbox",
    "HttpCaptureClient",
    "default_debug_snap_dir",
    "open_saved_image",
    "save_debug_snap",
)

_SLICE3_RPC_REEXPORTS: tuple[str, ...] = (
    "LoopbackCaptureReceiver",
    "_RPC_METHODS",
)


@pytest.mark.parametrize("name", _SLICE3_TRANSPORT_REEXPORTS)
def test_outbox_transport_reexport_identity(name: str) -> None:
    """Slice 3: transport symbols must be the same object via session.workflow."""
    from session import outbox_transport

    assert getattr(session_workflow, name) is getattr(outbox_transport, name)


@pytest.mark.parametrize("name", _SLICE3_RPC_REEXPORTS)
def test_rpc_server_reexport_identity(name: str) -> None:
    """Slice 3: RPC symbols must be the same object via session.workflow."""
    from session import rpc_server

    assert getattr(session_workflow, name) is getattr(rpc_server, name)


def test_rpc_arity_reexport_identity() -> None:
    from session import rpc_server

    assert session_workflow._RPC_ARITY is rpc_server._RPC_ARITY


_SLICE4_STORE_REEXPORTS: tuple[str, ...] = (
    "ProcessingStore",
    "default_work_order_scorer",
    "format_profile_summary_row",
)


@pytest.mark.parametrize("name", _SLICE4_STORE_REEXPORTS)
def test_processing_store_reexport_identity(name: str) -> None:
    """Slice 4: store symbols must be the same object via session.workflow."""
    from session import processing_store

    assert getattr(session_workflow, name) is getattr(processing_store, name)


_SLICE4_OVERLAY_REEXPORTS: tuple[str, ...] = (
    "build_slide_image_overlay",
)


@pytest.mark.parametrize("name", _SLICE4_OVERLAY_REEXPORTS)
def test_slide_image_overlay_reexport_identity(name: str) -> None:
    """Slice 4: overlay helper stays importable via session.workflow."""
    from verify import slide_image_overlay

    assert getattr(session_workflow, name) is getattr(slide_image_overlay, name)
