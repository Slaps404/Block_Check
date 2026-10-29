"""Characterization coverage for the ``session.workflow`` public import façade.

Locks every symbol imported by production code and tests from the facade.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from session import workflow as session_workflow
from session.workflow_public import PUBLIC_SYMBOLS


_REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo call sites live here; skipping worktrees and venvs keeps the scan fast.
_SCAN_DIRS = ("code", "tests", "tools", "scripts")


def _scan_session_workflow_imports() -> set[str]:
    """Collect every name imported via ``from session.workflow import ...``."""
    symbols: set[str] = set()
    paths = [p for d in _SCAN_DIRS for p in (_REPO_ROOT / d).rglob("*.py")]
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "session.workflow" not in text:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
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
