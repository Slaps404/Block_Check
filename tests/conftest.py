"""
Pytest fixtures for v2 claimed-pair MVP tests.

Phase 3 fixtures are in archive/phase3/tests/conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CODE_DIR = _REPO_ROOT / "code"
_TOOLS_DIR = _REPO_ROOT / "tools"
_TOOL_SUBDIRS = (
    "capture",
    "manifest",
    "identity",
    "scoring_diagnostics",
    "visual_audit",
    "diagnostics",
    "calibration",
)
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_CAPTURE_DIR = _CODE_DIR / "capture"
if _CAPTURE_DIR.is_dir() and str(_CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAPTURE_DIR))
for _subdir in _TOOL_SUBDIRS:
    _path = _TOOLS_DIR / _subdir
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.append(str(_path))
