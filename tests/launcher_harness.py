"""Shared harness for the launch-flag tests (test_launch_flags, test_hybrid_launch).

Owns the PowerShell plumbing for `tools/start_live_session.ps1`, a `main()`
patch that records what reaches SessionWorkflow/PiCaptureRuntime, and a
self-consistent synthetic Hybrid Configuration.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

requires_powershell = pytest.mark.skipif(
    shutil.which("powershell") is None,
    reason="Windows PowerShell binary not available on this platform",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
# Some tests run `tools/*.py` directly, so `tools/` must be importable.
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

PS1 = TOOLS_DIR / "start_live_session.ps1"

BASE_ARGV = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]


def load_launcher_functions_ps() -> str:
    """PowerShell that dot-sources the launcher's function block only.

    Slicing from `function Show-Usage` to `# Main` keeps the script's own
    main body from running, so a test can call one function directly.
    """
    ps1 = str(PS1).replace("'", "''")
    tools_dir = str(TOOLS_DIR).replace("'", "''")
    return f"""
$PSScriptRoot = '{tools_dir}'
$text = Get-Content -LiteralPath '{ps1}' -Raw
$start = $text.IndexOf('function Show-Usage')
$endMarker = [string]::Concat([char]35, ' Main')
$end = $text.IndexOf($endMarker)
if ($start -lt 0 -or $end -lt 0 -or $end -le $start) {{
  throw 'Could not slice launcher functions from script'
}}
. ([scriptblock]::Create($text.Substring($start, $end - $start)))
"""


def run_powershell_file(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def patch_main_dependencies(monkeypatch, *, durable_session_mode: str | None = None) -> dict:
    """Fake every `pi_session.main()` side effect; return what it built.

    The returned dict gains ``"workflow"`` (SessionWorkflow kwargs) and
    ``"runtime"`` (PiCaptureRuntime kwargs) as main() reaches them. The
    runtime fake takes ``**kwargs`` so new constructor kwargs never break it.
    ``durable_session_mode=None`` leaves the faked session without a
    ``session_mode`` attribute, which main()'s session-mode guard treats as "skip".
    """
    from app import pi_session

    captured: dict = {}

    class _Workflow:
        def snapshot(self):
            return type("Snap", (), {"phase": "blocks"})()

        def awaiting_capture_blocks(self):
            return ()

    class _Identity:
        def __init__(self, number):
            self.number = number
            if durable_session_mode is not None:
                self.session_mode = durable_session_mode

    class _Store:
        def resume_session(self, number):
            return _Identity(number)

    class _Camera:
        def close(self):
            pass

    def _workflow(**kwargs):
        captured["workflow"] = kwargs
        return _Workflow()

    def _runtime_init(self, workflow, camera, **kwargs):
        captured["runtime"] = kwargs

    import picamera2_adapter  # noqa: WPS433 -- delayed import target in main()

    monkeypatch.setattr(picamera2_adapter, "Picamera2Adapter", _Camera)
    monkeypatch.setattr(pi_session, "RemoteProcessingStore", lambda _url: _Store())
    monkeypatch.setattr(pi_session, "HttpCaptureClient", lambda _url: object())
    monkeypatch.setattr(pi_session, "PiOutbox", lambda _root: object())
    monkeypatch.setattr(pi_session, "FramingCalibrationStore", lambda _path: object())
    monkeypatch.setattr(pi_session, "SessionWorkflow", _workflow)
    monkeypatch.setattr(pi_session, "ActionLogger", lambda *a, **k: None)
    monkeypatch.setattr(pi_session, "_build_illumination", lambda **_: None)
    monkeypatch.setattr(pi_session, "install_camera_release_signals", lambda: ())
    monkeypatch.setattr(pi_session.PiCaptureRuntime, "__init__", _runtime_init)
    monkeypatch.setattr(pi_session.PiCaptureRuntime, "start", lambda self: None)
    monkeypatch.setattr(pi_session.PiCaptureRuntime, "close", lambda self: None)
    monkeypatch.setattr(pi_session, "_interactive_loop", lambda _w: None)
    return captured


def valid_synthetic_hybrid_config_payload() -> dict:
    """A config built from the loader's own catalog and hash helper.

    "Valid" means "the loader's own rules are satisfied", not a hand-picked
    expected value.
    """
    from retrieval.hybrid_configuration import (
        HYBRID_CONFIGURATION_SCHEMA_VERSION,
        REQUIRED_FALLBACK_IDS,
        current_implementation_hashes,
    )
    from retrieval.invariant_descriptors import descriptor_catalog

    spec = descriptor_catalog()[0]
    return {
        "schema_version": HYBRID_CONFIGURATION_SCHEMA_VERSION,
        "architecture": {
            "kind": "individual", "name": spec.name, "methods": [spec.name],
        },
        "descriptor_recipe": [
            {
                "name": spec.name, "version": spec.version, "dimension": spec.dimension,
                "comparison": spec.comparison, "prior_evidence": spec.prior_evidence,
            }
        ],
        "candidate_band_thresholds": {spec.name: 0.1},
        "veto": {"enabled": False, "threshold": None, "reason": "disabled: synthetic fixture"},
        "candidate_evidence": {
            "mean_candidate_count": 2.0, "median_candidate_count": 2.0,
            "p95_candidate_count": 3, "max_candidate_count": 4,
            "observed_runtime_seconds": 0.01, "estimated_runtime_seconds": 0.01,
            "full_comparison_reduction": 0.5,
        },
        "known_misses": [],
        "weak_stratum": None,
        "required_fallbacks": list(REQUIRED_FALLBACK_IDS),
        "provenance": {
            "manifest_path": "synthetic.csv", "manifest_hash": "0" * 64,
            "code_revision": "synthetic",
            "implementation_hashes": current_implementation_hashes(),
            "calibration_run_id": "synthetic-run-1",
        },
        "status": "proof_of_concept_not_production_approved",
    }


def write_hybrid_config(path: Path, payload: dict | None = None) -> Path:
    if payload is None:
        payload = valid_synthetic_hybrid_config_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
