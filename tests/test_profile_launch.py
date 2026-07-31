"""Launch plumbing for --profile (#168), mirroring --review-captures (#138)
one-for-one across argparse, main() propagation, and the PowerShell launcher.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("powershell") is None,
    reason="Windows PowerShell binary not available on this platform",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.append(str(_TOOLS_DIR))

import run_pi_session  # noqa: E402

_PS1 = _TOOLS_DIR / "start_live_session.ps1"


def _load_launcher_functions_ps() -> str:
    ps1 = str(_PS1).replace("'", "''")
    tools_dir = str(_TOOLS_DIR).replace("'", "''")
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


def _run_ps1_file(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_PS1),
            *args,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_powershell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_arg_parser_profile_defaults_false():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    )
    assert args.profile is False


def test_build_arg_parser_profile_store_true():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        [
            "--receiver-url",
            "http://127.0.0.1:8077",
            "--session",
            "1",
            "--profile",
        ]
    )
    assert args.profile is True


def test_main_passes_profile_from_argv(monkeypatch, tmp_path):
    captured: dict[str, bool] = {}

    class _Snap:
        phase = "blocks"

    class _Workflow:
        def snapshot(self):
            return _Snap()

        def awaiting_capture_blocks(self):
            return ()

    class _Camera:
        def close(self):
            pass

    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: type(
            "S",
            (),
            {
                "resume_session": lambda self, n: type(
                    "I", (), {"number": n}
                )(),
            },
        )(),
    )
    monkeypatch.setattr(run_pi_session, "HttpCaptureClient", lambda _url: object())
    monkeypatch.setattr(run_pi_session, "PiOutbox", lambda _root: object())
    monkeypatch.setattr(run_pi_session, "FramingCalibrationStore", lambda _path: object())
    monkeypatch.setattr(
        run_pi_session,
        "SessionWorkflow",
        lambda **kwargs: _Workflow(),
    )

    import picamera2_adapter  # noqa: WPS433 — delayed import target in main()

    monkeypatch.setattr(picamera2_adapter, "Picamera2Adapter", lambda: _Camera())

    def _capture_init(
        self,
        workflow,
        camera,
        *,
        capture_root,
        session_config=None,
        action_logger=None,
        review_captures=False,
        open_retrieval=False,
        hybrid=False,
        hybrid_shadow=False,
        profile=False,
    ):
        captured["profile"] = profile

    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "__init__", _capture_init)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "start", lambda self: None)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "close", lambda self: None)
    monkeypatch.setattr(run_pi_session, "_interactive_loop", lambda _w: None)

    argv = [
        "--receiver-url",
        "http://127.0.0.1:8077",
        "--session",
        "1",
        "--profile",
    ]
    assert run_pi_session.main(argv) == 0
    assert captured["profile"] is True

    captured.clear()
    argv_off = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    assert run_pi_session.main(argv_off) == 0
    assert captured["profile"] is False


def test_launcher_help_accepts_profile_flag():
    result = _run_ps1_file("-Mode", "pi", "--session", "1", "--profile", "--help")
    assert result.returncode == 0, result.stderr
    assert "--profile" in result.stdout


def test_launcher_parse_sets_profile_script_var():
    script = _load_launcher_functions_ps() + """
$script:ProfileMode = $false
Parse-RestArgs -List @('--profile')
if (-not $script:ProfileMode) { exit 3 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_appends_profile_when_set():
    script = _load_launcher_functions_ps() + """
$LocalTunnelPort = 8080
$cmd = Build-RemotePiCommand `
  -Repo '/home/esears/ljiblockcheck' `
  -Url 'http://192.168.50.1:8077' `
  -SessionNumber 3 `
  -ChromiumUrl 'http://127.0.0.1:8080' `
  -StartChromium $false `
  -DelaySec 1 `
  -KillStalePiSession $false `
  -ProfileMode $true
if ($cmd -notmatch '--profile') { exit 4 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_omits_profile_when_unset():
    script = _load_launcher_functions_ps() + """
$LocalTunnelPort = 8080
$cmd = Build-RemotePiCommand `
  -Repo '/home/esears/ljiblockcheck' `
  -Url 'http://192.168.50.1:8077' `
  -SessionNumber 3 `
  -ChromiumUrl 'http://127.0.0.1:8080' `
  -StartChromium $false `
  -DelaySec 1 `
  -KillStalePiSession $false `
  -ProfileMode $false
if ($cmd -match '--profile') { exit 5 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--bogus-flag"],
        ["--profil"],
    ],
)
def test_launcher_still_rejects_unknown_args(extra_args):
    result = _run_ps1_file("-Mode", "pi", "--session", "1", *extra_args)
    assert result.returncode == 2


def _fake_runtime_with_capture_logging(
    *, profile: bool, action_logger, store_spy, result_metadata: dict | None
):
    """A bare (``__init__``-free) PiCaptureRuntime with just the attributes
    `_attach_capture_logging`/`_capture_with_log` touch: profile-off must
    produce byte-identical output/files to today; profile-on is covered by
    the pure renderer/row-formatter tests in test_session_console.py and
    test_session_workflow.py."""

    class _FakeControllerSession:
        state = SimpleNamespace(name="AWAITING_CAPTURE")

    class _FakeController:
        def __init__(self, metadata):
            self.session = _FakeControllerSession()
            self.last_failure_timings = None
            self._result = (
                None if metadata is None else SimpleNamespace(metadata=metadata)
            )

        def _capture(self, captured_at):
            return self._result

    runtime = run_pi_session.PiCaptureRuntime.__new__(run_pi_session.PiCaptureRuntime)
    runtime._action_logger = action_logger
    runtime.profile = profile
    runtime.controller = _FakeController(result_metadata)
    runtime.workflow = SimpleNamespace(
        store=store_spy, session=SimpleNamespace(number=7)
    )
    runtime._attach_capture_logging()
    return runtime


def test_capture_with_log_profile_off_prints_no_profile_block(tmp_path, capsys):
    from action_logger import ActionLogger
    from datetime import datetime, timezone

    action_logger = ActionLogger(
        tmp_path / "actions.log", session_number=7
    )
    recorded_rows: list[tuple] = []
    store_spy = SimpleNamespace(
        record_profile_capture=lambda *a, **k: recorded_rows.append((a, k))
    )
    metadata = {
        "camera_capture_ms": 100,
        "publish_ms": 20,
        "consumer_ms": 30,
        "session_accept_ms": 5,
        "total_capture_ms": 155,
        "final_file_size_bytes": 123456,
        "capture_mode": "block",
        "counter": 1,
    }
    runtime = _fake_runtime_with_capture_logging(
        profile=False,
        action_logger=action_logger,
        store_spy=store_spy,
        result_metadata=metadata,
    )

    runtime.controller._capture(datetime.now(timezone.utc))

    out = capsys.readouterr().out
    assert "profile" not in out.lower()
    assert not recorded_rows


def test_capture_with_log_profile_off_writes_no_profile_summary_row(tmp_path):
    from action_logger import ActionLogger
    from datetime import datetime, timezone

    action_logger = ActionLogger(
        tmp_path / "actions.log", session_number=7
    )
    recorded_rows: list[tuple] = []
    store_spy = SimpleNamespace(
        record_profile_capture=lambda *a, **k: recorded_rows.append((a, k))
    )
    metadata = {
        "camera_capture_ms": 100,
        "publish_ms": 20,
        "consumer_ms": 30,
        "session_accept_ms": 5,
        "total_capture_ms": 155,
        "final_file_size_bytes": 123456,
        "capture_mode": "block",
        "counter": 1,
    }
    runtime = _fake_runtime_with_capture_logging(
        profile=False,
        action_logger=action_logger,
        store_spy=store_spy,
        result_metadata=metadata,
    )

    runtime.controller._capture(datetime.now(timezone.utc))

    assert recorded_rows == []
