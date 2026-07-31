"""Launch plumbing for --review-captures (#138)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

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


def test_install_camera_release_signals_registers_available_handlers(monkeypatch):
    """SIGHUP/SIGTERM must exit through SystemExit so finally closes the camera."""
    registered: dict[object, object] = {}

    def _fake_signal(sig, handler):
        registered[sig] = handler
        return None

    monkeypatch.setattr(run_pi_session.signal, "signal", _fake_signal)
    installed = run_pi_session.install_camera_release_signals()
    assert installed  # at least SIGTERM on Windows; SIGHUP+SIGTERM on Pi
    for name in installed:
        sig = getattr(run_pi_session.signal, name)
        assert registered[sig] is run_pi_session._request_shutdown
    with pytest.raises(SystemExit) as raised:
        run_pi_session._request_shutdown(None, None)
    assert raised.value.code == 0


def test_interactive_loop_returns_on_keyboard_interrupt(monkeypatch):
    def _boom(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _boom)
    run_pi_session._interactive_loop(object())  # type: ignore[arg-type]
    # Returned cleanly (no raise) so main()'s finally can close the camera.


def test_build_arg_parser_review_captures_defaults_false():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    )
    assert args.review_captures is False


def test_build_arg_parser_review_captures_store_true():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        [
            "--receiver-url",
            "http://127.0.0.1:8077",
            "--session",
            "1",
            "--review-captures",
        ]
    )
    assert args.review_captures is True


def test_main_passes_review_captures_from_argv(monkeypatch, tmp_path):
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
        captured["review_captures"] = review_captures

    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "__init__", _capture_init)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "start", lambda self: None)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "close", lambda self: None)
    monkeypatch.setattr(run_pi_session, "_interactive_loop", lambda _w: None)

    argv = [
        "--receiver-url",
        "http://127.0.0.1:8077",
        "--session",
        "1",
        "--review-captures",
    ]
    assert run_pi_session.main(argv) == 0
    assert captured["review_captures"] is True

    captured.clear()
    argv_off = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    assert run_pi_session.main(argv_off) == 0
    assert captured["review_captures"] is False


def test_launcher_help_accepts_review_captures_flag():
    result = _run_ps1_file("-Mode", "pi", "--session", "1", "--review-captures", "--help")
    assert result.returncode == 0, result.stderr
    assert "--review-captures" in result.stdout


def test_launcher_parse_sets_review_captures_script_var():
    script = _load_launcher_functions_ps() + """
$script:ReviewCaptures = $false
Parse-RestArgs -List @('--review-captures')
if (-not $script:ReviewCaptures) { exit 3 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_appends_review_captures_when_set():
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
  -ReviewCaptures $true
if ($cmd -notmatch '--review-captures') { exit 4 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_omits_review_captures_when_unset():
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
  -ReviewCaptures $false
if ($cmd -match '--review-captures') { exit 5 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--bogus-flag"],
        ["--review-capture"],
    ],
)
def test_launcher_still_rejects_unknown_args(extra_args):
    result = _run_ps1_file("-Mode", "pi", "--session", "1", *extra_args)
    assert result.returncode == 2
