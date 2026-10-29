"""Launcher flag plumbing, one table for every Pi launch flag.

Each flag must travel the same path: `start_live_session.ps1` parses it
(Parse-RestArgs) and forwards it (Build-RemotePiCommand), then
`pi_session` parses it (argparse) and hands it to PiCaptureRuntime. All
PowerShell checks share one process (see `ps`), since each launch costs ~3 s.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from launcher_harness import (
    BASE_ARGV,
    PS1,
    load_launcher_functions_ps,
    patch_main_dependencies,
    requires_powershell,
    run_powershell_file,
    write_hybrid_config,
)

from app import pi_session  # noqa: E402


@dataclass(frozen=True)
class Flag:
    cli: str  # what the operator types; also what Build-RemotePiCommand emits
    attr: str  # argparse dest, and the PiCaptureRuntime kwarg main() forwards
    ps_name: str  # Parse-RestArgs $script: var, and the Build-RemotePiCommand param
    aliases: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.cli


BOOL_FLAGS = (
    Flag("--review-captures", "review_captures", "ReviewCaptures"),
    Flag("--triple-capture", "triple_capture", "TripleCapture"),
    Flag("--led-only", "led_only_blocks", "LedOnlyBlocks", aliases=("--led-only-blocks",)),
    Flag("--open-retrieval", "open_retrieval", "OpenRetrieval"),
    Flag("--hybrid", "hybrid", "Hybrid"),
    Flag("--hybrid-shadow", "hybrid_shadow", "HybridShadow"),
    Flag("--profile", "profile", "ProfileMode"),
    Flag("--allow-occupied-baseline", "allow_occupied_baseline", "AllowOccupiedBaseline"),
    Flag("--skip-removal-wait", "skip_removal_wait", "SkipRemovalWait"),
)
# (cli spelling, flag) for every accepted spelling, aliases included.
BOOL_SPELLINGS = [(f.cli, f) for f in BOOL_FLAGS] + [
    (alias, f) for f in BOOL_FLAGS for alias in f.aliases
]
HYBRID_ATTRS = {"hybrid", "hybrid_shadow"}

# A junk flag and a one-letter typo of a real flag (no prefix matching).
UNKNOWN_ARGS = ["--bogus-flag", "--hybrid-shado"]
CONFLICTING_MODE_ARGS = [
    ["--open-retrieval", "--hybrid"],
    ["--open-retrieval", "--hybrid-shadow"],
    ["--hybrid", "--hybrid-shadow"],
]
# (id, -ResumeSession, extra params, expected run_receiver.py args). A
# mode flag reaches the receiver on --start-session only; --resume N attaches
# to a session whose mode is already durable, so the flag must be dropped.
RECEIVER_ARG_CASES = [
    ("start", "$null", "", "--start-session"),
    ("resume", "7", "", "--resume 7"),
    ("start-open", "$null", "-OpenRetrieval $true", "--start-session --open-retrieval"),
    ("start-hybrid", "$null", "-Hybrid $true", "--start-session --hybrid"),
    ("start-shadow", "$null", "-HybridShadow $true", "--start-session --hybrid-shadow"),
    ("resume-open", "7", "-OpenRetrieval $true", "--resume 7"),
    ("resume-hybrid", "7", "-Hybrid $true", "--resume 7"),
    ("resume-shadow", "7", "-HybridShadow $true", "--resume 7"),
]

_BUILD_COMMON = (
    "-Repo '/home/esears/ljiblockcheck' -Url 'http://192.168.50.1:8077' "
    "-SessionNumber 3 -ChromiumUrl 'http://127.0.0.1:8080' -DelaySec 1 "
    "-KillStalePiSession $false"
)


def _ps_list(items) -> str:
    return "@(" + ", ".join("'" + s.replace("'", "''") + "'" for s in items) + ")"


def _ps_script() -> str:
    """One PowerShell program that records every launcher fact as JSON."""
    lines = [
        "$results = @{}",
        f"$ps1 = '{PS1}'",
        "function Invoke-Launcher([string[]]$CaseArgs) {",
        "  $global:LASTEXITCODE = $null",
        "  try { $out = & $ps1 -Mode pi @CaseArgs *>&1 | Out-String }",
        "  catch { return @{ code = -1; out = \"$_\" } }",
        "  return @{ code = $LASTEXITCODE; out = ($out -replace '[^\\x00-\\x7F]', '?') }",
        "}",
    ]
    # Run the real script in-process (`&`); its `exit N` sets $LASTEXITCODE.
    file_cases = {f"unknown:{a}": [a] for a in UNKNOWN_ARGS}
    file_cases.update({f"conflict:{' '.join(c)}": c for c in CONFLICTING_MODE_ARGS})
    for key, args in file_cases.items():
        lines.append(
            f"$results['file:{key}'] = Invoke-Launcher {_ps_list(['--session', '1', *args])}"
        )

    lines.append(load_launcher_functions_ps())
    for spelling, flag in BOOL_SPELLINGS:
        lines += [
            f"$script:{flag.ps_name} = $false",
            f"Parse-RestArgs -List {_ps_list([spelling])}",
            f"$results['parse:{spelling}'] = $script:{flag.ps_name}",
        ]
    lines += [
        "$script:CameraProfile = 'imx477'",
        "Parse-RestArgs -List @('--camera-profile', 'v2.1')",
        "$results['parse:--camera-profile'] = $script:CameraProfile",
        "$LocalTunnelPort = 8080",
    ]
    all_off = " ".join(f"-{f.ps_name} $false" for f in BOOL_FLAGS)
    lines.append(
        f"$results['build:off'] = Build-RemotePiCommand {_BUILD_COMMON} "
        f"-StartChromium $false {all_off}"
    )
    for flag in BOOL_FLAGS:
        lines.append(
            f"$results['build:{flag.cli}'] = Build-RemotePiCommand {_BUILD_COMMON} "
            f"-StartChromium $false -{flag.ps_name} $true"
        )
    lines += [
        f"$results['build:--camera-profile'] = Build-RemotePiCommand {_BUILD_COMMON} "
        "-StartChromium $false -CameraProfile 'imx219'",
        f"$results['build:chromium'] = Build-RemotePiCommand {_BUILD_COMMON} "
        "-StartChromium $true",
    ]
    for key, resume, extra, _ in RECEIVER_ARG_CASES:
        lines.append(
            f"$results['receiver:{key}'] = Get-ReceiverSessionArgs "
            f"-ResumeSession {resume} {extra}"
        )
    lines.append("Write-Output ('@@JSON@@' + ($results | ConvertTo-Json -Compress -Depth 4))")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def ps(tmp_path_factory) -> dict:
    script = tmp_path_factory.mktemp("launcher") / "launch_flags.ps1"
    script.write_text(_ps_script(), encoding="utf-8")
    result = run_powershell_file(script)
    for line in result.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    pytest.fail(f"launcher probe printed no results:\n{result.stdout}\n{result.stderr}")


def _flag_pattern(cli: str) -> str:
    # Whole-token match, so --hybrid never matches inside --hybrid-shadow.
    return rf"(?<!\S){re.escape(cli)}(?!\S)"


# --------------------------------------------------------------------------
# PowerShell launcher
# --------------------------------------------------------------------------


@requires_powershell
@pytest.mark.parametrize("arg", UNKNOWN_ARGS)
def test_launcher_rejects_unknown_args(ps, arg):
    run = ps[f"file:unknown:{arg}"]
    assert run["code"] == 2, run["out"]
    assert "Unknown argument" in run["out"]


@requires_powershell
@pytest.mark.parametrize("args", CONFLICTING_MODE_ARGS, ids=" ".join)
def test_launcher_rejects_conflicting_scoring_mode_flags(ps, args):
    run = ps[f"file:conflict:{' '.join(args)}"]
    assert run["code"] == 2, run["out"]


@requires_powershell
@pytest.mark.parametrize("spelling", [s for s, _ in BOOL_SPELLINGS])
def test_parse_rest_args_sets_script_var(ps, spelling):
    assert ps[f"parse:{spelling}"] is True


@requires_powershell
def test_parse_rest_args_sets_camera_profile(ps):
    assert ps["parse:--camera-profile"] == "v2.1"


@requires_powershell
@pytest.mark.parametrize("flag", BOOL_FLAGS, ids=str)
def test_build_remote_pi_command_appends_flag_when_set(ps, flag):
    assert re.search(_flag_pattern(flag.cli), ps[f"build:{flag.cli}"])


@requires_powershell
@pytest.mark.parametrize("flag", BOOL_FLAGS, ids=str)
def test_build_remote_pi_command_omits_flag_when_unset(ps, flag):
    assert not re.search(_flag_pattern(flag.cli), ps["build:off"])


@requires_powershell
def test_build_remote_pi_command_appends_camera_profile(ps):
    assert "--camera-profile 'imx219'" in ps["build:--camera-profile"]


@requires_powershell
def test_build_remote_pi_command_recovers_kiosk_chromium_from_handoff_exit(ps):
    """The kiosk process, not Chromium's one-shot handoff PID, is authoritative."""
    cmd = ps["build:chromium"]
    assert re.search(r'--user-data-dir="?\$KIOSK_PROFILE', cmd)
    assert 'pgrep -f -- "--user-data-dir=$KIOSK_PROFILE"' in cmd
    assert 'while [ "$attempt" -le 3 ]' in cmd
    assert "CHROME_PID" not in cmd


@requires_powershell
@pytest.mark.parametrize(
    "key, expected", [(c[0], c[3]) for c in RECEIVER_ARG_CASES], ids=[c[0] for c in RECEIVER_ARG_CASES]
)
def test_get_receiver_session_args(ps, key, expected):
    assert ps[f"receiver:{key}"] == expected


# --------------------------------------------------------------------------
# pi_session argparse + main()
# --------------------------------------------------------------------------


def test_arg_parser_defaults_every_flag_off():
    args = pi_session.build_arg_parser().parse_args(BASE_ARGV)
    for flag in BOOL_FLAGS:
        assert getattr(args, flag.attr) is False, flag.cli
    assert args.camera_profile == "imx477"


def test_arg_parser_accepts_camera_profile_imx219():
    args = pi_session.build_arg_parser().parse_args([*BASE_ARGV, "--camera-profile", "imx219"])
    assert args.camera_profile == "imx219"


@pytest.mark.parametrize(
    "spelling, flag", [(None, None), *BOOL_SPELLINGS],
    ids=["no-flag", *[s for s, _ in BOOL_SPELLINGS]],
)
def test_main_forwards_flag_to_pi_capture_runtime(monkeypatch, tmp_path, spelling, flag):
    captured = patch_main_dependencies(monkeypatch)
    argv = list(BASE_ARGV)
    if flag is not None:
        argv.append(spelling)
        if flag.attr in HYBRID_ATTRS:
            argv += ["--hybrid-config", str(write_hybrid_config(tmp_path / "handoff.json"))]

    assert pi_session.main(argv) == 0
    for other in BOOL_FLAGS:
        assert bool(captured["runtime"].get(other.attr, False)) is (other is flag), other.cli


@pytest.mark.parametrize(
    "mode_flag, label, absent",
    [
        (None, "Mode: normal verification", None),
        ("--open-retrieval", "Mode: Open Retrieval", None),
        ("--hybrid", "Mode: Hybrid", "Hybrid Shadow"),
        ("--hybrid-shadow", "Mode: Hybrid Shadow", None),
    ],
)
def test_main_prints_scoring_mode_label(monkeypatch, capsys, tmp_path, mode_flag, label, absent):
    patch_main_dependencies(monkeypatch)
    argv = list(BASE_ARGV)
    if mode_flag:
        argv.append(mode_flag)
    if mode_flag in ("--hybrid", "--hybrid-shadow"):
        argv += ["--hybrid-config", str(write_hybrid_config(tmp_path / "handoff.json"))]

    assert pi_session.main(argv) == 0
    stdout = capsys.readouterr().out
    assert label in stdout
    if absent:
        assert absent not in stdout


@pytest.mark.parametrize("flag", [
    "--triple-capture", "--allow-occupied-baseline", "--skip-removal-wait",
])
def test_led_flag_rejects_conflicts_before_hardware_start(flag):
    with pytest.raises(SystemExit) as exc:
        pi_session.main([*BASE_ARGV, "--led-only", flag])
    assert exc.value.code == 2


def test_install_camera_release_signals_registers_available_handlers(monkeypatch):
    """SIGHUP/SIGTERM must exit through SystemExit so finally closes the camera."""
    registered: dict[object, object] = {}

    def _fake_signal(sig, handler):
        registered[sig] = handler

    monkeypatch.setattr(pi_session.signal, "signal", _fake_signal)
    installed = pi_session.install_camera_release_signals()
    assert installed  # at least SIGTERM on Windows; SIGHUP+SIGTERM on Pi
    for name in installed:
        assert registered[getattr(pi_session.signal, name)] is pi_session._request_shutdown
    with pytest.raises(SystemExit) as raised:
        pi_session._request_shutdown(None, None)
    assert raised.value.code == 0


def test_interactive_loop_returns_on_keyboard_interrupt(monkeypatch):
    def _boom(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _boom)
    # Returning cleanly (no raise) lets main()'s finally close the camera.
    pi_session._interactive_loop(object())  # type: ignore[arg-type]


def test_capture_with_log_profile_off_prints_and_records_nothing(tmp_path, capsys):
    """Profile off must leave capture output untouched: no profile block, no row.

    Profile-on is covered by the renderer/row tests in test_session_console.py
    and test_session_workflow.py.
    """
    from action_logger import ActionLogger

    recorded_rows: list[tuple] = []
    metadata = {
        "camera_capture_ms": 100, "publish_ms": 20, "consumer_ms": 30,
        "session_accept_ms": 5, "total_capture_ms": 155,
        "final_file_size_bytes": 123456, "capture_mode": "block", "counter": 1,
    }

    class _FakeController:
        session = SimpleNamespace(state=SimpleNamespace(name="AWAITING_CAPTURE"))
        last_failure_timings = None

        def _capture(self, captured_at):
            return SimpleNamespace(metadata=metadata)

    # `__init__`-free runtime with only what _attach_capture_logging touches.
    runtime = pi_session.PiCaptureRuntime.__new__(pi_session.PiCaptureRuntime)
    runtime._action_logger = ActionLogger(tmp_path / "actions.log", session_number=7)
    runtime.profile = False
    runtime.controller = _FakeController()
    runtime.workflow = SimpleNamespace(
        store=SimpleNamespace(record_profile_capture=lambda *a, **k: recorded_rows.append((a, k))),
        session=SimpleNamespace(number=7),
    )
    runtime._attach_capture_logging()

    runtime.controller._capture(datetime.now(timezone.utc))

    assert "profile" not in capsys.readouterr().out.lower()
    assert recorded_rows == []
