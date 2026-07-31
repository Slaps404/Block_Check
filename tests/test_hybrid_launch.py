"""Launch plumbing for --hybrid / --hybrid-shadow (#247).

Mirrors test_open_retrieval_launch.py / test_profile_launch.py one-for-one:
argparse defaults/store_true, main() propagation + startup mode label, the
PowerShell launcher's parse/help/Build-RemotePiCommand wiring -- plus the
conflicting-flag rejection this prefactor exists to add.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
from session.session_mode import SessionMode  # noqa: E402

_PS1 = _TOOLS_DIR / "start_live_session.ps1"
_STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# #249 synthetic Hybrid Configuration fixture helper
# --------------------------------------------------------------------------
# #245: the curated calibration manifest and #242's real handoff do not exist
# yet, so every test here exercises the loader/startup-gate contract with a
# self-consistent synthetic handoff -- built from the SAME production
# descriptor catalog and implementation-hash helper the loader itself uses,
# so "valid" here means "the loader's own rules are satisfied", not a
# hand-picked value that happens to match a hardcoded expectation.


def _valid_synthetic_hybrid_config_payload() -> dict:
    from session.hybrid_configuration import (
        HYBRID_CONFIGURATION_SCHEMA_VERSION,
        REQUIRED_FALLBACK_IDS,
        current_implementation_hashes,
    )
    from verify.invariant_descriptors import descriptor_catalog

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


def _write_valid_hybrid_config(path: Path) -> Path:
    path.write_text(json.dumps(_valid_synthetic_hybrid_config_payload()), encoding="utf-8")
    return path


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


# --------------------------------------------------------------------------
# argparse: --hybrid / --hybrid-shadow defaults + store_true
# --------------------------------------------------------------------------


def test_build_arg_parser_hybrid_defaults_false():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    )
    assert args.hybrid is False
    assert args.hybrid_shadow is False


def test_build_arg_parser_hybrid_store_true():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        [
            "--receiver-url",
            "http://127.0.0.1:8077",
            "--session",
            "1",
            "--hybrid",
        ]
    )
    assert args.hybrid is True
    assert args.hybrid_shadow is False


def test_build_arg_parser_hybrid_shadow_store_true():
    parser = run_pi_session.build_arg_parser()
    args = parser.parse_args(
        [
            "--receiver-url",
            "http://127.0.0.1:8077",
            "--session",
            "1",
            "--hybrid-shadow",
        ]
    )
    assert args.hybrid_shadow is True
    assert args.hybrid is False


# --------------------------------------------------------------------------
# main(): conflicting scoring-mode flags rejected loudly at startup, before
# any store/camera/workflow side effect
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_flags",
    [
        ["--open-retrieval", "--hybrid"],
        ["--open-retrieval", "--hybrid-shadow"],
        ["--hybrid", "--hybrid-shadow"],
        ["--open-retrieval", "--hybrid", "--hybrid-shadow"],
    ],
)
def test_main_rejects_conflicting_scoring_mode_flags_before_any_side_effect(
    monkeypatch, capsys, extra_flags
):
    """Two-or-more scoring-mode flags must exit non-zero with a clear stderr
    message, and must never reach RemoteProcessingStore/the camera -- i.e.
    the rejection happens strictly before "capture can begin"."""
    touched: list[str] = []
    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: touched.append("store") or object(),
    )

    argv = [
        "--receiver-url",
        "http://127.0.0.1:8077",
        "--session",
        "1",
        *extra_flags,
    ]
    assert run_pi_session.main(argv) == 2
    assert touched == []
    stderr = capsys.readouterr().err
    for flag in extra_flags:
        assert flag in stderr


def _patch_main_dependencies(
    monkeypatch, captured: dict, *, durable_session_mode: str | None = None
):
    """Patch every `main()` side effect below the #269 mode-mismatch guard.

    ``durable_session_mode`` stands in for `sessions.session_mode` as
    returned by the (faked) `resume_session` -- i.e. what
    `tools/run_pi_session.py::main`'s startup guard compares the Pi's own
    resolved flags against (see `SessionIdentity.session_mode`). Left
    ``None``, the fake identity has no `session_mode` attribute at all,
    which the guard's `getattr(session, "session_mode", None)` treats as
    "unknown, skip" -- preserving every pre-existing caller of this helper
    that never set out to exercise the guard. Callers that pass a real mode
    flag AND want to observe unrelated behavior (label text, propagation)
    must pass a matching ``durable_session_mode`` or the new guard rejects
    the mismatch before reaching anything this helper patches below it.
    """

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

    class _Identity:
        def __init__(self, number):
            self.number = number
            if durable_session_mode is not None:
                self.session_mode = durable_session_mode

    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: type(
            "S",
            (),
            {"resume_session": lambda self, n: _Identity(n)},
        )(),
    )
    monkeypatch.setattr(run_pi_session, "HttpCaptureClient", lambda _url: object())
    monkeypatch.setattr(run_pi_session, "PiOutbox", lambda _root: object())
    monkeypatch.setattr(run_pi_session, "FramingCalibrationStore", lambda _path: object())
    monkeypatch.setattr(run_pi_session, "SessionWorkflow", lambda **kwargs: _Workflow())

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
        captured["open_retrieval"] = open_retrieval
        captured["hybrid"] = hybrid
        captured["hybrid_shadow"] = hybrid_shadow

    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "__init__", _capture_init)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "start", lambda self: None)
    monkeypatch.setattr(run_pi_session.PiCaptureRuntime, "close", lambda self: None)
    monkeypatch.setattr(run_pi_session, "_interactive_loop", lambda _w: None)


def test_main_passes_hybrid_from_argv_and_prints_hybrid_label(monkeypatch, capsys, tmp_path):
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode="hybrid")
    config_path = _write_valid_hybrid_config(tmp_path / "handoff.json")

    argv = [
        "--receiver-url",
        "http://127.0.0.1:8077",
        "--session",
        "1",
        "--hybrid",
        "--hybrid-config",
        str(config_path),
    ]
    assert run_pi_session.main(argv) == 0
    assert captured == {
        "open_retrieval": False,
        "hybrid": True,
        "hybrid_shadow": False,
    }
    stdout = capsys.readouterr().out
    assert "Mode: Hybrid" in stdout
    assert "Hybrid Shadow" not in stdout


def test_main_passes_hybrid_shadow_from_argv_and_prints_hybrid_shadow_label(
    monkeypatch, capsys, tmp_path
):
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode="hybrid_shadow")
    config_path = _write_valid_hybrid_config(tmp_path / "handoff.json")

    argv = [
        "--receiver-url",
        "http://127.0.0.1:8077",
        "--session",
        "1",
        "--hybrid-shadow",
        "--hybrid-config",
        str(config_path),
    ]
    assert run_pi_session.main(argv) == 0
    assert captured == {
        "open_retrieval": False,
        "hybrid": False,
        "hybrid_shadow": True,
    }
    stdout = capsys.readouterr().out
    assert "Mode: Hybrid Shadow" in stdout


# --------------------------------------------------------------------------
# #250: the descriptor recipe named in the loaded Hybrid Configuration must
# reach SessionWorkflow, which is what threads it to
# ProcessingStore.freeze_hybrid_pool via poll_drain.
# --------------------------------------------------------------------------


def test_main_threads_session_mode_and_descriptor_recipe_into_session_workflow(
    monkeypatch, capsys, tmp_path
):
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode="hybrid")
    workflow_kwargs: dict[str, object] = {}

    class _Snap:
        phase = "blocks"

    class _Workflow:
        def snapshot(self):
            return _Snap()

        def awaiting_capture_blocks(self):
            return ()

    def _session_workflow_spy(**kwargs):
        workflow_kwargs.update(kwargs)
        return _Workflow()

    monkeypatch.setattr(run_pi_session, "SessionWorkflow", _session_workflow_spy)
    payload = _valid_synthetic_hybrid_config_payload()
    config_path = tmp_path / "handoff.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        "--hybrid", "--hybrid-config", str(config_path),
    ]
    assert run_pi_session.main(argv) == 0

    assert workflow_kwargs["session_mode"] is SessionMode.HYBRID
    expected_names = tuple(entry["name"] for entry in payload["descriptor_recipe"])
    assert workflow_kwargs["hybrid_descriptor_names"] == expected_names
    assert workflow_kwargs["hybrid_candidate_configuration"] == {
        "architecture_kind": payload["architecture"]["kind"],
        "architecture_name": payload["architecture"]["name"],
        "architecture_methods": payload["architecture"]["methods"],
        "candidate_band_thresholds": payload["candidate_band_thresholds"],
    }


def test_main_normal_mode_threads_empty_descriptor_names(monkeypatch, capsys):
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured)
    workflow_kwargs: dict[str, object] = {}

    class _Snap:
        phase = "blocks"

    class _Workflow:
        def snapshot(self):
            return _Snap()

        def awaiting_capture_blocks(self):
            return ()

    def _session_workflow_spy(**kwargs):
        workflow_kwargs.update(kwargs)
        return _Workflow()

    monkeypatch.setattr(run_pi_session, "SessionWorkflow", _session_workflow_spy)

    argv = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    assert run_pi_session.main(argv) == 0

    assert workflow_kwargs["session_mode"] is SessionMode.NORMAL
    assert workflow_kwargs["hybrid_descriptor_names"] == ()
    assert workflow_kwargs["hybrid_candidate_configuration"] is None


# --------------------------------------------------------------------------
# Confirmed HIGH-severity fix (adversarial review, post-#269): the Pi's own
# resolved SessionMode and the durable sessions.session_mode column this
# session was actually started with must never silently disagree -- that gap
# is exactly what let a Hybrid work order fall through to the full N-by-N
# scoring path #269 exists to close. `main()` must hard-refuse (nonzero exit,
# naming both values) before the camera opens or SessionWorkflow is built.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode_flag, resolved_mode_value",
    [("--hybrid", "hybrid"), ("--hybrid-shadow", "hybrid_shadow")],
)
def test_main_refuses_to_start_on_session_mode_mismatch(
    monkeypatch, capsys, tmp_path, mode_flag, resolved_mode_value
):
    """Divergence paths (a)/(b): a `--resume`/`-Mode pi` launch can leave the
    Pi resolving HYBRID or HYBRID_SHADOW from its own flags while the
    session's durable `sessions.session_mode` still says 'normal' (e.g. it
    was started without any mode flag). Starting anyway is the confirmed
    defect: `finish_work_order` reads the durable 'normal' and runs full
    N-by-N scoring. main() must refuse before any camera/workflow side
    effect, exiting nonzero with a message naming BOTH the durable value and
    what the Pi resolved."""
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode="normal")
    config_path = _write_valid_hybrid_config(tmp_path / "handoff.json")

    import picamera2_adapter  # noqa: WPS433 — delayed import target in main()

    camera_opened: list[bool] = []
    monkeypatch.setattr(
        picamera2_adapter, "Picamera2Adapter",
        lambda: camera_opened.append(True) or None,
    )

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        mode_flag, "--hybrid-config", str(config_path),
    ]
    assert run_pi_session.main(argv) == 2
    assert camera_opened == []
    assert captured == {}
    stderr = capsys.readouterr().err
    assert "normal" in stderr
    assert resolved_mode_value in stderr


@pytest.mark.parametrize(
    "mode_flag, durable_mode",
    [("--hybrid", "hybrid"), ("--hybrid-shadow", "hybrid_shadow")],
)
def test_main_starts_normally_when_resolved_mode_matches_durable_mode(
    monkeypatch, capsys, tmp_path, mode_flag, durable_mode
):
    """Non-vacuous positive control for the guard above: matching modes must
    start exactly as before -- proves the new check compares values instead
    of always rejecting."""
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode=durable_mode)
    config_path = _write_valid_hybrid_config(tmp_path / "handoff.json")

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        mode_flag, "--hybrid-config", str(config_path),
    ]
    assert run_pi_session.main(argv) == 0
    assert captured["open_retrieval"] is False


def test_main_normal_mode_startup_unaffected_by_mode_mismatch_guard(monkeypatch, capsys):
    """NORMAL-mode negative control: no mode flags, durable mode 'normal' --
    the new guard must never fire on this, the pre-existing default path, so
    normal-mode startup stays byte-for-byte unchanged."""
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured, durable_session_mode="normal")

    argv = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    assert run_pi_session.main(argv) == 0
    assert captured == {
        "open_retrieval": False,
        "hybrid": False,
        "hybrid_shadow": False,
    }
    stdout = capsys.readouterr().out
    assert "Mode: normal verification" in stdout


# --------------------------------------------------------------------------
# #249: Hybrid Configuration blocks --hybrid/--hybrid-shadow startup loudly,
# before any store/camera side effect; NORMAL/OPEN_RETRIEVAL are unaffected
# --------------------------------------------------------------------------


def test_main_hybrid_blocks_startup_when_config_missing_before_any_side_effect(
    monkeypatch, capsys, tmp_path
):
    touched: list[str] = []
    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: touched.append("store") or object(),
    )
    missing_path = tmp_path / "does_not_exist.json"

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        "--hybrid", "--hybrid-config", str(missing_path),
    ]
    assert run_pi_session.main(argv) == 2
    assert touched == []
    stderr = capsys.readouterr().err
    assert "Hybrid" in stderr
    assert str(missing_path) in stderr


def test_main_hybrid_shadow_blocks_startup_when_config_missing(monkeypatch, capsys, tmp_path):
    touched: list[str] = []
    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: touched.append("store") or object(),
    )
    missing_path = tmp_path / "does_not_exist.json"

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        "--hybrid-shadow", "--hybrid-config", str(missing_path),
    ]
    assert run_pi_session.main(argv) == 2
    assert touched == []


def test_main_hybrid_blocks_startup_on_malformed_json(monkeypatch, capsys, tmp_path):
    touched: list[str] = []
    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: touched.append("store") or object(),
    )
    config_path = tmp_path / "handoff.json"
    config_path.write_text("{not valid json", encoding="utf-8")

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        "--hybrid", "--hybrid-config", str(config_path),
    ]
    assert run_pi_session.main(argv) == 2
    assert touched == []
    assert "not valid JSON" in capsys.readouterr().err


def _run_real_cli(config_path: Path) -> subprocess.CompletedProcess[str]:
    """Drive the actual `tools/run_pi_session.py` CLI in a real subprocess.

    F1 (#249 review): the confirmed live bug was a bare `ValueError`/
    `TypeError` escaping past `run_pi_session.main`'s
    `except HybridConfigurationError` handler, printing a Python traceback
    and exiting 1 instead of a clean message at exit 2. An in-process call
    to `main()` cannot observe that seam -- an uncaught exception there just
    propagates into the test process. Only a real subprocess proves the
    operator-visible behavior: the process's own exit code and whether a
    traceback reached its stderr.
    """
    return subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / "run_pi_session.py"),
            "--receiver-url", "http://127.0.0.1:1",
            "--session", "1",
            "--hybrid",
            "--hybrid-config", str(config_path),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda payload: payload["candidate_band_thresholds"].__setitem__(
                next(iter(payload["candidate_band_thresholds"])), "abc"
            ),
            id="string_threshold",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("known_misses", 5),
            id="known_misses_int",
        ),
        pytest.param(
            lambda payload: payload["architecture"].__setitem__("methods", 5),
            id="architecture_methods_int",
        ),
    ],
)
def test_real_cli_exits_2_with_message_not_traceback_on_malformed_handoff(tmp_path, mutate):
    """F1: every malformed-payload class must exit 2 with a clear stderr
    message, never exit 1 with a Python traceback."""
    payload = _valid_synthetic_hybrid_config_payload()
    mutate(payload)
    config_path = tmp_path / "handoff.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_real_cli(config_path)

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "Traceback" not in result.stderr
    assert "Hybrid configuration invalid" in result.stderr


def test_main_hybrid_blocks_startup_on_wrong_schema_version(monkeypatch, capsys, tmp_path):
    touched: list[str] = []
    monkeypatch.setattr(
        run_pi_session,
        "RemoteProcessingStore",
        lambda _url: touched.append("store") or object(),
    )
    payload = _valid_synthetic_hybrid_config_payload()
    payload["schema_version"] = 999
    config_path = tmp_path / "handoff.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    argv = [
        "--receiver-url", "http://127.0.0.1:8077", "--session", "1",
        "--hybrid", "--hybrid-config", str(config_path),
    ]
    assert run_pi_session.main(argv) == 2
    assert touched == []
    stderr = capsys.readouterr().err
    assert "expected" in stderr and "999" in stderr


@pytest.mark.parametrize("mode_flag", [None, "--open-retrieval"])
def test_main_normal_and_open_retrieval_unaffected_by_missing_hybrid_config(
    monkeypatch, capsys, mode_flag
):
    """A missing/invalid --hybrid-config must never block NORMAL or
    OPEN_RETRIEVAL: `load_hybrid_configuration` must not even be called."""
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured)

    def _fail(_path):
        pytest.fail("load_hybrid_configuration was called outside Hybrid/Hybrid Shadow")

    monkeypatch.setattr(run_pi_session, "load_hybrid_configuration", _fail)

    argv = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    if mode_flag:
        argv.append(mode_flag)
    assert run_pi_session.main(argv) == 0


def test_main_with_no_mode_flag_still_prints_normal_verification(monkeypatch, capsys):
    """Regression: absent Hybrid flags must not disturb the existing
    normal-verification label/plumbing."""
    captured: dict[str, bool] = {}
    _patch_main_dependencies(monkeypatch, captured)

    argv = ["--receiver-url", "http://127.0.0.1:8077", "--session", "1"]
    assert run_pi_session.main(argv) == 0
    assert captured == {
        "open_retrieval": False,
        "hybrid": False,
        "hybrid_shadow": False,
    }
    stdout = capsys.readouterr().out
    assert "Mode: normal verification" in stdout


# --------------------------------------------------------------------------
# PiCaptureRuntime: hybrid/hybrid_shadow set the explicit mode, and boot
# into the existing capture flow (open_retrieval stays False)
# --------------------------------------------------------------------------


def test_pi_capture_runtime_hybrid_flag_resolves_hybrid_mode(tmp_path):
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=_STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:0"),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
    )

    class _FakeCamera:
        def close(self):
            pass

    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        hybrid=True,
    )

    assert runtime.session_mode is SessionMode.HYBRID
    assert runtime.open_retrieval is False


def test_pi_capture_runtime_hybrid_shadow_flag_resolves_hybrid_shadow_mode(tmp_path):
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=_STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:0"),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
    )

    class _FakeCamera:
        def close(self):
            pass

    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
        hybrid_shadow=True,
    )

    assert runtime.session_mode is SessionMode.HYBRID_SHADOW
    assert runtime.open_retrieval is False


def test_pi_capture_runtime_default_mode_is_normal(tmp_path):
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=_STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:0"),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
    )

    class _FakeCamera:
        def close(self):
            pass

    runtime = run_pi_session.PiCaptureRuntime(
        workflow,
        _FakeCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=run_pi_session.SessionConfig(baseline_frames=1),
    )

    assert runtime.session_mode is SessionMode.NORMAL
    assert runtime.open_retrieval is False


def test_pi_capture_runtime_rejects_conflicting_mode_flags(tmp_path):
    from session.session_mode import SessionModeConflictError
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    session = store.start_session(started_at=_STARTED_AT)
    workflow = SessionWorkflow(
        session=session,
        store=store,
        outbox=PiOutbox(tmp_path / "outbox"),
        transport=HttpCaptureClient("http://127.0.0.1:0"),
        framing_calibration=FramingCalibrationStore(
            tmp_path / "pi_local" / "framing_calibration.json"
        ),
    )

    class _FakeCamera:
        def close(self):
            pass

    with pytest.raises(SessionModeConflictError):
        run_pi_session.PiCaptureRuntime(
            workflow,
            _FakeCamera(),
            capture_root=tmp_path / "pi_captures",
            session_config=run_pi_session.SessionConfig(baseline_frames=1),
            open_retrieval=True,
            hybrid=True,
        )


# --------------------------------------------------------------------------
# PowerShell launcher: help text, Parse-RestArgs, Build-RemotePiCommand
# --------------------------------------------------------------------------


def test_launcher_help_accepts_hybrid_flags():
    result = _run_ps1_file("-Mode", "pi", "--session", "1", "--hybrid", "--help")
    assert result.returncode == 0, result.stderr
    assert "--hybrid" in result.stdout
    assert "--hybrid-shadow" in result.stdout


def test_launcher_parse_sets_hybrid_script_var():
    script = _load_launcher_functions_ps() + """
$script:Hybrid = $false
Parse-RestArgs -List @('--hybrid')
if (-not $script:Hybrid) { exit 3 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_parse_sets_hybrid_shadow_script_var():
    script = _load_launcher_functions_ps() + """
$script:HybridShadow = $false
Parse-RestArgs -List @('--hybrid-shadow')
if (-not $script:HybridShadow) { exit 3 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_appends_hybrid_when_set():
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
  -Hybrid $true
if ($cmd -notmatch '--hybrid(?!-shadow)') { exit 4 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_appends_hybrid_shadow_when_set():
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
  -HybridShadow $true
if ($cmd -notmatch '--hybrid-shadow') { exit 4 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_remote_pi_command_omits_hybrid_flags_when_unset():
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
  -Hybrid $false `
  -HybridShadow $false
if ($cmd -match '--hybrid') { exit 5 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--open-retrieval", "--hybrid"],
        ["--open-retrieval", "--hybrid-shadow"],
        ["--hybrid", "--hybrid-shadow"],
    ],
)
def test_launcher_rejects_conflicting_scoring_mode_flags(extra_args):
    result = _run_ps1_file("-Mode", "pi", "--session", "1", *extra_args)
    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--bogus-flag"],
        ["--hybrid-shado"],
        ["--hybri"],
    ],
)
def test_launcher_still_rejects_unknown_args(extra_args):
    result = _run_ps1_file("-Mode", "pi", "--session", "1", *extra_args)
    assert result.returncode == 2


# --------------------------------------------------------------------------
# #269: one flag on the top-level launcher must reach BOTH the receiver
# (Get-ReceiverSessionArgs, consumed by Start-ReceiverWindow) and the Pi
# (Build-RemotePiCommand, already tested above) -- not require typing it
# twice.
# --------------------------------------------------------------------------


def test_get_receiver_session_args_start_session_plain():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession $null
if ($args -ne '--start-session') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_resume_plain():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession 7
if ($args -ne '--resume 7') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_appends_open_retrieval():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession $null -OpenRetrieval $true
if ($args -ne '--start-session --open-retrieval') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_appends_hybrid():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession $null -Hybrid $true
if ($args -ne '--start-session --hybrid') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_appends_hybrid_shadow():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession $null -HybridShadow $true
if ($args -ne '--start-session --hybrid-shadow') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# Confirmed HIGH-severity fix (adversarial review, post-#269): --resume N
# attaches to an already-durable session, so a mode flag alongside it used
# to be appended to the receiver command line and then silently discarded by
# run_receiver.py's --resume branch -- an operator could believe the flag
# did something when it did not. Get-ReceiverSessionArgs must not
# manufacture the flag for --resume at all, for any of the three modes.
# --------------------------------------------------------------------------


def test_get_receiver_session_args_resume_ignores_open_retrieval():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession 7 -OpenRetrieval $true
if ($args -ne '--resume 7') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_resume_ignores_hybrid():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession 7 -Hybrid $true
if ($args -ne '--resume 7') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_receiver_session_args_resume_ignores_hybrid_shadow():
    script = _load_launcher_functions_ps() + """
$args = Get-ReceiverSessionArgs -ResumeSession 7 -HybridShadow $true
if ($args -ne '--resume 7') { exit 6 }
"""
    result = _run_powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# tools/run_receiver.py: --open-retrieval/--hybrid/--hybrid-shadow resolve
# into store.start_session(session_mode=...) (#269's real, traced path from
# the launch flag to the durable sessions.session_mode column).
# --------------------------------------------------------------------------


def test_run_receiver_start_session_with_hybrid_flag_persists_the_mode(tmp_path):
    """#269's real, traced path: run_receiver.py --hybrid resolves the mode
    and threads it into the SAME direct, same-process store.start_session
    call that mints the session -- proven by running the real module's
    main() (not a hand-rolled parallel parser) against a real bound
    receiver, then reading the durable row back with a fresh local store.

    PYTHONUNBUFFERED=1 is required in the child's env: with stdout piped
    (not a tty), Python block-buffers and the "SESSION" line would never
    reach this test's readline() within any timeout -- the exact hazard
    tools/start_live_session.ps1's own receiver-launch comment documents.
    A background reader thread (not a bare readline() loop) is what makes
    the wall-clock deadline below actually enforceable: readline() itself
    has no timeout parameter and would otherwise block past the deadline
    check if the child ever stalled.
    """
    import os
    import queue
    import threading
    import time

    from session.workflow import ProcessingStore

    root = tmp_path / "receiver_root"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [
            sys.executable, "-u", str(_TOOLS_DIR / "run_receiver.py"),
            "--root", str(root), "--host", "127.0.0.1", "--port", "0",
            "--start-session", "--hybrid",
        ],
        cwd=_REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    lines: "queue.Queue[str]" = queue.Queue()

    def _pump():
        for line in iter(proc.stdout.readline, ""):
            lines.put(line)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 15
        seen = ""
        found = False
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            seen += line
            if "SESSION" in line:
                found = True
                break
        if not found:
            pytest.fail(f"receiver never printed SESSION; output so far:\n{seen}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        reader.join(timeout=2)

    # Windows: a just-killed process holding a WAL-mode sqlite3 connection
    # can leave a brief window where re-opening the same file raises a
    # transient "disk I/O error" before the OS fully releases its handle.
    # Retry briefly rather than asserting on file-handle timing.
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            store = ProcessingStore(root)
            assert store._session_mode(1) == "hybrid"
            return
        except Exception as exc:  # pragma: no cover -- Windows handle-release race
            last_exc = exc
            time.sleep(0.3)
    raise last_exc


def test_run_receiver_rejects_conflicting_mode_flags(tmp_path):
    root = tmp_path / "receiver_root"
    result = subprocess.run(
        [
            sys.executable, str(_TOOLS_DIR / "run_receiver.py"),
            "--root", str(root), "--host", "127.0.0.1",
            "--start-session", "--hybrid", "--open-retrieval",
        ],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "one session scoring-mode flag" in result.stdout + result.stderr


def test_run_receiver_help_accepts_hybrid_flags(tmp_path):
    result = subprocess.run(
        [sys.executable, str(_TOOLS_DIR / "run_receiver.py"), "--help"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--hybrid" in result.stdout
    assert "--hybrid-shadow" in result.stdout
    assert "--open-retrieval" in result.stdout
