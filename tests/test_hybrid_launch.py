"""Hybrid-specific launch behavior.

Generic flag plumbing (argparse, launcher parse/forward, mode labels) for
--hybrid/--hybrid-shadow lives in test_launch_flags.py. This file covers what
only the scoring modes do: conflict rejection, the Hybrid Configuration
startup gate, the durable-mode guard, and the receiver's persisted mode.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import pytest

from launcher_harness import (
    BASE_ARGV,
    REPO_ROOT,
    TOOLS_DIR,
    patch_main_dependencies,
    valid_synthetic_hybrid_config_payload,
    write_hybrid_config,
)

from app import pi_session  # noqa: E402
from session.session_mode import SessionMode  # noqa: E402

_STARTED_AT = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


def _forbid_store(monkeypatch) -> list[str]:
    """Record any RemoteProcessingStore construction (the first side effect)."""
    touched: list[str] = []
    monkeypatch.setattr(
        pi_session, "RemoteProcessingStore", lambda _url: touched.append("store") or object()
    )
    return touched


# --------------------------------------------------------------------------
# main(): conflicting scoring-mode flags rejected before any side effect
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
    touched = _forbid_store(monkeypatch)

    assert pi_session.main([*BASE_ARGV, *extra_flags]) == 2
    assert touched == []
    stderr = capsys.readouterr().err
    for flag in extra_flags:
        assert flag in stderr


# --------------------------------------------------------------------------
# The loaded descriptor recipe must reach SessionWorkflow, which threads
# it to ProcessingStore.freeze_hybrid_pool via poll_drain.
# --------------------------------------------------------------------------


def test_main_threads_session_mode_and_descriptor_recipe_into_session_workflow(
    monkeypatch, tmp_path
):
    captured = patch_main_dependencies(monkeypatch, durable_session_mode="hybrid")
    payload = valid_synthetic_hybrid_config_payload()
    config_path = write_hybrid_config(tmp_path / "handoff.json", payload)

    assert pi_session.main([*BASE_ARGV, "--hybrid", "--hybrid-config", str(config_path)]) == 0

    workflow_kwargs = captured["workflow"]
    assert workflow_kwargs["session_mode"] is SessionMode.HYBRID
    assert workflow_kwargs["hybrid_descriptor_names"] == tuple(
        entry["name"] for entry in payload["descriptor_recipe"]
    )
    assert workflow_kwargs["hybrid_candidate_configuration"] == {
        "architecture_kind": payload["architecture"]["kind"],
        "architecture_name": payload["architecture"]["name"],
        "architecture_methods": payload["architecture"]["methods"],
        "candidate_band_thresholds": payload["candidate_band_thresholds"],
    }


def test_main_normal_mode_threads_empty_descriptor_names(monkeypatch):
    captured = patch_main_dependencies(monkeypatch)

    assert pi_session.main(list(BASE_ARGV)) == 0

    workflow_kwargs = captured["workflow"]
    assert workflow_kwargs["session_mode"] is SessionMode.NORMAL
    assert workflow_kwargs["hybrid_descriptor_names"] == ()
    assert workflow_kwargs["hybrid_candidate_configuration"] is None


# --------------------------------------------------------------------------
# The Pi's resolved mode and the durable sessions.session_mode
# must never silently disagree. main() hard-refuses before the camera opens.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode_flag, resolved_mode_value",
    [("--hybrid", "hybrid"), ("--hybrid-shadow", "hybrid_shadow")],
)
def test_main_refuses_to_start_on_session_mode_mismatch(
    monkeypatch, capsys, tmp_path, mode_flag, resolved_mode_value
):
    """A `--resume`/`-Mode pi` launch can resolve HYBRID while the durable
    column says 'normal'; finish_work_order would then run full N-by-N scoring.
    The refusal must name both values."""
    captured = patch_main_dependencies(monkeypatch, durable_session_mode="normal")
    config_path = write_hybrid_config(tmp_path / "handoff.json")

    import picamera2_adapter  # noqa: WPS433 -- delayed import target in main()

    camera_opened: list[bool] = []
    monkeypatch.setattr(
        picamera2_adapter, "Picamera2Adapter", lambda: camera_opened.append(True) or None
    )

    argv = [*BASE_ARGV, mode_flag, "--hybrid-config", str(config_path)]
    assert pi_session.main(argv) == 2
    assert camera_opened == []
    assert captured == {}
    stderr = capsys.readouterr().err
    assert "normal" in stderr
    assert resolved_mode_value in stderr


@pytest.mark.parametrize(
    "mode_flag, durable_mode",
    [(None, "normal"), ("--hybrid", "hybrid"), ("--hybrid-shadow", "hybrid_shadow")],
)
def test_main_starts_normally_when_resolved_mode_matches_durable_mode(
    monkeypatch, tmp_path, mode_flag, durable_mode
):
    """Positive control for the guard: matching modes start, so the check
    compares values instead of always rejecting."""
    captured = patch_main_dependencies(monkeypatch, durable_session_mode=durable_mode)
    argv = list(BASE_ARGV)
    if mode_flag:
        argv += [mode_flag, "--hybrid-config", str(write_hybrid_config(tmp_path / "h.json"))]

    assert pi_session.main(argv) == 0
    assert captured["runtime"]["open_retrieval"] is False


# --------------------------------------------------------------------------
# A bad Hybrid Configuration blocks --hybrid/--hybrid-shadow startup
# before any side effect; NORMAL/OPEN_RETRIEVAL never load it.
# --------------------------------------------------------------------------


def _wrong_schema_payload() -> dict:
    payload = valid_synthetic_hybrid_config_payload()
    payload["schema_version"] = 999
    return payload


def _mutated_payload(key: str, value, *, nested: bool = False) -> dict:
    payload = valid_synthetic_hybrid_config_payload()
    if nested:
        payload[key].update(value)
    else:
        payload[key] = value
    return payload


@pytest.mark.parametrize(
    "mode_flag, write, expected_stderr",
    [
        pytest.param("--hybrid", None, ["Hybrid", "{path}"], id="hybrid-missing"),
        pytest.param("--hybrid-shadow", None, ["Hybrid", "{path}"], id="shadow-missing"),
        pytest.param(
            "--hybrid",
            lambda p: p.write_text("{not valid json", encoding="utf-8"),
            ["not valid JSON"],
            id="malformed-json",
        ),
        pytest.param(
            "--hybrid",
            lambda p: write_hybrid_config(p, _wrong_schema_payload()),
            ["expected", "999"],
            id="wrong-schema-version",
        ),
        pytest.param(
            "--hybrid",
            lambda p: write_hybrid_config(p, _mutated_payload("known_misses", 5)),
            ["Hybrid configuration invalid"],
            id="known_misses_int",
        ),
        pytest.param(
            "--hybrid",
            lambda p: write_hybrid_config(
                p, _mutated_payload("architecture", {"methods": 5}, nested=True)
            ),
            ["Hybrid configuration invalid"],
            id="architecture_methods_int",
        ),
    ],
)
def test_main_hybrid_blocks_startup_on_bad_config_before_any_side_effect(
    monkeypatch, capsys, tmp_path, mode_flag, write, expected_stderr
):
    touched = _forbid_store(monkeypatch)
    config_path = tmp_path / "handoff.json"
    if write is not None:
        write(config_path)

    assert pi_session.main([*BASE_ARGV, mode_flag, "--hybrid-config", str(config_path)]) == 2
    assert touched == []
    stderr = capsys.readouterr().err
    for text in expected_stderr:
        assert text.format(path=config_path) in stderr


def test_real_cli_exits_2_with_message_not_traceback_on_malformed_handoff(tmp_path):
    """A bare ValueError/TypeError must not escape main()'s
    HybridConfigurationError handler as a traceback at exit 1. Only a
    real subprocess sees the process exit code and stderr traceback; the other
    malformed shapes run in-process above."""
    payload = valid_synthetic_hybrid_config_payload()
    thresholds = payload["candidate_band_thresholds"]
    thresholds[next(iter(thresholds))] = "abc"
    config_path = write_hybrid_config(tmp_path / "handoff.json", payload)

    result = subprocess.run(
        [
            sys.executable, str(TOOLS_DIR / "run_pi_session.py"),
            "--receiver-url", "http://127.0.0.1:1", "--session", "1",
            "--hybrid", "--hybrid-config", str(config_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "Traceback" not in result.stderr
    assert "Hybrid configuration invalid" in result.stderr


@pytest.mark.parametrize("mode_flag", [None, "--open-retrieval"])
def test_main_normal_and_open_retrieval_unaffected_by_missing_hybrid_config(
    monkeypatch, mode_flag
):
    patch_main_dependencies(monkeypatch)

    def _fail(_path):
        pytest.fail("load_hybrid_configuration was called outside Hybrid/Hybrid Shadow")

    monkeypatch.setattr(pi_session, "load_hybrid_configuration", _fail)

    argv = list(BASE_ARGV)
    if mode_flag:
        argv.append(mode_flag)
    assert pi_session.main(argv) == 0


# --------------------------------------------------------------------------
# PiCaptureRuntime: each mode flag resolves its explicit SessionMode and boots
# the existing capture flow (open_retrieval stays False)
# --------------------------------------------------------------------------


def _build_runtime(tmp_path, **mode_kwargs):
    from session.workflow import (
        FramingCalibrationStore,
        HttpCaptureClient,
        PiOutbox,
        ProcessingStore,
        SessionWorkflow,
    )

    store = ProcessingStore(tmp_path / "processing")
    workflow = SessionWorkflow(
        session=store.start_session(started_at=_STARTED_AT),
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

    return pi_session.PiCaptureRuntime(
        workflow,
        _FakeCamera(),
        capture_root=tmp_path / "pi_captures",
        session_config=pi_session.SessionConfig(baseline_frames=1),
        **mode_kwargs,
    )


@pytest.mark.parametrize(
    "mode_kwargs, expected_mode",
    [
        ({}, SessionMode.NORMAL),
        ({"hybrid": True}, SessionMode.HYBRID),
        ({"hybrid_shadow": True}, SessionMode.HYBRID_SHADOW),
    ],
)
def test_pi_capture_runtime_resolves_session_mode(tmp_path, mode_kwargs, expected_mode):
    runtime = _build_runtime(tmp_path, **mode_kwargs)

    assert runtime.session_mode is expected_mode
    assert runtime.open_retrieval is False


# --------------------------------------------------------------------------
# tools/run_receiver.py: mode flags resolve into store.start_session(
# session_mode=...), the traced path to the durable column.
# --------------------------------------------------------------------------


def test_run_receiver_start_session_with_hybrid_flag_persists_the_mode(tmp_path):
    """Runs the real run_receiver.py, then reads the durable row back.

    PYTHONUNBUFFERED=1: with piped stdout Python block-buffers, so "SESSION"
    would never reach readline(). The reader thread makes the deadline
    enforceable, since readline() has no timeout.
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
            sys.executable, "-u", str(TOOLS_DIR / "run_receiver.py"),
            "--root", str(root), "--host", "127.0.0.1", "--port", "0",
            "--start-session", "--hybrid",
        ],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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

    # Windows can briefly refuse to reopen a just-killed process's WAL sqlite
    # file ("disk I/O error"); retry rather than assert on handle timing.
    last_exc: Exception | None = None
    for _attempt in range(5):
        try:
            assert ProcessingStore(root)._session_mode(1) == "hybrid"
            return
        except Exception as exc:  # pragma: no cover -- Windows handle-release race
            last_exc = exc
            time.sleep(0.3)
    raise last_exc


def test_run_receiver_rejects_conflicting_mode_flags(monkeypatch, capsys, tmp_path):
    from app import receiver

    touched: list[str] = []
    monkeypatch.setattr(receiver, "ProcessingStore", lambda *a, **k: touched.append("store"))
    monkeypatch.setattr(sys, "argv", [
        "run_receiver.py", "--root", str(tmp_path / "receiver_root"),
        "--host", "127.0.0.1", "--start-session", "--hybrid", "--open-retrieval",
    ])

    assert receiver.main() == 2
    assert touched == []
    assert "one session scoring-mode flag" in capsys.readouterr().err
