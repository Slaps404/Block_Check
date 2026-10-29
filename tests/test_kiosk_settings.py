"""Kiosk gear panel: /settings routes over the real capture-settings runtime."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from kiosk.relay import KioskRelay
from kiosk.server import KioskServer
from test_capture_settings import _on_scan_block, _runtime

_INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "code" / "kiosk" / "static" / "index.html"
)


def _request(server, path, payload=None):
    """GET (no payload) or POST JSON; HTTP errors return their status too."""
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        server.url + path, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310 (localhost only)
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def served(tmp_path):
    """A real PiCaptureRuntime on Scan Block behind the real kiosk server."""
    runtime = _runtime(tmp_path)
    _on_scan_block(runtime)
    server = KioskServer(KioskRelay(runtime), host="127.0.0.1", port=0)
    server.start()
    try:
        yield runtime, server
    finally:
        server.stop()


def test_get_settings_returns_rows_with_editability(served):
    runtime, server = served

    status, body = _request(server, "/settings")

    assert status == 200
    assert body == json.loads(json.dumps(runtime.settings_status()))
    assert body["active_role"] == "block"
    assert all(row["editable"] for row in body["settings"])


def test_get_settings_reports_locked_camera_rows_off_scan_block(served):
    runtime, server = served
    runtime.scan_block("12345678")

    rows = {row["name"]: row for row in _request(server, "/settings")[1]["settings"]}

    assert rows["block.exposure_us"]["editable"] is False
    assert "Scan Block" in rows["block.exposure_us"]["locked_reason"]


def test_get_settings_is_404_on_a_handle_without_settings():
    server = KioskServer(KioskRelay(object()), host="127.0.0.1", port=0)
    server.start()
    try:
        status, body = _request(server, "/settings")
    finally:
        server.stop()

    assert status == 404
    assert "unavailable" in body["error"]


def test_post_settings_applies_and_returns_the_runtime_result(served):
    runtime, server = served

    status, body = _request(
        server, "/settings", {"name": "block.exposure_us", "value": 20000}
    )

    assert status == 200 and body["ok"] is True
    result = body["result"]
    assert result["setting"]["value"] == 20000
    assert result["setting"]["changed"] is True
    assert result["recalibrated"] is True
    assert "recalibrating" in result["message"]
    assert runtime.capture_settings.value("block.exposure_us") == 20000


def test_post_settings_range_error_surfaces_the_real_message(served):
    runtime, server = served

    status, body = _request(
        server, "/settings", {"name": "block.exposure_us", "value": 5}
    )

    assert status == 400
    assert body == {
        "ok": False, "error": "block.exposure_us must be between 100 and 200000",
    }
    assert runtime.capture_settings.overrides() == {}


def test_post_settings_state_refusal_surfaces_the_real_message(served):
    runtime, server = served
    runtime.scan_block("12345678")

    status, body = _request(
        server, "/settings", {"name": "block.exposure_us", "value": 20000}
    )

    assert status == 200
    assert body["ok"] is False
    assert body["error"] == (
        "camera settings can only change on Scan Block with the table clear"
    )
    assert runtime.capture_settings.overrides() == {}


def test_command_set_range_error_is_not_a_generic_verb_error(served):
    _, server = served

    status, body = _request(
        server, "/command", {"name": "set", "args": ["block.exposure_us", "5"]}
    )

    assert status == 400
    assert body["error"] == "block.exposure_us must be between 100 and 200000"


def test_post_settings_rejects_a_malformed_body(served):
    _, server = served

    status, body = _request(server, "/settings", {"value": 5})

    assert status == 400
    assert body["ok"] is False


def test_post_settings_reset_restores_defaults(served):
    runtime, server = served
    runtime.set_setting("detection.motion_threshold", 0.05)

    status, body = _request(server, "/settings/reset", {})

    assert status == 200 and body["ok"] is True
    assert body["result"]["reset"] == ["detection.motion_threshold"]
    assert runtime.capture_settings.overrides() == {}


# Client-side contracts pinned textually (no browser in CI).
def test_gear_and_settings_panel_markup_exist():
    html = _INDEX_HTML.read_text(encoding="utf-8")

    for snippet in (
        'id="settings-gear"', 'id="settings-panel"', 'id="settings-rows"',
        'id="settings-save"', 'id="settings-reset"', 'id="settings-close"',
        'fetch("/settings"', '"/settings/reset"', "Recalibrating…",
        "row.locked_reason", "RESET TO DEFAULTS",
    ):
        assert snippet in html, snippet


def test_gear_lives_on_the_panel_outside_every_screen_container():
    """The gear is a direct child of #panel, so no screen layout hides it."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    panel_open = html.index('<div class="panel" id="panel">')
    content_open = html.index('<div class="content" id="content">')
    gear = html.index('id="settings-gear"')

    assert panel_open < gear < content_open
    assert '$("settings-gear").style.display' not in html


def test_gear_shows_on_every_catalog_screen_and_opens_the_panel():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for client behavior test")
    html = _INDEX_HTML.read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    script = script[:script.index("  // ---- boot: fetch catalog once")]
    harness = r'''
const assert = require("node:assert/strict");
class Element {
  constructor(id) {
    this.id = id; this.style = {}; this.dataset = {}; this.children = [];
    this.listeners = {}; this.clientWidth = 1200; this.clientHeight = 700;
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
  }
  set innerHTML(value) { this.children = []; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  removeAttribute() {}
  setAttribute() {}
  querySelectorAll() { return []; }
}
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  },
  createElement: tag => new Element(tag),
  querySelector: selector => new Element(selector),
  addEventListener() {},
};
const window = { location: { origin: "http://localhost" }, addEventListener() {} };
const setInterval = () => 1;
const clearInterval = () => {};
const fetched = [];
const fetch = async url => {
  fetched.push(url);
  return { ok: true, json: async () => ({ active_role: "block", settings: [] }) };
};
'''
    checks = r'''
(async () => {
  command = async () => {};
  for (const id of Object.keys(FALLBACK_CATALOG.screens)) {
    render({ screen: id, phase: "blocks", capture_mode: "block",
             capture_state: "WAITING_FOR_SCAN", effective_engaged: true });
    assert.notEqual($("settings-gear").style.display, "none", id);
  }
  await $("settings-gear").listeners.click();
  assert.equal(settingsOpen, true);
  assert.ok(fetched.includes("/settings"));
  $("settings-close").listeners.click();
  assert.equal(settingsOpen, false);
})().catch(err => { console.error(err); process.exit(1); });
'''
    result = subprocess.run(
        [node, "-"], input=harness + script + checks,
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
