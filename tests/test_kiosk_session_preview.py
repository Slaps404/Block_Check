"""Exercise temporary preview navigation using the real client JavaScript."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_preview_returns_to_latest_state_without_changing_session_flags():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for client behavior test")
    html = (Path(__file__).resolve().parents[1]
            / "code/kiosk/static/index.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    # Suppress HTTP boot; drive real rendering and click handlers below.
    script = script[:script.index("  // ---- boot: fetch catalog once")]
    harness = r'''
const assert = require("node:assert/strict");
class Element {
  constructor(id) {
    this.id = id; this.style = {}; this.dataset = {}; this.children = [];
    this.listeners = {}; this.clientWidth = 1200; this.clientHeight = 700;
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  set innerHTML(value) { this.children = []; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  removeAttribute() {}
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
'''
    checks = r'''
let commands = [];
command = async name => { commands.push(name); };
ui.engaged = true;
const flags = JSON.stringify(ui);
const state = {
  screen: "06", phase: "blocks", capture_mode: "block",
  capture_state: "WAITING_FOR_SCAN", effective_engaged: true,
};
render(state);
assert.equal($("preview-toggle").style.display, "");
$("preview-toggle").listeners.click();
assert.equal($("content").style.display, "none");
assert.equal($("idle-preview").style.display, "");
assert.equal($("idle-top").children[0].textContent, "BACK TO SESSION");
assert.equal(bootPreviewActive, true);
assert.match($("idle-image").src, /^\/preview-frame\?/);
assert.equal($("preview-toggle").style.display, "none");
// Workflow advances during preview. Returning must use this new snapshot.
render({ ...state, screen: "12", phase: "slides", capture_mode: "slide",
         capture_state: "EMPTY" });
assert.equal($("idle-top").children[0].textContent, "BACK TO SESSION");
$("idle-top").children[0].listeners.click();
assert.equal($("headline").textContent, "Place Slide");
assert.equal($("content").style.display, "");
assert.equal(bootPreviewActive, false);
assert.equal(JSON.stringify(ui), flags);
assert.deepEqual(commands, []);
render({ online: false });
assert.equal($("preview-toggle").disabled, true);
render(state);
assert.equal($("preview-toggle").disabled, false);
// Completion closes the temporary view; boot keeps its original start button.
setSessionPreview(true);
render({ ...state, screen: "21", phase: "finalized" });
assert.equal(sessionPreviewOpen, false);
assert.equal($("preview-toggle").style.display, "none");
render({ screen: "01", effective_engaged: false, phase: "blocks" });
assert.equal($("idle-top").children[0].textContent, "START SESSION");
assert.equal($("preview-toggle").style.display, "none");
'''
    result = subprocess.run(
        [node, "-"], input=harness + script + checks,
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
