"""Pi-local HTTP shell that serves the kiosk to Chromium ``--kiosk`` (#121).

Stdlib only -- a ``ThreadingHTTPServer`` bound to ``localhost`` so a browser on
the same Pi can render the UI; no framework, no new dependency to deploy. Four
routes map onto the two relay seams plus the static screen catalog:

* ``GET  /``        -> the one extracted screen (static HTML, deck runtime stripped)
* ``GET  /state``   -> ``relay.state()`` as JSON (read path; the page polls this)
* ``GET  /catalog`` -> the static :data:`kiosk.screens.CATALOG` (state-free; cache at boot)
* ``POST /command`` -> ``relay.command(name, *args)`` (write path; one tap)
* ``GET  /inspection`` -> ``relay.inspection_descriptors(capture_id)`` as JSON
  (#153: ordered contact-sheet descriptors for one results-ready slide)

The server owns no session logic; it is a thin transport over :class:`KioskRelay`.
Runs on its own daemon thread so it coexists beside the text console's operator
loop without taking over the process.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast
from urllib.parse import parse_qs, urlparse

from kiosk.screens import CATALOG
from store.remote import TransportError

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Minimal grey JPEG returned when no live preview frame exists yet.
_PREVIEW_PLACEHOLDER_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7),01444\x1f\x27=9=82<.7\xff\xc0\x00"
    b"\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01"
    b"\x05\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03"
    b"\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03"
    b"\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05"
    b"\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0"
    b"$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghij"
    b"stuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
    b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
    b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6"
    b"\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3"
    b"\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb"
    b"\xd5\xff\xd9"
)

_UI_FLAGS = (
    "engaged",
    "finish_blocks_guard",
    "finish_slides_guard",
    "view_results_guard",
    "recapture_guard",
)


def _ui_flags(query: str) -> dict[str, bool]:
    """Parse the client-owned routing flags from a /state query string.
    A flag is true iff present as 1/true/yes."""
    params = parse_qs(query)
    truthy = {"1", "true", "yes"}
    return {
        name: (params.get(name, ["0"])[0].lower() in truthy) for name in _UI_FLAGS
    }


class _KioskHTTPServer(ThreadingHTTPServer):
    """Carries the relay + static dir to every request handler (typed)."""

    def __init__(self, server_address, handler, *, relay, static_dir: Path):
        super().__init__(server_address, handler)
        self.relay = relay
        self.static_dir = static_dir


class _Handler(BaseHTTPRequestHandler):
    @property
    def kiosk(self) -> _KioskHTTPServer:
        return cast(_KioskHTTPServer, self.server)

    def log_message(self, *args) -> None:  # keep test/operator output clean
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file(self.kiosk.static_dir / "index.html", "text/html")
        elif parsed.path == "/state":
            self._send_json(200, self.kiosk.relay.state(_ui_flags(parsed.query)))
        elif parsed.path == "/catalog":
            # Static, state-independent screen descriptions; cacheable at boot.
            self._send_json(200, CATALOG)
        elif parsed.path == "/review-still":
            self._send_review_still()
        elif parsed.path == "/preview-frame":
            self._send_preview_frame()
        elif parsed.path == "/inspection-sheet":
            self._send_inspection_sheet(parse_qs(parsed.query))
        elif parsed.path == "/results-evidence":
            self._send_results_evidence(parse_qs(parsed.query))
        elif parsed.path == "/inspection":
            self._send_inspection_descriptors(parse_qs(parsed.query))
        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/command":
            self._send_json(404, {"error": f"not found: {self.path}"})
            return
        payload = self._read_json_body()
        name = payload.get("name")
        args = payload.get("args") or []
        if not isinstance(name, str) or not isinstance(args, list):
            self._send_json(400, {"ok": False, "error": "expected {name, args[]}"})
            return
        try:
            result = self.kiosk.relay.command(name, *args)
        except ValueError as exc:  # unknown verb / bad arity -- deterministic reject
            self._send_json(400, {"ok": False, "error": str(exc)})
        except TransportError as exc:  # link down: not fatal, surfaced to the UI
            self._send_json(200, {"ok": False, "offline": True, "error": str(exc)})
        except RuntimeError as exc:
            # A verb raced against its own state (e.g. a double-tapped CALIBRATE
            # NOW whose confirm_empty landed after the phase already advanced)
            # raises RuntimeError. It is a benign no-op at the seam, not a
            # server fault -- surface {ok: False}, never a 500. (Ordered AFTER
            # TransportError, which subclasses RuntimeError.)
            self._send_json(200, {"ok": False, "error": str(exc)})
        else:
            self._send_json(200, {"ok": True, "result": str(result)})

    # -- helpers -----------------------------------------------------------
    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return {}

    def _send_json(self, status: int, body: object) -> None:
        self._send_bytes(status, "application/json", json.dumps(body).encode())

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._send_json(404, {"error": f"missing asset: {path.name}"})
            return
        self._send_bytes(200, content_type, data)

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Kiosk is single-client localhost; never let a stale screen cache.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_review_still(self) -> None:
        jpeg = self.kiosk.relay.review_still_jpeg()
        if jpeg is None:
            self._send_json(404, {"error": "no pending capture"})
            return
        self._send_bytes(200, "image/jpeg", jpeg)

    def _send_preview_frame(self) -> None:
        jpeg = self.kiosk.relay.latest_preview_jpeg()
        if jpeg is None:
            self._send_bytes(503, "image/jpeg", _PREVIEW_PLACEHOLDER_JPEG)
            return
        self._send_bytes(200, "image/jpeg", jpeg)

    def _send_inspection_descriptors(self, params: dict[str, list[str]]) -> None:
        """#153: ``GET /inspection?capture_id=<id>`` -- the ordered contact-
        sheet descriptors (``kiosk.inspection.project_inspection``) for one
        results-ready slide, so the client drives ``/inspection-sheet?path=``
        fetches off returned paths rather than reconstructing them itself."""
        capture_id = (params.get("capture_id") or [None])[0]
        descriptors = (
            self.kiosk.relay.inspection_descriptors(capture_id) if capture_id else None
        )
        if descriptors is None:
            self._send_json(404, {"error": "no inspection data"})
            return
        self._send_json(200, descriptors)

    def _send_inspection_sheet(self, params: dict[str, list[str]]) -> None:
        path = (params.get("path") or [None])[0]
        png = self.kiosk.relay.inspection_sheet_bytes(path) if path else None
        if png is None:
            self._send_json(404, {"error": "no inspection sheet"})
            return
        self._send_bytes(200, "image/png", png)

    def _send_results_evidence(self, params: dict[str, list[str]]) -> None:
        path = (params.get("path") or [None])[0]
        jpeg = self.kiosk.relay.results_evidence_bytes(path) if path else None
        if jpeg is None:
            self._send_json(404, {"error": "no results evidence"})
            return
        self._send_bytes(200, "image/jpeg", jpeg)


class KioskServer:
    """Owns the localhost HTTP server and its daemon thread."""

    def __init__(
        self,
        relay,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        static_dir: str | Path | None = None,
    ):
        static = Path(static_dir) if static_dir else _STATIC_DIR
        self._httpd = _KioskHTTPServer(
            (host, port), _Handler, relay=relay, static_dir=static
        )
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode()
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._httpd.serve_forever, name="kiosk-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._httpd.server_close()
