# SessionWorkflow runs on the Pi; the processing-computer store is a remote RPC proxy

**Status:** accepted (2026-07-06)

Issue #101 asks an operator to accept a *deployed two-machine* system, but the
integrated `SessionWorkflow` has only ever run in a single process (the PRD
acceptance seam). The two machines must be split for real. We decided the
`SessionWorkflow` runs on the **Raspberry Pi**, and the processing computer's
`ProcessingStore` is reached as a **remote proxy over the dedicated Ethernet
link**, because the workflow is camera-coupled (it constructs `CaptureSession`,
per-phase empty-backlight baselines, and stillness detection in its constructor,
`session_workflow.py:1704`) and the operator physically stands at the capture
station. The camera is the anchor: whatever runs the workflow must be local to
the camera, and the camera is bolted to the Pi.

The split crosses the wire on the **narrow** dependency. The store is a
request/response, latency-tolerant surface (~20 methods, each "call → get a
dataclass back"); the camera is chatty, streaming, and latency-critical (live
preview, sub-second stillness detection, per-phase baselines) and its code is
already implemented and tested on Pi hardware. Remoting the store is cheap and
honors the PRD split ("the Pi owns capture, live preview frames, per-phase
baselines," line 87); remoting the camera would be expensive and contradict it.

## Command transport

- **One wire surface: the store proxy.** Operator commands never cross the
  network. The command adapter (`session_console.dispatch`) runs on the Pi in
  the same process as the workflow; during bring-up the operator SSHes into the
  Pi to drive it. There is no remote command mode and no main-computer UI. The
  future touchscreen (out of scope, PRD line 139) is also Pi-local, so this
  needs no revisiting when the touchscreen lands.
- **Generic `/rpc` envelope for the store, not ~20 REST routes.** The
  workflow→store chatter goes through a single `POST /sessions/{n}/rpc` carrying
  `{method, args}`. The surface is large and grows as the workflow evolves;
  ~20 hand-written route/client pairs would be boilerplate to keep in sync. The
  "arbitrary method" risk is muted on this isolated, no-auth, dedicated-Ethernet
  link (PRD lines 90, 148).
- **Whitelisted method registry, never `getattr` dispatch.** The `/rpc` handler
  dispatches through an explicit `{name: callable}` registry (mirroring
  `session_console._COMMANDS`), so the Pi cannot invoke arbitrary store
  attributes even if the link is ever less isolated than assumed.
- **Binary captures stay on explicit routes.** `GET /sessions/{n}/status` and
  block `POST /sessions/{n}/captures` retain their bring-up/debug contracts.
  Slide PNG bytes cross `POST /sessions/{n}/slides` with framed identity/decode
  metadata and a stable request ID. The processing computer first writes a
  processing-local staging file, then records the durable slide capture; a
  Pi-local filename is never treated as a processing-computer path. These
  routes remain separate from JSON `/rpc` so large PNGs are not base64 encoded.

## Mid-command disconnect (idempotency)

A store command can execute on the processing computer and have its response
lost to a cable pull; the Pi's RPC client then retries. Domain-duplicate
rejection is **not** idempotency: a retried `scan_block` would return
"Block already scanned" and a retried `resolve_claim` would return "Slide
already processed" instead of the original success — the latter would deny the
operator the `PASS`/`REVIEW` verdict they are waiting on.

We **generalize the existing `receipts` idempotency ledger** (`receive_capture`,
`session_workflow.py:890`: `capture_id`-keyed `receipts` table, `BEGIN
IMMEDIATE` serialization, original-response replay) to the mutating RPC surface.
The Pi attaches a per-request idempotency key; the store records
`request_id → original response` and replays it on a repeat. Scope:

- **Read-only commands** — nothing; retry is naturally safe.
- **Non-idempotent trio** (`scan_block`, `record_slide_capture`,
  `resolve_claim`) — must replay the original response.
- **`record_event`** — minor double-append on retry; the key dedupes it.

The key also disambiguates a lost-ack retry (same key → replay) from a genuine
operator double-scan (different key, same block → reject) — which "treat a
duplicate as success" cannot do.

**Watch later (noted, accepted for bring-up):**
- The **client must mint one stable request-id per logical command and reuse it
  across retries** — a fresh id per attempt defeats the ledger silently. This is
  the load-bearing client rule, mirroring how `capture_id` is stable per capture.
- The ledger grows per session and stores full response payloads; if it becomes
  a size/serialization problem, revisit (bound it, or store only the trio).

## Considered options

- **Option 2 — workflow on the processing computer, thin Pi capture agent.**
  Rejected: it remotes the camera (streaming preview + baselines + stillness over
  HTTP), requires rewriting working, hardware-tested Pi capture code, and
  contradicts the PRD's "Pi owns operator actions / capture / preview" split.
- **UI on the main computer.** Rejected as a driver of placement. Keeping the
  workflow on the Pi but moving only the UI adds a *second* wire surface
  (render/command, main→Pi) for no benefit; moving the workflow to follow the UI
  is just Option 2. The UI follows the operator, the operator follows the camera,
  the camera is on the Pi.
- **Explicit REST route per store method.** Rejected: ~20 route/client pairs to
  maintain and extend, when the only endpoints we hand-inspect (`/status`,
  `/captures`) are kept explicit anyway.

## Boundary rules (resolved — "proxy the store" does NOT cover these)

1. **Framing calibration is Pi-local; `store.root` never crosses the boundary.**
   The workflow builds `self.store.root / "framing_calibration.json"`
   (`session_workflow.py:1722`), a path on the *main computer* that is
   meaningless on the Pi. A grep confirms `FramingCalibrationStore` is read
   **only by the workflow** (constructor + `view`/`approve`/`recalibrate`,
   `session_workflow.py:1872-1904`) — nothing on the processing/store side
   (`receive_capture`, `prepare_finalization`, preprocessing) consumes it. It is
   pure camera state. **Decision:** the Pi entry point injects an explicit
   Pi-local `FramingCalibrationStore` (the ctor already accepts one), and the
   default stops reaching for `self.store.root`. `root` is never proxied.
2. **`_record_finalization_error` is promoted to a public store method.** The
   store already uses it pervasively *internally* (`begin_finalization:731`,
   `prepare_finalization:785/793/809`, `complete_finalization:838`). Its one
   external caller (`poll_finalization:1975`) wraps a `try` around **both**
   `self.outbox.delete_acknowledged()` (a Pi-side op) and
   `store.complete_finalization()`, so the failure can originate on the Pi where
   the store cannot observe it — the Pi legitimately needs to say "record this
   finalization error." **Decision:** promote to public `record_finalization_error`
   and add it to the `/rpc` whitelist. It was only private because callers were
   in-process; recording a finalization error is a real public operation.

## Consequences

- A future reader must not "simplify" this to a single-process deployment — that
  is the test seam, not the product. #101 acceptance requires the real split.
- The boundary needs its own tests (per #101 / Handoff A DoD): command
  serialization, an induced mid-command disconnect, and finalization over the
  wire — while the in-process seam stays green.
- No changes to segmentation, scoring, gates, or `PASS_THRESHOLD`. This is
  integration wiring only.
