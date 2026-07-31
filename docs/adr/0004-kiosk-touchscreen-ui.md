# Kiosk touchscreen UI is a Pi-local second renderer of the workflow, served as HTML to Chromium

**Status:** accepted (2026-07-07)

The `LJI Block Check Kiosk` wireframe (23 screens) is the operator-facing
touchscreen UI reserved by PRD line 139 and named "future / out of scope" in
`PROJECT_CONTEXT.md`. We decided to build it **now**, as a *second renderer*
alongside the existing text console rather than a rewrite of any workflow logic.
The seam already exists: `session_console.py`'s own docstring says "a future
touchscreen replaces this module without moving any workflow logic," and it
exposes exactly three verbs — `render_summary`, `render_events`, `dispatch`.
The kiosk consumes the same `SessionSummary` / `WorkflowEvent` /
`WorkflowSnapshot` dataclasses and issues the same ~10 dispatch commands.

Because ADR 0002 put `SessionWorkflow` **on the Pi** (it is camera-coupled), the
UI, the renderer, and the authoritative state are all on the same machine. The
kiosk is therefore a genuinely local attachment, not a networked client. This is
what ADR 0002 anticipated: "The future touchscreen is also Pi-local, so this
needs no revisiting when the touchscreen lands."

## Decision

- **Render with Chromium `--kiosk` on the Pi, not a Python GUI toolkit.** The
  wireframe is already HTML/CSS with CSS-variable theming; the display is
  physically on the Pi (DisplayPort). We extract the per-screen markup + theme
  variables and **drop the `<x-dc>` / `deck-stage.js` presentation-deck runtime**
  (that runtime is authoring/slideshow scaffolding, not a shippable app shell).
- **A Pi-local relay replaces `session_console.py` as the renderer.** It reads
  the in-process workflow's summary/events/snapshot, serves them to Chromium on
  `localhost`, and forwards operator taps back to the same local workflow via the
  existing dispatch verbs. **No new processing-computer surface** is added; the
  relay sits beside the console, not below the store proxy.
- **Detached but easily attachable.** The relay subscribes to the workflow the
  same way the console does. v1 can run the console and the kiosk side by side;
  neither owns the workflow. This keeps the kiosk removable if hardware QA
  surfaces a blocker.
- **Work order is derived passively from decoded slide QRs** (`WO: —` until the
  first decode). The wireframe's standalone work-order scan screen (03) is
  deferred; if a pre-declared work order proves useful, it can be added later
  without changing this contract.
- **Calibration stays manual for v1.** The operator taps to confirm an empty
  backlight (`confirm_empty`); the wireframe's auto-countdown and
  auto-detect-clear-backlight ideas (screen 04) are deferred because both touch
  baseline-capture correctness, which is out of scope for a presentation layer.
- **Screen 11 ("⚠ EXPERIMENT — DO NOT USE") is excluded** from the build.

## Considered options

- **Python GUI toolkit (Qt / Tk / Kivy) instead of Chromium.** Rejected: it
  discards the finished HTML/CSS wireframe and its theming, and reimplements
  layout the design already specifies. Chromium renders the design as authored.
- **Host the UI on the processing computer.** Rejected for the same reason ADR
  0002 rejected it as a placement driver: the operator stands at the camera, the
  camera is on the Pi, and a PC-hosted UI would add a *second* wire surface
  (render/command across the link) for no benefit. The UI follows the operator;
  the operator follows the camera; the camera is on the Pi.
- **Keep the deck runtime (`deck-stage.js`) and ship the `.dc.html` directly.**
  Rejected: it is a design-presentation harness (slide navigation, authoring
  affordances), not an operator app; screen transitions must be driven by
  workflow events, not deck navigation.

## Open detail (noted, not a blocker)

- **Screen 02 "resume" needs the session number delivered to the Pi.**
  `resume_session(None)` latest-session discovery is deliberately
  processing-computer-local and **not** proxied (`remote_store.py`); deployed Pis
  are launched with an explicit `--session N` that `tools/run_receiver.py`
  prints. So the kiosk cannot ask "is there a session to resume?" the way the
  wireframe implies — the number has to reach the Pi (e.g. surfaced from the
  launch argument the Pi already runs with, or a small Pi-local record). This is
  an implementation detail of the resume screen, not a change to this decision.
  Block- and slide-phase restart recovery both already exist (commit cf4642f).

## Consequences

- The relay is a rendering/transport layer only. It must not add domain logic,
  a second source of truth, or a new mutating path — every state change still
  goes through the workflow's existing dispatch verbs and the ADR 0002
  `request_id` idempotency ledger.
- A cable pull cannot break the UI↔workflow path (both Pi-local). It stalls only
  the workflow's calls out to the processing-computer store, which the workflow
  already models as `TransportError`. The kiosk's offline banner (screen 22) is
  driven by the workflow's store-link health, and verdict-affecting actions are
  disabled while offline — connection health, not UI liveness.
- The kiosk is a presentation layer: no changes to segmentation, scoring, gates,
  `PASS_THRESHOLD`, or the capture state machine. Screen-to-state binding is a
  read of signals that already exist.
