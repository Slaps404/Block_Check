---
status: accepted
date: 2026-07-09
---

# Debug capture-review gate pauses before commit rather than deduping after

## Context

During hardware bring-up the operator needs to see each captured still and
retake it if the framing, focus, or backlight is wrong ("capture quality", not
verdict reasoning — the QC-artifact pipeline on the processing computer is out
of scope). This is a Pi-local concern: the preview frames and the published
full-res PNG both live on the Pi, and the kiosk server is Pi-local, so no
cross-machine transport is involved.

The obvious-looking shortcut is to let a capture commit as normal and simply
retake over the top, keeping only the newest per identity at finalize. That
does **not** work against the current store, which is deliberately
first-writer-wins:

- **Blocks** are one row per `(session, block_id)` and the capture write is
  guarded `WHERE capture_id IS NULL` (`code/session_workflow.py:1224`). A retake
  of an already-captured block hits `updated == 0` and
  `raise ValueError("capture does not match one unique unfilled set")`. The only
  overwrite path is `recapture=True` gated on `preprocessing_status='failed'`.
- **Slide verdicts** are first-writer-wins keyed on `block_id`
  (`WHERE ... verdict IS NULL`, `code/session_workflow.py:1679`). A second
  capture of the same slide returns `ClaimOutcome(False, "Slide already
  processed")` — no fresh verdict, just a stray appended still row.

So there are no post-commit duplicates to dedupe: blocks throw, slides refuse to
re-score. Making commit-then-overwrite work would require inverting the store's
`IS NULL` compare-and-set guards — a change to the production durability model
on the processing computer, the highest-blast-radius surface in the system.

## Decision

Add an opt-in debug review gate, enabled by a `--review-captures` launch flag on
`run_pi_session` (plumbed through `tools/start_live_session.bat`), **off by
default** so production behaviour is byte-for-byte unchanged. When on, every
capture (blocks **and** slides) pauses on a new `AWAITING_ACCEPT` capture state
*before* the still is uploaded/scored. The kiosk shows the held still
fit-to-screen with **ACCEPT** and **RETAKE**:

- **ACCEPT** runs the existing capture consumer (upload for blocks,
  `capture_slide` for slides) and advances normally to `WAITING_FOR_REMOVAL`, so
  the verdict flow (screens 14/15/16) is untouched.
- **RETAKE** discards the held still and does an immediate re-shoot (reusing the
  `retry_capture` path; the operator repositions first, as on screen 17). It
  must re-preserve the scanned `block_id`, which `accept_capture_result` clears
  on capture (`code/capture/capture_session.py:263`).

The gate holds at the single point between "still saved to the Pi's local
`CaptureStore`" (`code/capture/capture_runtime.py:137`) and the consumer call
(`:146`). The camera loop must **not** block while waiting — it transitions to
`AWAITING_ACCEPT` and returns, so preview frames keep flowing and ACCEPT/RETAKE
arrive as ordinary kiosk verbs on the command thread.

## Consequences

- Nothing crosses to the processing computer until ACCEPT, so the store's
  first-writer-wins guards are respected, not fought.
- A new `AWAITING_ACCEPT` state, an `accept_capture` verb, a widened
  `retry_capture` (valid from `AWAITING_ACCEPT`), a router rule, and a new review
  screen (catalog + `index.html` fallback) are required. All Pi-local, all
  behind the flag.
- Slide retakes yield a genuine fresh verdict (the reject never committed).
- Because the gate is off by default and self-describing via the new state, the
  kiosk needs no separate debug flag — it renders the review screen whenever it
  sees `AWAITING_ACCEPT`.

## Considered and rejected

- **Commit-then-dedupe / keep-newest at finalize** — impossible without
  inverting the store's durability guards (see Context); also "newest wins"
  would keep a *worse* retake.
- **Client-only overlay retake** (like the screen-18 duplicate flash) — the bad
  capture is already committed by the time the operator reacts.
