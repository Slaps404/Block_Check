---
status: accepted
---

# Kiosk operator controls for Open Retrieval work orders (START/FINISH bracket)

## Context

[ADR-0009](0009-workorder-scoped-n2-retrieval-mode.md) established Open Retrieval:
a work-order-scoped N² identification mode driven by an operator **start/finish
bracket**, with the lifecycle `capturing → finalized → scoring → results-ready`.
ADR-0009 deliberately left one thread open — *"workflow/state-machine design still
open — see grilling session."* This ADR closes that thread for the **kiosk control
surface** (issue #155).

Two facts about the deployed kiosk shape the decision:

1. The work-order verbs (`start_work_order` / `finish_work_order`) are wired
   end-to-end — dispatch table (`code/session/console.py` `_COMMANDS`),
   arity-checked, proxied over `/rpc` (commit 2cec80d) — but the **only trigger
   is the `pi>`
   interactive console**. The touchscreen cannot open or close a work order, so
   the Open Retrieval flow never engages from the kiosk.
2. The kiosk's **"START SESSION" button is only a client-side `engaged` latch**
   (`screens.py` `action:"engage"` → `setFlag("engaged", true)`); the real backend
   session bind is deferred (#119c). The session is effectively **ambient** — it
   exists from process startup. So a "start session" step carries no backend
   meaning in this mode.

## Decision

### 2026-07-16 kiosk-layout amendment

The state gate now distinguishes a fresh session from a between-orders
session using durable, session-scoped `has_work_orders` state:

- No work orders: preserve the full startup preview and `START SESSION` label;
  in Open Retrieval that button dispatches `start_work_order` directly and
  opens the capture bracket.
- One or more work orders, none open: preserve an approximately same-size live
  preview with small `START NEW WORK ORDER` and `VIEW RESULTS` footer buttons.
- Results remain on the preview hub until the operator taps `VIEW RESULTS`.
  `GO BACK` clears the `view_results_guard` and returns to that preview.
- The prior `END SESSION` affordance is removed from this hub because its
  client-only disengage latch cannot close the ambient backend session.

`PiCaptureRuntime` seeds both `work_order_open` and `has_work_orders` from the
store at startup and maintains them locally, avoiding a database RPC on every
kiosk poll. This amendment supersedes the older screen wording below; the
idempotency and work-order lifecycle decisions remain unchanged.

1. **Opening a work order is the single operator entry gesture in
   `--open-retrieval` mode.** The first button retains the familiar `START
   SESSION` label and later buttons say `START NEW WORK ORDER`, but both dispatch
   `start_work_order`; there is no separate backend "start session" transition.
   *Rejected: auto-opening a work order on session start.* It is new backend
   lifecycle logic (ADR-0009 defers automatic transitions to v2) and it makes the
   operator-facing START button — the entire point of #155 — vestigial.

2. **`work_order_open` serves as the `engaged` gate in open-retrieval mode.**
   One bracket stays open across both ordered phases: blocks first, then slides.
   `FINISH WORK ORDER` is absent during block capture and appears only while
   slide capture is idle (`capture_mode == slide`, `capture_state == EMPTY`).
   Dispatching it closes the bracket and returns to the between-orders preview.
   Normal closed-set `engage`/`disengage` behavior is unchanged.

3. **Button gating is router-driven, not descriptor-driven.** `relay.state()` gains
   two read-only projections — `open_retrieval` and `work_order_open` — and
   `select_screen` (`code/kiosk/router.py`) picks screens/buttons from them, keyed
   off the button catalog in `code/kiosk/screens.py`. This matches the
   existing `results_table` precedent (routed on `view_results_guard` +
   `results_ready_work_orders`). No new renderer descriptor field is introduced; the
   frozen button-action vocabulary (`dispatch`/`guard`/`back`/`engage`/`disengage`)
   is unchanged, and the two new buttons are plain `action:"dispatch"` verbs.

4. **Double-START is guarded in the store, not only the UI.**
   `ProcessingStore.start_work_order` becomes **idempotent**: if a `capturing` row
   already exists for the session it returns that row's id instead of inserting a
   second. The UI *additionally* hides/disables START while `work_order_open`, but
   correctness — no orphan `capturing` row — is guaranteed by the store even under a
   Pi restart mid-bracket, a `pi>` console start, or a rapid double-tap.
   The UI's `work_order_open` is a **local `PiCaptureRuntime` boolean** (set on start
   success, cleared on finish, **seeded from the store once at runtime init** for
   restart recovery). Both the touchscreen and the console dispatch through
   `PiCaptureRuntime`, so the single boolean stays authoritative for the UI without a
   per-poll store round-trip.
   *Rejected: a per-poll authoritative store read.* It adds a network call to the
   750 ms hot path and carries the `RemoteProcessingStore`/`/rpc`-whitelist crash
   hazard for every new store method the Pi calls.

5. **Each later work order restarts at blocks.** If the session already has a
   work order, `ProcessingStore.start_work_order` durably resets phase from
   `slides` to `blocks` before inserting the new bracket. `PiCaptureRuntime`
   then synchronizes its live controller to that durable phase.

6. **Missing receiver state fails safe.** Runtime startup handles only the exact
   deterministic `unknown method` error from a work-order state RPC; transport
   errors and every other store error still propagate. A receiver missing
   `open_work_order_id` always blocks startup with a clear instruction to restart
   the processing receiver. If only `has_work_orders` is missing, runtime may
   infer `True` only when `open_work_order_id` positively observed an open
   bracket. With no positively observed bracket it blocks startup instead of
   silently guessing `False`.

## Scope

**In scope (#155):** work-order dispatch buttons; the `open_retrieval` +
`work_order_open` projections; router gating for slide-idle finish and results
guard/back; idempotent start; durable phase/controller reset for later orders;
and boot-seeded local flags with narrow receiver fallback.

**Out of scope (already owned elsewhere):** all verdict / boundary rules —
single-block (M=1) threshold-only fallback, "claimed block not in this order",
gate-failed pairs — remain solely in `work_order_evaluator.evaluate_work_order`. The
results-table UI is prior-issue work. No new verdict path is added by #155.

## Consequences

- The kiosk gains **no new verdict path**; the N² verdict stays in
  `evaluate_work_order`.
- `start_work_order`'s contract changes from *"always insert"* to
  *"insert-or-return-existing"*. Post-`FINISH` the row is `finalized`, so a
  subsequent START resets phase to `blocks`, inserts a fresh `capturing` row,
  and synchronizes the block controller.
- A bracket cannot close during block capture; finish is available only after
  entering slide capture and reaching its idle placement screen.
- `GO BACK` clears the results guard without changing backend lifecycle.
- The local UI flag may momentarily disagree with the DB, but only in the
  enable/disable direction; it can never cause an orphan row because the store guard
  is authoritative. After a restart mid-bracket the seed restores the correct
  START-vs-FINISH affordance and captures are preserved.
