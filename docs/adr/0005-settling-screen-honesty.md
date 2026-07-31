# SETTLING must not reuse "Place" or session "Processing" copy on the kiosk

**Status:** accepted (2026-07-09)  
**Evidence:** kiosk workflow QA (session 6) — settling-screen honesty bugs K2, K5.

During hands-on kiosk QA, operators placed a block or slide but the screen kept
saying **Place Block** for 3–5 seconds. Code mapped both `EMPTY` and `SETTLING`
to Place (old OQ1). `SETTLING` means "specimen detected, waiting for motion to
stop"; it is Pi-local and unrelated to laptop preprocessing or the `processing`
screen used during `draining_blocks` / `finalizing`.

## Decision

When `capture_state == SETTLING`, the kiosk shows shared screen id **`hold_still`**
with headline **Hold Still** (centred Place-style layout, no progress bar, no
buttons, status bar on). Not Place, not Capturing, not session Processing.

**Block ID:** show **BLOCK ID: …** as the sub line on Place (07) → Hold Still →
Capturing (08) → Remove (09) until Scan Block (06). Prefer `pending_block_id` /
last scan event pre-capture, then `latest_block_id` after publish. Slide-mode
Hold Still has no block-ID sub.

## Considered options

| Option | Outcome |
| --- | --- |
| A. Keep Place during settle | Rejected — misleading |
| B. Route SETTLING → `processing` | Rejected — wrong semantics |
| C. Route SETTLING → Capturing | Rejected — shutter not fired yet |
| D. Dedicated Hold Still router screen | **Accepted** |
| E. Client headline swap on 07/12 | Rejected — prefer honest screen id |

## Consequences

- Router R19/R20 are EMPTY-only; SETTLING → `hold_still`.
- Catalog + fallback HTML + `preview_kiosk` include `hold_still`.
- Design_spec empty-sub on 08/09 is deliberately diverged for block-ID persistence.
- Does not shorten settle time or unblock upload lock; debug telemetry still needed.
