---
status: accepted
---

> The capture-time visibility and batch-dispatch parts of decisions 5, 7, and 8
> are superseded by ADR 0017. The complete within-work-order scoring and
> rank/margin verdict rules remain accepted.

# Work-order-scoped N² retrieval mode for the kiosk

## Context

The deployed kiosk does **closed-set verification**: each slide carries a decoded
identity (barcode → claimed block), and the pipeline scores that *one* claimed
(slide, block) pair against a fixed `PASS_THRESHOLD` (0.85, an uncalibrated
placeholder — Blocker #1). Absolute scores are not comparable across tissue types
(lung true-pairs score low; spread pancreas sections have slop), so no single fixed
threshold serves every tissue.

## Decision

Add an opt-in **session-wide** launch mode — **Open Retrieval**, enabled by the
`--open-retrieval` flag (default off), threaded PowerShell → SSH → `argparse` — that
replaces per-slide threshold verification with **within-work-order N² identification**:

1. Scope the N² to **one work order** at a time. Work orders are queueable; each is
   scored only against itself. This bounds cost and makes the confuser set the right
   one — the other blocks that could plausibly be confused, not the whole corpus. The
   work order is defined by the operator's **start/finish bracket** — everything captured
   between the two is one N² set. Decoded barcode identity supplies each slide's *claim*
   (which block it says it is), not the grouping. Fails safe: a mis-bracketed extra block
   just becomes another confuser and surfaces in the results. (Barcode-defined grouping
   is a v2 hardening.)
2. The comparison is **bipartite**: each slide is scored against **every block in the
   work order** (K slides × M blocks). Many slides may share one block. For a slide,
   the blocks are ranked; `B₁` is the top match.
3. Turn calibration into **ranking**. The verdict is a hybrid — claim label for
   *correctness*, runner-up margin for *confidence*:
   - **PASS** iff `B₁ == claim` **and** no other block is within the margin of `B₁`.
   - **REVIEW** otherwise (single verdict — no reason split in v1).
   The margin band is measured relative to **`B₁` (the winner)**, not the claim, so it
   stays a valid confidence signal even when the claim is wrong.
4. On REVIEW, the operator inspects an **expandable contact sheet** (reusing
   `code/contact_sheet.py`, full 5-panel) for the flagged pairs, each specimen labeled
   with its unique ID, to visually confirm the (non-)match. The claim block is always
   surfaced. The renderer sits behind a **seam** (swappable interface) so a different
   contact-sheet script can replace the 5-panel one later without pipeline changes.
5. **No per-slide verdict during capture** — captures just confirm "captured." The
   operator explicitly **starts** and **finishes** a work order; finishing triggers the
   batch N². (Auto-finish when all claimed blocks are scanned is a deferred v2 option.)
6. Results are presented as a **sortable table** (roll-up): one row per slide,
   color-coded PASS/REVIEW, REVIEW sorted to the top, each row expandable to its
   contact sheet(s). This is a new UI shape — today's kiosk has no list/summary screen.
7. **Concurrent work orders**: finishing an order must not block the kiosk. Scoring runs
   as an **async background job**; the operator can start a new order while a prior one
   scores. This introduces a work-order lifecycle: `capturing → finalized → scoring →
   results-ready`.
8. **Execution placement**: the N² batch runs on the **processing computer** (where the
   images and horsepower already live, via the existing capture-receiver seam), not on
   the Pi. Orders are scored **one at a time** (a queue; hardware parallelism deferred).
   Results (verdict table + contact-sheet PNGs) are returned to the Pi and persisted to
   CSV for audit. v1 shows results for orders finished **in the current session**; a
   historical/external session browser is deferred (the persisted CSVs enable it later).
   Future option: run scoring on the lab's own Linux servers.

Reuses the standalone N² engine (`tools/scoring_diagnostics/pair_diagnostics.py`) and
the production scorer, preserving the one-way `code/` dependency and "never makes a
verdict" contract of the diagnostic layer — the *verdict* stays in the kiosk pipeline.

## Boundary rules

- **Single-block order (M=1, no runner-up):** Match Margin is undefined, so fall back to
  the absolute `PASS_THRESHOLD` for that lone pair, and tag the row "unverified —
  threshold only" so it is never mistaken for a ranked pass. (Deemed rare in practice.)
- **Claimed block not scanned in the order:** REVIEW, row labeled "claimed block not in
  this order" (usually the operator missed scanning a block).
- **Pair fails quality gates (no score):** that pair cannot be a match candidate; if the
  *claimed* pair itself gate-fails, the slide is REVIEW (fail-closed, as today).

## Consequences

- Sidesteps `PASS_THRESHOLD` calibration for this mode: ranking is invariant to the
  per-tissue score offset that makes a fixed threshold meaningless.
- Changes workflow timing: scoring is deferred to a batch after captures, not streamed
  per slide (workflow/state-machine design still open — see grilling session).
- Introduces a runtime, unsupervised margin (top-1 vs runner-up) distinct from the
  existing supervised **Near-Miss Margin** (true vs best-wrong) in `CONTEXT.md`. The
  runtime concept is named **Match Margin** (`MATCH_MARGIN`, default 0.05); both terms
  are now in `CONTEXT.md`, cross-referenced so they cannot be conflated.
