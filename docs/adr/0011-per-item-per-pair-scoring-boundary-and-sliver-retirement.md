---
status: accepted
---

# Per-item / per-pair scoring boundary, and retirement of the sliver gate

Extends ADR 0009 (work-order-scoped N² retrieval). Tracked by issue #158.
Implementation complete. The equality test in Consequences is the guard, not a
decision gate.

## Context

The N² work-order scorer (`default_work_order_scorer`, `session/workflow.py:3305`)
reuses the **single-pair** `pipeline.decide_claim` in a nested loop over K slides ×
M blocks — deliberately, to share one production scorer with the live single-claim
path (`ProcessingStore.resolve_claim`) for batch/live parity.

The side effect: everything `decide_claim` does runs **per pair**, including work
that depends only on one specimen. Measured empirically (synthetic 3-block × 4-slide
work order, counting real calls; script in scratchpad):

- `build_locked_score_cache` (`verify/scorer.py:121`) — which runs `radial_normalize_mask`
  (the 256×256 canonical warp) + `_component_features` (`connectedComponentsWithStats`)
  — fires **24 = 2·M·K** times (once for the block side, once for the slide side, on
  every pair). A per-item cache needs only **M+K = 7**.
- The quality gates (`verify/gates.py`) also run per pair, though each check keys off a
  single specimen (mask coverage, contours, `roi_ok`) except one pair-coupled rule
  (sliver-AND, see below).

What is **already** correct: segmentation runs once per item (block masks are
`cv2.imread` from a PNG written at capture; slides segmented once per work order).
The N² loop is **fully serial** (plain `for`/`for`); the only concurrency is a
`ThreadPoolExecutor` (default 1 worker) that schedules whole jobs via threads (shared
heap, no pickling) — so there is **no** parallel re-segmentation or per-worker mask
duplication. The waste is purely the per-pair recompute of pair-independent facts.

## Decision

1. **Draw an explicit per-item / per-pair boundary.**
   - **Per item (N+M), owned by preparation:** segmentation (already) → radial
     normalization → `LockedScoreCache`, plus all per-item gate facts (mask quality,
     `roi_ok`). Preparation becomes the owner of "is this specimen usable + its
     normalized cache" — a coherent home that dissolves the earlier layering question.
   - **Per pair (N×M), owned by the scorer:** only the irreducible work — the boolean
     combination of per-item gate facts, the locked rotation-search alignment
     (`align_normalized_masks`), and features computed on the *aligned* slide mask
     (`verify/scorer.py:145`, genuinely pair-dependent).
   - Target: **M+K** normalizations instead of 2·M·K. Route the loop through the
     existing cache-taking entry point `score_routed_caches` (`verify/scorer.py:126`) rather
     than the per-pair-rebuilding `score_pair_result_routed`.

2. **Retire the sliver-like quality gate** (`_is_sliver_like` AND rule, `verify/gates.py:65`).
   Sliver-like is a *shape* property (thin, low-coverage, high-aspect mask); it was
   used as a proxy for *indiscriminable* tissue (identity — e.g. skin, which resembles
   many others). That is a proxy error: a thin specimen can be perfectly discriminable,
   and a sliver filter wrongly cuts it. The correct detector of indiscriminability is
   **open retrieval failing to surface a confident top match** — which is what actually
   demonstrates a specimen cannot be told apart. Archive (remove the call, keep the
   function + constants) so it is reversible if skin returns as a target.

## Considered options (sliver)

- **Keep sliver, just hoisted per-item** — the AND rule is a cheap boolean over two
  per-item slivers; it hoists to N+M with *zero* behavior change. Rejected: keeps a
  known-wrong proxy.
- **Per-item sliver → REVIEW (fail if either side is sliver)** — stricter, protects
  both paths. Rejected: still the wrong proxy, just more aggressive.
- **Archive (chosen)** — sliverness is not the right signal; open retrieval is.

## Resolved decisions (grill, 2026-07-16)

The decisions above set the target; the wiring specifics below were left open in
draft and resolved in a design interview with the owner.

**Sequencing.** Ship **decision 2 (retire sliver) first**, as its own change, then
decision 1 (perf refactor) second. The two are independent and the equality test only
guards decision 1.

**Scope — cache only, gates untouched.** Do **not** hoist the gates. Each gate check
keys off a single specimen and operates on the in-memory `specimen.mask` (never touches
disk), so it is already cheap and correctly per-item in effect. Leaving the gate suite
in place per-pair reintroduces no I/O and keeps failure `reason`/`stage` strings
bit-stable — which makes open question 3 below **not applicable** (no gate recomposition
happens).

1. **Block-side precompute site → precompute pass in `default_work_order_scorer`.**
   A pre-pass at the top of the N² loop builds both `block_caches` and `slide_caches`
   before scoring starts. Chosen over touching `_load_block_result` because it treats
   both sides uniformly, avoids branching on the disk-load path, and leaves the live
   single-claim path untouched.
2. **Cache-threading seam → option (a): `scorer=` passthrough on `decide_claim`.**
   Add an optional `scorer=` parameter to `decide_claim` (`code/session/pipeline.py`),
   forwarded to the `scorer=` hook that already exists on `compose_prepared_pair`
   (`code/verify/pair_composition.py:36`). The work-order loop injects a closure that
   looks up the pre-built caches and calls `score_routed_caches`. The live single-claim
   path leaves `scorer=` at its default → unchanged. Rejected attaching the cache to
   `PreparedSpecimen` (circular-import risk + pollutes the shared live path).
   - **Cache lookup keys on object identity (`id(result)`)**, not the `item_id` string.
     The loop passes the exact same specimen objects down through `decide_claim`, so
     identity matches; block and slide caches live in separate dicts, so no collision.
     Robust against any future change to the `item_id` string format.
3. **Gate recomposition ordering — not applicable.** Because gates are not hoisted
   (see Scope above), the first-failure ordering is untouched and no reason/stage
   strings can drift.

## Consequences

- **The perf refactor (decision 1) is behavior-preserving.** Guard it with a
  tolerance-based equality test: score a real/synthetic work order both ways and
  assert every score identical within `1e-9`, **and** assert the cache-build count
  drops from 2·M·K to M+K. Ship it as its own change, separate from decision 2.
- **Sliver retirement (decision 2) is a deliberate behavior change**, not covered by
  the equality test: pairs where both masks are sliver-like currently REVIEW; after
  removal they can score and PASS. It also affects the **live single-claim path**
  (`resolve_claim` shares `decide_claim`), which has **no retrieval backstop**.
  Accepted because (a) production is moving to open-retrieval until a model is trained,
  so single-claim is transitional, and (b) `PASS_THRESHOLD` parity between the paths
  was never validated (it is an uncalibrated placeholder), so "identical verdicts" was
  never a real guarantee to preserve.
- ADR 0009's "pair fails quality gates → fail-closed REVIEW" boundary rule is otherwise
  unchanged; only the sliver rule leaves the gate suite.
- Micro-note (not in scope): in the point-layout branch the slide's cached
  `component_features` is unused (`verify/scorer.py:143-146` recomputes features on the
  *aligned* slide mask), so a per-item cache need not build slide component features.
  Left as a follow-up; not worth complicating the first refactor.
