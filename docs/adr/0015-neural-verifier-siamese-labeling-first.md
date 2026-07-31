# Neural verifier: labeling-first Siamese experiment in-repo

**Status:** accepted (2026-07-09) — labeling framework locked in grill; training details still provisional

LJI Block Check stays a classical CV product. A neural block–slide verifier is
an optional experiment that may later support PASS/REVIEW on claimed pairs.
We will not train or integrate a model until a labeled pair corpus exists, and
we will not put that work on the production path until it beats classical
scoring on held-out block families and fails closed to REVIEW.

## Decision

- **Same repo, experiment first.** Live under `experiments/neural_verifier/`.
  Promote to something like `code/ml_verifier/` only after validation. No
  separate repo: the experiment depends on manifests, classical masks, scores,
  near-miss diagnostics, and PASS/REVIEW semantics.
- **Phase 1 is a labeling harness, not training.** A Streamlit review tool
  lets a human curate sparse hard negatives per block and write labels to a
  new labeled CSV without mutating the frozen corpus. Acceptance criteria
  live with the labeling harness under `experiments/neural_verifier/`
  (lab notes filed in the Obsidian vault).
- **Intended model (phase 2+):** a two-tower / Siamese verifier that scores
  claimed pairs for PASS/REVIEW support — not open-ended top-1 retrieval.
- **Inputs start as classical mask normalized crops** (bbox → pad → fixed
  canvas, preserve aspect ratio). Do not train segmentation first unless
  review shows segmentation is the bottleneck. Full-frame RGB is a leakage
  probe, not the first training input (cassette color, labels, barcodes,
  batch lighting).
- **Splits are by block family / specimen / workorder group**, never random
  image or pair splits. All slides from one physical block family stay in one
  split.
- **Sparse pair table, not n² storage or n² classical mining at scale.**
  At ~41 sets, all-pairs classical diagnostics are a dry-run seed only. At
  600–800 live sets they will be too slow and not trustworthy enough as the
  primary near-miss source. Hard negatives are human-curated by browsing
  same `work_order` candidates from live capture inventory; easy negatives
  are sampled later from other work orders (rule deferred).
- **Positive/negative tiers drive training inclusion.** Clear + realistic
  positives are main training; hard positives get lower weight; uncertain and
  exclude stay out of training; fully rearranged true positives stay in a
  challenge set, not the main train mix.
- **Evaluation is conservative verification.** False accepts (wrong pair →
  PASS) are the primary danger; false reviews are safer. Report
  tier-stratified metrics; retrieval@k is a stress test only.
- **Backbone preference (provisional):** frozen DINOv2 (or DINOv3 if
  access/license is cleared) plus a small projection head; CosineEmbeddingLoss
  to prove plumbing, then InfoNCE / hard-negative mining. Agents build
  tooling and audits; they do not hand-tune weights.
- **Safeguards:** gitignore `outputs/`, `runs/`, `models/`, weight files; no
  ML deps on the production import path.

## Considered options

- **Separate ML repo** — rejected: shared manifests, masks, classical scores,
  and PASS/REVIEW semantics would fork immediately.
- **Train before labeling / use classical scores as labels** — rejected:
  near misses and hard cases need human review; classical score is not ground
  truth.
- **Segmentation-first neural model** — deferred: classical masks are the
  stable geometry input unless review proves otherwise.
- **Full-image Siamese from day one** — rejected as primary path: high
  shortcut/leakage risk; keep as ablation only.
- **Production PASS/REVIEW integration in v1** — rejected: labeling harness
  and held-out family evaluation come first.

## Resolved in grill

- **`relationships.csv` origin:** produced by `freeze_corpus` (commit
  `ea8bc3f`, Refs #129), not a hand-authored upload. Current frozen copy:
  `outputs/runtime_campaigns/v3-corpus/relationships.csv` — 41 `true_pair` +
  79 auto-selected `hard_negative` rows (best wrong identity plus nearby
  competitors within score gap 0.04). Image paths resolve against the main
  project `images/pi_images_v3/` tree.
- **Freeze stays immutable; labels are an overlay.** Hand-picks and review
  labels write to something like `relationships_labeled.csv` (or equivalent
  under `experiments/neural_verifier/`). They do not rewrite
  `v3-corpus/relationships.csv`. A later corpus version may re-freeze if we
  want a new classical seed; that is separate from labeling.
- **Framework first; 41-set corpus is a dry-run, not the training set.**
  Build the picker / labeled overlay / inventory export path now so it is
  ready when ~600–800 live sets exist. Do not plan to train the verifier on
  the 41 v3 sets alone.
- **Live capture storage is fine as-is.** Keep using processing-root
  `sessions.sqlite3` + session folders (`captures/`, `slide_captures/`,
  artifacts). Transfer into training = **export** true pairs (and later
  labels) into a growing inventory / pair CSV under
  `experiments/neural_verifier/`. Do not hand-edit `v3-corpus/`; that freeze
  stays a classical benchmark seed only.
- **Hard-negative selection: human browse is enough; classical rank is optional.**
  Hard negatives do **not** need to be the single nearest miss — any
  confusing wrong pair in the browse pool is valid. Same-`work_order` (live)
  or same-tissue (v3 dry-run) browse + human pick remains the primary path.
  Classical all-pairs / IoU sort is an optional aid when the pool is large
  and hard to scan, not a requirement. Soft floor ≥1 when candidates exist.
- **Live / large-N browse key remains same `work_order`.** Tissue is not on
  the barcode. Full n² at 600–800 sets stays out of scope as the primary
  miner. Easy negatives later from other work orders (rule TBD).
- **v3 dry-run** has tissue and no WO: same-tissue pool is fine; `lung` ≠
  `lungs`.
- **No hard-negative quota.** Soft floor: at least one pick when any
  same-work-order (or dry-run same-tissue) candidates exist.
- **v1 labeled schema is slim.** Identity + paths + `work_order` (and tissue
  when known), plus on hard-negative pick: `is_match=0`,
  `negative_tier=hard_negative`, `pair_source=manual_negative`,
  `use_for_training=yes`, `review_status=reviewed_ok`, optional `notes`.
  True pairs get a light confirm/skip for now. Deferred: full 7-button
  taxonomy, layout/fragment fields, `visual_reason`, easy-negative sampling.
- **CSV remains viable at 600–800 sets.** Sparse pair table stays thousands
  of rows, not n². No training DB required for scale alone. Live SQLite
  remains capture authority; training catalog is exported CSV.
- **Train/val/test split by `work_order`.** All pairs from one WO stay in
  one split so same-WO hard negatives cannot leak across folds. Later work
  orders (including ones with similar tissue) are the real held-out test;
  that is intentional generalization, not contamination.
- **Live export = claimed true pairs without classical scores.** One row per
  block+linked-slide claim: paths, `block_id`, `work_order`, stain,
  optional verdict. Classical `score` is omitted from the training inventory
  (useful as optional UI context later, not required). Unpaired captures
  stay out of v1 export.
- **Hard negatives reuse other true-pair slides.** A slide that is the
  positive for block B is a valid hard-negative candidate for block A in the
  same work order. Images are shared across pair rows; what must not cross
  splits is the work order, not the image’s dual role as true-pair and
  hard-neg.
- **App + labeled overlay live under `experiments/neural_verifier/`.**
  Streamlit entrypoint (e.g. `review_pairs.py`) reads inventory / live
  export CSVs and writes `relationships_labeled.csv` there. Frozen
  `v3-corpus/` is read-only dry-run input only.
- **Neural starts parallel to classical, not as a replacement.** Classical
  remains production PASS/REVIEW; neural is a side score / REVIEW-support
  candidate until held-out work-order tests show it is consistently better
  (or better on the failure modes that matter). Classical may later be
  partially eclipsed; replace vs ensemble is deferred until after labeling
  and first training runs. Fail closed to REVIEW.
- **Promotion criteria stay qualitative until more live data exists.**
  Direction only: evaluate on held-out work orders; false accepts (wrong
  pair → PASS) are the primary danger and must not get worse than
  classical; fewer unnecessary REVIEWs are secondary and must not buy
  false accepts. The 41-set corpus is too small to set numeric gates or
  promote to `code/ml_verifier/`. Revisit numbers after a real multi-WO
  labeled batch.
- **Hard-neg browse UX validated in prototype.** Side-by-side block +
  candidate slide, scroll within same WO (live) / same tissue (v3 dry-run),
  pick/unpick. Keep that interaction for the real labeling harness;
  `experiments/neural_verifier/prototype/` remains throwaway.
- **Same-tissue / same-WO manual labeling stays viable.** Hard negatives do
  not need to be the absolute nearest miss — confusing wrong pairs are
  enough. Classical IoU ranking is optional help for large pools, not the
  gate for labeling.

## Open detail (grill targets)

- (none blocking for labeling-harness v1 — further grill as training
  approaches)

## Sources

- Labeling harness: `experiments/neural_verifier/` (vault: neural verifier
  labeling harness + siamese notes)
- A prior frozen-corpus experiment may be consulted as historical evidence; its
  runtime-harness writer has been retired.
- Adjacent classical near-miss mindset: ADR 0003
- Related visual QA: `tools/scoring_diagnostics/scoring_contact_sheet.py`
