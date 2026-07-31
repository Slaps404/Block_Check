# Lessons Learned

This is the canonical do-not-repeat file. Keep durable lessons here instead of
copying them into project context.

## Product Direction

- Treat the system as claimed-pair verification, not open retrieval.
- Barcode/metadata is the identity anchor.
- Computer vision is a conservative safety check.
- PASS requires strong evidence; uncertainty routes to REVIEW.
- False-PASS risk matters more than open top-1 ranking.

## Do Not Repeat

- Do not make a newly synced Pi client require a newly added PC receiver RPC
  at startup without either coordinated receiver restart or a narrow
  rolling-upgrade fallback. `Sync-LJIPi` updates Pi files only; an already
  running receiver keeps its old in-memory RPC whitelist.

- Treat tuned segmentation thresholds and camera calibration as one fixed-camera
  imaging contract. Enabling AE/AWB changes the image distribution underneath
  color/brightness thresholds even when the auto result is stable. Removing
  AE/AWB alone is incomplete if auto-tuned absolute luma/clipping gates remain:
  those universal gates can reject the restored, hardware-known-good fixed block
  and slide exposures. Verify the fixed control lock, retain luma/clipping as
  telemetry, and fail closed on chromatic occupancy. This detector is
  intentionally neutral-blind: neutral objects, clear glass, a lens cap, or an
  unlit backlight may still pass and remain a hardware/operator QA risk.

- When applying calibrated camera controls, merge them over the complete role
  controls instead of replacing those controls. Replacement silently dropped
  `NoiseReductionMode=Minimal` (the `cdn_off` equivalent); Picamera2 restored
  its HighQuality still default, enabled Pi 5 temporal denoise (TDN), and
  aborted the preview-to-still switch.
- Never label requested or preview-verified controls as observed still metadata.
  The current capture metadata is a lock fingerprint, not proof of the exact
  still-request values; future hardening should capture metadata from the still
  request itself.
- Do not chase open retrieval accuracy as the primary product metric.
- Do not treat cross-tissue separation as proof of safety; same-tissue near
  misses are the hard cases.
- Do not tune PASS/REVIEW thresholds without hard near-miss mismatches.
- Do not use holdout data during iteration unless explicitly approved.
- Do not change segmentation, descriptors, scoring, and thresholding in the
  same iteration.
- Do not promote Shape Context, TPS warping, full rotation invariance, SWD, or
  standalone HOG without clear true-vs-near-miss improvement.
- Do not bypass `prepare_specimen` with ad-hoc load/parse/ROI/clean chains in
  production-shaped scoring or diagnostics.
- Do not treat `NO_VISUAL_CONTRADICTION` or a high visual score as proof of
  identity.
- Do not cite old "14-set HE-only" or iPhone calibration results as current Pi
  product truth.
- Do not promote arbitrary best-subset slide matching to production unless
  residual tissue is explicitly evaluated and hard near-miss diagnostics show
  no false-PASS increase.
- Do not re-propose a new shape *distance* (Chamfer, rotation-invariant
  matching) as the esophagus fix. The limit is outline feature entropy, not the
  metric (see Known Failed below). Refine capture / add non-silhouette features
  or route sparse specimens to REVIEW instead.
- Do not partially revert a merged experiment (e.g. QuPath classifiers) and
  let agents "restore" pieces from chat context. That produces a Frankenstein:
  production imports one path, tests/docs/constants still describe another.
  When backing out an approach, excise it in one PR (code + tests + docs +
  constants) and archive reference code under `archive/` instead of leaving dead
  imports in `code/`.

## Known Failed / Weak Approaches

- Paraffin-relative faint block-mask adapter (2026-07-01): recovered the block
  37 colon tail, but also admitted ambiguous optical halo/background regions
  and nearly saturated its 5% paraffin-window safety cap. Archived unpromoted;
  do not wire it into production without a stronger false-positive constraint
  and full-mask visual approval.
- Hu/Zernike descriptor path: weak cross-modal signal; old production direction
  is now legacy.
- Shape Context: did not improve true-vs-near-miss separability.
- SWD boundary-cloud descriptor (2026-06-15): AUC about 0.509 on dev; failed
  because cross-modal boundary density asymmetry made true pairs and near
  misses hard to distinguish.
- HOG silhouette descriptor (2026-06-15): AUC about 0.495 overall; lung signal
  improved but esophagus was anti-discriminative.
- Normalized-mask scoring is useful but not sufficient proof of identity; recent
  diagnostics show weak separation between true pairs and same-tissue near
  misses.
- Chamfer / rotation-invariant contour matching (2026-06-25): centroid+RMS-
  normalized contour Chamfer with D4 / fine-rotation sweep, proposed to replace
  strict mask-IoU on sparse tissue (esophagus, lung). On esophagus it out-ranks
  IoU (AUC 0.40 -> 0.71 on identical masks, confirming IoU is the wrong metric
  for thin ribbon shapes) but still cannot threshold: 48/48 same-tissue impostor
  overlap. Cause is information-theoretic, not tuning -- normalized esophagus
  outlines are not specimen-distinctive (most-similar pair set_05<->set_03 is an
  IMPOSTOR at dist 0.069, beating every true pair). Also degrades lung
  (AUC 0.67 vs production near-perfect). Gemini's proposed gamma=12 is
  miscalibrated for RMS-normalized space (dists 0.07-0.88 -> exp(-d/12) ~ 0.95,
  no discrimination); raw Chamfer is not rotation-invariant (AUC 0.22), so it
  merely re-implements the existing rotation_search D4 machinery. Decision: keep
  D4; the esophagus bottleneck is outline feature entropy + capture quality.
  Probe: outputs/diagnostics/grill_chamfer_detail.csv.

## Resolved Hard Cases

- **set_030 (kidney) — resolved max-radius normalization defect.** The slide's
  848px residual is a plausible ribbon-cutting remnant: only 1.41% of total
  tissue, but remote enough to inflate the old maximum radius by 54% and shrink
  the two dominant pieces. RMS-radius normalization retains the residual as IoU
  evidence while weighting its scale influence by area and squared distance.
  On the fixed true/near-miss set, its routed margin changes -0.3897 -> +0.0382.
  This is improved geometric consistency, not proof of specimen identity.

## Known Un-Separable Pairs (capture / depth ceilings, v3 41-set)

Documented in iteration 042. These true pairs stay un-separable under **every**
scoring configuration (dense IoU, sparse point_layout, blends). The limit is the
mask/capture, NOT the router or the threshold — correctly route to REVIEW. Do not
re-attack with a new metric or routing feature; a fix requires better capture or
segmentation of the specific defect named.

- **set_024 (lung-panc) — block surface vs deep-layer mismatch; unfixable now.**
  Long pancreas: on the slide it is stringy/sparse-but-connected; on the block
  the strands amalgamate/fuse into one mass as they spread into deeper layers, and
  the block image sees *through* those deeper layers, which do not match the
  slide's surface section. Segmentation is correct; the surfaces genuinely
  differ. Not fixable without perfectly scanning only the block's top surface or
  training an ML model for it.
- **set_027 (lung-panc) — same "long pink"/pancreas depth issue as set_024.**
  Logged as an end risk. NOTE: it is the lone routing sign-flip (dense -0.080 ->
  sparse +0.064) but rests on n=1 within lung-panc; do not build routing features
  for it alone.
- **set_026 (lung-panc) — block/slide shape mismatch + extra block dot.** Same
  depth-of-section family as 024. Block shape does not match the slide, and the
  block has a small dot absent on the slide. Unresolvable by simple methods.
- **set_036 / set_037 (colon) — faint purple-paraffin tail not segmented.** The
  colon has a small tail that is fainter and tinted with purple paraffin; current
  segmentation does not capture it, so block and slide masks disagree. The
  archived paraffin-relative faint block-mask adapter (see Known Failed, 2026-07-01)
  was the attempt to recover exactly this and was shelved for admitting ambiguous
  halo/background — no accepted fix yet.
- **set_009 (lung) — ambiguously shaped ovoid.** Single near-featureless ovoid
  blob; low outline entropy, so IoU and point_layout both hover near zero margin.
  Expected un-separable, same class as the single-blob lung ceiling.

## Next-Step Principle

Every new experiment should test one clear hypothesis, use dev/hard cases only,
write outputs to a named experiment or diagnostic folder, and end with a short
summary of whether the hypothesis survived.
