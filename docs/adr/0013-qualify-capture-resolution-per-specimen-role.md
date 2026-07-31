---
status: accepted
date: 2026-07-20
---

# Qualify capture resolution per specimen role

Slide and block preprocessing use different segmentation algorithms, so a
resolution that preserves slide masks does not qualify block masks. Capture
dimensions and spatial-scale controls will therefore be configured and
qualified per specimen role: blocks remain 4056x3040 for now, while
half-resolution slides are the first promotion candidate and quarter-resolution
slides remain experimental. A role may move to a lower resolution only after
its own Segmentation Qualification.

Capture validation must compare typed still dimensions with the configured
dimensions for that role rather than a hardcoded shared size. Lower-resolution
camera configurations must preserve the full sensor field of view. Relative
fractions remain relative; absolute pixel lengths and areas scale explicitly;
intensity thresholds and the scorer's fixed canonical grid are not blindly
scaled. QR decoding remains a separate qualification concern.

Visual Segmentation QA is the primary resolution-qualification gate. Reviewers
must compare the raw capture and overlays and confirm that meaningful tissue
fragments are preserved without new background leakage. Component preservation,
area drift, centroid shift, and IoU are guardrails used to rank suspicious
cases, not universal pass/fail substitutes for visual correctness. In
particular, a full-resolution mask is not ground truth when it visibly includes
capture grain, slide markings, or background texture as tissue.

For sparse masks, the default numeric review contract is: preserve every
meaningful tissue fragment, keep absolute area drift within 5%, and keep
centroid shift within 2 pixels on the 256 grid. A candidate outside those
guardrails needs explicit visual adjudication; a candidate inside them still
fails if its overlay loses tissue or admits background.

Qualification results must be stratified by Capture Profile Provenance. The
production acceptance set must use the current fixed-AWB, role-locked, native
PNG profile with the requested spatial smoothing mode. Captures from earlier
AWB-broken or unsmoothed profiles remain useful adversarial controls, but they
must not be pooled into the representative pass rate or drive thresholds for
the current camera contract.
