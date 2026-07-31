---
status: accepted
---

# Retrain QuPath classifiers in Python, do not port serialized trees

Accepted 2026-07-17. Tracked by the spec issue "Config-driven QuPath-JSON to
retrain tool for block segmentation".

## Context

A QuPath pixel classifier (OpenCV RTrees serialized in a JSON export) segments
block tissue well: it rejects blurry deep tissue via Laplacian-of-Gaussian +
gradient-magnitude features and keeps the sharp top surface a slide is cut from.
We want that behavior inside the OpenCV pipeline. Two ways to get it there:

- **(a) Port** — load QuPath's already-trained trees and reproduce its feature
  computation bit-exactly so the trees see the inputs they were trained on.
- **(b) Retrain** — parse the JSON only as a *feature recipe* (channels, ops,
  sigmas, downsample factor, class map) and fit fresh `cv2.ml.RTrees` on
  features computed in Python.

## Decision

Retrain (option b). Parse the QuPath JSON as a feature recipe, fit fresh
RTrees on OpenCV-computed features, and never load QuPath's serialized trees.

## Consequences

- **Porting is brittle.** It demands bit-exact feature reproduction across two
  libraries (QuPath/ImageJ vs OpenCV) whose kernels differ in normalization,
  convolution-vs-correlation, border handling, and float rounding. Small
  numeric drift lands inputs in the wrong tree branches, so a faithfully copied
  tree misbehaves.
- **Retraining absorbs the numeric differences.** It only needs the same
  feature *family* at the same scales; it fits new trees on OpenCV's own feature
  values, so library-level kernel differences never have to be matched.
- **Cost:** training data (Groovy-exported annotations) must be available, and
  RTrees has randomness, so a fixed RNG seed is recorded in the recipe sidecar
  for reproducible reruns.
- **Enables a reusable tool.** Because the recipe is parsed rather than
  hard-coded, any QuPath classifier of the same feature family retrains by
  pointing the tool at a different JSON + annotation set — no code change.

## Reference

- Spec issue: Config-driven QuPath-JSON to retrain tool for block segmentation
  (drafted alongside this ADR).
