## Goal

What specific PiBlockCheck milestone or issue does this PR solve?

## Scope

Files intentionally changed:

-

Files intentionally not changed:

-

## Risk Level

Check one — how intensely should this be reviewed?

- [ ] Low — docs, comments, tooling, no pipeline/scoring behavior change
- [ ] Medium — touches one pipeline stage, behavior change is scoped and testable
- [ ] High — touches scoring/thresholds/routing, or changes behavior across multiple pipeline stages

If Medium/High, what's the blast radius if this is wrong?

-

## Test / Validation

How did you test this?

-

## PiBlockCheck Pipeline Stage

Check one:

- [ ] Image loading
- [ ] Preprocessing
- [ ] Barcode / QR / Data Matrix decoding
- [ ] Block-label to slide-label matching
- [ ] Tissue segmentation / silhouette detection
- [ ] Multi-section detection
- [ ] Feature comparison / similarity scoring
- [ ] Confidence scoring
- [ ] Manual review outputs
- [ ] Tests / tooling / docs

## Real-Image Risk

What could fail on real histology images?

Consider lighting, rotation, scale, focus, labels, grid/lattice artifacts, background noise, and multiple tissue sections.

## AI Review Requested

Check only the reviewers that are useful for this PR:

- [ ] Scope creep / overengineering
- [ ] Tests / reproducibility
- [ ] Classical CV robustness
- [ ] Barcode / QR / Data Matrix robustness
- [ ] Pipeline boundaries

## Notes for AI Reviewers

- Keep feedback narrow and tied to this PR’s stated goal.
- Do not suggest broad rewrites.
- Do not suggest ML unless this PR explicitly involves ML.
- Prefer small, testable fixes over architecture changes.
- Every blocking concern should point to a specific file, line, behavior, or missing test.