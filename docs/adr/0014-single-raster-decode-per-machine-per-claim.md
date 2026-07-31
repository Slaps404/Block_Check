# Single raster decode per machine per slide claim

## Status

accepted

## Context

The slide-claim path re-decoded the same ~12 MP capture PNG several times per
claim. On the Pi, `capture_slide` decoded the still for QR identity even when a
keyboard-scanner payload made that decode unnecessary. On the processing
computer, `resolve_claim` decoded the slide once for Specimen Preprocessing and
`_finalize_claim` decoded the *same* PNG a second time purely to render the claim
QC sheet (`perf_audit.md` §5). Each decode is a fresh PNG decompression plus a
large BGR allocation (~70-80 ms on the dev machine).

## Decision

Decode each slide capture **once per machine per claim**.

- **Pi:** decode the raster only when a QR Search is actually needed — i.e. only
  on the camera-QR identity path. When a keyboard-scanner payload is present,
  skip the decode entirely (the pixels are never read).
- **Processing computer:** `resolve_claim` performs one Raster Decode and threads
  that BGR frame through Specimen Preprocessing and claim QC. The re-decode in
  `_finalize_claim` is removed. The `if frame is None` guard and its
  `PreparationFailure(role="slide", reason="could not read image: …")` are
  relocated alongside the moved decode so the fail-closed verdict is byte-identical.

The frame is owned by `resolve_claim` as a local value, released at function
scope. No capture-scoped value object is introduced.

## Considered options

- **A — `resolve_claim` owns the decode; slide-prep seam becomes frame-in.**
  Chosen. Clean ownership: prep and QC are pure consumers of one frame; the 36 MB
  array dies when the function returns. The array-taking entry points
  (`prepare_specimen_from_image`, `decode_slide_identity`) already exist.
- **B — slide prep keeps the path but *returns* the frame it decoded.** Rejected:
  staples the 36 MB frame onto `PreparedSpecimen`, which then flows onward into
  scoring code that has no use for it and blurs when the frame is safe to release.

## Consequences

- The injectable `slide_preprocessor` seam flips from `Callable[[Path], …]` to
  `Callable[[np.ndarray], …]`; its default and the injected-callable test fakes
  are updated. The work-order N² path threads the same per-capture frame. The
  offline `pipeline.py` path is unchanged (separate seam, out of scope).
- The camera-QR path deliberately still decodes **twice** (once Pi for QR Search,
  once main for the claim) because a decoded frame cannot cross the Pi→main wire —
  it re-serializes to PNG. Collapsing that to one decode requires hoisting the QR
  Search onto the processing computer, which is issue #163's job. This decision is
  the socket #163 plugs into: once QR runs on main, the single `resolve_claim`
  frame also feeds identity, and the camera path drops to one decode with no
  further rework here.
