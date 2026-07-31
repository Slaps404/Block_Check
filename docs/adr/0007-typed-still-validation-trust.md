---
status: accepted
date: 2026-07-10
---

# Session trusts typed still validation, never free-form metadata

## Context

Publishing a full-resolution capture currently decodes the PNG up to four times:
three times inside `CaptureStore.publish` (source, staging copy, final hard
link) and once again in `CaptureSession._valid_still` when accepting the result.
The fourth decode is redundant on the production `CaptureController` path, but
removing it without a clear trust boundary is unsafe: `CaptureResult.metadata`
is already a free-form bag for camera settings, ROI, and block identity, so a
caller could invent dimension keys and silently bypass fail-closed checks.

Alternatives considered:

- **Reserved metadata keys** (`validated_by`, `validated_width`, …) — easy to
  wire, but easy to spoof and surprising to future readers.
- **Keep the fourth decode forever** — simplest, but leaves measurable latency
  on every accepted capture.
- **Typed validation field** set only by `CaptureStore` / `CaptureController`.

## Decision

Carry store-proven still facts as a typed field (e.g. `ValidatedStill` with
width, height, and format/suffix) on `CaptureRecord` and `CaptureResult`.
`CaptureSession.accept_capture_result` may skip re-decode only when that typed
field is present and reports 4056×3040 PNG. If the field is absent, keep the
existing `_valid_still` OpenCV decode so direct unit-test callers and any
non-store path remain fail-closed. Free-form `metadata` must never authorize
dimension trust.

## Consequences

- Publication optimizations can return validation facts once and reuse them.
- Spoofed metadata cannot bypass still checks.
- Session unit tests that build `CaptureResult.success(path)` without a store
  still exercise the decode fallback.
- Downstream code must treat missing typed validation as “untrusted,” not as
  “already OK.”
