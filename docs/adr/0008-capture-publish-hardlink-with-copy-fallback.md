---
status: accepted
date: 2026-07-10
---

# Capture publication hard-links when possible, copies across filesystems

## Context

`CaptureStore.publish` must turn a source PNG into a collision-safe final name
under `output_dir` without overwriting existing files. Today it always
byte-copies into a staging file, then hard-links staging → final. That is safe
when source and destination are on different filesystems (outbox ingest,
`feed_captures`), but it pays a full 4056×3040 PNG copy on the common Pi camera
path where pending and published share one capture root.

Alternatives considered: always copy (simple, slow); split APIs for camera vs
outbox (easy to misuse); hard-link only and reject cross-FS sources (breaks
real callers).

## Decision

Keep a single `publish()` path. Prefer `os.link(source, final)` after one
validation decode. On `FileExistsError`, advance the counter and retry. On
cross-filesystem failure (`EXDEV`), fall back to copy-into-`output_dir` then
link. After a successful same-FS hard-link, deleting the pending source name
must leave the final name intact (two directory entries, one inode).

## Consequences

- Camera pending→published avoids a full-file copy when mounts match.
- Outbox / feed tools keep working across mounts via the fallback.
- Tests must cover collision safety, pending unlink after hard-link, and an
  `EXDEV` (or equivalent) fallback path.
- Callers must not assume the source file bytes are uniquely owned after
  publish until they unlink their own path.
