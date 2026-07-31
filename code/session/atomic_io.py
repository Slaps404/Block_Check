"""Atomic filesystem writes shared by session workflow adapters (#201 slice 2)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_name(path: Path) -> Path:
    """A per-call-unique staging path beside ``path`` (#250 review F4).

    A FIXED temp name (e.g. ``.name.tmp``) is safe only when at most one
    writer ever targets ``path`` at a time. `ProcessingStore.freeze_hybrid_
    pool`'s archive write proved that assumption false: a retried duplicate
    RPC request can run concurrently with the original attempt (both well
    past the client's 10s timeout), and two writers sharing one temp name can
    interleave -- writer B still writing into the inode writer A already
    ``os.replace``'d into place, corrupting the published file. Every caller
    of `atomic_bytes`/`atomic_json` gets this for free; none of them assume
    or rely on the exact temp filename (grepped `tests/` and `code/` before
    making this change).
    """
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def atomic_bytes(path: Path, body: bytes) -> None:
    temporary = _temporary_name(path)
    try:
        with temporary.open("wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = _temporary_name(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
