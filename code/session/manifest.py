"""Manifest loading for v2 claimed-pair verification.

CSV rows are asserted block↔slide claims. Required: block_path, slide_path.
Optional claim_id (auto-generated). One block may appear in multiple rows.

Code map
--------
ManifestValidationError
    Fatal validation: missing columns or duplicate claim_id.
ClaimRow
    claim_id, block_path, slide_path, missing_files tuple.
load_manifest(path, check_files=False)   ← pipeline entry
    Parse CSV; optional file-existence check (non-fatal).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

REQUIRED_COLUMNS = {"block_path", "slide_path"}


class ManifestValidationError(ValueError):
    """Raised for fatal manifest validation failures (missing columns, duplicate IDs)."""


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    block_path: str
    slide_path: str
    missing_files: tuple = ()


def load_manifest(
    path: str | Path,
    *,
    check_files: bool = False,
) -> List[ClaimRow]:
    """Load and validate claims from a manifest CSV.

    Args:
        path: Path to the manifest CSV.
        check_files: If True, check whether block_path and slide_path exist
            and record any missing paths in ClaimRow.missing_files.
            Missing files are non-fatal; the pipeline converts them to REVIEW.

    Raises:
        ManifestValidationError: for missing required columns or duplicate claim IDs.
    """
    path = Path(path)
    rows: List[ClaimRow] = []
    seen_ids: set[str] = set()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_cols:
            raise ManifestValidationError(
                f"Manifest missing required columns: {', '.join(sorted(missing_cols))}"
            )
        for i, row in enumerate(reader):
            claim_id = (row.get("claim_id") or "").strip() or f"claim_{i:04d}"
            if claim_id in seen_ids:
                raise ManifestValidationError(
                    f"Duplicate claim_id in manifest: {claim_id!r}"
                )
            seen_ids.add(claim_id)

            block_path = row["block_path"]
            slide_path = row["slide_path"]

            missing: list[str] = []
            if check_files:
                if not Path(block_path).exists():
                    missing.append(block_path)
                if not Path(slide_path).exists():
                    missing.append(slide_path)

            rows.append(ClaimRow(
                claim_id=claim_id,
                block_path=block_path,
                slide_path=slide_path,
                missing_files=tuple(missing),
            ))

    return rows
