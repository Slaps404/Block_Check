"""ADR identifiers must be unique so numeric citations are unambiguous (#198)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ADR_DIR = Path(__file__).resolve().parent.parent / "docs" / "adr"
NUMBER_RE = re.compile(r"^(\d{4})-")


def _adr_numbers():
    numbers = []
    for path in ADR_DIR.glob("*.md"):
        match = NUMBER_RE.match(path.name)
        assert match, f"ADR filename missing numeric prefix: {path.name}"
        numbers.append((match.group(1), path.name))
    return numbers


def test_adr_filename_numbers_are_unique():
    numbers = _adr_numbers()
    counts = Counter(number for number, _ in numbers)
    duplicates = {number: [name for n, name in numbers if n == number]
                  for number, count in counts.items() if count > 1}
    assert not duplicates, f"Duplicate ADR numbers: {duplicates}"
