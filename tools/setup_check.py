"""
setup_check.py — pre-flight verification for the slide QR/DataMatrix
decoding stack.

zxing-cpp is the primary slide decoder and cv2 (already required for the
rest of the pipeline) is the fallback. This script runs both imports,
prints install guidance on failure, and exits non-zero. Intended to run
BEFORE the Phase 3.5 pipeline so a missing decoder surfaces up front,
not mid-batch.

Usage:
    python tools/setup_check.py
"""

from __future__ import annotations

import sys
from typing import Callable

ZXING_GUIDANCE = """\
Install steps:
  pip install zxing-cpp"""

CV2_GUIDANCE = """\
Install steps:
  pip install opencv-python"""

DECODERS = (
    ("zxingcpp", ZXING_GUIDANCE),
    ("cv2", CV2_GUIDANCE),
)


def _try_import(module_name: str) -> tuple[bool, str]:
    """Attempt the import and return (success, error_message)."""
    try:
        __import__(module_name)
        return True, ""
    except Exception as exc:    # pylint: disable=broad-except
        return False, f"{type(exc).__name__}: {exc}"


def check_dependencies(
    importer: Callable[[str], tuple[bool, str]] = _try_import,
) -> int:
    """Run the decoder check. Returns 0 on success, non-zero on failure.

    `importer` is exposed as an arg so tests can simulate a missing
    decoder without actually uninstalling it.
    """
    print("Slide decoder setup check")
    failures: list[tuple[str, str]] = []
    for mod, guidance in DECODERS:
        ok, msg = importer(mod)
        if ok:
            print(f"  [OK] {mod}")
        else:
            print(f"  [FAIL] {mod}: {msg}")
            failures.append((mod, guidance))

    if failures:
        print()
        print("Missing or broken decoders: " + ", ".join(mod for mod, _ in failures))
        for mod, guidance in failures:
            print()
            print(f"{mod}:")
            print(guidance)
        print()
        print("Re-run this script after installing the missing decoders.")
        return 1

    print("All slide decoders available.")
    return 0


if __name__ == "__main__":
    sys.exit(check_dependencies())
