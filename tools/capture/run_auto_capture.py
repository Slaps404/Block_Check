"""Launch the Raspberry Pi automatic capture console."""
from pathlib import Path
import sys

_CODE_DIR = Path(__file__).resolve().parents[2] / "code"
_CAPTURE_DIR = _CODE_DIR / "capture"
for _path in (_CODE_DIR, _CAPTURE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from capture_console import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
