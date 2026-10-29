"""Launch the Pi session; implementation lives in app.pi_session."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "code", ROOT / "code" / "capture"):
    sys.path.insert(0, str(directory))

from app.pi_session import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
