"""Launch the processing-computer receiver; implementation lives in app.receiver."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "code", ROOT / "code" / "capture"):
    sys.path.insert(0, str(directory))

from app.receiver import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
