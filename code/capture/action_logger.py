"""Append-only timestamped workflow action log for Pi session operators."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class ActionLogger:
    """Write milestone lines to a file and an operator-visible sink."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_number: int,
        print_sink: Callable[[str], object] = print,
    ) -> None:
        self.path = Path(path)
        self.session_number = session_number
        self.print_sink = print_sink

    def log(self, event: str, **fields: object) -> None:
        line = self._format_line(event, **fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.print_sink(line)

    def _format_line(self, event: str, **fields: object) -> str:
        tokens = [f"session={self.session_number}"]
        if event:
            tokens.append(f"event={event}")
        for key, value in fields.items():
            tokens.append(f"{key}={value}")
        return f"{self._timestamp()}  " + "  ".join(tokens)

    @staticmethod
    def _timestamp() -> str:
        now = datetime.now(timezone.utc)
        millis = now.microsecond // 1000
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"
