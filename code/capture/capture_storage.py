"""Durable, collision-safe publication for validated specimen captures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from uuid import uuid4

import cv2

from constants import CAPTURE_DIMENSIONS


class PublicationError(RuntimeError):
    """A candidate still could not be safely validated and published."""


@dataclass(frozen=True)
class ValidatedStill:
    width: int
    height: int
    format: str


@dataclass(frozen=True)
class CaptureRecord:
    counter: int
    path: Path
    role: str
    captured_at: datetime
    block_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validated: ValidatedStill | None = None


class CaptureStore:
    _COUNTER_FILE = ".capture_counter"
    _CAPTURE_PATTERN = re.compile(r"^capture_(\d+)_")

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        source: str | Path,
        role: str,
        *,
        captured_at: datetime,
        block_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CaptureRecord:
        source_path = Path(source)
        if role == "slide":
            self._downscale_oversized_slide(source_path)
        validated = self._validate_still(source_path, role)
        self._validate_identity(role, block_id)
        timestamp = self._utc_timestamp(captured_at)

        counter = self._next_counter()
        staging_path: Path | None = None
        try:
            try:
                final_path, counter = self._link_with_collision_retry(
                    source_path, counter, role, block_id, timestamp
                )
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                staging_path = self.output_dir / f".capture-{uuid4().hex}.tmp"
                shutil.copyfile(source_path, staging_path)
                final_path, counter = self._link_with_collision_retry(
                    staging_path, counter, role, block_id, timestamp
                )

            self._write_counter(counter)
        except PublicationError:
            raise
        except (OSError, ValueError) as exc:
            raise PublicationError(f"capture publication failed: {exc}") from exc
        finally:
            if staging_path is not None:
                staging_path.unlink(missing_ok=True)

        return CaptureRecord(
            counter=counter,
            path=final_path,
            role=role,
            block_id=block_id,
            captured_at=captured_at.astimezone(timezone.utc),
            metadata=dict(metadata or {}),
            validated=validated,
        )

    def _link_with_collision_retry(
        self,
        link_source: Path,
        counter: int,
        role: str,
        block_id: str | None,
        timestamp: str,
    ) -> tuple[Path, int]:
        while True:
            final_path = self.output_dir / self._filename(
                counter, role, block_id, timestamp
            )
            try:
                os.link(link_source, final_path)
                return final_path, counter
            except FileExistsError:
                counter += 1

    def _next_counter(self) -> int:
        persisted = 0
        counter_path = self.output_dir / self._COUNTER_FILE
        if counter_path.is_file():
            try:
                persisted = int(counter_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError) as exc:
                raise PublicationError(f"invalid persistent counter: {exc}") from exc

        published = 0
        for path in self.output_dir.glob("capture_*.png"):
            match = self._CAPTURE_PATTERN.match(path.name)
            if match:
                published = max(published, int(match.group(1)))
        return max(persisted, published) + 1

    def _write_counter(self, counter: int) -> None:
        temporary = self.output_dir / f".{self._COUNTER_FILE}-{uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="ascii") as stream:
                stream.write(f"{counter}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output_dir / self._COUNTER_FILE)
        finally:
            temporary.unlink(missing_ok=True)

    _PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

    @classmethod
    def _downscale_oversized_slide(cls, source_path: Path) -> None:
        """Shrink an over-large slide PNG in place before validation.

        Store-time-only shrink: the native sensor contract is unchanged. This
        exists purely to cut the
        stored/transmitted slide PNG down to ``CAPTURE_DIMENSIONS["slide"]``
        so downstream transfer/decode/memory cost drops with it. Dimensions
        are peeked straight from the PNG header (no pixel decode) so an
        already-correct-size slide (the current production config, and every
        block capture) costs nothing extra: ``publish`` still decodes the
        source exactly once in that case.
        """
        dims = CAPTURE_DIMENSIONS.get("slide")
        if dims is None:
            return
        exp_w, exp_h = dims
        peeked = cls._peek_png_dimensions(source_path)
        if peeked is None:
            return
        width, height = peeked
        if width <= exp_w and height <= exp_h:
            return
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        resized = cv2.resize(image, (exp_w, exp_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(source_path), resized)

    @classmethod
    def _peek_png_dimensions(cls, path: Path) -> tuple[int, int] | None:
        """Read a PNG's width/height from its IHDR chunk, without decoding.

        Returns ``None`` for anything not readable as a PNG header; callers
        fall through to the existing ``_validate_still`` error path, which
        already reports missing/unreadable/wrong-dimension captures.
        """
        try:
            with path.open("rb") as stream:
                header = stream.read(24)
        except OSError:
            return None
        if len(header) < 24 or header[:8] != cls._PNG_SIGNATURE:
            return None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height

    @classmethod
    def _validate_still(
        cls, path: Path, role: str, *, require_png_suffix: bool = True
    ) -> ValidatedStill:
        if require_png_suffix and path.suffix.lower() != ".png":
            raise PublicationError("capture must use the .png extension")
        if not path.is_file():
            raise PublicationError(f"capture file does not exist: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PublicationError(f"capture cannot be reopened: {path}")
        dims = CAPTURE_DIMENSIONS.get(role)
        if dims is None:
            raise PublicationError(f"unknown capture role: {role!r}")
        exp_w, exp_h = dims
        if image.shape[:2] != (exp_h, exp_w):
            raise PublicationError(
                f"capture dimensions for {role} must be {exp_w}x{exp_h}; "
                f"got {image.shape[1]}x{image.shape[0]}"
            )
        del image
        return ValidatedStill(
            width=exp_w,
            height=exp_h,
            format=".png",
        )

    @staticmethod
    def _validate_identity(role: str, block_id: str | None) -> None:
        if role not in {"block", "slide"}:
            raise PublicationError("role must be 'block' or 'slide'")
        if role == "block" and not (
            block_id is not None and len(block_id) == 8 and block_id.isascii()
            and block_id.isdigit()
        ):
            raise PublicationError("block captures require an eight-digit block ID")
        if role == "slide" and block_id is not None:
            raise PublicationError("slide captures cannot carry a block ID")

    @staticmethod
    def _utc_timestamp(captured_at: datetime) -> str:
        if captured_at.tzinfo is None:
            raise PublicationError("captured_at must be timezone-aware")
        return captured_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _filename(
        counter: int, role: str, block_id: str | None, timestamp: str
    ) -> str:
        identity = f"{block_id}_" if block_id is not None else ""
        return f"capture_{counter:06d}_{role}_{identity}{timestamp}.png"
