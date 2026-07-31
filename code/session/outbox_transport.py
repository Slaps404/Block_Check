"""Outbound capture transport: Pi outbox publication and HTTP client (#201 slice 3)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from capture_storage import CaptureStore
from slide.qr import SlideQRResult
import store.wire as store_wire
from session.atomic_io import atomic_bytes as _atomic_bytes
from session.atomic_io import atomic_json as _atomic_json
from session.atomic_io import sha256 as _sha256
from session.workflow_types import (
    CaptureTransport,
    OutboxCapture,
    OutboxSlide,
    UploadReceipt,
)


def _slide_capture_id(
    captured_at: datetime, checksum: str, result: SlideQRResult,
) -> str:
    """Build a stable, human-pairable slide capture ID.

    The barcode claim is a filename aid only; the stored decode audit remains
    authoritative.  Unexpected/non-eight-digit claims are deliberately named
    ``unresolved`` rather than creating a misleading filename.
    """
    claim = result.block_id if (
        result.success and result.block_id is not None
        and re.fullmatch(r"\d{8}", result.block_id)
    ) else "unresolved"
    return f"slide_{claim}_{captured_at:%Y%m%dT%H%M%SZ}_{checksum[:12]}"


class PiOutbox:
    """Publish validated captures locally before any transport sees them."""

    _BLOCK_CAPTURE_PATTERN = re.compile(
        r"^(capture_[0-9]+_block_([0-9]{8})_([0-9]{8}T[0-9]{6}Z))\.png$"
    )

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._store = CaptureStore(self.directory)
        self._recover_published_captures()

    def _recover_published_captures(self) -> None:
        """Close the crash window between PNG publication and metadata commit."""
        pattern = re.compile(
            r"^capture_\d+_block_(\d{8})_(\d{8}T\d{6}Z)\.png$"
        )
        for path in self.directory.glob("capture_*_block_*.png"):
            metadata_path = path.with_suffix(".json")
            if metadata_path.exists():
                continue
            match = pattern.match(path.name)
            if match is None:
                continue
            captured_at = datetime.strptime(
                match.group(2), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc)
            _atomic_json(
                metadata_path,
                {
                    "capture_id": path.stem,
                    "block_id": match.group(1),
                    "checksum": _sha256(path),
                    "captured_at": captured_at.isoformat(),
                    "path": path.name,
                    "state": "pending",
                },
            )

    def publish_block(
        self, source: str | Path, block_id: str, captured_at: datetime, *,
        recapture: bool = False,
        profile: bool = False,
    ) -> OutboxCapture:
        record = self._store.publish(
            source, "block", block_id=block_id, captured_at=captured_at
        )
        capture_id = record.path.stem
        checksum = _sha256(record.path)
        metadata = {
            "capture_id": capture_id,
            "block_id": block_id,
            "checksum": checksum,
            "captured_at": record.captured_at.isoformat(),
            "path": record.path.name,
            "state": "pending",
            "recapture": recapture,
            "profile": profile,
        }
        _atomic_json(self.directory / f"{capture_id}.json", metadata)
        return OutboxCapture(
            capture_id, record.path, block_id, checksum, record.captured_at,
            recapture=recapture,
            profile=profile,
        )

    def publish_slide(
        self, source: str | Path, captured_at: datetime, *,
        result: SlideQRResult, duration_ms: float,
        profile: bool = False,
        supersedes: str | None = None,
    ) -> OutboxSlide:
        """Durably publish a slide and decode audit before any HTTP attempt.

        ``supersedes`` (#256 follow-up) is the armed-recapture tag: when set,
        it names the existing Hybrid slide capture this NEW capture should
        replace once replayed (see ``replay_slides``). Baked into the durable
        metadata at publish time so it survives a crash before replay, same
        as ``profile`` above.
        """
        if captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        body = Path(source).read_bytes()
        checksum = hashlib.sha256(body).hexdigest()
        utc = captured_at.astimezone(timezone.utc)
        capture_id = _slide_capture_id(utc, checksum, result)
        directory = self.directory / "slides"
        directory.mkdir(exist_ok=True)
        path = directory / f"{capture_id}.png"
        _atomic_bytes(path, body)
        _atomic_json(directory / f"{capture_id}.json", {
            "capture_id": capture_id, "path": path.name, "checksum": checksum,
            "captured_at": utc.isoformat(), "result": store_wire.encode(result),
            "duration_ms": float(duration_ms), "state": "pending",
            # #258: thread the --profile gate across the durable outbox
            # entry, mirroring publish_block's own "profile" metadata key,
            # so a crash between this write and replay_slides still recovers
            # the flag from disk rather than silently dropping it to False.
            "profile": profile,
            "supersedes": supersedes,
        })
        return OutboxSlide(
            capture_id, path, utc, result, float(duration_ms), profile=profile,
            supersedes=supersedes,
        )

    def pending_slides(self) -> tuple[OutboxSlide, ...]:
        entries, _ = self._read_slide_entries()
        return entries

    def invalid_slide_entries(self) -> tuple[str, ...]:
        _, invalid = self._read_slide_entries()
        return invalid

    def _read_slide_entries(
        self,
    ) -> tuple[tuple[OutboxSlide, ...], tuple[str, ...]]:
        directory = self.directory / "slides"
        entries: list[OutboxSlide] = []
        invalid: list[str] = []
        for metadata_path in directory.glob("slide_*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                state = metadata.get("state", "pending")
                if state not in {"pending", "acknowledged"}:
                    raise ValueError("invalid slide outbox state")
                if state == "acknowledged":
                    continue
                capture_id = str(metadata["capture_id"])
                path = (directory / str(metadata["path"])).resolve()
                if (
                    path.parent != directory.resolve()
                    or metadata_path.name != f"{capture_id}.json"
                    or path.name != f"{capture_id}.png"
                    or not path.is_file()
                    or _sha256(path) != str(metadata["checksum"])
                ):
                    raise ValueError("slide outbox metadata or capture is invalid")
                supersedes = metadata.get("supersedes")
                entries.append(OutboxSlide(
                    capture_id, path,
                    datetime.fromisoformat(metadata["captured_at"]),
                    store_wire.decode(SlideQRResult, metadata["result"]),
                    float(metadata["duration_ms"]),
                    profile=bool(metadata.get("profile", False)),
                    supersedes=str(supersedes) if supersedes is not None else None,
                ))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(metadata_path.stem)
        return (
            tuple(sorted(entries, key=lambda entry: entry.capture_id)),
            tuple(sorted(invalid)),
        )

    def replay_slides(self, session_number: int, store: object) -> tuple[str, ...]:
        """Replay durable slide captures FIFO; leave failures pending.

        A slide durably tagged ``supersedes`` (#256 follow-up -- an
        operator-armed recapture, see ``SessionWorkflow.
        arm_hybrid_recapture``) replays through ``store.
        recapture_hybrid_slide`` instead of the ordinary ``store.
        record_slide_capture``, naming the superseded capture id. A
        recapture the store rejects (identity mismatch, or no such durable
        capture) is NOT retried and NOT silently dropped: it is acknowledged
        like any other decided outcome, and the rejection reason is recorded
        as a durable, operator-visible event via the ALREADY-whitelisted
        ``store.record_event`` (no new store method), guarded by its own
        stable request id so a replay after a crash cannot double-log it.
        """
        acknowledged: list[str] = []
        for slide in self.pending_slides():
            try:
                if slide.supersedes is not None:
                    outcome = store.recapture_hybrid_slide(
                        session_number, slide.supersedes, slide.path,
                        captured_at=slide.captured_at, result=slide.result,
                        duration_ms=slide.duration_ms,
                        request_id=f"slide:{slide.capture_id}",
                    )
                    if not outcome.accepted:
                        store.record_event(
                            session_number, "hybrid_recapture_rejected",
                            f"Recapture for {slide.supersedes} was not "
                            f"accepted: {outcome.message}",
                            request_id=f"slide:{slide.capture_id}:rejected",
                        )
                else:
                    store.record_slide_capture(
                        session_number, slide.path, captured_at=slide.captured_at,
                        result=slide.result, duration_ms=slide.duration_ms,
                        request_id=f"slide:{slide.capture_id}",
                        profile=slide.profile,
                    )
            except (OSError, RuntimeError, URLError):
                break
            metadata_path = slide.path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["state"] = "acknowledged"
            _atomic_json(metadata_path, metadata)
            acknowledged.append(slide.capture_id)
        return tuple(acknowledged)

    def entries(self) -> tuple[OutboxCapture, ...]:
        """Reconstruct only entries whose metadata and path fully agree."""
        entries, _ = self._read_entries()
        return entries

    def invalid_entries(self) -> tuple[str, ...]:
        """Return corrupt entry IDs so they remain visible and block finalization."""
        _, invalid = self._read_entries()
        return invalid

    def _read_entries(
        self,
    ) -> tuple[tuple[OutboxCapture, ...], tuple[str, ...]]:
        entries: list[OutboxCapture] = []
        invalid: list[str] = []
        root = self.directory.resolve()
        for metadata_path in self.directory.glob("capture_*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                capture_id = str(metadata["capture_id"])
                path_value = str(metadata["path"])
                path = (self.directory / path_value).resolve()
                match = self._BLOCK_CAPTURE_PATTERN.fullmatch(Path(path_value).name)
                captured_at = datetime.fromisoformat(metadata["captured_at"])
                if (
                    path.parent != root
                    or Path(path_value).name != path_value
                    or match is None
                    or metadata_path.name != f"{capture_id}.json"
                    or match.group(1) != capture_id
                    or match.group(2) != str(metadata["block_id"])
                    or captured_at.tzinfo is None
                    or captured_at.astimezone(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ) != match.group(3)
                    or metadata.get("state", "pending")
                    not in {"pending", "acknowledged"}
                ):
                    raise ValueError("outbox metadata does not match capture naming")
                checksum = str(metadata["checksum"])
                if not path.is_file() or _sha256(path) != checksum:
                    raise ValueError("outbox capture is missing or corrupt")
                entries.append(
                    OutboxCapture(
                        capture_id,
                        path,
                        str(metadata["block_id"]),
                        checksum,
                        captured_at,
                        str(metadata.get("state", "pending")),
                        bool(metadata.get("recapture", False)),
                        bool(metadata.get("profile", False)),
                    )
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(metadata_path.stem)
        return (
            tuple(sorted(entries, key=lambda entry: entry.capture_id)),
            tuple(sorted(invalid)),
        )

    def pending(self) -> tuple[OutboxCapture, ...]:
        return tuple(entry for entry in self.entries() if entry.state == "pending")

    def acknowledge(self, receipt: UploadReceipt) -> None:
        metadata_path = self.directory / f"{receipt.capture_id}.json"
        entry = next(
            (item for item in self.entries() if item.capture_id == receipt.capture_id),
            None,
        )
        if entry is None:
            raise ValueError("receipt does not match a durable outbox entry")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if entry.checksum != receipt.checksum or not receipt.acknowledged:
            raise ValueError("receipt does not match the durable capture")
        # This transition is monotonic: stale status data cannot turn an
        # acknowledged entry back into pending work.
        if metadata.get("state", "pending") == "acknowledged":
            return
        metadata["state"] = "acknowledged"
        metadata["receipt"] = asdict(receipt)
        _atomic_json(metadata_path, metadata)

    def delete_acknowledged(self) -> tuple[str, ...]:
        """Remove only durably-acknowledged captures; idempotent and safe to retry."""
        deleted = []
        for entry in self.entries():
            if entry.state != "acknowledged":
                continue
            entry.path.unlink(missing_ok=True)
            entry.path.with_suffix(".json").unlink(missing_ok=True)
            deleted.append(entry.capture_id)
        slide_directory = self.directory / "slides"
        for metadata_path in slide_directory.glob("slide_*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("state") != "acknowledged":
                    continue
                capture_id = str(metadata["capture_id"])
                (slide_directory / f"{capture_id}.png").unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                deleted.append(capture_id)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(deleted)

    def replay(
        self, session_number: int, transport: CaptureTransport
    ) -> tuple[UploadReceipt, ...]:
        """Poll once, then replay the pending FIFO until connectivity fails."""
        try:
            transport.status(session_number)
        except (OSError, RuntimeError, URLError):
            return ()
        receipts: list[UploadReceipt] = []
        for capture in self.pending():
            try:
                receipt = transport.upload(session_number, capture)
            except (OSError, RuntimeError, URLError):
                break
            self.acknowledge(receipt)
            receipts.append(receipt)
        return tuple(receipts)


def default_debug_snap_dir() -> Path:
    """Laptop folder shared with Desktop\\capture.bat (pi_captures)."""
    return Path.home() / "Desktop" / "pi_captures"


def open_saved_image(path: Path) -> None:
    """Open a saved image with the OS default viewer (best-effort)."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
    except OSError:
        # Viewer missing / association broken — snap file is still on disk.
        return


def save_debug_snap(
    body: bytes, *, dest_dir: Path | None = None, open_image: bool = False
) -> Path:
    """Write one debug PNG under dest_dir; return the absolute path."""
    dest = Path(dest_dir) if dest_dir is not None else default_debug_snap_dir()
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = dest / f"snap_{stamp}_{uuid4().hex[:8]}.png"
    path.write_bytes(body)
    if open_image:
        open_saved_image(path)
    return path.resolve()


class HttpCaptureClient:
    """HTTP transport adapter used by the Pi-side workflow."""

    def __init__(self, base_url: str, *, timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self, session_number: int) -> Mapping[str, object]:
        request = Request(f"{self.base_url}/sessions/{session_number}/status")
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def upload(self, session_number: int, capture: OutboxCapture) -> UploadReceipt:
        request = Request(
            f"{self.base_url}/sessions/{session_number}/captures",
            data=capture.path.read_bytes(),
            method="POST",
            headers={
                "Content-Type": "image/png",
                "X-Capture-Id": capture.capture_id,
                "X-Block-Id": capture.block_id,
                "X-Checksum-Sha256": capture.checksum,
                "X-Captured-At": capture.captured_at.isoformat(),
                "X-Block-Recapture": "true" if capture.recapture else "false",
                "X-Profile": "true" if capture.profile else "false",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"capture upload rejected: {detail}") from exc
        return UploadReceipt(
            payload["capture_id"], payload["acknowledged"], payload["checksum"]
        )

    def debug_snap(self, path: Path) -> str:
        """POST a debug still to /debug/snap; return the laptop saved path."""
        request = Request(
            f"{self.base_url}/debug/snap",
            data=Path(path).read_bytes(),
            method="POST",
            headers={"Content-Type": "image/png"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"debug snap rejected: {detail}") from exc
        saved = payload.get("path")
        if not saved:
            raise RuntimeError("debug snap response missing path")
        return str(saved)

    def upload_profile_curve(self, session_number: int, path: str | Path) -> str:
        """POST the Pi-local motion curve into the session bundle (#172).

        Mirrors `debug_snap`'s raw-bytes upload idiom; called once from
        `PiCaptureRuntime.end_session` after the local writer is flushed.
        """
        request = Request(
            f"{self.base_url}/sessions/{session_number}/profile-curve",
            data=Path(path).read_bytes(),
            method="POST",
            headers={"Content-Type": "text/csv"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"profile curve upload rejected: {detail}") from exc
        saved = payload.get("path")
        if not saved:
            raise RuntimeError("profile curve upload response missing path")
        return str(saved)

    def upload_profile_config(self, session_number: int, path: str | Path) -> str:
        """POST profiled settling controls into the session bundle."""
        request = Request(
            f"{self.base_url}/sessions/{session_number}/profile-config",
            data=Path(path).read_bytes(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"profile config upload rejected: {detail}") from exc
        saved = payload.get("path")
        if not saved:
            raise RuntimeError("profile config upload response missing path")
        return str(saved)
