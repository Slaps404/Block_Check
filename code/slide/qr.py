"""Slide barcode decode for claimed-pair verification.

4-step pipeline: full-frame decode → locate+crop → enhancement variants →
label-region fallback. Never raises; failure → SlideQRResult(success=False).

Code map
--------
SlideQRResult
    success flag + parsed fields or error reason.
decode_slide_qr(bgr)   ← tools/identity entry
    Run full decode pipeline; return result dataclass.
parse_qr_payload(payload)
    Parse workorder_blockID_number_stain or legacy email format.
_engines, _try_variants, _locate_and_zoom, _label_rect_crop,
_try_crop_with_rotations, _try_decode_pipeline, _to_gray
    zxing/cv2 decode helpers and image variants.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded imports — degrade gracefully if engine is missing
# ---------------------------------------------------------------------------

try:
    import zxingcpp as _zxingcpp
    _HAS_ZXING = True
except Exception as _exc:
    _zxingcpp = None  # type: ignore[assignment]
    _HAS_ZXING = False
    log.info("zxing-cpp not available: %s", _exc)

try:
    from slide.label_mask import find_label_rect as _find_label_rect
    _HAS_LABEL_MASK = True
except Exception as _exc:
    _find_label_rect = None  # type: ignore[assignment]
    _HAS_LABEL_MASK = False
    log.info("slide_label_mask not available: %s", _exc)

# ---------------------------------------------------------------------------
# Module-level detector (cv2) — instantiated once
# ---------------------------------------------------------------------------

_det = cv2.QRCodeDetector()

# Rotation angles for the fallback sweep
_ROTATION_ANGLES = (90, 180, 270, 33, -33, 45, -45)

# Target minimum dimension for upscaled crops
_UPSCALE_TARGET = 600

# Operator-facing identity decode budget. Native decoder calls cannot be
# interrupted mid-call, so this deadline is checked between bounded routes.
DECODE_BUDGET_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodeCandidate:
    """One symbol payload returned by a replaceable decoder adapter."""

    engine: str
    symbology: str
    preprocessing: str
    payload: str


@dataclass(frozen=True)
class DecodeAttempt:
    """Auditable grammar decision for one decoded candidate."""

    engine: str
    symbology: str
    preprocessing: str
    payload: str
    accepted: bool
    reason: str


class SlideCodeDecoder(Protocol):
    """Replaceable boundary for one barcode decoding engine."""

    def decode(
        self, image: np.ndarray, preprocessing: str
    ) -> tuple[DecodeCandidate, ...]: ...


class ZxingSlideCodeDecoder:
    """Decode QR and Data Matrix symbols in one zxing scan."""

    def __init__(
        self,
        *,
        try_rotate: bool = True,
        try_downscale: bool = True,
        try_invert: bool = False,
        prefer_single: bool = False,
    ) -> None:
        self._try_rotate = try_rotate
        self._try_downscale = try_downscale
        self._try_invert = try_invert
        self._prefer_single = prefer_single

    def decode(
        self, image: np.ndarray, preprocessing: str
    ) -> tuple[DecodeCandidate, ...]:
        if not _HAS_ZXING:
            return ()
        options = {
            "formats": (
                _zxingcpp.BarcodeFormat.QRCode,
                _zxingcpp.BarcodeFormat.DataMatrix,
            ),
            "try_rotate": self._try_rotate,
            "try_downscale": self._try_downscale,
            "try_invert": self._try_invert,
        }
        results = ()
        if self._prefer_single:
            result = _zxingcpp.read_barcode(image, **options)
            if result is not None:
                parsed = parse_qr_payload(result.text)
                if parsed.get("success"):
                    results = (result,)
                else:
                    results = _zxingcpp.read_barcodes(image, **options)
        else:
            results = _zxingcpp.read_barcodes(image, **options)
        candidates = []
        for result in results:
            symbology = str(result.format).replace(" ", "")
            if symbology not in ("QRCode", "DataMatrix"):
                continue
            candidates.append(DecodeCandidate(
                "zxing", symbology, preprocessing, result.text
            ))
        return tuple(candidates)


class OpenCvQrDecoder:
    """QR-only fallback adapter backed by OpenCV."""

    def decode(
        self, image: np.ndarray, preprocessing: str
    ) -> tuple[DecodeCandidate, ...]:
        decoded, _, _ = _det.detectAndDecode(image)
        if not decoded:
            return ()
        return (DecodeCandidate("cv2", "QRCode", preprocessing, decoded),)


@dataclass(frozen=True)
class SlideQRResult:
    """Decoded + parsed result for a single slide image.

    On failure: success=False, reason explains why; all other fields None.
    On success: success=True, reason="ok", raw_payload set; parsed fields
    populated according to format ("current" or "legacy").
    """

    success: bool
    reason: str
    raw_payload: Optional[str]
    format: Optional[str]          # "current" | "legacy"
    block_id: Optional[str]
    slide_num: Optional[str]
    stain: Optional[str]
    work_order: Optional[str]
    email: Optional[str]
    genotype: Optional[str]
    engine: Optional[str]          # "zxing" | "cv2"
    preprocessing: Optional[str]
    symbology: Optional[str] = None
    attempts: tuple[DecodeAttempt, ...] = ()

    @property
    def lab_work_order(self) -> Optional[str]:
        """Laboratory work-order number decoded from a current slide QR.

        ``work_order`` remains the wire-compatible field name. This explicit
        accessor distinguishes the lab identifier from a local
        ``work_order_id`` capture/scoring-bracket record.
        """
        return self.work_order


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def parse_qr_payload(payload: str) -> dict:
    """Parse a slide QR payload into named fields. Never raises.

    Supports two formats, discriminated by '@' in field 0:
      current: work_order_block_id_slide_num_stain
      legacy:  email_genotype_slide_num_stain  (genotype may contain a space)

    Both formats have exactly 4 '_'-separated fields. The legacy genotype
    field may include an internal space (e.g. 'WT 3') — it is preserved
    verbatim.

    Returns a dict with 'success' bool and either parsed fields or 'reason'.
    """
    if not isinstance(payload, str) or payload == "":
        return {"success": False, "reason": "empty payload"}

    # Scanner/decoder prefixes may contain framing controls or leading spaces.
    # The contract permits removing them only at the start of the payload.
    normalized = payload.lstrip("".join(chr(value) for value in range(33)))
    if normalized == "":
        return {"success": False, "reason": "empty payload"}
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        return {"success": False, "reason": "control character in payload"}

    parts = normalized.split("_")
    if len(parts) < 4:
        return {
            "success": False,
            "reason": f"expected at least 4 _-separated fields, got {len(parts)}",
        }

    if not all(parts):
        return {"success": False, "reason": "empty field in payload"}

    if "@" in parts[0]:
        # Semantic fields are email, middle genotype, slide number, stain.
        # Joining the middle preserves historical genotypes containing '_'.
        email = parts[0]
        genotype = "_".join(parts[1:-2])
        slide_num, stain = parts[-2:]
        if not genotype:
            return {"success": False, "reason": "empty genotype in payload"}
        return {
            "success": True,
            "format": "legacy",
            "email": email,
            "genotype": genotype,
            "slide_num": slide_num,
            "stain": stain,
            "block_id": None,
            "work_order": None,
        }

    if len(parts) != 4:
        return {
            "success": False,
            "reason": f"current payload requires 4 fields, got {len(parts)}",
        }
    work_order, block_id, slide_num, stain = parts
    if not work_order.isascii() or not work_order.isdigit():
        return {"success": False, "reason": "work order must be numeric"}
    if not (len(block_id) == 8 and block_id.isascii() and block_id.isdigit()):
        return {
            "success": False,
            "reason": "block ID must contain exactly eight numeric digits",
        }
    return {
        "success": True,
        "format": "current",
        "work_order": work_order,
        "block_id": block_id,
        "slide_num": slide_num,
        "stain": stain,
        "email": None,
        "genotype": None,
    }


def scanner_identity(payload: str) -> SlideQRResult:
    """Resolve a handheld-scanner payload through the same grammar the
    camera path uses, tagged engine/preprocessing 'scanner'."""
    return select_slide_identity((
        DecodeCandidate("scanner", "QRCode", "scanner", payload),
    ))


def select_slide_identity(
    candidates: tuple[DecodeCandidate, ...] | list[DecodeCandidate],
) -> SlideQRResult:
    """Choose a production identity by grammar, never by decoder order."""
    attempts: list[DecodeAttempt] = []
    legacy: tuple[DecodeCandidate, dict] | None = None
    for candidate in candidates:
        parsed = parse_qr_payload(candidate.payload)
        accepted = bool(parsed.get("success") and parsed.get("format") == "current")
        if accepted:
            reason = "ok"
        elif parsed.get("success"):
            reason = "legacy payload cannot establish a production claim"
            if legacy is None:
                legacy = (candidate, parsed)
        else:
            reason = str(parsed.get("reason", "malformed payload"))
        attempts.append(DecodeAttempt(
            candidate.engine, candidate.symbology, candidate.preprocessing,
            candidate.payload, accepted, reason,
        ))
        if accepted:
            return SlideQRResult(
                True, "ok", candidate.payload, "current", parsed["block_id"],
                parsed["slide_num"], parsed["stain"], parsed["work_order"],
                None, None, candidate.engine, candidate.preprocessing,
                candidate.symbology, tuple(attempts),
            )

    if legacy is not None:
        candidate, parsed = legacy
        return SlideQRResult(
            False, "legacy payload cannot establish a production claim",
            candidate.payload, "legacy", None, parsed["slide_num"],
            parsed["stain"], None, parsed["email"], parsed["genotype"],
            candidate.engine, candidate.preprocessing, candidate.symbology,
            tuple(attempts),
        )
    reason = attempts[-1].reason if attempts else "no supported code decoded"
    return SlideQRResult(
        False, reason, None, None, None, None, None, None, None, None,
        None, None, None, tuple(attempts),
    )


def _variant_images(
    gray: np.ndarray, prefix: str = ""
) -> Iterable[tuple[str, np.ndarray]]:
    yield f"{prefix}raw", gray
    try:
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        yield f"{prefix}otsu", otsu
    except Exception:
        pass
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yield f"{prefix}clahe", clahe.apply(gray)
    except Exception:
        pass


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8)
    ).apply(gray)


def _route_variants(
    routes: dict[str, np.ndarray], name: str, image: np.ndarray
) -> None:
    """Add the measured raw/CLAHE 0/180 route family."""
    routes[f"{name}+raw"] = image
    routes[f"{name}+clahe"] = _clahe(image)
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    routes[f"{name}+rot180+raw"] = rotated
    routes[f"{name}+rot180+clahe"] = _clahe(rotated)


def _perspective_label_crop(
    bgr: np.ndarray, rect
) -> Optional[np.ndarray]:
    """Map the detected rotated label directly into one upright crop."""
    width = max(1, int(round(rect.size[0] * 1.08)))
    height = max(1, int(round(rect.size[1] * 1.08)))
    expanded = cv2.boxPoints((
        rect.center, (float(width), float(height)), rect.angle,
    )).astype(np.float32)
    sums = expanded.sum(axis=1)
    differences = np.diff(expanded, axis=1).reshape(-1)
    source = np.array([
        expanded[np.argmin(sums)],
        expanded[np.argmin(differences)],
        expanded[np.argmax(sums)],
        expanded[np.argmax(differences)],
    ], dtype=np.float32)
    destination = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        bgr, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    if warped.size == 0:
        return None
    gray = _to_gray(warped)
    if gray.shape[1] > gray.shape[0]:
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray


def _label_rect_found(bgr: np.ndarray) -> bool:
    """True when find_label_rect locates a frosted/yellow label band."""
    if not _HAS_LABEL_MASK:
        return False
    try:
        return bool(_find_label_rect(bgr).found)
    except Exception:
        return False


def _with_label_debug(
    bgr: np.ndarray, result: SlideQRResult
) -> SlideQRResult:
    """Append a label-vs-QR debug note on decode failure for operators.

    Distinguishes "we never found the frosted label rectangle" from
    "label was found but no QR payload decoded" so console triage is clear.
    """
    if result.success:
        return result
    if "label rectangle" in result.reason:
        return result
    # Payload parse / legacy failures already decoded a symbol — label status
    # is not the useful split for those cases.
    if result.raw_payload is not None:
        return result
    note = (
        "label rectangle found; QR not readable enough"
        if _label_rect_found(bgr)
        else "label rectangle not found"
    )
    return SlideQRResult(
        False,
        f"{result.reason} ({note})",
        result.raw_payload,
        result.format,
        result.block_id,
        result.slide_num,
        result.stain,
        result.work_order,
        result.email,
        result.genotype,
        result.engine,
        result.preprocessing,
        result.symbology,
        result.attempts,
    )


def _label_search_routes(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Build only label-relative routes with measured unique contribution."""
    if not _HAS_LABEL_MASK:
        return {}
    try:
        rect = _find_label_rect(bgr)
        if not rect.found:
            return {}
        height, width = bgr.shape[:2]
        points = rect.box_pts
        x0 = max(0, int(points[:, 0].min()))
        x1 = min(width, int(points[:, 0].max()))
        y0 = max(0, int(points[:, 1].min()))
        y1 = min(height, int(points[:, 1].max()))
        if x1 <= x0 or y1 <= y0:
            return {}
        whole = _to_gray(bgr[y0:y1, x0:x1])

        cx, cy = rect.center
        ax0 = max(0, int(cx - 0.14 * width))
        ax1 = min(width, int(cx + 0.08 * width))
        ay0 = max(0, int(cy - 0.23 * height))
        ay1 = min(height, int(cy + 0.06 * height))
        anchor = _to_gray(bgr[ay0:ay1, ax0:ax1])

        by0 = max(0, int(cy - 0.42 * height))
        by1 = min(height, int(cy + 0.32 * height))
        bx1 = min(width, int(cx + 0.12 * width))
        broad = _to_gray(bgr[by0:by1, 0:bx1])

        routes: dict[str, np.ndarray] = {}
        perspective = _perspective_label_crop(bgr, rect)
        if perspective is not None:
            _route_variants(routes, "perspective", perspective)
        _route_variants(routes, "label", whole)
        _route_variants(routes, "anchor", anchor)
        _route_variants(routes, "broad", broad)
        return routes
    except Exception:
        return {}


def _run_route(
    image: np.ndarray,
    preprocessing: str,
    adapters: tuple[SlideCodeDecoder, ...],
    candidates: list[DecodeCandidate],
    seen: set[tuple[str, str, str]],
) -> Optional[SlideQRResult]:
    for adapter in adapters:
        try:
            decoded = adapter.decode(image, preprocessing)
        except Exception:
            continue
        for candidate in decoded:
            key = (candidate.engine, candidate.symbology, candidate.payload)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    selected = select_slide_identity(candidates)
    if selected.success or selected.format == "legacy":
        return selected
    return None


def decode_slide_identity(
    bgr_image: np.ndarray,
    *,
    adapters: tuple[SlideCodeDecoder, ...] | None = None,
) -> SlideQRResult:
    """Decode all candidate symbologies and return only a production claim."""
    if not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return select_slide_identity(())
    try:
        gray = _to_gray(bgr_image)
    except Exception as exc:
        return _with_label_debug(
            bgr_image,
            SlideQRResult(
                False, f"grayscale conversion failed: {exc}", None, None, None,
                None, None, None, None, None, None, None, None, (),
            ),
        )

    started = time.perf_counter()
    candidates: list[DecodeCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    fast_zxing = ZxingSlideCodeDecoder(
        try_rotate=False, try_downscale=False, try_invert=False,
        prefer_single=True,
    )
    standard_zxing = ZxingSlideCodeDecoder(prefer_single=True)
    opencv = OpenCvQrDecoder()

    def within_budget() -> bool:
        return time.perf_counter() - started < DECODE_BUDGET_SECONDS

    def try_route(
        name: str, image: np.ndarray,
        route_adapters: tuple[SlideCodeDecoder, ...],
    ) -> Optional[SlideQRResult]:
        if not within_budget():
            return None
        return _run_route(
            image, name, route_adapters, candidates, seen
        )

    try:
        if adapters is not None:
            injected = tuple(adapters)
            result = try_route("full+raw", gray, injected)
            if result is not None:
                return result
            result = try_route("full+clahe", _clahe(gray), injected)
            if result is not None:
                return result
        else:
            for route_adapters in ((fast_zxing,), (standard_zxing,)):
                result = try_route("full+raw", gray, route_adapters)
                if result is not None:
                    return result
            result = try_route("full+clahe", _clahe(gray), (standard_zxing,))
            if result is not None:
                return result

        label_routes = _label_search_routes(bgr_image) if within_budget() else {}
        if adapters is not None:
            route_adapters = tuple(adapters)
            route_order = (
                "perspective+raw", "perspective+clahe",
                "perspective+rot180+raw", "perspective+rot180+clahe",
                "label+raw", "label+clahe", "anchor+raw",
                "label+rot180+raw", "anchor+clahe",
                "anchor+rot180+clahe", "broad+raw",
                "broad+rot180+raw", "broad+clahe",
                "broad+rot180+clahe",
            )
            for name in route_order:
                image = label_routes.get(name)
                if image is None:
                    continue
                result = try_route(name, image, route_adapters)
                if result is not None:
                    return result
        else:
            label_raw = label_routes.get("label+raw")
            if label_raw is not None:
                result = try_route(
                    "label+raw", label_raw, (standard_zxing, opencv)
                )
                if result is not None:
                    return result
            for name in (
                "perspective+raw", "perspective+clahe",
                "perspective+rot180+raw", "perspective+rot180+clahe",
                "label+clahe", "anchor+raw", "label+rot180+raw",
                "anchor+clahe", "anchor+rot180+clahe", "broad+raw",
                "broad+rot180+raw", "broad+clahe",
                "broad+rot180+clahe",
            ):
                image = label_routes.get(name)
                if image is None:
                    continue
                result = try_route(name, image, (standard_zxing,))
                if result is not None:
                    return result

        if within_budget():
            localized = _locate_and_zoom(gray)
            if localized is not None:
                route_adapters = tuple(adapters) if adapters is not None else (opencv,)
                result = try_route(
                    "localized+raw", localized, route_adapters
                )
                if result is not None:
                    return result
    except Exception as exc:
        if not candidates:
            empty = select_slide_identity(())
            return _with_label_debug(
                bgr_image,
                SlideQRResult(
                    False, f"decode pipeline error: {exc}", empty.raw_payload,
                    empty.format, empty.block_id, empty.slide_num, empty.stain,
                    empty.work_order, empty.email, empty.genotype, empty.engine,
                    empty.preprocessing, empty.symbology, empty.attempts,
                ),
            )
    return _with_label_debug(bgr_image, select_slide_identity(candidates))


# ---------------------------------------------------------------------------
# Internal decode helpers
# ---------------------------------------------------------------------------

def _engines(
    gray: np.ndarray,
    *,
    use_zxing: bool = True,
    use_cv2: bool = True,
) -> Optional[tuple]:
    """Try zxing first, then cv2 fallback on a single grayscale image.

    Returns (engine_name, payload_str) on first hit, or None.
    """
    # zxing-cpp — restrict to QR so a Data Matrix on the same slide can't be
    # returned in place of the slide QR.
    if use_zxing and _HAS_ZXING:
        try:
            results = _zxingcpp.read_barcodes(
                gray, formats=_zxingcpp.BarcodeFormat.QRCode
            )
            if results:
                return ("zxing", results[0].text)
        except Exception:
            pass

    # cv2 decode is speculative fallback only. cv2 localization remains
    # load-bearing elsewhere in this module.
    if use_cv2:
        try:
            decoded, _, _ = _det.detectAndDecode(gray)
        except cv2.error:
            decoded = ""
        if decoded:
            return ("cv2", decoded)

    return None


def _try_variants(
    gray: np.ndarray,
    *,
    use_zxing: bool = True,
    use_cv2: bool = True,
) -> Optional[tuple]:
    """Try all preprocessing variants (raw, Otsu, CLAHE) on a grayscale image.

    Returns (engine_name, payload_str, preprocessing_label) on first hit.
    """
    # raw
    hit = _engines(gray, use_zxing=use_zxing, use_cv2=use_cv2)
    if hit:
        return (hit[0], hit[1], "raw")

    # Otsu binarization
    try:
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        hit = _engines(otsu, use_zxing=use_zxing, use_cv2=use_cv2)
        if hit:
            return (hit[0], hit[1], "otsu")
    except Exception:
        pass

    # CLAHE normalization
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        hit = _engines(enhanced, use_zxing=use_zxing, use_cv2=use_cv2)
        if hit:
            return (hit[0], hit[1], "clahe")
    except Exception:
        pass

    return None


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale (no-op if already 2-D)."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _locate_and_zoom(gray: np.ndarray) -> Optional[np.ndarray]:
    """Use cv2.QRCodeDetector.detect() at multiple scales to find a QR box,
    then crop with margin and upscale.

    Returns an upscaled grayscale crop, or None if not located.
    """
    for scale in (1.0, 0.6, 0.4, 0.3):
        if scale != 1.0:
            gg = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            gg = gray
        try:
            ok, pts = _det.detect(gg)
        except cv2.error:
            ok = False
        if not ok or pts is None:
            continue

        # pts are in scaled space — map back to full resolution
        p = (pts.reshape(-1, 2) / scale).astype(int)
        x0 = int(p[:, 0].min())
        y0 = int(p[:, 1].min())
        x1 = int(p[:, 0].max())
        y1 = int(p[:, 1].max())

        mx = int(0.4 * max(1, x1 - x0))
        my = int(0.4 * max(1, y1 - y0))
        H, W = gray.shape
        x0 = max(0, x0 - mx)
        y0 = max(0, y0 - my)
        x1 = min(W, x1 + mx)
        y1 = min(H, y1 + my)

        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        f = max(1.0, float(_UPSCALE_TARGET) / max(crop.shape))
        return cv2.resize(
            crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC
        )

    return None


def _label_rect_crop(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Try find_label_rect() as an additional localizer fallback.

    Returns a grayscale crop of the label region, or None on failure.
    The crop is upscaled to _UPSCALE_TARGET on its short side.
    """
    if not _HAS_LABEL_MASK:
        return None
    try:
        rect = _find_label_rect(bgr)
        if not rect.found:
            return None
        # Build a tight axis-aligned bounding box from the box_pts corners
        pts = rect.box_pts
        x0 = int(max(0, pts[:, 0].min()))
        y0 = int(max(0, pts[:, 1].min()))
        x1 = int(min(bgr.shape[1], pts[:, 0].max()))
        y1 = int(min(bgr.shape[0], pts[:, 1].max()))
        if x1 <= x0 or y1 <= y0:
            return None
        crop_bgr = bgr[y0:y1, x0:x1]
        crop_gray = _to_gray(crop_bgr)
        f = max(1.0, float(_UPSCALE_TARGET) / max(crop_gray.shape))
        if f > 1.0:
            return cv2.resize(
                crop_gray, None, fx=f, fy=f,
                interpolation=cv2.INTER_CUBIC,
            )
        return crop_gray
    except Exception:
        return None


def _try_crop_with_rotations(
    crop: np.ndarray,
    *,
    use_zxing: bool = True,
    use_cv2: bool = True,
) -> Optional[tuple]:
    """Try variants then rotation fallback on a grayscale crop.

    Returns (engine, payload, preprocessing) or None.
    """
    hit = _try_variants(crop, use_zxing=use_zxing, use_cv2=use_cv2)
    if hit:
        return hit

    # Rotation fallback
    ch, cw = crop.shape[:2]
    cx = cw / 2.0
    cy = ch / 2.0
    for ang in _ROTATION_ANGLES:
        try:
            M = cv2.getRotationMatrix2D((cx, cy), float(ang), 1.0)
            rot = cv2.warpAffine(
                crop, M, (cw, ch), borderValue=255
            )
            hit = _try_variants(rot, use_zxing=use_zxing, use_cv2=use_cv2)
            if hit:
                label = f"{hit[2]}+rot{ang:+d}"
                return (hit[0], hit[1], label)
        except Exception:
            continue

    return None


def _try_decode_pipeline(
    bgr_image: np.ndarray,
    gray: np.ndarray,
    *,
    use_zxing: bool,
    use_cv2: bool,
) -> Optional[tuple]:
    """Run the full image/crop/label pipeline for one decoder policy."""
    hit = _try_variants(gray, use_zxing=use_zxing, use_cv2=use_cv2)
    if hit:
        return hit

    crop = _locate_and_zoom(gray)
    if crop is not None:
        hit = _try_crop_with_rotations(
            crop, use_zxing=use_zxing, use_cv2=use_cv2
        )
        if hit:
            return hit

    label_crop = _label_rect_crop(bgr_image)
    if label_crop is not None:
        hit = _try_crop_with_rotations(
            label_crop, use_zxing=use_zxing, use_cv2=use_cv2
        )
        if hit:
            return hit

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_slide_qr(bgr_image: np.ndarray) -> SlideQRResult:
    """Decode the slide-side identity QR code from a BGR image.

    Pipeline (short-circuits on first success):
      1. Full-frame: try variants on the whole grayscale frame.
      2. Localize: cv2.QRCodeDetector multi-scale -> crop+margin -> upscale
         -> variants + rotation fallback.
      3. Fallback: find_label_rect() label-region crop (optional; only if
         cv2.detect path didn't succeed).

    Returns a frozen SlideQRResult. Never raises.
    """
    # Input guard
    if not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return SlideQRResult(
            success=False, reason="invalid image input",
            raw_payload=None, format=None, block_id=None,
            slide_num=None, stain=None, work_order=None,
            email=None, genotype=None, engine=None, preprocessing=None,
        )

    try:
        gray = _to_gray(bgr_image)
    except Exception as exc:
        return SlideQRResult(
            success=False, reason=f"grayscale conversion failed: {exc}",
            raw_payload=None, format=None, block_id=None,
            slide_num=None, stain=None, work_order=None,
            email=None, genotype=None, engine=None, preprocessing=None,
        )

    engine_hit: Optional[tuple] = None  # (engine, payload, preprocessing)

    # Stages 1-3 are wrapped in a catch-all so the public contract holds:
    # any unexpected error inside the decode pipeline (e.g. a non-cv2.error
    # raised by an engine or a resize on a degenerate array) degrades to a
    # clean failure result rather than propagating.
    try:
        # Run zxing through the full pipeline before cv2 gets decode fallback
        # credit on any variant or crop.
        engine_hit = _try_decode_pipeline(
            bgr_image, gray, use_zxing=True, use_cv2=False
        )
        if engine_hit is None:
            engine_hit = _try_decode_pipeline(
                bgr_image, gray, use_zxing=False, use_cv2=True
            )
    except Exception as exc:
        if engine_hit is None:
            return SlideQRResult(
                success=False, reason=f"decode pipeline error: {exc}",
                raw_payload=None, format=None, block_id=None,
                slide_num=None, stain=None, work_order=None,
                email=None, genotype=None, engine=None, preprocessing=None,
            )

    # ------------------------------------------------------------------
    # No decode at all
    # ------------------------------------------------------------------
    if engine_hit is None:
        return SlideQRResult(
            success=False, reason="no QR code decoded",
            raw_payload=None, format=None, block_id=None,
            slide_num=None, stain=None, work_order=None,
            email=None, genotype=None, engine=None, preprocessing=None,
        )

    engine_name, payload, preprocessing = engine_hit

    # ------------------------------------------------------------------
    # Parse payload
    # ------------------------------------------------------------------
    parsed = parse_qr_payload(payload)
    if not parsed["success"]:
        return SlideQRResult(
            success=False,
            reason=f"decode ok but parse failed: {parsed['reason']}",
            raw_payload=payload, format=None, block_id=None,
            slide_num=None, stain=None, work_order=None,
            email=None, genotype=None,
            engine=engine_name, preprocessing=preprocessing,
        )

    return SlideQRResult(
        success=True,
        reason="ok",
        raw_payload=payload,
        format=parsed.get("format"),
        block_id=parsed.get("block_id"),
        slide_num=parsed.get("slide_num"),
        stain=parsed.get("stain"),
        work_order=parsed.get("work_order"),
        email=parsed.get("email"),
        genotype=parsed.get("genotype"),
        engine=engine_name,
        preprocessing=preprocessing,
    )


__all__ = [
    "DecodeAttempt",
    "DecodeCandidate",
    "OpenCvQrDecoder",
    "SlideCodeDecoder",
    "SlideQRResult",
    "ZxingSlideCodeDecoder",
    "decode_slide_identity",
    "parse_qr_payload",
    "scanner_identity",
    "select_slide_identity",
    "decode_slide_qr",
]
