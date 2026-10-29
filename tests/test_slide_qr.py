"""
Unit tests for code/slide/qr.py.

Covers:
  - parse_qr_payload: current format, legacy format, malformed input
  - decode_slide_qr: synthetic QR round-trip, blank image failure path

All tests are fast and do NOT depend on pi_images.
"""

from __future__ import annotations

import numpy as np
import pytest

import slide.qr as slide_qr_module
from slide.qr import (
    DecodeCandidate,
    SlideQRResult,
    decode_slide_identity,
    decode_slide_qr,
    parse_qr_payload,
    select_slide_identity,
)


# ---------------------------------------------------------------------------
# parse_qr_payload — current format
# ---------------------------------------------------------------------------

class TestParseQrPayloadCurrent:

    def test_maps_four_fields(self):
        result = parse_qr_payload("7842_51151378_01_HE")
        assert result["success"] is True
        assert result["format"] == "current"
        assert result["work_order"] == "7842"
        assert result["block_id"] == "51151378"
        assert result["slide_num"] == "01"
        assert result["stain"] == "HE"
        assert result["email"] is None
        assert result["genotype"] is None

    def test_result_exposes_lab_work_order_without_renaming_wire_field(self):
        """The QR number is the lab job, not the local bracket id."""
        result = SlideQRResult(
            True, "ok", "12080_51151378_01_HE", "current", "51151378",
            "01", "HE", "12080", None, None, "zxing", "raw",
        )

        assert result.lab_work_order == "12080"
        assert result.work_order == "12080"  # legacy/wire compatibility


# ---------------------------------------------------------------------------
# parse_qr_payload — legacy format
# ---------------------------------------------------------------------------

class TestParseQrPayloadLegacy:

    def test_maps_four_fields(self):
        result = parse_qr_payload("rgupta@lji.org_WT 3_01_HE")
        assert result["success"] is True
        assert result["format"] == "legacy"
        assert result["email"] == "rgupta@lji.org"
        assert result["genotype"] == "WT 3"
        assert result["slide_num"] == "01"
        assert result["stain"] == "HE"
        assert result["block_id"] is None
        assert result["work_order"] is None

    def test_spaced_genotype_preserved_verbatim(self):
        """Internal space in genotype must survive unchanged."""
        result = parse_qr_payload("user@example.com_TWKO 4_03_MT")
        assert result["genotype"] == "TWKO 4"


# ---------------------------------------------------------------------------
# parse_qr_payload — malformed input
# ---------------------------------------------------------------------------

class TestParseQrPayloadMalformed:

    @pytest.mark.parametrize("bad_input", [
        "",          # empty string
        None,        # non-string
        42,          # non-string int
        [],          # non-string list
        0,           # falsy int
        "a_b_c_d_e", # too many fields
        "a_b_c",     # too few fields
        "x_y",       # too few fields
        "7842__01_HE",  # empty field
    ])
    def test_bad_input_returns_failure(self, bad_input):
        result = parse_qr_payload(bad_input)
        assert result["success"] is False
        assert "reason" in result

    @pytest.mark.parametrize("payload", [
        "abc_51151378_01_HE",
        "7842_5115137_01_HE",
        "7842_511513789_01_HE",
        "7842_51151A78_01_HE",
    ])
    def test_current_payload_requires_numeric_work_order_and_eight_digit_block_id(
        self, payload
    ):
        assert parse_qr_payload(payload)["success"] is False

    def test_variable_length_numeric_work_order_is_allowed(self):
        result = parse_qr_payload("12080_51137181_01_HE")
        assert result["success"] is True
        assert result["work_order"] == "12080"

    def test_legacy_genotype_preserves_embedded_underscores(self):
        result = parse_qr_payload("sara+czi@lji.org_SM13_Good_01_HE")
        assert result["success"] is True
        assert result["genotype"] == "SM13_Good"
        assert result["block_id"] is None

    def test_allowed_leading_control_whitespace_is_removed(self):
        result = parse_qr_payload("\x1d\r\n 12080_51137181_01_HE")
        assert result["success"] is True

    def test_trailing_or_internal_control_characters_are_rejected(self):
        for payload in (
            "12080_51137181_01_HE\n",
            "12080_51137181_01_\x00HE",
        ):
            assert parse_qr_payload(payload)["success"] is False


class TestCandidateSelection:
    def test_zxing_fast_read_expands_when_first_payload_is_malformed(
        self, monkeypatch
    ):
        class Barcode:
            def __init__(self, text):
                self.text = text
                self.format = "QRCode"

        class FakeZxing:
            class BarcodeFormat:
                QRCode = "qr"
                DataMatrix = "dm"

            def read_barcode(self, image, **kwargs):
                return Barcode("malformed")

            def read_barcodes(self, image, **kwargs):
                return (
                    Barcode("malformed"),
                    Barcode("12080_51137181_01_HE"),
                )

        monkeypatch.setattr(slide_qr_module, "_HAS_ZXING", True)
        monkeypatch.setattr(slide_qr_module, "_zxingcpp", FakeZxing())
        decoder = slide_qr_module.ZxingSlideCodeDecoder(prefer_single=True)

        candidates = decoder.decode(
            np.ones((20, 20), dtype=np.uint8), "full+raw"
        )

        assert [candidate.payload for candidate in candidates] == [
            "malformed", "12080_51137181_01_HE",
        ]

    def test_valid_current_payload_wins_over_decoder_order(self):
        result = select_slide_identity((
            DecodeCandidate("zxing", "QRCode", "raw", "not_an_identity"),
            DecodeCandidate(
                "zxing", "DataMatrix", "raw", "12080_51137181_01_HE"
            ),
        ))

        assert result.success is True
        assert result.block_id == "51137181"
        assert result.symbology == "DataMatrix"
        assert len(result.attempts) == 2
        assert result.attempts[0].accepted is False
        assert result.attempts[1].accepted is True

    def test_legacy_candidate_is_diagnostic_not_a_production_claim(self):
        result = select_slide_identity((
            DecodeCandidate(
                "zxing", "QRCode", "raw",
                "sara+czi@lji.org_SM13_Good_01_HE",
            ),
        ))

        assert result.success is False
        assert result.format == "legacy"
        assert result.block_id is None
        assert result.reason == "legacy payload cannot establish a production claim"

    def test_replaceable_adapters_are_all_considered(self):
        class FakeAdapter:
            def __init__(self, candidate):
                self.candidate = candidate

            def decode(self, image, preprocessing):
                if preprocessing != "full+raw":
                    return ()
                return (self.candidate,)

        image = np.ones((20, 20, 3), dtype=np.uint8) * 255
        result = decode_slide_identity(image, adapters=(
            FakeAdapter(DecodeCandidate("qr", "QRCode", "raw", "bad")),
            FakeAdapter(DecodeCandidate(
                "dm", "DataMatrix", "raw", "12080_51137181_01_HE"
            )),
        ))

        assert result.success is True
        assert result.engine == "dm"
        assert result.symbology == "DataMatrix"

    def test_bounded_scheduler_reaches_full_clahe_without_otsu_sweep(self):
        class ClaheOnlyAdapter:
            def __init__(self):
                self.routes = []

            def decode(self, image, preprocessing):
                self.routes.append(preprocessing)
                if preprocessing != "full+clahe":
                    return ()
                return (DecodeCandidate(
                    "test", "QRCode", preprocessing,
                    "12080_51137181_01_HE",
                ),)

        adapter = ClaheOnlyAdapter()
        image = np.ones((20, 20, 3), dtype=np.uint8) * 255

        result = decode_slide_identity(image, adapters=(adapter,))

        assert result.success is True
        assert result.preprocessing == "full+clahe"
        assert adapter.routes == ["full+raw", "full+clahe"]

    def test_perspective_label_route_can_establish_identity(self, monkeypatch):
        class Rect:
            found = True
            center = (50.0, 50.0)
            size = (60.0, 40.0)
            angle = 0.0
            box_pts = np.array([
                [20.0, 30.0], [80.0, 30.0],
                [80.0, 70.0], [20.0, 70.0],
            ], dtype=np.float32)

        class PerspectiveAdapter:
            def decode(self, image, preprocessing):
                if preprocessing != "perspective+raw":
                    return ()
                return (DecodeCandidate(
                    "test", "QRCode", preprocessing,
                    "12080_51137181_01_HE",
                ),)

        monkeypatch.setattr(slide_qr_module, "_HAS_LABEL_MASK", True)
        monkeypatch.setattr(
            slide_qr_module, "_find_label_rect", lambda image: Rect()
        )
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = decode_slide_identity(
            image, adapters=(PerspectiveAdapter(),)
        )

        assert result.success is True
        assert result.preprocessing == "perspective+raw"


# ---------------------------------------------------------------------------
# decode_slide_qr — synthetic QR round-trip
# ---------------------------------------------------------------------------

def test_decode_slide_qr_synthetic_round_trip():
    """Render a current-format QR with qrcode, decode it, assert the full result."""
    qrcode = pytest.importorskip("qrcode")
    payload = "7842_51151378_01_HE"
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    rgb = np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))

    result = decode_slide_qr(rgb[:, :, ::-1].copy())

    assert result.success is True, f"Expected success but got: {result.reason}"
    assert result.raw_payload == payload
    assert result.format == "current"
    assert result.work_order == "7842"
    assert result.block_id == "51151378"
    assert result.slide_num == "01"
    assert result.stain == "HE"
    assert result.engine in ("cv2", "zxing")
    assert isinstance(result.preprocessing, str) and result.preprocessing
    assert isinstance(result, SlideQRResult)
    with pytest.raises((AttributeError, TypeError)):
        result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Internal engine policy
# ---------------------------------------------------------------------------

class TestDecodeEnginePolicy:

    class _FakeBarcode:
        def __init__(self, text):
            self.text = text

    class _FakeZxing:
        class BarcodeFormat:
            QRCode = object()

        def __init__(self, results):
            self.results = results
            self.calls = 0

        def read_barcodes(self, gray, formats=None):
            if (
                self.results
                and isinstance(self.results[0], list)
            ):
                index = min(self.calls, len(self.results) - 1)
                self.calls += 1
                return self.results[index]
            self.calls += 1
            return self.results

    class _FakeDetector:
        def __init__(self, payload):
            self.payload = payload

        def detectAndDecode(self, gray):
            return self.payload, None, None

    class _FakePyzbar:
        class _Hit:
            data = b"pyzbar_payload"

        def decode(self, gray):
            return [self._Hit()]

    def test_zxing_wins_when_zxing_and_cv2_both_decode(self, monkeypatch):
        gray = np.ones((20, 20), dtype=np.uint8) * 255
        monkeypatch.setattr(slide_qr_module, "_HAS_ZXING", True)
        monkeypatch.setattr(
            slide_qr_module,
            "_zxingcpp",
            self._FakeZxing([self._FakeBarcode("zxing_payload")]),
        )
        monkeypatch.setattr(
            slide_qr_module,
            "_det",
            self._FakeDetector("cv2_payload"),
        )

        assert slide_qr_module._engines(gray) == ("zxing", "zxing_payload")

    def test_cv2_runs_as_fallback_when_zxing_misses(self, monkeypatch):
        gray = np.ones((20, 20), dtype=np.uint8) * 255
        monkeypatch.setattr(slide_qr_module, "_HAS_ZXING", True)
        monkeypatch.setattr(slide_qr_module, "_zxingcpp", self._FakeZxing([]))
        monkeypatch.setattr(
            slide_qr_module,
            "_det",
            self._FakeDetector("cv2_payload"),
        )

        assert slide_qr_module._engines(gray) == ("cv2", "cv2_payload")

    def test_pyzbar_is_not_a_decode_fallback(self, monkeypatch):
        gray = np.ones((20, 20), dtype=np.uint8) * 255
        monkeypatch.setattr(slide_qr_module, "_HAS_ZXING", True)
        monkeypatch.setattr(slide_qr_module, "_zxingcpp", self._FakeZxing([]))
        monkeypatch.setattr(slide_qr_module, "_det", self._FakeDetector(""))
        monkeypatch.setattr(slide_qr_module, "_HAS_PYZBAR", True, raising=False)
        monkeypatch.setattr(
            slide_qr_module, "_pyzbar", self._FakePyzbar(), raising=False
        )

        assert slide_qr_module._engines(gray) is None

    def test_zxing_gets_all_preprocessing_variants_before_cv2(self, monkeypatch):
        bgr = np.ones((20, 20, 3), dtype=np.uint8) * 255
        monkeypatch.setattr(slide_qr_module, "_HAS_ZXING", True)
        monkeypatch.setattr(
            slide_qr_module,
            "_zxingcpp",
            self._FakeZxing([
                [],
                [self._FakeBarcode("7842_51151378_01_HE")],
            ]),
        )
        monkeypatch.setattr(
            slide_qr_module,
            "_det",
            self._FakeDetector("7842_00000000_99_MT"),
        )

        result = decode_slide_qr(bgr)

        assert result.success is True
        assert result.engine == "zxing"
        assert result.raw_payload == "7842_51151378_01_HE"


# ---------------------------------------------------------------------------
# decode_slide_qr — failure path on blank image
# ---------------------------------------------------------------------------

class TestDecodeSlideQrFailure:

    def test_blank_image_fails_with_reason_and_no_fields(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = decode_slide_qr(blank)
        assert result.success is False
        assert isinstance(result.reason, str) and result.reason
        for field in ("raw_payload", "format", "block_id", "slide_num",
                      "stain", "work_order", "email", "genotype",
                      "engine", "preprocessing"):
            assert getattr(result, field) is None, (
                f"{field} should be None on failure"
            )

    @pytest.mark.parametrize("image", [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.random.default_rng(42).integers(0, 256, (300, 300, 3), dtype=np.uint8),
        np.array([]),
    ], ids=["black", "noise", "empty"])
    def test_does_not_raise(self, image):
        result = decode_slide_qr(image)
        assert isinstance(result, SlideQRResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# decode_slide_identity — label-rectangle debug note on failure
# ---------------------------------------------------------------------------

class TestDecodeSlideIdentityLabelDebug:
    """Operator triage: label missing vs QR unreadable."""

    def test_blank_image_notes_label_not_found(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = decode_slide_identity(blank)
        assert result.success is False
        assert "label rectangle not found" in result.reason

    def test_label_found_notes_qr_not_readable(self, monkeypatch):
        class Rect:
            found = True
            center = (50.0, 50.0)
            size = (60.0, 40.0)
            angle = 0.0
            box_pts = np.array([
                [20.0, 30.0], [80.0, 30.0],
                [80.0, 70.0], [20.0, 70.0],
            ], dtype=np.float32)

        class EmptyAdapter:
            def decode(self, image, preprocessing):
                return ()

        monkeypatch.setattr(slide_qr_module, "_HAS_LABEL_MASK", True)
        monkeypatch.setattr(
            slide_qr_module, "_find_label_rect", lambda image: Rect()
        )
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = decode_slide_identity(image, adapters=(EmptyAdapter(),))

        assert result.success is False
        assert "label rectangle found; QR not readable enough" in result.reason
        assert result.raw_payload is None
