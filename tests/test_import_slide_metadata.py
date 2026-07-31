"""Tests for the slide-label OCR catalog importer."""

from __future__ import annotations

import sqlite3

import cv2
import numpy as np

from slide.label_mask import LabelRect
from tools.catalog import import_slide_metadata as catalog


def test_match_organs_matches_phrases_before_splitting():
    vocabulary = catalog.load_vocabulary()

    matches = catalog.match_organs("Heart/Panc, small intestine; esophgus", vocabulary)

    assert [(match.name, match.status) for match in matches] == [
        ("esophagus", "confirmed"),
        ("heart", "confirmed"),
        ("intestine", "confirmed"),
        ("pancreas", "confirmed"),
    ]


def test_match_organs_does_not_fuzzy_match_stain_text_as_paw():
    vocabulary = catalog.load_vocabulary()

    matches = catalog.match_organs("Kidney PAS", vocabulary)

    assert [match.name for match in matches] == ["kidney"]


def test_discover_slide_images_excludes_generated_qc_panels(tmp_path):
    assert cv2.imwrite(
        str(tmp_path / "slide_53523999_20260720T051938Z_example.png"),
        np.zeros((20, 20, 3), dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(tmp_path / "slide_53523999_20260720T051938Z_example_claim_qc.png"),
        np.zeros((20, 20, 3), dtype=np.uint8),
    )

    found = catalog.discover_slide_images([tmp_path])

    assert [path.name for path in found] == ["slide_53523999_20260720T051938Z_example.png"]


def test_import_records_raw_ocr_and_current_multi_organ_matches(tmp_path, monkeypatch):
    slides = tmp_path / "slides"
    slides.mkdir()
    image_path = slides / "slide_53523999_20260720T051938Z_example.png"
    assert cv2.imwrite(str(image_path), np.full((80, 120, 3), 255, dtype=np.uint8))
    rect = LabelRect(
        found=True,
        center=(60.0, 40.0),
        size=(80.0, 30.0),
        angle=0.0,
        box_pts=np.array([[20, 25], [100, 25], [100, 55], [20, 55]], dtype=np.float32),
        label_side="top",
    )
    monkeypatch.setattr(catalog, "find_label_rect", lambda image: rect)
    database = tmp_path / "catalog.sqlite3"

    summary = catalog.import_slide_metadata([slides], database, ocr=lambda crop: "Lung Panc")
    second_summary = catalog.import_slide_metadata([slides], database, ocr=lambda crop: "Heart")

    assert summary == catalog.ImportSummary(1, 1, 0, 0, 1)
    assert second_summary == catalog.ImportSummary(1, 1, 0, 0, 1)
    with sqlite3.connect(database) as db:
        image = db.execute("SELECT block_id FROM images").fetchone()
        raw_text = db.execute("SELECT raw_text, error FROM ocr_results").fetchone()
        image_count = db.execute("SELECT COUNT(*) FROM images").fetchone()
        result_count = db.execute("SELECT COUNT(*) FROM ocr_results").fetchone()
        organs = db.execute(
            "SELECT organ, status FROM image_organs ORDER BY organ"
        ).fetchall()
    assert image == ("53523999",)
    assert raw_text == ("Lung Panc", None)
    assert image_count == (1,)
    assert result_count == (2,)
    assert organs == [("heart", "confirmed")]


def test_import_falls_back_to_full_image_when_label_is_missing(tmp_path, monkeypatch):
    slides = tmp_path / "slides"
    slides.mkdir()
    assert cv2.imwrite(str(slides / "slide_no_label.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(
        catalog,
        "find_label_rect",
        lambda image: LabelRect(False, (0.0, 0.0), (0.0, 0.0), 0.0, np.zeros((4, 2)), "none"),
    )

    def full_image_ocr(crop):
        assert crop.shape == (20, 20, 3)
        return "Heart"

    summary = catalog.import_slide_metadata(
        [slides], tmp_path / "catalog.sqlite3", ocr=full_image_ocr
    )

    assert summary == catalog.ImportSummary(1, 1, 1, 0, 1)
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as db:
        result = db.execute("SELECT raw_text, ocr_region FROM ocr_results").fetchone()
        organs = db.execute("SELECT organ FROM image_organs").fetchall()
    assert result == ("Heart", "full_image")
    assert organs == [("heart",)]
