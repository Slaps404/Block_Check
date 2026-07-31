"""Behavior tests for the QuPath-to-RTrees fidelity review command."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qupath_fidelity_check import main, run_fidelity_check  # noqa: E402
from verify.qupath_features import FeatureRecipe, FeatureSpec  # noqa: E402
from verify.qupath_rtrees import train_rtrees_segmenter  # noqa: E402


def _write_fixture(
    root: Path, *, missing_reference: bool = False
) -> tuple[Path, Path, Path, Path, Path]:
    images = root / "images"
    references = root / "references"
    images.mkdir()
    references.mkdir()
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, 12:, 2] = 220
    labels = np.zeros((24, 24), dtype=np.uint8)
    labels[:, 12:] = 1
    for name in ("depth.png", "tail.png"):
        assert cv2.imwrite(str(images / name), image)
        if not missing_reference or name != "tail.png":
            assert cv2.imwrite(str(references / name), labels)

    recipe = FeatureRecipe(
        channels=("Red",),
        features=(FeatureSpec("GAUSSIAN", 1.0),),
        downsample=1.0,
        class_map={0: "Background", 1: "Tissue"},
    )
    model = root / "block.yml.gz"
    sidecar = root / "block.recipe.json"
    train_rtrees_segmenter(
        recipe,
        [(image, labels)],
        positive_class_id=1,
        model_path=model,
        sidecar_path=sidecar,
        rng_seed=9,
    )
    manifest = root / "review.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_name", "hard_case", "gate1_disposition"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "image_name": "depth.png",
                    "hard_case": "depth-of-section",
                    "gate1_disposition": "approved",
                },
                {
                    "image_name": "tail.png",
                    "hard_case": "faint-tail",
                    "gate1_disposition": "review",
                },
            )
        )
    return images, references, model, sidecar, manifest


def test_fidelity_check_writes_per_block_metrics_and_five_panel_review(tmp_path):
    images, references, model, sidecar, manifest = _write_fixture(tmp_path)
    output = tmp_path / "review-output"

    summary = run_fidelity_check(
        images,
        references,
        model,
        sidecar,
        manifest,
        output,
        min_iou=0.99,
    )

    assert summary.passed is True
    assert summary.aggregate_iou == pytest.approx(1.0)
    assert (output / "depth.review.png").is_file()
    assert (output / "tail.review.png").is_file()
    rows = list(csv.DictReader((output / "fidelity.csv").open(encoding="utf-8")))
    assert [row["gate1_disposition"] for row in rows] == ["approved", "review"]
    assert all(float(row["rtrees_iou"]) >= 0.99 for row in rows)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["gate_2"]["passed"] is True
    assert report["gate_1_dispositions"] == {"approved": 1, "review": 1}


def test_fidelity_check_rejects_missing_reference_instead_of_skipping(tmp_path):
    images, references, model, sidecar, manifest = _write_fixture(
        tmp_path, missing_reference=True
    )

    with pytest.raises(ValueError, match="missing QuPath reference mask.*tail.png"):
        run_fidelity_check(
            images, references, model, sidecar, manifest, tmp_path / "out", min_iou=0.99
        )


def test_command_exits_nonzero_when_gate_2_fidelity_is_below_bar(tmp_path, monkeypatch):
    images, references, model, sidecar, manifest = _write_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qupath_fidelity_check.py",
            "--images", str(images),
            "--references", str(references),
            "--model", str(model),
            "--sidecar", str(sidecar),
            "--review-manifest", str(manifest),
            "--output-dir", str(tmp_path / "out"),
            "--min-iou", "1.0",
        ],
    )
    reference = cv2.imread(str(references / "tail.png"), cv2.IMREAD_GRAYSCALE)
    reference[0, 0] = 1
    assert cv2.imwrite(str(references / "tail.png"), reference)

    with pytest.raises(SystemExit, match="1"):
        main()
