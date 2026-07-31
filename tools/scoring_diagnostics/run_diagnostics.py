"""Run all-pairs scorer diagnostics for Pi block/slide images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pair_diagnostics import run_all_pairs_diagnostic  # noqa: E402
from robust_normalization import NORMALIZATION_MODES  # noqa: E402


def discover_dataset_paths(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return block and slide image paths for supported dataset layouts.

    Legacy Pi captures use ``set_NN_block_silhouette_...jpg`` names. The v3
    PNG captures use ``block_NNN_...png`` / ``slide_NNN_...png`` names.
    """
    patterns = (
        ("*block_silhouette*.jpg", "*slide*.jpg"),
        ("block_*.png", "slide_*.png"),
    )
    for block_pattern, slide_pattern in patterns:
        block_paths = sorted(dataset_dir.glob(block_pattern))
        slide_paths = sorted(dataset_dir.glob(slide_pattern))
        if block_paths and slide_paths:
            return block_paths, slide_paths
    return [], []


def load_manifest_paths(
    manifest_csv: Path,
    root: Path,
) -> tuple[list[Path], list[Path], dict[str, tuple[str, str]]]:
    """Source block/slide image lists and a ground-truth map from a manifest.

    The v2 dataset uses opaque timestamp filenames the glob cannot match, so
    both the image lists and the (set_id, tissue) map come from the manifest.

    Returns (block_paths, slide_paths, path_metadata):
      - block_paths: unique ``root / block_path`` values, sorted.
      - slide_paths: unique ``root / slide_path`` values, sorted.
      - path_metadata: keyed by ``(root / relpath).as_posix()`` (the same
        convention run_all_pairs_diagnostic looks up) -> (set_id, tissue),
        with blocks mapped to block_tissue and slides to slide_tissue.
    """
    with open(manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    blocks: dict[str, Path] = {}
    slides: dict[str, Path] = {}
    path_metadata: dict[str, tuple[str, str]] = {}

    for row in rows:
        set_id = row["set_id"]

        block_path = root / row["block_path"]
        block_key = block_path.as_posix()
        blocks[block_key] = block_path
        path_metadata[block_key] = (set_id, row["block_tissue"])

        slide_path = root / row["slide_path"]
        slide_key = slide_path.as_posix()
        slides[slide_key] = slide_path
        path_metadata[slide_key] = (set_id, row["slide_tissue"])

    block_paths = sorted(blocks.values(), key=lambda p: p.as_posix())
    slide_paths = sorted(slides.values(), key=lambda p: p.as_posix())
    return block_paths, slide_paths, path_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="outputs/diagnostics/all_pairs.csv",
        help="CSV output path for diagnostic scores.",
    )
    parser.add_argument(
        "--dataset",
        default="images/pi_images_v3",
        help=(
            "Image folder to run on (e.g. images/pi_images_v2 for the "
            "re-exposed A/B set)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/pi_images_v3_manifest.csv",
        help=(
            "Manifest CSV to source image lists and "
            "ground-truth labels from, for opaque-filename datasets. When set, "
            "the --dataset glob is bypassed. Defaults to the v3 PNG manifest; "
            "pass an empty string to use dataset discovery."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Discover current images directly from --dataset, ignoring manifests.",
    )
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=3,
        help="Number of true-pair locked-alignment PNGs to write.",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="rms",
        help="Diagnostic normalization mode; production default is rms.",
    )
    args = parser.parse_args()

    if args.manifest and not args.no_manifest:
        manifest_csv = ROOT / args.manifest
        if not manifest_csv.is_file():
            parser.error(f"manifest file not found: {manifest_csv}")
        block_paths, slide_paths, path_metadata = load_manifest_paths(
            manifest_csv, ROOT
        )
        if not block_paths or not slide_paths:
            parser.error(
                f"manifest yielded no block/slide rows: {manifest_csv}"
            )
        run_all_pairs_diagnostic(
            block_paths,
            slide_paths,
            ROOT / args.output,
            path_metadata=path_metadata,
            normalization_mode=args.normalization,
        )
        return

    dataset_dir = ROOT / args.dataset
    if not dataset_dir.is_dir():
        parser.error(f"dataset folder not found: {dataset_dir}")

    block_paths, slide_paths = discover_dataset_paths(dataset_dir)
    if not block_paths or not slide_paths:
        parser.error(
            f"no supported block/slide images found in {dataset_dir} "
            "(expected legacy set_NN JPGs or v3 block_NNN/slide_NNN PNGs)."
        )
    run_all_pairs_diagnostic(
        block_paths,
        slide_paths,
        ROOT / args.output,
        normalization_mode=args.normalization,
    )


if __name__ == "__main__":
    main()
