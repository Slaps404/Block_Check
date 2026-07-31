#!/usr/bin/env python3
"""
add_extra_slides_for_blocks.py

Add newly captured slide images for blocks already present in images/pi_images.

Use case:
  - You already have a block image named like:
      set_07_block_silhouette_lung_HE_TWKO2_WO7842.jpg
  - Later you capture additional slide images for that block/animal.
  - This script asks for tissue + animal, finds the matching block, asks
    for the new slide stain, and writes:
      set_07_slide_lung_MT_TWKO2_WO7842.jpg

Important:
  - Raw images are left untouched.
  - Refuses to overwrite existing slides unless --overwrite is passed.
  - If more than one block matches, it asks which set to use.

Run:
    python tools/capture/add_extra_slides_for_blocks.py
    python tools/capture/add_extra_slides_for_blocks.py \
        --raw ./pi_captures --pi-images ./images/pi_images
    python tools/capture/add_extra_slides_for_blocks.py \
        --raw ./pi_captures --pi-images ./images/pi_images \
        --batch-tissue lung --batch-animal "TWKO 2"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

VALID_TISSUES = {"lung", "lungs", "esophagus", "skin"}
VALID_STAINS = {"HE", "MT", "PAS", "PSRFG", "SMA"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_WO_PATTERN = re.compile(r"^WO\d+$|^\d+$")


@dataclass(frozen=True)
class BlockRecord:
    path: Path
    set_num: int
    tissue: str
    stain: str
    genotype: str
    work_order: str


def discover_images(raw_dir: Path) -> list[Path]:
    """Return raw slide images sorted by modification time/capture order."""
    images = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
    return images


def _open_with_os(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  Could not open image: {exc}")
        print(f"  Open manually: {path.resolve()}")


def show_image(path: Path) -> None:
    """Display image non-blocking; questions follow immediately.

    Closes any previous preview window before opening the new one.
    If OpenCV is unavailable, opens in the OS default viewer (no wait).
    """
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        _open_with_os(path)
        return

    img = cv2.imread(str(path))
    if img is None:
        _open_with_os(path)
        return

    max_h = 650
    h, w = img.shape[:2]
    if h > max_h:
        scale = max_h / float(h)
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.putText(
        img, path.name,
        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
    )

    window_name = f"Slide preview — {path.name}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
    except Exception as exc:
        print(f"  OpenCV window failed: {exc}")
        preview_file = (
            Path(tempfile.gettempdir()) / f"preview_{path.stem}.jpg"
        )
        cv2.imwrite(str(preview_file), img)
        print(f"  Wrote preview: {preview_file}")
        _open_with_os(preview_file)


def close_preview() -> None:
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass


def normalize_token(value: str) -> str:
    return "_".join(value.strip().split())


def normalize_tissue(value: str) -> str:
    """Normalize tissue for matching; treats lung/lungs as equivalent."""
    v = normalize_token(value).lower()
    if v == "lungs":
        return "lung"
    return v


def normalize_genotype(value: str) -> str:
    """Normalize animal/genotype names like 'TWKO 2' -> 'TWKO2'."""
    return "".join(
        value.strip().upper().replace("-", "").replace("_", "").split()
    )


def parse_block_filename(path: Path) -> BlockRecord | None:
    """Parse set_NN_block_silhouette_<tissue>_<stain>_<genotype>[_<wo>]."""
    parts = path.stem.split("_")
    if len(parts) < 7:
        return None
    if parts[0] != "set":
        return None
    try:
        set_num = int(parts[1])
    except ValueError:
        return None
    if parts[2] != "block" or parts[3] != "silhouette":
        return None

    remainder = parts[4:]  # tissue..., stain, genotype[, work_order]
    if len(remainder) < 3:
        return None

    # Work order is optional. Detect by WO\d+ or pure-digit last token.
    last = remainder[-1].upper()
    if len(remainder) >= 4 and _WO_PATTERN.match(last):
        tissue = "_".join(remainder[:-3]).lower()
        stain = remainder[-3].upper()
        genotype = remainder[-2]
        work_order = last
    else:
        tissue = "_".join(remainder[:-2]).lower()
        stain = remainder[-2].upper()
        genotype = remainder[-1]
        work_order = ""

    return BlockRecord(
        path=path,
        set_num=set_num,
        tissue=tissue,
        stain=stain,
        genotype=genotype,
        work_order=work_order,
    )


def load_blocks(pi_images_dir: Path) -> list[BlockRecord]:
    blocks: list[BlockRecord] = []
    for path in sorted(pi_images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        record = parse_block_filename(path)
        if record is not None:
            blocks.append(record)
    return blocks


def prompt_nonempty(prompt_text: str) -> str:
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        print("  (required — please enter a value)")


def prompt_tissue_and_animal(
    *,
    batch_tissue: str | None,
    batch_animal: str | None,
) -> tuple[str, str]:
    if batch_tissue:
        tissue = normalize_tissue(batch_tissue)
        print(f"  Tissue: {tissue} (from --batch-tissue)")
    else:
        tissue = normalize_tissue(
            prompt_nonempty(
                "  Tissue for existing block (e.g. lung, esophagus): "
            )
        )
    if tissue not in {normalize_tissue(t) for t in VALID_TISSUES}:
        print(
            f"  Note: '{tissue}' is not in the usual list "
            f"{sorted(VALID_TISSUES)} — using it anyway."
        )

    if batch_animal:
        animal = normalize_genotype(batch_animal)
        print(f"  Animal/genotype: {animal} (from --batch-animal)")
    else:
        animal = normalize_genotype(
            prompt_nonempty(
                "  Animal/genotype for existing block"
                " (e.g. TWKO 2, WT3): "
            )
        )

    return tissue, animal


def prompt_stain() -> str:
    stain = normalize_token(
        prompt_nonempty("  New slide stain (HE / MT / PAS / PSRFG / SMA): ")
    ).upper()
    if stain not in VALID_STAINS:
        print(
            f"  Note: '{stain}' is not in the usual list "
            f"{sorted(VALID_STAINS)} — using it anyway."
        )
    return stain


def choose_block(matches: list[BlockRecord]) -> BlockRecord:
    if len(matches) == 1:
        chosen = matches[0]
        print(
            f"  Matched existing block: "
            f"set_{chosen.set_num:02d} ({chosen.path.name})"
        )
        return chosen

    print("  Multiple existing blocks match that tissue + animal:")
    for i, block in enumerate(matches, start=1):
        print(
            f"    {i}. set_{block.set_num:02d} | tissue={block.tissue}"
            f" | stain={block.stain} | animal={block.genotype}"
            f" | wo={block.work_order} | file={block.path.name}"
        )

    while True:
        raw = input("  Choose block number from list: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("  Please enter a number from the list.")
            continue
        if 1 <= idx <= len(matches):
            return matches[idx - 1]
        print("  Number out of range.")


def build_slide_name(block: BlockRecord, new_stain: str, ext: str) -> str:
    base = (
        f"set_{block.set_num:02d}_slide_"
        f"{block.tissue}_{new_stain}_{block.genotype}"
    )
    if block.work_order:
        base += f"_{block.work_order}"
    return base + ext.lower()


def copy_or_fail(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {dst}\n"
            "This usually means that slide/stain already exists for this "
            "set. Use --overwrite only if replacement is intentional."
        )
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add extra slide captures for already-tagged block images."
    )
    parser.add_argument(
        "--raw", default="./pi_captures",
        help="Folder containing newly captured slide images.",
    )
    parser.add_argument(
        "--pi-images", default="./images/pi_images",
        help="Folder containing existing blocks; new slides written here.",
    )
    parser.add_argument(
        "--batch-tissue", default=None,
        help="Use the same tissue for every raw slide.",
    )
    parser.add_argument(
        "--batch-animal", default=None,
        help="Use the same animal/genotype for every raw slide.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing an existing slide filename.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    pi_images_dir = Path(args.pi_images)

    if not raw_dir.is_dir():
        print(f"ERROR: raw slide folder not found: {raw_dir.resolve()}")
        return 2
    if not pi_images_dir.is_dir():
        print(f"ERROR: pi_images folder not found: {pi_images_dir.resolve()}")
        return 2

    raw_slides = discover_images(raw_dir)
    if not raw_slides:
        print(f"No raw slide images found in {raw_dir.resolve()}")
        return 0

    blocks = load_blocks(pi_images_dir)
    if not blocks:
        print(
            f"ERROR: no block_silhouette images found in "
            f"{pi_images_dir.resolve()}"
        )
        return 2

    print(f"\nFound {len(raw_slides)} new slide image(s).")
    print(f"Found {len(blocks)} existing block image(s) to match against.\n")

    written: list[tuple[str, str]] = []

    for idx, slide_src in enumerate(raw_slides, start=1):
        print(f"--- New slide {idx}/{len(raw_slides)}: {slide_src.name} ---")
        show_image(slide_src)

        tissue, animal = prompt_tissue_and_animal(
            batch_tissue=args.batch_tissue,
            batch_animal=args.batch_animal,
        )

        matches = [
            b for b in blocks
            if normalize_tissue(b.tissue) == tissue
            and normalize_genotype(b.genotype) == animal
        ]

        if not matches:
            print(
                f"  NO MATCH: no existing block found for "
                f"tissue={tissue}, animal={animal}. Skipping.\n"
            )
            close_preview()
            continue

        block = choose_block(matches)
        new_stain = prompt_stain()
        close_preview()

        new_name = build_slide_name(block, new_stain, slide_src.suffix)
        dst = pi_images_dir / new_name

        copy_or_fail(slide_src, dst, overwrite=args.overwrite)
        written.append((slide_src.name, new_name))

        print(f"  -> {new_name}\n")

    print("=" * 60)
    print(
        f"Done. {len(written)} new slide file(s) written to "
        f"{pi_images_dir.resolve()}"
    )
    print("Original raw slide images were left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
