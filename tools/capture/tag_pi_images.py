#!/usr/bin/env python3
"""
tag_pi_images.py

Renames already-captured Pi images into the project filename convention:

    set_NN_<role>_<tissue>_<stain>_<genotype>_<workorder>.<ext>

Capture order assumption (confirmed by Zeke):
  - Images are taken in pairs.
  - Within each pair the SLIDE is captured first, then its BLOCK.
  - So the sorted-by-time sequence is: slide, block, slide, block, ...
  - Each pair becomes one set, numbered sequentially starting at set_01.

For each pair you type the metadata ONCE (for the slide). The script applies the
same tissue / stain / genotype / work order to the block that follows, only
changing the role token from 'slide' to 'block_silhouette'. The block carries
the paired slide's stain as metadata, matching the project convention.

Workflow:
  1. Put the raw captured images in one folder (default ./pi_captures).
  2. Run this script. It lists the images sorted by modification time.
  3. For each pair it pops up the slide and block images (small window, top-right),
     then asks for the slide's metadata and closes the window when you're done.
  4. It writes renamed copies into the output folder (default ./images/pi_images_v3),
     leaving the originals untouched.

Run:
    python tools/capture/tag_pi_images.py
    python tools/capture/tag_pi_images.py --raw ./pi_captures --out ./images/pi_images_v3 --start 1
"""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

VIEWER_HEIGHT = 750


class ImageViewer:
    """Reusable windowed viewer. Call .show(path) per image; .close() at the end."""

    def __init__(self):
        self._ok = False
        self._root = None
        self._label = None
        self._photo = None
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            self._tk = tk
            self._Image = Image
            self._ImageTk = ImageTk
            self._ok = True
        except Exception as e:
            print(f"  [Windowed viewer unavailable ({e}); using system viewer.]")

    def show(self, path: Path):
        if not self._ok:
            _open_with_system(path)
            return
        try:
            if self._root is None:
                self._root = self._tk.Tk()
                self._root.title("LJI Image Tagger")
                self._root.attributes("-topmost", True)
                self._label = self._tk.Label(self._root)
                self._label.pack()

            img = self._Image.open(str(path))
            w, h = img.size
            scale = VIEWER_HEIGHT / h
            new_size = (max(1, int(w * scale)), VIEWER_HEIGHT)
            img = img.resize(new_size, self._Image.LANCZOS)

            self._photo = self._ImageTk.PhotoImage(img)
            self._label.configure(image=self._photo)

            self._root.update_idletasks()
            screen_w = self._root.winfo_screenwidth()
            x = max(0, screen_w - new_size[0] - 20)
            self._root.geometry(f"{new_size[0]}x{VIEWER_HEIGHT}+{x}+20")
            self._root.title(f"LJI Image Tagger \u2014 {path.name}")
            self._root.update()
        except Exception as e:
            print(f"  [Viewer error ({e}); opening with system viewer.]")
            _open_with_system(path)

    def close(self):
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None


def _open_with_system(path: Path):
    """Fallback: open in OS default viewer."""
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"  [Could not open image: {e}]  Open manually: {path}")


# Valid values for light validation. Extend as the project grows.
VALID_TISSUES = {"lung", "lungs", "esophagus"}
VALID_STAINS = {"HE", "MT", "PAS", "PSRFG", "SMA"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_images(raw_dir: Path) -> list[Path]:
    """Return image files in raw_dir sorted by modification time (capture order)."""
    images = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    # Sort by modification time so capture order is preserved.
    images.sort(key=lambda p: p.stat().st_mtime)
    return images


def prompt_nonempty(prompt_text: str, viewer: ImageViewer | None = None) -> str:
    """Ask until the user types something non-empty."""
    while True:
        if viewer is not None:
            viewer.pump()
        value = input(prompt_text).strip()
        if value:
            return value
        print("  (required — please enter a value)")


def prompt_metadata(viewer: ImageViewer | None = None) -> dict:
    """Collect the slide metadata that is shared with its block."""
    tissue = prompt_nonempty("  Tissue (lung / lungs / esophagus): ", viewer).lower()
    if tissue not in VALID_TISSUES:
        print(f"  Note: '{tissue}' is not in the usual list {sorted(VALID_TISSUES)} "
              f"— using it anyway.")

    stain = prompt_nonempty("  Stain (HE / MT / PAS / PSRFG / SMA): ", viewer).upper()
    if stain not in VALID_STAINS:
        print(f"  Note: '{stain}' is not in the usual list {sorted(VALID_STAINS)} "
              f"— using it anyway.")

    genotype = prompt_nonempty("  Genotype (e.g. TWKO4, WT3, NAIVE): ", viewer)
    work_order = prompt_nonempty("  Work order number (e.g. 7842): ", viewer)
    # Normalize work order so it always starts with WO and has no duplicate prefix.
    work_order = work_order.upper().lstrip("WO")
    work_order = f"WO{work_order}"

    return {
        "tissue": tissue,
        "stain": stain,
        "genotype": genotype,
        "work_order": work_order,
    }


def build_name(set_num: int, role: str, meta: dict, ext: str) -> str:
    """Assemble the canonical filename for one image."""
    return (
        f"set_{set_num:02d}_{role}_"
        f"{meta['tissue']}_{meta['stain']}_{meta['genotype']}_{meta['work_order']}{ext}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag Pi capture pairs (slide then block).")
    parser.add_argument("--raw", default="./pi_captures",
                        help="Folder containing the raw captured images.")
    parser.add_argument("--out", default="./images/pi_images_v3",
                        help="Folder to write renamed copies into.")
    parser.add_argument("--start", type=int, default=1,
                        help="Set number to start counting from (default 1).")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if not raw_dir.is_dir():
        print(f"ERROR: raw folder not found: {raw_dir.resolve()}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(raw_dir)
    if not images:
        print(f"No images found in {raw_dir.resolve()}")
        return

    if len(images) % 2 != 0:
        print(f"WARNING: found {len(images)} images — not an even number. "
              f"The last unpaired image will be skipped. Check your capture set.")

    num_pairs = len(images) // 2
    print(f"\nFound {len(images)} images = {num_pairs} pairs (slide, block).")
    print("For each pair, the SLIDE is first and the BLOCK is second.\n")

    set_num = args.start
    renamed = []
    viewer = ImageViewer()

    try:
        for pair_index in range(num_pairs):
            slide_src = images[pair_index * 2]
            block_src = images[pair_index * 2 + 1]

            print(f"--- Pair {pair_index + 1}  →  set_{set_num:02d} ---")
            print(f"  SLIDE image: {slide_src.name}")
            print(f"  BLOCK image: {block_src.name}")

            viewer.show(slide_src)
            meta = prompt_metadata(viewer)
            viewer.close()

            slide_name = build_name(set_num, "slide", meta, slide_src.suffix.lower())
            block_name = build_name(set_num, "block_silhouette", meta, block_src.suffix.lower())

            shutil.copy2(slide_src, out_dir / slide_name)
            shutil.copy2(block_src, out_dir / block_name)

            renamed.append((slide_src.name, slide_name))
            renamed.append((block_src.name, block_name))

            print(f"  -> {slide_name}")
            print(f"  -> {block_name}\n")

            set_num += 1
    finally:
        viewer.close()

    print("=" * 60)
    print(f"Done. {len(renamed)} files written to {out_dir.resolve()}")
    print(f"Sets {args.start:02d} through {set_num - 1:02d} created.")
    print("Originals in the raw folder were left untouched.")


if __name__ == "__main__":
    main()
