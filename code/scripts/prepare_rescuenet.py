"""Prepare the RescueNet official test split as a zero-shot flood test set.

RescueNet (Rahnemoonfar, Chowdhury & Murphy, Scientific Data 2023; CDLA-Permissive)
is a UAV dataset from Hurricane Michael (2018) with 11 classes. Its index masks
use: 0 background, 1 water, 2 building-no-damage, 3 building-medium-damage,
4 building-major-damage, 5 building-total-destruction, 6 vehicle, 7 road-clear,
8 road-blocked, 9 tree, 10 pool. We map ``water`` (index 1) to flood (255) and
everything else to 0, which mirrors the FloodNet mapping of the combined corpus
(water + road-flooded -> flood; pools are not flood).

The water index was verified on the downloaded label files (Task 7 step 2 of
the 2026-11-17 plan): run ``--inspect`` to print the label histogram.

    python -I scripts/prepare_rescuenet.py --raw <download dir> --out <dataset dir>
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RESCUENET_WATER_INDEX = 1
SOURCE = "rescuenet_test"


def binarise_rescuenet_mask(label: np.ndarray, water_index: int = RESCUENET_WATER_INDEX) -> np.ndarray:
    """0/255 uint8 mask with 255 where ``label == water_index``."""
    return np.where(label == water_index, 255, 0).astype(np.uint8)


def find_pairs(raw: Path) -> list[tuple[Path, Path]]:
    """Match RescueNet images to their label PNGs.

    The release names labels ``<id>_lab.png`` for image ``<id>.jpg``; the Kaggle
    mirror keeps that convention. Falls back to identical stems.
    """
    labels = {p.stem.replace("_lab", ""): p for p in raw.rglob("*.png")}
    pairs = []
    for img in sorted(raw.rglob("*.jpg")):
        lab = labels.get(img.stem)
        if lab is not None:
            pairs.append((img, lab))
    return pairs


def inspect_labels(raw: Path, n: int = 20) -> Counter:
    hist: Counter = Counter()
    for _, lab in find_pairs(raw)[:n]:
        arr = cv2.imread(str(lab), cv2.IMREAD_UNCHANGED)
        vals, counts = np.unique(arr, return_counts=True)
        hist.update(dict(zip(vals.tolist(), counts.tolist())))
    return hist


def write_manifest(out: Path) -> Path:
    """manifest.csv for every Image/<name>.jpg with Mask/<name>.png present."""
    rows = []
    for img in sorted((out / "Image").glob("*.jpg")):
        mask = out / "Mask" / (img.stem + ".png")
        if not mask.exists():
            continue
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        rows.append({"image": img.name, "mask": mask.name, "source": SOURCE,
                     "source_image": img.name, "width": m.shape[1], "height": m.shape[0],
                     "flood_fraction": f"{float((m > 127).mean()):.4f}"})
    path = out / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["image", "mask", "source", "source_image", "width", "height", "flood_fraction"])
        w.writeheader(); w.writerows(rows)
    return path


def prepare(raw: Path, out: Path, water_index: int = RESCUENET_WATER_INDEX) -> int:
    (out / "Image").mkdir(parents=True, exist_ok=True)
    (out / "Mask").mkdir(parents=True, exist_ok=True)
    n = 0
    for img, lab in find_pairs(raw):
        name = f"rescuenet_{img.stem}"
        shutil.copy(img, out / "Image" / f"{name}.jpg")
        arr = cv2.imread(str(lab), cv2.IMREAD_UNCHANGED)
        if arr.ndim == 3:
            arr = arr[..., 0]
        cv2.imwrite(str(out / "Mask" / f"{name}.png"), binarise_rescuenet_mask(arr, water_index))
        n += 1
    write_manifest(out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=False)
    ap.add_argument("--inspect", action="store_true", help="print the label-value histogram and exit")
    ap.add_argument("--water_index", type=int, default=RESCUENET_WATER_INDEX)
    a = ap.parse_args()
    raw = Path(a.raw)
    if a.inspect:
        pairs = find_pairs(raw)
        print(f"{len(pairs)} image/label pairs under {raw}")
        for v, c in sorted(inspect_labels(raw).items()):
            print(f"  label {v:3d}: {c:>10d} px")
        return
    n = prepare(raw, Path(a.out), a.water_index)
    print(f"wrote {n} pairs to {a.out}")


if __name__ == "__main__":
    main()
