"""FSSD <-> flood-combined-fsegnet duplicate gate (pHash, hamming <= 8).

Hard gate before any training on the combined dataset: FloodLite's existing
checkpoints are trained on FSSD, so a near-duplicate of an FSSD image inside
the combined evaluation roles (floodnet_test, external_test, external_pool)
would leak training data into the zero-shot / held-out evaluations. Such hits
fail the gate (exit 1). Hits in train-side roles (kaggle, extra_train) are
reported as warnings only — they do not corrupt any planned evaluation.

Uses the same algorithm and threshold as the combined dataset's internal
cross-source dedup (DCT pHash, hamming distance <= 8), so the two checks are
directly comparable. Torch-free: runs anywhere Python + numpy/scipy/PIL run.

Usage:
    python scripts/check_fssd_overlap.py --fssd_root /path/to/FSSD \
        --combined_root /path/to/dataset_combined \
        [--threshold 8] [--out runs/fssd_overlap_report.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floodlite.manifest import load_manifest
from floodlite.phash import phash_file, find_near_duplicates

# Roles used for held-out / zero-shot evaluation: an FSSD near-duplicate here
# invalidates the evaluation, so it blocks. Train-side hits only warn.
EVAL_ROLES = {"floodnet_test", "external_test", "external_pool"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMG_DIR_NAMES = ("images", "Image", "IMAGES")


def collect_fssd_images(root: str | Path) -> list[Path]:
    """All FSSD images under <root>/**/{train,val}/{images,Image,IMAGES}/."""
    root = Path(root)
    files: list[Path] = []
    seen_dirs: set[Path] = set()  # Windows globbing is case-insensitive
    for split in ("train", "val"):
        for dir_name in IMG_DIR_NAMES:
            for img_dir in root.rglob(f"{split}/{dir_name}"):
                img_dir = img_dir.resolve()
                if img_dir in seen_dirs:
                    continue
                seen_dirs.add(img_dir)
                files.extend(p for p in sorted(img_dir.iterdir())
                             if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No FSSD images found under {root!s}; expected "
            "dataset/{train,val}/{images,labels}.")
    return files


def _hash_files(paths: list[Path], desc: str) -> np.ndarray:
    hashes = [phash_file(p) for p in tqdm(paths, desc=desc, unit="img")]
    return np.array(hashes, dtype=np.uint64)


def run_check(fssd_root: str | Path, combined_root: str | Path,
              threshold: int = 8) -> dict:
    """Hash both datasets, find cross-dataset near-duplicates, classify them."""
    combined_root = Path(combined_root)
    fssd_files = collect_fssd_images(fssd_root)
    rows = load_manifest(combined_root)
    combined_files = [combined_root / "Image" / r["image"] for r in rows]

    fssd_hashes = _hash_files(fssd_files, "FSSD")
    combined_hashes = _hash_files(combined_files, "combined")

    hits = find_near_duplicates(fssd_hashes, combined_hashes, threshold=threshold)
    blocking, warnings = [], []
    for i, j, dist in hits:
        row = rows[j]
        record = {
            "fssd_image": str(fssd_files[i]),
            "combined_image": row["image"],
            "source": row["source"],
            "role": row["role"],
            "distance": dist,
        }
        (blocking if row["role"] in EVAL_ROLES else warnings).append(record)

    return {
        "threshold": threshold,
        "n_fssd_images": len(fssd_files),
        "n_combined_images": len(combined_files),
        "n_hits": len(hits),
        "blocking": sorted(blocking, key=lambda h: h["distance"]),
        "warnings": sorted(warnings, key=lambda h: h["distance"]),
        "passed": not blocking,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fssd_root", required=True)
    ap.add_argument("--combined_root", required=True)
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--out", default="runs/fssd_overlap_report.json")
    args = ap.parse_args()

    report = run_check(args.fssd_root, args.combined_root, args.threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nFSSD images: {report['n_fssd_images']}  "
          f"combined images: {report['n_combined_images']}  "
          f"hits (hamming <= {report['threshold']}): {report['n_hits']}")
    for h in report["blocking"]:
        print(f"  BLOCKING  d={h['distance']:2d}  {h['role']:>14s}  "
              f"{h['combined_image']}  <->  {h['fssd_image']}")
    for h in report["warnings"]:
        print(f"  warning   d={h['distance']:2d}  {h['role']:>14s}  "
              f"{h['combined_image']}  <->  {h['fssd_image']}")
    print(f"Report written to {out}")
    print("GATE PASSED" if report["passed"] else
          "GATE FAILED: FSSD near-duplicates found in evaluation roles")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
