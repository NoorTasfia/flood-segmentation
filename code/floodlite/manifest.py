"""Manifest-driven splits for the flood-combined-fsegnet dataset.

The combined dataset (Kaggle Flood Area Segmentation + FloodNet + AIFloodSense)
ships a ``manifest.csv`` with one row per image/mask pair and these roles:

- ``kaggle``        290 images with a duplicate-aware 5-fold assignment
                    (``kaggle_fold``); duplicates share a ``kaggle_group`` so
                    copies never straddle a fold boundary.
- ``extra_train``   FloodNet train+val (1,825) — always training data.
- ``floodnet_test`` FloodNet test (424) — held-out test, identical per fold.
- ``external_test`` AIFloodSense test (94) — zero-shot external set; never train.
- ``external_pool`` AIFloodSense train (376) — reserved; never used.

Split per fold f: train = kaggle(fold != f) + extra_train; val = kaggle(fold == f)
minus within-source duplicate copies; test = floodnet_test; external = external_test.

Torch-free: shared by the training loaders (floodlite.data) and the
FSSD-overlap dedup gate (scripts/check_fssd_overlap.py).
"""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path

_UNSET = {"", "0", "-1", "false", "False", "nan", "None"}


def _is_set(value: str | None) -> bool:
    return value is not None and value.strip() not in _UNSET


def load_manifest(root: str | Path) -> list[dict]:
    """Read manifest.csv under ``root``, dropping rows flagged ``exclude``."""
    path = Path(root) / "manifest.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return [r for r in rows if not _is_set(r.get("exclude"))]


def split_manifest(rows: list[dict], fold: int, n_folds: int = 5) -> dict[str, list[dict]]:
    """Partition manifest rows into train/val/test/external for one fold."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0,{n_folds}); got {fold}")
    split: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "external": []}
    for r in rows:
        role = r["role"]
        is_duplicate_copy = _is_set(r.get("duplicate_of"))
        if role == "kaggle":
            if int(r["kaggle_fold"]) == fold:
                if not is_duplicate_copy:  # copies are never evaluation data
                    split["val"].append(r)
            else:
                split["train"].append(r)
        elif role == "extra_train":
            split["train"].append(r)
        elif role == "floodnet_test":
            if not is_duplicate_copy:
                split["test"].append(r)
        elif role == "external_test":
            if not is_duplicate_copy:
                split["external"].append(r)
        # external_pool: reserved, never used for training or evaluation
    return split


def pairs_from_rows(root: str | Path, rows: list[dict]) -> list[tuple[Path, Path]]:
    """(image_path, mask_path) pairs for manifest rows; every file must exist."""
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    for r in rows:
        img = root / "Image" / r["image"]
        mask = root / "Mask" / r["mask"]
        if not img.exists():
            raise FileNotFoundError(f"Image listed in manifest is missing: {img}")
        if not mask.exists():
            raise FileNotFoundError(f"Mask listed in manifest is missing: {mask}")
        pairs.append((img, mask))
    return pairs


def source_balance_weights(sources: list[str]) -> list[float]:
    """Inverse-frequency sampling weights so each source has equal total weight.

    With FloodNet:Kaggle ~10:1 (and water fraction 14% vs 41%), uniform
    sampling would let FloodNet dominate every batch; these weights make a
    WeightedRandomSampler draw each source equally often in expectation.
    """
    counts = Counter(sources)
    return [1.0 / counts[s] for s in sources]
