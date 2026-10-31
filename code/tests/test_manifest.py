"""Tests for floodlite.manifest — manifest-driven splits for flood-combined-fsegnet.

Runs without torch: split logic and sampler weights are pure Python/CSV.
Synthetic manifests mirror the real manifest.csv columns documented in the
dataset README (roles: kaggle / extra_train / floodnet_test / external_test /
external_pool; duplicate-aware kaggle_fold; exclude flags).
"""
from __future__ import annotations
import csv
from pathlib import Path

import pytest

from floodlite.manifest import (
    load_manifest,
    split_manifest,
    pairs_from_rows,
    source_balance_weights,
)

COLUMNS = [
    "image", "mask", "source", "source_image", "file_format", "role",
    "source_split", "kaggle_fold", "kaggle_group", "duplicate_of",
    "width", "height", "flood_fraction", "exclude", "exclude_reason",
]


def _row(image, source, role, kaggle_fold=-1, duplicate_of="", exclude=""):
    return {
        "image": image, "mask": image.replace(".jpg", ".png"),
        "source": source, "source_image": image, "file_format": "jpeg",
        "role": role, "source_split": "", "kaggle_fold": str(kaggle_fold),
        "kaggle_group": "-1", "duplicate_of": duplicate_of,
        "width": "64", "height": "64", "flood_fraction": "0.3",
        "exclude": exclude, "exclude_reason": "near-dup" if exclude else "",
    }


@pytest.fixture
def combined_root(tmp_path: Path) -> Path:
    rows = [
        _row("kaggle_000.jpg", "kaggle", "kaggle", kaggle_fold=0),
        _row("kaggle_001.jpg", "kaggle", "kaggle", kaggle_fold=0,
             duplicate_of="kaggle_000.jpg"),
        _row("kaggle_002.jpg", "kaggle", "kaggle", kaggle_fold=1),
        _row("kaggle_003.jpg", "kaggle", "kaggle", kaggle_fold=1),
        _row("kaggle_004.jpg", "kaggle", "kaggle", kaggle_fold=2),
        _row("kaggle_005.jpg", "kaggle", "kaggle", kaggle_fold=2),
        _row("floodnet_100.jpg", "floodnet", "extra_train"),
        _row("floodnet_101.jpg", "floodnet", "extra_train"),
        _row("floodnet_102.jpg", "floodnet", "extra_train", exclude="1"),
        _row("floodnet_200.jpg", "floodnet", "floodnet_test"),
        _row("floodnet_201.jpg", "floodnet", "floodnet_test"),
        _row("afs_300.jpg", "afs", "external_test"),
        _row("afs_400.jpg", "afs", "external_pool"),
    ]
    (tmp_path / "Image").mkdir()
    (tmp_path / "Mask").mkdir()
    for r in rows:
        (tmp_path / "Image" / r["image"]).touch()
        (tmp_path / "Mask" / r["mask"]).touch()
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_load_manifest_drops_excluded_rows(combined_root):
    rows = load_manifest(combined_root)
    names = {r["image"] for r in rows}
    assert "floodnet_102.jpg" not in names
    assert len(rows) == 12


def test_split_roles_are_disjoint_and_correct(combined_root):
    rows = load_manifest(combined_root)
    split = split_manifest(rows, fold=0)

    train = {r["image"] for r in split["train"]}
    val = {r["image"] for r in split["val"]}
    test = {r["image"] for r in split["test"]}
    external = {r["image"] for r in split["external"]}

    # val = kaggle fold 0, minus within-source duplicate copies
    assert val == {"kaggle_000.jpg"}
    # train = kaggle folds != 0 plus all extra_train (non-excluded)
    assert train == {"kaggle_002.jpg", "kaggle_003.jpg", "kaggle_004.jpg",
                     "kaggle_005.jpg", "floodnet_100.jpg", "floodnet_101.jpg"}
    # test = FloodNet held-out test, identical for every fold
    assert test == {"floodnet_200.jpg", "floodnet_201.jpg"}
    # external = AIFloodSense external_test only (never external_pool)
    assert external == {"afs_300.jpg"}

    assert not (train & val) and not (train & test) and not (val & test)
    assert not (external & (train | val | test))


def test_duplicate_copy_never_lands_in_val(combined_root):
    rows = load_manifest(combined_root)
    for fold in range(3):
        split = split_manifest(rows, fold=fold)
        assert "kaggle_001.jpg" not in {r["image"] for r in split["val"]}


def test_duplicate_copy_allowed_in_train_for_other_folds(combined_root):
    rows = load_manifest(combined_root)
    split = split_manifest(rows, fold=1)  # fold 0 rows go to train
    assert "kaggle_001.jpg" in {r["image"] for r in split["train"]}


def test_changing_fold_changes_val(combined_root):
    rows = load_manifest(combined_root)
    assert {r["image"] for r in split_manifest(rows, fold=1)["val"]} == \
        {"kaggle_002.jpg", "kaggle_003.jpg"}


def test_pairs_from_rows_builds_existing_paths(combined_root):
    rows = load_manifest(combined_root)
    pairs = pairs_from_rows(combined_root, split_manifest(rows, fold=0)["test"])
    assert len(pairs) == 2
    for img, mask in pairs:
        assert img.exists() and mask.exists()
        assert img.parent.name == "Image" and mask.parent.name == "Mask"


def test_pairs_from_rows_raises_on_missing_file(combined_root):
    rows = load_manifest(combined_root)
    (combined_root / "Image" / "floodnet_200.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        pairs_from_rows(combined_root, split_manifest(rows, fold=0)["test"])


def test_source_balance_weights_equalize_sources():
    sources = ["floodnet"] * 9 + ["kaggle"]
    w = source_balance_weights(sources)
    assert len(w) == 10
    # Total weight per source is equal -> expected 50/50 sampling.
    assert sum(w[:9]) == pytest.approx(sum(w[9:]))
    # All floodnet samples share one weight; kaggle sample is 9x heavier.
    assert w[9] == pytest.approx(9 * w[0])
