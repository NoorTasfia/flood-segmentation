"""Tests for make_combined_loaders / source_balanced_sampler (floodlite.data).

Requires torch + cv2 (skipped where unavailable); the underlying split logic
is tested torch-free in test_manifest.py.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=Exception)
cv2 = pytest.importorskip("cv2", exc_type=Exception)

from floodlite.data import make_combined_loaders, source_balanced_sampler  # noqa: E402

COLUMNS = [
    "image", "mask", "source", "source_image", "file_format", "role",
    "source_split", "kaggle_fold", "kaggle_group", "duplicate_of",
    "width", "height", "flood_fraction", "exclude", "exclude_reason",
]


@pytest.fixture
def combined_root(tmp_path: Path) -> Path:
    """Synthetic combined dataset: 12 kaggle (folds 0-4) + floodnet + afs rows."""
    rng = np.random.default_rng(0)
    rows = []

    def add(image, source, role, fold=-1):
        rows.append({
            "image": image, "mask": Path(image).stem + ".png",
            "source": source, "source_image": image, "file_format": "jpeg",
            "role": role, "source_split": "", "kaggle_fold": str(fold),
            "kaggle_group": "-1", "duplicate_of": "", "width": "64",
            "height": "64", "flood_fraction": "0.3", "exclude": "",
            "exclude_reason": "",
        })

    for i in range(10):
        add(f"kaggle_{i:03d}.jpg", "kaggle", "kaggle", fold=i % 5)
    for i in range(6):
        add(f"floodnet_{i:03d}.jpg", "floodnet", "extra_train")
    for i in range(4):
        add(f"floodnet_t{i:03d}.jpg", "floodnet", "floodnet_test")
    for i in range(3):
        add(f"afs_{i:03d}.jpg", "afs", "external_test")

    (tmp_path / "Image").mkdir()
    (tmp_path / "Mask").mkdir()
    for r in rows:
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        msk = (rng.integers(0, 2, (64, 64), dtype=np.uint8) * 255)
        cv2.imwrite(str(tmp_path / "Image" / r["image"]), img)
        cv2.imwrite(str(tmp_path / "Mask" / r["mask"]), msk)
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_loaders_have_all_splits_and_correct_shapes(combined_root):
    loaders = make_combined_loaders(combined_root, fold=0, batch_size=2,
                                    img_size=64, num_workers=0)
    assert set(loaders) == {"train", "val", "test", "external"}
    for name in ("train", "val", "test", "external"):
        x, y = next(iter(loaders[name]))
        assert x.shape[1:] == (3, 64, 64)
        assert y.shape[1:] == (1, 64, 64)
        assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_split_sizes_match_manifest_roles(combined_root):
    loaders = make_combined_loaders(combined_root, fold=0, batch_size=1,
                                    img_size=64, num_workers=0)
    # fold 0: train = 8 kaggle (folds 1-4) + 6 extra_train; val = 2 kaggle
    assert len(loaders["train"].dataset) == 14
    assert len(loaders["val"].dataset) == 2
    assert len(loaders["test"].dataset) == 4
    assert len(loaders["external"].dataset) == 3


def test_train_loader_uses_source_balanced_sampler(combined_root):
    loaders = make_combined_loaders(combined_root, fold=0, batch_size=2,
                                    img_size=64, num_workers=0)
    from torch.utils.data import WeightedRandomSampler
    assert isinstance(loaders["train"].sampler, WeightedRandomSampler)

    unbalanced = make_combined_loaders(combined_root, fold=0, batch_size=2,
                                       img_size=64, num_workers=0,
                                       balance_sources=False)
    assert not isinstance(unbalanced["train"].sampler, WeightedRandomSampler)


def test_eval_loaders_are_deterministic(combined_root):
    loaders = make_combined_loaders(combined_root, fold=0, batch_size=2,
                                    img_size=64, num_workers=0)
    x1, _ = next(iter(loaders["test"]))
    x2, _ = next(iter(loaders["test"]))
    assert torch.equal(x1, x2)


def test_sampler_draws_sources_evenly(combined_root):
    sources = ["floodnet"] * 90 + ["kaggle"] * 10
    sampler = source_balanced_sampler(sources, seed=42)
    draws = list(sampler)
    kaggle_share = sum(1 for i in draws if i >= 90) / len(draws)
    assert 0.3 < kaggle_share < 0.7  # ~0.5 expected; 0.1 without balancing


def test_mislabelled_gif_image_still_loads(combined_root):
    """45 combined files named .jpg are really WebP/PNG/GIF; cv2 can't read GIF."""
    from PIL import Image
    gif_path = combined_root / "Image" / "kaggle_001.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(gif_path, format="GIF")
    loaders = make_combined_loaders(combined_root, fold=0, batch_size=1,
                                    img_size=64, num_workers=0,
                                    balance_sources=False)
    for x, _ in loaders["train"]:  # must not crash on the GIF
        assert x.shape[1:] == (3, 64, 64)
