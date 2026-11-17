"""Tests for the RescueNet zero-shot test set: mask binarisation and loader."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_rescuenet import RESCUENET_WATER_INDEX, binarise_rescuenet_mask, write_manifest  # noqa: E402


def test_binarise_maps_only_the_water_index_to_255():
    label = np.array([[0, 1, 2], [10, 1, 7]], dtype=np.uint8)
    out = binarise_rescuenet_mask(label, water_index=1)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 255, 0], [0, 255, 0]]


def test_water_index_is_the_documented_class_one():
    # RescueNet class list: 0 background, 1 water, 2-5 buildings, 6 vehicle,
    # 7 road-clear, 8 road-blocked, 9 tree, 10 pool (verified on the files, Task 7 step 2)
    assert RESCUENET_WATER_INDEX == 1


def test_write_manifest_records_flood_fraction(tmp_path: Path):
    masks = tmp_path / "Mask"; masks.mkdir()
    imgs = tmp_path / "Image"; imgs.mkdir()
    cv2 = pytest.importorskip("cv2", exc_type=Exception)
    m = np.zeros((4, 4), np.uint8); m[:2] = 255
    cv2.imwrite(str(masks / "rescuenet_1.png"), m)
    cv2.imwrite(str(imgs / "rescuenet_1.jpg"), np.zeros((4, 4, 3), np.uint8))
    write_manifest(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv", encoding="utf-8")))
    assert rows[0]["image"] == "rescuenet_1.jpg" and rows[0]["mask"] == "rescuenet_1.png"
    assert rows[0]["source"] == "rescuenet_test"
    assert float(rows[0]["flood_fraction"]) == pytest.approx(0.5)


def test_loader_yields_normalised_batches(tmp_path: Path):
    torch = pytest.importorskip("torch", exc_type=Exception)
    cv2 = pytest.importorskip("cv2", exc_type=Exception)
    from floodlite.data import make_rescuenet_loader
    (tmp_path / "Image").mkdir(); (tmp_path / "Mask").mkdir()
    for i in range(10):
        cv2.imwrite(str(tmp_path / "Image" / f"rescuenet_{i}.jpg"), np.random.randint(0, 255, (64, 64, 3), np.uint8))
        cv2.imwrite(str(tmp_path / "Mask" / f"rescuenet_{i}.png"), (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255)
    write_manifest(tmp_path)
    loader = make_rescuenet_loader(tmp_path, batch_size=8, img_size=256)
    x, y = next(iter(loader))
    assert tuple(x.shape) == (8, 3, 256, 256) and tuple(y.shape) == (8, 1, 256, 256)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    assert len(loader.dataset) == 10
