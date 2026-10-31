"""Tests for floodlite.phash — perceptual hashing used by the FSSD-overlap gate.

Torch-free on purpose: the dedup gate must run on machines without a working
torch install.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from floodlite.phash import phash_file, phash_image, hamming_matrix, find_near_duplicates


def _structured_image(seed: int, size: int = 128) -> Image.Image:
    """A smooth, structured RGB image (random low-frequency blobs).

    Smooth structure (not white noise) because pHash compares low-frequency
    DCT content — exactly what real photographs carry.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BILINEAR)


def test_identical_images_have_distance_zero(tmp_path: Path):
    img = _structured_image(0)
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    img.save(p1)
    img.save(p2)
    h1, h2 = phash_file(p1), phash_file(p2)
    assert h1 == h2
    d = hamming_matrix(np.array([h1], dtype=np.uint64), np.array([h2], dtype=np.uint64))
    assert d[0, 0] == 0


def test_reencoded_and_resized_copy_is_near_duplicate(tmp_path: Path):
    """A JPEG-recompressed, downscaled copy must stay within hamming <= 8."""
    img = _structured_image(1, size=256)
    orig, copy = tmp_path / "orig.png", tmp_path / "copy.jpg"
    img.save(orig)
    img.resize((192, 192), Image.BILINEAR).save(copy, quality=70)
    d = hamming_matrix(
        np.array([phash_file(orig)], dtype=np.uint64),
        np.array([phash_file(copy)], dtype=np.uint64),
    )
    assert d[0, 0] <= 8


def test_distinct_images_are_far_apart():
    h1 = phash_image(_structured_image(2))
    h2 = phash_image(_structured_image(3))
    d = hamming_matrix(np.array([h1], dtype=np.uint64), np.array([h2], dtype=np.uint64))
    assert d[0, 0] > 8


def test_hamming_matrix_known_values():
    a = np.array([0b0000, 0b0111], dtype=np.uint64)
    b = np.array([0b0001], dtype=np.uint64)
    d = hamming_matrix(a, b)
    assert d.shape == (2, 1)
    assert d[0, 0] == 1  # 0000 ^ 0001
    assert d[1, 0] == 2  # 0111 ^ 0001 = 0110


def test_find_near_duplicates_returns_matching_pairs(tmp_path: Path):
    img = _structured_image(4, size=256)
    (tmp_path / "x.png").parent.mkdir(exist_ok=True)
    img.save(tmp_path / "x.png")
    img.resize((200, 200)).save(tmp_path / "x_copy.jpg", quality=80)

    hashes_a = np.array([phash_file(tmp_path / "x.png"),
                         phash_image(_structured_image(5))], dtype=np.uint64)
    hashes_b = np.array([phash_file(tmp_path / "x_copy.jpg")], dtype=np.uint64)

    hits = find_near_duplicates(hashes_a, hashes_b, threshold=8)
    assert (0, 0) in {(i, j) for i, j, _ in hits}
    assert all(dist <= 8 for _, _, dist in hits)
    # The unrelated image must not match.
    assert (1, 0) not in {(i, j) for i, j, _ in hits}
