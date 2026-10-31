"""Perceptual hashing (pHash) for cross-dataset duplicate detection.

Implements the standard DCT pHash (Zauner 2010), bit-compatible with the
`imagehash` library's ``phash`` default parameters: 32x32 grayscale resize,
2-D DCT-II, top-left 8x8 low-frequency block thresholded at its median.
The flood-combined-fsegnet dataset's own internal dedup used the same
algorithm with hamming distance <= 8, so the FSSD-overlap gate reuses it.

Torch-free by design: this must run on machines without a working torch
install (the dedup gate runs wherever the raw datasets were downloaded).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

HASH_BITS = 64


def phash_image(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit perceptual hash of a PIL image."""
    size = hash_size * highfreq_factor
    gray = img.convert("L").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    freq = dct(dct(pixels, axis=0), axis=1)
    low = freq[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def phash_file(path: str | Path) -> int:
    """64-bit perceptual hash of an image file (any PIL-readable format)."""
    with Image.open(path) as img:
        return phash_image(img)


def _unpack(hashes: np.ndarray) -> np.ndarray:
    """(N,) uint64 -> (N, 64) bit matrix."""
    h = np.ascontiguousarray(hashes, dtype=np.uint64)
    return np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1)


def hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Pairwise hamming distances between two uint64 hash arrays.

    Returns an (len(a), len(b)) uint8 matrix; computed in row chunks so the
    intermediate bit tensors stay small for ~4k x 3k comparisons.
    """
    abits = _unpack(a)
    bbits = _unpack(b)
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for start in range(0, len(a), chunk):
        block = abits[start:start + chunk]
        out[start:start + chunk] = (
            block[:, None, :] != bbits[None, :, :]
        ).sum(axis=2, dtype=np.uint8)
    return out


def find_near_duplicates(hashes_a: np.ndarray, hashes_b: np.ndarray,
                         threshold: int = 8) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with hamming(hashes_a[i], hashes_b[j]) <= threshold."""
    dists = hamming_matrix(np.asarray(hashes_a, dtype=np.uint64),
                           np.asarray(hashes_b, dtype=np.uint64))
    ii, jj = np.nonzero(dists <= threshold)
    return [(int(i), int(j), int(dists[i, j])) for i, j in zip(ii, jj)]
