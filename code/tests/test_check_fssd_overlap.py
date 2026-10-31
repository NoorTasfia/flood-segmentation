"""Tests for scripts/check_fssd_overlap.py — the FSSD <-> combined dedup gate.

Hard gate before any training on flood-combined-fsegnet: a near-duplicate of
an FSSD image inside the combined evaluation roles (floodnet_test,
external_test, external_pool) would leak FSSD training data into the
zero-shot / held-out evaluations, so those hits must fail the gate. Hits in
train-side roles (kaggle, extra_train) are reported as warnings only.
"""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_fssd_overlap.py"
spec = importlib.util.spec_from_file_location("check_fssd_overlap", SCRIPT)
overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)


def _structured(seed: int, size: int = 128) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BILINEAR)


def _make_fssd(root: Path) -> None:
    for split, stems in (("train", ["a", "b"]), ("val", ["c"])):
        img_dir = root / "dataset" / split / "images"
        lbl_dir = root / "dataset" / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for stem in stems:
            seed = ord(stem)
            _structured(seed).save(img_dir / f"{stem}.png")
            Image.new("L", (128, 128)).save(lbl_dir / f"{stem}.png")


def _make_combined(root: Path) -> None:
    (root / "Image").mkdir(parents=True)
    (root / "Mask").mkdir(parents=True)
    rows = [
        # near-duplicate of FSSD train image 'a' in a TRAIN-side role -> warning
        ("kaggle_000.jpg", "kaggle", "kaggle", ord("a")),
        # near-duplicate of FSSD val image 'c' in an EVAL role -> blocking
        ("floodnet_200.jpg", "floodnet", "floodnet_test", ord("c")),
        # unrelated external image -> no hit
        ("afs_300.jpg", "afs", "external_test", 999),
    ]
    for name, _, _, seed in rows:
        img = _structured(seed, size=192)
        img.save(root / "Image" / name, quality=85)
        Image.new("L", (192, 192)).save(root / "Mask" / name.replace(".jpg", ".png"))
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask", "source", "source_image", "file_format",
                    "role", "source_split", "kaggle_fold", "kaggle_group",
                    "duplicate_of", "width", "height", "flood_fraction",
                    "exclude", "exclude_reason"])
        for name, source, role, _ in rows:
            w.writerow([name, name.replace(".jpg", ".png"), source, name,
                        "jpeg", role, "", "0" if role == "kaggle" else "-1",
                        "-1", "", "192", "192", "0.3", "", ""])


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    fssd, combined = tmp_path / "fssd", tmp_path / "combined"
    _make_fssd(fssd)
    _make_combined(combined)
    return fssd, combined


def test_run_check_finds_planted_duplicates_and_fails_gate(roots):
    fssd, combined = roots
    report = overlap.run_check(fssd, combined, threshold=8)

    assert report["n_fssd_images"] == 3
    assert report["n_combined_images"] == 3
    assert report["passed"] is False

    blocking = {(h["combined_image"], h["role"]) for h in report["blocking"]}
    warnings = {(h["combined_image"], h["role"]) for h in report["warnings"]}
    assert ("floodnet_200.jpg", "floodnet_test") in blocking
    assert ("kaggle_000.jpg", "kaggle") in warnings
    assert all(h["distance"] <= 8 for h in report["blocking"] + report["warnings"])
    # every hit names the FSSD file it collides with
    assert all(h["fssd_image"] for h in report["blocking"] + report["warnings"])


def test_run_check_passes_when_no_duplicates(tmp_path: Path):
    fssd, combined = tmp_path / "fssd", tmp_path / "combined"
    _make_fssd(fssd)
    _make_combined(combined)
    # replace the two colliding images with unrelated content
    _structured(1001, size=192).save(combined / "Image" / "kaggle_000.jpg")
    _structured(1002, size=192).save(combined / "Image" / "floodnet_200.jpg")
    report = overlap.run_check(fssd, combined, threshold=8)
    assert report["passed"] is True
    assert report["blocking"] == [] and report["warnings"] == []
