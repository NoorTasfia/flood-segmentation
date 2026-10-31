"""Tests for scripts/eval_zero_shot.py (D1: FSSD checkpoints -> combined eval roles).

Needs torch (+ cv2 for the synthetic dataset); skipped where unavailable.
"""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=Exception)
cv2 = pytest.importorskip("cv2", exc_type=Exception)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_zero_shot.py"
spec = importlib.util.spec_from_file_location("eval_zero_shot", SCRIPT)
ezs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ezs)

COLUMNS = [
    "image", "mask", "source", "source_image", "file_format", "role",
    "source_split", "kaggle_fold", "kaggle_group", "duplicate_of",
    "width", "height", "flood_fraction", "exclude", "exclude_reason",
]


@pytest.fixture
def combined_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i, (prefix, source, role) in enumerate(
            [("floodnet_t", "floodnet", "floodnet_test")] * 3
            + [("afs_", "afs", "external_test")] * 2):
        name = f"{prefix}{i:03d}.jpg"
        rows.append(dict(zip(COLUMNS, [
            name, f"{prefix}{i:03d}.png", source, name, "jpeg", role, "",
            "-1", "-1", "", "64", "64", "0.3", "", ""])))
    (tmp_path / "Image").mkdir()
    (tmp_path / "Mask").mkdir()
    for r in rows:
        cv2.imwrite(str(tmp_path / "Image" / r["image"]),
                    rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(tmp_path / "Mask" / r["mask"]),
                    rng.integers(0, 2, (64, 64), dtype=np.uint8) * 255)
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


class _TinySeg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_evaluate_per_image_shapes_and_keys(combined_root):
    loaders = ezs.make_eval_loaders(combined_root, img_size=64, batch_size=2,
                                    num_workers=0)
    assert set(loaders) == {"floodnet_test", "external_test"}
    res = ezs.evaluate_per_image(_TinySeg(), loaders["external_test"], "cpu")
    assert res["n_images"] == 2
    for k in ("iou_mean", "iou_std", "accuracy_mean", "f1_mean"):
        assert k in res
    assert len(res["per_image_iou"]) == 2
    assert all(0.0 <= v <= 1.0 for v in res["per_image_iou"])


def test_evaluate_onnx_per_image_matches_schema(combined_root, tmp_path):
    onnx = pytest.importorskip("onnx", exc_type=Exception)
    pytest.importorskip("onnxruntime", exc_type=Exception)
    from onnx import TensorProto, helper
    w = np.zeros((1, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 64, 64])],
        [helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # stay within the installed onnxruntime's max IR version
    model_path = tmp_path / "tiny.onnx"
    onnx.save(model, str(model_path))

    loaders = ezs.make_eval_loaders(combined_root, img_size=64, batch_size=1,
                                    num_workers=0)
    res = ezs.evaluate_onnx_per_image(model_path, loaders["floodnet_test"])
    assert res["n_images"] == 3
    assert "iou_mean" in res and len(res["per_image_iou"]) == 3
