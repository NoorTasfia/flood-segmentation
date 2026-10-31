"""Tests for benchmark_onnxruntime thread pinning (torch-free: needs only onnx + ort).

The M2 benchmark reports both the default ORT thread pool and a pinned
single-thread configuration (portability story for weak edge CPUs), so the
benchmark helper must accept an intra-op thread count.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", exc_type=Exception)
ort = pytest.importorskip("onnxruntime", exc_type=Exception)

from floodlite.benchmark import benchmark_onnxruntime  # noqa: E402

EXPECTED_KEYS = {"p50_ms", "p95_ms", "mean_ms", "std_ms", "fps"}


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    """A minimal 1-conv ONNX model with the (1,3,H,W) input benchmarks expect."""
    from onnx import TensorProto, helper
    w = np.zeros((1, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        name="tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 32, 32])],
        initializer=[helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # stay within the installed onnxruntime's max IR version
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_default_threading_returns_expected_schema(tiny_onnx):
    res = benchmark_onnxruntime(tiny_onnx, img_size=32, n_warm=2, n_iter=5)
    assert set(res) == EXPECTED_KEYS
    assert res["p50_ms"] > 0


def test_single_thread_pinning_returns_expected_schema(tiny_onnx):
    res = benchmark_onnxruntime(tiny_onnx, img_size=32, n_warm=2, n_iter=5,
                                intra_op_threads=1)
    assert set(res) == EXPECTED_KEYS
    assert res["p50_ms"] > 0
