"""Tests for floodlite.mixed_precision: sensitivity-guided FP32 fallback for
static INT8 PTQ. Selection is a pure function; MAC accounting is tested on a
tiny hand-built ONNX graph."""
from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx", exc_type=Exception)
np = pytest.importorskip("numpy")

from floodlite.mixed_precision import (  # noqa: E402
    group_key,
    group_quant_nodes,
    int8_mac_share,
    node_macs,
    select_fp32_groups,
)


@pytest.mark.parametrize("name,key", [
    ("/encoder/blocks.3/blocks.3.1/conv_dw/Conv", "encoder/blocks.3"),          # timm MobileNetV3 / Lite0
    ("/encoder/conv_stem/Conv", "encoder/conv_stem"),
    ("/encoder/_blocks.12/_se_reduce/Conv", "encoder/_blocks.12"),              # smp EfficientNet-B0 teacher
    ("/encoder/model/stages_1/stages_1.2/conv/Conv", "encoder/model/stages_1"),  # timm MobileViT
    ("/encoder/model/stages_3/stages_3.1/transformer/MatMul_7", "encoder/model/stages_3"),
    ("/encoder/model/stem/conv/Conv", "encoder/model/stem"),
    ("/encoder/model/final_conv/conv/Conv", "encoder/model/final_conv"),
    ("/decoder/blocks.2/conv1/conv1.0/Conv", "decoder/blocks.2"),
    ("/segmentation_head/segmentation_head.0/Conv", "segmentation_head"),
])
def test_group_key_is_one_encoder_stage_decoder_block_or_head(name, key):
    assert group_key(name) == key


def test_select_adds_most_sensitive_groups_until_budget_met():
    sens = {"a": 0.3, "b": 0.1, "c": 0.01}
    losses = {0: 0.3, 1: 0.1, 2: 0.015, 3: 0.0}
    calls = []

    def loss_after(selected):
        calls.append(list(selected))
        return losses[len(selected)]

    assert select_fp32_groups(sens, loss_after, tau=0.02) == ["a", "b"]
    assert calls[0] == []            # the all-INT8 artifact is tried first


def test_select_returns_empty_when_all_int8_is_within_budget():
    assert select_fp32_groups({"a": 0.5}, lambda s: 0.01, tau=0.02) == []


def test_select_returns_every_group_when_budget_never_met():
    sens = {"a": 0.3, "b": 0.2}
    assert select_fp32_groups(sens, lambda s: 1.0, tau=0.02) == ["a", "b"]


def _two_conv_graph(tmp_path):
    """x[1,3,8,8] -> Conv(3->4, 3x3, pad 1) -> Conv(4->2, 1x1) -> y[1,2,8,8]."""
    from onnx import TensorProto, helper, numpy_helper
    w1 = numpy_helper.from_array(np.zeros((4, 3, 3, 3), np.float32), "w1")
    w2 = numpy_helper.from_array(np.zeros((2, 4, 1, 1), np.float32), "w2")
    n1 = helper.make_node("Conv", ["x", "w1"], ["h"], name="/encoder/model/stem/conv/Conv",
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n2 = helper.make_node("Conv", ["h", "w2"], ["y"], name="/decoder/blocks.0/conv1/conv1.0/Conv",
                          kernel_shape=[1, 1])
    g = helper.make_graph([n1, n2], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8, 8])],
                          initializer=[w1, w2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    p = tmp_path / "two_conv.onnx"
    onnx.save(m, str(p))
    return p


def test_node_macs_counts_conv_multiply_adds(tmp_path):
    macs = node_macs(_two_conv_graph(tmp_path))
    assert macs["/encoder/model/stem/conv/Conv"] == 8 * 8 * 4 * (3 * 3 * 3)
    assert macs["/decoder/blocks.0/conv1/conv1.0/Conv"] == 8 * 8 * 2 * (4 * 1 * 1)


def test_int8_mac_share_excludes_fp32_nodes(tmp_path):
    p = _two_conv_graph(tmp_path)
    total = 8 * 8 * 4 * 27 + 8 * 8 * 2 * 4
    assert int8_mac_share(p, excluded=[]) == pytest.approx(1.0)
    assert int8_mac_share(p, excluded=["/encoder/model/stem/conv/Conv"]) == pytest.approx((8 * 8 * 2 * 4) / total)


def test_group_quant_nodes_groups_by_stage(tmp_path):
    groups = group_quant_nodes(_two_conv_graph(tmp_path))
    assert groups == {"encoder/model/stem": ["/encoder/model/stem/conv/Conv"],
                      "decoder/blocks.0": ["/decoder/blocks.0/conv1/conv1.0/Conv"]}
