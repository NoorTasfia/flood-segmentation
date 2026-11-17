"""Sensitivity-guided mixed-precision INT8 PTQ (FP32 fallback by layer group).

On top of the per-channel static QDQ recipe, this module decides *which layer
groups to leave in FP32*:

1. ``group_quant_nodes`` splits the quantizable nodes (Conv/Gemm/MatMul) of an
   ONNX graph into groups: one per encoder stage, one per decoder block, and
   the segmentation head.
2. The caller measures each group's *sensitivity*: the IoU drop on the
   calibration images when only that group is quantized.
3. ``select_fp32_groups`` moves groups to FP32 in descending sensitivity until
   the mixed artifact's loss on the calibration images is within ``tau``.

Everything the selection sees comes from the calibration images; the
validation fold is only used afterwards to report the result.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnx

QUANT_OPS = ("Conv", "Gemm", "MatMul")
_STAGE = re.compile(r"^(_?blocks|stages)[._]\d+$")


def group_key(node_name: str) -> str:
    """Map an ONNX node name to its layer group.

    Encoder nodes group by stage (``encoder/blocks.3``, ``encoder/_blocks.12``,
    ``encoder/model/stages_1``) or, outside the stages, by their module
    (``encoder/conv_stem``, ``encoder/model/stem``); decoder nodes by block
    (``decoder/blocks.2``); the head is one group (``segmentation_head``).
    """
    parts = [p for p in node_name.split("/") if p]
    if not parts:
        return node_name
    if parts[0] == "decoder":
        return "/".join(parts[:2])
    if parts[0] == "segmentation_head":
        return parts[0]
    for i, p in enumerate(parts):
        if _STAGE.match(p):
            return "/".join(parts[: i + 1])
    return "/".join(parts[: min(3, max(1, len(parts) - 1))])


def quant_nodes(model: onnx.ModelProto) -> list[str]:
    return [n.name for n in model.graph.node if n.op_type in QUANT_OPS]


def group_quant_nodes(onnx_path: str | Path) -> dict[str, list[str]]:
    """Group the quantizable node names of an ONNX file by ``group_key``."""
    model = onnx.load(str(onnx_path))
    groups: dict[str, list[str]] = {}
    for name in quant_nodes(model):
        groups.setdefault(group_key(name), []).append(name)
    return groups


def select_fp32_groups(sensitivity: dict[str, float],
                       loss_after: Callable[[list[str]], float],
                       tau: float = 0.02) -> list[str]:
    """Greedy FP32 fallback: add groups in descending sensitivity until
    ``loss_after(selected) <= tau``; return every group if never satisfied.

    ``loss_after`` receives the list of groups kept in FP32 and returns the
    IoU loss (FP32 minus mixed) of the resulting artifact on the calibration
    images. It is called first with ``[]``.
    """
    order = sorted(sensitivity, key=lambda g: sensitivity[g], reverse=True)
    selected: list[str] = []
    if loss_after(selected) <= tau:
        return selected
    for g in order:
        selected.append(g)
        if loss_after(selected) <= tau:
            return selected
    return selected


# ---------------------------------------------------------------- MAC accounting

def _shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    out: dict[str, list[int]] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        out[vi.name] = dims
    for init in model.graph.initializer:
        out[init.name] = list(init.dims)
    return out


def node_macs(onnx_path: str | Path) -> dict[str, int]:
    """Multiply-accumulates per Conv/Gemm/MatMul node (batch dimension counted as 1)."""
    model = onnx.load(str(onnx_path))
    shapes = _shape_map(model)
    macs: dict[str, int] = {}
    for n in model.graph.node:
        if n.op_type not in QUANT_OPS:
            continue
        out_shape = shapes.get(n.output[0])
        if out_shape is None:
            macs[n.name] = 0
            continue
        out_elems = int(np.prod(out_shape[1:])) if len(out_shape) > 1 else int(np.prod(out_shape))
        if n.op_type == "Conv":
            w = shapes.get(n.input[1], [0, 0])
            groups = next((a.i for a in n.attribute if a.name == "group"), 1)
            cin_per_group = w[1] if len(w) > 1 else 0
            kernel = int(np.prod(w[2:])) if len(w) > 2 else 1
            macs[n.name] = out_elems * cin_per_group * kernel
        else:  # Gemm / MatMul: out elements x inner dimension of input A
            a = shapes.get(n.input[0], [1])
            macs[n.name] = out_elems * int(a[-1])
    return macs


def int8_mac_share(onnx_path: str | Path, excluded: Iterable[str]) -> float:
    """Share of Conv/Gemm/MatMul MACs that remain INT8 when ``excluded`` stay FP32."""
    macs = node_macs(onnx_path)
    total = sum(macs.values())
    if total == 0:
        return 0.0
    kept_fp32 = sum(macs[n] for n in excluded if n in macs)
    return (total - kept_fp32) / total
