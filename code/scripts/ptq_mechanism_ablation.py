"""PTQ mechanism ablation (reviewer item C1/M2/M3).

The paper's INT8 recipe uses per-tensor weight scales. Per-channel weights are
the usual remedy for depthwise MobileNets, so a reviewer can fairly ask whether
the MobileNetV3-Small / MobileViT-XXS collapse is a recipe artefact rather than
an architecture property. This script re-quantizes the FSSD no-KD students (the
Table 4 checkpoints) and the teacher under controlled variants that share the
SAME fold, FP32 export and calibration batches:

  pt          per-tensor weights        (the paper's recipe, replication check)
  pc          per-channel weights       (everything else identical)
  pc_keep_hs  per-channel, and every Conv/Gemm/MatMul fed by a HardSwish or
              HardSigmoid output stays FP32 (tests the HardSwish-range claim)
  pc_keep_mm  per-channel, MatMul/Gemm (attention + MLP) stay FP32; only Conv
              is quantized (tests the attention claim for MobileViT)
  pc_keep_early  MobileViT only: per-channel, the stem and stages_0/1 Convs
              (located by ptq_mobilevit_diagnostic.py) stay FP32

Variants that would be a no-op for a model (e.g. pc_keep_hs on a graph with no
HardSwish) are skipped. Metrics follow quantize_all.py (batch-mean over the
fold's validation loader, batch 8) so numbers are comparable with Table 4.

Usage:
    python scripts/ptq_mechanism_ablation.py --data_root <FSSD> \
        --ckpt_dir runs/3fold --out runs/3fold/ptq_mechanism.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process

from floodlite.data import make_loaders
from floodlite.export import export_onnx
from floodlite.models import STUDENT_BACKBONES, make_student, make_teacher
from floodlite.quantize import evaluate_onnx, file_size_mb

QUANT_OPS = ("Conv", "Gemm", "MatMul")
EARLY_PREFIXES = ("/encoder/model/stem/", "/encoder/model/stages_0/",
                  "/encoder/model/stages_1/")


class ListReader:
    def __init__(self, batches, input_name):
        self._it = iter([{input_name: b} for b in batches])

    def get_next(self):
        return next(self._it, None)


def preprocess(fp_path: Path) -> Path:
    out = fp_path.with_name(fp_path.stem + "_preproc.onnx")
    try:
        quant_pre_process(str(fp_path), str(out), skip_optimization=False,
                          skip_onnx_shape=False, skip_symbolic_shape=False,
                          auto_merge=True)
    except Exception:
        quant_pre_process(str(fp_path), str(out), skip_optimization=False,
                          skip_onnx_shape=False, skip_symbolic_shape=True)
    return out


def graph_info(path: Path) -> dict:
    g = onnx.load(str(path)).graph
    producer = {o: n for n in g.node for o in n.output}
    op_counts: dict[str, int] = {}
    for n in g.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    # timm exports HardSwish as x * HardSigmoid(x) (and SE gates as
    # x * HardSigmoid(.)), so a "hard-activation output" is either a
    # HardSwish node or a Mul with a HardSigmoid input.
    def is_hard_act(node) -> bool:
        if node.op_type in ("HardSwish", "HardSigmoid"):
            return True
        return node.op_type == "Mul" and any(
            i in producer and producer[i].op_type == "HardSigmoid"
            for i in node.input)
    fed_by_hs = [n.name for n in g.node if n.op_type in QUANT_OPS and any(
        i in producer and is_hard_act(producer[i]) for i in n.input)]
    return {"op_counts": op_counts, "fed_by_hardswish": fed_by_hs}


def quantize(preproc: Path, out: Path, batches, *, per_channel: bool,
             op_types=QUANT_OPS, exclude=()):
    sess = ort.InferenceSession(str(preproc), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    del sess
    quantize_static(
        str(preproc), str(out), ListReader(batches, name),
        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8, per_channel=per_channel,
        op_types_to_quantize=list(op_types), nodes_to_exclude=list(exclude),
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.999,
                       "CalibMaxIntermediateOutputs": 50},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="+",
                    default=list(STUDENT_BACKBONES) + ["teacher"])
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out", default="runs/3fold/ptq_mechanism.json")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="run only these variants (others kept from --out)")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_meta"] = {"onnxruntime": ort.__version__, "torch": torch.__version__,
                        "onnx": onnx.__version__, "calib_images": 80,
                        "calib": "Percentile 99.999, 10 batches x 8 train-fold images",
                        "metric": "batch-mean IoU over fold val loader (as quantize_all.py)"}

    for fold in args.folds:
        torch.manual_seed(1000 + fold)
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=8)
        calib = []
        for x, _ in tr:
            calib.append(x.numpy().astype("float32"))
            if len(calib) == 10:
                break
        fk = f"fold_{fold}"
        results.setdefault(fk, {})
        for m in args.models:
            if m == "teacher":
                ckpt = Path(args.ckpt_dir) / f"teacher_fold{fold}.pt"
                model = make_teacher()
            else:
                ckpt = Path(args.ckpt_dir) / f"{m}_none_fold{fold}.pt"
                model = make_student(m)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            fp = work / f"{m}_fold{fold}_fp32.onnx"
            export_onnx(model, fp, img_size=256, opset=17)
            pre = preprocess(fp)
            info = graph_info(pre)
            fp_m = evaluate_onnx(pre, va)
            rec = {"fp32": fp_m, "fp32_size_mb": file_size_mb(fp),
                   "op_counts": info["op_counts"],
                   "n_quant_nodes_fed_by_hardswish": len(info["fed_by_hardswish"]),
                   "variants": {}}
            variants = {"pt": dict(per_channel=False),
                        "pc": dict(per_channel=True)}
            if info["fed_by_hardswish"]:
                variants["pc_keep_hs"] = dict(per_channel=True,
                                              exclude=info["fed_by_hardswish"])
            if info["op_counts"].get("MatMul", 0) > 0:
                variants["pc_keep_mm"] = dict(per_channel=True, op_types=("Conv",))
            if m == "mobilevit_xxs":
                early = [n.name for n in onnx.load(str(pre)).graph.node
                         if n.op_type == "Conv" and n.name.startswith(EARLY_PREFIXES)]
                variants["pc_keep_early"] = dict(per_channel=True, exclude=early)
            old = results[fk].get(m, {}).get("variants", {})
            if args.variants:
                rec["variants"] = {k: v for k, v in old.items() if k not in args.variants}
                variants = {k: v for k, v in variants.items() if k in args.variants}
            for vname, kw in variants.items():
                q = work / f"{m}_fold{fold}_{vname}_int8.onnx"
                t0 = time.time()
                try:
                    quantize(pre, q, calib, **kw)
                    qm = evaluate_onnx(q, va)
                    rec["variants"][vname] = {
                        "iou": qm["iou"], "delta_iou": qm["iou"] - fp_m["iou"],
                        "metrics": qm, "size_mb": file_size_mb(q)}
                    print(f"{fk} {m:18s} {vname:11s} FP32 {fp_m['iou']:.4f} -> "
                          f"INT8 {qm['iou']:.4f} ({qm['iou']-fp_m['iou']:+.4f}) "
                          f"{file_size_mb(q):.2f} MB  {time.time()-t0:.0f}s", flush=True)
                except Exception as e:  # record, keep going
                    rec["variants"][vname] = {"error": repr(e)}
                    print(f"{fk} {m} {vname} FAILED: {e!r}", flush=True)
            results[fk][m] = rec
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
