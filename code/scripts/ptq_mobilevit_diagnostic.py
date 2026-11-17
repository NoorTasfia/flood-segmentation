"""Locate MobileViT-XXS's INT8 collapse (follow-up to ptq_mechanism_ablation.py).

Per-channel weights and FP32 attention MatMuls both leave MobileViT-XXS ~0.23-0.33
IoU below FP32, so the damage sits in its quantized Conv layers. This script
quantizes ONE layer group at a time (per-channel, same recipe and calibration
batches as the ablation; every other node stays FP32) and reports each group's
IoU drop, plus the complementary "all but this group" run.

Groups follow the timm/smp node names: stem+stages_0/1 (MobileNetV2 blocks),
stages_k.0 (downsampling inverted residual), stages_k.1 (MobileViT block convs:
local kxk, 1x1 in, proj, fusion), final_conv, decoder, segmentation head.

Usage:
    python scripts/ptq_mobilevit_diagnostic.py --data_root <FSSD> \
        --work_dir <dir with mobilevit_xxs_fold{f}_fp32_preproc.onnx> \
        --out runs/3fold/ptq_mobilevit_groups.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnx
import torch

from floodlite.data import make_loaders
from floodlite.quantize import evaluate_onnx
from ptq_mechanism_ablation import quantize

GROUPS = {
    "stem": ("/encoder/model/stem/",),
    "s0": ("/encoder/model/stages_0/",),
    "s1": ("/encoder/model/stages_1/",),
    "stem_s0_s1": ("/encoder/model/stem/", "/encoder/model/stages_0/", "/encoder/model/stages_1/"),
    "s2_down": ("/encoder/model/stages_2/stages_2.0/",),
    "s2_mvit": ("/encoder/model/stages_2/stages_2.1/",),
    "s3_down": ("/encoder/model/stages_3/stages_3.0/",),
    "s3_mvit": ("/encoder/model/stages_3/stages_3.1/",),
    "s4_down": ("/encoder/model/stages_4/stages_4.0/",),
    "s4_mvit": ("/encoder/model/stages_4/stages_4.1/",),
    "final_conv": ("/encoder/model/final_conv/",),
    "decoder": ("/decoder/",),
    "head": ("/segmentation_head/",),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs/3fold/ptq_mobilevit_groups.json")
    ap.add_argument("--groups", nargs="+", default=None, help="subset of GROUPS")
    ap.add_argument("--modes", nargs="+", default=["only", "all_but"])
    args = ap.parse_args()
    work = Path(args.work_dir)
    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res["_meta"] = {"recipe": "per-channel QDQ INT8, Conv only (MatMul FP32), "
                    "Percentile 99.999, same 80 calibration images as ptq_mechanism.json",
                    "only": "only this group quantized, rest FP32",
                    "all_but": "every Conv except this group quantized"}
    for fold in args.folds:
        torch.manual_seed(1000 + fold)
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=8)
        calib = []
        for x, _ in tr:
            calib.append(x.numpy().astype("float32"))
            if len(calib) == 10:
                break
        pre = work / f"mobilevit_xxs_fold{fold}_fp32_preproc.onnx"
        convs = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
        fp_iou = evaluate_onnx(pre, va)["iou"]
        fk = f"fold_{fold}"
        res.setdefault(fk, {"groups": {}})["fp32_iou"] = fp_iou
        for g, prefixes in GROUPS.items():
            if args.groups and g not in args.groups:
                continue
            members = [c for c in convs if c.startswith(prefixes)]
            others = [c for c in convs if c not in members]
            rec = res[fk]["groups"].get(g, {})
            rec["n_conv"] = len(members)
            for mode, exclude in (("only", others), ("all_but", members)):
                if mode not in args.modes:
                    continue
                q = work / f"mvit_diag_fold{fold}_{g}_{mode}.onnx"
                t0 = time.time()
                quantize(pre, q, calib, per_channel=True, op_types=("Conv",), exclude=exclude)
                iou = evaluate_onnx(q, va)["iou"]
                rec[mode] = {"iou": iou, "delta_iou": iou - fp_iou}
                q.unlink(missing_ok=True)
                print(f"{fk} {g:11s} {mode:7s} IoU {iou:.4f} ({iou-fp_iou:+.4f}) "
                      f"{time.time()-t0:.0f}s", flush=True)
            res[fk]["groups"][g] = rec
            out_path.write_text(json.dumps(res, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
