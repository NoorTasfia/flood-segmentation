"""Sensitivity-guided mixed-precision INT8 PTQ (resubmission plan Task 6).

For each (model, fold): export/preprocess the FP32 ONNX, draw the same 80
calibration images as ptq_mechanism_ablation.py, measure each layer group's
sensitivity (IoU drop on the calibration images when only that group is
quantized, per-channel), then greedily keep the most sensitive groups in FP32
until the mixed artifact's calibration loss is within --tau. The validation
fold is touched only to report the final artifact.

Usage (FSSD checkpoints):
    python scripts/ptq_mixed_precision.py --dataset fssd --data_root <FSSD> \
        --ckpt_dir <dir with {model}_none_fold{f}.pt, teacher_fold{f}.pt> \
        --work_dir <onnx work dir> --out runs/3fold/ptq_mixed.json
Combined-corpus MobileViT (none/comb/gated checkpoints):
    python scripts/ptq_mixed_precision.py --dataset combined --data_root <combined> \
        --ckpt_dir <combined-ckpts> --models mobilevit_xxs --kd none comb gated \
        --work_dir <dir> --out runs/combined/ptq_mixed_mobilevit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from floodlite.data import make_combined_loaders, make_loaders
from floodlite.export import export_onnx
from floodlite.metrics import compute_metrics
from floodlite.mixed_precision import group_quant_nodes, int8_mac_share, select_fp32_groups
from floodlite.models import STUDENT_BACKBONES, make_student, make_teacher
from floodlite.quantize import evaluate_onnx, file_size_mb
from ptq_mechanism_ablation import preprocess, quantize


def iou_on_batches(onnx_path: Path, batches: list[tuple[np.ndarray, torch.Tensor]]) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ious = []
    for x, y in batches:
        logits = torch.from_numpy(sess.run(None, {name: x})[0])
        ious.append(compute_metrics(logits, y).iou)
    return float(np.mean(ious))


def calibration_batches(train_loader, n_batches: int = 10):
    out = []
    for x, y in train_loader:
        out.append((x.numpy().astype("float32"), y))
        if len(out) == n_batches:
            break
    return out


def load_model(name: str, kd: str, fold: int, ckpt_dir: Path):
    if name == "teacher":
        return make_teacher(), ckpt_dir / f"teacher_fold{fold}.pt", f"teacher_fold{fold}"
    tag = f"{name}_{kd}_fold{fold}"
    return make_student(name), ckpt_dir / f"{tag}.pt", tag


def run_one(pre: Path, work: Path, tag: str, calib, va, tau: float) -> dict:
    groups = group_quant_nodes(pre)
    all_nodes = [n for ns in groups.values() for n in ns]
    x_calib = [b[0] for b in calib]
    fp_calib = iou_on_batches(pre, calib)

    def quantized_iou(exclude: list[str], path: Path) -> float:
        quantize(pre, path, x_calib, per_channel=True, exclude=exclude)
        return iou_on_batches(path, calib)

    # 1. sensitivity: only this group quantized
    sens: dict[str, float] = {}
    tmp = work / f"{tag}_mp_tmp.onnx"
    for g, nodes in groups.items():
        t0 = time.time()
        others = [n for n in all_nodes if n not in nodes]
        sens[g] = fp_calib - quantized_iou(others, tmp)
        print(f"  sens {g:34s} {sens[g]:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # 2. greedy fallback on the calibration images
    trajectory: list[dict] = []

    def loss_after(selected: list[str]) -> float:
        exclude = [n for g in selected for n in groups[g]]
        loss = fp_calib - quantized_iou(exclude, tmp)
        trajectory.append({"fp32_groups": list(selected), "calib_loss": loss})
        print(f"  greedy fp32={selected} calib_loss={loss:+.4f}", flush=True)
        return loss

    selected = select_fp32_groups(sens, loss_after, tau=tau)
    exclude = [n for g in selected for n in groups[g]]

    # 3. final artifact, reported on the validation fold
    mixed = work / f"{tag}_mixed_int8.onnx"
    quantize(pre, mixed, x_calib, per_channel=True, exclude=exclude)
    tmp.unlink(missing_ok=True)
    fp_val = evaluate_onnx(pre, va)
    mx_val = evaluate_onnx(mixed, va)
    pc = work / f"{tag}_pc_int8.onnx"
    if not pc.exists():
        quantize(pre, pc, x_calib, per_channel=True)
    pc_val = evaluate_onnx(pc, va)
    return {
        "groups": {g: len(ns) for g, ns in groups.items()},
        "sensitivity_calib": sens,
        "fp32_calib_iou": fp_calib,
        "trajectory": trajectory,
        "selected_fp32_groups": selected,
        "n_fp32_nodes": len(exclude),
        "budget_met": bool(trajectory and trajectory[-1]["calib_loss"] <= tau),
        "fp32_iou": fp_val["iou"],
        "pc_iou": pc_val["iou"],
        "mixed_iou": mx_val["iou"],
        "delta_pc": pc_val["iou"] - fp_val["iou"],
        "delta_mixed": mx_val["iou"] - fp_val["iou"],
        "metrics_mixed": mx_val,
        "int8_mac_share": int8_mac_share(pre, exclude),
        "size_mb_fp32": file_size_mb(pre),
        "size_mb_pc": file_size_mb(pc),
        "size_mb_mixed": file_size_mb(mixed),
        "artifact": str(mixed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("fssd", "combined"), default="fssd")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=list(STUDENT_BACKBONES) + ["teacher"])
    ap.add_argument("--kd", nargs="+", default=["none"], help="student KD tags (combined: none comb gated)")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--out", default="runs/3fold/ptq_mixed.json")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_meta"] = {
        "onnxruntime": ort.__version__, "torch": torch.__version__, "onnx": onnx.__version__,
        "tau": args.tau, "dataset": args.dataset,
        "recipe": "per-channel static QDQ INT8, Percentile 99.999, Conv/Gemm/MatMul; FP32 fallback "
                  "by layer group chosen greedily by calibration-image sensitivity (80 train images)",
        "selection_sees": "calibration images only; validation fold used for the reported numbers",
    }
    for fold in args.folds:
        torch.manual_seed(1000 + fold)
        if args.dataset == "fssd":
            tr, va = make_loaders(args.data_root, fold=fold, batch_size=8, num_workers=args.num_workers)
        else:
            ld = make_combined_loaders(args.data_root, fold=fold, batch_size=8, num_workers=args.num_workers)
            tr, va = ld["train"], ld["val"]
        calib = calibration_batches(tr)
        fk = f"fold_{fold}"
        results.setdefault(fk, {})
        for m in args.models:
            for kd in (["none"] if m == "teacher" else args.kd):
                model, ckpt, tag = load_model(m, kd, fold, Path(args.ckpt_dir))
                if not ckpt.exists():
                    print(f"[skip] {ckpt} missing", flush=True); continue
                model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.eval()
                pre = work / f"{tag}_fp32_preproc.onnx"
                if not pre.exists():
                    fp = work / f"{tag}_fp32.onnx"
                    export_onnx(model, fp, img_size=256, opset=17)
                    pre = preprocess(fp)
                print(f"== {fk} {tag}", flush=True)
                t0 = time.time()
                rec = run_one(pre, work, tag, calib, va, args.tau)
                rec["seconds"] = time.time() - t0
                results[fk][tag] = rec
                print(f"   FP32 {rec['fp32_iou']:.4f}  pc {rec['delta_pc']:+.4f}  mixed {rec['delta_mixed']:+.4f} "
                      f"fp32_groups={rec['selected_fp32_groups']} int8_mac_share={rec['int8_mac_share']:.2f} "
                      f"size {rec['size_mb_mixed']:.1f} MB  ({rec['seconds']:.0f}s)", flush=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
