"""D3b: per-channel INT8 PTQ of the combined-dataset (D2) students.

Companion to ``quantize_combined.py`` (per-tensor recipe behind Table T11).
Re-quantizes the same checkpoints with the per-channel weight-scale variant
of ``ptq_mechanism_ablation.py`` (static QDQ, QInt8 weights + activations,
Percentile-99.999 calibration, 10 batches x 8 = 80 fold-training images) and
evaluates FP32 and INT8 ONNX per image on the fold's val split, the FloodNet
test split and the AIFloodSense external split.

Reuses the pre-processed FP32 exports written by ``quantize_combined.py``
(``{tag}_fold{N}_fp32_preproc.onnx``) when present; otherwise exports from
``--ckpt_dir``.

Usage:
    python scripts/quantize_combined_pc.py --data_root /path/to/combined \
        --exports_dir /path/to/combined-exports [--ckpt_dir ...] \
        [--folds 0 1 2] [--models mobilenetv3_small ...] [--kd none comb] \
        [--out runs/combined/quantized_pc.json]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnxruntime as ort
import torch

from floodlite.data import make_combined_loaders
from floodlite.models import STUDENT_BACKBONES, make_student
from floodlite.quantize import file_size_mb
from eval_zero_shot import evaluate_onnx_per_image
from ptq_mechanism_ablation import preprocess, quantize
from quantize_combined import SPLITS, _brief


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--exports_dir", required=True)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=list(STUDENT_BACKBONES))
    ap.add_argument("--kd", nargs="+", default=["none", "comb"])
    ap.add_argument("--out", default="runs/combined/quantized_pc.json")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    exports = Path(args.exports_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["recipe"] = ("static QDQ, per-channel weights (QInt8), QInt8 activations, "
                         "Percentile 99.999, 10 batches x 8 = 80 fold-train calibration images "
                         "(ptq_mechanism_ablation.quantize, per_channel=True)")
    results["onnxruntime"] = ort.__version__
    results["torch"] = torch.__version__

    for fold in args.folds:
        torch.manual_seed(1000 + fold)
        loaders = make_combined_loaders(args.data_root, fold=fold, batch_size=8,
                                        num_workers=args.num_workers)
        calib = []
        for x, _ in loaders["train"]:
            calib.append(x.numpy().astype("float32"))
            if len(calib) == 10:
                break
        fk = f"fold_{fold}"
        results.setdefault(fk, {})
        for sname in args.models:
            for kd in args.kd:
                tag = f"{sname}_{kd}"
                fp_onnx = exports / f"{tag}_fold{fold}_fp32.onnx"
                pre = exports / f"{tag}_fold{fold}_fp32_preproc.onnx"
                if not pre.exists():
                    if args.ckpt_dir is None:
                        print(f"SKIP {tag} fold {fold}: no preproc export and no --ckpt_dir")
                        continue
                    from floodlite.export import export_onnx
                    model = make_student(sname).cpu()
                    model.load_state_dict(torch.load(
                        Path(args.ckpt_dir) / f"{tag}_fold{fold}.pt", map_location="cpu"))
                    model.eval()
                    export_onnx(model, fp_onnx, img_size=256, opset=17)
                    pre = preprocess(fp_onnx)
                q = exports / f"{tag}_fold{fold}_pc_int8.onnx"
                t0 = time.time()
                try:
                    quantize(pre, q, calib, per_channel=True)
                    entry = {"fp32_size_mb": file_size_mb(fp_onnx) if fp_onnx.exists() else None,
                             "int8_pc_size_mb": file_size_mb(q),
                             "quantize_seconds": round(time.time() - t0, 1)}
                    for split in SPLITS:
                        fp = evaluate_onnx_per_image(pre, loaders[split])
                        qm = evaluate_onnx_per_image(q, loaders[split])
                        entry[split] = {"fp32": _brief(fp), "int8_pc": _brief(qm)}
                    results[fk][tag] = entry
                    t = entry["test"]
                    print(f"fold {fold} {tag:26s} "
                          f"val {entry['val']['fp32']['iou_mean']:.3f}->{entry['val']['int8_pc']['iou_mean']:.3f}  "
                          f"test flood {t['fp32'].get('iou_flood_mean', 0):.3f}->{t['int8_pc'].get('iou_flood_mean', 0):.3f}  "
                          f"FAR {t['fp32'].get('false_alarm_rate', 0):.3f}->{t['int8_pc'].get('false_alarm_rate', 0):.3f}  "
                          f"ext {entry['external']['fp32']['iou_mean']:.3f}->{entry['external']['int8_pc']['iou_mean']:.3f}  "
                          f"{time.time()-t0:.0f}s", flush=True)
                except Exception as e:  # record, keep going
                    results[fk][tag] = {"error": repr(e)}
                    print(f"fold {fold} {tag}: FAILED {e!r}", flush=True)
                out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
