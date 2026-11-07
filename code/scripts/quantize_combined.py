"""D3: INT8 PTQ of the combined-dataset (D2) students, evaluated per image.

Applies the exact recipe behind runs/3fold/quantized.json
(``quantize_int8_onnx_static``: static QDQ, per-tensor, Percentile-99.999%
calibration, 10 batches x 8 = 80 calibration images drawn from the fold's
training loader) to every ``{student}_{kd}_fold{N}.pt`` checkpoint of the D2
runs, then evaluates FP32 and INT8 ONNX per image (macro IoU, flood-only IoU,
false-alarm rate) on the fold's val split, the FloodNet test split and the
AIFloodSense external split.

Usage:
    python scripts/quantize_combined.py --data_root /path/to/combined \
        --ckpt_dir /path/to/combined-ckpts [--folds 0 1 2] [--kd none comb] \
        [--exports_dir exports_combined] [--out runs/combined/quantized.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from floodlite.data import make_combined_loaders
from floodlite.models import make_student, STUDENT_BACKBONES
from floodlite.quantize import quantize_int8_onnx_static, file_size_mb
from eval_zero_shot import evaluate_onnx_per_image

SPLITS = ("val", "test", "external")


def _brief(res: dict) -> dict:
    keep = ("n_images", "iou_mean", "iou_std", "iou_flood_mean", "iou_flood_std",
            "false_alarm_rate", "precision_mean", "recall_mean", "per_image_iou")
    return {k: res[k] for k in keep if k in res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--kd", nargs="+", default=["none", "comb"])
    ap.add_argument("--exports_dir", default="exports_combined")
    ap.add_argument("--out", default="runs/combined/quantized.json")
    args = ap.parse_args()

    exports = Path(args.exports_dir)
    exports.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"recipe": "quantize_int8_onnx_static (per-tensor, Percentile 99.999, 80 calib imgs)",
                     "onnxruntime": __import__("onnxruntime").__version__}

    for fold in args.folds:
        loaders = make_combined_loaders(args.data_root, fold=fold, batch_size=8, num_workers=2)
        results[f"fold_{fold}"] = {}
        for sname in STUDENT_BACKBONES:
            for kd in args.kd:
                tag = f"{sname}_{kd}"
                ckpt = ckpt_dir / f"{tag}_fold{fold}.pt"
                if not ckpt.exists():
                    print(f"SKIP {ckpt} (missing)")
                    continue
                model = make_student(sname).cpu()
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                fp_onnx = exports / f"{tag}_fold{fold}_fp32.onnx"
                int8_onnx = exports / f"{tag}_fold{fold}_int8.onnx"
                try:
                    quantize_int8_onnx_static(model, fp_onnx_path=fp_onnx, int8_onnx_path=int8_onnx,
                                              calib_loader=loaders["train"], n_calib_batches=10)
                    entry = {"fp32_size_mb": file_size_mb(fp_onnx), "int8_size_mb": file_size_mb(int8_onnx)}
                    for split in SPLITS:
                        fp = evaluate_onnx_per_image(fp_onnx, loaders[split])
                        q = evaluate_onnx_per_image(int8_onnx, loaders[split])
                        entry[split] = {"fp32": _brief(fp), "int8": _brief(q)}
                    results[f"fold_{fold}"][tag] = entry
                    t = entry["test"]
                    print(f"fold {fold} {tag:26s} "
                          f"val {entry['val']['fp32']['iou_mean']:.3f}->{entry['val']['int8']['iou_mean']:.3f}  "
                          f"test flood {t['fp32'].get('iou_flood_mean', 0):.3f}->{t['int8'].get('iou_flood_mean', 0):.3f}  "
                          f"FAR {t['fp32'].get('false_alarm_rate', 0):.3f}->{t['int8'].get('false_alarm_rate', 0):.3f}  "
                          f"ext {entry['external']['fp32']['iou_mean']:.3f}->{entry['external']['int8']['iou_mean']:.3f}",
                          flush=True)
                except Exception as e:  # keep going; record the failure
                    results[f"fold_{fold}"][tag] = {"error": repr(e)}
                    print(f"fold {fold} {tag}: FAILED {e!r}", flush=True)
                out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
