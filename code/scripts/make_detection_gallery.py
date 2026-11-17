"""Figure: what the deployed models detect (qualitative gallery).

Rows are scenes: three FSSD fold-0 validation images (best, median and
worst EfficientNet-Lite0 IoU), one flooded FloodNet test scene and two
water-free FloodNet test scenes.  Columns are the input, the ground truth
and three deployed EfficientNet-Lite0 UNets run through ONNX Runtime:
FSSD-trained FP32, FSSD-trained INT8 (per-channel recipe) and the
combined-corpus-trained FP32 model.  Overlays colour true positives green,
false positives red and false negatives blue.

Usage:
  python scripts/make_detection_gallery.py --fssd_root <FSSD> --combined_root <combined>
      --exports exports --pc_exports <ptq_work dir> --combined_exports <combined-exports>
      [--out ../manuscript/figures/F7_detections]
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from floodlite.data import collect_pairs  # torch-free import path

ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TP, FP, FN = (0.18, 0.72, 0.42), (0.85, 0.23, 0.23), (0.16, 0.47, 0.84)
INK, INK2 = "#141414", "#555555"
DRY_SCENES = ["floodnet_7985.jpg", "floodnet_7647.jpg"]   # water-free FloodNet test scenes


def fold0_val_pairs(fssd_root: Path, n_folds=5, seed=42):
    """Replicates floodlite.data.make_loaders fold-0 validation split."""
    from sklearn.model_selection import KFold
    tr, va = collect_pairs(fssd_root)
    pairs = tr + va
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(len(pairs))))
    return [pairs[i] for i in splits[0][1]]


def to_tensor(img: Image.Image) -> np.ndarray:
    x = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]


class Model:
    def __init__(self, path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, img: Image.Image) -> np.ndarray:
        return self.sess.run(None, {self.name: to_tensor(img)})[0][0, 0] > 0


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum(); union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else float("nan")


def overlay(img: Image.Image, pred, gt, size=(320, 240), alpha=0.55):
    base = np.asarray(img.convert("RGB").resize(size, Image.BILINEAR), np.float32) / 255.0
    rs = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    p, g = rs(pred), rs(gt)
    out = base.copy()
    for m, c in ((p & g, TP), (p & ~g, FP), (~p & g, FN)):
        out[m] = (1 - alpha) * base[m] + alpha * np.array(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fssd_root", required=True)
    ap.add_argument("--combined_root", required=True)
    ap.add_argument("--exports", default=str(ROOT / "exports"))
    ap.add_argument("--pc_exports", required=True)
    ap.add_argument("--combined_exports", required=True)
    ap.add_argument("--out", default=str(ROOT.parent / "manuscript" / "figures" / "F7_detections"))
    args = ap.parse_args()

    models = [
        ("FSSD-trained, FP32", Model(Path(args.exports) / "efficientnet_lite0_fold0_fp32_preproc.onnx")),
        ("FSSD-trained, INT8 per-channel", Model(Path(args.pc_exports) / "efficientnet_lite0_fold0_pc_int8.onnx")),
        ("Combined-trained, FP32", Model(Path(args.combined_exports) / "efficientnet_lite0_none_fold0_fp32_preproc.onnx")),
    ]

    # --- FSSD fold-0 validation scenes: best / median / worst by FP32 IoU
    val = fold0_val_pairs(Path(args.fssd_root))
    scored = []
    for ip, mp in val:
        img = Image.open(ip); gt = np.asarray(Image.open(mp).convert("L").resize((256, 256), Image.NEAREST)) > 127
        scored.append((iou(models[0][1](img), gt), ip, mp))
    scored = [s for s in scored if not np.isnan(s[0])]
    scored.sort(key=lambda t: -t[0])
    picks = [scored[0], scored[len(scored) // 2], scored[-1]]
    scenes = [(f"FSSD val ({tag})", ip, mp) for (s, ip, mp), tag in zip(picks, ("best", "median", "hardest"))]

    # --- FloodNet test scenes: one flooded, two water-free
    croot = Path(args.combined_root)
    rows = list(csv.DictReader(open(croot / "manifest.csv", encoding="utf-8")))
    wet = [r for r in rows if r["source"] == "floodnet" and r["role"] == "floodnet_test"
           and r["exclude"] == "False" and 0.25 < float(r["flood_fraction"]) < 0.6]
    wet.sort(key=lambda r: r["image"])
    r = wet[len(wet) // 3]
    scenes.append(("FloodNet test, flooded", croot / "Image" / r["image"], croot / "Mask" / r["mask"]))
    for name in DRY_SCENES:
        rr = next(x for x in rows if x["image"] == name)
        scenes.append(("FloodNet test, water-free", croot / "Image" / name, croot / "Mask" / rr["mask"]))

    # --- render
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(scenes), ncol, figsize=(2.3 * ncol, 1.78 * len(scenes) + 0.9), dpi=220)
    fig.patch.set_facecolor("white")
    heads = ["Input", "Ground truth"] + [m[0] for m in models]
    for j, h in enumerate(heads):
        axes[0, j].set_title(h, fontsize=9, color=INK, pad=5)
    for i, (label, ip, mp) in enumerate(scenes):
        img = Image.open(ip)
        gt = np.asarray(Image.open(mp).convert("L").resize((256, 256), Image.NEAREST)) > 127
        axes[i, 0].imshow(np.asarray(img.convert("RGB").resize((320, 240))))
        gtv = np.asarray(img.convert("RGB").resize((320, 240)), np.float32) / 255.0
        g = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((320, 240), Image.NEAREST)) > 127
        gtv[g] = 0.45 * gtv[g] + 0.55 * np.array(FN)
        axes[i, 1].imshow(gtv)
        axes[i, 0].set_ylabel(label, fontsize=8, color=INK2, rotation=90, labelpad=4)
        for j, (mname, m) in enumerate(models):
            pred = m(img)
            axes[i, 2 + j].imshow(overlay(img, pred, gt))
            if gt.any():
                txt = f"IoU {iou(pred, gt):.2f}"
            else:
                txt = f"{pred.mean():.0%} flagged as water" if pred.any() else "no water predicted"
            axes[i, 2 + j].set_xlabel(txt, fontsize=8, color=(FP if (not gt.any() and pred.any()) else INK))
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#dddddd")
    fig.legend(handles=[Patch(color=TP, label="true positive (water found)"),
                        Patch(color=FP, label="false positive (false alarm)"),
                        Patch(color=FN, label="false negative (missed water)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("What the deployed EfficientNet-Lite0 UNets detect (ONNX Runtime, fold-0 checkpoints)",
                 fontsize=11, fontweight="bold", color=INK, y=0.995)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.945, bottom=0.06, wspace=0.04, hspace=0.22)
    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print("wrote", out.with_suffix(".png"))
    for label, ip, _ in scenes:
        print(" ", label, Path(ip).name)


if __name__ == "__main__":
    main()
