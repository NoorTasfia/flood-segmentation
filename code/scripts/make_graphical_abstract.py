"""Graphical abstract for the FloodLite manuscript (Elsevier: >= 1328 x 531 px).

Three panels, all values read from the authoritative result files:
  1. INT8 post-training quantization stability (runs/3fold/summary.json + quantized.json)
  2. False alarms on water-free scenes, FSSD-trained vs combined-trained
     (runs/3fold/zero_shot_3fold.json, runs/combined/summary.json)
  3. Distillation copies teacher false alarms (runs/combined/stats_combined.json)

Usage:  python scripts/make_graphical_abstract.py [--out ../manuscript/figures/graphical_abstract]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STUDENTS = [("mobilenetv3_small", "MobileNetV3-S"),
            ("efficientnet_lite0", "EffNet-Lite0"),
            ("mobilevit_xxs", "MobileViT-XXS")]


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def int8_ious() -> tuple[list[float], list[float]]:
    q = load("runs/3fold/quantized.json")
    fp, i8 = [], []
    for s, _ in STUDENTS:
        folds = [q[k][s] for k in q if k.startswith("fold_") and s in q[k] and "int8_iou" in q[k][s]]
        fp.append(float(np.mean([f["fp32_iou"] for f in folds])))
        i8.append(float(np.mean([f["int8_iou"] for f in folds])))
    return fp, i8


def fars() -> tuple[list[float], list[float]]:
    zs = load("runs/3fold/zero_shot_3fold.json")["floodnet_test"]
    cb = load("runs/combined/summary.json")["models"]
    return ([zs[f"{s}_none"]["false_alarm_rate"] for s, _ in STUDENTS],
            [cb[f"{s}_none"]["test"]["false_alarm_rate"]["mean"] for s, _ in STUDENTS])


def inheritance() -> tuple[list[float], list[float]]:
    inh = load("runs/combined/stats_combined.json")["teacher_inheritance"]
    return ([inh[s]["none"]["p_fa_given_teacher_fa"] for s, _ in STUDENTS],
            [inh[s]["comb"]["p_fa_given_teacher_fa"] for s, _ in STUDENTS])


def style(ax, ymax, pct=False):
    ax.set_ylim(0, ymax)
    step = 0.2 if ymax > 0.6 else 0.1
    ax.set_yticks(np.arange(0, min(ymax, 1.0) + 1e-9, step))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", length=0, colors=INK2, labelsize=10)
    ax.tick_params(axis="x", length=0, colors=INK, labelsize=10.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def paired(ax, a, b, la, lb, fmt):
    x = np.arange(len(STUDENTS))
    w = 0.36
    for off, vals, col, lab in ((-w / 2 - 0.01, a, BLUE, la), (w / 2 + 0.01, b, ORANGE, lb)):
        bars = ax.bar(x + off, vals, w, color=col, label=lab, edgecolor="white", linewidth=1.5)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015 * ax.get_ylim()[1], fmt(v),
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x, [n for _, n in STUDENTS])
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10,
              handlelength=1.0, columnspacing=1.0, labelcolor=INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "manuscript" / "figures" / "graphical_abstract"))
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.28, 5.31), dpi=300)
    fig.patch.set_facecolor("white")

    titles = [
        ("1  Deploy: INT8 must suit the backbone",
         "Lite0 INT8 runs at 58 FPS on a laptop CPU;\nthe other two collapse under the same recipe"),
        ("2  Flood-only data cannot say “no flood”",
         "Share of 215 dry FloodNet scenes with any\npredicted water (no-KD students)"),
        ("3  Distillation copies teacher errors",
         "On dry scenes where the teacher false-alarms,\nhow often the student does too"),
    ]

    fp, i8 = int8_ious()
    ax = axes[0]
    style(ax, 1.25)
    paired(ax, fp, i8, "FP32", "INT8", lambda v: f"{v:.2f}")
    ax.set_ylabel("IoU on FSSD", color=INK2, fontsize=10)

    a, b = fars()
    ax = axes[1]
    style(ax, 1.25, pct=True)
    paired(ax, a, b, "FSSD-trained", "+ water-free images", lambda v: f"{v:.0%}")

    n, c = inheritance()
    ax = axes[2]
    style(ax, 0.48, pct=True)
    paired(ax, n, c, "no KD", "combined KD", lambda v: f"{v:.0%}")
    ax.text(0.98, 0.86, "KD also lifts flood IoU\n+0.03 to +0.07", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=INK2)

    for ax, (t, sub) in zip(axes, titles):
        ax.set_title(sub, fontsize=9.5, color=INK2, loc="left", pad=8)
        ax.text(0, 1.115, t, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK, va="bottom")

    fig.suptitle("FloodLite: lightweight flood segmentation for CPU and browser deployment",
                 x=0.012, ha="left", fontsize=14.5, fontweight="bold", color=INK, y=0.975)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.76, wspace=0.22)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print("wrote", out.with_suffix(".png"), out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
