"""Figure: FloodLite network architecture and distillation signals.

Draws the UNet encoder-decoder used for every model (segmentation-models-
pytorch Unet, encoder depth 5, decoder channels 256/128/64/32/16, 3x3 conv
head), with the reference network (teacher) on top and a compact network
(student) below, the three loss terms of floodlite/losses.py, and the
decoder-block-2 feature hook of floodlite/models.py.  An FSSD validation
image and its mask are used as the input/output thumbnails.

Usage:  python scripts/make_architecture_figure.py --fssd_root <FSSD root>
        [--out ../manuscript/figures/F6_architecture]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W_IN, H_IN, DPI = 14.0, 7.2, 250
ASPECT = H_IN / W_IN
INK, INK2 = "#141414", "#555555"
ENC, DEC, HEAD = "#c9ddf5", "#f6d3c3", "#d9d9d9"      # block fills
ENC_E, DEC_E, HEAD_E = "#2a78d6", "#eb6834", "#777777"
SKIP, DOWN, UP, KD, PRED = "#2e9e6b", "#2a78d6", "#eb6834", "#8e44ad", "#d93a3a"
FSSD_SAMPLE = "HS604.png"

STAGES = [("stage 1", "1/2"), ("stage 2", "1/4"), ("stage 3", "1/8"), ("stage 4", "1/16"), ("stage 5", "1/32")]
DEC_CH = [256, 128, 64, 32, 16]


def rbox(ax, x, y, w, h, face, edge, lw=1.2, r=0.006, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ASPECT, facecolor=face, edgecolor=edge,
                                linewidth=lw, linestyle=ls, zorder=z))


def arrow(ax, p0, p1, color, lw=1.4, style="-|>", ms=10, z=3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, linewidth=lw,
                                 color=color, zorder=z, shrinkA=0, shrinkB=0, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def unet(ax, x0, y0, label, sub, color_label, hook=True):
    """Draw one UNet at origin (x0, y0); returns dict of anchor points."""
    bw, bh, gap = 0.052, 0.048, 0.012
    enc_x = [x0 + i * (bw + gap) for i in range(5)]
    enc_y = [y0 + 0.165 - i * 0.041 for i in range(5)]           # descending staircase
    dec_x = [x0 + 5 * (bw + gap) + 0.03 + i * (bw + gap) for i in range(5)]
    dec_y = [y0 + 0.165 - (4 - i) * 0.041 for i in range(5)]     # ascending staircase
    # encoder
    for i, (nm, res) in enumerate(STAGES):
        rbox(ax, enc_x[i], enc_y[i], bw, bh, ENC, ENC_E)
        ax.text(enc_x[i] + bw / 2, enc_y[i] + bh * 0.64, nm, ha="center", va="center", fontsize=7.4, color=INK, zorder=4)
        ax.text(enc_x[i] + bw / 2, enc_y[i] + bh * 0.28, res, ha="center", va="center", fontsize=6.6, color=INK2, zorder=4)
        if i < 4:
            arrow(ax, (enc_x[i] + bw, enc_y[i] + bh * 0.35), (enc_x[i + 1], enc_y[i + 1] + bh * 0.75), DOWN, lw=1.3, ms=8)
    # bottleneck -> decoder
    for i in range(5):
        rbox(ax, dec_x[i], dec_y[i], bw, bh, DEC, DEC_E)
        ax.text(dec_x[i] + bw / 2, dec_y[i] + bh * 0.64, f"block {i}", ha="center", va="center", fontsize=7.4, color=INK, zorder=4)
        ax.text(dec_x[i] + bw / 2, dec_y[i] + bh * 0.28, f"{DEC_CH[i]} ch", ha="center", va="center", fontsize=6.6, color=INK2, zorder=4)
        if i < 4:
            arrow(ax, (dec_x[i] + bw, dec_y[i] + bh * 0.75), (dec_x[i + 1], dec_y[i + 1] + bh * 0.35), UP, lw=1.3, ms=8)
    arrow(ax, (enc_x[4] + bw, enc_y[4] + bh * 0.5), (dec_x[0], dec_y[0] + bh * 0.5), UP, lw=1.3, ms=8)
    # skip connections: encoder stage i (i=0..3) -> decoder block 4-i-? (same resolution)
    for i in range(4):
        j = 3 - i   # stage1(1/2)->block3 (1/2 res), stage2->block2, stage3->block1, stage4->block0
        p0 = (enc_x[i] + bw / 2, enc_y[i] + bh)
        p1 = (dec_x[j] + bw / 2, dec_y[j] + bh)
        ax.plot([p0[0], p0[0], p1[0], p1[0]], [p0[1], p0[1] + 0.012 + i * 0.000, p1[1] + 0.012, p1[1]],
                color=SKIP, lw=1.2, zorder=1)
        ax.plot([p1[0]], [p1[1]], marker="v", color=SKIP, markersize=4, zorder=3)
    # head
    hx, hy = dec_x[4] + bw + 0.022, dec_y[4]
    rbox(ax, hx, hy, 0.052, bh, HEAD, HEAD_E)
    ax.text(hx + 0.026, hy + bh * 0.64, "3x3 conv", ha="center", va="center", fontsize=7.2, color=INK, zorder=4)
    ax.text(hx + 0.026, hy + bh * 0.28, "1 logit map", ha="center", va="center", fontsize=6.6, color=INK2, zorder=4)
    arrow(ax, (dec_x[4] + bw, dec_y[4] + bh / 2), (hx, hy + bh / 2), UP, lw=1.3, ms=8)
    # labels
    ax.text(x0, y0 + 0.265, label, ha="left", va="center", fontsize=10.5, fontweight="bold", color=color_label)
    ax.text(x0, y0 + 0.237, sub, ha="left", va="center", fontsize=8.2, color=INK2)
    ax.text(enc_x[2] + bw / 2, y0 - 0.012, "encoder (ImageNet-pretrained backbone)", ha="center", va="top", fontsize=7.6, color=ENC_E)
    ax.text(dec_x[2] + bw / 2, y0 - 0.012, "UNet decoder (shared topology)", ha="center", va="top", fontsize=7.6, color=DEC_E)
    return {"in": (enc_x[0], enc_y[0] + bh / 2), "out": (hx + 0.052, hy + bh / 2),
            "hook": (dec_x[2] + bw / 2, dec_y[2]), "hook_top": (dec_x[2] + bw / 2, dec_y[2] + bh),
            "logit": (hx + 0.026, hy), "logit_top": (hx + 0.026, hy + bh), "y0": y0, "bh": bh}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fssd_root", required=True)
    ap.add_argument("--out", default=str(ROOT.parent / "manuscript" / "figures" / "F6_architecture"))
    args = ap.parse_args()

    fssd = Path(args.fssd_root) / "dataset" / "val"
    img = Image.open(fssd / "images" / FSSD_SAMPLE).convert("RGB").resize((320, 240))
    gt = np.asarray(Image.open(fssd / "labels" / FSSD_SAMPLE).convert("L").resize((320, 240))) > 127

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.text(0.5, 0.965, "FloodLite network architecture and distillation signals",
             ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.932, "Every model is a UNet: a pretrained encoder with five stages, a five-block decoder with skip connections, "
             "and a 3x3 convolution head producing one flood logit map at 256 x 256.",
             ha="center", va="center", fontsize=9, color=INK2)

    # input / output thumbnails
    a = fig.add_axes([0.03, 0.33, 0.10, 0.10 * 0.75 / ASPECT]); a.imshow(img); a.axis("off")
    ax.text(0.08, 0.315, "Input RGB image\n256 x 256, ImageNet-normalised", ha="center", va="top", fontsize=7.6, color=INK2)
    a = fig.add_axes([0.875, 0.40, 0.10, 0.10 * 0.75 / ASPECT]); a.imshow(gt, cmap="Blues", vmin=0, vmax=1.3); a.axis("off")
    ax.text(0.925, 0.385, "Flood mask\nsigmoid(logits) > 0.5", ha="center", va="top", fontsize=7.6, color=INK2)

    # two UNets
    T = unet(ax, 0.165, 0.60, "Reference network (teacher): UNet + EfficientNet-B0",
             "6.25 M parameters; trained first with the task loss only, then frozen", ENC_E)
    S = unet(ax, 0.165, 0.115, "Compact network (student): UNet + MobileNetV3-Small / EfficientNet-Lite0 / MobileViT-XXS",
             "3.1 to 5.6 M parameters; trained with the task loss plus optional distillation terms", DEC_E)

    # input/output arrows
    for M in (T, S):
        arrow(ax, (0.135, 0.40), M["in"], INK2, lw=1.1, ms=8, z=1)
        arrow(ax, M["out"], (0.87, 0.46), PRED, lw=1.3, ms=9, z=1)

    # losses
    def loss_box(x, y, w, h, title, body, color):
        rbox(ax, x, y, w, h, "white", color, lw=1.4, r=0.008, z=5)
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=color, zorder=6)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=7.4, color=INK, zorder=6, linespacing=1.2)

    # task loss (student output vs ground truth)
    loss_box(0.775, 0.03, 0.20, 0.072, "Task loss (always on)",
             r"$0.5\,\mathrm{BCE} + 0.5\,(1-\mathrm{Dice})$ vs ground truth", PRED)
    arrow(ax, S["logit"], (0.875, 0.102), PRED, lw=1.2, ms=8, ls="--")

    # response KD (logits)
    loss_box(0.615, 0.47, 0.215, 0.075, "Response distillation (alpha = 0.5)",
             r"$T^{2}\,\mathrm{H}\!\left(\sigma(z_T/T),\,\sigma(z_S/T)\right),\; T = 4$", KD)
    arrow(ax, T["logit"], (0.78, 0.545), KD, lw=1.2, ms=8, ls="--")
    arrow(ax, S["logit_top"], (0.78, 0.47), KD, lw=1.2, ms=8, ls="--")

    # feature KD (decoder block 2)
    loss_box(0.30, 0.455, 0.25, 0.085, "Feature distillation (beta = 0.1)",
             "MSE between decoder block 2 activations (1/8 res.)\nafter a learned 1x1 channel adapter on the student side", KD)
    arrow(ax, T["hook"], (0.425, 0.54), KD, lw=1.2, ms=8, ls="--")
    arrow(ax, S["hook_top"], (0.425, 0.455), KD, lw=1.2, ms=8, ls="--")

    # total loss
    ax.text(0.385, 0.012, r"Student objective:  $\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\mathrm{task}} + \alpha\,\mathcal{L}_{\mathrm{resp}} + \beta\,\mathcal{L}_{\mathrm{feat}}$;"
            r"  no-KD ($\alpha=\beta=0$), response-only, feature-only and combined configurations are compared.",
            ha="center", va="bottom", fontsize=8.6, color=INK)

    # legend
    lx, ly = 0.02, 0.80
    items = [(DOWN, "-", "downsampling (stride-2 stage)"), (UP, "-", "upsampling (x2 block)"),
             (SKIP, "-", "skip connection"), (KD, "--", "distillation signal"),
             (PRED, "-", "prediction / task loss")]
    rbox(ax, lx - 0.008, ly - 0.16, 0.145, 0.225, "white", "#bbbbbb", lw=0.9, r=0.006, z=5)
    ax.text(lx, ly + 0.045, "Legend", ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK, zorder=6)
    for i, (c, ls, lab) in enumerate(items):
        yy = ly + 0.012 - i * 0.028
        arrow(ax, (lx, yy), (lx + 0.026, yy), c, lw=1.4, ms=8, ls=ls, z=6)
        ax.text(lx + 0.032, yy, lab, ha="left", va="center", fontsize=6.9, color=INK, zorder=6)
    ax.add_patch(Rectangle((lx, ly - 0.148), 0.012, 0.012, facecolor=ENC, edgecolor=ENC_E, lw=0.8, zorder=6))
    ax.text(lx + 0.016, ly - 0.142, "encoder", ha="left", va="center", fontsize=6.9, color=INK, zorder=6)
    ax.add_patch(Rectangle((lx + 0.062, ly - 0.148), 0.012, 0.012, facecolor=DEC, edgecolor=DEC_E, lw=0.8, zorder=6))
    ax.text(lx + 0.078, ly - 0.142, "decoder", ha="left", va="center", fontsize=6.9, color=INK, zorder=6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
