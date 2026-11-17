"""Figure: how an image becomes a flood mask in the released code.

Six-step inference path (read, preprocess, ONNX Runtime session, logits,
sigmoid + threshold, use), with tensor shapes, the Python calls that
implement each step, the browser equivalent, and a worked example on an
FSSD image using the deployed EfficientNet-Lite0 INT8 export.

Usage:  python scripts/make_inference_flow.py --fssd_root <FSSD> --pc_exports <ptq_work dir>
        [--out ../manuscript/figures/F8_inference_flow]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W_IN, H_IN, DPI = 14.0, 6.4, 250
ASPECT = H_IN / W_IN
INK, INK2 = "#141414", "#555555"
BLUE, GREEN, ORANGE, GREY = "#2a78d6", "#2e9e6b", "#eb6834", "#8d8d8d"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FSSD_SAMPLE = "HS604.png"

STEPS = [
    ("1  Read image", "any size, RGB\n(camera, drone, file)", "PIL.Image.open(path)", BLUE),
    ("2  Preprocess", "resize 256 x 256 (bilinear)\n/255, ImageNet mean/std\nHWC to NCHW float32\n1 x 3 x 256 x 256",
     "floodlite.data transforms", BLUE),
    ("3  Run the network", "ONNX Runtime session\none .onnx file: FP32 or INT8\nCPU provider (laptop, desktop)\nWebAssembly (browser)",
     "ort.InferenceSession(...).run", GREEN),
    ("4  Logits", "1 x 1 x 256 x 256\nreal-valued, per pixel", "output name: logits", GREEN),
    ("5  Decide", "sigmoid, threshold 0.5\nbinary mask\nresize back to input size\n(nearest neighbour)", "mask = sigmoid(z) > 0.5", ORANGE),
    ("6  Use", "overlay for an operator\nflooded-area fraction\nalert if above threshold\nIoU / false-alarm audit", "floodlite.metrics", ORANGE),
]

PY_CODE = (
    "import numpy as np, onnxruntime as ort\n"
    "from PIL import Image\n"
    "\n"
    "sess = ort.InferenceSession('efficientnet_lite0_int8.onnx',\n"
    "                            providers=['CPUExecutionProvider'])\n"
    "img  = Image.open(path).convert('RGB').resize((256, 256))\n"
    "x    = (np.asarray(img, np.float32) / 255 - MEAN) / STD\n"
    "x    = x.transpose(2, 0, 1)[None]              # 1x3x256x256\n"
    "z    = sess.run(None, {'input': x})[0]          # 1x1x256x256\n"
    "mask = 1 / (1 + np.exp(-z[0, 0])) > 0.5         # True = water\n"
    "flooded_fraction = mask.mean()"
)
JS_CODE = (
    "// browser: same .onnx file, onnxruntime-web (WebAssembly)\n"
    "const sess = await ort.InferenceSession.create('efficientnet_lite0_fp32.onnx');\n"
    "const out  = await sess.run({ input: new ort.Tensor('float32', x, [1, 3, 256, 256]) });\n"
    "const mask = out.logits.data.map(z => z > 0);   // sigmoid(z) > 0.5  <=>  z > 0"
)


def rbox(ax, x, y, w, h, face, edge, lw=1.3, r=0.008, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ASPECT, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color, lw=1.6, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color,
                                 zorder=3, shrinkA=0, shrinkB=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fssd_root", required=True)
    ap.add_argument("--pc_exports", required=True)
    ap.add_argument("--out", default=str(ROOT.parent / "manuscript" / "figures" / "F8_inference_flow"))
    args = ap.parse_args()

    import onnxruntime as ort
    img = Image.open(Path(args.fssd_root) / "dataset" / "val" / "images" / FSSD_SAMPLE).convert("RGB")
    x = ((np.asarray(img.resize((256, 256), Image.BILINEAR), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    sess = ort.InferenceSession(str(Path(args.pc_exports) / "efficientnet_lite0_fold0_pc_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    z = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    prob = 1 / (1 + np.exp(-z)); mask = prob > 0.5
    base = np.asarray(img.resize((320, 240)), np.float32) / 255
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((320, 240), Image.NEAREST)) > 127
    ov = base.copy(); ov[m] = 0.45 * base[m] + 0.55 * np.array([0.16, 0.47, 0.84])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.5, 0.96, "How an image becomes a flood mask: the FloodLite inference path",
             ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.925, "The same six steps run on a laptop or desktop CPU (ONNX Runtime) and in a browser "
             "(ONNX Runtime Web); only the runtime object changes.", ha="center", va="center", fontsize=9, color=INK2)

    # ---- step boxes
    n = len(STEPS); bw, gap = 0.138, 0.0245
    x0 = (1 - n * bw - (n - 1) * gap) / 2
    by, bh = 0.60, 0.235
    for i, (head, body, code, col) in enumerate(STEPS):
        bx = x0 + i * (bw + gap)
        rbox(ax, bx, by, bw, bh, "white", col, lw=1.6)
        ax.text(bx + bw / 2, by + bh - 0.03, head, ha="center", va="center", fontsize=9.6, fontweight="bold", color=col, zorder=4)
        ax.text(bx + bw / 2, by + bh / 2 - 0.005, body, ha="center", va="center", fontsize=7.9, color=INK, zorder=4, linespacing=1.25)
        ax.text(bx + bw / 2, by + 0.022, code, ha="center", va="center", fontsize=6.9, color=INK2, zorder=4,
                family="DejaVu Sans Mono")
        if i < n - 1:
            arrow(ax, (bx + bw, by + bh / 2), (bx + bw + gap, by + bh / 2), GREY)

    # ---- worked example thumbnails under steps 1, 4 and 5/6
    th_w = 0.12; th_h = th_w * 0.75 / ASPECT
    ty = 0.36
    ex = [(x0 + bw / 2 - th_w / 2, np.asarray(img.resize((320, 240))), "input (HS604, FSSD)"),
          (x0 + 3 * (bw + gap) + bw / 2 - th_w / 2, z, "logits (blue < 0 < red)"),
          (x0 + 4 * (bw + gap) + bw / 2 - th_w / 2, mask, "mask, sigmoid > 0.5"),
          (x0 + 5 * (bw + gap) + bw / 2 - th_w / 2, ov, f"overlay: {mask.mean():.0%} of pixels water")]
    for (tx, im, cap) in ex:
        a = fig.add_axes([tx, ty, th_w, th_h]); a.axis("off")
        if im.ndim == 2 and im.dtype != bool:
            a.imshow(im, cmap="RdBu_r", vmin=-8, vmax=8)
        elif im.dtype == bool:
            a.imshow(im, cmap="Blues", vmin=0, vmax=1.3)
        else:
            a.imshow(im)
        ax.text(tx + th_w / 2, ty - 0.012, cap, ha="center", va="top", fontsize=7.6, color=INK2)
    ax.text(x0 + 1.5 * (bw + gap) + bw / 2, ty + th_h / 2,
            "worked example:\nEfficientNet-Lite0 INT8\n(per-channel), fold 0,\nONNX Runtime CPU",
            ha="center", va="center", fontsize=8.2, color=INK2, linespacing=1.3)
    arrow(ax, (ex[0][0] + th_w + 0.005, ty + th_h / 2), (ex[1][0] - 0.005, ty + th_h / 2), GREY, lw=1.2, ms=9)

    # ---- code boxes
    rbox(ax, 0.03, 0.03, 0.555, 0.265, "#f7f7f7", "#bbbbbb", lw=1.0)
    ax.text(0.045, 0.275, "Python (floodlite, ONNX Runtime CPU)", ha="left", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0.045, 0.255, PY_CODE, ha="left", va="top", fontsize=7.0, color=INK, family="DejaVu Sans Mono", linespacing=1.3, zorder=4)
    rbox(ax, 0.605, 0.03, 0.365, 0.265, "#f7f7f7", "#bbbbbb", lw=1.0)
    ax.text(0.62, 0.275, "JavaScript (browser, onnxruntime-web)", ha="left", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0.62, 0.255, JS_CODE, ha="left", va="top", fontsize=6.4, color=INK, family="DejaVu Sans Mono", linespacing=1.35, zorder=4)
    ax.text(0.62, 0.075, "Batch of N images: stack to N x 3 x 256 x 256; the\nmodel accepts a dynamic batch dimension.",
            ha="left", va="top", fontsize=7.4, color=INK2, linespacing=1.3)

    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print("wrote", out.with_suffix(".png"), f"example flooded fraction {mask.mean():.3f}")


if __name__ == "__main__":
    main()
