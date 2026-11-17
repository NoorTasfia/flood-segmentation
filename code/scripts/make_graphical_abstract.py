"""Graphical abstract for the FloodLite manuscript (Elsevier: >= 1328 x 531 px).

Workflow-style figure in three boxed panels, read left to right:
  1. Train and distil on public data   (FSSD sample + teacher/student diagram)
  2. Quantize to INT8 and deploy         (recipe ablation chart + three targets)
  3. Check what responders will see      (dry scene: FSSD-trained vs retrained,
                                          false-alarm chart, KD inheritance)

Every number is read from the authoritative result files
(runs/3fold/PROVENANCE.md): ptq_mechanism.json (INT8 recipe ablation),
quantized.json (FP32 IoU), lat_m2.json / host2/lat_ort_cpu.json /
lat_wasm.json / host2/lat_wasm_node.json (latency), zero_shot_3fold.json and
combined/summary.json (false-alarm rates), combined/stats_combined.json
(teacher-inheritance and KD contrasts).  Parameter counts are Table 1.

The two photographs are an FSSD validation image with its ground-truth mask
and a water-free FloodNet test scene with the predictions of the FSSD-trained
and the combined-trained EfficientNet-Lite0 (fold 0, FP32 ONNX exports), run
through ONNX Runtime at build time.

Usage (needs numpy, matplotlib, Pillow, onnxruntime):
  python scripts/make_graphical_abstract.py \
      --fssd_root C:/.../flood-data/fssd --combined_root C:/.../flood-data/combined \
      --combined_exports C:/.../flood-data/combined-exports \
      [--out ../manuscript/figures/graphical_abstract]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W_IN, H_IN, DPI = 16.0, 6.8, 250
ASPECT = H_IN / W_IN

INK, INK2, GRID = "#141414", "#555555", "#e6e6e6"
BLUE, GREEN, ORANGE, RED, GREY = "#2a78d6", "#2e9e6b", "#eb6834", "#d93a3a", "#8d8d8d"
PANEL = {"train": BLUE, "deploy": GREEN, "check": ORANGE}
STUDENTS = [("mobilenetv3_small", "MobileNetV3-S", 3.59),
            ("efficientnet_lite0", "EfficientNet-Lite0", 5.61),
            ("mobilevit_xxs", "MobileViT-XXS", 3.09)]
TEACHER_PARAMS = 6.25  # Table 1, UNet+EfficientNet-B0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

FSSD_SAMPLE = "HS604.png"           # dataset/val
DRY_SAMPLE = "floodnet_7985.jpg"    # combined corpus, role floodnet_test, flood_fraction 0


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ----------------------------------------------------------------- numbers
def numbers() -> dict:
    n: dict = {}
    m = load("runs/3fold/ptq_mechanism.json")
    folds = [k for k in m if k.startswith("fold_")]
    nets = [s for s, _, _ in STUDENTS] + ["teacher"]
    n["fp32"] = {k: float(np.mean([m[f][k]["fp32"]["iou"] for f in folds])) for k in nets}
    n["pt"] = {k: float(np.mean([m[f][k]["variants"]["pt"]["iou"] for f in folds])) for k in nets}
    n["pc"] = {k: float(np.mean([m[f][k]["variants"]["pc"]["iou"] for f in folds])) for k in nets}

    q = load("runs/3fold/quantized.json")
    n["fp32_headline"] = {s: float(np.mean([q[f][s]["fp32_iou"] for f in folds if s in q[f]]))
                          for s, _, _ in STUDENTS}
    n["fp32_headline"]["teacher"] = float(load("runs/3fold/summary.json")["teacher"]["iou"]["mean"])
    n["lite0_int8_iou"] = float(np.mean([q[f]["efficientnet_lite0"]["int8_iou"] for f in folds]))
    n["lite0_size_x"] = float(np.mean([q[f]["efficientnet_lite0"]["size_reduction_x"] for f in folds]))

    m2 = load("runs/3fold/lat_m2.json")
    n["m2_lite0_int8_fps"] = m2["efficientnet_lite0_int8_ort_cpu"]["fps"]
    n["m2_lite0_int8_ms"] = m2["efficientnet_lite0_int8_ort_cpu"]["p50_ms"]
    n["m2_teacher_ms"] = m2["teacher_ort_cpu"]["p50_ms"]
    h2 = load("runs/host2/lat_ort_cpu.json")
    ratios = [h2[f"{s}_int8_pt"]["p50_ms"] / h2[f"{s}_fp32"]["p50_ms"] for s, _, _ in STUDENTS]
    n["x86_int8_slow_lo"], n["x86_int8_slow_hi"] = min(ratios), max(ratios)
    n["x86_fp32_fps_lo"] = min(h2[f"{s}_fp32"]["fps"] for s, _, _ in STUDENTS)
    n["x86_fp32_fps_hi"] = max(h2[f"{s}_fp32"]["fps"] for s, _, _ in STUDENTS)
    wasm = load("runs/3fold/lat_wasm.json")
    fps = [1000 / wasm[f"{s}_fp32"]["p50_ms"] for s, _, _ in STUDENTS]
    n["wasm_fps_lo"], n["wasm_fps_hi"] = min(fps), max(fps)
    w2 = load("runs/host2/lat_wasm_node.json")
    sp = [w2[f"{s}_fp32"]["p50_ms"] / w2[f"{s}_int8_pt"]["p50_ms"] for s, _, _ in STUDENTS]
    n["wasm_int8_speedup_lo"], n["wasm_int8_speedup_hi"] = min(sp), max(sp)

    zs = load("runs/3fold/zero_shot_3fold.json")["floodnet_test"]
    cb = load("runs/combined/summary.json")["models"]
    n["far_fssd"] = [zs[f"{s}_none"]["false_alarm_rate"] for s, _, _ in STUDENTS]
    n["far_comb"] = [cb[f"{s}_none"]["test"]["false_alarm_rate"]["mean"] for s, _, _ in STUDENTS]
    st = load("runs/combined/stats_combined.json")
    n["n_dry"] = st["n_dry"]
    inh = st["teacher_inheritance"]
    n["inh_none"] = [inh[s]["none"]["p_fa_given_teacher_fa"] for s, _, _ in STUDENTS]
    n["inh_comb"] = [inh[s]["comb"]["p_fa_given_teacher_fa"] for s, _, _ in STUDENTS]
    kd = st["kd_contrast"]
    d = [kd[s]["flood_iou_comb_minus_none"]["mean"] for s, _, _ in STUDENTS]
    n["kd_gain_lo"], n["kd_gain_hi"] = min(d), max(d)
    return n


# ----------------------------------------------------------------- images
def to_tensor(img: Image.Image) -> np.ndarray:
    x = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]


def predict(onnx_path: Path, img: Image.Image) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: to_tensor(img)})[0][0, 0]
    return logits > 0


def overlay(img: Image.Image, mask: np.ndarray, rgb, alpha=0.55, size=(400, 300)) -> np.ndarray:
    base = np.asarray(img.convert("RGB").resize(size, Image.BILINEAR), np.float32) / 255.0
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    col = np.array(rgb, np.float32)[None, None, :]
    out = base.copy()
    out[m] = (1 - alpha) * base[m] + alpha * col
    return out


def hex2rgb(h: str):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


# ----------------------------------------------------------------- drawing
def rbox(ax, x, y, w, h, edge, face="white", lw=1.6, r=0.012, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       mutation_aspect=ASPECT, linewidth=lw, edgecolor=edge,
                       facecolor=face, linestyle=ls, zorder=1)
    ax.add_patch(p)
    return p


def chip(ax, x, y, w, h, text, color, fs=9.6, bold_first=True, face=None):
    rbox(ax, x, y, w, h, color, face=face or "white", lw=1.3, r=0.008)
    if bold_first and "\n" in text:
        head, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h * 0.70, head, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, rest, ha="center", va="center", fontsize=fs - 0.8,
                color=INK2, zorder=3, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3, linespacing=1.2)


def arrow(ax, p0, p1, color=INK2, lw=1.6, style="-|>", ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, zorder=2,
                                 shrinkA=0, shrinkB=0))


def chevron(ax, x, y, w=0.018, h=0.07, color=GREY):
    ax.add_patch(Polygon([(x, y - h / 2), (x + w, y), (x, y + h / 2)], closed=True,
                         facecolor=color, edgecolor="none", zorder=2))


def panel_frame(ax, x0, x1, color, title, number):
    rbox(ax, x0, 0.045, x1 - x0, 0.80, color, face="#fbfbfb", lw=2.0, r=0.016)
    ax.text(x0 + 0.014, 0.878, number, ha="left", va="center", fontsize=15, fontweight="bold",
            color="white", zorder=4,
            bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="none"))
    ax.text(x0 + 0.045, 0.878, title, ha="left", va="center", fontsize=14,
            fontweight="bold", color=INK, zorder=4)


def style_axes(ax, ymax, pct=False):
    ax.set_ylim(0, ymax)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", length=0, colors=INK2, labelsize=8.5)
    ax.tick_params(axis="x", length=0, colors=INK, labelsize=8.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("none")
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fssd_root", required=True)
    ap.add_argument("--combined_root", required=True)
    ap.add_argument("--exports", default=str(ROOT / "exports"))
    ap.add_argument("--combined_exports", required=True)
    ap.add_argument("--out", default=str(ROOT.parent / "manuscript" / "figures" / "graphical_abstract"))
    args = ap.parse_args()
    n = numbers()

    # ---- photographs and predictions
    fssd = Path(args.fssd_root) / "dataset" / "val"
    img_f = Image.open(fssd / "images" / FSSD_SAMPLE)
    gt = np.asarray(Image.open(fssd / "labels" / FSSD_SAMPLE).convert("L")) > 127
    fssd_vis = overlay(img_f, gt, hex2rgb(BLUE), 0.5)

    img_d = Image.open(Path(args.combined_root) / "Image" / DRY_SAMPLE)
    pred_fssd = predict(Path(args.exports) / "efficientnet_lite0_fold0_fp32_preproc.onnx", img_d)
    pred_comb = predict(Path(args.combined_exports) / "efficientnet_lite0_none_fold0_fp32_preproc.onnx", img_d)
    dry_in = np.asarray(img_d.convert("RGB").resize((400, 300), Image.BILINEAR)) / 255.0
    dry_fssd = overlay(img_d, pred_fssd, hex2rgb(RED), 0.6)
    dry_comb = overlay(img_d, pred_comb, hex2rgb(RED), 0.6)
    fa_frac_fssd, fa_frac_comb = pred_fssd.mean(), pred_comb.mean()

    # ---- canvas
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.text(0.015, 0.965, "FloodLite: lightweight flood segmentation that is deployable, not just accurate",
             ha="left", va="center", fontsize=19.5, fontweight="bold", color=INK)
    fig.text(0.015, 0.923, "Three compact UNets and one larger reference network, benchmarked on two CPUs and in a browser in "
             "FP32 and INT8, then checked on water-free scenes that flood-only benchmarks never show.",
             ha="left", va="center", fontsize=11, color=INK2)

    # =============================== panel 1: train and distil
    x0, x1 = 0.015, 0.325
    panel_frame(ax, x0, x1, PANEL["train"], "Train and distil on public data", "1")

    a = fig.add_axes([x0 + 0.015, 0.605, 0.118, 0.118 * (300 / 400) / ASPECT])
    a.imshow(fssd_vis); a.axis("off")
    ax.text(x0 + 0.015 + 0.059, 0.593, "FSSD image with ground truth", ha="center", va="top",
            fontsize=8.3, color=INK2)
    chip(ax, x0 + 0.143, 0.735, 0.167, 0.085,
         "FSSD: 663 flood images (RGB)\n3-fold cross-validation,\nno water-free scenes at all", BLUE, fs=9.0)
    chip(ax, x0 + 0.143, 0.625, 0.167, 0.095,
         "Combined corpus: 3,009 images\n855 water-free (FloodNet, AIFloodSense,\nKaggle), duplicate-checked vs FSSD", BLUE, fs=9.0)

    tx, ty, tw, th = x0 + 0.055, 0.435, 0.20, 0.095
    chip(ax, tx, ty, tw, th,
         f"Reference network (teacher)\nUNet + EfficientNet-B0, {TEACHER_PARAMS:.1f} M params, IoU {n['fp32_headline']['teacher']:.3f}",
         BLUE, fs=10, face="#eaf2fc")
    arrow(ax, (tx + tw / 2, ty), (tx + tw / 2, 0.315), color=BLUE, lw=2.0)
    ax.text(tx + tw / 2 + 0.008, 0.372, "knowledge distillation\n(response, feature, combined)",
            ha="left", va="center", fontsize=8.6, color=INK2, linespacing=1.15)
    ax.plot([x0 + 0.062, x1 - 0.062], [0.315, 0.315], color=BLUE, lw=2.0, zorder=2)

    sw, gap = 0.092, 0.009
    sx = x0 + (x1 - x0 - 3 * sw - 2 * gap) / 2
    for i, (key, name, params) in enumerate(STUDENTS):
        cx = sx + i * (sw + gap)
        arrow(ax, (cx + sw / 2, 0.315), (cx + sw / 2, 0.275), color=BLUE, lw=2.0)
        chip(ax, cx, 0.175, sw, 0.10,
             f"{name}\n{params:.1f} M params\nIoU {n['fp32_headline'][key]:.3f}", BLUE, fs=9.2)
    ax.text((x0 + x1) / 2, 0.112,
            "On saturated FSSD the compact networks match the reference network\n"
            "without distillation. On the harder combined corpus, distillation\n"
            f"lifts flood IoU by +{n['kd_gain_lo']:.2f} to +{n['kd_gain_hi']:.2f}.",
            ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.25)

    # =============================== panel 2: quantize and deploy
    x0, x1 = 0.345, 0.655
    panel_frame(ax, x0, x1, PANEL["deploy"], "Quantize to INT8 and deploy", "2")
    cw, ch, cy = 0.088, 0.062, 0.76
    labels = ["ONNX export\n(opset 17, 256 x 256)", "INT8 static PTQ\n80 calibration images",
              "ONNX Runtime\nCPU and WebAssembly"]
    for i, lab in enumerate(labels):
        cx = x0 + 0.014 + i * (cw + 0.018)
        chip(ax, cx, cy, cw, ch, lab, GREEN, fs=9.0)
        if i < 2:
            arrow(ax, (cx + cw, cy + ch / 2), (cx + cw + 0.018, cy + ch / 2), color=GREEN, lw=2.0)

    c = fig.add_axes([x0 + 0.03, 0.355, 0.27, 0.29])
    style_axes(c, 1.08)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    nets = [s for s, _, _ in STUDENTS] + ["teacher"]
    names = [nm for _, nm, _ in STUDENTS] + ["Reference net"]
    xs = np.arange(len(nets)); bw = 0.26
    series = [("FP32", n["fp32"], GREY), ("INT8 per-tensor (default)", n["pt"], RED),
              ("INT8 per-channel", n["pc"], GREEN)]
    for j, (lab, vals, col) in enumerate(series):
        v = [vals[k] for k in nets]
        bars = c.bar(xs + (j - 1) * bw, v, bw, color=col, label=lab, edgecolor="white", linewidth=0.8)
        for b, val in zip(bars, v):
            c.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                   fontsize=7.4, color=INK)
    c.set_xticks(xs, names)
    c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.17), ncol=3, frameon=False, fontsize=8.0,
             handlelength=1.0, columnspacing=0.9)
    ax.text(x0 + 0.014, 0.712, "IoU on FSSD after INT8: per-channel scales repair every CNN,\n"
            "the per-tensor default collapses them; MobileViT stays broken",
            ha="left", va="center", fontsize=9.0, fontweight="bold", color=INK, linespacing=1.2)

    tw3, gap3 = 0.094, 0.009
    tx0 = x0 + (x1 - x0 - 3 * tw3 - 2 * gap3) / 2
    targets = [
        ("Laptop CPU (M2 Pro)",
         f"Lite0 INT8: {n['m2_lite0_int8_fps']:.0f} FPS\n"
         f"{n['m2_teacher_ms'] / n['m2_lite0_int8_ms']:.1f}x faster, {n['lite0_size_x']:.1f}x smaller\n"
         f"than reference, IoU {n['lite0_int8_iou']:.3f}"),
        ("x86 desktop (Ryzen 5)",
         f"INT8 is {n['x86_int8_slow_lo']:.1f}-{n['x86_int8_slow_hi']:.1f}x slower\n"
         f"than FP32 here, so ship\nFP32: {n['x86_fp32_fps_lo']:.0f}-{n['x86_fp32_fps_hi']:.0f} FPS"),
        ("Browser (WASM)",
         f"FP32: {n['wasm_fps_lo']:.0f}-{n['wasm_fps_hi']:.1f} FPS\n"
         f"INT8 loads on ORT-Web 1.22,\n{n['wasm_int8_speedup_lo']:.1f}-{n['wasm_int8_speedup_hi']:.1f}x faster"),
    ]
    for i, (head, body) in enumerate(targets):
        cx = tx0 + i * (tw3 + gap3)
        rbox(ax, cx, 0.085, tw3, 0.16, GREEN, face="#edf7f1", lw=1.3, r=0.008)
        ax.text(cx + tw3 / 2, 0.218, head, ha="center", va="center", fontsize=8.2, fontweight="bold",
                color=INK, zorder=3)
        ax.text(cx + tw3 / 2, 0.148, body, ha="center", va="center", fontsize=7.7, color=INK2,
                zorder=3, linespacing=1.22)
    ax.text((x0 + x1) / 2, 0.283, "Same INT8 file, different verdict per CPU", ha="center",
            va="center", fontsize=9.3, fontweight="bold", color=INK)

    # =============================== panel 3: check false alarms
    x0, x1 = 0.675, 0.985
    panel_frame(ax, x0, x1, PANEL["check"], "Check what responders will see", "3")
    iw = 0.094; ih = iw * (300 / 400) / ASPECT
    caps = [("Water-free scene\n(FloodNet test)", dry_in),
            (f"FSSD-trained model:\n{fa_frac_fssd:.0%} of pixels \"water\"", dry_fssd),
            (f"Retrained with water-free\ndata: {fa_frac_comb:.0%} flagged", dry_comb)]
    for i, (cap, im) in enumerate(caps):
        ix = x0 + 0.012 + i * (iw + 0.0075)
        a = fig.add_axes([ix, 0.655, iw, ih]); a.imshow(im); a.axis("off")
        for s in a.spines.values():
            s.set_visible(True); s.set_edgecolor(RED if i == 1 else GRID); s.set_linewidth(1.2)
        ax.text(ix + iw / 2, 0.645, cap, ha="center", va="top", fontsize=7.8,
                color=(RED if i == 1 else INK2), linespacing=1.15)

    c = fig.add_axes([x0 + 0.04, 0.245, 0.255, 0.285])
    style_axes(c, 1.18, pct=True)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    xs = np.arange(3); bw = 0.36
    for j, (lab, vals, col) in enumerate([("FSSD-trained", n["far_fssd"], RED),
                                          ("Retrained with water-free images", n["far_comb"], GREEN)]):
        bars = c.bar(xs + (j - 0.5) * bw, vals, bw - 0.03, color=col, label=lab, edgecolor="white")
        for b, val in zip(bars, vals):
            c.text(b.get_x() + b.get_width() / 2, val + 0.025, f"{val:.0%}", ha="center", va="bottom",
                   fontsize=8.2, color=INK)
    c.set_xticks(xs, [nm for _, nm, _ in STUDENTS])
    c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.13), ncol=2, frameon=False, fontsize=8.2,
             handlelength=1.0, columnspacing=1.0)
    ax.text(x0 + 0.014, 0.585, f"Share of {n['n_dry']} dry scenes with any predicted water",
            ha="left", va="center", fontsize=9.3, fontweight="bold", color=INK)

    rbox(ax, x0 + 0.012, 0.068, x1 - x0 - 0.024, 0.135, ORANGE, face="#fdf0ea", lw=1.3, r=0.008)
    ax.text((x0 + x1) / 2, 0.136,
            "Distillation copies the reference network's false alarms. Where it\n"
            f"false-alarms, the compact networks do too in {min(n['inh_none']):.0%}-{max(n['inh_none']):.0%} of cases\n"
            f"without KD and in {min(n['inh_comb']):.0%}-{max(n['inh_comb']):.0%} with combined KD.\n"
            "Audit a network's false alarms before distilling from it.",
            ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.25, zorder=3)

    chevron(ax, 0.329, 0.46); chevron(ax, 0.659, 0.46)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print("wrote", out.with_suffix(".png"), out.with_suffix(".pdf"))
    print(f"dry-scene false-alarm fraction: FSSD-trained {fa_frac_fssd:.3f}, combined-trained {fa_frac_comb:.3f}")


if __name__ == "__main__":
    main()
