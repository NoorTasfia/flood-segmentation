"""D1: zero-shot evaluation of FSSD-trained models on the combined dataset.

Evaluates existing FSSD checkpoints (no fine-tuning) on the two held-out
roles of flood-combined-fsegnet:

- ``floodnet_test``  (424 FloodNet test images; aerial, mostly unflooded)
- ``external_test``  (94 AIFloodSense images; never used for training anywhere)

Metrics are computed **per image** (macro mean +/- std, with the per-image IoU
list kept in the JSON for paired bootstrap/Wilcoxon analysis), not
batch-averaged. When ``--exports`` is given, the FP32 and INT8 ONNX exports
are evaluated too, which quantifies whether INT8 degrades more than FP32
under domain shift.

Hard gate: run scripts/check_fssd_overlap.py first; this evaluation is only
valid if no FSSD near-duplicates sit in these roles.

Usage:
    python scripts/eval_zero_shot.py --combined_root /path/to/dataset_combined \
        --ckpt_dir runs/3fold [--exports exports] [--fold 0] \
        [--kd none comb] [--out runs/3fold/zero_shot.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_ROLES = ("floodnet_test", "external_test")


def make_eval_loaders(root: str | Path, *, img_size: int = 256,
                      batch_size: int = 8, num_workers: int = 2) -> dict:
    """Eval-only DataLoaders for the combined dataset's held-out roles."""
    from torch.utils.data import DataLoader
    from floodlite.data import FSSD
    from floodlite.manifest import load_manifest, pairs_from_rows

    root = Path(root)
    rows = load_manifest(root)
    loaders = {}
    for role in EVAL_ROLES:
        pairs = pairs_from_rows(root, [r for r in rows if r["role"] == role])
        ds = FSSD(pairs, img_size=img_size, train=False)
        loaders[role] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)
    return loaders


def evaluate_per_image(model, loader, device) -> dict:
    """Per-image (macro) metrics of a PyTorch model over a loader."""
    from floodlite.metrics import evaluate_per_image as _impl
    return _impl(model, loader, device)


def evaluate_onnx_per_image(onnx_path: str | Path, loader) -> dict:
    """Per-image metrics of an ONNX model (FP32 or INT8) over a loader."""
    import torch
    import onnxruntime as ort
    from floodlite.metrics import per_image_record, summarise_per_image

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    per_image = []
    for x, y in loader:
        xn = x.numpy().astype(np.float32)
        for i in range(xn.shape[0]):
            logits = sess.run(None, {in_name: xn[i:i + 1]})[0]
            per_image.append(per_image_record(torch.from_numpy(logits), y[i:i + 1]))
    return summarise_per_image(per_image)


def main() -> None:
    import torch
    from floodlite.models import (make_teacher, make_baseline_mobilenetv2,
                                  make_student, STUDENT_BACKBONES)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combined_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--exports", default=None,
                    help="Directory with *.onnx exports; enables the FP32/INT8 "
                         "ONNX domain-shift comparison.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--kd", nargs="+", default=["none", "comb"],
                    choices=["none", "resp", "feat", "comb"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="runs/3fold/zero_shot.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = make_eval_loaders(args.combined_root, batch_size=args.batch_size)
    ckpt_dir = Path(args.ckpt_dir)
    out: dict = {"fold": args.fold, "roles": {r: {} for r in EVAL_ROLES}}

    def _eval_pt(tag: str, factory, ckpt: Path) -> None:
        if not ckpt.exists():
            print(f"SKIP (no checkpoint): {ckpt}")
            return
        model = factory()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        for role, loader in loaders.items():
            res = evaluate_per_image(model, loader, device)
            out["roles"][role][tag] = res
            print(f"{role:>14s}  {tag:35s} IoU {res['iou_mean']:.4f} "
                  f"+/- {res['iou_std']:.4f}  (n={res['n_images']})")

    _eval_pt("teacher", make_teacher, ckpt_dir / f"teacher_fold{args.fold}.pt")
    _eval_pt("baseline_mobilenetv2", make_baseline_mobilenetv2,
             ckpt_dir / f"baseline_mobilenetv2_fold{args.fold}.pt")
    for sname in STUDENT_BACKBONES:
        for kd in args.kd:
            _eval_pt(f"{sname}_{kd}", lambda s=sname: make_student(s),
                     ckpt_dir / f"{sname}_{kd}_fold{args.fold}.pt")

    if args.exports:
        exports = Path(args.exports)
        for sname in STUDENT_BACKBONES:
            for precision in ("fp32", "int8"):
                op = next((c for c in (
                    exports / f"{sname}_{precision}.onnx",
                    exports / f"{sname}_fold{args.fold}_{precision}.onnx",
                ) if c.exists()), None)
                if op is None:
                    print(f"SKIP (no ONNX): {sname}_{precision}")
                    continue
                tag = f"{sname}_onnx_{precision}"
                for role, loader in loaders.items():
                    res = evaluate_onnx_per_image(op, loader)
                    out["roles"][role][tag] = res
                    print(f"{role:>14s}  {tag:35s} IoU {res['iou_mean']:.4f} "
                          f"+/- {res['iou_std']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
