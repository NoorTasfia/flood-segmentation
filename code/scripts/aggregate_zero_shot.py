"""Aggregate per-fold zero-shot results into cross-fold mean +/- std.

Reads runs/3fold/zero_shot_fold{N}.json (produced by eval_zero_shot.py, one
per fold) and prints, per (role, model), the cross-fold mean and std of the
image-macro metrics — the numbers Table T6 reports. Torch-free.

Usage:
    python scripts/aggregate_zero_shot.py [--dir runs/3fold] [--out runs/3fold/zero_shot_3fold.json]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

AGG_KEYS = ("iou_mean", "iou_flood_mean", "false_alarm_rate")


def aggregate(folder: str | Path) -> dict:
    """{role: {model: {metric means/stds across folds, n_folds}}}."""
    folder = Path(folder)
    fold_files = sorted(folder.glob("zero_shot_fold*.json"))
    if not fold_files:
        raise FileNotFoundError(f"No zero_shot_fold*.json under {folder}")
    folds = [json.loads(p.read_text())["roles"] for p in fold_files]

    out: dict = {}
    roles = {role for f in folds for role in f}
    for role in sorted(roles):
        out[role] = {}
        models = {m for f in folds for m in f.get(role, {})}
        for model in sorted(models):
            entries = [f[role][model] for f in folds
                       if model in f.get(role, {})]
            agg = {"n_folds": len(entries)}
            for key in AGG_KEYS:
                vals = [e[key] for e in entries if key in e]
                if vals:
                    agg[key] = float(np.mean(vals))
                    agg[f"{key.replace('_mean', '')}_std_folds"] = float(np.std(vals))
            # carry the (fold-invariant) image counts through
            for key in ("n_images", "n_flood_images", "n_noflood_images"):
                if key in entries[0]:
                    agg[key] = entries[0][key]
            out[role][model] = agg
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="runs/3fold")
    ap.add_argument("--out", default="runs/3fold/zero_shot_3fold.json")
    args = ap.parse_args()

    out = aggregate(args.dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    for role, models in out.items():
        print(f"=== {role} ===")
        for model, a in models.items():
            parts = [f"IoU {a['iou_mean']:.3f}+/-{a.get('iou_std_folds', 0):.3f}"]
            if "iou_flood_mean" in a:
                parts.append(f"floodIoU {a['iou_flood_mean']:.3f}"
                             f"+/-{a.get('iou_flood_std_folds', 0):.3f}")
            if "false_alarm_rate" in a:
                parts.append(f"FAR {a['false_alarm_rate']:.3f}"
                             f"+/-{a.get('false_alarm_rate_std_folds', 0):.3f}")
            print(f"  {model:35s} n_folds={a['n_folds']}  " + "  ".join(parts))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
