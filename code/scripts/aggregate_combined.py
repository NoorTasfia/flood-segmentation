"""Aggregate D2 combined-dataset folds into mean +/- std per model and split.

Usage: python scripts/aggregate_combined.py [runs/combined] [-o runs/combined/summary.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SPLITS = ("val", "test", "external")
METRICS = ("iou_mean", "iou_flood_mean", "false_alarm_rate")


def load_folds(root: Path) -> dict[str, dict]:
    folds = {}
    for p in sorted(root.glob("results_fold*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for fold_key, blocks in data.items():
            folds[fold_key] = blocks
    return folds


def per_model(blocks: dict) -> dict[str, dict]:
    out = {"teacher": blocks["teacher"]}
    if blocks.get("baseline_mobilenetv2"):
        out["baseline_mobilenetv2"] = blocks["baseline_mobilenetv2"]
    out.update(blocks["students"])
    return out


def aggregate(folds: dict[str, dict]) -> dict:
    fold_keys = sorted(folds)
    models = list(per_model(folds[fold_keys[0]]))
    summary: dict = {"folds": fold_keys, "models": {}}
    for m in models:
        entry: dict = {}
        for split in SPLITS:
            entry[split] = {}
            for metric in METRICS:
                vals = [per_model(folds[f])[m][split].get(metric) for f in fold_keys]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                entry[split][metric] = {
                    "mean": statistics.fmean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "per_fold": vals,
                }
        summary["models"][m] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs/combined")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    summary = aggregate(load_folds(root))
    out = Path(args.out) if args.out else root / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"folds: {summary['folds']}")
    for m, e in summary["models"].items():
        bits = []
        for split in SPLITS:
            for metric, label in (("iou_mean", "IoU"), ("iou_flood_mean", "flood"), ("false_alarm_rate", "FAR")):
                v = e[split].get(metric)
                if v is not None and not (split != "test" and metric == "iou_flood_mean"):
                    bits.append(f"{split}.{label} {v['mean']:.3f}±{v['std']:.3f}")
        print(f"{m:26s} " + "  ".join(bits))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
