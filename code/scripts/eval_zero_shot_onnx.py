"""Zero-shot evaluation of additional ONNX artifacts, merged into the fold JSON.

Complements ``eval_zero_shot.py`` (which writes ``zero_shot_fold{N}.json``):
evaluates ONNX files ``{model}_fold{N}_{suffix}.onnx`` from ``--artifacts``
per image on the two held-out roles of flood-combined-fsegnet and merges the
results into the existing per-fold JSON under the tag
``{model}_onnx_{tag}``. Used for the per-channel INT8 artifacts of
``ptq_mechanism_ablation.py`` (suffix ``pc_int8``, tag ``int8_pc``).
Re-run ``aggregate_zero_shot.py`` afterwards.

Usage:
    python scripts/eval_zero_shot_onnx.py --combined_root /path/to/combined \
        --artifacts /path/to/ptq_work --fold 0 --suffix pc_int8 --tag int8_pc \
        [--models mobilenetv3_small efficientnet_lite0 mobilevit_xxs] \
        [--json runs/3fold/zero_shot_fold0.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_zero_shot import EVAL_ROLES, evaluate_onnx_per_image, make_eval_loaders


def main() -> None:
    from floodlite.models import STUDENT_BACKBONES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combined_root", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--suffix", default="pc_int8")
    ap.add_argument("--tag", default="int8_pc")
    ap.add_argument("--models", nargs="+", default=list(STUDENT_BACKBONES))
    ap.add_argument("--json", default=None,
                    help="per-fold JSON to merge into (default runs/3fold/zero_shot_fold{N}.json)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--kd", default=None,
                    help="student KD tag for combined-corpus artifacts named {model}_{kd}_fold{N}_{suffix}.onnx")
    ap.add_argument("--rescuenet_root", default=None,
                    help="prepared RescueNet test split; adds the rescuenet_test role")
    ap.add_argument("--roles", nargs="+", default=None,
                    help="evaluate only these roles (default: all available)")
    args = ap.parse_args()

    json_path = Path(args.json or f"runs/3fold/zero_shot_fold{args.fold}.json")
    if json_path.exists():
        out = json.loads(json_path.read_text())
    else:
        out = {"fold": args.fold, "roles": {r: {} for r in EVAL_ROLES}}
    loaders = make_eval_loaders(args.combined_root, num_workers=args.num_workers,
                                rescuenet_root=args.rescuenet_root)
    if args.roles:
        loaders = {r: l for r, l in loaders.items() if r in args.roles}
    for m in args.models:
        stem = f"{m}_{args.kd}_fold{args.fold}" if args.kd else f"{m}_fold{args.fold}"
        op = Path(args.artifacts) / f"{stem}_{args.suffix}.onnx"
        if not op.exists():
            print(f"SKIP (no ONNX): {op}")
            continue
        tag = f"{m}_{args.kd}_onnx_{args.tag}" if args.kd else f"{m}_onnx_{args.tag}"
        for role, loader in loaders.items():
            res = evaluate_onnx_per_image(op, loader)
            out["roles"].setdefault(role, {})[tag] = res
            print(f"{role:>14s}  {tag:35s} IoU {res['iou_mean']:.4f} +/- {res['iou_std']:.4f}"
                  f"  floodIoU {res.get('iou_flood_mean', float('nan')):.4f}"
                  f"  FAR {res.get('false_alarm_rate', float('nan')):.4f}", flush=True)
        json_path.write_text(json.dumps(out, indent=2))
    print(f"Merged into {json_path}")


if __name__ == "__main__":
    main()
