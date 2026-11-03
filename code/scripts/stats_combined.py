"""Per-image paired statistics for the D2 combined-dataset runs (torch-free).

Reads runs/combined/results_fold{0,1,2}.json (per-image IoU on the fixed
FloodNet test split, identical image order in every fold) and the dataset's
manifest.csv (flood_fraction == 0 marks a water-free image). For each student:

  * flood-IoU contrast comb - none over flood-containing test images, and
    FAR contrast comb - none over water-free test images. Per-image values are
    averaged across the three folds' checkpoints first (the test set is the
    same image list in every fold), then a percentile bootstrap over images
    gives a 95% CI.
  * teacher-inheritance check: over water-free images and all three folds,
    P(student false alarm | teacher false alarm) vs
    P(student false alarm | teacher correct rejection), for none and comb.

On a water-free image per-image IoU is 1 for a correct rejection and 0 for
any predicted water, so a false alarm is exactly IoU < 1.

Usage:  python scripts/stats_combined.py --data_root /path/to/combined \
            [--runs runs/combined] [--out runs/combined/stats_combined.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from floodlite.manifest import load_manifest, split_manifest  # noqa: E402

FOLDS = (0, 1, 2)
STUDENTS = ("mobilenetv3_small", "efficientnet_lite0", "mobilevit_xxs")


def boot_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(lo), float(hi)],
        "p_le_0": float((means <= 0).mean()),
        "n": int(len(deltas)),
    }


def load_test_iou(runs: Path) -> dict[str, np.ndarray]:
    """model -> array [fold, image] of per-image test IoU."""
    per: dict[str, list] = {}
    for f in FOLDS:
        blocks = json.loads((runs / f"results_fold{f}.json").read_text(encoding="utf-8"))[f"fold_{f}"]
        models = {"teacher": blocks["teacher"], **blocks["students"]}
        for name, m in models.items():
            per.setdefault(name, []).append(np.asarray(m["test"]["per_image_iou"], dtype=float))
    return {k: np.stack(v) for k, v in per.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--runs", default="runs/combined")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = Path(args.runs)

    test_rows = split_manifest(load_manifest(args.data_root), 0)["test"]
    dry = np.array([float(r["flood_fraction"]) == 0.0 for r in test_rows])
    iou = load_test_iou(runs)
    n_img = iou["teacher"].shape[1]
    assert n_img == len(dry), f"manifest test rows {len(dry)} != per-image IoU {n_img}"

    fa = {k: (v[:, dry] < 1.0) for k, v in iou.items()}  # [fold, dry image]
    # consistency with the stored FAR
    for f in FOLDS:
        blocks = json.loads((runs / f"results_fold{f}.json").read_text(encoding="utf-8"))[f"fold_{f}"]
        stored = blocks["teacher"]["test"]["false_alarm_rate"]
        assert abs(fa["teacher"][f].mean() - stored) < 1e-9, (f, fa["teacher"][f].mean(), stored)

    out: dict = {"n_test": int(n_img), "n_dry": int(dry.sum()), "n_flood": int((~dry).sum()),
                 "kd_contrast": {}, "teacher_inheritance": {}}
    t_fa = fa["teacher"]
    for s in STUDENTS:
        none, comb = iou[f"{s}_none"], iou[f"{s}_comb"]
        d_flood = (comb[:, ~dry] - none[:, ~dry]).mean(axis=0)
        d_far = (fa[f"{s}_comb"].astype(float) - fa[f"{s}_none"].astype(float)).mean(axis=0)
        out["kd_contrast"][s] = {
            "flood_iou_comb_minus_none": boot_ci(d_flood),
            "far_comb_minus_none": boot_ci(d_far),
        }
        inh = {}
        for kd in ("none", "comb"):
            sfa = fa[f"{s}_{kd}"]
            inh[kd] = {
                "p_fa_given_teacher_fa": float(sfa[t_fa].mean()) if t_fa.any() else None,
                "p_fa_given_teacher_ok": float(sfa[~t_fa].mean()),
                "n_teacher_fa": int(t_fa.sum()),
                "n_teacher_ok": int((~t_fa).sum()),
            }
        out["teacher_inheritance"][s] = inh

    dest = Path(args.out) if args.out else runs / "stats_combined.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"test images {n_img}: {int((~dry).sum())} flood, {int(dry.sum())} water-free")
    for s in STUDENTS:
        c = out["kd_contrast"][s]
        fl, fr = c["flood_iou_comb_minus_none"], c["far_comb_minus_none"]
        print(f"{s:20s} dFloodIoU {fl['mean']:+.3f} [{fl['ci95'][0]:+.3f},{fl['ci95'][1]:+.3f}]"
              f"  dFAR {fr['mean']:+.3f} [{fr['ci95'][0]:+.3f},{fr['ci95'][1]:+.3f}]")
        for kd in ("none", "comb"):
            i = out["teacher_inheritance"][s][kd]
            print(f"   {kd:4s} P(FA|teacher FA)={i['p_fa_given_teacher_fa']:.3f} "
                  f"(n={i['n_teacher_fa']})  P(FA|teacher OK)={i['p_fa_given_teacher_ok']:.3f} (n={i['n_teacher_ok']})")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
