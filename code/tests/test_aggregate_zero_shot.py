"""Tests for scripts/aggregate_zero_shot.py (3-fold zero-shot aggregation)."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_zero_shot.py"
spec = importlib.util.spec_from_file_location("aggregate_zero_shot", SCRIPT)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _fold_json(iou, far):
    return {"roles": {"external_test": {"teacher": {
        "iou_mean": iou, "iou_std": 0.1, "n_images": 94}},
        "floodnet_test": {"teacher": {
            "iou_mean": iou - 0.4, "iou_std": 0.3, "n_images": 424,
            "iou_flood_mean": iou - 0.2, "false_alarm_rate": far,
            "n_flood_images": 209, "n_noflood_images": 215}}}}


def test_aggregate_means_and_stds_across_folds(tmp_path: Path):
    for f, (iou, far) in enumerate([(0.70, 0.9), (0.72, 0.8), (0.68, 1.0)]):
        (tmp_path / f"zero_shot_fold{f}.json").write_text(
            json.dumps(_fold_json(iou, far)))
    out = agg.aggregate(tmp_path)
    t = out["external_test"]["teacher"]
    assert t["n_folds"] == 3
    assert abs(t["iou_mean"] - 0.70) < 1e-9
    assert t["iou_std_folds"] > 0
    fn = out["floodnet_test"]["teacher"]
    assert abs(fn["false_alarm_rate"] - 0.9) < 1e-9
    assert abs(fn["iou_flood_mean"] - 0.50) < 1e-9


def test_aggregate_handles_missing_models_per_fold(tmp_path: Path):
    j0 = _fold_json(0.7, 0.9)
    j1 = _fold_json(0.7, 0.9)
    del j1["roles"]["external_test"]["teacher"]
    (tmp_path / "zero_shot_fold0.json").write_text(json.dumps(j0))
    (tmp_path / "zero_shot_fold1.json").write_text(json.dumps(j1))
    out = agg.aggregate(tmp_path)
    assert out["external_test"]["teacher"]["n_folds"] == 1
