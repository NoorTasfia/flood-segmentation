"""Tests for scripts/merge_results.py: add one config's student blocks to a
per-fold results file without touching what is already there."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_results import merge_results, main  # noqa: E402


def _base():
    return {"fold_1": {"teacher": {"val": {"iou_mean": 0.9}},
                       "students": {"mobilenetv3_small_none": {"val": {"iou_mean": 0.8}},
                                    "mobilenetv3_small_comb": {"val": {"iou_mean": 0.81}}}},
            "fold_2": {"teacher": {}, "students": {"x_none": {}}}}


def _extra():
    return {"fold_1": {"teacher": {"val": {"iou_mean": 0.95}},
                       "students": {"mobilenetv3_small_gated": {"val": {"iou_mean": 0.82}}}}}


def test_merge_adds_new_student_block():
    out = merge_results(_base(), _extra(), fold=1)
    assert out["fold_1"]["students"]["mobilenetv3_small_gated"]["val"]["iou_mean"] == 0.82


def test_merge_keeps_existing_blocks_and_other_folds():
    out = merge_results(_base(), _extra(), fold=1)
    assert out["fold_1"]["students"]["mobilenetv3_small_none"]["val"]["iou_mean"] == 0.8
    assert out["fold_1"]["teacher"]["val"]["iou_mean"] == 0.9      # extra's teacher ignored
    assert out["fold_2"] == _base()["fold_2"]


def test_merge_refuses_to_overwrite_existing_key():
    extra = {"fold_1": {"students": {"mobilenetv3_small_comb": {"val": {"iou_mean": 0.0}}}}}
    with pytest.raises(KeyError):
        merge_results(_base(), extra, fold=1)


def test_cli_writes_back_to_base(tmp_path: Path):
    base = tmp_path / "results_fold1.json"
    extra = tmp_path / "results_gated_fold1.json"
    base.write_text(json.dumps(_base()))
    extra.write_text(json.dumps(_extra()))
    main(["--base", str(base), "--extra", str(extra), "--fold", "1"])
    merged = json.loads(base.read_text())
    assert set(merged["fold_1"]["students"]) == {"mobilenetv3_small_none", "mobilenetv3_small_comb",
                                                 "mobilenetv3_small_gated"}
