"""Per-image metric semantics on flood-absent images.

215 of the 424 FloodNet-test images contain no flood pixels. An
empty-ground-truth, empty-prediction image is a *correct* prediction and must
score IoU 1.0 (the old 0/(0+eps)=0 behaviour capped a perfect model at ~0.49
macro IoU). Flood-absent images additionally feed a false-alarm rate, and a
flood-only IoU is reported over the images that do contain water.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=Exception)

from floodlite.metrics import compute_metrics, evaluate_per_image, summarise_per_image  # noqa: E402

NEG = -100.0  # logit -> sigmoid ~ 0
POS = 100.0   # logit -> sigmoid ~ 1


def test_empty_gt_empty_pred_scores_perfect():
    logits = torch.full((1, 1, 8, 8), NEG)
    target = torch.zeros((1, 1, 8, 8))
    m = compute_metrics(logits, target)
    assert m.iou == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.accuracy == pytest.approx(1.0, abs=1e-6)


def test_empty_gt_nonempty_pred_scores_zero():
    logits = torch.full((1, 1, 8, 8), POS)
    target = torch.zeros((1, 1, 8, 8))
    m = compute_metrics(logits, target)
    assert m.iou == pytest.approx(0.0, abs=1e-6)


def test_nonempty_gt_unchanged():
    logits = torch.full((1, 1, 8, 8), POS)
    target = torch.ones((1, 1, 8, 8))
    assert compute_metrics(logits, target).iou == pytest.approx(1.0, abs=1e-6)


def test_summary_splits_flood_and_noflood_images():
    per_image = [
        # two flood images: IoU 0.8 and 0.6
        {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "iou": 0.8,
         "gt_flood": True, "pred_flood": True},
        {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "iou": 0.6,
         "gt_flood": True, "pred_flood": True},
        # two no-flood images: one clean rejection, one false alarm
        {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "iou": 1.0,
         "gt_flood": False, "pred_flood": False},
        {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "iou": 0.0,
         "gt_flood": False, "pred_flood": True},
    ]
    s = summarise_per_image(per_image)
    assert s["n_images"] == 4
    assert s["n_flood_images"] == 2
    assert s["n_noflood_images"] == 2
    assert s["iou_flood_mean"] == pytest.approx(0.7)
    assert s["false_alarm_rate"] == pytest.approx(0.5)
    assert s["iou_mean"] == pytest.approx((0.8 + 0.6 + 1.0 + 0.0) / 4)


class _ConstNeg(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1, *x.shape[2:]), NEG)


def test_evaluate_per_image_records_flood_flags():
    xs = torch.randn(3, 3, 8, 8)
    ys = torch.stack([torch.ones(1, 8, 8), torch.zeros(1, 8, 8),
                      torch.zeros(1, 8, 8)])
    loader = [(xs, ys)]
    s = evaluate_per_image(_ConstNeg(), loader, "cpu")
    # predicts nothing: misses the flood image, cleanly rejects the empty ones
    assert s["n_flood_images"] == 1 and s["n_noflood_images"] == 2
    assert s["false_alarm_rate"] == pytest.approx(0.0)
    assert s["iou_flood_mean"] == pytest.approx(0.0, abs=1e-6)
    assert s["per_image_iou"][1] == pytest.approx(1.0)
