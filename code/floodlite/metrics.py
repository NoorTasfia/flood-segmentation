"""Segmentation metrics."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class SegMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b + eps)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    """Compute pixel accuracy, precision, recall, F1, IoU for binary masks."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    tgt = (target >= 0.5).float()

    tp = (pred * tgt).sum()
    fp = (pred * (1 - tgt)).sum()
    fn = ((1 - pred) * tgt).sum()
    tn = ((1 - pred) * (1 - tgt)).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    if (tp + fp + fn).item() == 0:
        # No flood in ground truth and none predicted: a correct rejection.
        # (0/(0+eps)=0 would cap a perfect model at ~0.5 macro IoU on sets
        # where half the images contain no water, e.g. FloodNet test.)
        return SegMetrics(acc.item(), 1.0, 1.0, 1.0, 1.0)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    iou = _safe_div(tp, tp + fp + fn)

    return SegMetrics(acc.item(), prec.item(), rec.item(), f1.item(), iou.item())


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "iou")


def summarise_per_image(per_image: list[dict]) -> dict:
    """Macro mean/std per metric over a per-image metric list (+ IoU list kept).

    When the per-image dicts carry ``gt_flood``/``pred_flood`` flags, also
    reports flood-only IoU (over images that contain water) and the
    false-alarm rate (share of water-free images where water was predicted).
    """
    import numpy as np
    out: dict = {"n_images": len(per_image)}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in per_image])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    out["per_image_iou"] = [float(m["iou"]) for m in per_image]

    if per_image and "gt_flood" in per_image[0]:
        flood = [m for m in per_image if m["gt_flood"]]
        noflood = [m for m in per_image if not m["gt_flood"]]
        out["n_flood_images"] = len(flood)
        out["n_noflood_images"] = len(noflood)
        if flood:
            ious = np.array([m["iou"] for m in flood])
            out["iou_flood_mean"] = float(ious.mean())
            out["iou_flood_std"] = float(ious.std())
        if noflood:
            out["false_alarm_rate"] = float(
                np.mean([m["pred_flood"] for m in noflood]))
    return out


def per_image_record(logits: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> dict:
    """One image's metric dict, with gt/pred flood-presence flags."""
    m = compute_metrics(logits, target, threshold)
    rec = {k: getattr(m, k) for k in METRIC_KEYS}
    rec["gt_flood"] = bool((target >= 0.5).any().item())
    rec["pred_flood"] = bool((torch.sigmoid(logits) >= threshold).any().item())
    return rec


@torch.no_grad()
def evaluate_per_image(model, loader, device) -> dict:
    """Per-image (macro) metrics of a PyTorch model over a loader."""
    model = model.to(device)
    model.eval()
    per_image = []
    for x, y in loader:
        logits = model(x.to(device))
        y = y.to(device)
        for i in range(logits.size(0)):
            per_image.append(per_image_record(logits[i:i + 1], y[i:i + 1]))
    return summarise_per_image(per_image)
