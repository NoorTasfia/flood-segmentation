"""Tests for ground-truth-gated distillation (floodlite.losses).

The gate distils only where the teacher is right: response and feature KD
terms are masked by the per-pixel teacher-correctness map.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=Exception)

from floodlite.losses import (  # noqa: E402
    FeatureKDLoss,
    response_kd_loss,
    teacher_correct_mask,
)


def test_mask_marks_teacher_correct_pixels():
    t = torch.tensor([[[[3., -3.], [3., -3.]]]])
    y = torch.tensor([[[[1., 0.], [0., 1.]]]])
    assert teacher_correct_mask(t, y).tolist() == [[[[1., 1.], [0., 0.]]]]


def test_mask_is_detached_float():
    t = torch.randn(2, 1, 8, 8, requires_grad=True)
    y = (torch.rand(2, 1, 8, 8) > 0.5).float()
    m = teacher_correct_mask(t, y)
    assert m.dtype == torch.float32 and not m.requires_grad and m.shape == y.shape


def test_masked_response_equals_unmasked_when_teacher_perfect(tiny_batch):
    _, y = tiny_batch
    t = (y * 2 - 1) * 6          # confident and always right
    s = torch.randn_like(y)
    full = response_kd_loss(s, t)
    gated = response_kd_loss(s, t, mask=teacher_correct_mask(t, y))
    assert torch.allclose(gated, full)


def test_masked_response_is_zero_when_teacher_all_wrong(tiny_batch):
    _, y = tiny_batch
    t = (1 - y) * 12 - 6         # confident and always wrong
    s = torch.randn_like(y)
    gated = response_kd_loss(s, t, mask=teacher_correct_mask(t, y))
    assert gated.item() == 0.0 and torch.isfinite(gated)


def test_masked_response_averages_over_correct_pixels_only():
    y = torch.tensor([[[[1., 0.]]]])
    t = torch.tensor([[[[6., 6.]]]])      # right on pixel 0, wrong on pixel 1
    s = torch.tensor([[[[0., 0.]]]])
    m = teacher_correct_mask(t, y)
    gated = response_kd_loss(s, t, mask=m)
    only_first = response_kd_loss(s[..., :1], t[..., :1])
    assert torch.allclose(gated, only_first)


def test_feature_mask_is_area_averaged_and_safe():
    torch.manual_seed(0)
    loss = FeatureKDLoss(4, 8)
    f_s = torch.randn(1, 4, 32, 32)
    f_t = torch.randn(1, 8, 32, 32)
    mask = torch.zeros(1, 1, 256, 256)
    mask[..., :128, :] = 1                # top half correct
    assert loss(f_s, f_t, mask=mask).item() > 0
    assert loss(f_s, f_t, mask=torch.zeros(1, 1, 256, 256)).item() == 0.0


def test_feature_mask_all_ones_matches_unmasked():
    torch.manual_seed(0)
    loss = FeatureKDLoss(4, 8)
    f_s = torch.randn(1, 4, 32, 32)
    f_t = torch.randn(1, 8, 32, 32)
    full = loss(f_s, f_t)
    assert torch.allclose(loss(f_s, f_t, mask=torch.ones(1, 1, 256, 256)), full, atol=1e-6)
