"""Shared pytest fixtures for floodlite tests.

torch is imported lazily: the phash/manifest tests (and the FSSD-overlap
dedup gate they cover) must run on machines where torch is unavailable or
its DLLs fail to load.
"""
from __future__ import annotations
import pytest

try:
    import torch
except Exception:  # pragma: no cover - ImportError or Windows DLL OSError
    torch = None


@pytest.fixture
def tiny_batch():
    """A single 3-channel 256x256 image and binary mask, on CPU."""
    if torch is None:
        pytest.skip("torch unavailable")
    x = torch.randn(1, 3, 256, 256)
    y = (torch.rand(1, 1, 256, 256) > 0.5).float()
    return x, y


@pytest.fixture
def cpu_device():
    if torch is None:
        pytest.skip("torch unavailable")
    return torch.device("cpu")
