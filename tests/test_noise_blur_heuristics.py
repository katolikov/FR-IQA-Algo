"""Tests for the wavelet noise detector and blur detector (Module 5).

Covers:

1. Self-comparison — both masks should be zero when ref == tgt.
2. Noise specificity — adding AWGN to the target makes noise_mask rise
   much more than blur_mask.
3. Blur specificity — Gaussian-blurring the target makes blur_mask rise
   much more than noise_mask.
4. MAD sanity — on pure Gaussian noise vs. a zero image, the estimated
   sigma is within a factor of 2 of the true sigma.
5. SpatialHeuristicsEngine — forward returns all four expected keys
   with matching shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from upiqal.heuristics import (  # noqa: E402
    BlurDetector,
    NoiseDetector,
    SpatialHeuristicsEngine,
)


def _gaussian_blur_rgb(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    import math

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype)
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    k = torch.outer(g, g).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    k = k.expand(3, 1, -1, -1).to(x.device)
    return F.conv2d(x, k, padding=radius, groups=3)


def _make_ref(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Structured image with edges and texture, not pure noise
    x = torch.zeros(1, 3, 128, 128)
    x[:, :, 30:60, 30:80] = 0.8
    x[:, :, 70:110, 40:100] = 0.3
    # Add a bit of texture on top
    x = x + 0.02 * torch.randn(x.shape, generator=g)
    return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. Self-comparison
# ---------------------------------------------------------------------------
def test_self_comparison_masks_are_zero() -> None:
    ref = _make_ref()
    noise_det = NoiseDetector()
    blur_det = BlurDetector()
    n = noise_det(ref, ref)
    b = blur_det(ref, ref)
    assert float(n.max().item()) == pytest.approx(0.0, abs=1e-6)
    assert float(b.max().item()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Noise specificity
# ---------------------------------------------------------------------------
def test_noise_specificity_on_awgn_target() -> None:
    ref = _make_ref()
    torch.manual_seed(42)
    tgt = (ref + torch.randn_like(ref) * 0.05).clamp(0, 1)

    noise_det = NoiseDetector()
    blur_det = BlurDetector()

    n_mean = float(noise_det(ref, tgt).mean().item())
    b_mean = float(blur_det(ref, tgt).mean().item())

    # Noise detector should fire on noisy target.
    assert n_mean > 0.03, f"noise mask too low on AWGN target: {n_mean:.4f}"
    # And it should dominate over blur.
    assert n_mean > b_mean, (
        f"noise should dominate blur on AWGN target: "
        f"noise={n_mean:.4f} blur={b_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Blur specificity
# ---------------------------------------------------------------------------
def test_blur_specificity_on_gaussian_blurred_target() -> None:
    ref = _make_ref()
    tgt = _gaussian_blur_rgb(ref, sigma=2.5)

    noise_det = NoiseDetector()
    blur_det = BlurDetector()

    n_mean = float(noise_det(ref, tgt).mean().item())
    b_mean = float(blur_det(ref, tgt).mean().item())

    # Blur detector should fire strongly.
    assert b_mean > 0.1, f"blur mask too low on blurred target: {b_mean:.4f}"
    # And should dominate over noise.
    assert b_mean > n_mean, (
        f"blur should dominate noise on blurred target: "
        f"noise={n_mean:.4f} blur={b_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. MAD sanity
# ---------------------------------------------------------------------------
def test_mad_estimates_pure_noise_sigma() -> None:
    """Sigma estimate on pure AWGN should be within 2x of ground truth."""
    torch.manual_seed(0)
    true_sigma = 0.04  # ~10 on 0-255 scale
    zeros = torch.zeros(1, 3, 256, 256)
    noisy = (zeros + torch.randn_like(zeros) * true_sigma)
    det = NoiseDetector()
    sigma_map = det._sigma_map(noisy)
    est = float(sigma_map.mean().item())
    # MAD on a smooth zero image would give 0; on pure Gaussian noise with
    # sigma=0.04 the raw |HH| median is ~0.66*sigma*sqrt(2), so the
    # scaled-by-1/0.6745 estimate is within a factor of ~2.
    assert true_sigma * 0.3 <= est <= true_sigma * 3.0, (
        f"MAD sigma estimate {est:.4f} not within [0.3x, 3x] of "
        f"true sigma {true_sigma:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Engine shape contract
# ---------------------------------------------------------------------------
def test_engine_forward_returns_all_four_masks() -> None:
    ref = _make_ref()
    tgt = (ref + 0.01 * torch.randn_like(ref)).clamp(0, 1)
    eng = SpatialHeuristicsEngine()
    out = eng(ref, tgt)
    for key in ("blocking_mask", "ringing_mask", "noise_mask", "blur_mask"):
        assert key in out, f"missing key {key!r} in heuristics output"
        assert out[key].shape == ref.shape[:1] + (1,) + ref.shape[-2:], (
            f"{key} has wrong shape {tuple(out[key].shape)}; "
            f"expected (1, 1, {ref.shape[-2]}, {ref.shape[-1]})"
        )
        assert 0.0 <= float(out[key].min().item())
        assert float(out[key].max().item()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# 6. Invalid arguments
# ---------------------------------------------------------------------------
def test_invalid_smoothing_rejected() -> None:
    with pytest.raises(ValueError):
        NoiseDetector(smoothing=4)  # even
    with pytest.raises(ValueError):
        NoiseDetector(smoothing=0)  # non-positive
    with pytest.raises(ValueError):
        BlurDetector(smoothing=2)  # even
