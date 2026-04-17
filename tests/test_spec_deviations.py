"""Regression tests for the Module 5 + pipeline spec-alignment changes.

Covers:

* NFA a-contrario blocking (item 1): self-comparison returns zero mask;
  natural noise doesn't trigger; synthetic 8x8 flat-block images do.
* Systematic/statistical error quotient for ringing (item 2): structural
  contract unchanged (self-comparison zero, differential non-negative).
* Multi-level Haar wavelet for noise (item 3): self-comparison zero;
  AWGN target triggers; pure smooth target does not.
* Learnable A-DISTS sigmoid (item 5): sigmoid_gain / sigmoid_bias are
  nn.Parameter instances when requested, plain buffers otherwise.
* Learnable aggregation head (item 6): the six aggregation scalars are
  nn.Parameter instances by default.
* NLL-style score normalisation (item 7): score_mode="nll" on identical
  inputs returns exactly 1.0; score_mode="sigmoid" returns the legacy
  0.9526.
* Multi-scale pyramid (item 4): passing ref_full/tgt_full through
  UPIQAL.forward does not break the pipeline and leaves the diagnostic
  tensor at the feature-level (H, W) resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use the VGG16 mock from the main test module so these tests don't need
# the 528 MB pretrained checkpoint.
from tests.test_upiqal import _mock_vgg16  # noqa: E402

from upiqal.heuristics import (  # noqa: E402
    JPEGBlockingDetector,
    GibbsRingingDetector,
    NoiseDetector,
)
from upiqal.features import DeepStatisticalExtractor  # noqa: E402
from upiqal.model import UPIQAL  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def natural_pair():
    torch.manual_seed(42)
    ref = torch.rand(1, 3, 128, 128) * 0.5 + 0.3
    tgt = (ref + 0.02 * torch.randn_like(ref)).clamp(0, 1)
    return ref, tgt


@pytest.fixture
def blocky_pair():
    torch.manual_seed(7)
    ref = torch.rand(1, 3, 128, 128) * 0.5 + 0.3
    tgt = ref.clone()
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            m = tgt[:, :, i:i + 8, j:j + 8].mean(dim=(-2, -1), keepdim=True)
            tgt[:, :, i:i + 8, j:j + 8] = m
    return ref, tgt


# ---------------------------------------------------------------------------
# Item 1: NFA blocking
# ---------------------------------------------------------------------------
def test_nfa_blocking_self_is_zero(natural_pair):
    ref, _ = natural_pair
    det = JPEGBlockingDetector()
    assert float(det(ref, ref).mean().item()) == pytest.approx(0.0)


def test_nfa_blocking_ignores_natural_noise(natural_pair):
    ref, tgt = natural_pair
    det = JPEGBlockingDetector()
    # Default params are calibrated so small AWGN does not trigger.
    assert float(det(ref, tgt).mean().item()) == pytest.approx(0.0, abs=1e-6)


def test_nfa_blocking_detects_synthetic_blocks(blocky_pair):
    ref, tgt = blocky_pair
    det = JPEGBlockingDetector()
    mean = float(det(ref, tgt).mean().item())
    assert mean > 0.05, (
        f"blocking detector should fire on 8x8 flat-block image, got {mean:.4f}"
    )


def test_nfa_binomial_tail_monotone():
    """P[K>=k] should be monotone non-increasing in k."""
    det = JPEGBlockingDetector()
    p = 0.05
    n = 200
    values = [det._binom_tail_ge(k, n, p) for k in range(0, n + 1, 20)]
    for a, b in zip(values, values[1:]):
        assert a >= b - 1e-12, f"non-monotone binomial tail: {values}"


# ---------------------------------------------------------------------------
# Item 2: eps ringing
# ---------------------------------------------------------------------------
def test_epsilon_ringing_self_is_zero(natural_pair):
    ref, _ = natural_pair
    det = GibbsRingingDetector()
    assert float(det(ref, ref).mean().item()) == pytest.approx(0.0)


def test_epsilon_ringing_differential_nonneg(natural_pair):
    ref, tgt = natural_pair
    det = GibbsRingingDetector()
    out = det(ref, tgt)
    assert float(out.min().item()) >= 0.0
    assert float(out.max().item()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Item 3: multi-level wavelet noise
# ---------------------------------------------------------------------------
def test_multilevel_wavelet_self_is_zero(natural_pair):
    ref, _ = natural_pair
    det = NoiseDetector(levels=3)
    assert float(det(ref, ref).max().item()) == pytest.approx(0.0, abs=1e-6)


def test_multilevel_wavelet_fires_on_awgn(natural_pair):
    ref, _ = natural_pair
    torch.manual_seed(1)
    tgt = (ref + 0.05 * torch.randn_like(ref)).clamp(0, 1)
    det = NoiseDetector(levels=3)
    mean = float(det(ref, tgt).mean().item())
    assert mean > 0.03, f"wavelet noise detector missed AWGN, mean={mean:.4f}"


def test_multilevel_wavelet_levels_monotone_noise(natural_pair):
    """Adding more decomposition levels should not reduce the detector
    response on pure AWGN (it should pick up the same or more scales)."""
    ref, _ = natural_pair
    torch.manual_seed(1)
    tgt = (ref + 0.05 * torch.randn_like(ref)).clamp(0, 1)
    m1 = float(NoiseDetector(levels=1)(ref, tgt).mean().item())
    m3 = float(NoiseDetector(levels=3)(ref, tgt).mean().item())
    assert m3 >= m1 - 1e-4, (
        f"3-level wavelet mean ({m3:.4f}) below 1-level ({m1:.4f})"
    )


def test_invalid_levels_rejected():
    with pytest.raises(ValueError):
        NoiseDetector(levels=0)


# ---------------------------------------------------------------------------
# Item 5: learnable A-DISTS sigmoid
# ---------------------------------------------------------------------------
def test_learnable_sigmoid_is_parameter():
    ds = DeepStatisticalExtractor(pretrained=False)
    assert isinstance(ds.sigmoid_gain, nn.Parameter)
    assert isinstance(ds.sigmoid_bias, nn.Parameter)
    assert float(ds.sigmoid_gain.item()) == pytest.approx(5.0)
    assert float(ds.sigmoid_bias.item()) == pytest.approx(1.0)


def test_fixed_sigmoid_is_buffer():
    ds = DeepStatisticalExtractor(pretrained=False, learnable_sigmoid=False)
    assert not isinstance(ds.sigmoid_gain, nn.Parameter)
    assert not isinstance(ds.sigmoid_bias, nn.Parameter)


# ---------------------------------------------------------------------------
# Item 6: learnable aggregation head
# ---------------------------------------------------------------------------
@patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
def test_learnable_aggregation_weights_are_parameters(_mock):
    model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
    for name in (
        "w_color", "w_anomaly", "w_structure", "w_heuristic",
        "score_scale", "score_center",
    ):
        val = getattr(model, name)
        assert isinstance(val, nn.Parameter), f"{name} is not an nn.Parameter"


@patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
def test_fixed_aggregation_weights_are_floats(_mock):
    model = UPIQAL(
        pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3,
        learnable_aggregation=False,
    )
    for name in (
        "w_color", "w_anomaly", "w_structure", "w_heuristic",
        "score_scale", "score_center",
    ):
        val = getattr(model, name)
        assert not isinstance(val, nn.Parameter)


# ---------------------------------------------------------------------------
# Item 7: NLL score normalisation
# ---------------------------------------------------------------------------
@patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
def test_nll_score_identical_inputs_is_one(_mock):
    torch.manual_seed(0)
    ref = torch.rand(1, 3, 64, 64) * 0.5 + 0.3
    model = UPIQAL(
        pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3,
        score_mode="nll",
    )
    model.eval()
    out = model(ref, ref.clone())
    score = float(out["score"].item())
    # All dissimilarity signals are ~0 on identical inputs -> NLL ~ 0 ->
    # exp(0) == 1.  Allow a tiny tolerance for the Sinkhorn residual.
    assert score > 0.99, f"NLL score on identical inputs was {score:.4f}"


@patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
def test_sigmoid_score_identical_inputs_is_legacy(_mock):
    torch.manual_seed(0)
    ref = torch.rand(1, 3, 64, 64) * 0.5 + 0.3
    model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
    model.eval()
    out = model(ref, ref.clone())
    score = float(out["score"].item())
    # Legacy behaviour: calibrated ~0.9526 on identity (tolerate mock VGG).
    assert 0.9 < score < 0.98, f"sigmoid score on identical inputs was {score:.4f}"


def test_invalid_score_mode_rejected():
    with pytest.raises(ValueError):
        UPIQAL(pretrained_vgg=False, score_mode="mystery")


# ---------------------------------------------------------------------------
# Item 4: multi-scale pyramid
# ---------------------------------------------------------------------------
@patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
def test_multiscale_pyramid_shapes(_mock):
    torch.manual_seed(0)
    ref_full = torch.rand(1, 3, 128, 128)
    tgt_full = torch.rand(1, 3, 128, 128)
    ref = F.interpolate(ref_full, size=(64, 64), mode="bilinear", align_corners=False)
    tgt = F.interpolate(tgt_full, size=(64, 64), mode="bilinear", align_corners=False)
    model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
    model.eval()
    out = model(ref, tgt, ref_full=ref_full, tgt_full=tgt_full)
    # Diagnostic tensor lives at the feature-level (smaller) resolution.
    assert out["diagnostic_tensor"].shape == (1, 7, 64, 64)
    assert out["score"].shape == (1,)
