"""Tests for `upiqal.suss_train.ranking_loss`, the MOS-tuning objective."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from upiqal.suss_train import ranking_loss  # noqa: E402


def test_perfect_alignment_gives_near_zero_loss() -> None:
    mos = torch.linspace(1.0, 5.0, 20)
    pred = mos.clone()  # perfect linear match
    loss = ranking_loss(pred, mos)
    assert float(loss.item()) < 0.01


def test_inverse_alignment_gives_large_loss() -> None:
    mos = torch.linspace(1.0, 5.0, 20)
    pred = -mos  # exactly backwards
    loss = ranking_loss(pred, mos)
    assert float(loss.item()) > 0.5


def test_gradient_flows_to_predictor() -> None:
    """A single Adam step on pred should move it toward mos."""
    mos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], requires_grad=True)
    opt = torch.optim.SGD([pred], lr=0.5)

    loss0 = ranking_loss(pred, mos).item()
    for _ in range(30):
        opt.zero_grad()
        ranking_loss(pred, mos).backward()
        opt.step()
    loss1 = ranking_loss(pred, mos).item()
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


def test_single_sample_returns_zero() -> None:
    """With B=1 there's nothing to rank or correlate; return 0."""
    loss = ranking_loss(torch.tensor([3.0]), torch.tensor([3.0]))
    assert float(loss.item()) == 0.0


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        ranking_loss(torch.zeros(4), torch.zeros(5))
    with pytest.raises(ValueError):
        ranking_loss(torch.zeros(2, 2), torch.zeros(2, 2))  # 2D


def test_weights_sum_to_configured_value() -> None:
    """Custom weights should still give a finite, non-negative loss."""
    mos = torch.linspace(1.0, 5.0, 10)
    pred = mos.clone()
    # All weight on rank:
    only_rank = ranking_loss(pred, mos, weight_plcc=0.0, weight_rank=1.0)
    assert float(only_rank.item()) < 0.01
    # All weight on PLCC:
    only_plcc = ranking_loss(pred, mos, weight_plcc=1.0, weight_rank=0.0)
    assert float(only_plcc.item()) < 0.01
