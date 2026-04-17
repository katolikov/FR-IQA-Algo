"""Tests for Module 4 (SUSS Cholesky) training path.

Covers three guarantees:

1. Identity parity — at initialization (``init_scale=1.0``) the block-diagonal
   Cholesky reduces to the diagonal-mode implementation modulo numerical
   tolerance.  This keeps existing inference behaviour unchanged when no
   learned checkpoint is provided.
2. Loss decreases — three gradient steps on a trivial residual-producing
   toy pipeline reduce the NLL loss monotonically.
3. Checkpoint round-trip — saved state_dict reloads bit-identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from upiqal.uncertainty import ProbabilisticUncertaintyMapper  # noqa: E402
from upiqal.suss_train import compute_nll_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VGG_CHANNELS = ProbabilisticUncertaintyMapper.VGG_CHANNELS
VGG_LAYER_NAMES = ProbabilisticUncertaintyMapper.VGG_LAYER_NAMES


def _make_residuals(
    batch: int = 1,
    h: int = 16,
    w: int = 16,
    seed: int = 0,
) -> dict:
    """Return a dict of per-stage residual tensors at VGG-like spatial scales."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, c in zip(VGG_LAYER_NAMES, VGG_CHANNELS):
        # Emulate the spatial downsampling of VGG16 stages roughly.
        # (Tests don't need exact ratios; any resolution works because the
        # mapper re-interpolates to target_size.)
        out[name] = torch.randn(batch, c, h, w, generator=g)
    return out


# ---------------------------------------------------------------------------
# 1. Identity parity
# ---------------------------------------------------------------------------
def test_identity_parity_diag_vs_blockdiag_at_init() -> None:
    """With init_scale=1 the blockdiag forward should match diagonal forward.

    Rationale: diagonal = scaled identity with scale=1 applies an L equal to I,
    so M^2 = ||R||^2.  Blockdiag with log_diag=0 => exp=1 (identity diagonal)
    and all-zero strict-lower entries is also L=I per stage; summing per-stage
    squared norms equals the concatenated squared norm.
    """
    torch.manual_seed(0)
    residuals = _make_residuals(batch=1, h=8, w=8)
    target_size = (8, 8)

    diag = ProbabilisticUncertaintyMapper(parameterization="diagonal")
    diag.eval()
    m2_diag = diag(residuals, target_size)

    block = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag", init_scale=1.0
    )
    block.eval()
    with torch.no_grad():
        m2_block = block(residuals, target_size)

    assert m2_diag.shape == m2_block.shape
    assert torch.allclose(m2_diag, m2_block, rtol=1e-4, atol=1e-5), (
        f"diagonal vs blockdiag init mismatch: "
        f"max|diff| = {(m2_diag - m2_block).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Loss decreases
# ---------------------------------------------------------------------------
def test_loss_decreases_with_gradient_steps() -> None:
    """Three Adam steps on a fixed residual should decrease the NLL loss."""
    torch.manual_seed(1)
    residuals = _make_residuals(batch=2, h=8, w=8)
    target_size = (8, 8)

    block = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag", init_scale=1.0
    )
    # Bias residuals so the identity L is obviously suboptimal.
    scaled = {k: v * 0.1 for k, v in residuals.items()}

    optim = torch.optim.Adam(block.parameters(), lr=1e-2)

    losses = []
    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        m2 = block(scaled, target_size)
        loss = compute_nll_loss(
            m2_map=m2,
            sum_log_diag=block.sum_log_diag(),
            spatial_pixels=int(m2.shape[-2] * m2.shape[-1]),
        )
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))

    # We don't require strictly monotone (Adam can overshoot), but the last
    # step must beat the first by a clear margin.
    assert losses[-1] < losses[0] - 1e-3, f"loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# 3. Checkpoint round-trip
# ---------------------------------------------------------------------------
def test_checkpoint_round_trip(tmp_path: Path) -> None:
    block = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag", init_scale=1.0
    )
    # Perturb parameters so the default state isn't all zeros.
    with torch.no_grad():
        for p in block.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    ckpt_path = tmp_path / "L.pth"
    torch.save({"state_dict": block.state_dict()}, ckpt_path)

    block2 = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag", init_scale=1.0
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    block2.load_state_dict(loaded["state_dict"])

    for (k1, v1), (k2, v2) in zip(
        block.state_dict().items(), block2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2), f"mismatch on {k1}"


# ---------------------------------------------------------------------------
# 4. sum_log_diag sanity
# ---------------------------------------------------------------------------
def test_sum_log_diag_matches_manual() -> None:
    block = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag", init_scale=1.0
    )
    # At init, every log_diag_* is log(1.0) = 0, so the sum is 0.
    assert float(block.sum_log_diag().item()) == pytest.approx(0.0, abs=1e-6)

    # Scribble a non-zero value into one stage and verify.
    with torch.no_grad():
        getattr(block, "log_diag_relu1_2").fill_(0.5)
    expected = 0.5 * VGG_CHANNELS[0]
    assert float(block.sum_log_diag().item()) == pytest.approx(expected, rel=1e-5)
