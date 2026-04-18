"""Self-supervised training utilities for Module 4 (SUSS Cholesky factor).

This module is inference-free: it only defines the pieces needed to train
the Cholesky factor L of ``upiqal.uncertainty.ProbabilisticUncertaintyMapper``
via a maximum-likelihood objective on imperceptibly-augmented image pairs.

Objective
---------
For a residual ``R = phi(x) - phi(x_tilde)`` drawn from
``N(0, Sigma)`` with ``Sigma^{-1} = L L^T``, the negative log-likelihood
(up to a constant) is

    NLL(R) = 0.5 * ||L^T R||_2^2  -  log|det L|
           = 0.5 * M2(R)          -  sum_k sum_c log_diag_k[c]

This is averaged over the batch and spatial dimensions.  The ``-log|det L|``
term is what prevents the degenerate solution ``L -> 0``.

Augmentations
-------------
`ImperceptibleAugment` applies one of a set of perceptually-minor
perturbations to produce ``x_tilde`` from ``x``:

    * identity (no change)
    * additive Gaussian noise, sigma in [0.5, 3.0] / 255
    * Gaussian blur, sigma in [0.0, 0.6] px
    * small translation, +/- 1 pixel
    * brightness/contrast jitter, +/- 3%
    * light JPEG re-encode, quality in [85, 100]  (optional; requires PIL)
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PIL is an optional dependency for the JPEG augmentation.
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - PIL always present in our env
    _PIL_AVAILABLE = False


# --------------------------------------------------------------------------
# Augmentations
# --------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Ranges for the imperceptible perturbations."""

    noise_sigma_range: Tuple[float, float] = (0.5 / 255.0, 3.0 / 255.0)
    blur_sigma_range: Tuple[float, float] = (0.0, 0.6)
    translation_max_px: int = 1
    brightness_range: Tuple[float, float] = (-0.03, 0.03)
    contrast_range: Tuple[float, float] = (-0.03, 0.03)
    jpeg_quality_range: Tuple[int, int] = (85, 100)
    # Relative probabilities (normalized internally).
    weights: Tuple[float, ...] = (
        1.0,  # identity
        1.5,  # noise
        1.0,  # blur
        1.0,  # translation
        1.0,  # brightness/contrast
        1.0,  # jpeg
    )


class ImperceptibleAugment:
    """Stateless sampler that returns an augmented copy of an image tensor.

    Input is expected as a ``(B, 3, H, W)`` tensor with values in ``[0, 1]``.
    Augmentations are applied independently per batch element.
    """

    _KINDS = ("identity", "noise", "blur", "translate", "bc", "jpeg")

    def __init__(self, config: Optional[AugmentConfig] = None) -> None:
        self.config = config or AugmentConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B,3,H,W) tensor; got shape {x.shape}")
        kinds = self._KINDS
        weights = list(self.config.weights)
        if not _PIL_AVAILABLE:
            # Drop the JPEG kind if PIL isn't available.
            kinds = tuple(k for k in kinds if k != "jpeg")
            weights = weights[: len(kinds)]
        B = x.shape[0]
        out = torch.empty_like(x)
        for b in range(B):
            kind = random.choices(kinds, weights=weights, k=1)[0]
            out[b] = self._apply(x[b], kind)
        return out.clamp(0.0, 1.0)

    # --------------------------------------------------------------
    def _apply(self, img: torch.Tensor, kind: str) -> torch.Tensor:
        cfg = self.config
        if kind == "identity":
            return img
        if kind == "noise":
            lo, hi = cfg.noise_sigma_range
            sigma = random.uniform(lo, hi)
            return img + torch.randn_like(img) * sigma
        if kind == "blur":
            lo, hi = cfg.blur_sigma_range
            sigma = random.uniform(lo, hi)
            return _gaussian_blur(img, sigma)
        if kind == "translate":
            r = cfg.translation_max_px
            if r <= 0:
                return img
            dx = random.randint(-r, r)
            dy = random.randint(-r, r)
            return _roll_translate(img, dx, dy)
        if kind == "bc":
            blo, bhi = cfg.brightness_range
            clo, chi = cfg.contrast_range
            delta = random.uniform(blo, bhi)
            gamma = 1.0 + random.uniform(clo, chi)
            mean = img.mean(dim=(-2, -1), keepdim=True)
            return (img - mean) * gamma + mean + delta
        if kind == "jpeg":
            qlo, qhi = cfg.jpeg_quality_range
            q = random.randint(qlo, qhi)
            return _jpeg_reencode(img, q)
        raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------
# Helper tensor ops used by augmentations
# --------------------------------------------------------------------------
def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """1D-separable Gaussian blur for a (3, H, W) tensor."""
    if sigma <= 1e-6:
        return img
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(
        -radius, radius + 1, dtype=img.dtype, device=img.device
    )
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    C = img.shape[0]
    # (C, 1, 1, K)
    kh = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kv = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = img.unsqueeze(0)
    x = F.conv2d(x, kh, padding=(0, radius), groups=C)
    x = F.conv2d(x, kv, padding=(radius, 0), groups=C)
    return x.squeeze(0)


def _roll_translate(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate with edge-replicate padding via roll + edge fix-up."""
    if dx == 0 and dy == 0:
        return img
    out = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))
    # Replicate the exposed edges so we don't inject wrap-around content.
    if dy > 0:
        out[..., :dy, :] = img[..., :1, :]
    elif dy < 0:
        out[..., dy:, :] = img[..., -1:, :]
    if dx > 0:
        out[..., :, :dx] = img[..., :, :1]
    elif dx < 0:
        out[..., :, dx:] = img[..., :, -1:]
    return out


def _jpeg_reencode(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip through JPEG at the given quality.  Requires PIL."""
    if not _PIL_AVAILABLE:  # pragma: no cover
        return img
    arr = (img.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    arr = arr.permute(1, 2, 0).contiguous().numpy()  # HWC
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    pil2 = Image.open(buf).convert("RGB")
    out = torch.from_numpy(
        _pil_to_array(pil2).copy()
    ).to(img.dtype).to(img.device) / 255.0
    return out.permute(2, 0, 1).contiguous()


def _pil_to_array(pil):  # small indirection so tests can stub PIL
    import numpy as np  # local import to keep top of file clean

    return np.asarray(pil)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def ranking_loss(
    pred: torch.Tensor,
    mos: torch.Tensor,
    weight_plcc: float = 0.5,
    weight_rank: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable ranking loss for MOS-correlation tuning.

    Combines two standard IQA surrogates:

    * ``1 - PLCC``: linear Pearson correlation between centered
      predictions and MOS.  Pushes the predictor into linear alignment
      with MOS.
    * Margin ranking: for every pair (i, j) with distinct MOS, penalise
      the case where the predictor's ordering disagrees with MOS via
      ``relu(-(mos_i - mos_j) * (pred_i - pred_j))``.  This is a
      differentiable proxy for SROCC — zero when every pair is ordered
      correctly.

    The convex combination yields a stable loss surface whose minimum
    coincides with perfect SROCC and perfect PLCC simultaneously.

    ``pred`` and ``mos`` must already point the same way (higher pred
    ⇔ higher mos).  Flip the sign of whichever side is needed before
    calling (UPIQAL score is "higher better", KADID DMOS is
    "higher worse").
    """
    if pred.shape != mos.shape or pred.ndim != 1:
        raise ValueError(
            f"pred and mos must both be 1D of equal length; "
            f"got {pred.shape} vs {mos.shape}"
        )
    if pred.shape[0] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    pred_c = pred - pred.mean()
    mos_c = mos - mos.mean()
    num = (pred_c * mos_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (mos_c ** 2).sum() + eps)
    plcc = num / den
    loss_plcc = 1.0 - plcc

    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    mos_diff = mos.unsqueeze(0) - mos.unsqueeze(1)
    loss_rank = torch.relu(-mos_diff * pred_diff).mean()

    return weight_plcc * loss_plcc + weight_rank * loss_rank


def compute_nll_loss(
    m2_map: torch.Tensor,
    sum_log_diag: torch.Tensor,
    spatial_pixels: int,
) -> torch.Tensor:
    """Negative log-likelihood under ``R ~ N(0, (L L^T)^{-1})``.

    Parameters
    ----------
    m2_map : torch.Tensor
        Per-pixel squared Mahalanobis distances, shape ``(B, 1, H, W)``.
        Produced by ``ProbabilisticUncertaintyMapper.forward``.
    sum_log_diag : torch.Tensor
        Scalar ``log|det L|`` for the block-diagonal Cholesky factor.  For
        the ``"diagonal"`` parameterization this is zero (no learnable
        diagonal) and the loss reduces to ``0.5 * mean(M^2)``.
    spatial_pixels : int
        Number of "samples" per batch element for the log-det term.  The
        NLL is summed over spatial positions in its full form, but we
        average M^2 over space to keep the optimization well-scaled; we
        therefore subtract ``sum_log_diag`` once per image to give it
        commensurate weight rather than once per pixel.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.  Lower is better.
    """
    del spatial_pixels  # kept for API clarity; current impl averages over space
    mahalanobis_term = 0.5 * m2_map.mean()
    return mahalanobis_term - sum_log_diag


# --------------------------------------------------------------------------
# Single training step
# --------------------------------------------------------------------------
def training_step(
    *,
    ref: torch.Tensor,
    normalizer: nn.Module,
    deep_stats: nn.Module,
    uncertainty: nn.Module,
    augment: ImperceptibleAugment,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run one forward pass and return the scalar loss (no optimizer step).

    Parameters
    ----------
    ref : torch.Tensor
        Reference images, shape ``(B, 3, H, W)`` in ``[0, 1]``.
    normalizer, deep_stats, uncertainty : nn.Module
        UPIQAL sub-modules.  Only ``uncertainty`` will have
        ``requires_grad=True`` parameters.
    augment : ImperceptibleAugment
        Sampler for the target image ``x_tilde``.
    """
    if ref.ndim != 4 or ref.shape[1] != 3:
        raise ValueError(f"expected (B,3,H,W) ref; got {tuple(ref.shape)}")

    tgt = augment(ref)

    ref_n, tgt_n = normalizer(ref, tgt)

    # Deep features / residuals.  VGG parameters are frozen inside
    # DeepStatisticalExtractor (requires_grad=False), so autograd only
    # flows back to the Cholesky parameters.
    deep_out = deep_stats(ref_n, tgt_n)
    residuals = deep_out["residuals"]

    H, W = ref.shape[-2:]
    m2 = uncertainty(residuals, target_size=(H, W))  # (B, 1, H, W)

    sum_log_diag = uncertainty.sum_log_diag()
    spatial = int(m2.shape[-2] * m2.shape[-1])
    loss = compute_nll_loss(m2, sum_log_diag, spatial)

    info = {
        "mahalanobis_mean": m2.detach().mean(),
        "sum_log_diag": sum_log_diag.detach(),
        "loss": loss.detach(),
    }
    return loss, info


# --------------------------------------------------------------------------
# Simple epoch runner (used by train_uncertainty.py)
# --------------------------------------------------------------------------
def one_epoch(
    *,
    batches,  # iterable of (B,3,H,W) tensors already on device
    normalizer: nn.Module,
    deep_stats: nn.Module,
    uncertainty: nn.Module,
    augment: ImperceptibleAugment,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = 1.0,
) -> List[float]:
    """Run one epoch; return the list of per-step loss values."""
    uncertainty.train()
    losses: List[float] = []
    for ref in batches:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_step(
            ref=ref,
            normalizer=normalizer,
            deep_stats=deep_stats,
            uncertainty=uncertainty,
            augment=augment,
        )
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(uncertainty.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return losses
