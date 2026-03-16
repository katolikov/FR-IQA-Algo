"""Module 1: Universal Preprocessing and Normalization.

Implements Minmax scaling and piece-wise linear histogram matching to
guarantee robust generalization across disparate imaging domains (natural
photographic, MRI, endoscopic, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Normalizer(nn.Module):
    """Universal preprocessing and normalization module.

    Applies either standard ImageNet normalization or piece-wise linear
    histogram matching depending on the ``mode`` parameter.

    Parameters
    ----------
    mode : str
        ``"imagenet"`` for standard ImageNet mean/std normalization (default),
        ``"histogram"`` for piece-wise linear histogram matching.
    low_pct : float
        Lower percentile for histogram matching (default 2.0).
    high_pct : float
        Upper percentile for histogram matching (default 98.0).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        mode: str = "imagenet",
        low_pct: float = 2.0,
        high_pct: float = 98.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.low_pct = low_pct
        self.high_pct = high_pct

        # Register ImageNet constants as buffers (move with .to(device))
        self.register_buffer(
            "mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize a reference / target pair.

        Parameters
        ----------
        ref : torch.Tensor
            Reference image, shape ``(B, C, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Target (distorted) image, same shape as *ref*.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Normalized ``(ref, tgt)`` tensors scaled to ``[-1, 1]`` range
            (ImageNet mode) or matched to reference histogram.
        """
        ref = self._minmax_scale(ref)
        tgt = self._minmax_scale(tgt)

        if self.mode == "imagenet":
            ref = (ref - self.mean) / self.std
            tgt = (tgt - self.mean) / self.std
        elif self.mode == "histogram":
            tgt = self._piecewise_histogram_match(ref, tgt)
            # After matching, apply ImageNet normalization
            ref = (ref - self.mean) / self.std
            tgt = (tgt - self.mean) / self.std
        return ref, tgt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax_scale(x: torch.Tensor) -> torch.Tensor:
        """Scale tensor to ``[0, 1]`` per-sample.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Scaled tensor with values in ``[0, 1]``.
        """
        B = x.shape[0]
        flat = x.view(B, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        denom = (hi - lo).clamp(min=1e-8)
        flat_scaled = (flat - lo) / denom
        return flat_scaled.view_as(x)

    def _piecewise_histogram_match(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Piece-wise linear histogram matching per channel.

        Aligns the 2% and 98% percentiles of *tgt* to those of *ref* using
        a simple linear rescale per channel, neutralizing systemic intensity
        shifts before perceptual comparison.

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, C, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Target image ``(B, C, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Histogram-matched target.
        """
        B, C, H, W = ref.shape
        ref_flat = ref.view(B, C, -1)
        tgt_flat = tgt.view(B, C, -1)

        lo_q = self.low_pct / 100.0
        hi_q = self.high_pct / 100.0

        ref_lo = torch.quantile(ref_flat.float(), lo_q, dim=2, keepdim=True)
        ref_hi = torch.quantile(ref_flat.float(), hi_q, dim=2, keepdim=True)
        tgt_lo = torch.quantile(tgt_flat.float(), lo_q, dim=2, keepdim=True)
        tgt_hi = torch.quantile(tgt_flat.float(), hi_q, dim=2, keepdim=True)

        scale = (ref_hi - ref_lo) / (tgt_hi - tgt_lo).clamp(min=1e-8)
        matched = (tgt_flat - tgt_lo) * scale + ref_lo
        return matched.view(B, C, H, W).clamp(0.0, 1.0)
