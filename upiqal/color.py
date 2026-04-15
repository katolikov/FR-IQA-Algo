"""Module 2: Chromatic Transport Evaluator.

Implements differentiable sRGB -> Oklab conversion and Sinkhorn-Knopp
approximation of the Earth Mover's Distance (EMD) for 3D color histograms.
Operates entirely on GPU tensors to avoid CPU bottlenecks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChromaticTransportEvaluator(nn.Module):
    """Evaluates chromatic degradation via optimal transport in Oklab space.

    Parameters
    ----------
    patch_size : int
        Spatial patch size for local color histogram extraction (default 16).
    n_bins : int
        Number of bins per Oklab channel for the 3D histogram (default 8).
    sinkhorn_iters : int
        Number of Sinkhorn-Knopp iterations (default 20).
    sinkhorn_reg : float
        Entropy regularization coefficient gamma (default 0.1).
    """

    def __init__(
        self,
        patch_size: int = 16,
        n_bins: int = 8,
        sinkhorn_iters: int = 20,
        sinkhorn_reg: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_reg = sinkhorn_reg

    # ------------------------------------------------------------------
    # sRGB -> linear RGB -> Oklab   (fully differentiable)
    # ------------------------------------------------------------------

    @staticmethod
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        """Convert sRGB [0,1] to linear RGB via the inverse transfer function.

        Parameters
        ----------
        x : torch.Tensor
            sRGB image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Linear-light RGB in ``[0, 1]``.
        """
        return torch.where(
            x <= 0.04045,
            x / 12.92,
            ((x + 0.055) / 1.055).clamp(min=0.0) ** 2.4,
        )

    @staticmethod
    def linear_rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear RGB to Oklab (L, a, b).

        Uses the exact Oklab formulation by Bjorn Ottosson:
        linear RGB -> LMS (via M1) -> cube-root -> Lab (via M2).

        Parameters
        ----------
        rgb : torch.Tensor
            Linear RGB ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Oklab ``(B, 3, H, W)`` with L in ~[0,1], a/b in ~[-0.5, 0.5].
        """
        # M1: linear sRGB -> approximate LMS
        M1 = torch.tensor(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ],
            dtype=rgb.dtype,
            device=rgb.device,
        )
        # M2: cube-root LMS -> Oklab
        M2 = torch.tensor(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ],
            dtype=rgb.dtype,
            device=rgb.device,
        )

        B, C, H, W = rgb.shape
        pixels = rgb.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)
        lms = (pixels @ M1.T).clamp(min=0.0)
        lms_g = torch.sign(lms) * torch.abs(lms).clamp(min=1e-12).pow(1.0 / 3.0)
        lab = lms_g @ M2.T
        return lab.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)

    def srgb_to_oklab(self, x: torch.Tensor) -> torch.Tensor:
        """Full sRGB -> Oklab pipeline.

        Parameters
        ----------
        x : torch.Tensor
            sRGB image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Oklab tensor ``(B, 3, H, W)``.
        """
        return self.linear_rgb_to_oklab(self.srgb_to_linear(x.clamp(0.0, 1.0)))

    # ------------------------------------------------------------------
    # Sinkhorn-Knopp EMD approximation
    # ------------------------------------------------------------------

    def _sinkhorn_emd(
        self,
        hist_r: torch.Tensor,
        hist_t: torch.Tensor,
        bin_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate EMD between two histograms via Sinkhorn-Knopp.

        Parameters
        ----------
        hist_r : torch.Tensor
            Reference histogram weights ``(P, N)`` where P is number of patches,
            N is number of bins (flattened 3D histogram).
        hist_t : torch.Tensor
            Target histogram weights ``(P, N)``.
        bin_centers : torch.Tensor
            3D coordinates of bin centers ``(N, 3)`` in Oklab space.

        Returns
        -------
        torch.Tensor
            Approximate EMD per patch ``(P,)``.
        """
        eps = 1e-6
        # Identify patches with degenerate (all-zero) histograms; we'll zero
        # out their EMD contribution at the end so they can't poison the
        # Sinkhorn iteration with NaNs.
        sum_r = hist_r.sum(dim=-1, keepdim=True)
        sum_t = hist_t.sum(dim=-1, keepdim=True)
        empty_mask = (sum_r.squeeze(-1) <= eps) | (sum_t.squeeze(-1) <= eps)  # (P,)

        # Normalize histograms to probability distributions
        hist_r = hist_r / sum_r.clamp(min=eps)
        hist_t = hist_t / sum_t.clamp(min=eps)

        # Ground distance matrix: pairwise L2 in Oklab
        # (N, 3) -> (N, N)
        cost = torch.cdist(bin_centers.unsqueeze(0), bin_centers.unsqueeze(0)).squeeze(0)

        # Gibbs kernel
        K = torch.exp(-cost / self.sinkhorn_reg)  # (N, N)

        # Sinkhorn iterations  (P, N) x (N, N) -> (P, N)
        u = torch.ones_like(hist_r)
        for _ in range(self.sinkhorn_iters):
            v = hist_t / (u @ K.T + eps)
            u = hist_r / (v @ K + eps)

        # Transport plan T_ij = u_i * K_ij * v_j
        # EMD = sum_ij T_ij * C_ij  for each patch
        T = u.unsqueeze(-1) * K.unsqueeze(0) * v.unsqueeze(-2)  # (P, N, N)
        emd = (T * cost.unsqueeze(0)).sum(dim=(-1, -2))  # (P,)
        # Sanitise any NaN/Inf produced by extreme degeneracies and force
        # patches that started with empty histograms to zero EMD.
        emd = torch.nan_to_num(emd, nan=0.0, posinf=0.0, neginf=0.0)
        if empty_mask.any():
            emd = torch.where(empty_mask, torch.zeros_like(emd), emd)
        return emd

    # ------------------------------------------------------------------
    # Patch-based color histogram extraction
    # ------------------------------------------------------------------

    def _extract_histograms(
        self, oklab: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract soft 3D color histograms from non-overlapping patches.

        Parameters
        ----------
        oklab : torch.Tensor
            Oklab image ``(B, 3, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Histogram weights ``(B * Ph * Pw, N_bins^3)``.
            - Bin center coordinates ``(N_bins^3, 3)``.
        """
        B, C, H, W = oklab.shape
        ps = self.patch_size
        nb = self.n_bins

        # Pad to multiple of patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            oklab = F.pad(oklab, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = oklab.shape
        Ph, Pw = Hp // ps, Wp // ps

        # Unfold into patches: (B, 3, Ph, ps, Pw, ps)
        patches = oklab.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, Ph, Pw, 3, ps, ps)
        patches = patches.reshape(B * Ph * Pw, 3, ps * ps)  # (P, 3, N_pix)
        patches = patches.permute(0, 2, 1)  # (P, N_pix, 3)

        # Bin edges and centers
        # L in [0,1], a/b in roughly [-0.5, 0.5]; use data-agnostic range
        lo = patches.min(dim=1).values.min(dim=0).values  # (3,)
        hi = patches.max(dim=1).values.max(dim=0).values  # (3,)
        # Safety margin
        lo = lo - 0.01
        hi = hi + 0.01
        # If any channel is constant (lo == hi within tolerance) widen the
        # range so torch.linspace returns a non-degenerate set of edges.
        for c in range(3):
            if (hi[c] - lo[c]) < 1e-4:
                lo[c] = lo[c] - 0.05
                hi[c] = hi[c] + 0.05

        edges = [torch.linspace(lo[c].item(), hi[c].item(), nb + 1, device=oklab.device) for c in range(3)]
        centers = [(edges[c][:-1] + edges[c][1:]) / 2.0 for c in range(3)]

        # 3D grid of bin centers (nb^3, 3)
        grid = torch.stack(
            torch.meshgrid(centers[0], centers[1], centers[2], indexing="ij"), dim=-1
        ).reshape(-1, 3)

        # Hard-assign each pixel to closest bin -> histogram
        # (P, N_pix, 1, 3) vs (1, 1, nb^3, 3) -> (P, N_pix, nb^3)
        dists = torch.cdist(patches, grid.unsqueeze(0).expand(patches.shape[0], -1, -1))
        assignments = dists.argmin(dim=-1)  # (P, N_pix)

        # Build histograms via scatter
        n_total_bins = nb ** 3
        hist = torch.zeros(patches.shape[0], n_total_bins, device=oklab.device)
        hist.scatter_add_(1, assignments, torch.ones_like(assignments, dtype=hist.dtype))

        return hist, grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the spatial Color Degradation Map.

        Parameters
        ----------
        ref : torch.Tensor
            Reference sRGB image ``(B, 3, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Target sRGB image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Color degradation map ``(B, 1, H, W)`` (bilinear-upsampled from
            the patch grid).
        """
        B, C, H, W = ref.shape
        ps = self.patch_size

        # Convert to Oklab
        oklab_r = self.srgb_to_oklab(ref)
        oklab_t = self.srgb_to_oklab(tgt)

        # Pad dimensions
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        Hp = H + pad_h
        Wp = W + pad_w
        Ph, Pw = Hp // ps, Wp // ps

        # Histograms for both
        hist_r, grid_r = self._extract_histograms(oklab_r)
        hist_t, grid_t = self._extract_histograms(oklab_t)

        # Use shared bin centers (average)
        grid = (grid_r + grid_t) / 2.0

        # Sinkhorn EMD per patch
        emd = self._sinkhorn_emd(hist_r, hist_t, grid)  # (B*Ph*Pw,)

        # Reshape to spatial grid and upsample
        emd_map = emd.view(B, 1, Ph, Pw)
        color_map = F.interpolate(emd_map, size=(H, W), mode="bilinear", align_corners=False)
        return color_map
