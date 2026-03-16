"""Module 4: Probabilistic Uncertainty Mapper.

Implements the SUSS-inspired spatial Mahalanobis distance calculation using
a Cholesky-parameterized precision matrix.  The lower-triangular factor L is
a placeholder (identity) for inference; the actual L would be learned during
an offline self-supervised generative training phase.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticUncertaintyMapper(nn.Module):
    """Compute the spatial Mahalanobis distance anomaly map.

    For every spatial location ``(x, y)`` the residual feature vector

    .. math::

        R(x, y) = \\Phi_r(x, y) - \\Phi_t(x, y)

    is projected through the Cholesky factor ``L`` of the precision matrix:

    .. math::

        \\mathcal{M}^2(x, y) = \\| L^T R(x, y) \\|_2^2

    Parameters
    ----------
    feature_dim : int
        Total number of channels across all VGG layers that will be
        concatenated (default 1472 = 64+128+256+512+512 for VGG16 relu layers).
    init_scale : float
        Scale for the diagonal of the placeholder Cholesky factor (default 1.0).
        In production this would be replaced by the learned factor.
    """

    # Channel counts for the five VGG16 stages we hook into
    VGG_CHANNELS = (64, 128, 256, 512, 512)

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim is None:
            feature_dim = sum(self.VGG_CHANNELS)  # 1472

        self.feature_dim = feature_dim

        # Placeholder Cholesky factor: scaled identity (diagonal)
        # In production this would be a learned sparse lower-triangular matrix.
        # We store only the diagonal for efficiency.
        self.register_buffer(
            "L_diag",
            torch.ones(feature_dim) * init_scale,
        )

    def forward(
        self,
        residuals: Dict[str, torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute the spatial Mahalanobis distance map.

        Parameters
        ----------
        residuals : dict[str, torch.Tensor]
            Per-layer residual feature maps ``{layer_name: (B, C_k, H_k, W_k)}``.
        target_size : tuple[int, int]
            ``(H, W)`` to which all residual maps are resized before
            concatenation.

        Returns
        -------
        torch.Tensor
            Global anomaly map ``(B, 1, H, W)`` containing
            :math:`\\mathcal{M}^2(x,y)` at every spatial location.
        """
        # Upsample all residual maps to target_size and concatenate
        aligned: list[torch.Tensor] = []
        for name in sorted(residuals.keys()):
            r = residuals[name]
            if r.shape[2:] != target_size:
                r = F.interpolate(
                    r, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(r)

        # R: (B, D, H, W) where D = sum of all channel dims
        R = torch.cat(aligned, dim=1)
        B, D, H, W = R.shape

        # Apply Cholesky factor: L^T R
        # With diagonal L this is simply element-wise scaling per channel
        L_diag = self.L_diag[:D].view(1, D, 1, 1)
        LR = L_diag * R  # (B, D, H, W)

        # Mahalanobis distance squared: ||L^T R||_2^2 along channel dim
        M2 = (LR * LR).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return M2
