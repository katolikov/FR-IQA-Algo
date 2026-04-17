"""Module 4: Probabilistic Uncertainty Mapper.

Implements the SUSS-inspired spatial Mahalanobis distance calculation using
a Cholesky-parameterized precision matrix.  Two parameterizations are
supported:

* ``"diagonal"``: a single scaled-identity / per-channel diagonal factor
  applied to residuals concatenated across all VGG stages.  This is the
  legacy inference-only mode and produces bit-identical output to the
  previous implementation when ``init_scale=1.0``.

* ``"blockdiag"``: a block-diagonal Cholesky with one lower-triangular
  factor ``L_k`` per VGG stage ``k``.  Each ``L_k`` has ``C_k * (C_k+1) / 2``
  learnable parameters.  Residuals are left at their native spatial
  resolution for the per-stage matrix multiply, and the resulting scalar
  maps are bilinearly upsampled and summed.  This is the mode used for
  the SUSS self-supervised training (see ``upiqal/suss_train.py``).

The forward pass returns the squared Mahalanobis distance map
:math:`\\mathcal{M}^2(x, y) = \\| L^T R(x, y) \\|_2^2` as a
``(B, 1, H, W)`` tensor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Valid parameterization identifiers for the Cholesky factor.
_PARAMETERIZATIONS = ("diagonal", "blockdiag")

# Clamp range for log-diagonal entries to keep exp(log_diag) numerically sane
# during training.  exp(-5) ~ 6.7e-3, exp(5) ~ 148.
_LOG_DIAG_MIN = -5.0
_LOG_DIAG_MAX = 5.0


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
    feature_dim : int, optional
        Total number of channels across all VGG layers that will be
        concatenated (default 1472 = 64+128+256+512+512 for VGG16 relu
        layers).  Used only by the ``"diagonal"`` mode.
    init_scale : float
        Scale for the initial diagonal of the Cholesky factor (default 1.0).
        With ``init_scale=1.0`` the block-diagonal mode at initialization
        is mathematically equivalent to the diagonal mode and reduces to
        a plain sum of per-channel squared residuals.
    parameterization : {"diagonal", "blockdiag"}
        Structure of the Cholesky factor.  Defaults to ``"diagonal"`` for
        backwards compatibility; use ``"blockdiag"`` for trained / trainable
        factors.
    """

    # Channel counts for the five VGG16 stages we hook into
    VGG_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512, 512)
    VGG_LAYER_NAMES: Tuple[str, ...] = (
        "relu1_2",
        "relu2_2",
        "relu3_3",
        "relu4_3",
        "relu5_3",
    )

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        init_scale: float = 1.0,
        parameterization: str = "diagonal",
    ) -> None:
        super().__init__()
        if parameterization not in _PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization must be one of {_PARAMETERIZATIONS}; "
                f"got {parameterization!r}"
            )
        self.parameterization = parameterization

        if feature_dim is None:
            feature_dim = sum(self.VGG_CHANNELS)  # 1472
        self.feature_dim = feature_dim

        if parameterization == "diagonal":
            # Legacy inference-only factor: a scaled identity diagonal.
            self.register_buffer(
                "L_diag",
                torch.ones(feature_dim) * init_scale,
            )
        else:  # blockdiag
            # One lower-triangular Cholesky factor per VGG stage.
            # Parameterized as (strict lower tril entries) + (log diagonal)
            # so the diagonal of L is exp(log_diag) > 0, which keeps the
            # precision matrix L L^T positive definite by construction.
            init_log_diag_value = float(torch.log(torch.tensor(init_scale)))
            for k, (name, c) in enumerate(
                zip(self.VGG_LAYER_NAMES, self.VGG_CHANNELS)
            ):
                # Strictly below-diagonal entries, flattened.
                n_tril = c * (c - 1) // 2
                self.register_parameter(
                    f"tril_{name}",
                    nn.Parameter(torch.zeros(n_tril)),
                )
                # Per-channel log-diagonal.
                self.register_parameter(
                    f"log_diag_{name}",
                    nn.Parameter(
                        torch.full((c,), init_log_diag_value)
                    ),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_L(self, layer_name: str, channels: int) -> torch.Tensor:
        """Reconstruct the lower-triangular Cholesky factor for one stage."""
        tril = getattr(self, f"tril_{layer_name}")
        log_diag = getattr(self, f"log_diag_{layer_name}")
        log_diag = log_diag.clamp(min=_LOG_DIAG_MIN, max=_LOG_DIAG_MAX)

        L = tril.new_zeros(channels, channels)
        tril_idx = torch.tril_indices(channels, channels, offset=-1)
        L[tril_idx[0], tril_idx[1]] = tril
        # Diagonal from exp(log_diag) -> guaranteed positive.
        diag_idx = torch.arange(channels, device=L.device)
        L[diag_idx, diag_idx] = torch.exp(log_diag)
        return L

    def sum_log_diag(self) -> torch.Tensor:
        """Sum of log-diagonal entries across all stages.

        Equals ``log|det L|`` for the block-diagonal Cholesky factor.  Used
        by the NLL loss during SUSS training.
        """
        if self.parameterization != "blockdiag":
            # Diagonal mode has no learnable diagonal; return 0 so the NLL
            # loss short-circuits to plain Mahalanobis.
            return torch.zeros((), device=self._any_param_or_buffer().device)
        total = None
        for name in self.VGG_LAYER_NAMES:
            log_diag = getattr(self, f"log_diag_{name}").clamp(
                min=_LOG_DIAG_MIN, max=_LOG_DIAG_MAX
            )
            s = log_diag.sum()
            total = s if total is None else total + s
        assert total is not None
        return total

    def _any_param_or_buffer(self) -> torch.Tensor:
        """Return an arbitrary tensor owned by the module (for device info)."""
        for p in self.parameters():
            return p
        for b in self.buffers():
            return b
        # Module has no tensors; fall back to CPU
        return torch.zeros(())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        residuals: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Compute the spatial Mahalanobis distance map.

        Parameters
        ----------
        residuals : dict[str, torch.Tensor]
            Per-layer residual feature maps
            ``{layer_name: (B, C_k, H_k, W_k)}``.
        target_size : tuple[int, int]
            ``(H, W)`` to which the per-pixel Mahalanobis maps are resized
            before aggregation.

        Returns
        -------
        torch.Tensor
            Global anomaly map ``(B, 1, H, W)`` containing
            :math:`\\mathcal{M}^2(x,y)` at every spatial location.
        """
        if self.parameterization == "diagonal":
            return self._forward_diagonal(residuals, target_size)
        return self._forward_blockdiag(residuals, target_size)

    # ---- diagonal (legacy) -------------------------------------------
    def _forward_diagonal(
        self,
        residuals: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        aligned: List[torch.Tensor] = []
        for name in sorted(residuals.keys()):
            r = residuals[name]
            if r.shape[2:] != target_size:
                r = F.interpolate(
                    r, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(r)
        R = torch.cat(aligned, dim=1)
        _, D, _, _ = R.shape
        L_diag = self.L_diag[:D].view(1, D, 1, 1)
        LR = L_diag * R
        M2 = (LR * LR).sum(dim=1, keepdim=True)
        return M2

    # ---- block-diagonal ----------------------------------------------
    def _forward_blockdiag(
        self,
        residuals: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        H, W = target_size
        total: Optional[torch.Tensor] = None
        for name, c in zip(self.VGG_LAYER_NAMES, self.VGG_CHANNELS):
            if name not in residuals:
                # Skip any stage that wasn't provided (defensive).
                continue
            R_k = residuals[name]  # (B, C_k, H_k, W_k)
            if R_k.shape[1] != c:
                raise ValueError(
                    f"Residual for layer {name!r} has {R_k.shape[1]} channels, "
                    f"expected {c}."
                )
            L_k = self._build_L(name, c)  # (C_k, C_k), lower-triangular
            # L_k^T R_k  -> contract over channel dim at native resolution.
            # einsum: 'ij,bjhw->bihw' with A = L_k^T, i.e. A[i,j] = L_k[j,i]
            LR = torch.einsum("ji,bjhw->bihw", L_k, R_k)
            M2_k = (LR * LR).sum(dim=1, keepdim=True)  # (B, 1, H_k, W_k)
            if M2_k.shape[-2:] != (H, W):
                M2_k = F.interpolate(
                    M2_k, size=target_size, mode="bilinear", align_corners=False
                )
            total = M2_k if total is None else total + M2_k
        if total is None:
            # No residuals supplied: return an all-zero map.  We fall back
            # to a tiny tensor since we don't know the batch size.
            return torch.zeros((1, 1, H, W))
        return total
