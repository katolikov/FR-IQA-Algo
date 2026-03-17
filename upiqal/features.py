"""Module 3: Hierarchical Deep Statistical Extractor.

Implements the A-DISTS-inspired adaptive structure/texture separation using
VGG16 feature maps, 2D Hanning-window L2 pooling, dispersion index, and
texture probability mapping.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Default path to the local VGG16 weights file (relative to this source file).
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_DEFAULT_VGG16_WEIGHTS = _WEIGHTS_DIR / "vgg16-397923af.pth"


# ======================================================================
# Hanning-window L2 pooling
# ======================================================================


def make_hanning_kernel(size: int) -> torch.Tensor:
    """Create a normalized 2D Hanning window kernel.

    .. math::

        w(m, n) = \\frac{1}{\\Omega}
        \\bigl[1 - \\cos\\bigl(\\frac{2\\pi m}{N-1}\\bigr)\\bigr]
        \\bigl[1 - \\cos\\bigl(\\frac{2\\pi n}{N-1}\\bigr)\\bigr]

    where :math:`\\Omega` ensures :math:`\\sum w = 1`.

    Parameters
    ----------
    size : int
        Kernel spatial dimension (square).

    Returns
    -------
    torch.Tensor
        Kernel of shape ``(1, 1, size, size)`` summing to 1.
    """
    coords = torch.arange(size, dtype=torch.float32)
    w1 = 1.0 - torch.cos(2.0 * math.pi * coords / (size - 1))
    w2d = w1.unsqueeze(1) * w1.unsqueeze(0)
    w2d = w2d / w2d.sum()
    return w2d.unsqueeze(0).unsqueeze(0)


class L2HanningPool(nn.Module):
    """Localized L2 pooling with a Hanning window via grouped convolution.

    For each channel *C* of the input feature map, computes the localized
    mean (L2 norm style) using ``torch.nn.functional.conv2d`` with a fixed
    Hanning kernel.

    Parameters
    ----------
    kernel_size : int
        Spatial size of the Hanning window (default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer("_kernel", make_hanning_kernel(kernel_size))

    def _get_kernel(self, channels: int) -> torch.Tensor:
        """Expand the single-channel kernel to a grouped conv kernel."""
        return self._kernel.expand(channels, 1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hanning-windowed L2 pooling.

        .. math::

            \\mu_{C}(x,y) = \\sqrt{\\sum_{m,n} w(m,n) \\cdot
            \\Phi_C(x-m, y-n)^2 + \\epsilon}

        Parameters
        ----------
        x : torch.Tensor
            Feature map ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Localized L2 means ``(B, C, H, W)``.
        """
        C = x.shape[1]
        pad = self.kernel_size // 2
        kernel = self._get_kernel(C)
        x_sq = x * x
        mu_sq = F.conv2d(x_sq, kernel, padding=pad, groups=C)
        return torch.sqrt(mu_sq + 1e-12)


# ======================================================================
# VGG16 feature extractor with forward hooks
# ======================================================================

# Layers to intercept (ReLU outputs after named conv blocks)
_HOOK_LAYERS: Dict[str, str] = {
    "relu1_2": "features.3",
    "relu2_2": "features.8",
    "relu3_3": "features.15",
    "relu4_3": "features.22",
    "relu5_3": "features.29",
}


class VGG16FeatureExtractor(nn.Module):
    """Fixed VGG16 backbone with forward hooks at five ReLU stages.

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet-pretrained weights (default True).
    weights_path : str or Path or None
        Explicit path to a local ``.pth`` file.  When *None* (the default)
        and ``pretrained=True``, the extractor looks for the bundled file at
        ``<project_root>/weights/vgg16-397923af.pth``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        # Always create the architecture without triggering any download.
        vgg = models.vgg16(weights=None)

        if pretrained:
            wpath = Path(weights_path) if weights_path else _DEFAULT_VGG16_WEIGHTS
            if not wpath.exists():
                raise FileNotFoundError(
                    f"Local VGG16 weights not found at {wpath}.  "
                    "Run 'python weights/download_vgg16.py' to fetch them."
                )
            state = torch.load(wpath, map_location="cpu", weights_only=True)
            vgg.load_state_dict(state)
        self.features = vgg.features
        # Freeze all parameters
        for p in self.features.parameters():
            p.requires_grad_(False)

        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, layer_path in _HOOK_LAYERS.items():
            parts = layer_path.split(".")
            module = self.features
            for part in parts[1:]:
                module = module[int(part)]

            def _hook_fn(mod, inp, out, _name=name):
                self._activations[_name] = out

            self._hooks.append(module.register_forward_hook(_hook_fn))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Normalized image ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer name to activation tensor.
        """
        self._activations.clear()
        self.features(x)
        return dict(self._activations)


# ======================================================================
# Deep Statistical Module (A-DISTS inspired)
# ======================================================================

_EPS = 1e-12
_C1 = 1e-6
_C2 = 1e-6


class DeepStatisticalExtractor(nn.Module):
    """Hierarchical deep feature statistics with adaptive texture separation.

    Computes localized spatial mean, variance, cross-covariance at each VGG16
    stage.  Derives the dispersion index and texture probability mask
    :math:`P_{tex}`.

    Parameters
    ----------
    kernel_size : int
        Hanning window size for spatial pooling (default 7).
    pretrained : bool
        Load pretrained VGG16 weights (default True).
    sigmoid_gain : float
        Gain ``k`` for the sigmoid mapping :math:`P_{tex} = \\sigma(k \\cdot D)`.
    sigmoid_bias : float
        Bias ``b`` for the sigmoid mapping :math:`P_{tex} = \\sigma(k \\cdot (D - b))`.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        pretrained: bool = True,
        sigmoid_gain: float = 5.0,
        sigmoid_bias: float = 1.0,
        weights_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.vgg = VGG16FeatureExtractor(
            pretrained=pretrained, weights_path=weights_path
        )
        self.pool = L2HanningPool(kernel_size)
        self.kernel_size = kernel_size
        self.sigmoid_gain = sigmoid_gain
        self.sigmoid_bias = sigmoid_bias

        # Build Hanning kernel for mean computation (non-L2)
        self.register_buffer("_hann", make_hanning_kernel(kernel_size))

    def _windowed_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Compute localized mean via Hanning-windowed convolution."""
        C = x.shape[1]
        pad = self.kernel_size // 2
        kernel = self._hann.expand(C, 1, -1, -1)
        return F.conv2d(x, kernel, padding=pad, groups=C)

    def _windowed_variance(
        self, x: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Compute localized variance: E[X^2] - E[X]^2."""
        C = x.shape[1]
        pad = self.kernel_size // 2
        kernel = self._hann.expand(C, 1, -1, -1)
        ex2 = F.conv2d(x * x, kernel, padding=pad, groups=C)
        return (ex2 - mu * mu).clamp(min=0.0)

    def _windowed_covariance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute localized cross-covariance: E[XY] - E[X]E[Y]."""
        C = x.shape[1]
        pad = self.kernel_size // 2
        kernel = self._hann.expand(C, 1, -1, -1)
        exy = F.conv2d(x * y, kernel, padding=pad, groups=C)
        return exy - mu_x * mu_y

    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute deep statistical features and texture probability mask.

        Parameters
        ----------
        ref : torch.Tensor
            Normalized reference image ``(B, 3, H, W)``.
        tgt : torch.Tensor
            Normalized target image ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys:
            - ``"l_maps"`` : list of luminance similarity maps per layer
            - ``"s_maps"`` : list of structure/texture similarity maps per layer
            - ``"p_tex"``  : list of texture probability maps per layer
            - ``"residuals"`` : dict of residual feature maps per layer
            - ``"mu_r"``  : dict of reference means per layer
        """
        feats_r = self.vgg(ref)
        feats_t = self.vgg(tgt)

        l_maps: List[torch.Tensor] = []
        s_maps: List[torch.Tensor] = []
        p_tex_maps: List[torch.Tensor] = []
        residuals: Dict[str, torch.Tensor] = {}
        mu_r_dict: Dict[str, torch.Tensor] = {}

        for name in _HOOK_LAYERS:
            phi_r = feats_r[name]
            phi_t = feats_t[name]

            # Localized means via L2 pooling (as per paper formula)
            mu_r = self.pool(phi_r)
            mu_t = self.pool(phi_t)

            # Also need standard means for variance/covariance
            mean_r = self._windowed_mean(phi_r)
            mean_t = self._windowed_mean(phi_t)

            # Variances and cross-covariance
            var_r = self._windowed_variance(phi_r, mean_r)
            var_t = self._windowed_variance(phi_t, mean_t)
            cov_rt = self._windowed_covariance(phi_r, phi_t, mean_r, mean_t)

            # Luminance similarity: l(x,y) = (2*mu_r*mu_t + c1) / (mu_r^2 + mu_t^2 + c1)
            l_map = (2.0 * mu_r * mu_t + _C1) / (mu_r ** 2 + mu_t ** 2 + _C1)

            # Structure/texture similarity: s(x,y) = (2*cov + c2) / (var_r + var_t + c2)
            s_map = (2.0 * cov_rt + _C2) / (var_r + var_t + _C2)

            # Dispersion index: D = var_r / (mu_r + eps)
            # Using the L2 pooled means as local mean proxy
            dispersion = var_r / (mu_r + _EPS)

            # Texture probability: P_tex = sigmoid(k * (D - b))
            p_tex = torch.sigmoid(
                self.sigmoid_gain * (dispersion - self.sigmoid_bias)
            )

            l_maps.append(l_map)
            s_maps.append(s_map)
            p_tex_maps.append(p_tex)
            residuals[name] = phi_r - phi_t
            mu_r_dict[name] = mu_r

        return {
            "l_maps": l_maps,
            "s_maps": s_maps,
            "p_tex": p_tex_maps,
            "residuals": residuals,
            "mu_r": mu_r_dict,
        }
