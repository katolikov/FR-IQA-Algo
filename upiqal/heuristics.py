"""Module 5: Spatial Artifact Heuristics Engine.

Implements vectorized detection of:
- JPEG blocking artifacts (modulo-8 cross-difference filters + NFA)
- Gibbs ringing artifacts (Sobel edge detection + morphological dilation +
  variance ratio thresholding)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# JPEG Blocking Artifact Detector
# ======================================================================


class JPEGBlockingDetector(nn.Module):
    """Detect JPEG 8x8 DCT blocking artifacts via cross-difference filtering.

    Computes horizontal and vertical absolute cross-differences on the
    luminance channel, accumulates energy at modulo-8 offsets, and flags
    grid phases whose energy exceeds a statistical threshold (NFA-based).

    The detector is *differential*: it computes the blocking mask for both
    ``ref`` and ``tgt`` and returns only the excess blocking introduced in
    the target. This avoids false positives on reference images that contain
    intrinsic periodic content (e.g., stripes, tiled patterns).

    Parameters
    ----------
    block_size : int
        DCT block size (default 8).
    nfa_threshold : float
        Inverse energy-ratio threshold for flagging a grid phase. A phase is
        considered blocked when ``boundary_energy / interior_energy >
        1/nfa_threshold``. Default ``0.25`` → ratio > 4, which catches
        moderate pixel-domain DCT quantization without firing on natural
        textures. (The old default of 0.01 → ratio > 100 was too strict and
        missed realistic blocking artifacts.)
    """

    def __init__(
        self,
        block_size: int = 8,
        nfa_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.nfa_threshold = nfa_threshold

        # Horizontal cross-difference kernel: [1, -1]
        h_kernel = torch.tensor([[[[1.0, -1.0]]]])  # (1,1,1,2)
        self.register_buffer("h_kernel", h_kernel)

        # Vertical cross-difference kernel: [[1], [-1]]
        v_kernel = torch.tensor([[[[1.0], [-1.0]]]])  # (1,1,2,1)
        self.register_buffer("v_kernel", v_kernel)

    @staticmethod
    def _rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance: 0.299R + 0.587G + 0.114B.

        Parameters
        ----------
        x : torch.Tensor
            RGB image ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Luminance ``(B, 1, H, W)``.
        """
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)

    def _cross_diffs(self, img: torch.Tensor):
        """Return horizontal and vertical absolute luminance differences."""
        if img.shape[1] == 3:
            lum = self._rgb_to_luminance(img)
        else:
            lum = img
        e_h = F.conv2d(lum, self.h_kernel, padding=(0, 0)).abs()  # (B,1,H,W-1)
        e_v = F.conv2d(lum, self.v_kernel, padding=(0, 0)).abs()  # (B,1,H-1,W)
        return lum, e_h, e_v

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Compute the differential blocking artifact mask.

        Strategy:
          1. Compute per-pixel absolute horizontal & vertical cross-differences
             for both images.
          2. Subtract ``ref`` energy from ``tgt`` to get EXCESS edge energy
             introduced in the target (true degradation, not intrinsic edges).
          3. For each candidate block-grid phase ``k ∈ [0, bs-1]``, compute
             the mean excess energy at boundary lines vs. interior lines.
             Flag the phase with the highest energy ratio when it exceeds
             ``1 / nfa_threshold`` (default 4).
          4. Produce a binary mask marking boundary columns/rows at the
             winning phase.

        Making the test differential-by-construction (step 2) both fixes:
          - Bug #4: false positives on intrinsic periodic content in ref
            (e.g., self-comparison of a striped image no longer triggers).
          - Bug #3: previously-strict absolute-ratio tests couldn't detect
            real JPEG Q≤10 (tgt boundary ratio 2.0× vs ref 1.58× → differential
            ratio 2.3× on diff-energy is detectable).

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, 3, H, W)`` or ``(B, 1, H, W)``.
        tgt : torch.Tensor
            Target image ``(B, 3, H, W)`` or ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Binary blocking mask ``(B, 1, H, W)``.
        """
        _, e_h_ref, e_v_ref = self._cross_diffs(ref)
        lum_t, e_h_tgt, e_v_tgt = self._cross_diffs(tgt)
        de_h = (e_h_tgt - e_h_ref).clamp(min=0.0)  # excess horizontal edge energy
        de_v = (e_v_tgt - e_v_ref).clamp(min=0.0)  # excess vertical edge energy

        B, _, H, W = lum_t.shape
        bs = self.block_size
        threshold = 1.0 / self.nfa_threshold

        mask_h = torch.zeros(B, 1, H, W, device=lum_t.device)
        ratios_h, cols_per_k = [], []
        for k in range(bs):
            cols = torch.arange(k, W - 1, bs, device=lum_t.device)
            if cols.numel() == 0:
                ratios_h.append(None); cols_per_k.append(None); continue
            a_k = de_h[:, :, :, cols].mean(dim=(2, 3))  # (B, 1)
            other_cols = torch.arange(W - 1, device=lum_t.device)
            other_cols = other_cols[other_cols % bs != k]
            bg = de_h[:, :, :, other_cols].mean(dim=(2, 3)) if other_cols.numel() else a_k
            ratio = a_k / (bg + 1e-8)
            ratios_h.append(ratio); cols_per_k.append(cols)
        if any(r is not None for r in ratios_h):
            stacked = torch.stack(
                [r if r is not None else torch.zeros_like(ratios_h[0])
                 for r in ratios_h], dim=0)
            best_r_h, best_k_h = stacked.max(dim=0)
            is_blocked_h = (best_r_h > threshold).float()
            for b in range(B):
                if is_blocked_h[b, 0].item() > 0:
                    k = int(best_k_h[b, 0].item())
                    cols = cols_per_k[k]
                    if cols is not None and cols.numel() > 0:
                        mask_h[b, 0, :, cols] = 1.0

        mask_v = torch.zeros(B, 1, H, W, device=lum_t.device)
        ratios_v, rows_per_k = [], []
        for k in range(bs):
            rows = torch.arange(k, H - 1, bs, device=lum_t.device)
            if rows.numel() == 0:
                ratios_v.append(None); rows_per_k.append(None); continue
            a_k = de_v[:, :, rows, :].mean(dim=(2, 3))
            other_rows = torch.arange(H - 1, device=lum_t.device)
            other_rows = other_rows[other_rows % bs != k]
            bg = de_v[:, :, other_rows, :].mean(dim=(2, 3)) if other_rows.numel() else a_k
            ratio = a_k / (bg + 1e-8)
            ratios_v.append(ratio); rows_per_k.append(rows)
        if any(r is not None for r in ratios_v):
            stacked = torch.stack(
                [r if r is not None else torch.zeros_like(ratios_v[0])
                 for r in ratios_v], dim=0)
            best_r_v, best_k_v = stacked.max(dim=0)
            is_blocked_v = (best_r_v > threshold).float()
            for b in range(B):
                if is_blocked_v[b, 0].item() > 0:
                    k = int(best_k_v[b, 0].item())
                    rows = rows_per_k[k]
                    if rows is not None and rows.numel() > 0:
                        mask_v[b, 0, rows, :] = 1.0

        return ((mask_h + mask_v) > 0).float()


# ======================================================================
# Gibbs Ringing Artifact Detector
# ======================================================================


class GibbsRingingDetector(nn.Module):
    """Detect Gibbs ringing artifacts near sharp edges.

    Pipeline:
    1. Sobel edge detection -> binary edge map ``E``
    2. Morphological dilation (via max-pool) -> dilated edge ``E ⊕ B``
    3. Proximity mask ``M_prox = (E ⊕ B) - E``
    4. Local variance in proximity zone vs. background
    5. Threshold: ``sigma^2(M_prox) / sigma^2(M_bg) > gamma``

    Parameters
    ----------
    edge_threshold : float
        Threshold ``tau_E`` for binarizing the Sobel gradient magnitude
        (default 0.1).
    dilation_size : int
        Structuring element size for morphological dilation (default 5).
    variance_ratio_threshold : float
        Threshold ``gamma`` for the variance ratio (default 2.0).
    local_window : int
        Window size for local variance computation (default 7).
    """

    def __init__(
        self,
        edge_threshold: float = 0.1,
        dilation_size: int = 5,
        variance_ratio_threshold: float = 2.0,
        local_window: int = 7,
    ) -> None:
        super().__init__()
        self.edge_threshold = edge_threshold
        self.dilation_size = dilation_size
        self.variance_ratio_threshold = variance_ratio_threshold
        self.local_window = local_window

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)

    def _sobel_edge(self, lum: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude.

        Parameters
        ----------
        lum : torch.Tensor
            Luminance ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Gradient magnitude ``(B, 1, H, W)``.
        """
        gx = F.conv2d(lum, self.sobel_x, padding=1)
        gy = F.conv2d(lum, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)

    def _local_variance(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean local variance within a binary mask.

        Parameters
        ----------
        x : torch.Tensor
            Input luminance ``(B, 1, H, W)``.
        mask : torch.Tensor
            Binary mask ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Per-sample variance ``(B, 1)``.
        """
        ws = self.local_window
        pad = ws // 2
        ones_k = torch.ones(1, 1, ws, ws, device=x.device, dtype=x.dtype)

        # Local mean and mean of squares
        local_sum = F.conv2d(x * mask, ones_k, padding=pad)
        local_cnt = F.conv2d(mask, ones_k, padding=pad).clamp(min=1.0)
        local_mean = local_sum / local_cnt

        local_sq_sum = F.conv2d(x * x * mask, ones_k, padding=pad)
        local_var = (local_sq_sum / local_cnt) - local_mean ** 2
        local_var = local_var.clamp(min=0.0) * mask

        # Average variance across valid locations
        total = mask.sum(dim=(2, 3)).clamp(min=1.0)
        return local_var.sum(dim=(2, 3)) / total  # (B, 1)

    def _compute_ringing_mask(self, img: torch.Tensor) -> torch.Tensor:
        """Compute ringing mask for a single image.

        Parameters
        ----------
        img : torch.Tensor
            Image ``(B, 3, H, W)`` or ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Binary ringing mask ``(B, 1, H, W)``.
        """
        if img.shape[1] == 3:
            lum = self._rgb_to_luminance(img)
        else:
            lum = img

        # 1. Sobel edge detection
        grad_mag = self._sobel_edge(lum)
        E = (grad_mag > self.edge_threshold).float()

        # 2. Morphological dilation via max-pool
        ds = self.dilation_size
        pad = ds // 2
        E_dilated = F.max_pool2d(E, kernel_size=ds, stride=1, padding=pad)

        # 3. Proximity mask: M_prox = dilated - original edges
        M_prox = (E_dilated - E).clamp(min=0.0)

        # 4. Background mask
        M_bg = 1.0 - E_dilated

        # 5. Variance ratio
        var_prox = self._local_variance(lum, M_prox)  # (B, 1)
        var_bg = self._local_variance(lum, M_bg)  # (B, 1)
        ratio = var_prox / (var_bg + 1e-12)

        # 6. Flag entire proximity zone if ratio > gamma
        is_ringing = (ratio > self.variance_ratio_threshold).float()  # (B, 1)
        ringing_mask = M_prox * is_ringing.unsqueeze(-1).unsqueeze(-1)
        return ringing_mask

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Compute differential Gibbs ringing mask.

        Computes ringing for both reference and target independently,
        then returns only the excess ringing in the target (artifacts
        introduced by degradation, not inherent image structure).

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, 3, H, W)`` or ``(B, 1, H, W)``.
        tgt : torch.Tensor
            Target image ``(B, 3, H, W)`` or ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Differential ringing mask ``(B, 1, H, W)``.
        """
        ringing_ref = self._compute_ringing_mask(ref)
        ringing_tgt = self._compute_ringing_mask(tgt)
        return (ringing_tgt - ringing_ref).clamp(min=0.0)


# ======================================================================
# Combined heuristics module
# ======================================================================


class SpatialHeuristicsEngine(nn.Module):
    """Combines JPEG blocking and Gibbs ringing detection.

    Parameters
    ----------
    block_size : int
        DCT block size (default 8).
    nfa_threshold : float
        NFA threshold for blocking detection (default 0.01).
    edge_threshold : float
        Sobel threshold for ringing detection (default 0.1).
    dilation_size : int
        Morphological dilation kernel size (default 5).
    variance_ratio_threshold : float
        Variance ratio gamma for ringing (default 2.0).
    """

    def __init__(
        self,
        block_size: int = 8,
        nfa_threshold: float = 0.35,
        edge_threshold: float = 0.1,
        dilation_size: int = 5,
        variance_ratio_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocking = JPEGBlockingDetector(block_size, nfa_threshold)
        self.ringing = GibbsRingingDetector(
            edge_threshold, dilation_size, variance_ratio_threshold
        )

    def forward(
        self, ref: torch.Tensor, tgt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run both artifact detectors.

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, 3, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Target image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"blocking_mask"`` and ``"ringing_mask"``, each ``(B, 1, H, W)``.
        """
        return {
            "blocking_mask": self.blocking(ref, tgt),
            "ringing_mask": self.ringing(ref, tgt),
        }
