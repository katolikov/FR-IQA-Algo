"""Module 5: Spatial Artifact Heuristics Engine.

Implements vectorized detection of:
- JPEG blocking artifacts (modulo-8 cross-difference filters + true
  a-contrario Number-of-False-Alarms binomial tail test)
- Gibbs ringing artifacts (Sobel edge detection + morphological dilation
  + systematic-to-statistical error quotient eps, lifted from CT imaging)
- Additive Gaussian noise (multi-level Haar wavelet HH-subband + Donoho
  MAD estimator across 3 decomposition levels)
- Blur / loss of detail (high-frequency energy attenuation in target vs
  reference, i.e. edge-spread)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# JPEG Blocking Artifact Detector
# ======================================================================


class JPEGBlockingDetector(nn.Module):
    """Detect JPEG 8x8 DCT blocking artifacts via a-contrario cross-differences.

    Computes horizontal and vertical absolute cross-differences on the
    luminance channel, accumulates energy at modulo-8 offsets, and flags
    grid phases whose Number-of-False-Alarms (NFA) falls below 1.

    **A-contrario test.** For each candidate phase ``k`` of the 8-pixel
    grid, let

    - ``N_k``  = number of candidate boundary positions at phase k,
    - ``T``    = a robust threshold on the differential cross-difference
                 (we use the 95th percentile of the target's differential
                 horizontal/vertical energy, ignoring zeros),
    - ``p``    = empirical probability that a non-boundary difference
                 exceeds ``T`` (~ 0.05 by construction, but the threshold
                 is re-estimated per image so self-calibrates),
    - ``k_k``  = number of boundary positions at phase k whose
                 differential cross-difference exceeds ``T``.

    Under the null hypothesis "no artificial grid at phase k", ``k_k`` is
    Binomial(N_k, p).  The NFA is

        NFA(k) = N_tests * P[K >= k_k]

    with ``N_tests = bs`` (one per candidate phase).  A phase is flagged
    blocking when ``NFA(k) < nfa_threshold`` (default 1.0 - the classical
    "meaningful" threshold from Desolneux-Moisan-Morel).

    The detector is *differential*: it compares target excess energy to
    the reference, so intrinsic periodic content in the reference (tiles,
    fences, brickwork) does not trigger false positives.

    Parameters
    ----------
    block_size : int
        DCT block size (default 8).
    nfa_threshold : float
        Meaningfulness threshold on NFA (default 1.0 - a phase is flagged
        when the expected number of false alarms across all tested phases
        is below 1).  **Lower = stricter.**
    percentile : float
        Percentile of the non-zero differential cross-difference used as
        the threshold ``T`` for the binomial test (default 95.0).
    """

    def __init__(
        self,
        block_size: int = 8,
        nfa_threshold: float = 1e-2,
        percentile: float = 95.0,
    ) -> None:
        super().__init__()
        if not 0.0 < percentile < 100.0:
            raise ValueError(
                f"percentile must be in (0, 100); got {percentile}"
            )
        self.block_size = block_size
        self.nfa_threshold = float(nfa_threshold)
        self.percentile = float(percentile)

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

    # ------------------------------------------------------------------
    # A-contrario binomial tail (log-domain for numerical stability)
    # ------------------------------------------------------------------
    @staticmethod
    def _binom_tail_ge(k: int, n: int, p: float) -> float:
        """Return P[Binom(n, p) >= k] using log-sum-exp for stability."""
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        # Sum over i in [k, n] of C(n, i) p^i (1-p)^(n-i)
        import math

        log_p = math.log(p)
        log_q = math.log(1.0 - p)
        # Compute log C(n, i) iteratively, then use logsumexp.
        log_pmf: list[float] = []
        log_c = 0.0  # log C(n, 0) = 0
        for i in range(0, k):
            # advance to log C(n, i+1) = log C(n, i) + log((n-i)/(i+1))
            log_c += math.log((n - i) / (i + 1))
        for i in range(k, n + 1):
            log_pmf.append(log_c + i * log_p + (n - i) * log_q)
            if i < n:
                log_c += math.log((n - i) / (i + 1))
        m = max(log_pmf)
        return math.exp(m + math.log(sum(math.exp(v - m) for v in log_pmf)))

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Compute the differential blocking artifact mask via NFA.

        Strategy (differential + a-contrario):
          1. Compute per-pixel absolute horizontal/vertical cross-differences
             for both images; take the clamped (non-negative) difference
             ``tgt - ref`` so intrinsic periodic content does NOT count.
          2. Pick a threshold ``T`` = the ``percentile`` quantile of the
             positive differential (self-calibrating per image).
          3. For each phase ``k in [0, bs-1]``, count boundary positions
             whose differential exceeds ``T``; compute NFA via the
             binomial tail and flag phases with ``NFA(k) < nfa_threshold``.
          4. Emit a binary mask marking boundary rows/columns at every
             flagged phase.

        Parameters
        ----------
        ref, tgt : torch.Tensor
            Images ``(B, 3, H, W)`` or ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Binary blocking mask ``(B, 1, H, W)``.
        """
        _, e_h_ref, e_v_ref = self._cross_diffs(ref)
        lum_t, e_h_tgt, e_v_tgt = self._cross_diffs(tgt)
        de_h = (e_h_tgt - e_h_ref).clamp(min=0.0)  # (B, 1, H, W-1)
        de_v = (e_v_tgt - e_v_ref).clamp(min=0.0)  # (B, 1, H-1, W)

        B, _, H, W = lum_t.shape
        bs = self.block_size

        mask_h = self._nfa_axis(de_h, axis="h", B=B, H=H, W=W, bs=bs)
        mask_v = self._nfa_axis(de_v, axis="v", B=B, H=H, W=W, bs=bs)
        return ((mask_h + mask_v) > 0).float()

    def _nfa_axis(
        self,
        de: torch.Tensor,
        axis: str,
        B: int,
        H: int,
        W: int,
        bs: int,
    ) -> torch.Tensor:
        """Per-axis a-contrario test. ``axis`` is "h" or "v"."""
        out = torch.zeros(B, 1, H, W, device=de.device, dtype=de.dtype)
        # Per-batch threshold T = percentile of nonzero differential energy.
        for b in range(B):
            flat = de[b].flatten()
            # Only positive entries carry information under the null.
            nonzero = flat[flat > 0]
            if nonzero.numel() < bs:  # not enough data to calibrate
                continue
            T = float(
                torch.quantile(nonzero, self.percentile / 100.0).item()
            )
            # Empirical null-hypothesis hit rate: how often a generic
            # position exceeds T in the differential map.
            p = float((de[b] > T).float().mean().item())
            if p <= 0.0 or p >= 1.0:
                continue
            n_tests = bs
            for k in range(bs):
                if axis == "h":
                    idx = torch.arange(k, W - 1, bs, device=de.device)
                    if idx.numel() == 0:
                        continue
                    cells = de[b, :, :, idx]
                else:
                    idx = torch.arange(k, H - 1, bs, device=de.device)
                    if idx.numel() == 0:
                        continue
                    cells = de[b, :, idx, :]
                n_k = int(cells.numel())
                k_k = int((cells > T).sum().item())
                pvalue = self._binom_tail_ge(k_k, n_k, p)
                nfa = n_tests * pvalue
                if nfa < self.nfa_threshold:
                    if axis == "h":
                        out[b, 0, :, idx] = 1.0
                    else:
                        out[b, 0, idx, :] = 1.0
        return out


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
        epsilon_threshold: float = 2.5,
    ) -> None:
        super().__init__()
        self.edge_threshold = edge_threshold
        self.dilation_size = dilation_size
        self.variance_ratio_threshold = variance_ratio_threshold
        self.local_window = local_window
        # Systematic-to-statistical error quotient threshold (CT-imaging
        # formulation, Tang et al. 2012 "Quantification of ring artifact
        # visibility in CT"). Higher epsilon => more evidence of ringing.
        self.epsilon_threshold = float(epsilon_threshold)

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

    def _mean_abs_hf(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean absolute high-pass residual inside a binary mask.

        The residual ``x - lowpass(x)`` isolates oscillatory content; its
        mean absolute value inside ``mask`` is the "systematic" error
        term in the eps quotient.
        """
        ws = self.local_window
        pad = ws // 2
        ones_k = torch.ones(1, 1, ws, ws, device=x.device, dtype=x.dtype) / float(
            ws * ws
        )
        lp = F.conv2d(x, ones_k, padding=pad)
        hf = (x - lp).abs()
        num = (hf * mask).sum(dim=(2, 3))
        den = mask.sum(dim=(2, 3)).clamp(min=1.0)
        return num / den  # (B, 1)

    def _compute_ringing_mask(self, img: torch.Tensor) -> torch.Tensor:
        """Compute ringing mask for a single image.

        Uses the systematic-to-statistical error quotient ``eps`` adapted
        from CT ring-artifact quantification:

        .. math::

            \\epsilon = \\frac{|\\overline{|hf|_{\\text{prox}}} -
                               \\overline{|hf|_{\\text{bg}}}|}
                              {\\sigma(hf_{\\text{bg}}) + \\varepsilon_0}

        where ``hf = I - LP(I)`` is the high-pass residual, ``prox`` is
        the edge-proximity mask, and ``bg`` is the complementary
        background.  A region is flagged as containing ringing when
        ``eps > epsilon_threshold`` AND the legacy variance ratio
        agrees (keeps prior sensitivity as a soft lower bound).

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

        # 5. Variance ratio (legacy statistical test)
        var_prox = self._local_variance(lum, M_prox)  # (B, 1)
        var_bg = self._local_variance(lum, M_bg)  # (B, 1)
        ratio = var_prox / (var_bg + 1e-12)

        # 6. eps = systematic-to-statistical error quotient
        hp_prox = self._mean_abs_hf(lum, M_prox)  # (B, 1)
        hp_bg = self._mean_abs_hf(lum, M_bg)  # (B, 1)
        sigma_bg = var_bg.clamp(min=1e-12).sqrt()
        eps = (hp_prox - hp_bg).abs() / (sigma_bg + 1e-8)

        # 7. Flag if BOTH tests agree the proximity zone is anomalous.
        is_ringing = (
            (ratio > self.variance_ratio_threshold)
            & (eps > self.epsilon_threshold)
        ).float()  # (B, 1)
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
# Additive Gaussian Noise Detector (wavelet MAD)
# ======================================================================


# Haar wavelet filters (2x2).  Scaled by 1/2 so that applying them to a
# constant image returns 0 in the detail subbands.
_HAAR_FILTERS = torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0]],    # LL (approximation)
        [[1.0, 1.0], [-1.0, -1.0]],  # LH (horizontal detail)
        [[1.0, -1.0], [1.0, -1.0]],  # HL (vertical detail)
        [[1.0, -1.0], [-1.0, 1.0]],  # HH (diagonal detail -> noise signature)
    ]
) * 0.5


class NoiseDetector(nn.Module):
    """Detect additive Gaussian noise via multi-level Haar HH-subband MAD.

    Applies ``levels`` successive Haar wavelet decompositions to the
    luminance channel.  At each level the LL approximation is recursed
    into and the diagonal (``HH``) subband is kept.  Each HH subband is
    converted to a per-pixel noise-sigma estimate by dividing the
    absolute values by 0.6745 (the Donoho MAD constant) and smoothed
    with a local box filter.  The per-level sigma maps are bilinearly
    upsampled to full resolution and fused by taking the **maximum**,
    which captures noise at whatever scale it dominates (single-pixel
    speckle at level 1, coarser additive noise at levels 2-3).

    The final mask is the non-negative differential between target and
    reference sigma maps, rescaled by ``noise_scale`` to ``[0, 1]``.

    Parameters
    ----------
    smoothing : int
        Side length of the box filter applied to ``|HH|`` before
        differencing (default 5, must be odd).
    noise_scale : float
        Divisor on the sigma estimate; a typical JPEG / mild camera
        noise (sigma ~10 on a 0-255 scale ~= 0.04) maps to mask values
        around 0.5 (default 0.04).
    levels : int
        Number of wavelet decomposition levels.  3 is the paper default
        (Donoho & Johnstone 1994); level 1 captures the finest speckle,
        levels 2-3 capture coarse additive noise (default 3).  Must be
        >= 1.
    """

    def __init__(
        self,
        smoothing: int = 5,
        noise_scale: float = 0.04,
        levels: int = 3,
    ) -> None:
        super().__init__()
        if smoothing < 1 or smoothing % 2 == 0:
            raise ValueError(
                f"smoothing must be a positive odd integer; got {smoothing}"
            )
        if levels < 1:
            raise ValueError(f"levels must be >= 1; got {levels}")
        self.smoothing = smoothing
        self.noise_scale = float(noise_scale)
        self.levels = int(levels)
        # Register the four Haar filters as a buffer so .to(device) moves them.
        self.register_buffer(
            "haar",
            _HAAR_FILTERS.unsqueeze(1).clone(),  # (4, 1, 2, 2)
        )
        # Box filter for local smoothing of |HH|.
        box = torch.ones(1, 1, smoothing, smoothing) / float(smoothing * smoothing)
        self.register_buffer("box", box)

    @staticmethod
    def _rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _sigma_map(self, img: torch.Tensor) -> torch.Tensor:
        """Return a smoothed per-pixel noise-sigma estimate via multi-level DWT.

        Fuses per-level sigma maps via element-wise max so noise at any
        scale shows up in the final map.

        Shape: same spatial dims as the input luminance (B, 1, H, W).
        """
        y = self._rgb_to_luminance(img)  # (B, 1, H, W)
        target_hw = y.shape[-2:]
        pad = self.smoothing // 2

        ll = y
        fused: Optional[torch.Tensor] = None
        for _lvl in range(self.levels):
            # Pad to even dims so stride-2 conv stays clean.
            if ll.shape[-2] % 2:
                ll = F.pad(ll, (0, 0, 0, 1), mode="replicate")
            if ll.shape[-1] % 2:
                ll = F.pad(ll, (0, 1, 0, 0), mode="replicate")
            if ll.shape[-1] < 2 or ll.shape[-2] < 2:
                break
            coeffs = F.conv2d(ll, self.haar, stride=2)
            ll = coeffs[:, 0:1]  # LL for next level
            hh = coeffs[:, 3:4]
            sigma = hh.abs() / 0.6745  # Donoho MAD factor
            sigma = F.conv2d(sigma, self.box, padding=pad)
            sigma = F.interpolate(
                sigma, size=target_hw, mode="bilinear", align_corners=False
            )
            fused = sigma if fused is None else torch.max(fused, sigma)
        assert fused is not None
        return fused

    def forward(
        self, ref: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        """Return the differential noise mask ``(B, 1, H, W)`` in ``[0, 1]``.

        0 means "no more noise in target than reference" and 1 means "a lot
        of extra noise at this location".
        """
        if ref.shape != tgt.shape:
            raise ValueError(
                f"ref and tgt shapes must match: {tuple(ref.shape)} vs "
                f"{tuple(tgt.shape)}"
            )
        sigma_ref = self._sigma_map(ref)
        sigma_tgt = self._sigma_map(tgt)
        excess = (sigma_tgt - sigma_ref).clamp(min=0.0) / max(
            self.noise_scale, 1e-6
        )
        return excess.clamp(0.0, 1.0)


# ======================================================================
# Blur / Loss-of-Detail Detector (edge-spread / HF attenuation)
# ======================================================================


class BlurDetector(nn.Module):
    """Detect generalized blur by measuring high-frequency energy loss.

    For each image we compute a local high-frequency energy map

    .. math::

        E(x, y) = \\mathrm{box}( (I - G_\\sigma * I)^2 )

    where ``G_\\sigma`` is a small Gaussian kernel.  The target-to-reference
    ratio ``E_t / E_r`` drops below 1 exactly where the target has lost
    spatial detail relative to the reference (i.e. blur).  The final mask
    is ``clamp(1 - E_t / E_r, 0, 1)`` so that 0 = "no detail lost" and 1 =
    "target is completely smoothed out at this location".

    This is the dedicated, spatially-precise complement to the generic
    ``(1 - deep_similarity)`` proxy previously used as the blur severity
    signal in the CLI.

    Parameters
    ----------
    blur_sigma : float
        Sigma of the internal Gaussian used to extract the low-pass
        component (default 1.5 px).
    smoothing : int
        Box-filter side length applied to the squared residual so the HF
        energy map is spatially smooth (default 7).
    min_ref_energy : float
        Floor on the reference HF energy below which the ratio is not
        evaluated (pixels in flat regions don't carry a meaningful blur
        signal; default 1e-4).
    """

    def __init__(
        self,
        blur_sigma: float = 1.5,
        smoothing: int = 7,
        min_ref_energy: float = 1e-4,
    ) -> None:
        super().__init__()
        if smoothing < 1 or smoothing % 2 == 0:
            raise ValueError(
                f"smoothing must be a positive odd integer; got {smoothing}"
            )
        self.min_ref_energy = float(min_ref_energy)
        k = _gaussian_kernel_2d(blur_sigma)  # (K, K)
        self.register_buffer(
            "gaussian",
            k.view(1, 1, k.shape[0], k.shape[1]).contiguous(),
        )
        box = torch.ones(1, 1, smoothing, smoothing) / float(smoothing * smoothing)
        self.register_buffer("box", box)

    @staticmethod
    def _rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _hf_energy(self, img: torch.Tensor) -> torch.Tensor:
        y = self._rgb_to_luminance(img)  # (B, 1, H, W)
        k = self.gaussian
        pad = k.shape[-1] // 2
        lo = F.conv2d(y, k, padding=pad)
        hf = (y - lo).pow(2)
        pad_b = self.box.shape[-1] // 2
        return F.conv2d(hf, self.box, padding=pad_b)

    def forward(
        self, ref: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        """Return the differential blur mask ``(B, 1, H, W)`` in ``[0, 1]``.

        0 means "target is at least as detailed as reference" and 1 means
        "target has completely lost detail at this location".
        """
        if ref.shape != tgt.shape:
            raise ValueError(
                f"ref and tgt shapes must match: {tuple(ref.shape)} vs "
                f"{tuple(tgt.shape)}"
            )
        e_ref = self._hf_energy(ref)
        e_tgt = self._hf_energy(tgt)
        # Stabilize the ratio by clamping the denominator rather than
        # adding an epsilon — that way, in valid regions where
        # ``e_ref == e_tgt``, the ratio is exactly 1.0 and the mask is
        # exactly 0 (important for self-comparison bit-parity tests).
        denom = e_ref.clamp(min=self.min_ref_energy)
        ratio = e_tgt / denom
        blur = (1.0 - ratio).clamp(0.0, 1.0)
        # Flat regions in the reference (below min_ref_energy) carry no
        # meaningful blur signal; mask them out entirely.
        valid = (e_ref > self.min_ref_energy).to(blur.dtype)
        return blur * valid


def _gaussian_kernel_2d(sigma: float) -> torch.Tensor:
    """Return a (K, K) Gaussian kernel with ``K = 2*ceil(3*sigma) + 1``."""
    import math as _math

    radius = max(1, int(_math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g1 = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    g1 = g1 / g1.sum()
    return torch.outer(g1, g1)


# ======================================================================
# Combined heuristics module
# ======================================================================


class SpatialHeuristicsEngine(nn.Module):
    """Combines JPEG blocking, Gibbs ringing, noise, and blur detection.

    Parameters
    ----------
    block_size : int
        DCT block size (default 8).
    nfa_threshold : float
        NFA threshold for blocking detection (default 0.35).
    edge_threshold : float
        Sobel threshold for ringing detection (default 0.1).
    dilation_size : int
        Morphological dilation kernel size (default 5).
    variance_ratio_threshold : float
        Variance ratio gamma for ringing (default 2.0).
    noise_smoothing : int
        Side length of the box filter for the noise sigma estimator
        (default 5, must be odd).
    noise_scale : float
        Divisor on the estimated sigma; smaller = more sensitive to noise
        (default 0.04).
    blur_sigma : float
        Sigma of the low-pass filter in the blur detector (default 1.5).
    blur_smoothing : int
        Side length of the box filter for the HF energy map (default 7,
        must be odd).
    """

    def __init__(
        self,
        block_size: int = 8,
        nfa_threshold: float = 1e-2,
        edge_threshold: float = 0.1,
        dilation_size: int = 5,
        variance_ratio_threshold: float = 2.0,
        noise_smoothing: int = 5,
        noise_scale: float = 0.04,
        blur_sigma: float = 1.5,
        blur_smoothing: int = 7,
        epsilon_threshold: float = 2.5,
        wavelet_levels: int = 3,
    ) -> None:
        super().__init__()
        self.blocking = JPEGBlockingDetector(block_size, nfa_threshold)
        self.ringing = GibbsRingingDetector(
            edge_threshold=edge_threshold,
            dilation_size=dilation_size,
            variance_ratio_threshold=variance_ratio_threshold,
            epsilon_threshold=epsilon_threshold,
        )
        self.noise = NoiseDetector(
            smoothing=noise_smoothing,
            noise_scale=noise_scale,
            levels=wavelet_levels,
        )
        self.blur = BlurDetector(
            blur_sigma=blur_sigma, smoothing=blur_smoothing
        )

    def forward(
        self, ref: torch.Tensor, tgt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run all four artifact detectors.

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, 3, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Target image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``blocking_mask``, ``ringing_mask``, ``noise_mask``,
            ``blur_mask``, each ``(B, 1, H, W)``.
        """
        return {
            "blocking_mask": self.blocking(ref, tgt),
            "ringing_mask": self.ringing(ref, tgt),
            "noise_mask": self.noise(ref, tgt),
            "blur_mask": self.blur(ref, tgt),
        }
