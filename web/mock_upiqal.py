"""Mock UPIQAL model for the web application.

Provides a lightweight stand-in that returns plausible outputs without
requiring VGG16 weights or heavy computation, so the app runs out of the box.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MockUPIQAL(nn.Module):
    """Lightweight mock of the UPIQAL model.

    Produces a realistic-looking FR-IQA score and a 5-channel diagnostic
    tensor using simple pixel-level operations (no deep network required).
    """

    def __init__(self) -> None:
        super().__init__()
        # Sobel kernels for edge-based heuristics
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        return (x * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    @torch.no_grad()
    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run a lightweight quality comparison.

        Parameters
        ----------
        ref, tgt : torch.Tensor
            ``(B, 3, H, W)`` images in ``[0, 1]``.

        Returns
        -------
        dict
            ``"score"`` : ``(B,)`` quality score in ``[0, 1]``.
            ``"diagnostic_tensor"`` : ``(B, 5, H, W)`` multi-channel mask.
        """
        B, C, H, W = ref.shape

        # --- Channel 0: Global anomaly (pixel-level L2 difference) ---
        diff = (ref - tgt).pow(2).sum(dim=1, keepdim=True).sqrt()
        anomaly = diff / (diff.max() + 1e-12)

        # --- Channel 1: Color degradation (per-pixel chroma distance) ---
        color_diff = (ref - tgt).abs().mean(dim=1, keepdim=True)
        color_map = color_diff / (color_diff.max() + 1e-12)

        # --- Channel 2: Structural similarity (inverted local MSE) ---
        # Use a simple local mean difference
        kernel = torch.ones(1, 1, 7, 7, device=ref.device) / 49.0
        ref_g = self._to_gray(ref)
        tgt_g = self._to_gray(tgt)
        mu_r = F.conv2d(ref_g, kernel, padding=3)
        mu_t = F.conv2d(tgt_g, kernel, padding=3)
        ssim_approx = (2.0 * mu_r * mu_t + 1e-4) / (mu_r.pow(2) + mu_t.pow(2) + 1e-4)

        # --- Channel 3: Blocking mask (modulo-8 gradient spikes) ---
        gx = F.conv2d(tgt_g, self.sobel_x, padding=1).abs()
        blocking = torch.zeros_like(tgt_g)
        # Flag pixels at 8-pixel grid boundaries
        for k in range(0, W, 8):
            if k < W:
                blocking[:, :, :, k] = gx[:, :, :, min(k, W - 1)]
        for k in range(0, H, 8):
            if k < H:
                blocking[:, :, k, :] = torch.max(
                    blocking[:, :, k, :],
                    F.conv2d(tgt_g, self.sobel_y, padding=1).abs()[:, :, min(k, H - 1), :],
                )
        blocking = (blocking > 0.05).float()

        # --- Channel 4: Ringing mask (high-freq near edges) ---
        grad_mag = (
            F.conv2d(tgt_g, self.sobel_x, padding=1).pow(2)
            + F.conv2d(tgt_g, self.sobel_y, padding=1).pow(2)
        ).sqrt()
        edges = (grad_mag > 0.15).float()
        dilated = F.max_pool2d(edges, kernel_size=5, stride=1, padding=2)
        ringing = (dilated - edges).clamp(min=0.0) * anomaly

        # --- Score ---
        score = 1.0 - anomaly.mean(dim=(1, 2, 3))
        score = score.clamp(0.0, 1.0)

        diagnostic = torch.cat(
            [anomaly, color_map, ssim_approx, blocking, ringing], dim=1
        )

        # --- Diagnostic statistics ---
        blocking_severity = float(blocking.mean().item()) * 100
        ringing_severity = float(ringing.mean().item()) * 100
        noise_severity = float(anomaly.mean().item()) * 100
        color_severity = float(color_map.mean().item()) * 100
        blur_severity = float((1.0 - ssim_approx).mean().item()) * 100

        severity_scores = {
            "blocking": round(min(blocking_severity * 5, 100.0), 1),
            "ringing": round(min(ringing_severity * 5, 100.0), 1),
            "noise": round(min(noise_severity * 3, 100.0), 1),
            "color_shift": round(min(color_severity * 3, 100.0), 1),
            "blur": round(min(blur_severity * 2, 100.0), 1),
        }

        artifact_labels = {
            "blocking": "Severe JPEG Blocking",
            "ringing": "Gibbs Ringing",
            "noise": "Noise / Granularity",
            "color_shift": "Color Shift",
            "blur": "Blur / Loss of Detail",
        }
        dominant_key = max(severity_scores, key=severity_scores.get)
        dominant_artifact = artifact_labels[dominant_key]

        affected_mask = (anomaly > 0.15).float()
        affected_area = round(float(affected_mask.mean().item()) * 100, 1)

        diagnostics = {
            "dominant_artifact": dominant_artifact,
            "severity_scores": severity_scores,
            "affected_area": affected_area,
        }

        return {
            "score": score,
            "diagnostic_tensor": diagnostic,
            "diagnostics": diagnostics,
        }
