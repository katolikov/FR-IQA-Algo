"""Central UPIQAL orchestrator module.

Wires together all five sub-modules and produces the final FR-IQA scalar
score and multi-channel Diagnostic Tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from upiqal.normalize import Normalizer
from upiqal.color import ChromaticTransportEvaluator
from upiqal.features import DeepStatisticalExtractor
from upiqal.uncertainty import ProbabilisticUncertaintyMapper
from upiqal.heuristics import SpatialHeuristicsEngine


def _safe_per_sample_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale a ``(B, 1, H, W)`` map to ``[0, 1]`` per sample.

    Returns an all-zero tensor for samples whose maximum is below ``1e-3``
    so identical / near-identical inputs cannot trigger pathological
    division by a near-zero peak.  The threshold (1e-3) is large enough
    to absorb MPS / float32 numerical drift that otherwise let 1e-4-scale
    Sinkhorn / EMD noise get amplified to full saturation.
    """
    B = x.shape[0]
    flat = x.view(B, -1)
    peak = flat.max(dim=1, keepdim=True).values  # (B, 1)
    safe_peak = peak.clamp(min=1e-3).view(B, 1, 1, 1)
    out = x / safe_peak
    is_small = (peak < 1e-3).view(B, 1, 1, 1)
    return torch.where(is_small, torch.zeros_like(out), out)


class UPIQAL(nn.Module):
    """Unified Probabilistic Image Quality and Artifact Locator.

    Orchestrates five cascading modules:

    1. **Normalizer** -- Minmax scaling + ImageNet or histogram matching.
    2. **ChromaticTransportEvaluator** -- Oklab EMD color degradation map.
    3. **DeepStatisticalExtractor** -- VGG16 A-DISTS-style adaptive features.
    4. **ProbabilisticUncertaintyMapper** -- Mahalanobis anomaly map.
    5. **SpatialHeuristicsEngine** -- JPEG blocking + Gibbs ringing masks.

    Parameters
    ----------
    norm_mode : str
        ``"imagenet"`` (default) or ``"histogram"`` for Module 1.
    pretrained_vgg : bool
        Load ImageNet-pretrained VGG16 weights (default True).
    color_patch_size : int
        Patch size for the chromatic evaluator (default 16).
    sinkhorn_iters : int
        Sinkhorn iterations for EMD approximation (default 20).
    alpha : float
        Base weight for structure similarity in the aggregation (default 0.5).
    beta : float
        Base weight for texture (variance) similarity (default 0.5).
    w_color : float
        Weight for the color degradation map in the final score (default 0.1).
    w_anomaly : float
        Weight for the Mahalanobis anomaly map (default 0.3).
    w_structure : float
        Weight for the deep feature structure/texture score (default 0.5).
    w_heuristic : float
        Weight for the heuristic artifact penalty (default 0.1).
    """

    def __init__(
        self,
        norm_mode: str = "imagenet",
        pretrained_vgg: bool = True,
        color_patch_size: int = 16,
        sinkhorn_iters: int = 20,
        alpha: float = 0.5,
        beta: float = 0.5,
        w_color: float = 0.1,
        w_anomaly: float = 0.3,
        w_structure: float = 0.5,
        w_heuristic: float = 0.1,
        score_scale: float = 10.0,
        score_center: float = 0.2,
        vgg_weights_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()

        # Learnable / tunable aggregation weights
        self.alpha = alpha
        self.beta = beta
        self.w_color = w_color
        self.w_anomaly = w_anomaly
        self.w_structure = w_structure
        self.w_heuristic = w_heuristic
        # Score calibration (matches upiqal_cli.run_pipeline): without the
        # scale/center, identity inputs map to sigmoid(0.5) ≈ 0.62 instead of
        # the expected 0.9526 upper bound.
        self.score_scale = score_scale
        self.score_center = score_center

        # Sub-modules
        self.normalizer = Normalizer(mode=norm_mode)
        self.chromatic = ChromaticTransportEvaluator(
            patch_size=color_patch_size,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.deep_stats = DeepStatisticalExtractor(
            pretrained=pretrained_vgg,
            weights_path=vgg_weights_path,
        )
        self.uncertainty = ProbabilisticUncertaintyMapper()
        self.heuristics = SpatialHeuristicsEngine()

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_deep_score(
        self,
        l_maps: list[torch.Tensor],
        s_maps: list[torch.Tensor],
        p_tex: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Aggregate per-layer structure / texture similarity into one map.

        Uses P_tex to interpolate between structure (l * cross-cov) and
        texture (variance similarity) weighting.

        Parameters
        ----------
        l_maps : list[torch.Tensor]
            Luminance similarity maps per VGG stage.
        s_maps : list[torch.Tensor]
            Structure/texture similarity maps per VGG stage.
        p_tex : list[torch.Tensor]
            Texture probability maps per VGG stage.
        target_size : tuple[int, int]
            Output spatial resolution ``(H, W)``.

        Returns
        -------
        torch.Tensor
            Aggregated deep similarity map ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        combined = None
        for l_map, s_map, pt in zip(l_maps, s_maps, p_tex):
            # Channel-average
            l_avg = l_map.mean(dim=1, keepdim=True)
            s_avg = s_map.mean(dim=1, keepdim=True)
            pt_avg = pt.mean(dim=1, keepdim=True)

            # Adaptive weighting: high P_tex -> rely on s(x,y);
            #                     low P_tex  -> rely on l(x,y)
            score = (1.0 - pt_avg) * l_avg + pt_avg * s_avg

            # Upsample
            if score.shape[2:] != target_size:
                score = F.interpolate(
                    score, size=target_size, mode="bilinear", align_corners=False
                )
            if combined is None:
                combined = score
            else:
                combined = combined + score

        # Average across layers
        return combined / len(l_maps)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run the full UPIQAL pipeline.

        Parameters
        ----------
        ref : torch.Tensor
            Reference image ``(B, 3, H, W)`` in ``[0, 1]``.
        tgt : torch.Tensor
            Distorted target image ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        dict
            ``"score"`` : scalar FR-IQA quality score in ``[0, 1]``
                          (1 = perfect, 0 = worst).
            ``"diagnostic_tensor"`` : ``(B, 5, H, W)`` multi-channel mask:
                - channel 0: Global Anomaly Map (continuous)
                - channel 1: Color Degradation Map (continuous)
                - channel 2: Deep Similarity Map (continuous)
                - channel 3: Blocking Mask (binary)
                - channel 4: Ringing Mask (binary)
        """
        B, C, H, W = ref.shape

        # ---- Module 1: Normalization ----
        ref_raw = ref.clone()
        tgt_raw = tgt.clone()
        ref_norm, tgt_norm = self.normalizer(ref, tgt)

        # ---- Module 2: Chromatic Transport (on raw [0,1] images) ----
        # Subtract the self-transport baseline: Sinkhorn regularisation
        # produces non-zero EMD even for identical distributions, and on
        # identity that residual peak feeds _safe_per_sample_normalize and
        # inflates color_shift to 100%.  Matches upiqal_cli.run_pipeline.
        color_map_raw = self.chromatic(ref_raw, tgt_raw)   # (B,1,H,W)
        color_baseline = self.chromatic(ref_raw, ref_raw)  # (B,1,H,W)
        color_map = (color_map_raw - color_baseline).clamp(min=0.0)

        # ---- Module 3: Deep Statistics ----
        deep_out = self.deep_stats(ref_norm, tgt_norm)

        # ---- Module 4: Probabilistic Uncertainty ----
        anomaly_map = self.uncertainty(
            deep_out["residuals"], target_size=(H, W)
        )  # (B,1,H,W)

        # ---- Module 5: Spatial Heuristics (on raw images) ----
        # Differential detectors need both ref and tgt: blocking uses the
        # ref modulo-8 grid energy as a baseline; ringing compares variance
        # excess near edges in both images.
        heur = self.heuristics(ref_raw, tgt_raw)
        blocking_mask = heur["blocking_mask"]  # (B,1,H,W)
        ringing_mask = heur["ringing_mask"]  # (B,1,H,W)

        # ---- Aggregation ----
        # Deep similarity map (higher = more similar).  Clamp to [0, 1] so
        # downstream consumers (diagnostics, blur severity = 1 - deep_sim)
        # cannot drift outside the documented range when l/s maps go slightly
        # negative for low-contrast / mismatched-statistics regions.
        deep_sim = self._aggregate_deep_score(
            deep_out["l_maps"],
            deep_out["s_maps"],
            deep_out["p_tex"],
            target_size=(H, W),
        ).clamp(0.0, 1.0)  # (B,1,H,W)

        # Normalize anomaly map to [0, 1] per sample.  When the per-sample
        # peak is essentially zero (identical/near-identical images) we emit
        # an all-zero map instead of dividing by a tiny clamp, which used to
        # blow up to ~1e12 and propagate NaNs through the diagnostics.
        anomaly_norm = _safe_per_sample_normalize(anomaly_map)

        # Normalize color map to [0, 1] using the same safe path.
        color_norm = _safe_per_sample_normalize(color_map)

        # Heuristic penalty: fraction of pixels flagged
        heur_penalty = (blocking_mask + ringing_mask).clamp(0.0, 1.0).mean(dim=(2, 3))  # (B,1)

        # Final score: weighted combination (higher = better)
        # deep_sim is similarity (higher=better), anomaly_norm is distance (lower=better)
        score_spatial = (
            self.w_structure * deep_sim
            - self.w_anomaly * anomaly_norm
            - self.w_color * color_norm
        )
        score_per_pixel = score_spatial.mean(dim=(2, 3))  # (B,1)
        score = (
            self.score_scale * (score_per_pixel - self.score_center)
            - self.w_heuristic * heur_penalty
        )
        # Clamp and shift to [0,1]
        score = torch.sigmoid(score).squeeze(1)  # (B,)

        # Diagnostic tensor: 5 channels
        diagnostic = torch.cat(
            [anomaly_norm, color_norm, deep_sim, blocking_mask, ringing_mask],
            dim=1,
        )  # (B, 5, H, W)

        return {
            "score": score,
            "diagnostic_tensor": diagnostic,
        }
