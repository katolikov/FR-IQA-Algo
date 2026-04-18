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
        score_mode: str = "sigmoid",
        learnable_aggregation: bool = True,
        vgg_weights_path: Union[str, Path, None] = None,
        uncertainty_parameterization: str = "diagonal",
        uncertainty_weights: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # Aggregation weights.  The paper describes a "fully connected
        # regression head that aggregates the weighted sums of local
        # log-probabilities".  We implement that as four learnable
        # scalars (nn.Parameter) initialised to the legacy hand-tuned
        # values, so inference is bit-identical until someone trains
        # them; legacy float behaviour is restored by passing
        # learnable_aggregation=False.
        if learnable_aggregation:
            self.w_color = nn.Parameter(torch.tensor(float(w_color)))
            self.w_anomaly = nn.Parameter(torch.tensor(float(w_anomaly)))
            self.w_structure = nn.Parameter(torch.tensor(float(w_structure)))
            self.w_heuristic = nn.Parameter(torch.tensor(float(w_heuristic)))
            self.score_scale = nn.Parameter(torch.tensor(float(score_scale)))
            self.score_center = nn.Parameter(torch.tensor(float(score_center)))
        else:
            self.w_color = w_color
            self.w_anomaly = w_anomaly
            self.w_structure = w_structure
            self.w_heuristic = w_heuristic
            self.score_scale = score_scale
            self.score_center = score_center

        # score_mode: "sigmoid" (legacy, keeps current calibration) or
        # "nll" (paper-style "negative sum of log-probabilities normalised
        # to [0,1]").  The NLL mode maps log-probability-like signals
        # (anomaly, color-EMD, 1 - structure, heuristic penalty) through
        # an exponentially-decaying transformation, producing a score in
        # [0, 1] whose tail distribution more closely matches the paper's
        # prescription without requiring sigmoid re-calibration.
        if score_mode not in ("sigmoid", "nll"):
            raise ValueError(
                f"score_mode must be 'sigmoid' or 'nll'; got {score_mode!r}"
            )
        self.score_mode = score_mode

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
        self.uncertainty = ProbabilisticUncertaintyMapper(
            parameterization=uncertainty_parameterization,
        )
        if uncertainty_weights is not None:
            state = torch.load(
                str(uncertainty_weights), map_location="cpu", weights_only=True
            )
            # Allow loading either a bare state_dict or one wrapped under
            # the "state_dict" key (which is what train_uncertainty.py saves).
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.uncertainty.load_state_dict(state, strict=False)
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

    def forward(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
        ref_full: Optional[torch.Tensor] = None,
        tgt_full: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run the full UPIQAL pipeline.

        Parameters
        ----------
        ref, tgt : torch.Tensor
            Reference / distorted images ``(B, 3, H, W)`` in ``[0, 1]``.
            These are the deep-feature-level tensors (per the paper's
            Phase 1, typically downsampled to ~256 px on the longer side
            so VGG16 sees the expected spatial frequencies).
        ref_full, tgt_full : torch.Tensor, optional
            Full-resolution copies ``(B, 3, H', W')`` to feed Module 5
            (spatial heuristics).  When omitted, the heuristics run on
            ``ref`` / ``tgt`` at the same resolution as the rest of the
            pipeline (legacy behaviour).

        Returns
        -------
        dict
            ``"score"`` : scalar FR-IQA quality score in ``[0, 1]``.
            ``"diagnostic_tensor"`` : ``(B, 7, H, W)``; channels are
                0 anomaly, 1 color, 2 structure, 3 blocking, 4 ringing,
                5 noise, 6 blur.
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

        # ---- Module 5: Spatial Heuristics ----
        # Multi-scale pyramid: if full-resolution copies are supplied,
        # heuristics run at native resolution and the masks are then
        # downsampled back to the pipeline (H, W).  Otherwise the
        # pyramid-level tensors are used directly.
        if ref_full is not None and tgt_full is not None:
            heur = self.heuristics(ref_full, tgt_full)
            heur = {
                k: (
                    v
                    if v.shape[-2:] == (H, W)
                    else F.interpolate(
                        v, size=(H, W), mode="bilinear", align_corners=False
                    )
                )
                for k, v in heur.items()
            }
        else:
            heur = self.heuristics(ref_raw, tgt_raw)
        blocking_mask = heur["blocking_mask"]  # (B,1,H,W)
        ringing_mask = heur["ringing_mask"]  # (B,1,H,W)
        noise_mask = heur["noise_mask"]  # (B,1,H,W)
        blur_mask = heur["blur_mask"]  # (B,1,H,W)

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

        if self.score_mode == "sigmoid":
            # Legacy calibration (self-comparison -> ~0.9526, different
            # images -> ~0.85, cartoon -> ~0.72).  Weighted sum of mean
            # per-pixel similarity / dissimilarity signals, passed through
            # a calibrated sigmoid.
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
            score = torch.sigmoid(score).squeeze(1)  # (B,)
        else:  # "nll"
            # Paper prescription: negative sum of log-probabilities across
            # all perceptual components and spatial locations, normalised
            # to [0, 1].  We treat each per-pixel dissimilarity channel as
            # an empirical -log p of the target feature under the
            # reference distribution; their sum is an aggregate NLL, and
            # exp(-NLL / score_scale) maps [0, inf) to (0, 1] as the
            # canonical normalisation.  Self-comparison (all dissimilarity
            # signals ~ 0) -> score ~ 1; increasingly different images ->
            # exponentially-decaying score.
            nll_spatial = (
                self.w_anomaly * anomaly_norm
                + self.w_color * color_norm
                + self.w_structure * (1.0 - deep_sim)
            )
            nll_mean = nll_spatial.mean(dim=(2, 3))  # (B, 1)
            total_nll = nll_mean + self.w_heuristic * heur_penalty
            # score_scale acts as an inverse temperature: larger values
            # sharpen the mapping so moderate NLL already drags the score
            # visibly below 1.
            score = torch.exp(-self.score_scale * total_nll).squeeze(1)
            score = score.clamp(0.0, 1.0)

        # Diagnostic tensor: 7 channels.
        # Channel layout (indexed downstream by the CLI and web frontend):
        #   0 anomaly, 1 color, 2 structure, 3 blocking, 4 ringing,
        #   5 noise (wavelet-MAD),  6 blur (HF-attenuation).
        diagnostic = torch.cat(
            [
                anomaly_norm, color_norm, deep_sim,
                blocking_mask, ringing_mask,
                noise_mask, blur_mask,
            ],
            dim=1,
        )  # (B, 7, H, W)

        return {
            "score": score,
            "diagnostic_tensor": diagnostic,
        }
