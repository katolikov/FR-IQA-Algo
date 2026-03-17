#!/usr/bin/env python3
"""UPIQAL CLI — Standalone command-line tool for Full-Reference Image Quality Assessment.

Runs the five-module UPIQAL pipeline on a reference/target image pair and
saves diagnostic heatmaps (PNG) and a numerical report (JSON) to a
uniquely named output directory.

Modules executed sequentially:
    1. Normalizer             — Minmax scaling + ImageNet normalization
    2. ChromaticTransportEval — Oklab EMD color degradation map
    3. DeepStatisticalExtr    — VGG16 A-DISTS adaptive features
    4. UncertaintyMapper      — Mahalanobis distance anomaly map
    5. SpatialHeuristics      — JPEG blocking + Gibbs ringing masks

Usage:
    python upiqal_cli.py --reference ref.png --target tgt.png
    python upiqal_cli.py --reference ref.png --target tgt.png --name experiment1
    python upiqal_cli.py --reference ref.png --target tgt.png --max-side 256

Dependencies:
    torch>=2.0, torchvision>=0.15, numpy>=1.24, pillow>=10.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# UPIQAL module imports (real implementations from the upiqal/ package)
# ---------------------------------------------------------------------------
from upiqal.normalize import Normalizer
from upiqal.color import ChromaticTransportEvaluator
from upiqal.features import DeepStatisticalExtractor
from upiqal.uncertainty import ProbabilisticUncertaintyMapper
from upiqal.heuristics import SpatialHeuristicsEngine

# Package version (imported for the report)
from upiqal import __version__ as UPIQAL_VERSION


# ======================================================================
# Image I/O helpers
# ======================================================================


def load_image_as_tensor(path: str, max_side: int = 512) -> torch.Tensor:
    """Load an image file and return a ``(1, 3, H, W)`` float tensor in [0, 1].

    The longer side is resized to at most *max_side* pixels (aspect ratio
    preserved) using Lanczos resampling so that inference stays fast.

    Parameters
    ----------
    path : str
        Filesystem path to the image.
    max_side : int
        Maximum pixel dimension for the longer side (default 512).

    Returns
    -------
    torch.Tensor
        Image tensor ``(1, 3, H, W)`` with values in ``[0, 1]``.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
    return tensor


def apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply a JET-like colormap to a [0, 1] grayscale array.

    Returns an ``(H, W, 3)`` uint8 RGB image.  Piecewise-linear
    approximation matching ``web/main.py:_apply_colormap``.

    Parameters
    ----------
    gray : np.ndarray
        2D array with values in [0, 1].

    Returns
    -------
    np.ndarray
        RGB image ``(H, W, 3)`` as uint8.
    """
    x = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_channel_as_png(
    tensor: torch.Tensor,
    channel: int,
    filepath: str,
    use_colormap: bool = True,
) -> None:
    """Extract one channel from a diagnostic tensor and save as PNG.

    Parameters
    ----------
    tensor : torch.Tensor
        Diagnostic tensor ``(1, C, H, W)``.
    channel : int
        Channel index to extract.
    filepath : str
        Output PNG path.
    use_colormap : bool
        If True, apply JET colormap; otherwise save as grayscale.
    """
    arr = tensor[0, channel].cpu().numpy()

    # Normalize to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    if use_colormap:
        rgb = apply_jet_colormap(arr)
    else:
        # Grayscale → 3-channel for consistent PNG output
        rgb = (np.stack([arr] * 3, axis=-1) * 255).astype(np.uint8)

    Image.fromarray(rgb).save(filepath, format="PNG")


# ======================================================================
# Score aggregation (mirrors upiqal/model.py:_aggregate_deep_score)
# ======================================================================


def aggregate_deep_score(
    l_maps: list[torch.Tensor],
    s_maps: list[torch.Tensor],
    p_tex: list[torch.Tensor],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Aggregate per-layer structure/texture similarity into one map.

    Replicates ``UPIQAL._aggregate_deep_score`` from ``upiqal/model.py``.

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
        # Channel-average each map
        l_avg = l_map.mean(dim=1, keepdim=True)
        s_avg = s_map.mean(dim=1, keepdim=True)
        pt_avg = pt.mean(dim=1, keepdim=True)

        # Adaptive weighting: high P_tex → rely on s(x,y); low → rely on l(x,y)
        score = (1.0 - pt_avg) * l_avg + pt_avg * s_avg

        # Upsample to target resolution
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


# ======================================================================
# Diagnostics computation (mirrors web/mock_upiqal.py:111-142)
# ======================================================================

# Severity display multipliers (from mock_upiqal.py)
_SEVERITY_MULTIPLIERS = {
    "blocking": 5.0,
    "ringing": 5.0,
    "noise": 3.0,
    "color_shift": 3.0,
    "blur": 2.0,
}

_ARTIFACT_LABELS = {
    "blocking": "Severe JPEG Blocking",
    "ringing": "Gibbs Ringing",
    "noise": "Noise / Granularity",
    "color_shift": "Color Shift",
    "blur": "Blur / Loss of Detail",
}


def compute_diagnostics(
    anomaly_norm: torch.Tensor,
    color_norm: torch.Tensor,
    deep_sim: torch.Tensor,
    blocking_mask: torch.Tensor,
    ringing_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Compute artifact severity scores, dominant artifact, and affected area.

    Parameters
    ----------
    anomaly_norm : torch.Tensor
        Normalized anomaly map ``(B, 1, H, W)`` in [0, 1].
    color_norm : torch.Tensor
        Normalized color degradation ``(B, 1, H, W)`` in [0, 1].
    deep_sim : torch.Tensor
        Deep similarity map ``(B, 1, H, W)`` in [0, 1].
    blocking_mask : torch.Tensor
        Binary blocking mask ``(B, 1, H, W)``.
    ringing_mask : torch.Tensor
        Binary ringing mask ``(B, 1, H, W)``.

    Returns
    -------
    dict
        Diagnostics dict with ``dominant_artifact``, ``severity_scores``,
        and ``affected_area``.
    """
    # Raw severity percentages
    blocking_sev = float(blocking_mask.mean().item()) * 100
    ringing_sev = float(ringing_mask.mean().item()) * 100
    noise_sev = float(anomaly_norm.mean().item()) * 100
    color_sev = float(color_norm.mean().item()) * 100
    blur_sev = float((1.0 - deep_sim).mean().item()) * 100

    severity_scores = {
        "blocking": round(min(blocking_sev * _SEVERITY_MULTIPLIERS["blocking"], 100.0), 1),
        "ringing": round(min(ringing_sev * _SEVERITY_MULTIPLIERS["ringing"], 100.0), 1),
        "noise": round(min(noise_sev * _SEVERITY_MULTIPLIERS["noise"], 100.0), 1),
        "color_shift": round(min(color_sev * _SEVERITY_MULTIPLIERS["color_shift"], 100.0), 1),
        "blur": round(min(blur_sev * _SEVERITY_MULTIPLIERS["blur"], 100.0), 1),
    }

    # Dominant artifact is the one with the highest severity
    dominant_key = max(severity_scores, key=severity_scores.get)
    dominant_artifact = _ARTIFACT_LABELS[dominant_key]

    # Affected area: percentage of pixels where anomaly exceeds threshold
    affected_mask = (anomaly_norm > 0.15).float()
    affected_area = round(float(affected_mask.mean().item()) * 100, 1)

    return {
        "dominant_artifact": dominant_artifact,
        "severity_scores": severity_scores,
        "affected_area": affected_area,
    }


# ======================================================================
# Quality label from score
# ======================================================================


def score_label(score: float) -> str:
    """Return a human-readable quality label for a FR-IQA score."""
    if score >= 0.9:
        return "Excellent quality"
    elif score >= 0.7:
        return "Good quality"
    elif score >= 0.5:
        return "Moderate degradation"
    elif score >= 0.3:
        return "Poor quality"
    else:
        return "Severe degradation"


# ======================================================================
# Main pipeline
# ======================================================================


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full UPIQAL pipeline and write results to disk."""

    # ── Header ──────────────────────────────────────────────────────
    sep = "\u2500" * 30
    print(f"UPIQAL CLI v{UPIQAL_VERSION}")
    print(sep)
    print(f"Reference : {args.reference}")
    print(f"Target    : {args.target}")

    # ── Load images ─────────────────────────────────────────────────
    ref_tensor = load_image_as_tensor(args.reference, max_side=args.max_side)
    tgt_tensor = load_image_as_tensor(args.target, max_side=args.max_side)

    # Ensure matching spatial dimensions (resize target to reference)
    _, _, rh, rw = ref_tensor.shape
    _, _, th, tw = tgt_tensor.shape
    if (rh, rw) != (th, tw):
        tgt_tensor = F.interpolate(
            tgt_tensor, size=(rh, rw), mode="bilinear", align_corners=False
        )

    B, C, H, W = ref_tensor.shape
    print(f"Resolution: {H} x {W}")
    print(sep)

    # ── Initialize modules (before creating output dir, so failures
    #    like VGG16 download errors don't leave empty directories) ───
    timestamp = datetime.now()
    normalizer = Normalizer(mode="imagenet")
    chromatic = ChromaticTransportEvaluator(patch_size=16, sinkhorn_iters=20)

    # Deep feature extractor — try pretrained, fallback to random init
    try:
        deep_stats = DeepStatisticalExtractor(pretrained=True)
        vgg_status = "ImageNet (pretrained)"
    except Exception:
        deep_stats = DeepStatisticalExtractor(pretrained=False)
        vgg_status = "random initialization (pretrained weights unavailable)"
    print(f"  VGG16 weights: {vgg_status}")

    uncertainty = ProbabilisticUncertaintyMapper()
    heuristics = SpatialHeuristicsEngine()

    # Aggregation weights (defaults from upiqal/model.py)
    w_color = 0.1
    w_anomaly = 0.3
    w_structure = 0.5
    w_heuristic = 0.1

    # ── Create output directory (deferred until modules are ready) ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        tag = args.name if args.name else "results"
        dir_name = f"upiqal_{tag}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path(dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep raw copies for modules that operate on [0, 1] images
    ref_raw = ref_tensor.clone()
    tgt_raw = tgt_tensor.clone()

    # ── Module 1: Normalization ─────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_norm, tgt_norm = normalizer(ref_tensor, tgt_tensor)
    dt = time.perf_counter() - t0
    print(f"[1/5] Normalization       ... done ({dt:.2f}s)")

    # ── Module 2: Chromatic Transport ───────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        color_map = chromatic(ref_raw, tgt_raw)  # (B, 1, H, W)
    dt = time.perf_counter() - t0
    print(f"[2/5] Chromatic Transport ... done ({dt:.2f}s)")

    # ── Module 3: Deep Statistics ───────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        deep_out = deep_stats(ref_norm, tgt_norm)
    dt = time.perf_counter() - t0
    print(f"[3/5] Deep Statistics     ... done ({dt:.2f}s)")

    # ── Module 4: Probabilistic Uncertainty ─────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        anomaly_map = uncertainty(deep_out["residuals"], target_size=(H, W))
    dt = time.perf_counter() - t0
    print(f"[4/5] Uncertainty Mapping ... done ({dt:.2f}s)")

    # ── Module 5: Spatial Heuristics ────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        heur = heuristics(tgt_raw)
    blocking_mask = heur["blocking_mask"]  # (B, 1, H, W)
    ringing_mask = heur["ringing_mask"]    # (B, 1, H, W)
    dt = time.perf_counter() - t0
    print(f"[5/5] Spatial Heuristics  ... done ({dt:.2f}s)")

    # ── Aggregation ─────────────────────────────────────────────────
    # Deep similarity map (higher = more similar)
    deep_sim = aggregate_deep_score(
        deep_out["l_maps"],
        deep_out["s_maps"],
        deep_out["p_tex"],
        target_size=(H, W),
    )  # (B, 1, H, W)

    # Normalize anomaly map to [0, 1] per sample
    anom_flat = anomaly_map.view(B, -1)
    anom_max = anom_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1).clamp(min=1e-12)
    anomaly_norm = anomaly_map / anom_max

    # Normalize color map to [0, 1]
    color_flat = color_map.view(B, -1)
    color_max = color_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1).clamp(min=1e-12)
    color_norm = color_map / color_max

    # Heuristic penalty: fraction of pixels flagged
    heur_penalty = (blocking_mask + ringing_mask).clamp(0.0, 1.0).mean(dim=(2, 3))  # (B, 1)

    # Final score: weighted combination (higher = better)
    score_spatial = (
        w_structure * deep_sim
        - w_anomaly * anomaly_norm
        - w_color * color_norm
    )
    score_per_pixel = score_spatial.mean(dim=(2, 3))  # (B, 1)
    final_score_tensor = score_per_pixel - w_heuristic * heur_penalty
    final_score_tensor = torch.sigmoid(final_score_tensor).squeeze(1)  # (B,)
    final_score = round(float(final_score_tensor[0].item()), 4)

    # ── Build diagnostic tensor (B, 5, H, W) ───────────────────────
    diagnostic = torch.cat(
        [anomaly_norm, color_norm, deep_sim, blocking_mask, ringing_mask],
        dim=1,
    )

    # ── Compute diagnostics ─────────────────────────────────────────
    diagnostics = compute_diagnostics(
        anomaly_norm, color_norm, deep_sim, blocking_mask, ringing_mask
    )

    # ── Print summary ───────────────────────────────────────────────
    print(sep)
    label = score_label(final_score)
    print(f"FR-IQA Score: {final_score} ({label})")
    print(f"Dominant Artifact: {diagnostics['dominant_artifact']}")
    print(f"Affected Area: {diagnostics['affected_area']}%")
    print(sep)

    # ── Save output PNGs ────────────────────────────────────────────
    # Channel 0: Global Anomaly Map (continuous → JET colormap)
    save_channel_as_png(diagnostic, 0, str(out_dir / "global_anomaly_map.png"), use_colormap=True)
    # Channel 1: Color Degradation Map (continuous → JET colormap)
    save_channel_as_png(diagnostic, 1, str(out_dir / "color_degradation_map.png"), use_colormap=True)
    # Channel 2: Structural Similarity Map (continuous → JET colormap)
    save_channel_as_png(diagnostic, 2, str(out_dir / "structural_similarity_map.png"), use_colormap=True)
    # Channel 3: JPEG Blocking Mask (binary → grayscale)
    save_channel_as_png(diagnostic, 3, str(out_dir / "jpeg_blocking_mask.png"), use_colormap=False)
    # Channel 4: Gibbs Ringing Mask (binary → grayscale)
    save_channel_as_png(diagnostic, 4, str(out_dir / "gibbs_ringing_mask.png"), use_colormap=False)

    # ── Save JSON report ────────────────────────────────────────────
    report = {
        "tool": f"UPIQAL FR-IQA v{UPIQAL_VERSION}",
        "timestamp": timestamp.isoformat(),
        "reference_image": str(Path(args.reference).resolve()),
        "target_image": str(Path(args.target).resolve()),
        "output_directory": str(out_dir),
        "image_resolution": {"height": H, "width": W},
        "score": final_score,
        "diagnostics": diagnostics,
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Final output listing ────────────────────────────────────────
    files = sorted(out_dir.iterdir())
    print(f"Saved: {out_dir}/")
    for i, fp in enumerate(files):
        connector = "\u2514\u2500\u2500" if i == len(files) - 1 else "\u251C\u2500\u2500"
        print(f"  {connector} {fp.name}")

    print()


# ======================================================================
# CLI entry point
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="upiqal_cli",
        description=(
            "UPIQAL — Unified Probabilistic Image Quality & Artifact Locator.\n"
            "Runs the full FR-IQA pipeline on a reference/target image pair\n"
            "and saves diagnostic heatmaps (PNG) and a report (JSON)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference (pristine) image.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target (distorted) image.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Custom label for the output directory name.  The directory will be "
            "named upiqal_<name>_YYYYMMDD_HHMMSS.  Defaults to 'results'."
        ),
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Maximum pixel dimension for the longer side (default: 512).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the output directory path entirely.  When set, --name "
            "and the timestamp are ignored."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Validate inputs and run the UPIQAL pipeline."""
    args = parse_args()

    # Validate input files exist
    if not os.path.isfile(args.reference):
        print(f"Error: reference image not found: {args.reference}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.target):
        print(f"Error: target image not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
