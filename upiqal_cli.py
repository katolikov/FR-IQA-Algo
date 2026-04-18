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
    python upiqal_cli.py --reference ref.raw --target tgt.raw --width 640 --height 480 --pixel_format RGB888
    python upiqal_cli.py --reference ref.nv21 --target tgt.nv21 --width 640 --height 480 --pixel_format NV21 --output_format npy
    python upiqal_cli.py --reference ref.nv12 --target tgt.nv12 --width 640 --height 480 --pixel_format NV12

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
from typing import Any, Dict, Optional

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
# Raw / binary format constants
# ======================================================================

_RAW_EXTENSIONS = {".raw", ".bin", ".nv21", ".nv12", ".yuv", ".npy"}
_PIXEL_FORMATS = ("NV21", "NV12", "GRAY8", "RGB888")


# ======================================================================
# Image I/O helpers
# ======================================================================


def _yuv420sp_to_rgb(
    data: bytes, width: int, height: int, chroma_order: str,
) -> np.ndarray:
    """Convert a YUV 4:2:0 semi-planar buffer to an ``(H, W, 3)`` uint8 RGB array.

    Parameters
    ----------
    data : bytes
        Raw bytes of the YUV frame (Y plane followed by interleaved chroma).
    width, height : int
        Frame dimensions in pixels.  Both must be > 0; for sub-sampled chroma
        to round-trip exactly, even dimensions are recommended.
    chroma_order : str
        ``"VU"`` for NV21 (V byte first, U byte second) or ``"UV"`` for NV12
        (U byte first, V byte second).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"YUV420SP requires positive dimensions, got {width}x{height}")
    if chroma_order not in ("VU", "UV"):
        raise ValueError(f"chroma_order must be 'VU' or 'UV', got {chroma_order!r}")

    y_size = width * height
    # Chroma is sub-sampled by 2 in each axis; round up to keep the trailing
    # row/column when the dimensions are odd.
    cw = (width + 1) // 2
    ch = (height + 1) // 2
    uv_size = cw * ch * 2
    expected = y_size + uv_size
    if len(data) < expected:
        fmt_name = "NV21" if chroma_order == "VU" else "NV12"
        raise ValueError(
            f"{fmt_name} data too short: expected {expected} bytes for "
            f"{width}x{height}, got {len(data)}"
        )

    y_plane = (
        np.frombuffer(data, dtype=np.uint8, count=y_size)
        .reshape(height, width)
        .astype(np.float32)
    )
    chroma = np.frombuffer(
        data, dtype=np.uint8, offset=y_size, count=uv_size,
    ).reshape(ch, cw, 2)

    if chroma_order == "VU":
        v_plane = chroma[:, :, 0].astype(np.float32)
        u_plane = chroma[:, :, 1].astype(np.float32)
    else:  # UV (NV12)
        u_plane = chroma[:, :, 0].astype(np.float32)
        v_plane = chroma[:, :, 1].astype(np.float32)

    # Upsample chroma planes to full resolution by pixel-replication, then
    # crop to the requested odd dimensions if necessary.
    u_full = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)[:height, :width]
    v_full = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)[:height, :width]

    # YUV → RGB (BT.601 full-range coefficients used for standard NV21/NV12).
    r = np.clip(y_plane + 1.370705 * (v_full - 128.0), 0, 255)
    g = np.clip(y_plane - 0.337633 * (u_full - 128.0) - 0.698001 * (v_full - 128.0), 0, 255)
    b = np.clip(y_plane + 1.732446 * (u_full - 128.0), 0, 255)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _nv21_to_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert NV21 (YUV420SP, VU interleaved) raw bytes to ``(H, W, 3)`` RGB."""
    return _yuv420sp_to_rgb(data, width, height, chroma_order="VU")


def _nv12_to_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert NV12 (YUV420SP, UV interleaved) raw bytes to ``(H, W, 3)`` RGB."""
    return _yuv420sp_to_rgb(data, width, height, chroma_order="UV")


def _npy_array_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary ``np.load`` result to an ``(H, W, 3)`` uint8 RGB image.

    Handles the shape and dtype variations commonly produced by CV pipelines:

    * 2D ``(H, W)`` grayscale -> tiled to RGB.
    * 3D ``(H, W, 1)`` -> squeezed and tiled to RGB.
    * 3D ``(H, W, 3)`` / ``(H, W, 4)`` -> kept (alpha dropped).
    * 3D ``(3, H, W)`` channels-first -> transposed.
    * 4D leading singleton (``(1, ...)``) -> squeezed and re-evaluated.

    Integer dtypes are rescaled by their full ``iinfo.max`` range; floating
    dtypes have ``NaN``/``Inf`` replaced with zero, and are scaled from
    ``[0, 1]`` if the observed maximum is <= 1.001, otherwise clipped to
    ``[0, 255]``.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f".npy payload is not an ndarray: {type(arr)!r}")
    if arr.size == 0:
        raise ValueError(f".npy array is empty (shape={arr.shape})")

    # ---- Shape normalization --------------------------------------------
    # Squeeze leading singleton batch dimensions like (1, H, W, 3).
    while arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        h0, h1, h2 = arr.shape
        if h2 == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif h2 == 3:
            pass  # already (H, W, 3)
        elif h2 == 4:
            arr = arr[:, :, :3]  # drop alpha
        elif h0 == 3 and h2 != 3:
            # Channels-first (3, H, W) -> (H, W, 3)
            arr = np.transpose(arr, (1, 2, 0))
        elif h0 == 1:
            arr = np.repeat(np.transpose(arr, (1, 2, 0)), 3, axis=2)
        else:
            raise ValueError(
                f".npy array has unsupported 3D shape {arr.shape}; "
                "expected (H, W), (H, W, 1), (H, W, 3/4), or (3, H, W)"
            )
    else:
        raise ValueError(
            f".npy array has unsupported ndim={arr.ndim} (shape={arr.shape}); "
            "expected 2D grayscale or 3D RGB"
        )

    # ---- dtype normalization --------------------------------------------
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_val = max(int(info.max), 1)
        scaled = arr.astype(np.float32) * (255.0 / max_val)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating):
        f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(f.max()) if f.size else 0.0
        if peak <= 1.001:
            f = f * 255.0
        return np.clip(f, 0, 255).astype(np.uint8)

    raise ValueError(f".npy array has unsupported dtype {arr.dtype!r}")


def load_raw_image(
    path: str, width: int, height: int, pixel_format: str,
) -> np.ndarray:
    """Load a raw/binary file and return an ``(H, W, 3)`` uint8 RGB array.

    Supported pixel formats: ``NV21``, ``NV12``, ``GRAY8``, ``RGB888``.
    Also handles ``.npy`` files (auto-detected from extension).  The file
    extensions ``.nv21`` / ``.nv12`` override the explicit *pixel_format*
    argument so that callers don't need to repeat themselves.
    """
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _npy_array_to_rgb_uint8(arr)

    raw_bytes = Path(path).read_bytes()
    fmt = pixel_format.upper()

    # Extension-based override: .nv21/.nv12 are unambiguous.
    if ext == ".nv21":
        fmt = "NV21"
    elif ext == ".nv12":
        fmt = "NV12"

    if fmt == "NV21":
        return _nv21_to_rgb(raw_bytes, width, height)
    if fmt == "NV12":
        return _nv12_to_rgb(raw_bytes, width, height)
    if fmt == "GRAY8":
        expected = width * height
        if expected <= 0:
            raise ValueError(
                f"GRAY8 requires positive width/height, got {width}x{height}"
            )
        if len(raw_bytes) < expected:
            raise ValueError(
                f"GRAY8 data too short: expected {expected} bytes, got {len(raw_bytes)}"
            )
        gray = np.frombuffer(raw_bytes, dtype=np.uint8, count=expected).reshape(height, width)
        return np.stack([gray, gray, gray], axis=-1)
    if fmt == "RGB888":
        expected = width * height * 3
        if expected <= 0:
            raise ValueError(
                f"RGB888 requires positive width/height, got {width}x{height}"
            )
        if len(raw_bytes) < expected:
            raise ValueError(
                f"RGB888 data too short: expected {expected} bytes, got {len(raw_bytes)}"
            )
        return np.frombuffer(raw_bytes, dtype=np.uint8, count=expected).reshape(height, width, 3)

    raise ValueError(
        f"Unsupported pixel_format: {pixel_format!r} (expected one of {_PIXEL_FORMATS})"
    )


def _is_raw_file(path: str) -> bool:
    """Return True if the file extension indicates a raw/binary format."""
    return Path(path).suffix.lower() in _RAW_EXTENSIONS


def _pyramid_target(hw: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Return new (H, W) with aspect preserved and longer side at most max_side."""
    h, w = hw
    scale = max_side / float(max(h, w))
    return (max(1, int(round(h * scale))), max(1, int(round(w * scale))))


def load_image_as_tensor(
    path: str,
    max_side: int = 512,
    width: int = 0,
    height: int = 0,
    pixel_format: str = "RGB888",
) -> torch.Tensor:
    """Load an image file and return a ``(1, 3, H, W)`` float tensor in [0, 1].

    For standard image files (PNG, JPEG, etc.) the *width*/*height*/*pixel_format*
    parameters are ignored.  For raw/binary files (``.raw``, ``.bin``, ``.nv21``,
    ``.npy``) the dimensions and pixel format are required.

    The longer side is resized to at most *max_side* pixels (aspect ratio
    preserved) using Lanczos resampling so that inference stays fast.
    """
    if _is_raw_file(path):
        if not width or not height:
            ext = Path(path).suffix.lower()
            if ext != ".npy":
                raise ValueError(
                    f"--width and --height are required for raw file: {path}"
                )
        rgb = load_raw_image(path, width, height, pixel_format)
        img = Image.fromarray(rgb)
    else:
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


# ======================================================================
# Composite diagnostic overlay ("all artefacts on one image")
# ======================================================================

# Colour per artefact class.  Chosen to be perceptually separable under
# both protanopia and deuteranopia (tested against Coblis / Sim Daltonism).
#                                (R,    G,    B)  uint8
_ARTIFACT_PALETTE: dict[str, tuple[int, int, int]] = {
    "blocking":    (224, 75, 14),    # orange-red   — hot HF grid
    "ringing":     (224, 168, 0),    # amber        — edge proximity
    "noise":       (123, 75, 194),   # purple       — stochastic
    "color_shift": (42,  176, 196),  # cyan         — chromatic
    "blur":        (88,  114, 160),  # blue-gray    — detail loss
}

# Display order for the composite legend (stable, independent of dict order).
_ARTIFACT_ORDER = ("blocking", "ringing", "noise", "color_shift", "blur")


def compose_diagnostic_overlay(
    target_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    threshold: float = 0.05,
    alpha: float = 0.55,
    draw_legend: bool = True,
) -> np.ndarray:
    """Blend the five artefact severity masks into a single RGB overlay.

    Per-pixel argmax picks the **dominant** artefact class; the max
    severity across classes picks the alpha weight.  Pixels whose max
    severity is below ``threshold`` are left untouched so clean regions
    of the target show through.

    Parameters
    ----------
    target_rgb : np.ndarray
        Target image, ``(H, W, 3)`` uint8.
    masks : dict[str, np.ndarray]
        Per-class severity maps in ``[0, 1]``; each entry is an
        ``(H, W)`` float array.  Missing entries are treated as zero
        (class silently skipped in the argmax).  Accepted keys match
        ``_ARTIFACT_ORDER``; extras are ignored.
    threshold : float
        Pixels whose max severity is ≤ this value pass through to the
        target unchanged (default 0.05).
    alpha : float
        Opacity scaling for the coloured overlay (default 0.55 — same as
        ``anomaly_overlay.png``).
    draw_legend : bool
        If ``True``, paint a small legend strip in the top-left corner
        with one colour chip and label per class.

    Returns
    -------
    np.ndarray
        Blended ``(H, W, 3)`` uint8 RGB image.
    """
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError(
            f"target_rgb must be (H, W, 3); got shape {target_rgb.shape}"
        )
    h, w = target_rgb.shape[:2]

    # Stack masks in a fixed class order.  Missing classes are padded
    # with zeros so the argmax still works.
    stack = np.zeros((len(_ARTIFACT_ORDER), h, w), dtype=np.float32)
    for i, name in enumerate(_ARTIFACT_ORDER):
        m = masks.get(name)
        if m is None:
            continue
        if m.shape != (h, w):
            # Resize with bilinear interpolation via PIL so we don't add
            # an OpenCV dependency.
            m = np.array(
                Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)
                                 ).resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        stack[i] = np.clip(m, 0.0, 1.0).astype(np.float32)

    # Per-pixel dominant class + its severity.
    label_map = stack.argmax(axis=0)  # (H, W) int in [0, 4]
    sev_map = stack.max(axis=0)       # (H, W) float in [0, 1]

    # Build an (H, W, 3) RGB layer using the palette of the dominant class.
    colour_lut = np.array(
        [_ARTIFACT_PALETTE[n] for n in _ARTIFACT_ORDER], dtype=np.float32,
    )  # (5, 3)
    colour_map = colour_lut[label_map]  # (H, W, 3) float in [0, 255]

    # Alpha = 0 where severity < threshold, else severity * alpha.
    active = (sev_map > threshold).astype(np.float32)
    alpha_mask = (sev_map * alpha * active)[..., None]  # (H, W, 1)

    blended = (
        (1.0 - alpha_mask) * target_rgb.astype(np.float32)
        + alpha_mask * colour_map
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_legend:
        _draw_legend(blended)
    return blended


def _draw_legend(rgb: np.ndarray) -> None:
    """Paint a small legend in the top-left corner of ``rgb`` in place."""
    from PIL import ImageDraw, ImageFont

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    # Try the system's default TrueType; fall back to the bitmap font.
    try:
        font = ImageFont.truetype("Arial.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()

    pad = 6
    chip_w, chip_h = 12, 12
    line_h = chip_h + 4
    # Measure the longest label to size the background box.
    labels = [n.replace("_", " ") for n in _ARTIFACT_ORDER]
    # Pillow 10+ removed font.getsize; use getbbox instead.
    label_w = max((draw.textbbox((0, 0), lbl, font=font)[2] for lbl in labels), default=60)
    box_w = pad * 2 + chip_w + 4 + label_w
    box_h = pad * 2 + line_h * len(_ARTIFACT_ORDER)
    draw.rectangle([(4, 4), (4 + box_w, 4 + box_h)], fill=(0, 0, 0, 170))
    y = 4 + pad
    for name, lbl in zip(_ARTIFACT_ORDER, labels):
        colour = _ARTIFACT_PALETTE[name]
        draw.rectangle(
            [(4 + pad, y), (4 + pad + chip_w, y + chip_h)],
            fill=colour,
        )
        draw.text(
            (4 + pad + chip_w + 4, y - 1),
            lbl,
            fill=(240, 240, 240, 255),
            font=font,
        )
        y += line_h
    rgb[:] = np.array(img, dtype=np.uint8)


def save_channel(
    tensor: torch.Tensor,
    channel: int,
    filepath: str,
    use_colormap: bool = True,
    output_format: str = "png",
) -> None:
    """Extract one channel from a diagnostic tensor and save in the chosen format.

    Supported *output_format* values:
    - ``png``  — Colormapped RGB PNG image.
    - ``npy``  — Raw float32 NumPy array (normalized [0, 1]).
    - ``raw`` / ``bin`` — Flat float32 binary dump.
    - ``nv21`` — NV21 (YUV420SP) binary from the grayscale channel.
    """
    arr = tensor[0, channel].cpu().numpy()

    # Normalize to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    fmt = output_format.lower()

    if fmt == "png":
        if use_colormap:
            rgb = apply_jet_colormap(arr)
        else:
            rgb = (np.stack([arr] * 3, axis=-1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(filepath, format="PNG")

    elif fmt == "npy":
        np.save(filepath, arr.astype(np.float32))

    elif fmt in ("raw", "bin"):
        arr.astype(np.float32).tofile(filepath)

    elif fmt == "nv21":
        _save_as_nv21(arr, filepath)

    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")


def _save_as_nv21(gray_01: np.ndarray, filepath: str) -> None:
    """Convert a [0,1] grayscale array to NV21 bytes and write to disk.

    The grayscale value is written as the Y plane; U and V are set to 128
    (neutral chroma) so the result is a valid monochrome NV21 frame.
    """
    h, w = gray_01.shape
    # Ensure even dimensions for NV21
    h2 = h if h % 2 == 0 else h - 1
    w2 = w if w % 2 == 0 else w - 1
    y_plane = (gray_01[:h2, :w2] * 255).clip(0, 255).astype(np.uint8)
    uv_plane = np.full((h2 // 2, w2), 128, dtype=np.uint8)  # VU interleaved, neutral
    with open(filepath, "wb") as f:
        f.write(y_plane.tobytes())
        f.write(uv_plane.tobytes())


# Backwards-compatible alias
save_channel_as_png = save_channel


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
# Diagnostics computation — shared with web/main.py:compute_diagnostics
# ======================================================================

# Severity display multipliers
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


def _hf_energy(img: torch.Tensor) -> torch.Tensor:
    """Compute a scalar high-frequency energy proxy per sample.

    Uses Laplacian variance on the luminance channel — a standard
    sharpness / high-frequency-content measure. Larger value → more
    high-frequency detail (sharpness or noise); smaller → smoother
    (blurrier or flatter).

    Parameters
    ----------
    img : torch.Tensor
        Image ``(B, 3, H, W)`` or ``(B, 1, H, W)`` in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Per-sample HF energy ``(B,)``.
    """
    if img.shape[1] == 3:
        w = torch.tensor([0.299, 0.587, 0.114],
                         device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        lum = (img * w).sum(dim=1, keepdim=True)
    else:
        lum = img
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=img.device, dtype=img.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(lum, kernel, padding=1)
    return lap.var(dim=(1, 2, 3))  # (B,)


def compute_diagnostics(
    anomaly_norm: torch.Tensor,
    color_norm: torch.Tensor,
    deep_sim: torch.Tensor,
    blocking_mask: torch.Tensor,
    ringing_mask: torch.Tensor,
    ref_raw: torch.Tensor = None,
    tgt_raw: torch.Tensor = None,
    noise_mask: torch.Tensor = None,
    blur_mask: torch.Tensor = None,
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
    ref_raw, tgt_raw : torch.Tensor, optional
        Reference / target images in ``[0, 1]`` used to compute a high-
        frequency-energy discriminator. When provided (recommended), the
        ``dominant_artifact`` selection is gated so that ``blur`` fires only
        when the target has LESS high-frequency content than the reference,
        and ``noise`` only when it has MORE. Without these, the classic
        max-over-severities logic is used (kept for backward compatibility).

    Returns
    -------
    dict
        Diagnostics dict with ``dominant_artifact``, ``severity_scores``,
        and ``affected_area``.
    """
    # Raw severity percentages (pre-multiplier, per-pixel mean)
    blocking_sev = float(blocking_mask.mean().item()) * 100
    ringing_sev = float(ringing_mask.mean().item()) * 100
    color_sev = float(color_norm.mean().item()) * 100
    # Prefer the dedicated wavelet / edge-spread detectors when available;
    # fall back to the legacy proxies (anomaly_norm / deep_sim) so older
    # callers still get a sane answer.
    if noise_mask is not None:
        noise_sev = float(noise_mask.mean().item()) * 100
    else:
        noise_sev = float(anomaly_norm.mean().item()) * 100
    if blur_mask is not None:
        blur_sev = float(blur_mask.mean().item()) * 100
    else:
        blur_sev = float((1.0 - deep_sim).mean().item()) * 100

    # Display severities: scale by perceptual multipliers and clamp at 100.
    # These numbers go into the report/UI and are NOT used for argmax below.
    severity_scores = {
        "blocking":    round(min(blocking_sev * _SEVERITY_MULTIPLIERS["blocking"], 100.0), 1),
        "ringing":     round(min(ringing_sev * _SEVERITY_MULTIPLIERS["ringing"], 100.0), 1),
        "noise":       round(min(noise_sev * _SEVERITY_MULTIPLIERS["noise"], 100.0), 1),
        "color_shift": round(min(color_sev * _SEVERITY_MULTIPLIERS["color_shift"], 100.0), 1),
        "blur":        round(min(blur_sev * _SEVERITY_MULTIPLIERS["blur"], 100.0), 1),
    }

    # ── Dominant-artifact selection ────────────────────────────────
    # Strategy: rank candidates by their CONTRIBUTION to the score drop
    # (raw_severity × pipeline_weight), not by the clamped display value.
    # This keeps the top-1 meaningful when several display channels saturate
    # at 100. Then apply a high-frequency-energy discriminator so generic
    # VGG dissimilarity isn't mislabeled as "blur" when the target is
    # actually SHARPER than the reference (noise / ringing / different
    # content) — see bugs #2 & #5 in analysis/report.md.
    #
    # Weights mirror those in run_pipeline (w_anomaly=0.3, w_color=0.1,
    # w_structure=0.5, w_heuristic=0.1 shared across blocking+ringing).
    # Contribution weights for dominant-artifact ranking.
    # Heuristic masks (blocking, ringing) are high-specificity evidence: when
    # they fire at all, they almost certainly indicate the named artifact,
    # so they get the largest multipliers here (even though their
    # contribution to the final numerical score is small, w_heuristic=0.1).
    # Color/blur/noise are generic deep-feature signals that overlap, so
    # they're weighted closer to their pipeline weights.
    contrib = {
        "blocking":    blocking_sev * 1.0,   # high specificity
        "ringing":     ringing_sev  * 1.0,   # high specificity
        "noise":       noise_sev    * 0.30,
        "color_shift": color_sev    * 0.30,  # raised so hue shift wins over generic VGG drift
        "blur":        blur_sev     * 0.50,
    }

    # HF-energy discriminator: only trust "blur" when tgt is smoother than
    # ref; only trust "noise" when tgt is rougher. Prevents generic
    # deep-similarity drop on unrelated images from being mislabeled as
    # "blur" (bug #5).
    if ref_raw is not None and tgt_raw is not None:
        with torch.no_grad():
            hf_ref = float(_hf_energy(ref_raw).mean().item())
            hf_tgt = float(_hf_energy(tgt_raw).mean().item())
        eps = 1e-8
        hf_ratio = hf_tgt / (hf_ref + eps)
        # 0.9 / 1.1 deadband is ~10% HF change; beyond that the direction is decisive.
        if hf_ratio > 1.1:
            # Target has MORE high-freq → not blur (noise / ringing territory)
            contrib["blur"] *= 0.0
            # Noise perturbs pixel-wise colors enough to fully saturate the
            # Sinkhorn EMD channel; downweight color_shift so noise can win.
            if noise_sev > 10.0:
                contrib["color_shift"] *= 0.3
        elif hf_ratio < 0.9:
            # Target is smoother → not noise / ringing (blur territory)
            contrib["noise"]   *= 0.3
            contrib["ringing"] *= 0.3
        else:
            # HF stable → spatial detail preserved.  If color EMD is high,
            # the delta is dominated by chromatic change, NOT blur.
            if color_sev >= 25.0:
                contrib["blur"] *= 0.3

    # Guard: if all contributions are negligible, report "None".
    # MPS / float32 drift can leave a single channel mildly non-zero on
    # near-identical inputs (e.g. deep_sim ≈ 0.88 → blur_sev ≈ 12%).
    # Treat that as device noise, not a real artifact.
    nonzero = {k: v for k, v in severity_scores.items() if v > 0.5}
    single_weak = (
        len(nonzero) == 1
        and next(iter(nonzero.values())) < 15.0
    )
    if max(contrib.values()) < 0.5 or single_weak:
        dominant_artifact = "None"
    else:
        dominant_key = max(contrib, key=contrib.get)
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
    raw_kw = dict(
        width=getattr(args, "width", 0) or 0,
        height=getattr(args, "height", 0) or 0,
        pixel_format=getattr(args, "pixel_format", "RGB888") or "RGB888",
    )
    # Multi-scale pyramid (Phase 1 of the paper): load at FULL resolution,
    # then downsample a separate copy for the deep-feature branch. The
    # heuristics (blocking/ringing/noise/blur) run on the original pixels
    # so grid-aligned artefacts aren't blurred away; Modules 1-4 use the
    # 256-ish-sided pyramid level for speed and to match the paper's
    # "minimum dimension ~ 256 px for standard feature extraction".
    # Opt out via --no-pyramid to restore single-scale behaviour.
    use_pyramid = bool(getattr(args, "pyramid", True))
    feature_side = int(getattr(args, "feature_side", 256))

    if use_pyramid:
        ref_full = load_image_as_tensor(args.reference, max_side=args.max_side, **raw_kw)
        tgt_full = load_image_as_tensor(args.target, max_side=args.max_side, **raw_kw)
        # Deep-feature copy: aggressively shrunk so VGG16 runs fast and
        # sees the same effective spatial frequencies as in the paper.
        ref_tensor = (
            F.interpolate(
                ref_full, size=_pyramid_target(ref_full.shape[-2:], feature_side),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
            if max(ref_full.shape[-2:]) > feature_side
            else ref_full
        )
        tgt_tensor = (
            F.interpolate(
                tgt_full, size=_pyramid_target(tgt_full.shape[-2:], feature_side),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
            if max(tgt_full.shape[-2:]) > feature_side
            else tgt_full
        )
    else:
        ref_full = load_image_as_tensor(args.reference, max_side=args.max_side, **raw_kw)
        tgt_full = load_image_as_tensor(args.target, max_side=args.max_side, **raw_kw)
        ref_tensor = ref_full
        tgt_tensor = tgt_full

    # Ensure matching spatial dimensions.
    #
    # Previously this bilinear-upsampled the target to the reference size,
    # which silently injected blur into the target and caused false "blur"/
    # "color shift" severity on same-content pairs at mismatched native
    # sizes (validation probe T7: score 0.95 -> 0.84).
    #
    # Fix: pick the SMALLER resolution (preserves sharpness of the lower-
    # res image; never upsamples) and downsample the larger one via bicubic
    # with antialiasing. If both already match, this is a no-op.
    # Reconcile ref/tgt spatial sizes at BOTH pyramid levels so the rest
    # of the pipeline sees matching shapes.
    def _match(a: torch.Tensor, b: torch.Tensor):
        _, _, ah, aw = a.shape
        _, _, bh, bw = b.shape
        if (ah, aw) == (bh, bw):
            return a, b
        ch, cw = min(ah, bh), min(aw, bw)
        if (ah, aw) != (ch, cw):
            a = F.interpolate(
                a, size=(ch, cw), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
        if (bh, bw) != (ch, cw):
            b = F.interpolate(
                b, size=(ch, cw), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
        return a, b

    ref_tensor, tgt_tensor = _match(ref_tensor, tgt_tensor)
    ref_full, tgt_full = _match(ref_full, tgt_full)

    B, C, H, W = ref_tensor.shape
    print(f"Resolution: {H} x {W}")
    print(sep)

    # ── Initialize modules (before creating output dir, so failures
    #    like VGG16 download errors don't leave empty directories) ───
    timestamp = datetime.now()
    normalizer = Normalizer(mode="imagenet")
    chromatic = ChromaticTransportEvaluator(patch_size=16, sinkhorn_iters=20)

    # Deep feature extractor — try pretrained from local weights, fallback to random init
    try:
        deep_stats = DeepStatisticalExtractor(pretrained=True)
        vgg_status = "ImageNet (pretrained, local)"
    except FileNotFoundError:
        print("  [warn] Local VGG16 weights not found. Run: python weights/download_vgg16.py")
        deep_stats = DeepStatisticalExtractor(pretrained=False)
        vgg_status = "random initialization (pretrained weights unavailable)"
    print(f"  VGG16 weights: {vgg_status}")

    uncertainty_weights_path = getattr(args, "uncertainty_weights", None)
    if uncertainty_weights_path:
        uncertainty = ProbabilisticUncertaintyMapper(parameterization="blockdiag")
        state = torch.load(
            uncertainty_weights_path, map_location="cpu", weights_only=True
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        uncertainty.load_state_dict(state, strict=False)
        uncertainty.eval()
        print(f"  Uncertainty weights: {uncertainty_weights_path} (blockdiag, loaded)")
    else:
        uncertainty = ProbabilisticUncertaintyMapper()
        print("  Uncertainty weights: identity diagonal (untrained)")
    heuristics = SpatialHeuristicsEngine()

    # Aggregation weights (defaults from upiqal/model.py).  Optionally
    # overridden by a MOS-tuned ckpt produced by train_aggregation.py.
    w_color = 0.1
    w_anomaly = 0.3
    w_structure = 0.5
    w_heuristic = 0.1
    score_scale = 10.0
    score_center = 0.2
    sigmoid_gain_override: Optional[float] = None
    sigmoid_bias_override: Optional[float] = None
    aggregation_weights_path = getattr(args, "aggregation_weights", None)
    if aggregation_weights_path:
        agg_ckpt = torch.load(
            aggregation_weights_path, map_location="cpu", weights_only=True
        )
        agg = agg_ckpt.get("parameters", agg_ckpt) if isinstance(agg_ckpt, dict) else {}
        w_color = float(agg.get("w_color", w_color))
        w_anomaly = float(agg.get("w_anomaly", w_anomaly))
        w_structure = float(agg.get("w_structure", w_structure))
        w_heuristic = float(agg.get("w_heuristic", w_heuristic))
        score_scale = float(agg.get("score_scale", score_scale))
        score_center = float(agg.get("score_center", score_center))
        if "sigmoid_gain" in agg:
            sigmoid_gain_override = float(agg["sigmoid_gain"])
        if "sigmoid_bias" in agg:
            sigmoid_bias_override = float(agg["sigmoid_bias"])
        print(
            f"  Aggregation weights: {aggregation_weights_path} (MOS-tuned)"
        )
    else:
        print("  Aggregation weights: hand-tuned defaults")

    # Apply optional A-DISTS sigmoid overrides to the already-built
    # deep_stats module.
    if sigmoid_gain_override is not None and hasattr(deep_stats, "sigmoid_gain"):
        with torch.no_grad():
            if isinstance(deep_stats.sigmoid_gain, torch.nn.Parameter):
                deep_stats.sigmoid_gain.copy_(torch.tensor(sigmoid_gain_override))
    if sigmoid_bias_override is not None and hasattr(deep_stats, "sigmoid_bias"):
        with torch.no_grad():
            if isinstance(deep_stats.sigmoid_bias, torch.nn.Parameter):
                deep_stats.sigmoid_bias.copy_(torch.tensor(sigmoid_bias_override))

    # ── Create output directory (deferred until modules are ready) ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        tag = args.name if args.name else "results"
        dir_name = f"upiqal_{tag}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path(dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep raw copies for modules that operate on [0, 1] images.
    # ref_raw / tgt_raw are the pyramid-feature level (used by Modules 2-4
    # for chromatic transport and the HF-energy discriminator at the
    # diagnostic stage).  ref_full / tgt_full are the original-resolution
    # pixels fed to Module 5 (heuristics), which need pixel-perfect grid
    # alignment for blocking and edge-exact locations for ringing.
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
        color_map_raw = chromatic(ref_raw, tgt_raw)  # (B, 1, H, W)
        # Subtract self-transport baseline: Sinkhorn regularization produces
        # non-zero EMD even for identical distributions.  Subtracting the
        # reference self-EMD removes this bias.
        color_baseline = chromatic(ref_raw, ref_raw)  # (B, 1, H, W)
        color_map = (color_map_raw - color_baseline).clamp(min=0.0)
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

    # ── Module 5: Spatial Heuristics (at FULL resolution) ──────────
    # The paper's Phase 1 explicitly keeps heuristic analysis at native
    # resolution so grid phases and ringing near edges aren't blurred.
    t0 = time.perf_counter()
    with torch.no_grad():
        heur = heuristics(ref_full, tgt_full)
    # Downsample masks to pipeline resolution for aggregation with the
    # other modules' maps.
    def _to_pipe(m: torch.Tensor) -> torch.Tensor:
        if m.shape[-2:] == (H, W):
            return m
        return F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    blocking_mask = _to_pipe(heur["blocking_mask"])
    ringing_mask = _to_pipe(heur["ringing_mask"])
    noise_mask = _to_pipe(heur["noise_mask"])
    blur_mask = _to_pipe(heur["blur_mask"])
    dt = time.perf_counter() - t0
    print(f"[5/5] Spatial Heuristics  ... done ({dt:.2f}s)")

    # ── Aggregation ─────────────────────────────────────────────────
    # Deep similarity map (higher = more similar).  Clamp to [0, 1] so the
    # downstream blur severity (1 - deep_sim) cannot drift outside its
    # documented range.
    deep_sim = aggregate_deep_score(
        deep_out["l_maps"],
        deep_out["s_maps"],
        deep_out["p_tex"],
        target_size=(H, W),
    ).clamp(0.0, 1.0)  # (B, 1, H, W)

    # Per-sample normalisation that returns zero for near-identical images
    # instead of dividing by a tiny clamp (avoids NaN propagation).
    def _safe_norm(x: torch.Tensor) -> torch.Tensor:
        flat = x.view(B, -1)
        peak = flat.max(dim=1, keepdim=True).values  # (B, 1)
        safe_peak = peak.clamp(min=1e-3).view(B, 1, 1, 1)
        out = x / safe_peak
        is_small = (peak < 1e-3).view(B, 1, 1, 1)
        return torch.where(is_small, torch.zeros_like(out), out)

    anomaly_norm = _safe_norm(anomaly_map)
    color_norm = _safe_norm(color_map)

    # Heuristic penalty: fraction of pixels flagged
    heur_penalty = (blocking_mask + ringing_mask).clamp(0.0, 1.0).mean(dim=(2, 3))  # (B, 1)

    # Final score: either the legacy sigmoid calibration or the paper's
    # negative-log-likelihood mapping.  Toggle via --score-mode.
    score_mode = getattr(args, "score_mode", "sigmoid")
    if score_mode == "sigmoid":
        score_spatial = (
            w_structure * deep_sim
            - w_anomaly * anomaly_norm
            - w_color * color_norm
        )
        score_per_pixel = score_spatial.mean(dim=(2, 3))  # (B, 1)
        final_score_tensor = (
            score_scale * (score_per_pixel - score_center)
            - w_heuristic * heur_penalty
        )
        final_score_tensor = torch.sigmoid(final_score_tensor).squeeze(1)  # (B,)
    else:  # "nll"
        nll_spatial = (
            w_anomaly * anomaly_norm
            + w_color * color_norm
            + w_structure * (1.0 - deep_sim)
        )
        total_nll = nll_spatial.mean(dim=(2, 3)) + w_heuristic * heur_penalty
        final_score_tensor = torch.exp(-score_scale * total_nll).squeeze(1)
        final_score_tensor = final_score_tensor.clamp(0.0, 1.0)
    final_score = round(float(final_score_tensor[0].item()), 4)

    # ── Build diagnostic tensor (B, 7, H, W) ───────────────────────
    diagnostic = torch.cat(
        [
            anomaly_norm, color_norm, deep_sim,
            blocking_mask, ringing_mask,
            noise_mask, blur_mask,
        ],
        dim=1,
    )

    # ── Compute diagnostics ─────────────────────────────────────────
    diagnostics = compute_diagnostics(
        anomaly_norm, color_norm, deep_sim, blocking_mask, ringing_mask,
        ref_raw=ref_raw, tgt_raw=tgt_raw,
        noise_mask=noise_mask, blur_mask=blur_mask,
    )

    # ── Print summary ───────────────────────────────────────────────
    print(sep)
    label = score_label(final_score)
    print(f"FR-IQA Score: {final_score} ({label})")
    print(f"Dominant Artifact: {diagnostics['dominant_artifact']}")
    print(f"Affected Area: {diagnostics['affected_area']}%")
    print(sep)

    # ── Save output maps ─────────────────────────────────────────────
    out_fmt = getattr(args, "output_format", "png") or "png"
    ext = out_fmt if out_fmt not in ("raw", "bin") else out_fmt
    ext_dot = f".{ext}"

    channel_info = [
        (0, "global_anomaly_map",         True),
        (1, "color_degradation_map",      True),
        (2, "structural_similarity_map",  True),
        (3, "jpeg_blocking_mask",         False),
        (4, "gibbs_ringing_mask",         False),
        (5, "gaussian_noise_mask",        True),
        (6, "blur_mask",                  True),
    ]
    for ch_idx, name, use_cmap in channel_info:
        save_channel(
            diagnostic, ch_idx,
            str(out_dir / f"{name}{ext_dot}"),
            use_colormap=use_cmap,
            output_format=out_fmt,
        )

    # ── Generate anomaly overlay at original resolution ─────────────
    alpha = 0.6  # heatmap opacity; 0.0 = target only, 1.0 = heatmap only

    # Load target image at its full native resolution
    if _is_raw_file(args.target):
        _orig_rgb = load_raw_image(
            args.target,
            raw_kw["width"], raw_kw["height"], raw_kw["pixel_format"],
        )
        orig_img = Image.fromarray(_orig_rgb)
    else:
        orig_img = Image.open(args.target).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Get the single-channel anomaly map and upsample to original resolution
    anom_np = anomaly_norm[0, 0].cpu().numpy()  # (H, W) in [0, 1]
    anom_full = np.array(
        Image.fromarray((anom_np * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0  # back to [0, 1]

    # Apply JET colormap and alpha-blend onto the original target
    heatmap_rgb = apply_jet_colormap(anom_full)  # (orig_h, orig_w, 3) uint8
    target_rgb = np.array(orig_img, dtype=np.float32)
    blended = (1.0 - alpha) * target_rgb + alpha * heatmap_rgb.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(str(out_dir / "anomaly_overlay.png"), format="PNG")

    # ── Composite diagnostic overlay (per-pixel dominant artefact) ──
    # Pulls all five artefact severity maps together and paints the
    # dominant class in its palette colour over the target image.  One
    # glance tells the user which artefacts dominate which regions.
    composite_masks_small = {
        "blocking":    blocking_mask[0, 0].cpu().numpy(),
        "ringing":     ringing_mask[0, 0].cpu().numpy(),
        "noise":       noise_mask[0, 0].cpu().numpy(),
        "color_shift": color_norm[0, 0].cpu().numpy(),
        "blur":        blur_mask[0, 0].cpu().numpy(),
    }
    composite = compose_diagnostic_overlay(
        np.array(orig_img, dtype=np.uint8),
        composite_masks_small,
        threshold=0.05,
        alpha=0.55,
        draw_legend=True,
    )
    Image.fromarray(composite).save(
        str(out_dir / "diagnostic_overlay.png"), format="PNG"
    )

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
        default=1024,
        help=(
            "Maximum pixel dimension for the longer side (default: 1024). "
            "Larger values preserve localized artifacts (e.g., 8x8 blocking, "
            "fine-scale ringing) at the cost of longer runtime. Values in "
            "[512, 2048] are typical."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the output directory path entirely.  When set, --name "
            "and the timestamp are ignored."
        ),
    )

    # --- Raw / binary input format options ---
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Image width in pixels (required for .raw/.bin/.nv21 inputs).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Image height in pixels (required for .raw/.bin/.nv21 inputs).",
    )
    parser.add_argument(
        "--pixel_format",
        default="RGB888",
        choices=list(_PIXEL_FORMATS),
        help="Pixel format of raw inputs (default: RGB888).",
    )

    # --- Output format ---
    parser.add_argument(
        "--output_format",
        default="png",
        choices=["png", "npy", "raw", "bin", "nv21"],
        help="Output format for diagnostic heatmaps (default: png).",
    )

    # --- Learned Cholesky factor (optional) ---
    parser.add_argument(
        "--uncertainty-weights",
        dest="uncertainty_weights",
        default=None,
        help=(
            "Path to a trained Cholesky checkpoint for Module 4 "
            "(produced by train_uncertainty.py). If omitted, the "
            "identity-diagonal baseline is used."
        ),
    )
    parser.add_argument(
        "--aggregation-weights",
        dest="aggregation_weights",
        default=None,
        help=(
            "Path to a MOS-tuned aggregation checkpoint produced by "
            "train_aggregation.py (8 scalars: the six pipeline weights "
            "plus A-DISTS sigmoid k/b). Overrides the hand-tuned defaults."
        ),
    )

    # --- Multi-scale pyramid (Phase 1 of the article) ---
    parser.add_argument(
        "--no-pyramid",
        dest="pyramid",
        action="store_false",
        default=True,
        help=(
            "Disable the multi-scale pyramid and run all modules at the "
            "same resolution (legacy behaviour). By default the "
            "heuristics (Module 5) run on the full-resolution image "
            "while Modules 1-4 use the --feature-side downsample."
        ),
    )
    parser.add_argument(
        "--feature-side",
        dest="feature_side",
        type=int,
        default=256,
        help=(
            "Maximum longer-side (in pixels) of the deep-feature "
            "pyramid level used by Modules 1-4 (default 256; paper "
            "default)."
        ),
    )

    # --- Score aggregation mode (Module-agnostic) ---
    parser.add_argument(
        "--score-mode",
        dest="score_mode",
        choices=["sigmoid", "nll"],
        default="sigmoid",
        help=(
            "Final-score normalisation formula. 'sigmoid' (default) "
            "matches the hand-calibrated legacy score; 'nll' is the "
            "paper-prescribed negative-log-likelihood mapping "
            "exp(-score_scale * total_nll)."
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

    # Validate raw format requirements
    for label, fpath in [("reference", args.reference), ("target", args.target)]:
        if _is_raw_file(fpath) and Path(fpath).suffix.lower() != ".npy":
            if not args.width or not args.height:
                print(
                    f"Error: --width and --height are required for {label} "
                    f"raw file: {fpath}",
                    file=sys.stderr,
                )
                sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
