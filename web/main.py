"""FastAPI server for the UPIQAL web application.

Serves the frontend and exposes a POST /api/compare endpoint that accepts
two uploaded images, runs the real UPIQAL model (with pretrained VGG16),
and returns the FR-IQA score plus Base64-encoded diagnostic heatmaps.

The heavy VGG16 model is loaded exactly once at startup (moved to CUDA if
available) and reused for every request.

Usage (from repo root):
    uvicorn main:app --app-dir web --reload --port 8000

Usage (from web/ directory):
    cd web
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

# Ensure the upiqal package (at repo root) is importable when running from web/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from upiqal import UPIQAL  # noqa: E402

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(title="UPIQAL - Image Quality Analyzer")

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Select the best available device (CUDA > MPS > CPU)
_device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# Model singleton (loaded once at startup, never reloaded per-request)
_model: Optional[UPIQAL] = None


def get_model() -> UPIQAL:
    """Return the global UPIQAL model, initialising it on first call.

    The pretrained VGG16 weights are downloaded on the very first run
    (~528 MB) and cached by torchvision.  The full model is moved to
    the best available device and set to eval mode.
    """
    global _model
    if _model is None:
        _model = UPIQAL(pretrained_vgg=True)
        _model.to(_device)
        _model.eval()
        print(f"UPIQAL model loaded on {_device}")
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nv21_to_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert NV21 (YUV420SP) raw bytes to ``(H, W, 3)`` uint8 RGB."""
    y_size = width * height
    uv_size = y_size // 2
    if len(data) < y_size + uv_size:
        raise ValueError(
            f"NV21 data too short: expected {y_size + uv_size} bytes "
            f"for {width}x{height}, got {len(data)}"
        )
    y = np.frombuffer(data, np.uint8, count=y_size).reshape(height, width).astype(np.float32)
    vu = np.frombuffer(data, np.uint8, offset=y_size, count=uv_size).reshape(height // 2, width // 2, 2)
    v = np.repeat(np.repeat(vu[:, :, 0].astype(np.float32), 2, 0), 2, 1)[:height, :width]
    u = np.repeat(np.repeat(vu[:, :, 1].astype(np.float32), 2, 0), 2, 1)[:height, :width]
    r = np.clip(y + 1.370705 * (v - 128), 0, 255)
    g = np.clip(y - 0.337633 * (u - 128) - 0.698001 * (v - 128), 0, 255)
    b = np.clip(y + 1.732446 * (u - 128), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _decode_raw_bytes(
    data: bytes,
    width: int,
    height: int,
    pixel_format: str,
    filename: str = "",
) -> np.ndarray:
    """Decode raw/binary upload bytes to ``(H, W, 3)`` uint8 RGB.

    Handles ``.npy`` (by extension in *filename*), NV21, GRAY8, RGB888.
    """
    ext = Path(filename).suffix.lower() if filename else ""

    if ext == ".npy":
        arr = np.load(io.BytesIO(data))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
        return arr

    fmt = pixel_format.upper()
    if fmt == "NV21" or ext == ".nv21":
        return _nv21_to_rgb(data, width, height)
    elif fmt == "GRAY8":
        gray = np.frombuffer(data, np.uint8, count=width * height).reshape(height, width)
        return np.stack([gray, gray, gray], axis=-1)
    elif fmt == "RGB888":
        return np.frombuffer(data, np.uint8, count=width * height * 3).reshape(height, width, 3)
    else:
        raise ValueError(f"Unsupported pixel_format: {pixel_format!r}")


_RAW_EXTENSIONS = {".raw", ".bin", ".nv21", ".npy"}


def read_image_as_tensor(
    data: bytes,
    max_side: int = 512,
    width: int = 0,
    height: int = 0,
    pixel_format: str = "RGB888",
    filename: str = "",
) -> tuple[torch.Tensor, int, int]:
    """Decode uploaded bytes to a ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    For standard image formats (PNG, JPEG) the raw-format parameters are
    ignored.  For raw/binary uploads the *width*, *height*, and
    *pixel_format* are used to interpret the byte stream.

    Returns
    -------
    tuple[torch.Tensor, int, int]
        The image tensor and the original (height, width) before any resizing.
    """
    ext = Path(filename).suffix.lower() if filename else ""
    is_raw = ext in _RAW_EXTENSIONS

    if is_raw:
        rgb = _decode_raw_bytes(data, width, height, pixel_format, filename)
        img = Image.fromarray(rgb)
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")

    # Store original dimensions before any resizing
    orig_w, orig_h = img.size

    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, orig_h, orig_w


def _apply_colormap(single_channel: np.ndarray) -> np.ndarray:
    """Apply a JET-like colormap to a [0, 1] grayscale array.

    Returns an ``(H, W, 3)`` uint8 RGB image.  Uses a simple manual
    interpolation to avoid a matplotlib import at runtime.
    """
    x = np.clip(single_channel, 0.0, 1.0)

    # Simple JET approximation via piecewise linear
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def tensor_channel_to_base64(
    tensor: torch.Tensor,
    channel: int,
    colormap: bool = True,
) -> str:
    """Extract one channel from a ``(1, C, H, W)`` tensor, render it as a
    PNG, and return the Base64-encoded string.
    """
    arr = tensor[0, channel].cpu().numpy()
    # Normalize to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    if colormap:
        rgb = _apply_colormap(arr)
    else:
        rgb = (np.stack([arr] * 3, axis=-1) * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a ``(1, 3, H, W)`` image tensor in [0,1] to Base64 PNG."""
    arr = tensor[0].cpu().permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Diagnostics computation
# ---------------------------------------------------------------------------

# Severity display multipliers (matching the original diagnostics output)
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


def compute_diagnostics(diag: torch.Tensor) -> Dict[str, Any]:
    """Derive severity scores and dominant artifact from the diagnostic tensor.

    Parameters
    ----------
    diag : torch.Tensor
        The 5-channel diagnostic tensor ``(1, 5, H, W)`` produced by
        ``UPIQAL.forward()``.  Channels:
        0 = anomaly, 1 = color, 2 = structure, 3 = blocking, 4 = ringing.

    Returns
    -------
    dict
        ``dominant_artifact``, ``severity_scores``, ``affected_area``.
    """
    anomaly = diag[0, 0]
    color_map = diag[0, 1]
    structure = diag[0, 2]
    blocking = diag[0, 3]
    ringing = diag[0, 4]

    blocking_sev = float(blocking.mean().item()) * 100
    ringing_sev = float(ringing.mean().item()) * 100
    noise_sev = float(anomaly.mean().item()) * 100
    color_sev = float(color_map.mean().item()) * 100
    blur_sev = float((1.0 - structure).mean().item()) * 100

    severity_scores = {
        "blocking": round(min(blocking_sev * _SEVERITY_MULTIPLIERS["blocking"], 100.0), 1),
        "ringing": round(min(ringing_sev * _SEVERITY_MULTIPLIERS["ringing"], 100.0), 1),
        "noise": round(min(noise_sev * _SEVERITY_MULTIPLIERS["noise"], 100.0), 1),
        "color_shift": round(min(color_sev * _SEVERITY_MULTIPLIERS["color_shift"], 100.0), 1),
        "blur": round(min(blur_sev * _SEVERITY_MULTIPLIERS["blur"], 100.0), 1),
    }

    dominant_key = max(severity_scores, key=severity_scores.get)
    dominant_artifact = _ARTIFACT_LABELS[dominant_key]

    affected_mask = (anomaly > 0.15).float()
    affected_area = round(float(affected_mask.mean().item()) * 100, 1)

    return {
        "dominant_artifact": dominant_artifact,
        "severity_scores": severity_scores,
        "affected_area": affected_area,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/compare")
async def compare(
    reference_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    width: int = Form(0),
    height: int = Form(0),
    pixel_format: str = Form("RGB888"),
    output_format: str = Form("png"),
):
    """Compare two images using the real UPIQAL model.

    Accepts multipart/form-data with ``reference_image`` and
    ``target_image`` fields, plus optional ``width``, ``height``,
    ``pixel_format``, and ``output_format`` for raw/binary inputs.

    Returns JSON with score, heatmaps, overlay, target image, and
    diagnostic statistics (dominant artifact, severity scores, affected area).
    """
    ref_bytes = await reference_image.read()
    tgt_bytes = await target_image.read()

    raw_kw = dict(
        width=width,
        height=height,
        pixel_format=pixel_format,
        filename=reference_image.filename or "",
    )
    ref_tensor, orig_h, orig_w = read_image_as_tensor(ref_bytes, **raw_kw)
    raw_kw["filename"] = target_image.filename or ""
    tgt_tensor, _, _ = read_image_as_tensor(tgt_bytes, **raw_kw)

    # Ensure same spatial dimensions (resize target to match reference)
    _, _, rh, rw = ref_tensor.shape
    _, _, th, tw = tgt_tensor.shape
    if (rh, rw) != (th, tw):
        tgt_tensor = F.interpolate(
            tgt_tensor, size=(rh, rw), mode="bilinear", align_corners=False
        )

    # Move tensors to model device
    model = get_model()
    ref_on_device = ref_tensor.to(_device)
    tgt_on_device = tgt_tensor.to(_device)

    # Run the real UPIQAL pipeline (all 5 modules)
    result = model(ref_on_device, tgt_on_device)

    score = float(result["score"][0].item())
    diag = result["diagnostic_tensor"]  # (1, 5, H, W)

    # Upsample diagnostic tensor and target to original input resolution
    if diag.shape[2:] != (orig_h, orig_w):
        diag = F.interpolate(
            diag, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
    if tgt_tensor.shape[2:] != (orig_h, orig_w):
        tgt_tensor = F.interpolate(
            tgt_tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    # Compute diagnostics from the diagnostic tensor
    diagnostics = compute_diagnostics(diag)

    # Encode heatmap channels as Base64 PNGs
    channel_names = ["anomaly", "color", "structure", "blocking", "ringing"]
    heatmaps = {}
    for i, name in enumerate(channel_names):
        use_colormap = name not in ("blocking", "ringing")
        heatmaps[name] = tensor_channel_to_base64(diag, i, colormap=use_colormap)

    # Create an overlay: anomaly heatmap blended onto the target image
    anomaly_arr = diag[0, 0].cpu().numpy()
    lo, hi = anomaly_arr.min(), anomaly_arr.max()
    if hi - lo > 1e-8:
        anomaly_arr = (anomaly_arr - lo) / (hi - lo)
    else:
        anomaly_arr = np.zeros_like(anomaly_arr)
    heatmap_rgb = _apply_colormap(anomaly_arr).astype(np.float32) / 255.0

    tgt_arr = tgt_tensor[0].cpu().permute(1, 2, 0).numpy()
    blended = np.clip(0.6 * tgt_arr + 0.4 * heatmap_rgb, 0, 1)
    blended_img = Image.fromarray((blended * 255).astype(np.uint8))
    buf = io.BytesIO()
    blended_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Target image as base64 (for opacity slider overlay in the frontend)
    target_b64 = tensor_to_base64(tgt_tensor)

    # Store diagnostic tensor for binary download endpoint
    _store_last_diagnostic(diag)

    return {
        "score": round(score, 4),
        "heatmaps": heatmaps,
        "overlay": overlay_b64,
        "target": target_b64,
        "diagnostics": diagnostics,
        "output_format": output_format,
    }


# ---------------------------------------------------------------------------
# Binary download support
# ---------------------------------------------------------------------------

_last_diagnostic: Optional[torch.Tensor] = None


def _store_last_diagnostic(diag: torch.Tensor) -> None:
    """Cache the most recent diagnostic tensor for download requests."""
    global _last_diagnostic
    _last_diagnostic = diag.cpu()


_CHANNEL_NAMES = ["anomaly", "color", "structure", "blocking", "ringing"]


@app.get("/api/download/{mask_name}")
async def download_mask(
    mask_name: str,
    fmt: str = "npy",
):
    """Download a diagnostic channel in a non-PNG binary format.

    Supported *fmt* values: ``npy``, ``raw``, ``bin``, ``nv21``.
    """
    from fastapi.responses import Response

    if _last_diagnostic is None:
        return Response(content="No analysis results available", status_code=404)

    if mask_name not in _CHANNEL_NAMES:
        return Response(content=f"Unknown mask: {mask_name}", status_code=400)

    ch_idx = _CHANNEL_NAMES.index(mask_name)
    arr = _last_diagnostic[0, ch_idx].numpy()

    # Normalize to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    fmt = fmt.lower()
    if fmt == "npy":
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32))
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="upiqal-{mask_name}.npy"'},
        )
    elif fmt in ("raw", "bin"):
        return Response(
            content=arr.astype(np.float32).tobytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="upiqal-{mask_name}.{fmt}"'},
        )
    elif fmt == "nv21":
        h, w = arr.shape
        h2, w2 = h - h % 2, w - w % 2
        y_plane = (arr[:h2, :w2] * 255).clip(0, 255).astype(np.uint8)
        uv_plane = np.full((h2 // 2, w2), 128, dtype=np.uint8)
        return Response(
            content=y_plane.tobytes() + uv_plane.tobytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="upiqal-{mask_name}.nv21"'},
        )
    else:
        return Response(content=f"Unsupported format: {fmt}", status_code=400)
