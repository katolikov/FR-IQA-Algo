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
from fastapi import FastAPI, File, UploadFile
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


def read_image_as_tensor(data: bytes, max_side: int = 512) -> torch.Tensor:
    """Decode uploaded bytes to a ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    Images are resized so that the longer side is at most *max_side* pixels
    to keep inference fast.  The returned tensor is contiguous and on CPU;
    it will be moved to the model device in the compare endpoint.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


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
):
    """Compare two images using the real UPIQAL model.

    Accepts multipart/form-data with ``reference_image`` and
    ``target_image`` fields.

    Returns JSON with score, heatmaps, overlay, target image, and
    diagnostic statistics (dominant artifact, severity scores, affected area).
    """
    ref_bytes = await reference_image.read()
    tgt_bytes = await target_image.read()

    ref_tensor = read_image_as_tensor(ref_bytes)
    tgt_tensor = read_image_as_tensor(tgt_bytes)

    # Ensure same spatial dimensions (resize target to match reference)
    _, _, rh, rw = ref_tensor.shape
    _, _, th, tw = tgt_tensor.shape
    if (rh, rw) != (th, tw):
        tgt_tensor = torch.nn.functional.interpolate(
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

    return {
        "score": round(score, 4),
        "heatmaps": heatmaps,
        "overlay": overlay_b64,
        "target": target_b64,
        "diagnostics": diagnostics,
    }
