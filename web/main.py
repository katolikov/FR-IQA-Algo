"""FastAPI server for the UPIQAL web application.

Serves the frontend and exposes a POST /api/compare endpoint that accepts
two uploaded images, runs the UPIQAL model (mock by default), and returns
the FR-IQA score plus Base64-encoded diagnostic heatmaps.

Usage (from repo root):
    uvicorn main:app --app-dir web --reload --port 8000

Usage (from web/ directory):
    cd web
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from mock_upiqal import MockUPIQAL

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(title="UPIQAL - Image Quality Analyzer")

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Model singleton (loaded once at startup)
_model: Optional[MockUPIQAL] = None


def get_model() -> MockUPIQAL:
    global _model
    if _model is None:
        _model = MockUPIQAL()
        _model.eval()
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_image_as_tensor(data: bytes, max_side: int = 512) -> torch.Tensor:
    """Decode uploaded bytes to a ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    Images are resized so that the longer side is at most *max_side* pixels
    to keep inference fast.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
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
    """Compare two images using the UPIQAL model.

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

    model = get_model()
    result = model(ref_tensor, tgt_tensor)

    score = float(result["score"][0].item())
    diag = result["diagnostic_tensor"]  # (1, 5, H, W)

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
        "diagnostics": result["diagnostics"],
    }
