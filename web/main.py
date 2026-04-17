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

import os

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# CORS — when the frontend is hosted elsewhere (e.g. Vercel static) and proxies
# /api/* here, we need to accept cross-origin requests. `UPIQAL_ALLOWED_ORIGINS`
# is a comma-separated list. Defaults to "*" for ease of local development.
_allowed = os.environ.get("UPIQAL_ALLOWED_ORIGINS", "*")
_allow_origins = [o.strip() for o in _allowed.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static & template files from `web/public/` so both the FastAPI host
# (local dev + CPU container) and Vercel's static hosting share the same files.
# Fall back to the legacy `web/templates/` + `web/static/` layout if present.
PUBLIC_DIR = Path(__file__).parent / "public"
LEGACY_TEMPLATES_DIR = Path(__file__).parent / "templates"
LEGACY_STATIC_DIR = Path(__file__).parent / "static"

if PUBLIC_DIR.exists():
    TEMPLATES_DIR = PUBLIC_DIR
    STATIC_DIR = PUBLIC_DIR / "static"
else:
    TEMPLATES_DIR = LEGACY_TEMPLATES_DIR
    STATIC_DIR = LEGACY_STATIC_DIR

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Force CPU.  MPS silently OOMs inside ProbabilisticUncertaintyMapper at
# larger resolutions: the command buffer errors async and the tensor
# returns all-zero, nuking the anomaly channel and the "affected area"
# metric.  CUDA is also disabled because the same user environment does
# not have it in production, and CPU is numerically authoritative.
_device: torch.device = torch.device("cpu")

# Model singleton (loaded once at startup, never reloaded per-request)
_model: Optional[UPIQAL] = None


def get_model() -> UPIQAL:
    """Return the global UPIQAL model, initialising it on first call.

    The pretrained VGG16 weights are loaded from the local
    ``weights/vgg16-397923af.pth`` file.  The full model is moved to
    the best available device and set to eval mode.
    """
    global _model
    if _model is None:
        # Resolve trained Cholesky factor. Priority:
        #   1. UPIQAL_UNCERTAINTY_WEIGHTS env var
        #   2. Default bundled ckpt at weights/L_cholesky_blockdiag.pth
        #   3. No trained factor (identity-diagonal fallback)
        env_weights = os.environ.get("UPIQAL_UNCERTAINTY_WEIGHTS") or None
        default_weights = _REPO_ROOT / "weights" / "L_cholesky_blockdiag.pth"
        if env_weights:
            uncertainty_weights: Optional[str] = env_weights
        elif default_weights.is_file():
            uncertainty_weights = str(default_weights)
        else:
            uncertainty_weights = None

        if uncertainty_weights:
            _model = UPIQAL(
                pretrained_vgg=True,
                uncertainty_parameterization="blockdiag",
                uncertainty_weights=uncertainty_weights,
            )
        else:
            _model = UPIQAL(pretrained_vgg=True)
        _model.to(_device)
        _model.eval()
        msg = f"UPIQAL model loaded on {_device}"
        if uncertainty_weights:
            msg += f" (uncertainty weights: {uncertainty_weights})"
        else:
            msg += " (uncertainty: identity-diagonal fallback)"
        print(msg)
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PIXEL_FORMATS = ("NV21", "NV12", "GRAY8", "RGB888")


def _yuv420sp_to_rgb(
    data: bytes, width: int, height: int, chroma_order: str,
) -> np.ndarray:
    """Convert a YUV 4:2:0 semi-planar buffer to ``(H, W, 3)`` uint8 RGB.

    *chroma_order* is ``"VU"`` for NV21 and ``"UV"`` for NV12.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"YUV420SP requires positive dimensions, got {width}x{height}")
    if chroma_order not in ("VU", "UV"):
        raise ValueError(f"chroma_order must be 'VU' or 'UV', got {chroma_order!r}")

    y_size = width * height
    cw = (width + 1) // 2
    ch = (height + 1) // 2
    uv_size = cw * ch * 2
    expected = y_size + uv_size
    if len(data) < expected:
        fmt_name = "NV21" if chroma_order == "VU" else "NV12"
        raise ValueError(
            f"{fmt_name} data too short: expected {expected} bytes "
            f"for {width}x{height}, got {len(data)}"
        )

    y = (
        np.frombuffer(data, np.uint8, count=y_size)
        .reshape(height, width)
        .astype(np.float32)
    )
    chroma = np.frombuffer(
        data, np.uint8, offset=y_size, count=uv_size,
    ).reshape(ch, cw, 2)

    if chroma_order == "VU":
        v_plane = chroma[:, :, 0].astype(np.float32)
        u_plane = chroma[:, :, 1].astype(np.float32)
    else:
        u_plane = chroma[:, :, 0].astype(np.float32)
        v_plane = chroma[:, :, 1].astype(np.float32)

    u = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)[:height, :width]

    r = np.clip(y + 1.370705 * (v - 128), 0, 255)
    g = np.clip(y - 0.337633 * (u - 128) - 0.698001 * (v - 128), 0, 255)
    b = np.clip(y + 1.732446 * (u - 128), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _nv21_to_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert NV21 (YUV420SP, VU interleaved) raw bytes to ``(H, W, 3)`` RGB."""
    return _yuv420sp_to_rgb(data, width, height, chroma_order="VU")


def _nv12_to_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert NV12 (YUV420SP, UV interleaved) raw bytes to ``(H, W, 3)`` RGB."""
    return _yuv420sp_to_rgb(data, width, height, chroma_order="UV")


def _npy_array_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary ``np.load`` result to an ``(H, W, 3)`` uint8 RGB image.

    Mirrors :func:`upiqal_cli._npy_array_to_rgb_uint8` so the web and CLI
    front-ends accept the same set of ``.npy`` shapes and dtypes.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f".npy payload is not an ndarray: {type(arr)!r}")
    if arr.size == 0:
        raise ValueError(f".npy array is empty (shape={arr.shape})")

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
            pass
        elif h2 == 4:
            arr = arr[:, :, :3]
        elif h0 == 3 and h2 != 3:
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


def _decode_raw_bytes(
    data: bytes,
    width: int,
    height: int,
    pixel_format: str,
    filename: str = "",
) -> np.ndarray:
    """Decode raw/binary upload bytes to ``(H, W, 3)`` uint8 RGB.

    Handles ``.npy`` (by extension in *filename*), NV21, NV12, GRAY8, RGB888.
    The file extensions ``.nv21`` / ``.nv12`` override the explicit
    *pixel_format* argument.
    """
    ext = Path(filename).suffix.lower() if filename else ""

    if ext == ".npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return _npy_array_to_rgb_uint8(arr)

    fmt = pixel_format.upper()
    if ext == ".nv21":
        fmt = "NV21"
    elif ext == ".nv12":
        fmt = "NV12"

    if fmt == "NV21":
        return _nv21_to_rgb(data, width, height)
    if fmt == "NV12":
        return _nv12_to_rgb(data, width, height)
    if fmt == "GRAY8":
        expected = width * height
        if expected <= 0:
            raise ValueError(
                f"GRAY8 requires positive width/height, got {width}x{height}"
            )
        if len(data) < expected:
            raise ValueError(
                f"GRAY8 data too short: expected {expected} bytes, got {len(data)}"
            )
        gray = np.frombuffer(data, np.uint8, count=expected).reshape(height, width)
        return np.stack([gray, gray, gray], axis=-1)
    if fmt == "RGB888":
        expected = width * height * 3
        if expected <= 0:
            raise ValueError(
                f"RGB888 requires positive width/height, got {width}x{height}"
            )
        if len(data) < expected:
            raise ValueError(
                f"RGB888 data too short: expected {expected} bytes, got {len(data)}"
            )
        return np.frombuffer(data, np.uint8, count=expected).reshape(height, width, 3)

    raise ValueError(
        f"Unsupported pixel_format: {pixel_format!r} (expected one of {_PIXEL_FORMATS})"
    )


_RAW_EXTENSIONS = {".raw", ".bin", ".nv21", ".nv12", ".yuv", ".npy"}


def read_image_as_tensor(
    data: bytes,
    # 768px balances detection fidelity with MPS memory (M1 Pro ~16 GB).
    # Raise to 1024/2048 on larger-VRAM machines via env override.
    max_side: int = 768,
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


def _hf_energy(img: torch.Tensor) -> torch.Tensor:
    """Laplacian-variance high-frequency energy proxy.

    Larger value -> more sharp detail (sharpness / noise); smaller ->
    smoother (blurrier / flatter).  Used by ``compute_diagnostics`` as a
    discriminator so generic VGG dissimilarity isn't mislabelled as blur
    when the target is actually sharper than the reference.
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
    return lap.var(dim=(1, 2, 3))


def compute_diagnostics(
    diag: torch.Tensor,
    ref_raw: Optional[torch.Tensor] = None,
    tgt_raw: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Derive severity scores and dominant artifact from the diagnostic tensor.

    Mirrors the CLI's ``compute_diagnostics`` (upiqal_cli.py): ranks
    candidates by raw contribution * specificity weight, gated by an
    HF-energy discriminator. This prevents the EMD channel from
    dominating every label and keeps "Blur" away from structurally
    different (non-blurred) pairs.

    Parameters
    ----------
    diag : torch.Tensor
        The diagnostic tensor ``(1, C, H, W)``. Channels:
        0 = anomaly, 1 = color, 2 = structure, 3 = blocking, 4 = ringing.
        Optional (present in C=7 tensors from the current model):
        5 = noise (wavelet-MAD), 6 = blur (HF attenuation).
    ref_raw, tgt_raw : torch.Tensor, optional
        Images in ``[0, 1]`` used by the HF-energy discriminator. When
        omitted, the discriminator is skipped.

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
    # Dedicated detectors (optional; fall back to the anomaly/structure
    # proxies for backward compatibility with older 5-channel tensors).
    has_noise_ch = diag.shape[1] >= 6
    has_blur_ch = diag.shape[1] >= 7

    blocking_sev = float(blocking.mean().item()) * 100
    ringing_sev = float(ringing.mean().item()) * 100
    color_sev = float(color_map.mean().item()) * 100
    if has_noise_ch:
        noise_sev = float(diag[0, 5].mean().item()) * 100
    else:
        noise_sev = float(anomaly.mean().item()) * 100
    if has_blur_ch:
        blur_sev = float(diag[0, 6].mean().item()) * 100
    else:
        blur_sev = float((1.0 - structure).mean().item()) * 100

    severity_scores = {
        "blocking":    round(min(blocking_sev * _SEVERITY_MULTIPLIERS["blocking"], 100.0), 1),
        "ringing":     round(min(ringing_sev  * _SEVERITY_MULTIPLIERS["ringing"],  100.0), 1),
        "noise":       round(min(noise_sev    * _SEVERITY_MULTIPLIERS["noise"],    100.0), 1),
        "color_shift": round(min(color_sev    * _SEVERITY_MULTIPLIERS["color_shift"], 100.0), 1),
        "blur":        round(min(blur_sev     * _SEVERITY_MULTIPLIERS["blur"],     100.0), 1),
    }

    # Contribution weights — high for specificity-rich heuristic masks,
    # moderate for deep-feature signals. Matches upiqal_cli.py.
    contrib = {
        "blocking":    blocking_sev * 1.0,
        "ringing":     ringing_sev  * 1.0,
        "noise":       noise_sev    * 0.30,
        "color_shift": color_sev    * 0.30,
        "blur":        blur_sev     * 0.50,
    }

    # HF-energy discriminator
    if ref_raw is not None and tgt_raw is not None:
        with torch.no_grad():
            hf_ref = float(_hf_energy(ref_raw).mean().item())
            hf_tgt = float(_hf_energy(tgt_raw).mean().item())
        hf_ratio = hf_tgt / (hf_ref + 1e-8)
        if hf_ratio > 1.1:
            contrib["blur"] *= 0.0
            if noise_sev > 10.0:
                contrib["color_shift"] *= 0.3
        elif hf_ratio < 0.9:
            contrib["noise"]   *= 0.3
            contrib["ringing"] *= 0.3
        else:
            if color_sev >= 25.0:
                contrib["blur"] *= 0.3

    # MPS / float32 drift can leave a single channel mildly non-zero on
    # near-identical inputs (deep_sim ≈ 0.88 on MPS → blur_sev ≈ 12%).
    # Treat a lone weak signal as device noise, not a real artifact.
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


@app.get("/paper", response_class=HTMLResponse)
async def paper():
    """Serve the UPIQAL research paper page."""
    html_path = TEMPLATES_DIR / "paper.html"
    if not html_path.exists():
        return HTMLResponse(content="paper.html not found", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz():
    """Lightweight liveness probe for container platforms (Fly / Render)."""
    return {"ok": True, "device": str(_device)}


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

    # Ensure same spatial dimensions. Fix (parity with CLI): downsample the
    # LARGER image to the SMALLER common size via bicubic + antialias. Never
    # upsamples, so no bilinear blur is injected into the target — this
    # eliminates the false "Blur" label on same-content pairs at different
    # native resolutions.
    _, _, rh, rw = ref_tensor.shape
    _, _, th, tw = tgt_tensor.shape
    if (rh, rw) != (th, tw):
        common_h = min(rh, th)
        common_w = min(rw, tw)
        if (rh, rw) != (common_h, common_w):
            ref_tensor = F.interpolate(
                ref_tensor, size=(common_h, common_w),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
        if (th, tw) != (common_h, common_w):
            tgt_tensor = F.interpolate(
                tgt_tensor, size=(common_h, common_w),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)

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

    # Compute diagnostics (HF discriminator uses raw pipeline-size tensors).
    diagnostics = compute_diagnostics(
        diag, ref_raw=ref_on_device, tgt_raw=tgt_on_device,
    )

    # Encode heatmap channels as Base64 PNGs.
    # Channels 5/6 (noise/blur) are only present when the current model
    # emits a 7-channel diagnostic tensor; older 5-channel tensors
    # simply skip them for backwards compatibility.
    channel_names = _CHANNEL_NAMES[: diag.shape[1]]
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


_CHANNEL_NAMES = [
    "anomaly", "color", "structure", "blocking", "ringing", "noise", "blur",
]


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
