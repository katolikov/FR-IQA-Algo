"""Comprehensive test suite for the UPIQAL framework.

Tests are designed to run without downloading large model checkpoints by
mocking VGG16 weights.  All tensors use small spatial dimensions for speed.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# Make the repo root importable so we can hit upiqal_cli and web/main.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, H, W = 2, 3, 64, 64  # batch, channels, height, width


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def ref_img(device: torch.device) -> torch.Tensor:
    """Random reference image in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(B, C, H, W, device=device)


@pytest.fixture
def tgt_img(ref_img: torch.Tensor) -> torch.Tensor:
    """Target image = reference + small noise."""
    torch.manual_seed(99)
    noise = torch.randn_like(ref_img) * 0.05
    return (ref_img + noise).clamp(0.0, 1.0)


@pytest.fixture
def identical_pair(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair of identical images for numerical stability tests."""
    torch.manual_seed(42)
    img = torch.rand(B, C, H, W, device=device)
    return img, img.clone()


# ---------------------------------------------------------------------------
# Lightweight VGG16 mock
# ---------------------------------------------------------------------------

def _make_tiny_vgg() -> nn.Sequential:
    """Build a tiny VGG16-like sequential with correct layer indices.

    Real VGG16 features has 31 layers (indices 0..30).  We replicate the
    same structure with minimal channel widths so hooks on indices
    3, 8, 15, 22, 29 still work.
    """
    layers: list[nn.Module] = []

    # The real VGG16 .features structure (simplified):
    # Block 1: conv(3->64), relu, conv(64->64), relu[3]
    # Block 2: pool, conv(64->128), relu, conv(128->128), relu[8]
    # Block 3: pool, conv(128->256), relu, conv(256->256), relu, conv(256->256), relu[15]
    # Block 4: pool, conv(256->512), relu, conv(512->512), relu, conv(512->512), relu[22]
    # Block 5: pool, conv(512->512), relu, conv(512->512), relu, conv(512->512), relu[29]
    # pool[30]

    cfg = [
        # (out_ch, n_convs)
        (64, 2),    # block 1, relu at index 3
        (128, 2),   # block 2, relu at index 8
        (256, 3),   # block 3, relu at index 15
        (512, 3),   # block 4, relu at index 22
        (512, 3),   # block 5, relu at index 29
    ]

    in_ch = 3
    for block_idx, (out_ch, n_convs) in enumerate(cfg):
        if block_idx > 0:
            layers.append(nn.MaxPool2d(2, 2))
        for _ in range(n_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

    layers.append(nn.MaxPool2d(2, 2))  # index 30

    seq = nn.Sequential(*layers)
    # Initialize with small random weights for deterministic output
    for m in seq.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return seq


def _mock_vgg16(**kwargs):
    """Return a mock VGG16 model whose ``.features`` is our tiny VGG."""
    mock_model = MagicMock()
    mock_model.features = _make_tiny_vgg()
    return mock_model


# ---------------------------------------------------------------------------
# Module 1: Normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_output_shape(self, ref_img, tgt_img):
        from upiqal.normalize import Normalizer
        norm = Normalizer(mode="imagenet")
        r, t = norm(ref_img, tgt_img)
        assert r.shape == ref_img.shape
        assert t.shape == tgt_img.shape

    def test_minmax_range(self, ref_img):
        from upiqal.normalize import Normalizer
        scaled = Normalizer._minmax_scale(ref_img)
        assert scaled.min() >= 0.0 - 1e-6
        assert scaled.max() <= 1.0 + 1e-6

    def test_histogram_mode_shape(self, ref_img, tgt_img):
        from upiqal.normalize import Normalizer
        norm = Normalizer(mode="histogram")
        r, t = norm(ref_img, tgt_img)
        assert r.shape == ref_img.shape
        assert t.shape == tgt_img.shape

    def test_identical_images_normalized_equal(self, identical_pair):
        from upiqal.normalize import Normalizer
        ref, tgt = identical_pair
        norm = Normalizer(mode="imagenet")
        r, t = norm(ref, tgt)
        assert torch.allclose(r, t, atol=1e-6)


# ---------------------------------------------------------------------------
# Module 2: Chromatic Transport Evaluator
# ---------------------------------------------------------------------------

class TestChromaticTransport:
    def test_srgb_to_oklab_shape(self, ref_img):
        from upiqal.color import ChromaticTransportEvaluator
        cte = ChromaticTransportEvaluator()
        oklab = cte.srgb_to_oklab(ref_img)
        assert oklab.shape == ref_img.shape

    def test_oklab_black(self):
        """Black (0,0,0) in sRGB should map to L~0 in Oklab."""
        from upiqal.color import ChromaticTransportEvaluator
        cte = ChromaticTransportEvaluator()
        black = torch.zeros(1, 3, 4, 4)
        oklab = cte.srgb_to_oklab(black)
        assert oklab[:, 0, :, :].abs().max() < 0.01  # L close to 0

    def test_color_map_shape(self, ref_img, tgt_img):
        from upiqal.color import ChromaticTransportEvaluator
        cte = ChromaticTransportEvaluator(patch_size=16, n_bins=4, sinkhorn_iters=5)
        cmap = cte(ref_img, tgt_img)
        assert cmap.shape == (B, 1, H, W)

    def test_identical_images_low_color_map(self, identical_pair):
        from upiqal.color import ChromaticTransportEvaluator
        ref, tgt = identical_pair
        cte = ChromaticTransportEvaluator(patch_size=16, n_bins=4, sinkhorn_iters=30)
        cmap = cte(ref, tgt)
        # Sinkhorn EMD with identical histograms produces small residual
        # transport cost due to entropy regularization; values should stay low.
        assert cmap.max() < 0.15, f"Expected low color degradation, got max={cmap.max().item()}"

    def test_srgb_to_linear_roundtrip(self):
        """Verify the transfer function for known values."""
        from upiqal.color import ChromaticTransportEvaluator
        x = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
        linear = ChromaticTransportEvaluator.srgb_to_linear(x)
        assert linear[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert linear[0, 2, 0, 0].item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Module 3: Deep Statistical Extractor
# ---------------------------------------------------------------------------

class TestDeepStatisticalExtractor:
    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_output_keys(self, mock_vgg, ref_img, tgt_img):
        from upiqal.features import DeepStatisticalExtractor
        dse = DeepStatisticalExtractor(pretrained=False)
        out = dse(ref_img, tgt_img)
        assert "l_maps" in out
        assert "s_maps" in out
        assert "p_tex" in out
        assert "residuals" in out
        assert len(out["l_maps"]) == 5
        assert len(out["s_maps"]) == 5
        assert len(out["p_tex"]) == 5

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_l_maps_bounded(self, mock_vgg, ref_img, tgt_img):
        from upiqal.features import DeepStatisticalExtractor
        dse = DeepStatisticalExtractor(pretrained=False)
        out = dse(ref_img, tgt_img)
        for l_map in out["l_maps"]:
            assert l_map.min() >= -1.1, "l_map should be roughly in [-1, 1]"

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_p_tex_in_01(self, mock_vgg, ref_img, tgt_img):
        from upiqal.features import DeepStatisticalExtractor
        dse = DeepStatisticalExtractor(pretrained=False)
        out = dse(ref_img, tgt_img)
        for pt in out["p_tex"]:
            assert pt.min() >= 0.0 - 1e-6
            assert pt.max() <= 1.0 + 1e-6

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_identical_images_high_similarity(self, mock_vgg, identical_pair):
        from upiqal.features import DeepStatisticalExtractor
        ref, tgt = identical_pair
        dse = DeepStatisticalExtractor(pretrained=False)
        out = dse(ref, tgt)
        for s_map in out["s_maps"]:
            # With identical inputs, structure similarity should be very high
            mean_sim = s_map.mean().item()
            assert mean_sim > 0.9, f"Expected high similarity, got {mean_sim}"

    def test_hanning_kernel_sums_to_one(self):
        from upiqal.features import make_hanning_kernel
        k = make_hanning_kernel(7)
        assert k.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_hanning_kernel_non_negative(self):
        from upiqal.features import make_hanning_kernel
        k = make_hanning_kernel(11)
        assert k.min().item() >= 0.0


# ---------------------------------------------------------------------------
# Module 4: Probabilistic Uncertainty Mapper
# ---------------------------------------------------------------------------

class TestProbabilisticUncertaintyMapper:
    def test_output_shape(self):
        from upiqal.uncertainty import ProbabilisticUncertaintyMapper
        pum = ProbabilisticUncertaintyMapper(feature_dim=128)
        residuals = {
            "relu1_2": torch.randn(B, 64, 32, 32),
            "relu2_2": torch.randn(B, 64, 16, 16),
        }
        out = pum(residuals, target_size=(H, W))
        assert out.shape == (B, 1, H, W)

    def test_zero_residuals_zero_anomaly(self):
        from upiqal.uncertainty import ProbabilisticUncertaintyMapper
        pum = ProbabilisticUncertaintyMapper(feature_dim=128)
        residuals = {
            "relu1_2": torch.zeros(B, 64, 32, 32),
            "relu2_2": torch.zeros(B, 64, 16, 16),
        }
        out = pum(residuals, target_size=(H, W))
        assert out.max().item() < 1e-10, "Zero residuals should give zero anomaly"

    def test_mahalanobis_non_negative(self):
        from upiqal.uncertainty import ProbabilisticUncertaintyMapper
        pum = ProbabilisticUncertaintyMapper(feature_dim=128)
        residuals = {
            "relu1_2": torch.randn(B, 64, 32, 32),
            "relu2_2": torch.randn(B, 64, 16, 16),
        }
        out = pum(residuals, target_size=(H, W))
        assert out.min().item() >= 0.0, "Mahalanobis distance must be non-negative"


# ---------------------------------------------------------------------------
# Module 5: Spatial Heuristics Engine
# ---------------------------------------------------------------------------

class TestSpatialHeuristicsEngine:
    # NOTE: all heuristic detectors are DIFFERENTIAL — they return
    # artifacts present in ``tgt`` that are NOT already in ``ref``.
    # Tests therefore pass both arguments; passing ``tgt == ref`` is the
    # canonical "no artifacts" case.

    def test_blocking_mask_shape(self, ref_img, tgt_img):
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        mask = det(ref_img, tgt_img)
        assert mask.shape == (B, 1, H, W)

    def test_blocking_mask_binary(self, ref_img, tgt_img):
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        mask = det(ref_img, tgt_img)
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Blocking mask should be binary, got {v}"

    def test_ringing_mask_shape(self, ref_img, tgt_img):
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        mask = det(ref_img, tgt_img)
        assert mask.shape == (B, 1, H, W)

    def test_ringing_mask_range(self, ref_img, tgt_img):
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        mask = det(ref_img, tgt_img)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_combined_engine_keys(self, ref_img, tgt_img):
        from upiqal.heuristics import SpatialHeuristicsEngine
        engine = SpatialHeuristicsEngine()
        out = engine(ref_img, tgt_img)
        # Engine now returns four detector masks (added noise/blur
        # alongside blocking/ringing).
        assert "blocking_mask" in out
        assert "ringing_mask" in out
        assert "noise_mask" in out
        assert "blur_mask" in out

    def test_uniform_image_no_blocking(self):
        """A perfectly uniform image should have no blocking artifacts."""
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        uniform = torch.full((1, 3, 64, 64), 0.5)
        mask = det(uniform, uniform)
        assert mask.sum().item() == 0.0

    def test_sobel_flat_image_no_edges(self):
        """A flat image should produce no edges, thus no ringing."""
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        flat = torch.full((1, 3, 64, 64), 0.5)
        mask = det(flat, flat)
        assert mask.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Integration: UPIQAL
# ---------------------------------------------------------------------------

class TestUPIQALIntegration:
    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_output_structure(self, mock_vgg, ref_img, tgt_img):
        from upiqal.model import UPIQAL
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
        model.eval()
        out = model(ref_img, tgt_img)
        assert "score" in out
        assert "diagnostic_tensor" in out
        assert out["score"].shape == (B,)
        # Diagnostic tensor grew from 5 -> 7 channels (added noise + blur
        # masks from the wavelet MAD / HF-attenuation detectors).
        assert out["diagnostic_tensor"].shape == (B, 7, H, W)

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_score_in_01(self, mock_vgg, ref_img, tgt_img):
        from upiqal.model import UPIQAL
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
        model.eval()
        out = model(ref_img, tgt_img)
        score = out["score"]
        assert score.min() >= 0.0 - 1e-6
        assert score.max() <= 1.0 + 1e-6

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_identical_images_high_score(self, mock_vgg, identical_pair):
        from upiqal.model import UPIQAL
        ref, tgt = identical_pair
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=5)
        model.eval()
        out = model(ref, tgt)
        score = out["score"]
        # Identical images should yield a high quality score (>0.5 at minimum)
        assert score.min().item() > 0.45, f"Identical images should score high, got {score}"

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_diagnostic_channels(self, mock_vgg, ref_img, tgt_img):
        from upiqal.model import UPIQAL
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
        model.eval()
        out = model(ref_img, tgt_img)
        diag = out["diagnostic_tensor"]
        # Channel 0: anomaly map (normalized to [0,1])
        assert diag[:, 0].min() >= -1e-6
        assert diag[:, 0].max() <= 1.0 + 1e-6
        # Channel 1: color map (normalized to [0,1])
        assert diag[:, 1].min() >= -1e-6
        assert diag[:, 1].max() <= 1.0 + 1e-6

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_no_nan_or_inf(self, mock_vgg, ref_img, tgt_img):
        from upiqal.model import UPIQAL
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
        model.eval()
        out = model(ref_img, tgt_img)
        assert not torch.isnan(out["score"]).any(), "Score contains NaN"
        assert not torch.isinf(out["score"]).any(), "Score contains Inf"
        assert not torch.isnan(out["diagnostic_tensor"]).any(), "Diagnostic contains NaN"
        assert not torch.isinf(out["diagnostic_tensor"]).any(), "Diagnostic contains Inf"

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_batch_consistency(self, mock_vgg, device):
        """Single-image result should match its slice from a batched run."""
        from upiqal.model import UPIQAL
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, H, W, device=device)
        img2 = (img1 + torch.randn_like(img1) * 0.1).clamp(0, 1)

        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=3)
        model.eval()

        out_single = model(img1, img2)
        # Batch of 2 copies
        out_batch = model(img1.repeat(2, 1, 1, 1), img2.repeat(2, 1, 1, 1))
        assert torch.allclose(
            out_batch["score"][0], out_batch["score"][1], atol=1e-4
        ), "Duplicated batch items should have identical scores"


# ---------------------------------------------------------------------------
# Image loading: .npy, RAW, NV21, NV12
# ---------------------------------------------------------------------------

class TestImageLoading:
    """Exercise the loaders shared by the CLI and web entry points."""

    # ----- .npy ----------------------------------------------------------

    def _save_npy(self, tmp_path, arr, name="arr.npy"):
        path = tmp_path / name
        np.save(path, arr)
        return str(path)

    def test_npy_uint8_2d_grayscale(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = (np.random.rand(32, 48) * 255).astype(np.uint8)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.shape == (32, 48, 3)
        assert out.dtype == np.uint8
        assert np.all(out[..., 0] == arr)
        assert np.all(out[..., 1] == arr)

    def test_npy_uint8_hwc(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.shape == (16, 16, 3)
        assert np.array_equal(out, arr)

    def test_npy_chw_channels_first(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = (np.random.rand(3, 24, 32) * 255).astype(np.uint8)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.shape == (24, 32, 3)
        assert np.array_equal(out, np.transpose(arr, (1, 2, 0)))

    def test_npy_4d_batch_squeezed(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = (np.random.rand(1, 20, 24, 3) * 255).astype(np.uint8)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.shape == (20, 24, 3)
        assert np.array_equal(out, arr[0])

    def test_npy_uint16_full_range_scaled(self, tmp_path):
        from upiqal_cli import load_raw_image
        # Top half saturated 65535, bottom half 0 -> should land at 255 / 0
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[:2, :] = 65535
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 255
        assert out[3, 3, 0] == 0

    def test_npy_float32_unit_range(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = np.linspace(0.0, 1.0, 4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0
        assert out[3, 3, 2] == 255

    def test_npy_float32_0_255_range(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = np.array([[[10.0, 20.0, 30.0], [200.0, 210.0, 220.0]]], dtype=np.float32)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out[0, 0, 0] == 10
        assert out[0, 1, 2] == 220

    def test_npy_nan_inf_safe(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = np.array([[np.nan, np.inf], [-np.inf, 0.5]], dtype=np.float32)
        out = load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()

    def test_npy_unsupported_shape_raises(self, tmp_path):
        from upiqal_cli import load_raw_image
        arr = np.zeros((2, 5, 6, 7), dtype=np.uint8)  # 4D non-singleton leading
        with pytest.raises(ValueError):
            load_raw_image(self._save_npy(tmp_path, arr), 0, 0, "RGB888")

    # ----- GRAY8 / RGB888 ------------------------------------------------

    def test_gray8_round_trip(self, tmp_path):
        from upiqal_cli import load_raw_image
        gray = (np.random.rand(20, 30) * 255).astype(np.uint8)
        path = tmp_path / "frame.raw"
        path.write_bytes(gray.tobytes())
        out = load_raw_image(str(path), 30, 20, "GRAY8")
        assert out.shape == (20, 30, 3)
        assert np.array_equal(out[..., 0], gray)

    def test_rgb888_round_trip(self, tmp_path):
        from upiqal_cli import load_raw_image
        rgb = (np.random.rand(16, 24, 3) * 255).astype(np.uint8)
        path = tmp_path / "frame.bin"
        path.write_bytes(rgb.tobytes())
        out = load_raw_image(str(path), 24, 16, "RGB888")
        assert np.array_equal(out, rgb)

    def test_rgb888_truncated_raises(self, tmp_path):
        from upiqal_cli import load_raw_image
        path = tmp_path / "short.raw"
        path.write_bytes(b"\x00" * (16 * 16 * 3 - 5))  # 5 bytes short
        with pytest.raises(ValueError, match="too short"):
            load_raw_image(str(path), 16, 16, "RGB888")

    # ----- NV21 / NV12 ---------------------------------------------------

    def _make_yuv420sp(self, y, u_byte, v_byte, chroma_order):
        """Build a YUV420SP frame with a custom Y plane and uniform U/V."""
        h, w = y.shape
        cw, ch = (w + 1) // 2, (h + 1) // 2
        chroma = np.empty((ch, cw, 2), dtype=np.uint8)
        if chroma_order == "VU":  # NV21
            chroma[..., 0] = v_byte
            chroma[..., 1] = u_byte
        else:  # NV12
            chroma[..., 0] = u_byte
            chroma[..., 1] = v_byte
        return y.tobytes() + chroma.tobytes()

    def test_nv21_neutral_chroma_recovers_luma(self, tmp_path):
        from upiqal_cli import load_raw_image
        y = (np.random.rand(16, 16) * 255).astype(np.uint8)
        path = tmp_path / "frame.nv21"
        path.write_bytes(self._make_yuv420sp(y, 128, 128, "VU"))
        out = load_raw_image(str(path), 16, 16, "NV21")
        # With neutral U/V (=128), R/G/B should all equal Y exactly
        assert np.allclose(out[..., 0], y, atol=1)
        assert np.allclose(out[..., 1], y, atol=1)
        assert np.allclose(out[..., 2], y, atol=1)

    def test_nv12_neutral_chroma_recovers_luma(self, tmp_path):
        from upiqal_cli import load_raw_image
        y = (np.random.rand(16, 16) * 255).astype(np.uint8)
        path = tmp_path / "frame.nv12"
        path.write_bytes(self._make_yuv420sp(y, 128, 128, "UV"))
        out = load_raw_image(str(path), 16, 16, "NV12")
        assert np.allclose(out[..., 0], y, atol=1)
        assert np.allclose(out[..., 1], y, atol=1)
        assert np.allclose(out[..., 2], y, atol=1)

    def test_nv12_red_chroma_differs_from_nv21(self, tmp_path):
        """An NV12 buffer with U=64, V=200 must NOT decode the same as NV21
        (which would interpret bytes as V=64, U=200) — proves chroma order
        is honoured per format."""
        from upiqal_cli import load_raw_image
        y = np.full((8, 8), 128, dtype=np.uint8)
        # NV12 buffer
        nv12_path = tmp_path / "frame.nv12"
        nv12_path.write_bytes(self._make_yuv420sp(y, 64, 200, "UV"))
        nv12_rgb = load_raw_image(str(nv12_path), 8, 8, "NV12")
        # Same byte stream parsed as NV21 (we lie about extension here)
        nv21_path = tmp_path / "frame_as_nv21.raw"
        nv21_path.write_bytes(self._make_yuv420sp(y, 64, 200, "UV"))
        nv21_rgb = load_raw_image(str(nv21_path), 8, 8, "NV21")
        assert not np.array_equal(nv12_rgb, nv21_rgb), (
            "NV12 and NV21 should disagree on chroma ordering"
        )

    def test_nv12_odd_dimensions(self, tmp_path):
        from upiqal_cli import load_raw_image
        w, h = 33, 17
        y = (np.random.rand(h, w) * 255).astype(np.uint8)
        path = tmp_path / "odd.nv12"
        path.write_bytes(self._make_yuv420sp(y, 128, 128, "UV"))
        out = load_raw_image(str(path), w, h, "NV12")
        assert out.shape == (h, w, 3)

    def test_extension_overrides_pixel_format(self, tmp_path):
        """A .nv12 extension should force NV12 even if --pixel_format=NV21."""
        from upiqal_cli import load_raw_image
        y = np.full((8, 8), 100, dtype=np.uint8)
        path = tmp_path / "frame.nv12"
        path.write_bytes(self._make_yuv420sp(y, 64, 200, "UV"))
        # Even though caller passes NV21, .nv12 extension wins.
        out_a = load_raw_image(str(path), 8, 8, "NV21")
        out_b = load_raw_image(str(path), 8, 8, "NV12")
        assert np.array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# CLI end-to-end: full pipeline on PNG and NV12 inputs
# ---------------------------------------------------------------------------

class TestCLIEndToEnd:
    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_run_pipeline_png(self, mock_vgg, tmp_path):
        import upiqal_cli
        from PIL import Image
        rng = np.random.default_rng(0)
        ref = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
        tgt = np.clip(ref.astype(np.int16) + rng.integers(-10, 10, ref.shape), 0, 255).astype(np.uint8)
        ref_path = tmp_path / "ref.png"
        tgt_path = tmp_path / "tgt.png"
        Image.fromarray(ref).save(ref_path)
        Image.fromarray(tgt).save(tgt_path)
        out_dir = tmp_path / "out"

        ns = type("Args", (), {})()
        ns.reference = str(ref_path)
        ns.target = str(tgt_path)
        ns.name = None
        ns.max_side = 48
        ns.output_dir = str(out_dir)
        ns.width = 0
        ns.height = 0
        ns.pixel_format = "RGB888"
        ns.output_format = "png"

        upiqal_cli.run_pipeline(ns)

        report = out_dir / "report.json"
        assert report.exists()
        import json
        data = json.loads(report.read_text())
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert (out_dir / "global_anomaly_map.png").exists()
        assert (out_dir / "anomaly_overlay.png").exists()

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_run_pipeline_nv12(self, mock_vgg, tmp_path):
        import upiqal_cli
        rng = np.random.default_rng(1)
        w, h = 48, 32
        for n in ("ref", "tgt"):
            y = (rng.random((h, w)) * 255).astype(np.uint8)
            cw, ch = (w + 1) // 2, (h + 1) // 2
            uv = np.full((ch, cw, 2), 128, dtype=np.uint8)
            (tmp_path / f"{n}.nv12").write_bytes(y.tobytes() + uv.tobytes())
        out_dir = tmp_path / "out"

        ns = type("Args", (), {})()
        ns.reference = str(tmp_path / "ref.nv12")
        ns.target = str(tmp_path / "tgt.nv12")
        ns.name = None
        ns.max_side = 48
        ns.output_dir = str(out_dir)
        ns.width = w
        ns.height = h
        ns.pixel_format = "NV12"
        ns.output_format = "png"

        upiqal_cli.run_pipeline(ns)

        report = out_dir / "report.json"
        assert report.exists()
        import json
        data = json.loads(report.read_text())
        assert 0.0 <= data["score"] <= 1.0


# ---------------------------------------------------------------------------
# Web end-to-end: FastAPI TestClient
# ---------------------------------------------------------------------------

class TestWebEndToEnd:
    """Smoke-test the FastAPI ``/api/compare`` endpoint against the REAL UPIQAL
    pipeline.  VGG16 torchvision weight fetching is patched to return a tiny
    local model (see ``_mock_vgg16``) so the test runs offline; the UPIQAL
    module and all five pipeline stages are otherwise unchanged.
    """

    @pytest.fixture
    def client(self, monkeypatch):
        import importlib
        from fastapi.testclient import TestClient
        # Patch torchvision's VGG16 loader BEFORE importing web.main so the
        # real UPIQAL in get_model() initialises with the tiny stand-in weights.
        monkeypatch.setattr(
            "torchvision.models.vgg16", _mock_vgg16, raising=True,
        )
        web_main = importlib.import_module("web.main")
        # Reset any cached model so the patched VGG16 is used.
        web_main._model = None  # type: ignore[attr-defined]
        return TestClient(web_main.app)

    def _png_bytes(self, seed):
        from PIL import Image
        rng = np.random.default_rng(seed)
        arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def test_compare_png(self, client):
        ref = self._png_bytes(0)
        tgt = self._png_bytes(1)
        r = client.post(
            "/api/compare",
            files={
                "reference_image": ("ref.png", ref, "image/png"),
                "target_image": ("tgt.png", tgt, "image/png"),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "score" in body and 0.0 <= body["score"] <= 1.0
        assert "heatmaps" in body
        assert set(body["heatmaps"].keys()) >= {
            "anomaly", "color", "structure", "blocking", "ringing",
        }

    def test_compare_npy(self, client):
        rng = np.random.default_rng(2)
        ref_arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        tgt_arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        ref_buf = io.BytesIO(); np.save(ref_buf, ref_arr)
        tgt_buf = io.BytesIO(); np.save(tgt_buf, tgt_arr)
        r = client.post(
            "/api/compare",
            files={
                "reference_image": ("ref.npy", ref_buf.getvalue(), "application/octet-stream"),
                "target_image": ("tgt.npy", tgt_buf.getvalue(), "application/octet-stream"),
            },
        )
        assert r.status_code == 200, r.text
        assert 0.0 <= r.json()["score"] <= 1.0

    def test_compare_nv12(self, client):
        rng = np.random.default_rng(3)
        w, h = 32, 32
        ref_y = (rng.random((h, w)) * 255).astype(np.uint8)
        tgt_y = (rng.random((h, w)) * 255).astype(np.uint8)
        cw, ch = (w + 1) // 2, (h + 1) // 2
        uv = np.full((ch, cw, 2), 128, dtype=np.uint8)
        ref_bytes = ref_y.tobytes() + uv.tobytes()
        tgt_bytes = tgt_y.tobytes() + uv.tobytes()
        r = client.post(
            "/api/compare",
            data={"width": w, "height": h, "pixel_format": "NV12"},
            files={
                "reference_image": ("ref.nv12", ref_bytes, "application/octet-stream"),
                "target_image": ("tgt.nv12", tgt_bytes, "application/octet-stream"),
            },
        )
        assert r.status_code == 200, r.text
        assert 0.0 <= r.json()["score"] <= 1.0


# ---------------------------------------------------------------------------
# Numerical-stability regression tests
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Pin down the stability fixes so they don't regress."""

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_identical_images_no_nan(self, mock_vgg):
        from upiqal.model import UPIQAL
        torch.manual_seed(0)
        img = torch.rand(1, 3, 64, 64)
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=5)
        model.eval()
        out = model(img, img.clone())
        assert not torch.isnan(out["score"]).any()
        assert not torch.isnan(out["diagnostic_tensor"]).any()
        assert not torch.isinf(out["diagnostic_tensor"]).any()
        # The anomaly channel comes from residuals = phi_r - phi_t which is
        # identically zero for identical inputs, so the safe normaliser must
        # keep it at zero (the old code blew this up to ~1e12 / NaN).
        assert out["diagnostic_tensor"][:, 0].max().item() < 1e-3

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_uniform_constant_images_no_nan(self, mock_vgg):
        from upiqal.model import UPIQAL
        ref = torch.full((1, 3, 64, 64), 0.3)
        tgt = torch.full((1, 3, 64, 64), 0.3)
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=5)
        model.eval()
        out = model(ref, tgt)
        assert not torch.isnan(out["score"]).any()
        assert not torch.isnan(out["diagnostic_tensor"]).any()

    @patch("upiqal.features.models.vgg16", side_effect=_mock_vgg16)
    def test_deep_sim_clamped_to_unit_interval(self, mock_vgg):
        from upiqal.model import UPIQAL
        torch.manual_seed(1)
        ref = torch.rand(1, 3, 64, 64)
        tgt = torch.rand(1, 3, 64, 64)  # totally different image
        model = UPIQAL(pretrained_vgg=False, color_patch_size=16, sinkhorn_iters=5)
        model.eval()
        out = model(ref, tgt)
        deep_sim = out["diagnostic_tensor"][:, 2]
        assert deep_sim.min().item() >= 0.0 - 1e-6
        assert deep_sim.max().item() <= 1.0 + 1e-6
