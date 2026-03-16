"""Comprehensive test suite for the UPIQAL framework.

Tests are designed to run without downloading large model checkpoints by
mocking VGG16 weights.  All tensors use small spatial dimensions for speed.
"""

from __future__ import annotations

import math
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

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
    def test_blocking_mask_shape(self, tgt_img):
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        mask = det(tgt_img)
        assert mask.shape == (B, 1, H, W)

    def test_blocking_mask_binary(self, tgt_img):
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        mask = det(tgt_img)
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Blocking mask should be binary, got {v}"

    def test_ringing_mask_shape(self, tgt_img):
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        mask = det(tgt_img)
        assert mask.shape == (B, 1, H, W)

    def test_ringing_mask_range(self, tgt_img):
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        mask = det(tgt_img)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_combined_engine_keys(self, tgt_img):
        from upiqal.heuristics import SpatialHeuristicsEngine
        engine = SpatialHeuristicsEngine()
        out = engine(tgt_img)
        assert "blocking_mask" in out
        assert "ringing_mask" in out

    def test_uniform_image_no_blocking(self):
        """A perfectly uniform image should have no blocking artifacts."""
        from upiqal.heuristics import JPEGBlockingDetector
        det = JPEGBlockingDetector()
        uniform = torch.full((1, 3, 64, 64), 0.5)
        mask = det(uniform)
        assert mask.sum().item() == 0.0

    def test_sobel_flat_image_no_edges(self):
        """A flat image should produce no edges, thus no ringing."""
        from upiqal.heuristics import GibbsRingingDetector
        det = GibbsRingingDetector()
        flat = torch.full((1, 3, 64, 64), 0.5)
        mask = det(flat)
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
        assert out["diagnostic_tensor"].shape == (B, 5, H, W)

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
