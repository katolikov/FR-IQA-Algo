"""Tests for `upiqal_cli.compose_diagnostic_overlay`.

The composite overlay takes five per-class artefact severity maps, argmaxes
over them per pixel, and blends the winning class's palette colour over
the target image.  We verify:

1. Empty masks (all zeros) leave the target untouched below the legend.
2. A single class mask tints only that region and in the expected colour.
3. Competing masks produce the class with the higher severity.
4. Pixels below the threshold pass through unchanged.
5. Missing classes in the input dict are tolerated (treated as zero).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from upiqal_cli import (  # noqa: E402
    compose_diagnostic_overlay,
    _ARTIFACT_PALETTE,
)


H, W = 64, 64


def _grey_target() -> np.ndarray:
    return np.full((H, W, 3), 128, dtype=np.uint8)


def _colour_of(rgb_pixel: np.ndarray, name: str) -> bool:
    """Has ``rgb_pixel`` moved toward the palette colour for ``name``?

    With alpha-blending at severity ~0.5-0.9 the pixel lands roughly
    half-way between grey (128) and the palette colour.  Euclidean-
    distance comparisons are unreliable at that mid-point, so we test
    the signed *direction* of change per channel: each axis must have
    moved toward its palette target by at least a small margin.
    """
    target = np.array(_ARTIFACT_PALETTE[name], dtype=np.int32)
    grey = np.array([128, 128, 128], dtype=np.int32)
    p = rgb_pixel.astype(np.int32)
    expected_sign = np.sign(target - grey)
    actual_sign = np.sign(p - grey)
    # Require every channel that needs to move to have moved in the
    # right direction by at least 5 units out of 255.
    for i in range(3):
        if expected_sign[i] == 0:
            continue  # palette channel is already at 128 — any value ok
        if actual_sign[i] != expected_sign[i]:
            return False
        if abs(int(p[i]) - 128) < 5:
            return False
    return True


def test_empty_masks_leave_target_mostly_untouched() -> None:
    tgt = _grey_target()
    out = compose_diagnostic_overlay(
        tgt,
        {"blocking": np.zeros((H, W)), "ringing": np.zeros((H, W))},
        draw_legend=False,
    )
    # Every pixel is still grey (±1 for rounding).
    assert np.abs(out.astype(int) - tgt.astype(int)).max() <= 1


def test_single_class_tints_expected_region_in_expected_colour() -> None:
    tgt = _grey_target()
    masks = {"blocking": np.zeros((H, W))}
    masks["blocking"][10:30, 10:30] = 0.9  # strong blocking in top-left square
    out = compose_diagnostic_overlay(tgt, masks, draw_legend=False)

    # Inside the square: pixels should lean toward the blocking colour.
    in_square = out[20, 20]
    assert _colour_of(in_square, "blocking"), (
        f"pixel {tuple(in_square.tolist())} not close to blocking colour"
    )
    # Outside the square: still grey.
    out_square = out[50, 50]
    assert np.allclose(out_square, tgt[50, 50], atol=1)


def test_higher_severity_wins_argmax() -> None:
    tgt = _grey_target()
    masks = {
        "blocking": np.full((H, W), 0.2),
        "blur":     np.full((H, W), 0.8),  # blur is stronger everywhere
    }
    out = compose_diagnostic_overlay(tgt, masks, draw_legend=False)
    # Sample a pixel away from the legend corner.
    pix = out[H // 2, W // 2]
    assert _colour_of(pix, "blur"), (
        f"blur (severity 0.8) should beat blocking (0.2); got {tuple(pix)}"
    )


def test_below_threshold_passes_through() -> None:
    tgt = _grey_target()
    masks = {"noise": np.full((H, W), 0.02)}  # below default threshold=0.05
    out = compose_diagnostic_overlay(
        tgt, masks, threshold=0.05, draw_legend=False,
    )
    # Everything should still be grey.
    assert np.abs(out.astype(int) - tgt.astype(int)).max() <= 1


def test_missing_classes_tolerated() -> None:
    tgt = _grey_target()
    # Only supply one class; the rest should silently be treated as zero.
    masks = {"ringing": np.full((H, W), 0.7)}
    out = compose_diagnostic_overlay(tgt, masks, draw_legend=False)
    pix = out[H // 2, W // 2]
    assert _colour_of(pix, "ringing")


def test_invalid_target_shape_rejected() -> None:
    with pytest.raises(ValueError):
        compose_diagnostic_overlay(np.zeros((H, W)), {})  # 2D, not 3D


def test_resize_mask_to_target() -> None:
    tgt = _grey_target()
    # Supply a mask half the target size; the helper should resize it.
    small = np.zeros((H // 2, W // 2))
    small[:, :] = 0.9
    out = compose_diagnostic_overlay(
        tgt, {"color_shift": small}, draw_legend=False,
    )
    pix = out[H // 2, W // 2]
    assert _colour_of(pix, "color_shift")
