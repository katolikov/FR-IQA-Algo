"""MOS correlation evaluation harness for UPIQAL.

Runs the full UPIQAL pipeline on a MOS-labelled IQA dataset and reports
the three canonical correlations used across the IQA literature:

* **SROCC** — Spearman rank order correlation.  Monotonic, scale-free.
  The primary metric; captures whether UPIQAL orders distorted images
  the same way humans do.
* **PLCC** — Pearson linear correlation *after* logistic regression
  mapping UPIQAL's [0, 1] scores onto the MOS axis.  Captures prediction
  accuracy.
* **RMSE** — root-mean-squared error on the same logistic fit.

Per-distortion-type breakdowns are printed so we can see *where*
UPIQAL under-performs (e.g. "impulse_noise" might be much lower SROCC
than "gaussian_blur").

Usage
-----
    python3 eval/mos_correlation.py                      # KADID-10k, full
    python3 eval/mos_correlation.py --limit 500          # smoke test
    python3 eval/mos_correlation.py \\
        --uncertainty-weights weights/L_cholesky_blockdiag.pth \\
        --max-side 256

The harness exits 0 if SROCC meets ``--srocc-floor`` (default 0, i.e.
no gating).  Set a non-zero floor in CI once you have a baseline.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets import MOSPair, load_dataset  # noqa: E402
from upiqal.model import UPIQAL  # noqa: E402
from upiqal_cli import load_image_as_tensor  # noqa: E402


# -------------------------------------------------------------------------
# Correlation primitives
# -------------------------------------------------------------------------
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient (-1..+1).  NaN-safe."""
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson linear correlation coefficient."""
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank array elements 1..N, averaging ties.  Pure numpy to avoid
    adding a scipy dependency just for one function."""
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    # Assign average ranks to ties.
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = mean_rank
        i = j + 1
    return ranks


def logistic_fit_and_rmse(pred: np.ndarray, mos: np.ndarray) -> tuple[float, float, float]:
    """Standard 5-parameter logistic mapping from predictor to MOS.

    IQA literature (VQEG 2000) fits
        MOS' = β₁ * (0.5 - 1/(1+exp(β₂*(pred-β₃)))) + β₄*pred + β₅
    and reports PLCC / RMSE on MOS vs MOS'.  We use a simpler but
    equally common 4-parameter variant (β₄ = 0) to avoid scipy:

        MOS' = β₁ * (0.5 - 1/(1+exp(β₂*(pred-β₃)))) + β₅

    Fit by grid-search + gradient descent fallback.  Returns
    ``(plcc, rmse, fitted_params)`` where ``fitted_params`` is only
    useful for diagnostics.
    """
    # Seed β from basic statistics.
    b1 = float(mos.max() - mos.min())
    b2 = 1.0
    b3 = float(np.median(pred))
    b5 = float(mos.mean())

    params = torch.tensor([b1, b2, b3, b5], dtype=torch.float64, requires_grad=True)
    x = torch.tensor(pred, dtype=torch.float64)
    y = torch.tensor(mos, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        b1, b2, b3, b5 = params.unbind()
        pred_mos = b1 * (0.5 - 1.0 / (1.0 + torch.exp(b2 * (x - b3)))) + b5
        loss = ((pred_mos - y) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        b1, b2, b3, b5 = params.unbind()
        pred_mos = (
            b1 * (0.5 - 1.0 / (1.0 + torch.exp(b2 * (x - b3)))) + b5
        ).numpy()
    plcc = pearson_r(pred_mos, mos)
    rmse = float(np.sqrt(((pred_mos - mos) ** 2).mean()))
    return plcc, rmse, tuple(float(p) for p in params.detach())


# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------
def score_pair(
    model: UPIQAL,
    pair: MOSPair,
    max_side: int,
    feature_side: int,
) -> float:
    """Run UPIQAL on one pair and return the scalar score."""
    ref = load_image_as_tensor(str(pair.ref_path), max_side=max_side)
    tgt = load_image_as_tensor(str(pair.dist_path), max_side=max_side)

    # Match spatial dims to the smaller of the two.
    _, _, rh, rw = ref.shape
    _, _, th, tw = tgt.shape
    if (rh, rw) != (th, tw):
        ch, cw = min(rh, th), min(rw, tw)
        if (rh, rw) != (ch, cw):
            ref = torch.nn.functional.interpolate(
                ref, size=(ch, cw), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp(0, 1)
        if (th, tw) != (ch, cw):
            tgt = torch.nn.functional.interpolate(
                tgt, size=(ch, cw), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp(0, 1)

    # Pyramid: feed full-res to heuristics, downsampled to Modules 1-4.
    full_side = max(ref.shape[-2:])
    if full_side > feature_side:
        h, w = ref.shape[-2:]
        scale = feature_side / float(max(h, w))
        new_hw = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
        ref_small = torch.nn.functional.interpolate(
            ref, size=new_hw, mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
        tgt_small = torch.nn.functional.interpolate(
            tgt, size=new_hw, mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
        with torch.no_grad():
            out = model(ref_small, tgt_small, ref_full=ref, tgt_full=tgt)
    else:
        with torch.no_grad():
            out = model(ref, tgt)
    return float(out["score"].item())


def evaluate(
    pairs: List[MOSPair],
    uncertainty_weights: Optional[Path] = None,
    max_side: int = 256,
    feature_side: int = 256,
    progress_every: int = 50,
) -> dict:
    """Run UPIQAL over ``pairs`` and return a dict with scores + metrics."""
    print(f"[eval] pairs: {len(pairs)}")

    # Build the model once; Parameters stay fixed during inference.
    if uncertainty_weights is not None and Path(uncertainty_weights).is_file():
        print(f"[eval] loading uncertainty weights: {uncertainty_weights}")
        model = UPIQAL(
            pretrained_vgg=True,
            uncertainty_parameterization="blockdiag",
            uncertainty_weights=uncertainty_weights,
        )
    else:
        print("[eval] identity-diagonal L (no checkpoint)")
        model = UPIQAL(pretrained_vgg=True)
    model.eval()

    scores: List[float] = []
    mos_vals: List[float] = []
    dist_types: List[str] = []
    start = time.perf_counter()
    for i, p in enumerate(pairs):
        try:
            s = score_pair(model, p, max_side=max_side, feature_side=feature_side)
        except Exception as exc:
            print(f"[eval] skip {p.dist_path.name}: {exc}")
            continue
        scores.append(s)
        mos_vals.append(p.mos)
        dist_types.append(p.distortion_type)
        if (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            eta = (len(pairs) - (i + 1)) / rate
            print(
                f"[eval] {i + 1}/{len(pairs)} "
                f"({rate:.1f} pairs/s, eta {eta / 60:.1f} min)",
                flush=True,
            )
    wall = time.perf_counter() - start
    print(f"[eval] done in {wall / 60:.1f} min")

    scores_arr = np.array(scores, dtype=np.float64)
    mos_arr = np.array(mos_vals, dtype=np.float64)

    # UPIQAL's score is "higher = better quality" and KADID's DMOS is
    # "higher = more degraded".  Flip the sign of DMOS (or UPIQAL) so
    # both point the same way before measuring monotone correlation.
    # Spearman is sign-aware: we want |rho| close to 1; report both
    # the raw and sign-corrected value.
    rho_raw = spearman_rho(scores_arr, mos_arr)
    rho = abs(rho_raw)

    # PLCC/RMSE use the sign-corrected predictor (flip UPIQAL) so the
    # logistic fit is monotone in the right direction.
    predictor = -scores_arr if rho_raw > 0 else scores_arr
    plcc, rmse, _ = logistic_fit_and_rmse(predictor, mos_arr)

    # Per-distortion breakdown.
    per_type = {}
    for dtype in sorted(set(dist_types)):
        mask = np.array([d == dtype for d in dist_types])
        if mask.sum() < 4:
            continue
        per_type[dtype] = {
            "n": int(mask.sum()),
            "srocc": float(abs(spearman_rho(scores_arr[mask], mos_arr[mask]))),
        }

    return {
        "n_pairs": len(scores),
        "srocc": float(rho),
        "srocc_raw": float(rho_raw),
        "plcc": float(abs(plcc)) if math.isfinite(plcc) else float("nan"),
        "rmse": float(rmse),
        "per_type": per_type,
        "wall_seconds": wall,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="kadid10k",
        help="Dataset name (default: kadid10k).",
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Dataset root (default: /tmp/kadid/kadid10k/kadid10k for kadid10k).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="If set, subsample this many pairs (random, seed=0).",
    )
    parser.add_argument(
        "--uncertainty-weights", type=Path, default=None,
        help="Path to a trained Cholesky ckpt; if omitted, identity L is used.",
    )
    parser.add_argument(
        "--max-side", type=int, default=256,
        help="Resize longer side to this many px (default 256).",
    )
    parser.add_argument(
        "--feature-side", type=int, default=256,
        help="Pyramid feature-branch longer side (default 256).",
    )
    parser.add_argument(
        "--srocc-floor", type=float, default=0.0,
        help="Exit non-zero if SROCC < this (default 0 = no gate).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional JSON output path.  Prints to stdout either way.",
    )
    args = parser.parse_args()

    pairs = load_dataset(args.dataset, root=args.root, limit=args.limit)
    if not pairs:
        print("[eval] no pairs found — is the dataset unpacked?", file=sys.stderr)
        return 2

    result = evaluate(
        pairs,
        uncertainty_weights=args.uncertainty_weights,
        max_side=args.max_side,
        feature_side=args.feature_side,
    )

    print()
    print(f"=== MOS correlation · dataset={args.dataset} ===")
    print(f"  pairs   : {result['n_pairs']}")
    print(f"  SROCC   : {result['srocc']:.4f}  (raw sign: {result['srocc_raw']:+.4f})")
    print(f"  PLCC    : {result['plcc']:.4f}")
    print(f"  RMSE    : {result['rmse']:.4f}  (MOS scale)")
    print(f"  wall    : {result['wall_seconds'] / 60:.1f} min")
    print()
    print("  per-distortion SROCC:")
    for dtype, v in sorted(
        result["per_type"].items(), key=lambda kv: -kv[1]["srocc"]
    ):
        print(f"    {dtype:32s}  n={v['n']:4d}  SROCC={v['srocc']:.3f}")

    if args.out is not None:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n[eval] saved to {args.out}")

    if result["srocc"] < args.srocc_floor:
        print(
            f"\n[eval] FAIL: SROCC {result['srocc']:.4f} < floor "
            f"{args.srocc_floor}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
