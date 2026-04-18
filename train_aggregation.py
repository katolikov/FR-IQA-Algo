"""Fine-tune the 8 aggregation / A-DISTS-sigmoid Parameters against MOS.

These `nn.Parameter`s were promoted from Python floats when the paper-
spec-deviation cleanup landed, but no optimiser has ever touched them:

* ``UPIQAL.w_color``, ``w_anomaly``, ``w_structure``, ``w_heuristic``
* ``UPIQAL.score_scale``, ``score_center``
* ``DeepStatisticalExtractor.sigmoid_gain`` (A-DISTS P_tex slope)
* ``DeepStatisticalExtractor.sigmoid_bias`` (A-DISTS P_tex offset)

They sit at their hand-tuned defaults forever.  This script wires them
up to Adam on the ranking loss from ``upiqal.suss_train.ranking_loss``
using a real MOS-labelled dataset (KADID-10k) and saves a small
checkpoint that the CLI / web can load.

Typical invocation (after fetching KADID):

    python3 train_aggregation.py \\
        --uncertainty-weights weights/L_cholesky_blockdiag.pth \\
        --dataset kadid10k \\
        --epochs 5 --batch-size 16 \\
        --lr 5e-3 \\
        --out weights/aggregation.pth

The loss is tiny (~hundreds of forward passes) so this finishes in
minutes on CPU.  The output checkpoint contains only the 8 trained
parameters + metadata; ``upiqal_cli.py --aggregation-weights PATH``
loads them on top of whatever Cholesky factor is in use.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from eval.datasets import MOSPair, held_out_split, load_dataset
from upiqal.model import UPIQAL
from upiqal.suss_train import ranking_loss
from upiqal_cli import load_image_as_tensor


# ---- utilities ----------------------------------------------------------
def _match_shapes(ref: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, rh, rw = ref.shape
    _, _, th, tw = tgt.shape
    if (rh, rw) == (th, tw):
        return ref, tgt
    ch, cw = min(rh, th), min(rw, tw)
    if (rh, rw) != (ch, cw):
        ref = F.interpolate(
            ref, size=(ch, cw), mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
    if (th, tw) != (ch, cw):
        tgt = F.interpolate(
            tgt, size=(ch, cw), mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
    return ref, tgt


def _score_batch(
    model: UPIQAL, batch: List[MOSPair], max_side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pred_scores, mos)`` both shape ``(len(batch),)``.

    Unlike inference we do **not** run in ``no_grad`` because we need
    gradients through the aggregation parameters.  VGG16 weights remain
    frozen via ``requires_grad_(False)`` set during model init.
    """
    preds = []
    targets = []
    for p in batch:
        ref = load_image_as_tensor(str(p.ref_path), max_side=max_side)
        tgt = load_image_as_tensor(str(p.dist_path), max_side=max_side)
        ref, tgt = _match_shapes(ref, tgt)
        out = model(ref, tgt)
        preds.append(out["score"])
        targets.append(torch.tensor([p.mos], dtype=torch.float32))
    return torch.cat(preds), torch.cat(targets)


def _collect_aggregation_params(model: UPIQAL) -> list[torch.nn.Parameter]:
    """Pull the 8 parameters we want to train and mark them leaf-trainable.

    Everything else is frozen (``requires_grad=False``).
    """
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    names = [
        "w_color", "w_anomaly", "w_structure", "w_heuristic",
        "score_scale", "score_center",
    ]
    trainable: list[torch.nn.Parameter] = []
    for name in names:
        obj = getattr(model, name)
        if isinstance(obj, torch.nn.Parameter):
            obj.requires_grad_(True)
            trainable.append(obj)

    # A-DISTS sigmoid (k, b) lives on the feature extractor.
    for attr in ("sigmoid_gain", "sigmoid_bias"):
        obj = getattr(model.deep_stats, attr, None)
        if isinstance(obj, torch.nn.Parameter):
            obj.requires_grad_(True)
            trainable.append(obj)

    return trainable


def _param_snapshot(params: list[torch.nn.Parameter]) -> dict[str, float]:
    """For logging: print the current scalar values."""
    return {
        f"p{i}": float(p.detach().item()) if p.numel() == 1 else float("nan")
        for i, p in enumerate(params)
    }


# ---- main --------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune UPIQAL aggregation weights on MOS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="kadid10k")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--uncertainty-weights", type=Path, default=None,
        help="Path to trained Cholesky L ckpt; loaded at model init.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("weights/aggregation.pth"),
        help="Where to save the trained 8 parameters (default weights/aggregation.pth).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- dataset + reference-disjoint split -----
    pairs = load_dataset(
        args.dataset,
        root=args.dataset_root,
        limit=args.train_limit + args.val_limit,
    )
    train_all, val_all = held_out_split(
        pairs, val_fraction=args.val_fraction, seed=args.seed,
    )
    # Respect per-split limits.
    if args.train_limit and len(train_all) > args.train_limit:
        train_all = random.Random(args.seed).sample(train_all, args.train_limit)
    if args.val_limit and len(val_all) > args.val_limit:
        val_all = random.Random(args.seed + 1).sample(val_all, args.val_limit)
    print(
        f"[agg] train={len(train_all)} val={len(val_all)}  "
        f"(split by reference image, {args.val_fraction * 100:.0f}% val)"
    )

    # ---- model --------------------------------------------------------
    if args.uncertainty_weights is not None and args.uncertainty_weights.is_file():
        print(f"[agg] loading L from {args.uncertainty_weights}")
        model = UPIQAL(
            pretrained_vgg=True,
            uncertainty_parameterization="blockdiag",
            uncertainty_weights=args.uncertainty_weights,
        )
    else:
        print("[agg] identity-diagonal L (no ckpt)")
        model = UPIQAL(pretrained_vgg=True)

    trainable = _collect_aggregation_params(model)
    n_params = sum(p.numel() for p in trainable)
    print(f"[agg] training {len(trainable)} parameters, {n_params} scalars total")
    print(f"[agg] initial values: {_param_snapshot(trainable)}")

    optim = torch.optim.Adam(trainable, lr=args.lr)

    # ---- training loop -----------------------------------------------
    history: list[dict[str, float]] = []
    best_val_rank = float("inf")
    best_state: dict[str, float] | None = None

    for epoch in range(args.epochs):
        # Shuffle + batch train split.
        rng = random.Random(args.seed + epoch)
        train = train_all[:]
        rng.shuffle(train)

        model.train()
        t0 = time.perf_counter()
        train_losses: list[float] = []
        for i in range(0, len(train), args.batch_size):
            batch = train[i : i + args.batch_size]
            if len(batch) < 2:
                continue
            pred, mos = _score_batch(model, batch, args.max_side)
            # Flip UPIQAL score: DMOS is "higher worse", our score is
            # "higher better".  Ranking loss wants both aligned.
            loss = ranking_loss(-pred, mos)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            train_losses.append(float(loss.item()))

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_pred, val_mos = _score_batch(
                model, val_all, args.max_side,
            )
            val_loss = float(ranking_loss(-val_pred, val_mos).item())
        elapsed = time.perf_counter() - t0

        print(
            f"[agg] epoch {epoch + 1}/{args.epochs} "
            f"train_loss={sum(train_losses) / max(1, len(train_losses)):.4f} "
            f"val_loss={val_loss:.4f}  ({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(train_losses) / max(1, len(train_losses)),
                "val_loss": val_loss,
                "params": _param_snapshot(trainable),
            }
        )
        if val_loss < best_val_rank:
            best_val_rank = val_loss
            best_state = {
                "w_color": float(model.w_color.detach().item()),
                "w_anomaly": float(model.w_anomaly.detach().item()),
                "w_structure": float(model.w_structure.detach().item()),
                "w_heuristic": float(model.w_heuristic.detach().item()),
                "score_scale": float(model.score_scale.detach().item()),
                "score_center": float(model.score_center.detach().item()),
                "sigmoid_gain": float(model.deep_stats.sigmoid_gain.detach().item()),
                "sigmoid_bias": float(model.deep_stats.sigmoid_bias.detach().item()),
            }

    assert best_state is not None
    print(f"[agg] best val_loss={best_val_rank:.4f}")
    print(f"[agg] final parameters: {best_state}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "parameters": best_state,
            "history": history,
            "config": vars(args),
        },
        args.out,
    )
    print(f"[agg] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
