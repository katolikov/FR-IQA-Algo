"""Train the SUSS Cholesky factor L of the UPIQAL uncertainty mapper.

Self-supervised objective: for each reference image ``x``, generate an
imperceptibly-augmented copy ``x_tilde`` and fit the block-diagonal Cholesky
factor ``L`` of the precision matrix ``Sigma^-1 = L L^T`` such that the
VGG feature residual ``R = phi(x) - phi(x_tilde)`` has high likelihood
under ``N(0, Sigma)``.

Example
-------
    python train_uncertainty.py \\
        --data-dir ./some_image_dir \\
        --epochs 10 \\
        --batch-size 4 \\
        --crop 256 \\
        --lr 1e-3 \\
        --out weights/L_cholesky_blockdiag.pth

The resulting checkpoint can be loaded at inference via::

    python upiqal_cli.py ref.png tgt.png \\
        --uncertainty-weights weights/L_cholesky_blockdiag.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator, List

import torch

from upiqal.features import DeepStatisticalExtractor
from upiqal.normalize import Normalizer
from upiqal.suss_train import (
    AugmentConfig,
    ImperceptibleAugment,
    one_epoch,
)
from upiqal.uncertainty import ProbabilisticUncertaintyMapper


# Importable list of image-file extensions we'll pick up from a data dir.
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_image_paths(data_dir: Path) -> List[Path]:
    paths = [
        p for p in data_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
    ]
    paths.sort()
    if not paths:
        raise SystemExit(
            f"No images found under {data_dir!s} "
            f"(looked for {sorted(_IMG_EXTS)})"
        )
    return paths


def _load_image(path: Path, crop: int, device: torch.device) -> torch.Tensor:
    """Load an image, random-crop to ``crop x crop``, return (3, crop, crop)."""
    # Local import so the CLI is usable even if PIL is broken for some reason.
    from PIL import Image  # type: ignore
    import numpy as np

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < crop:
        # Upscale short side so we can take a crop.
        scale = crop / float(min(w, h))
        img = img.resize(
            (max(crop, int(round(w * scale))), max(crop, int(round(h * scale)))),
            Image.LANCZOS,
        )
        w, h = img.size
    left = random.randint(0, w - crop)
    top = random.randint(0, h - crop)
    img = img.crop((left, top, left + crop, top + crop))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
    return t


def _batches(
    paths: List[Path],
    batch_size: int,
    crop: int,
    device: torch.device,
    steps_per_epoch: int,
) -> Iterator[torch.Tensor]:
    """Yield ``steps_per_epoch`` random batches from ``paths``."""
    for _ in range(steps_per_epoch):
        chosen = random.sample(paths, k=min(batch_size, len(paths)))
        while len(chosen) < batch_size:
            chosen.append(random.choice(paths))
        tensors = [_load_image(p, crop, device) for p in chosen]
        yield torch.stack(tensors, dim=0)  # (B, 3, crop, crop)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the SUSS Cholesky factor for UPIQAL Module 4.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory of reference images (searched recursively).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--init-scale", type=float, default=1.0,
        help="Initial value of exp(log_diag); 1.0 -> identity precision.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("weights/L_cholesky_blockdiag.pth"),
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="torch device (default: cpu).  CUDA is supported if available.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr-schedule", choices=["constant", "cosine"], default="cosine",
        help=(
            "Learning-rate schedule.  'cosine' decays lr from --lr down to "
            "--lr-min over the full training horizon, which lets L converge "
            "without the last few epochs undoing early progress. "
            "'constant' keeps lr fixed (legacy)."
        ),
    )
    parser.add_argument(
        "--lr-min", type=float, default=1e-6,
        help="Minimum lr for the cosine schedule (default 1e-6).",
    )
    parser.add_argument(
        "--extra-data-dir", type=Path, default=None,
        help=(
            "Optional second image directory whose contents are concatenated "
            "with --data-dir.  Useful for mixing BSDS500 + COCO etc."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    paths = _collect_image_paths(args.data_dir)
    if args.extra_data_dir is not None and args.extra_data_dir.exists():
        extra = _collect_image_paths(args.extra_data_dir)
        paths = paths + extra
        print(
            f"[train] found {len(paths)} image(s) "
            f"(main={len(paths) - len(extra)} under {args.data_dir}, "
            f"extra={len(extra)} under {args.extra_data_dir})"
        )
    else:
        print(f"[train] found {len(paths)} image(s) under {args.data_dir}")

    normalizer = Normalizer(mode="imagenet").to(device).eval()
    deep_stats = DeepStatisticalExtractor(pretrained=True).to(device).eval()
    uncertainty = ProbabilisticUncertaintyMapper(
        parameterization="blockdiag",
        init_scale=args.init_scale,
    ).to(device)
    # Freeze VGG (already frozen internally, but belt-and-suspenders).
    for p in deep_stats.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(uncertainty.parameters(), lr=args.lr)
    augment = ImperceptibleAugment(AugmentConfig())

    # Cosine schedule: smoothly anneal from args.lr down to args.lr_min across
    # the full (epochs * steps_per_epoch) horizon.  One schedule step per
    # epoch is enough since our epochs are short (~200 steps).
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs, eta_min=args.lr_min,
        )
    else:
        scheduler = None

    args.out.parent.mkdir(parents=True, exist_ok=True)

    history: List[float] = []
    for epoch in range(args.epochs):
        batches = _batches(
            paths,
            batch_size=args.batch_size,
            crop=args.crop,
            device=device,
            steps_per_epoch=args.steps_per_epoch,
        )
        losses = one_epoch(
            batches=batches,
            normalizer=normalizer,
            deep_stats=deep_stats,
            uncertainty=uncertainty,
            augment=augment,
            optimizer=optim,
            grad_clip=args.grad_clip,
        )
        history.extend(losses)
        mean_loss = sum(losses) / max(1, len(losses))
        current_lr = optim.param_groups[0]["lr"]
        print(
            f"[train] epoch {epoch + 1}/{args.epochs} "
            f"mean_loss={mean_loss:.4f} "
            f"lr={current_lr:.2e} "
            f"(first={losses[0]:.4f} last={losses[-1]:.4f})"
        )
        if scheduler is not None:
            scheduler.step()

    ckpt = {
        "state_dict": uncertainty.state_dict(),
        "parameterization": "blockdiag",
        "config": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "crop": args.crop,
            "lr": args.lr,
            "init_scale": args.init_scale,
        },
        "history": history,
    }
    torch.save(ckpt, args.out)
    print(f"[train] saved checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
