"""MOS-labelled IQA dataset loaders for the UPIQAL eval harness.

Each loader yields :class:`MOSPair` records; callers iterate, run UPIQAL,
and accumulate ``(predicted_score, mos, distortion_type)`` triples for
correlation analysis.

Supported datasets
------------------
* **KADID-10k** (Lin et al. 2019) — 81 reference images × 25 distortion
  types × 5 levels = 10 125 distorted images, MOS labels in ``dmos.csv``.
  Default location: ``/tmp/kadid/kadid10k/kadid10k/`` (matches the
  ``scripts/download_kadid10k.py`` output path).

* **TID2013** (Ponomarenko et al. 2013) — 25 references × 24 distortion
  types × 5 levels = 3 000 images, MOS in ``mos.txt``.
  *Not yet implemented — placeholder for future expansion.*
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class MOSPair:
    """One evaluation pair: reference + distorted + human rating."""

    ref_path: Path
    dist_path: Path
    mos: float
    distortion_type: str  # free-form label, e.g. "gaussian_blur"
    distortion_level: int  # 1..5 for KADID, 1..5 for TID2013


# -------------------------------------------------------------------------
# KADID-10k
# -------------------------------------------------------------------------
# Distortion type codes in KADID-10k filenames follow the pattern
# I<ref>_<type>_<level>.png where <type> is 01..25.  The README ships
# these category names:
_KADID_DISTORTION_TYPES = {
    "01": "gaussian_blur",
    "02": "lens_blur",
    "03": "motion_blur",
    "04": "color_diffusion",
    "05": "color_shift",
    "06": "color_quantisation",
    "07": "color_saturation_hsv",
    "08": "color_saturation_yvc",
    "09": "jpeg2000",
    "10": "jpeg",
    "11": "white_noise",
    "12": "white_noise_cc",  # colour-correlated
    "13": "impulse_noise",
    "14": "multiplicative_noise",
    "15": "denoise",
    "16": "brightness_change",
    "17": "mean_shift",
    "18": "jitter",
    "19": "non_eccentricity_patch",
    "20": "pixelate",
    "21": "quantisation",
    "22": "colour_block",
    "23": "high_sharpen",
    "24": "contrast_change",
    "25": "change_saturation",
}


def load_kadid10k(
    root: Path = Path("/tmp/kadid/kadid10k/kadid10k"),
    limit: Optional[int] = None,
    seed: int = 0,
) -> List[MOSPair]:
    """Load KADID-10k MOS pairs.

    Parameters
    ----------
    root : Path
        Directory containing ``dmos.csv`` and ``images/``.  Default path
        matches the ``scripts/download_kadid10k.py`` output.
    limit : int, optional
        If set, return a deterministic random subsample of this many
        pairs (seeded by ``seed``).  Useful for smoke tests.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    list[MOSPair]
        Ordered by reference ID, then distortion type, then level.
    """
    csv_path = root / "dmos.csv"
    imgs = root / "images"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"KADID dmos.csv not found at {csv_path}.  "
            "Run scripts/download_kadid10k.py first."
        )
    if not imgs.is_dir():
        raise FileNotFoundError(f"KADID images dir missing: {imgs}")

    pairs: List[MOSPair] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dist = imgs / row["dist_img"]
            ref = imgs / row["ref_img"]
            if not dist.is_file() or not ref.is_file():
                continue  # silently skip missing files (AppleDouble etc.)
            # dist filename: "I<ref>_<type>_<level>.png"
            stem = dist.stem.split("_")
            type_code = stem[1] if len(stem) >= 3 else "??"
            level = int(stem[2]) if len(stem) >= 3 and stem[2].isdigit() else 0
            pairs.append(
                MOSPair(
                    ref_path=ref,
                    dist_path=dist,
                    mos=float(row["dmos"]),
                    distortion_type=_KADID_DISTORTION_TYPES.get(
                        type_code, f"unknown_{type_code}"
                    ),
                    distortion_level=level,
                )
            )

    if limit is not None and limit > 0 and limit < len(pairs):
        import random
        rng = random.Random(seed)
        pairs = rng.sample(pairs, limit)

    return pairs


# -------------------------------------------------------------------------
# Dispatcher
# -------------------------------------------------------------------------
def load_dataset(
    name: str,
    root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[MOSPair]:
    """Single entry point used by ``mos_correlation.py``."""
    name = name.lower()
    if name in ("kadid10k", "kadid", "kadid-10k"):
        return load_kadid10k(
            root=root or Path("/tmp/kadid/kadid10k/kadid10k"),
            limit=limit,
        )
    raise ValueError(
        f"unknown dataset {name!r}; available: 'kadid10k'"
    )


def held_out_split(
    pairs: Iterable[MOSPair], val_fraction: float = 0.2, seed: int = 0
) -> tuple[List[MOSPair], List[MOSPair]]:
    """Split by REFERENCE image (not distorted pair) so the same ref
    never appears in both folds.  Prevents inflated SROCC from
    memorising reference textures during Phase-2 training."""
    import random

    pairs = list(pairs)
    refs = sorted({p.ref_path.stem for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(refs)
    n_val = max(1, int(round(len(refs) * val_fraction)))
    val_refs = set(refs[:n_val])
    train = [p for p in pairs if p.ref_path.stem not in val_refs]
    val = [p for p in pairs if p.ref_path.stem in val_refs]
    return train, val
