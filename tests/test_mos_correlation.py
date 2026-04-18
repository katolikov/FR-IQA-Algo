"""Unit tests for the MOS correlation primitives in eval/mos_correlation.py.

We don't exercise the full UPIQAL pipeline here (that's integration);
these tests just make sure SROCC/PLCC/RMSE + the KADID loader work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.mos_correlation import (  # noqa: E402
    logistic_fit_and_rmse,
    pearson_r,
    spearman_rho,
    _rankdata,
)


def test_rankdata_handles_ties() -> None:
    r = _rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    # Values: 10 → rank 1, 20s tie at ranks 2 & 3 → both get 2.5, 30 → rank 4.
    assert r.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_monotone() -> None:
    x = np.arange(20, dtype=np.float64)
    y = x ** 2  # strictly monotone in x
    rho = spearman_rho(x, y)
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_spearman_negated_is_minus_one() -> None:
    x = np.arange(20, dtype=np.float64)
    rho = spearman_rho(x, -x)
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_spearman_uncorrelated_near_zero() -> None:
    rng = np.random.default_rng(42)
    rho = spearman_rho(rng.standard_normal(200), rng.standard_normal(200))
    assert abs(rho) < 0.25  # independent draws ⇒ low |ρ|


def test_pearson_perfect_linear() -> None:
    x = np.arange(10, dtype=np.float64)
    y = 3.0 * x + 5.0
    r = pearson_r(x, y)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_logistic_fit_perfect_monotone() -> None:
    """If the predictor is a perfect monotone transform of MOS, the
    logistic fit should recover near-1.0 PLCC and near-0 RMSE."""
    mos = np.linspace(1.0, 5.0, 200, dtype=np.float64)
    pred = 1.0 / (1.0 + np.exp(-(mos - 3.0)))  # strictly monotone in MOS
    plcc, rmse, _ = logistic_fit_and_rmse(pred, mos)
    assert plcc > 0.99
    assert rmse < 0.15  # MOS scale is 1..5, so this is ~3 % rel error


def test_kadid_loader_handles_missing_dataset_gracefully() -> None:
    """Trying to load from a non-existent root raises FileNotFoundError
    with an actionable message (pointing at scripts/download_kadid10k.py)."""
    from eval.datasets import load_kadid10k

    with pytest.raises(FileNotFoundError, match="download_kadid10k"):
        load_kadid10k(root=Path("/nonexistent/kadid/root"))


def test_held_out_split_disjoint_by_reference() -> None:
    """`held_out_split` guarantees no reference image appears in both
    train and val — that's the whole point of splitting by reference."""
    from eval.datasets import MOSPair, held_out_split

    pairs = []
    for ref_id in range(10):
        for lvl in range(5):
            pairs.append(
                MOSPair(
                    ref_path=Path(f"/fake/I{ref_id:02d}.png"),
                    dist_path=Path(f"/fake/I{ref_id:02d}_01_{lvl + 1}.png"),
                    mos=float(lvl),
                    distortion_type="fake",
                    distortion_level=lvl + 1,
                )
            )
    train, val = held_out_split(pairs, val_fraction=0.2, seed=0)
    train_refs = {p.ref_path.stem for p in train}
    val_refs = {p.ref_path.stem for p in val}
    assert train_refs.isdisjoint(val_refs)
    assert len(val_refs) >= 1
    assert len(train) + len(val) == len(pairs)
