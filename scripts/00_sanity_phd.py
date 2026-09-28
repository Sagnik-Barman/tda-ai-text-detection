"""Sanity check: does the from-scratch PHD estimator recover known dimensions?

Samples point clouds from shapes whose intrinsic dimension we know, embeds some of
them in high ambient dimension with a random rotation (as token embeddings live in
R^768), and compares PHD / MLE / TwoNN to the truth. Also checks the three MST
backends agree. Run this first -- if it fails, nothing downstream can be trusted.

    python scripts/00_sanity_phd.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect.persistence import mst_weights, pairwise_distances  # noqa: E402
from topodetect.phd import mle_dimension, phd, twonn_dimension  # noqa: E402


def random_rotation(d_in: int, d_out: int, rng) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d_out, d_out)))
    return Q[:, :d_in].T  # (d_in, d_out) with orthonormal rows


def sphere(n, k, rng):
    """Uniform on the k-sphere S^k in R^{k+1} (intrinsic dim k)."""
    x = rng.normal(size=(n, k + 1))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def swiss_roll(n, rng):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21 * rng.random(n)
    return np.column_stack([t * np.cos(t), h, t * np.sin(t)])


def torus(n, rng, R=3.0, r=1.0):
    u, v = 2 * np.pi * rng.random(n), 2 * np.pi * rng.random(n)
    return np.column_stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])


def main(n: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cases = [
        ("circle S^1 in R^768", 1, sphere(n, 1, rng) @ random_rotation(2, 768, rng)),
        ("sphere S^2 in R^3", 2, sphere(n, 2, rng)),
        ("swiss roll in R^3", 2, swiss_roll(n, rng)),
        ("torus T^2 in R^3", 2, torus(n, rng)),
        ("cube [0,1]^3 in R^768", 3, rng.random((n, 3)) @ random_rotation(3, 768, rng)),
        ("Gaussian R^5 in R^768", 5, rng.normal(size=(n, 5)) @ random_rotation(5, 768, rng)),
        ("cube [0,1]^8 in R^768", 8, rng.random((n, 8)) @ random_rotation(8, 768, rng)),
        ("Gaussian R^10", 10, rng.normal(size=(n, 10))),
    ]

    # 1) backends agree
    D = pairwise_distances(rng.normal(size=(200, 16)))
    a, b, c = (np.sort(mst_weights(D, k)) for k in ("scipy", "prim", "unionfind"))
    assert np.allclose(a, b) and np.allclose(a, c), "MST backends disagree!"
    print("MST backends (scipy / Prim / union-find) agree on 199 edges.\n")

    # 2) dimension recovery
    print(f"{'shape':<24}{'true':>5}{'PHD':>8}{'MLE':>8}{'TwoNN':>8}{'sec':>7}")
    print("-" * 60)
    rows = []
    for name, d_true, X in cases:
        t0 = time.time()
        d_phd = phd(X, seed=seed)
        dt = time.time() - t0
        d_mle, d_2nn = mle_dimension(X), twonn_dimension(X)
        rows.append((name, d_true, d_phd, d_mle, d_2nn))
        print(f"{name:<24}{d_true:>5}{d_phd:>8.2f}{d_mle:>8.2f}{d_2nn:>8.2f}{dt:>7.2f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, det = phd(cases[5][2], seed=seed, return_details=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].loglog(det["sizes"], det["lengths"], "o-")
        ax[0].set(title=f"MST length scaling (Gaussian R^5): slope={det['slope']:.3f}",
                  xlabel="n (subsample size)", ylabel="E_1 = sum of MST edges")
        names = [r[0] for r in rows]
        y = np.arange(len(rows))
        ax[1].scatter([r[1] for r in rows], y, marker="|", s=300, c="k", label="true")
        ax[1].scatter([r[2] for r in rows], y, label="PHD")
        ax[1].scatter([r[3] for r in rows], y, marker="x", label="MLE")
        ax[1].set_yticks(y, names, fontsize=8)
        ax[1].set_xlabel("intrinsic dimension")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / "phd_sanity.png", dpi=130)
        print(f"\nplot -> {out / 'phd_sanity.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
