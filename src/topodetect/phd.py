"""Intrinsic-dimension estimators for point clouds, written from scratch.

Persistent Homology Dimension (PHD)
-----------------------------------
Let X_n be n i.i.d. samples from a measure on a d-dimensional manifold and let

    E_alpha(X_n) = sum over MST edges e of |e|^alpha      (0 < alpha < d)

The MST edge lengths are exactly the finite H0 death times (see persistence.py),
so E_alpha is the "alpha-weighted total persistence" of H0. A classical result
(Steele 1988; Schweinhart 2019 for the PH version) says

    E_alpha(X_n) ~ C * n^{(d - alpha) / d}     as n -> infinity.

Take logs:  log E = const + m * log n   with slope   m = (d - alpha) / d,
so                           d = alpha / (1 - m).

Estimator: subsample the cloud at several sizes n_1 < ... < n_k, compute E_alpha
for each (median over a few random subsets), fit the slope m of log E vs log n
by least squares, and return alpha / (1 - m).

This mirrors the procedure of Tulchinskii et al., NeurIPS 2023 ("Intrinsic
Dimension Estimation for Robust Detection of AI-Generated Texts"): the point
cloud is the set of contextual token embeddings of one text; AI text tends to
have a lower PHD than human text.

Also included, as cheap complementary estimators:
  * MLE (Levina & Bickel 2004, with the MacKay-Ghahramani averaging)
  * TwoNN (Facco et al. 2017)
"""

from __future__ import annotations

import numpy as np

from .persistence import mst_weights, pairwise_distances


def mst_alpha_length(D: np.ndarray, alpha: float = 1.0, backend: str = "scipy") -> float:
    """E_alpha: sum of MST edge lengths raised to ``alpha``."""
    w = mst_weights(D, backend=backend)
    return float(np.sum(w ** alpha))


def phd(
    X: np.ndarray,
    alpha: float = 1.0,
    min_points: int = 40,
    max_points: int | None = None,
    n_sizes: int = 7,
    n_subsets: int = 7,
    n_reruns: int = 3,
    metric: str = "euclidean",
    backend: str = "scipy",
    seed: int | None = 0,
    return_details: bool = False,
):
    """Persistent-homology dimension of an (n, d) point cloud.

    Defaults follow the GPTID reference implementation: sizes from 40 up to
    (n - step) in 7 steps, median of several random subsets per size, and the
    slope averaged over 3 independent reruns.

    Returns ``nan`` if the cloud is too small (needs > min_points + n_sizes points).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if max_points is None:
        max_points = n
    max_points = min(max_points, n)
    if max_points - min_points < n_sizes:
        return (np.nan, {}) if return_details else np.nan

    step = max((max_points - min_points) // n_sizes, 1)
    sizes = np.arange(min_points, max_points - step + 1, step)
    if len(sizes) < 3:
        return (np.nan, {}) if return_details else np.nan

    rng = np.random.default_rng(seed)
    D_full = pairwise_distances(X, metric=metric)

    slopes, all_lengths = [], []
    for _ in range(n_reruns):
        lengths = []
        for s in sizes:
            reps = 3 if n <= 2 * s else n_subsets
            vals = []
            for _ in range(reps):
                idx = rng.choice(n, size=int(s), replace=False)
                vals.append(mst_alpha_length(D_full[np.ix_(idx, idx)], alpha, backend))
            lengths.append(np.median(vals))
        lengths = np.asarray(lengths)
        x, y = np.log(sizes), np.log(lengths)
        m = np.polyfit(x, y, 1)[0]
        slopes.append(m)
        all_lengths.append(lengths)

    m = float(np.mean(slopes))
    d = alpha / (1.0 - m) if m < 1 else np.nan
    if return_details:
        return d, dict(sizes=sizes, lengths=np.mean(all_lengths, axis=0), slope=m, slopes=slopes)
    return d


def mle_dimension(X: np.ndarray, k: int = 20, metric: str = "euclidean") -> float:
    """Levina-Bickel MLE of intrinsic dimension with MacKay-Ghahramani averaging."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = min(k, n - 1)
    if k < 3:
        return np.nan
    D = pairwise_distances(X, metric=metric)
    np.fill_diagonal(D, np.inf)
    T = np.sort(D, axis=1)[:, :k]  # T[:, j] = distance to (j+1)-th neighbour
    T = np.maximum(T, 1e-12)
    # inverse of the local estimate: (1/(k-1)) sum_{j<k} log(T_k / T_j)
    inv_m = np.log(T[:, -1:] / T[:, :-1]).mean(axis=1)
    inv_m = inv_m[np.isfinite(inv_m) & (inv_m > 0)]
    if inv_m.size == 0:
        return np.nan
    return float(1.0 / inv_m.mean())


def twonn_dimension(X: np.ndarray, discard_fraction: float = 0.1, metric: str = "euclidean") -> float:
    """TwoNN estimator: uses only the ratio of 2nd to 1st nearest-neighbour distances."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 5:
        return np.nan
    D = pairwise_distances(X, metric=metric)
    np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    ok = r[:, 0] > 1e-12
    mu = np.sort(r[ok, 1] / r[ok, 0])
    N = mu.size
    if N < 5:
        return np.nan
    # Facco et al.: the empirical CDF satisfies -log(1 - F(mu)) = d * log(mu).
    # Fit that line through the origin on the lower (1 - discard_fraction) of the data.
    F = np.arange(1, N + 1) / N
    keep = slice(0, int(N * (1 - discard_fraction)))
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom > 0 else np.nan
