"""From-scratch 0-dimensional persistent homology.

Key fact used throughout this project:

    For a Vietoris-Rips filtration of a finite metric space, the finite
    H0 death times are exactly the edge weights of a minimum spanning tree.

Reason: every point is born at eps = 0. As eps grows, two connected components
merge the first time an edge between them appears -- which is precisely what
Kruskal's algorithm does when it adds the cheapest edge joining two different
trees. So "run Kruskal with a union-find and record the weight of every merge"
*is* H0 persistence.

This module gives three implementations that must agree (see tests):
  * ``h0_persistence_union_find`` -- Kruskal + union-find, the textbook version
  * ``mst_weights_prim``          -- O(n^2) dense Prim in numpy, no dependencies
  * ``mst_weights`` (backend="scipy") -- fast C implementation, used in practice
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ----------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------
def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Dense pairwise distance matrix for an (n, d) point cloud."""
    X = np.asarray(X, dtype=np.float64)
    if metric == "euclidean":
        sq = np.einsum("ij,ij->i", X, X)
        D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2)
    elif metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + _EPS)
        D = 1.0 - np.clip(Xn @ Xn.T, -1.0, 1.0)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


# ----------------------------------------------------------------------------
# Union-find (disjoint set) -- the data structure behind H0
# ----------------------------------------------------------------------------
class UnionFind:
    """Disjoint-set forest with path halving and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.n_components = n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.n_components -= 1
        return True


def h0_persistence_union_find(D: np.ndarray) -> np.ndarray:
    """H0 persistence diagram of the Rips filtration on distance matrix ``D``.

    Returns an (n-1, 2) array of finite (birth, death) pairs; births are all 0.
    (The one infinite bar -- the component that never dies -- is omitted.)
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 2:
        return np.zeros((0, 2))
    iu, ju = np.triu_indices(n, k=1)
    w = D[iu, ju]
    order = np.argsort(w, kind="stable")
    uf = UnionFind(n)
    deaths = []
    for k in order:
        if uf.union(int(iu[k]), int(ju[k])):
            deaths.append(w[k])
            if uf.n_components == 1:
                break
    deaths = np.asarray(deaths)
    return np.column_stack([np.zeros_like(deaths), deaths])


# ----------------------------------------------------------------------------
# Minimum spanning tree weights (= H0 death times)
# ----------------------------------------------------------------------------
def mst_weights_prim(D: np.ndarray) -> np.ndarray:
    """Edge weights of an MST of the complete graph with weights ``D`` (dense Prim, O(n^2))."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 2:
        return np.zeros(0)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = D[0].copy()
    best[0] = np.inf
    out = np.empty(n - 1)
    for k in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = int(np.argmin(cand))
        out[k] = cand[j]
        in_tree[j] = True
        np.minimum(best, D[j], out=best)
    return out


def mst_weights_batched(D: np.ndarray) -> np.ndarray:
    """Prim's algorithm vectorised over a batch of dense graphs.

    D: (B, n, n) -> (B, n-1) MST edge weights (in insertion order, not sorted).
    Used for attention maps, where B = layers * heads = 144 graphs per text:
    one loop of n steps over (B, n) arrays is ~50x faster than 144 separate MSTs.
    """
    D = np.asarray(D, dtype=np.float64)
    B, n, _ = D.shape
    if n < 2:
        return np.zeros((B, 0))
    ar = np.arange(B)
    in_tree = np.zeros((B, n), dtype=bool)
    in_tree[:, 0] = True
    best = D[:, 0, :].copy()
    best[:, 0] = np.inf
    out = np.empty((B, n - 1))
    for k in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = cand.argmin(axis=1)
        out[:, k] = cand[ar, j]
        in_tree[ar, j] = True
        np.minimum(best, D[ar, j, :], out=best)
    return out


def mst_weights(D: np.ndarray, backend: str = "scipy") -> np.ndarray:
    """MST edge weights. ``backend`` in {"scipy", "prim", "unionfind"}."""
    if backend == "prim":
        return mst_weights_prim(D)
    if backend == "unionfind":
        return h0_persistence_union_find(D)[:, 1]
    if backend != "scipy":
        raise ValueError(backend)
    from scipy.sparse.csgraph import minimum_spanning_tree

    D = np.asarray(D, dtype=np.float64)
    if D.shape[0] < 2:
        return np.zeros(0)
    # scipy treats (near-)zero entries, |x| < ~1e-8, as "no edge", which would drop
    # edges between duplicate points. Adding the same constant to every edge does
    # not change which tree is minimal (all spanning trees have n-1 edges), so we
    # shift, solve, and shift back.
    shift = 1e-6
    Dp = D + shift
    np.fill_diagonal(Dp, 0.0)
    w = minimum_spanning_tree(Dp).data - shift
    return np.maximum(w, 0.0)


# ----------------------------------------------------------------------------
# Summaries of a persistence diagram (vectorisation)
# ----------------------------------------------------------------------------
def lifetimes(dgm: np.ndarray) -> np.ndarray:
    dgm = np.asarray(dgm, dtype=np.float64)
    if dgm.size == 0:
        return np.zeros(0)
    life = dgm[:, 1] - dgm[:, 0]
    return life[np.isfinite(life) & (life > 0)]


def persistence_entropy(life: np.ndarray) -> float:
    """Shannon entropy of normalised lifetimes (0 if empty)."""
    life = np.asarray(life, dtype=np.float64)
    s = life.sum()
    if life.size == 0 or s <= 0:
        return 0.0
    p = life / s
    return float(-(p * np.log(p + _EPS)).sum())


def diagram_summary(dgm: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Fixed-length statistics of a diagram -- robust to the number of bars."""
    life = lifetimes(dgm)
    n = life.size
    if n == 0:
        stats = dict(n_bars=0, total=0, total_sq=0, mean=0, std=0, max=0,
                     q25=0, q50=0, q75=0, entropy=0, norm_entropy=0, top5_share=0)
    else:
        srt = np.sort(life)[::-1]
        ent = persistence_entropy(life)
        stats = dict(
            n_bars=n,
            total=float(life.sum()),
            total_sq=float((life ** 2).sum()),
            mean=float(life.mean()),
            std=float(life.std()),
            max=float(srt[0]),
            q25=float(np.quantile(life, 0.25)),
            q50=float(np.quantile(life, 0.50)),
            q75=float(np.quantile(life, 0.75)),
            entropy=ent,
            norm_entropy=ent / np.log(n) if n > 1 else 0.0,
            top5_share=float(srt[:5].sum() / srt.sum()),
        )
    return {f"{prefix}{k}": float(v) for k, v in stats.items()}


def betti_curve(dgm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Number of bars alive at each filtration value in ``grid``."""
    dgm = np.asarray(dgm, dtype=np.float64)
    if dgm.size == 0:
        return np.zeros(len(grid))
    b, d = dgm[:, 0][:, None], dgm[:, 1][:, None]
    g = np.asarray(grid)[None, :]
    return ((b <= g) & (g < d)).sum(axis=0).astype(np.float64)
