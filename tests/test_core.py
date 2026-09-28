"""Unit tests for the from-scratch maths. Run: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topodetect.features.attention import (DEFAULT_THRESHOLDS, attention_features,  # noqa: E402
                                           betti_all_thresholds, feature_names, graph_betti)
from topodetect.features.cloud import cloud_features  # noqa: E402
from topodetect.persistence import (UnionFind, betti_curve, diagram_summary,  # noqa: E402
                                    h0_persistence_union_find, mst_weights, mst_weights_batched,
                                    pairwise_distances)
from topodetect.phd import mle_dimension, phd, twonn_dimension  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_union_find_basic():
    uf = UnionFind(5)
    assert uf.union(0, 1) and uf.union(3, 4) and not uf.union(1, 0)
    assert uf.n_components == 3
    assert uf.find(0) == uf.find(1) and uf.find(0) != uf.find(3)


def test_mst_backends_agree(rng):
    D = pairwise_distances(rng.normal(size=(120, 7)))
    ref = np.sort(mst_weights(D, "scipy"))
    assert np.allclose(ref, np.sort(mst_weights(D, "prim")))
    assert np.allclose(ref, np.sort(mst_weights(D, "unionfind")))
    Ds = np.stack([D, pairwise_distances(rng.normal(size=(120, 3)))])
    assert np.allclose(ref, np.sort(mst_weights_batched(Ds)[0]))


def test_h0_matches_ripser(rng):
    ripser = pytest.importorskip("ripser").ripser
    X = rng.normal(size=(80, 4))
    ours = np.sort(h0_persistence_union_find(pairwise_distances(X))[:, 1])
    theirs = ripser(X, maxdim=0)["dgms"][0][:, 1]
    theirs = np.sort(theirs[np.isfinite(theirs)])
    assert np.allclose(ours, theirs, atol=1e-5)


def test_mst_handles_duplicate_points():
    X = np.array([[0.0, 0], [0, 0], [1, 0], [1, 0], [3, 0]])
    w = np.sort(mst_weights(pairwise_distances(X)))
    assert np.allclose(w, [0, 0, 1, 2])


@pytest.mark.parametrize("d", [1, 2, 4])
def test_phd_recovers_dimension(rng, d):
    X = rng.random((500, d)) @ np.linalg.qr(rng.normal(size=(64, 64)))[0][:d]  # embed in R^64
    assert abs(phd(X, seed=0) - d) < 0.35 * d + 0.2
    assert abs(mle_dimension(X) - d) < 0.35 * d + 0.3
    assert abs(twonn_dimension(X) - d) < 0.4 * d + 0.3


def test_phd_too_small_is_nan(rng):
    assert np.isnan(phd(rng.normal(size=(30, 5))))


def test_diagram_summary_and_betti_curve():
    dgm = np.array([[0, 1.0], [0, 3.0], [0.5, 1.5]])
    s = diagram_summary(dgm)
    assert s["n_bars"] == 3 and s["total"] == pytest.approx(5.0) and s["max"] == pytest.approx(3.0)
    assert list(betti_curve(dgm, [0.25, 1.2, 2.0])) == [2, 2, 1]
    assert diagram_summary(np.zeros((0, 2)))["n_bars"] == 0


def test_betti_from_mst_equals_graph_search(rng):
    for _ in range(25):
        T = int(rng.integers(4, 60))
        A = rng.dirichlet(np.full(T, 0.1), size=T)
        S = np.maximum(A, A.T)
        np.fill_diagonal(S, 0)
        D = 1 - S
        np.fill_diagonal(D, 0)
        fast = betti_all_thresholds(S, mst_weights(D), DEFAULT_THRESHOLDS)
        slow = sum((list(graph_betti(S, t)) for t in DEFAULT_THRESHOLDS), [])
        assert fast == slow


def test_attention_features_shape_and_values(rng):
    L, H, T = 2, 3, 30
    A = rng.dirichlet(np.full(T, 0.2), size=(L, H, T))
    v = attention_features(A, tokens=["<s>"] + ["Ġa", "Ġ,"] * 14 + ["</s>"])
    assert v.shape == (len(feature_names(L, H)),)
    assert np.isfinite(v).all()
    # an identity-like "attend to self" head should give pat_self == 1
    eye = np.broadcast_to(np.eye(T), (1, 1, T, T)).copy()
    names = feature_names(1, 1)
    f = dict(zip(names, attention_features(eye)))
    assert f["attn_L0H0_pat_self"] == pytest.approx(1.0)
    assert f["attn_L0H0_b0@0.5"] == pytest.approx(1.0)  # T isolated points / T


def test_cloud_features_keys(rng):
    f = cloud_features(rng.normal(size=(120, 16)), h1=True)
    for k in ("cloud_phd", "cloud_mle", "cloud_twonn", "cloud_h0_total", "cloud_n_tokens"):
        assert k in f and np.isfinite(f[k])
