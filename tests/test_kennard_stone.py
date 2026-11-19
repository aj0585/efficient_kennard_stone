import numpy as np
import pytest
from scipy.spatial.distance import cdist

from efficient_kennard_stone import kennard_stone, train_test_split


def naive_kennard_stone(X, n_select, init="diameter"):
    """Textbook O(n^2) Kennard-Stone with lowest-index tie-breaking."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D = cdist(X, X)
    if init == "diameter":
        i, j = np.unravel_index(np.argmax(D), D.shape)
        first = [int(min(i, j)), int(max(i, j))]
    else:
        first = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    first = list(dict.fromkeys(first))[:n_select]  # n == 1 gives (0, 0)
    sel = list(first)
    dist = [np.inf] + ([D[first[0], first[1]]] if len(first) == 2 else [])
    m = D[first].min(axis=0)
    m[sel] = -1
    while len(sel) < n_select:
        s = int(np.argmax(m))
        sel.append(s)
        dist.append(m[s])
        m = np.minimum(m, D[s])
        m[sel] = -1
    return np.array(sel), np.array(dist)


@pytest.mark.parametrize("n,d", [(1, 3), (2, 2), (7, 1), (50, 2), (300, 5), (2500, 3), (3000, 20)])
@pytest.mark.parametrize("leaf_size", [1, 7, 32])
def test_matches_naive(n, d, leaf_size):
    rng = np.random.default_rng(n * 31 + d)
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d)
    idx, dist = kennard_stone(X, leaf_size=leaf_size, return_distances=True)
    ref_idx, ref_dist = naive_kennard_stone(X, n)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-10)


@pytest.mark.parametrize("leaf_size", [1, 5, 40])
def test_ties_on_integer_grid(leaf_size):
    # All distances are exact, so every tie must be broken by the lowest index.
    g = np.arange(6, dtype=np.float64)
    X = np.array(np.meshgrid(g, g, g[:4])).reshape(3, -1).T.copy()
    X = np.vstack([X, X[:10]])  # exact duplicates too
    idx = kennard_stone(X, leaf_size=leaf_size)
    ref, _ = naive_kennard_stone(X, len(X))
    np.testing.assert_array_equal(idx, ref)


def test_identical_points():
    X = np.ones((20, 3))
    np.testing.assert_array_equal(kennard_stone(X), np.arange(20))


def test_partial_selection_and_float32():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(5000, 4)).astype(np.float32)
    idx = kennard_stone(X, 500)
    ref, _ = naive_kennard_stone(X, 500)
    np.testing.assert_array_equal(idx, ref)


def test_mean_init():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 3))
    idx = kennard_stone(X, 100, init="mean", leaf_size=10)
    ref, _ = naive_kennard_stone(X, 100, init="mean")
    np.testing.assert_array_equal(idx, ref)


def test_explicit_init():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 2))
    idx = kennard_stone(X, 50, init=[5, 17])
    assert list(idx[:2]) == [5, 17]
    D = cdist(X, X)
    m = D[[5, 17]].min(0)
    m[[5, 17]] = -1
    for s in idx[2:]:
        assert s == np.argmax(m)
        m = np.minimum(m, D[s])
        m[idx[: list(idx).index(s) + 1]] = -1


def test_mahalanobis_equals_euclidean_on_whitened():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 3)) @ np.array([[3, 0, 0], [1, 1, 0], [0, 2, 0.5]])
    VI = np.linalg.inv(np.cov(X.T))
    D = cdist(X, X, "mahalanobis", VI=VI)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    idx = kennard_stone(X, 30, metric="mahalanobis")
    assert set(idx[:2]) == {i, j}


def test_leaf_size_does_not_change_result():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(c, 0.3, size=(1500, 3)) for c in (0, 5, 10)])
    a = kennard_stone(X, 2000, leaf_size=1)
    b = kennard_stone(X, 2000, leaf_size=60)
    c = kennard_stone(X, 2000, leaf_size=4500)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_train_test_split():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 3))
    y = np.arange(200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    assert len(X_tr) == 160 and len(X_te) == 40
    np.testing.assert_array_equal(y_tr, kennard_stone(X, 160))
    assert set(y_tr).isdisjoint(y_te) and len(set(y_tr) | set(y_te)) == 200
    np.testing.assert_array_equal(X_te, X[np.sort(y_te)])


def test_validation():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        kennard_stone(X, 11)
    with pytest.raises(ValueError):
        kennard_stone(X, 3, init=[1, 1])
    with pytest.raises(ValueError):
        kennard_stone(X, 3, metric="cosine")
    with pytest.raises(ValueError):
        kennard_stone(np.array([[np.nan, 1.0]]))
    assert kennard_stone(X, 0).shape == (0,)


@pytest.mark.parametrize("d", [3, 40])
def test_blocked_diameter_path(monkeypatch, d):
    # Force the dual-tree search to give up so the blocked BLAS search runs.
    from efficient_kennard_stone import _kennard_stone as mod

    monkeypatch.setattr(mod, "_DUALTREE_MIN_BUDGET", 0)
    monkeypatch.setattr(mod, "_DUALTREE_BUDGET_FRACTION", 0.0)
    monkeypatch.setattr(mod, "_DIAMETER_BLOCK", 97)
    rng = np.random.default_rng(d)
    X = rng.normal(size=(1500, d))
    idx = kennard_stone(X, 40)
    ref, _ = naive_kennard_stone(X, 40)
    np.testing.assert_array_equal(idx, ref)
    # Exact ties: integer grid.
    g = np.arange(5, dtype=np.float64)
    G = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T.copy()
    np.testing.assert_array_equal(kennard_stone(G, 30), naive_kennard_stone(G, 30)[0])


def test_init_array_and_leaf_size_validation():
    X = np.random.default_rng(0).normal(size=(50, 2))
    np.testing.assert_array_equal(kennard_stone(X, 5, init=np.array([3]))[:1], [3])
    with pytest.raises(ValueError):
        kennard_stone(X, 5, init=np.array([0.5]))
    with pytest.raises(ValueError):
        kennard_stone(X, 5, leaf_size=0)
