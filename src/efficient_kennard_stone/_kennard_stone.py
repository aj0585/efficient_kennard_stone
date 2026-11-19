"""Public API: Kennard-Stone selection and a train/test splitter."""

import numbers

import numpy as np
from numba import get_num_threads
from scipy.linalg import cholesky, solve_triangular
from sklearn.neighbors import BallTree
from sklearn.utils import check_array, indexable
from sklearn.utils.validation import _num_samples

from . import _kernels as K

__all__ = ["kennard_stone", "train_test_split"]

# Tree level whose node pairs seed the parallel diameter search (2^level nodes).
_DIAMETER_TASK_LEVEL = 6
# The dual-tree diameter search gives up, in favour of blocked BLAS brute force,
# once it has evaluated this fraction of all candidate pairs.
_DUALTREE_BUDGET_FRACTION = 1.0 / 32
_DUALTREE_MIN_BUDGET = 20_000_000
# Row block size of the blocked brute-force diameter search.
_DIAMETER_BLOCK = 2048
# Coarse tree nodes of about this many blocks are sorted by distance to the mean.
_DIAMETER_NODE_CHUNKS = 16
# Precision of the GEMM in that search. Any pair whose approximate squared
# distance comes within this relative slack of the best so far is recomputed
# exactly in float64, so the GEMM precision never affects the result.
_GEMM_DTYPE = np.float32
_GEMM_SLACK = 1e-4
# Selected samples buffered between batched updates (at most 62: bitmasks).
_BATCH = 62
# Lazy candidate refreshes allowed before a batch is applied early.
_REFRESH_BUDGET = 256
# Tree level at which batched updates are split into parallel tasks.
_SPLIT_LEVEL = 7


class _Tree:
    """Flat arrays of a scikit-learn BallTree (the tree object is discarded)."""

    def __init__(self, X, leaf_size):
        # BallTree works in float64; it copies float32 input only for the build.
        tree = BallTree(X, leaf_size=leaf_size)
        _, idx, node_data, node_bounds = tree.get_arrays()
        self.idx = np.ascontiguousarray(idx, dtype=np.int64)
        self.nstart = np.ascontiguousarray(node_data["idx_start"], dtype=np.int64)
        self.nend = np.ascontiguousarray(node_data["idx_end"], dtype=np.int64)
        self.is_leaf = np.ascontiguousarray(node_data["is_leaf"], dtype=np.bool_)
        self.radius = np.array(node_data["radius"], dtype=np.float64)
        self.centers = np.ascontiguousarray(node_bounds[0], dtype=np.float64)
        del tree, node_data, node_bounds
        self.n_nodes = self.nstart.shape[0]
        self.n_levels = int(np.log2(self.n_nodes + 1))
        self.first_leaf = self.n_nodes // 2
        # The kernels rely on a complete tree: every leaf on the last level.
        if not (self.is_leaf[self.first_leaf:].all()
                and not self.is_leaf[:self.first_leaf].any()):
            raise RuntimeError("unexpected BallTree layout")
        sizes = self.nend[self.first_leaf:] - self.nstart[self.first_leaf:]
        self.max_leaf = int(sizes.max())
        # Remaining rounding differences between sklearn's radii and ours are
        # absorbed by the relative tolerance in every bound (kernels._EPS).
        self.r = K.leaf_radii(X, self.idx, self.nstart, self.nend, self.centers,
                              self.radius, self.first_leaf)

    def position_of(self, rows):
        return np.array([np.flatnonzero(self.idx == v)[0] for v in rows], dtype=np.int64)


def _diameter(X, tree):
    """Exact most distant pair ``(i, j, distance)`` with ``i < j``."""
    n = X.shape[0]
    mean = X.mean(axis=0, dtype=np.float64)
    r0_rows = K.dist_to_point(X, mean)
    # Lower bound: farthest-point sweeps starting from the point farthest from the mean.
    a = int(K.argmax_first(r0_rows))
    L, bi, bj = -1.0, a, a
    for _ in range(10):
        da = K.dist_to_point(X, np.asarray(X[a], dtype=np.float64))
        b = int(K.argmax_first(da))
        if da[b] <= L:
            break
        L, bi, bj = float(da[b]), min(a, b), max(a, b)
        a = b
    del da
    if n < 2 or L <= 0.0:
        return 0, 1, 0.0

    # Distances to the mean, in tree order, give the bound d(p, q) <= r0_p + r0_q.
    r0 = r0_rows[tree.idx]
    del r0_rows
    r0max = K.node_max(r0, tree.nstart, tree.nend, tree.first_leaf)

    level = min(_DIAMETER_TASK_LEVEL, tree.n_levels - 1)
    nodes = np.arange(2 ** level - 1, 2 ** (level + 1) - 1)
    ga, gb = np.triu_indices(nodes.size)
    A, B = nodes[ga], nodes[gb]
    C = tree.centers
    ub = np.where(
        A == B,
        2.0 * tree.radius[A],
        np.linalg.norm(C[A] - C[B], axis=1) + tree.radius[A] + tree.radius[B],
    )
    ub = np.minimum(ub, r0max[A] + r0max[B]) * (1.0 + 1e-12)
    keep = np.flatnonzero(ub >= L)
    keep = keep[np.argsort(-ub[keep], kind="stable")]
    # Only samples with r0_p + max(r0) >= L can be part of the most distant pair.
    n_cand = int(np.count_nonzero((r0 + r0max[0]) * (1.0 + 1e-12) >= L))
    budget = max(_DUALTREE_MIN_BUDGET, int(_DUALTREE_BUDGET_FRACTION * n_cand * (n_cand - 1) / 2))
    d2, i, j, aborted = K.diameter_dualtree(
        X, tree.idx, tree.r, r0, tree.nstart, tree.nend, tree.is_leaf, tree.centers,
        tree.radius, r0max, A[keep].astype(np.int64), B[keep].astype(np.int64), L,
        tree.max_leaf, 4 * get_num_threads(), budget,
    )
    if aborted:
        # The bounds prune too little at leaf level (high intrinsic dimension):
        # brute force with BLAS, pruning only at the level of large nodes.
        return _diameter_blocked(X, mean, tree, r0, r0max, L, bi, bj)
    if i >= 0:
        return int(i), int(j), float(np.sqrt(d2))
    return int(bi), int(bj), float(L)


def _diameter_blocked(X, mean, tree, r0, r0max, L, bi, bj):
    """Exact most distant pair by blocked brute force with BLAS.

    The samples are split into chunks of about ``_DIAMETER_BLOCK``: each
    coarse tree node (about 16 chunks' worth of samples) is sorted by distance
    to the mean ``r0`` and cut into chunks. A chunk pair's upper bound is the
    smaller of the ball bound of their nodes and the sum of their largest
    ``r0``, so both spatially separated groups and points close to the mean
    are skipped. Chunk pairs are visited by decreasing bound until it drops
    below the best distance found. Squared distances come from
    ``|a|^2 + |b|^2 - 2 a.b`` with one GEMM per tile, and every pair that comes
    close to the best is recomputed exactly.
    """
    n = X.shape[0]
    blk = _DIAMETER_BLOCK
    level = int(np.clip(round(np.log2(max(1.0, n / (_DIAMETER_NODE_CHUNKS * blk)))), 0,
                        tree.n_levels - 1))
    # Chunks: (node, rows sorted by decreasing r0, max r0).
    ch_node, ch_rows, ch_r0max = [], [], []
    for v in range(2 ** level - 1, 2 ** (level + 1) - 1):
        lo, hi = tree.nstart[v], tree.nend[v]
        srt = np.argsort(-r0[lo:hi], kind="stable")
        rows, rr = tree.idx[lo:hi][srt], r0[lo:hi][srt]
        for c0 in range(0, hi - lo, blk):
            ch_node.append(v)
            ch_rows.append((rows[c0:c0 + blk], rr[c0:c0 + blk]))
            ch_r0max.append(rr[c0])
    ch_node, ch_r0max = np.array(ch_node), np.array(ch_r0max)
    ga, gb = np.triu_indices(ch_node.size)
    na, nb = ch_node[ga], ch_node[gb]
    C, rad = tree.centers, tree.radius
    ub = np.where(na == nb, 2.0 * rad[na],
                  np.linalg.norm(C[na] - C[nb], axis=1) + rad[na] + rad[nb])
    ub = np.minimum(ub, ch_r0max[ga] + ch_r0max[gb]) * (1.0 + 1e-12)
    keep = np.flatnonzero(ub >= L)
    keep = keep[np.argsort(-ub[keep], kind="stable")]
    A, B, ub = ga[keep], gb[keep], ub[keep]
    del ga, gb, na, nb, keep

    best_d2 = float(K._sqdist_rows(X, bi, bj)) if bi != bj else 0.0
    cache = {}

    def block(c, other):
        """Rows of chunk ``c`` that can still reach the best distance, centred."""
        rows, rr = ch_rows[c]
        cnt = int(np.count_nonzero((rr + ch_r0max[other]) * (1.0 + 1e-12) >= np.sqrt(best_d2)))
        key = (c, cnt)  # rows are sorted by r0, so the survivors are a prefix
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            Xv = (np.asarray(X[rows[:cnt]], dtype=np.float64) - mean).astype(_GEMM_DTYPE)
            cache[key] = (rows[:cnt], Xv, np.einsum("ij,ij->i", Xv, Xv, dtype=np.float64))
        return cache[key]

    for a, b, u in zip(A, B, ub):
        if u < np.sqrt(best_d2):
            break
        ri, Xi, sqi = block(a, b)
        rj, Xj, sqj = (ri, Xi, sqi) if a == b else block(b, a)
        if ri.shape[0] == 0 or rj.shape[0] == 0:
            continue
        G = Xi @ Xj.T
        # float32 dot products err by at most ~d * 6e-8 * |a| |b|.
        slack = (_GEMM_SLACK + 2e-7 * X.shape[1]) * (sqi.max() + sqj.max()) + 1e-300
        d2, i, j = K.tile_max(X, ri, rj, sqi, sqj, G, a == b, best_d2, slack)
        if i >= 0 and (d2 > best_d2 or (d2 == best_d2 and (i, j) < (bi, bj))):
            best_d2, bi, bj = d2, int(i), int(j)
    return int(bi), int(bj), float(np.sqrt(best_d2))


def _select(X, tree, first, n_select, stats=None):
    out_idx = np.empty(n_select, dtype=np.int64)
    out_dist = np.empty(n_select, dtype=np.float64)
    if stats is None:
        stats = np.zeros(2, dtype=np.int64)
    K.select_loop(X, tree.idx, tree.r, tree.nstart, tree.nend, tree.centers, tree.radius,
                  tree.first_leaf, tree.position_of(first[:n_select]), n_select,
                  _BATCH, _REFRESH_BUDGET, _SPLIT_LEVEL, out_idx, out_dist, stats)
    return out_idx, out_dist


def _whiten(X):
    """Map X so that Euclidean distances equal Mahalanobis distances."""
    Xc = np.asarray(X, dtype=np.float64) - X.mean(axis=0, dtype=np.float64)
    cov = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    Lc = cholesky(cov, lower=True)
    # Xw = Xc @ Lc^{-T}  <=>  Lc @ Xw^T = Xc^T
    return np.ascontiguousarray(solve_triangular(Lc, Xc.T, lower=True).T)


def kennard_stone(
    X,
    n_select=None,
    *,
    init="diameter",
    metric="euclidean",
    leaf_size=32,
    return_distances=False,
):
    """Select samples with the Kennard-Stone algorithm.

    The first two samples are the two most distant samples of ``X``; each
    following sample is the one whose distance to its nearest already-selected
    sample is largest. Ties go to the lowest row index, so the result is
    identical to the textbook O(n^2) implementation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data. float32 and float64 C-contiguous arrays are used without a copy.
    n_select : int, optional
        Number of samples to select. Defaults to ``n_samples`` (a complete
        Kennard-Stone ordering).
    init : {"diameter", "mean"}, int or sequence of int, default="diameter"
        How selection starts.

        - ``"diameter"``: the two most distant samples (classic Kennard-Stone).
          Exact, found by a pruned dual-tree search.
        - ``"mean"``: the sample farthest from the mean, then continue greedily.
        - int / sequence of int: these rows are selected first, in this order.
    metric : {"euclidean", "mahalanobis"}, default="euclidean"
        ``"mahalanobis"`` whitens the data first (allocates a float64 copy).
    leaf_size : int, default=32
        Leaf size of the ball tree used to prune distance computations. Only
        affects speed, never the result.
    return_distances : bool, default=False
        Also return, for each selected sample, its distance to the nearest
        sample selected before it (``inf`` for the first one).

    Returns
    -------
    indices : ndarray of shape (n_select,), int64
        Row indices of ``X`` in the order they were selected.
    distances : ndarray of shape (n_select,), float64
        Only if ``return_distances``.
    """
    X = check_array(X, dtype=[np.float64, np.float32], order="C", copy=False)
    n = X.shape[0]
    if n_select is None:
        n_select = n
    if not isinstance(n_select, numbers.Integral) or not 0 <= n_select <= n:
        raise ValueError(f"n_select must be an integer in [0, {n}], got {n_select!r}")
    n_select = int(n_select)
    if metric == "mahalanobis":
        X = _whiten(X)
    elif metric != "euclidean":
        raise ValueError(f"metric must be 'euclidean' or 'mahalanobis', got {metric!r}")

    if n_select == 0:
        empty = np.empty(0, dtype=np.int64)
        return (empty, np.empty(0)) if return_distances else empty

    if not isinstance(leaf_size, numbers.Integral) or leaf_size < 1:
        raise ValueError(f"leaf_size must be a positive integer, got {leaf_size!r}")
    mode = init if isinstance(init, str) else "indices"
    if mode == "indices":
        first = np.atleast_1d(np.asarray(init))
        if first.size == 0 or first.dtype.kind not in "iu":
            raise ValueError("init indices must be a non-empty sequence of integers")
        first = [int(v) for v in first.ravel()]
        if min(first) < -n or max(first) >= n or len({v % n for v in first}) != len(first):
            raise ValueError("init indices must be unique and within range")
        first = [v % n for v in first]
    elif mode not in ("diameter", "mean"):
        raise ValueError(f"init must be 'diameter', 'mean' or indices, got {init!r}")

    tree = _Tree(X, leaf_size)
    if mode == "diameter":
        first = [0] if n == 1 else list(_diameter(X, tree)[:2])
    elif mode == "mean":
        mean = X.mean(axis=0, dtype=np.float64)
        first = [int(K.argmax_first(K.dist_to_point(X, mean)))]
    out_idx, out_dist = _select(X, tree, first, n_select)
    if return_distances:
        return out_idx, out_dist
    return out_idx


def train_test_split(*arrays, test_size=None, train_size=None, **kwargs):
    """Split arrays into train and test subsets with Kennard-Stone.

    The training set is the Kennard-Stone selection on the first array (so it
    spans the data space); the remaining samples form the test set. Sizes
    follow :func:`sklearn.model_selection.train_test_split` (default test size
    0.25). Extra keyword arguments go to :func:`kennard_stone`.

    Returns
    -------
    splitting : list of length ``2 * len(arrays)``
        ``[X_train, X_test, y_train, y_test, ...]``. Training rows are in
        selection order; test rows keep their original order.
    """
    from sklearn.model_selection._split import _validate_shuffle_split
    from sklearn.utils import _safe_indexing

    if not arrays:
        raise ValueError("At least one array required as input")
    arrays = indexable(*arrays)
    n = _num_samples(arrays[0])
    n_train, _ = _validate_shuffle_split(n, test_size, train_size, default_test_size=0.25)
    train = kennard_stone(arrays[0], n_train, **kwargs)
    mask = np.ones(n, dtype=bool)
    mask[train] = False
    test = np.flatnonzero(mask)
    result = []
    for a in arrays:
        result.extend([_safe_indexing(a, train), _safe_indexing(a, test)])
    return result
