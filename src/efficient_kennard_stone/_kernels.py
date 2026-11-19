"""Numba-compiled kernels for the Kennard-Stone algorithm.

The samples are organised by a ball tree (built by scikit-learn's Cython
``BallTree``). Samples are addressed in *tree order*: ``idx[p]`` is the row of
``X`` stored at position ``p``. Node ``v`` covers positions
``nstart[v]:nend[v]`` and has children ``2v+1`` and ``2v+2``; the tree is
complete, so all leaves are on the last level. ``X`` is never copied.

Selection state
---------------
* ``m[p]``   an upper bound on the distance from sample ``p`` to the selected
             set; it is exact for the first ``max(ver[p], v_sync)`` selected
             samples (``-1`` once ``p`` itself is selected).
* ``mu[v]``, ``amu[v]``  max of ``m`` over node ``v`` and its position
             (``-1`` when every sample of the node is selected); ties go to the
             smaller row index, so ``amu[0]`` is the global candidate.

Selected samples are buffered (``S``, up to ``batch`` rows) and applied to all
samples together by :func:`_sync`, a parallel tree traversal that skips
every node / sample whose triangle-inequality lower bound shows it cannot be
improved. Between syncs, the candidate ``amu[0]`` is refreshed against the
buffered samples only; when its refreshed value is still the largest upper
bound, it is the exact Kennard-Stone choice. This turns the memory-bound
one-sample-per-pass update into a compute-bound batched one, while pruning
removes most distance computations for data of low intrinsic dimension.
"""

import numpy as np
from numba import njit, prange

# Allow re-association (SIMD reductions) and FMA, but keep exact sqrt and
# IEEE handling of inf (the "nothing selected yet" distance).
_FASTMATH = {"reassoc", "contract", "nsz"}

# Bounds are loosened by this fraction of the magnitudes involved, so that
# rounding in a bound can never cause a needed update to be skipped.
_EPS = 1e-12

_STACK = 512  # DFS stack; trees here have < 64 levels


@njit(cache=True, fastmath=_FASTMATH, inline="always")
def _sqdist_vec(x, X, j):
    """Squared distance between float64 vector ``x`` and row ``X[j]``."""
    acc = 0.0
    for t in range(x.shape[0]):
        diff = x[t] - np.float64(X[j, t])
        acc += diff * diff
    return acc


@njit(cache=True, fastmath=_FASTMATH, inline="always")
def _sqdist_rows(X, i, j):
    acc = 0.0
    for t in range(X.shape[1]):
        diff = np.float64(X[i, t]) - np.float64(X[j, t])
        acc += diff * diff
    return acc


@njit(cache=True, fastmath=_FASTMATH, inline="always")
def _sqdist_cc(C, a, b):
    acc = 0.0
    for t in range(C.shape[1]):
        diff = C[a, t] - C[b, t]
        acc += diff * diff
    return acc


@njit(cache=True, inline="always")
def _better(m_new, i_new, m_old, i_old):
    """Larger distance wins; equal distances go to the smaller row index."""
    return m_new > m_old or (m_new == m_old and i_new < i_old)


@njit(cache=True, inline="always")
def _pair_better(d_new, i_new, j_new, d_old, i_old, j_old):
    """Larger distance wins; ties go to the lexicographically smaller pair."""
    if d_new > d_old:
        return True
    if d_new == d_old:
        return i_new < i_old or (i_new == i_old and j_new < j_old)
    return False


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def dist_to_point(X, x):
    """Euclidean distance from every row of ``X`` to the float64 vector ``x``."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = np.sqrt(_sqdist_vec(x, X, i))
    return out


@njit(cache=True)
def argmax_first(v):
    """Index of the maximum of ``v``; the lowest index wins ties."""
    best = 0
    bv = v[0]
    for i in range(1, v.shape[0]):
        if v[i] > bv:
            bv = v[i]
            best = i
    return best


@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def leaf_radii(X, idx, nstart, nend, centers, radius, first_leaf):
    """Distance of each sample (tree order) to its leaf centroid.

    Leaf radii are raised to the largest such distance if needed, so they are
    consistent with the per-sample values computed here."""
    r = np.empty(idx.shape[0], dtype=np.float64)
    n_nodes = nstart.shape[0]
    for v in prange(first_leaf, n_nodes):
        c = centers[v]
        best = 0.0
        for p in range(nstart[v], nend[v]):
            dp = np.sqrt(_sqdist_vec(c, X, idx[p]))
            r[p] = dp
            if dp > best:
                best = dp
        if best > radius[v]:
            radius[v] = best
    return r


@njit(cache=True)
def node_max(values, nstart, nend, first_leaf):
    """Per-node maximum of a per-position array."""
    n_nodes = nstart.shape[0]
    out = np.empty(n_nodes, dtype=np.float64)
    for v in range(first_leaf, n_nodes):
        best = -np.inf
        for p in range(nstart[v], nend[v]):
            if values[p] > best:
                best = values[p]
        out[v] = best
    for v in range(first_leaf - 1, -1, -1):
        out[v] = max(out[2 * v + 1], out[2 * v + 2])
    return out


# --------------------------------------------------------------------------
# Diameter (most distant pair): parallel dual-tree branch and bound
# --------------------------------------------------------------------------

@njit(cache=True, fastmath=_FASTMATH, inline="always")
def _pair_ub(centers, radius, r0max, a, b):
    """Upper bound on distances between samples of nodes ``a`` and ``b``: the
    smaller of the ball bound and the bound through the global mean."""
    if a == b:
        ub = 2.0 * radius[a]
    else:
        ub = np.sqrt(_sqdist_cc(centers, a, b)) + radius[a] + radius[b]
    ub0 = r0max[a] + r0max[b]
    if ub0 < ub:
        ub = ub0
    return ub * (1.0 + _EPS)


@njit(cache=True, fastmath=_FASTMATH)
def _leaf_pair_max(X, idx, r, r0, nstart, nend, centers, radius, r0max, a, b, thr, P, Q):
    """Most distant pair between leaves ``a`` and ``b``, among samples that can
    still reach ``thr``. Returns (squared distance, i, j, pairs evaluated)
    with ``i < j``."""
    dab = 0.0 if a == b else np.sqrt(_sqdist_cc(centers, a, b))
    nP = 0
    for p in range(nstart[a], nend[a]):
        if (r[p] + dab + radius[b]) * (1.0 + _EPS) >= thr and \
                (r0[p] + r0max[b]) * (1.0 + _EPS) >= thr:
            P[nP] = idx[p]
            nP += 1
    nQ = 0
    if a == b:
        for t in range(nP):
            Q[t] = P[t]
        nQ = nP
    else:
        for q in range(nstart[b], nend[b]):
            if (r[q] + dab + radius[a]) * (1.0 + _EPS) >= thr and \
                    (r0[q] + r0max[a]) * (1.0 + _EPS) >= thr:
                Q[nQ] = idx[q]
                nQ += 1
    bd = -1.0
    bi = -1
    bj = -1
    for s in range(nP):
        i = P[s]
        q0 = s + 1 if a == b else 0
        for t in range(q0, nQ):
            j = Q[t]
            d2 = _sqdist_rows(X, i, j)
            lo = min(i, j)
            hi = max(i, j)
            if bi < 0 or _pair_better(d2, lo, hi, bd, bi, bj):
                bd = d2
                bi = lo
                bj = hi
    n_eval = nP * (nP - 1) // 2 if a == b else nP * nQ
    return bd, bi, bj, n_eval


@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def diameter_dualtree(X, idx, r, r0, nstart, nend, is_leaf, centers, radius, r0max,
                      task_a, task_b, L0, max_leaf, n_workers, budget):
    """Exact most distant pair given a lower bound ``L0`` on the diameter.

    ``task_a, task_b`` are node pairs (sorted by decreasing upper bound) that
    together cover all sample pairs; they are dealt round-robin to workers,
    each of which explores its node pairs depth-first, pruning any node pair
    whose upper bound cannot reach the best distance found so far by any
    worker. ``r0`` / ``r0max`` are distances to the global mean (per position /
    per-node maximum), which give a second, often tighter, upper bound.

    If the workers together evaluate more than ``budget`` sample pairs (the
    bounds are not pruning, so brute force would be cheaper), the search stops
    early and reports ``aborted``.

    Returns (squared distance, i, j, aborted) with ``i < j``; ``i = -1`` if no
    pair reaches ``L0`` (impossible when ``L0`` is an actual pair distance).
    """
    n_tasks = task_a.shape[0]
    n_workers = max(1, min(n_tasks, n_workers))
    shared_L = np.full(1, L0)  # read/written racily; any value is a valid bound
    abort = np.zeros(1, dtype=np.int64)
    local_budget = budget // n_workers + 1
    w_d = np.full(n_workers, -1.0)
    w_i = np.full(n_workers, -1, dtype=np.int64)
    w_j = np.full(n_workers, -1, dtype=np.int64)
    for w in prange(n_workers):
        sa = np.empty(_STACK, dtype=np.int64)
        sb = np.empty(_STACK, dtype=np.int64)
        P = np.empty(max_leaf, dtype=np.int64)
        Q = np.empty(max_leaf, dtype=np.int64)
        bd = -1.0
        bi = -1
        bj = -1
        used = 0
        for t in range(w, n_tasks, n_workers):
            sa[0] = task_a[t]
            sb[0] = task_b[t]
            top = 1
            while top > 0 and abort[0] == 0:
                top -= 1
                a = sa[top]
                b = sb[top]
                L = shared_L[0]
                if _pair_ub(centers, radius, r0max, a, b) < L:
                    continue
                if is_leaf[a] and is_leaf[b]:
                    d2, i, j, n_eval = _leaf_pair_max(X, idx, r, r0, nstart, nend, centers,
                                                      radius, r0max, a, b, L, P, Q)
                    used += n_eval
                    if used > local_budget:
                        abort[0] = 1
                    if i >= 0 and (bi < 0 or _pair_better(d2, i, j, bd, bi, bj)):
                        bd = d2
                        bi = i
                        bj = j
                        dd = np.sqrt(d2)
                        if dd > shared_L[0]:
                            shared_L[0] = dd
                elif a == b:
                    c1 = 2 * a + 1
                    c2 = 2 * a + 2
                    sa[top] = c1
                    sb[top] = c1
                    sa[top + 1] = c2
                    sb[top + 1] = c2
                    sa[top + 2] = c1
                    sb[top + 2] = c2
                    top += 3
                else:
                    # Split the larger non-leaf node; push the more promising
                    # child pair last so it is explored first.
                    if is_leaf[a] or (not is_leaf[b] and radius[b] > radius[a]):
                        c1 = 2 * b + 1
                        c2 = 2 * b + 2
                        if _pair_ub(centers, radius, r0max, a, c1) > \
                                _pair_ub(centers, radius, r0max, a, c2):
                            c1, c2 = c2, c1
                        sa[top] = a
                        sb[top] = c1
                        sa[top + 1] = a
                        sb[top + 1] = c2
                    else:
                        c1 = 2 * a + 1
                        c2 = 2 * a + 2
                        if _pair_ub(centers, radius, r0max, c1, b) > \
                                _pair_ub(centers, radius, r0max, c2, b):
                            c1, c2 = c2, c1
                        sa[top] = c1
                        sb[top] = b
                        sa[top + 1] = c2
                        sb[top + 1] = b
                    top += 2
        w_d[w] = bd
        w_i[w] = bi
        w_j[w] = bj
    bd = -1.0
    bi = -1
    bj = -1
    for w in range(n_workers):
        if w_i[w] >= 0 and (bi < 0 or _pair_better(w_d[w], w_i[w], w_j[w], bd, bi, bj)):
            bd = w_d[w]
            bi = w_i[w]
            bj = w_j[w]
    return bd, bi, bj, abort[0] != 0


@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def tile_max(X, rows_i, rows_j, sq_i, sq_j, G, same, thr2, slack):
    """Exact most distant pair within a tile of the blocked brute-force search.

    ``G = Xi @ Xj.T`` (BLAS) on rows centred at a common point, ``sq_*`` their
    squared norms, so ``sq_i + sq_j - 2 G`` approximates squared distances.
    Every pair whose approximation reaches ``thr2 - slack`` is recomputed
    exactly from ``X``; ``slack`` covers the rounding of the approximation.
    Returns (squared distance, i, j) with ``i < j`` (``i = -1`` if none).
    """
    nI = rows_i.shape[0]
    nJ = rows_j.shape[0]
    row_d = np.full(nI, -1.0)
    row_i = np.full(nI, -1, dtype=np.int64)
    row_j = np.full(nI, -1, dtype=np.int64)
    for a in prange(nI):
        cut = thr2 - slack
        bd = -1.0
        bi = -1
        bj = -1
        b0 = a + 1 if same else 0
        sa = sq_i[a]
        for b in range(b0, nJ):
            if sa + sq_j[b] - 2.0 * G[a, b] >= cut:
                i = rows_i[a]
                j = rows_j[b]
                d2 = _sqdist_rows(X, i, j)
                lo = min(i, j)
                hi = max(i, j)
                if bi < 0 or _pair_better(d2, lo, hi, bd, bi, bj):
                    bd = d2
                    bi = lo
                    bj = hi
                    if d2 - slack > cut:
                        cut = d2 - slack
        row_d[a] = bd
        row_i[a] = bi
        row_j[a] = bj
    bd = -1.0
    bi = -1
    bj = -1
    for a in range(nI):
        if row_i[a] >= 0 and (bi < 0 or _pair_better(row_d[a], row_i[a], row_j[a], bd, bi, bj)):
            bd = row_d[a]
            bi = row_i[a]
            bj = row_j[a]
    return bd, bi, bj


# --------------------------------------------------------------------------
# Greedy max-min selection
# --------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _recompute_leaf(idx, m, nstart, nend, mu, amu, v):
    best_m = -1.0
    best_p = -1
    best_i = 0
    for p in range(nstart[v], nend[v]):
        mp = m[p]
        if mp < 0.0:
            continue
        i = idx[p]
        if best_p < 0 or _better(mp, i, best_m, best_i):
            best_m = mp
            best_p = p
            best_i = i
    mu[v] = best_m
    amu[v] = best_p


@njit(cache=True, inline="always")
def _recompute_inner(idx, mu, amu, v):
    c1 = 2 * v + 1
    c2 = 2 * v + 2
    p1 = amu[c1]
    p2 = amu[c2]
    if p1 < 0:
        mu[v] = mu[c2]
        amu[v] = p2
    elif p2 < 0 or _better(mu[c1], idx[p1], mu[c2], idx[p2]):
        mu[v] = mu[c1]
        amu[v] = p1
    else:
        mu[v] = mu[c2]
        amu[v] = p2


@njit(cache=True)
def _update_path(idx, m, nstart, nend, mu, amu, first_leaf, p):
    """Refresh the node maxima on the path from position ``p``'s leaf to the root."""
    v = 0
    while v < first_leaf:
        c = 2 * v + 1
        v = c if p < nend[c] else c + 1
    _recompute_leaf(idx, m, nstart, nend, mu, amu, v)
    while v > 0:
        v = (v - 1) // 2
        _recompute_inner(idx, mu, amu, v)


@njit(cache=True, fastmath=_FASTMATH, inline="always")
def _node_mask(S, nb, centers, radius, mu, amu, v, msk, dv):
    """Subset of buffered samples (bitmask) that may improve node ``v``;
    stores their distances to the node centre in ``dv``."""
    if amu[v] < 0:
        return 0
    rv = radius[v]
    muv = mu[v]
    c = centers[v]
    out = 0
    for j in range(nb):
        if (msk >> j) & 1:
            acc = 0.0
            for t in range(c.shape[0]):
                diff = S[j, t] - c[t]
                acc += diff * diff
            dj = np.sqrt(acc)
            dv[j] = dj
            if dj - rv - _EPS * (dj + rv) < muv:
                out |= 1 << j
    return out


@njit(cache=True, fastmath=_FASTMATH)
def _sync_subtree(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                  S, nb, v_sync, first_leaf, root, mask):
    """Apply the buffered samples ``S[:nb]`` (bitmask ``mask``) to the subtree
    under ``root``, then refresh the subtree's node maxima."""
    stack_v = np.empty(_STACK, dtype=np.int64)
    stack_m = np.empty(_STACK, dtype=np.int64)
    dv = np.empty(64, dtype=np.float64)
    stack_v[0] = root
    stack_m[0] = mask
    top = 1
    while top > 0:
        top -= 1
        v = stack_v[top]
        msk = _node_mask(S, nb, centers, radius, mu, amu, v, stack_m[top], dv)
        if msk == 0:
            continue
        if v >= first_leaf:
            for p in range(nstart[v], nend[v]):
                mp = m[p]
                if mp < 0.0:
                    continue
                j0 = ver[p] - v_sync
                if j0 < 0:
                    j0 = 0
                rp = r[p]
                i = idx[p]
                for j in range(j0, nb):
                    if (msk >> j) & 1:
                        dj = dv[j]
                        if dj - rp - _EPS * (dj + rp) < mp:
                            acc = 0.0
                            for t in range(S.shape[1]):
                                diff = S[j, t] - np.float64(X[i, t])
                                acc += diff * diff
                            if acc < mp * mp:
                                mp = np.sqrt(acc)
                m[p] = mp
            _recompute_leaf(idx, m, nstart, nend, mu, amu, v)
        else:
            stack_v[top] = 2 * v + 1
            stack_m[top] = msk
            stack_v[top + 1] = 2 * v + 2
            stack_m[top + 1] = msk
            top += 2
    # Bottom-up refresh of the subtree's inner nodes, deepest level first.
    lo = root
    width = 1
    los = np.empty(64, dtype=np.int64)
    widths = np.empty(64, dtype=np.int64)
    n_lv = 0
    while lo < first_leaf:
        los[n_lv] = lo
        widths[n_lv] = width
        n_lv += 1
        lo = 2 * lo + 1
        width *= 2
    for t in range(n_lv - 1, -1, -1):
        for v in range(los[t], los[t] + widths[t]):
            _recompute_inner(idx, mu, amu, v)


@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def _sync_parallel(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                   S, nb, v_sync, first_leaf, f_first, fmask, todo, n_todo):
    for t in prange(n_todo):
        f = todo[t]
        _sync_subtree(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                      S, nb, v_sync, first_leaf, f_first + f, fmask[f])


@njit(cache=True, fastmath=_FASTMATH)
def _sync(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
          S, nb, v_sync, first_leaf, split_level):
    """Apply the buffered selected samples ``S[:nb]`` to every sample."""
    if nb == 0:
        return
    full = (1 << nb) - 1
    dv = np.empty(64, dtype=np.float64)
    # Frontier: the nodes on level `split` (or the leaves, if shallower).
    split = 0
    while split < split_level and (1 << split) - 1 < first_leaf:
        split += 1
    f_first = (1 << split) - 1
    n_front = 1 << split
    fmask = np.zeros(n_front, dtype=np.int64)
    if split == 0:
        fmask[0] = full
    else:
        upper = np.zeros(f_first, dtype=np.int64)
        upper[0] = full
        for v in range(f_first):
            msk = _node_mask(S, nb, centers, radius, mu, amu, v, upper[v], dv)
            for c in (2 * v + 1, 2 * v + 2):
                if c < f_first:
                    upper[c] = msk
                else:
                    fmask[c - f_first] = msk
    todo = np.empty(n_front, dtype=np.int64)
    n_todo = 0
    for f in range(n_front):
        if fmask[f] != 0:
            todo[n_todo] = f
            n_todo += 1
    if n_todo > 2:
        _sync_parallel(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                       S, nb, v_sync, first_leaf, f_first, fmask, todo, n_todo)
    else:
        for t in range(n_todo):
            f = todo[t]
            _sync_subtree(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                          S, nb, v_sync, first_leaf, f_first + f, fmask[f])
    for v in range(f_first - 1, -1, -1):
        _recompute_inner(idx, mu, amu, v)


@njit(cache=True, fastmath=_FASTMATH)
def select_loop(X, idx, r, nstart, nend, centers, radius, first_leaf,
                first_pos, n_select, batch, refresh_budget, split_level, out_idx, out_dist,
                stats):
    """Greedy max-min (farthest-point) selection on the ball tree.

    ``first_pos`` holds tree positions of samples selected first, in order
    (the initial pair for classic Kennard-Stone). After them, each step selects
    the unselected sample farthest from the selected set (ties: lowest row).
    ``out_idx`` / ``out_dist`` receive the row of each selected sample and its
    distance to the samples selected before it (inf for the first).
    ``stats`` counts (syncs, candidate refreshes).
    """
    n = idx.shape[0]
    n_nodes = nstart.shape[0]
    d = X.shape[1]
    k0 = first_pos.shape[0]
    m = np.full(n, np.inf)
    ver = np.zeros(n, dtype=np.int64)
    mu = np.full(n_nodes, np.inf)
    amu = nstart.copy()
    S = np.empty((batch, d), dtype=np.float64)
    nb = 0
    v_sync = 0
    n_sel = 0
    n_refresh = 0
    for k in range(n_select):
        while True:
            p = first_pos[k] if k < k0 else amu[0]
            e = max(ver[p], v_sync)
            if e < n_sel and k >= k0 and n_refresh >= refresh_budget:
                # Many candidates are going stale: apply the batch now.
                _sync(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                      S, nb, v_sync, first_leaf, split_level)
                stats[0] += 1
                v_sync = n_sel
                nb = 0
                n_refresh = 0
                continue
            if e < n_sel:
                # Refresh the candidate against the buffered samples.
                mp = m[p]
                i = idx[p]
                for j in range(e - v_sync, nb):
                    d2 = _sqdist_vec(S[j], X, i)
                    if d2 < mp * mp:
                        mp = np.sqrt(d2)
                m[p] = mp
                ver[p] = n_sel
                stats[1] += 1
                n_refresh += 1
                if k >= k0:
                    _update_path(idx, m, nstart, nend, mu, amu, first_leaf, p)
                    continue
            break
        out_idx[k] = idx[p]
        out_dist[k] = m[p]
        if k == n_select - 1:
            break
        m[p] = -1.0
        _update_path(idx, m, nstart, nend, mu, amu, first_leaf, p)
        i = idx[p]
        for t in range(d):
            S[nb, t] = X[i, t]
        nb += 1
        n_sel += 1
        if nb == batch or k == k0 - 1:
            n_refresh = 0
            _sync(X, idx, r, nstart, nend, centers, radius, m, mu, amu, ver,
                  S, nb, v_sync, first_leaf, split_level)
            stats[0] += 1
            v_sync = n_sel
            nb = 0
    return out_idx, out_dist
