"""Benchmark Kennard-Stone selection by stage, with peak memory.

Usage: python benchmarks/bench.py N D N_SELECT [--data gauss|uniform|blobs]
       [--leaf INT] [--dtype float64|float32] [--naive]

Each run should be its own process so peak RSS is meaningful.
"""

import argparse
import resource
import time

import numpy as np

from efficient_kennard_stone import _kennard_stone as ks
from efficient_kennard_stone import _kernels as K


def make_data(kind, n, d, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        X = rng.standard_normal((n, d), dtype=np.float64)
    elif kind == "uniform":
        X = rng.random((n, d), dtype=np.float64)
    elif kind == "blobs":
        centers = rng.normal(0, 10, size=(20, d))
        X = centers[rng.integers(0, 20, n)] + rng.standard_normal((n, d))
    else:
        raise ValueError(kind)
    return X.astype(dtype, copy=False)


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("d", type=int)
    p.add_argument("n_select", type=float, help="count, or fraction if < 1")
    p.add_argument("--data", default="gauss")
    p.add_argument("--leaf", type=int, default=32)
    p.add_argument("--dtype", default="float64")
    p.add_argument("--init", default="diameter")
    args = p.parse_args()

    k = int(args.n_select * args.n) if args.n_select < 1 else int(args.n_select)

    # Warm up / load JIT cache on tiny data so compile time is not measured.
    ks.kennard_stone(np.random.default_rng(1).random((3000, args.d)).astype(args.dtype), 50)

    X = make_data(args.data, args.n, args.d, np.dtype(args.dtype))
    base = peak_mb()

    t0 = time.perf_counter()
    tree = ks._Tree(X, args.leaf)
    t1 = time.perf_counter()
    if args.init == "diameter":
        first = list(ks._diameter(X, tree)[:2])
    else:
        first = [int(K.argmax_first(K.dist_to_point(X, X.mean(0, dtype=np.float64))))]
    t2 = time.perf_counter()
    stats = np.zeros(2, np.int64)
    ks._select(X, tree, first, k, stats)
    t3 = time.perf_counter()

    print(
        f"n={args.n:>8} d={args.d:>3} k={k:>8} data={args.data:<7} leaf={args.leaf:>3} "
        f"tree={t1 - t0:6.2f}s diameter={t2 - t1:7.2f}s select={t3 - t2:8.2f}s "
        f"total={t3 - t0:8.2f}s  X={X.nbytes / 2**20:6.0f}MB "
        f"peak={peak_mb():6.0f}MB (after data {base:6.0f}MB)",
        flush=True,
    )
    print(f"   syncs={stats[0]} refreshes/step={stats[1] / k:.2f} n_nodes={tree.n_nodes}")


if __name__ == "__main__":
    main()
