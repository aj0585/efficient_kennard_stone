"""Run time and memory against sample count, dimension and worst-case data.

Selects 1% of n samples (at least 10) with Kennard-Stone for n = 1,000 to
1,000,000 and d = 2, 10, 50, on four synthetic datasets:

* **Gaussian blobs**: 20 well separated isotropic clusters. Each blob fills
  all d dimensions, so at d = 50 there is little to prune inside a blob.
* **Swiss roll**: a 2-D sheet rolled up in 3-D, rotated into d dimensions
  (high ambient, low intrinsic dimension; in d = 2 a spiral).
* **Uniform hypercube**: fills all d dimensions (high intrinsic dimension:
  little for the triangle-inequality bounds to prune).
* **Hypersphere surface**: every sample is equally far from the mean, and
  almost every sample has a nearly diametral partner. This is the worst case
  for the exact diameter search, and as hard as the hypercube for selection.

The package is compared with two brute-force implementations that give the
same selection:

* **Textbook**: full n x n distance matrix (n <= 10,000).
* **Brute-force loop**: O(n) memory, one pass over the data per selected
  sample, diameter by chunked BLAS (n <= 100,000).

Their cost does not depend on the data, so they run on the Gaussian blobs only.

Every run happens in its own process, so its peak resident memory can be
measured (Linux/macOS). Data sets are generated once and cached as ``.npy``
files, so a run's memory baseline is exactly the loaded data. Results are
appended to ``examples/figures/scaling.csv``; rerunning skips finished runs
(delete the CSV to start over).

Outputs, written to ``examples/figures/``:

* ``scaling.png``: time and extra memory against n, one column per d.
* ``worst_case.png``: where the time goes at n = 1,000,000, per dataset and d.
* ``scaling.csv``: one row per run.

Run from the repository root (the full grid takes about an hour, most of it
the three 1,000,000-sample cases where the exact diameter cannot prune)::

    python examples/scaling.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
CSV = OUT / "scaling.csv"
CACHE = Path(os.environ.get("EKS_EXAMPLE_CACHE", Path(tempfile.gettempdir()) / "eks_example_data"))
SIZES = [1_000, 10_000, 100_000, 1_000_000]
DIMS = [2, 10, 50]
DATASETS = ["Gaussian blobs", "Swiss roll", "Uniform hypercube", "Hypersphere surface"]
METHODS = ["efficient-kennard-stone", "Brute-force loop", "Textbook"]
MAX_N = {"Brute-force loop": 100_000, "Textbook": 10_000}
TIMEOUT = 3600


def n_select(n):
    return max(10, n // 100)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def generate(name, n, d, seed=0):
    rng = np.random.default_rng(seed)
    if name == "Gaussian blobs":
        centers = rng.uniform(-30, 30, size=(20, d))
        X = centers[rng.integers(0, 20, n)] + rng.standard_normal((n, d))
    elif name == "Swiss roll":
        t = 1.5 * np.pi * (1 + 2 * rng.random(n))
        h = 21 * rng.random(n)
        if d == 2:
            X = np.column_stack([t * np.cos(t), t * np.sin(t)])
        else:
            R = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
            Q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
            X = R @ Q.T
        X += 0.05 * rng.standard_normal(X.shape)
    elif name == "Uniform hypercube":
        X = rng.random((n, d))
    elif name == "Hypersphere surface":
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    else:
        raise ValueError(name)
    return np.ascontiguousarray(X, dtype=np.float64)


def data_path(name, n, d):
    return CACHE / f"{name.replace(' ', '_').lower()}_{n}_{d}.npy"


def ensure_data(name, n, d):
    path = data_path(name, n, d)
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, generate(name, n, d))
        tmp.replace(path)
    return path


# --------------------------------------------------------------------------
# Baselines (same selection rule and tie-breaking as the package)
# --------------------------------------------------------------------------

def textbook(X, k):
    from scipy.spatial.distance import cdist

    D = cdist(X, X)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    sel = [min(i, j), max(i, j)][:k]
    m = D[sel].min(axis=0)
    m[sel] = -1
    while len(sel) < k:
        s = int(np.argmax(m))
        sel.append(s)
        np.minimum(m, D[s], out=m)
        m[sel] = -1
    return np.array(sel)


def brute_force_loop(X, k):
    n = X.shape[0]
    block = max(1, 2**24 // n)  # 16M distances (128 MiB) per chunk
    sq = np.einsum("ij,ij->i", X, X)
    best = (-1.0, 0, 0)
    for i0 in range(0, n, block):  # diameter: chunked GEMM, exact recheck of the max
        D2 = sq[i0:i0 + block, None] + sq[None, :] - 2.0 * X[i0:i0 + block] @ X.T
        r, c = np.unravel_index(np.argmax(D2), D2.shape)
        a, b = sorted((i0 + int(r), int(c)))
        d2 = float(((X[a] - X[b]) ** 2).sum())
        if d2 > best[0] or (d2 == best[0] and (a, b) < best[1:]):
            best = (d2, a, b)
    sel = [best[1], best[2]][:k]
    m = np.full(n, np.inf)
    for s in sel:
        np.minimum(m, np.sqrt(((X - X[s]) ** 2).sum(1)), out=m)
    m[sel] = -1
    while len(sel) < k:
        s = int(np.argmax(m))
        sel.append(s)
        np.minimum(m, np.sqrt(((X - X[s]) ** 2).sum(1)), out=m)
        m[sel] = -1
    return np.array(sel)


# --------------------------------------------------------------------------
# One run (in a fresh process)
# --------------------------------------------------------------------------

def peak_mb():
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def worker(method, name, n, d):
    from efficient_kennard_stone import _kennard_stone as ks, kennard_stone

    k = n_select(n)
    kennard_stone(np.random.default_rng(1).random((3000, d)), 50)  # load JIT cache
    X = np.load(data_path(name, n, d))
    base = peak_mb()
    stages = {}
    t0 = time.perf_counter()
    if method == "efficient-kennard-stone":
        # The steps of kennard_stone(X, k), timed separately.
        tree = ks._Tree(X, 32)
        t1 = time.perf_counter()
        first = list(ks._diameter(X, tree)[:2])
        t2 = time.perf_counter()
        idx, _ = ks._select(X, tree, first, k)
        t3 = time.perf_counter()
        stages = dict(t_tree=t1 - t0, t_diameter=t2 - t1, t_select=t3 - t2)
    elif method == "Brute-force loop":
        idx = brute_force_loop(X, k)
    else:
        idx = textbook(X, k)
    elapsed = time.perf_counter() - t0
    digest = hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]
    print(json.dumps(dict(method=method, dataset=name, n=n, d=d, k=k, seconds=elapsed,
                          extra_mb=peak_mb() - base, data_mb=X.nbytes / 2**20,
                          selection_sha1=digest, **stages)))


def run_grid():
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(
        columns=["method", "dataset", "n", "d"])
    finished = set(zip(done.method, done.dataset, done.n, done.d))
    # Baselines run on the Gaussian blobs only (their cost does not depend on the data).
    jobs = [(m, ds, n, d) for d in DIMS for n in SIZES for ds in DATASETS for m in METHODS
            if (m == METHODS[0] or (ds == DATASETS[0] and n <= MAX_N[m]))
            and (m, ds, n, d) not in finished]
    for method, name, n, d in jobs:
        ensure_data(name, n, d)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", method, name,
               str(n), str(d)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
            row = json.loads(res.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            row = dict(method=method, dataset=name, n=n, d=d, k=n_select(n),
                       seconds=float("inf"))
        except (IndexError, json.JSONDecodeError):
            print(res.stderr, file=sys.stderr)
            raise
        print(f"{method:<24} {name:<20} n={n:>9,} d={d:>2}  {row['seconds']:9.2f} s  "
              f"+{row.get('extra_mb', float('nan')):8.1f} MB", flush=True)
        pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot():
    import _style as S
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(CSV)
    # All implementations must agree wherever several were run.
    agree = df.groupby(["dataset", "n", "d"]).selection_sha1.nunique()
    mismatches = agree[agree > 1]
    if len(mismatches):
        print("WARNING: selections differ for\n", mismatches)
    else:
        print(f"All implementations made identical selections "
              f"({int((df.groupby(['dataset', 'n', 'd']).size() > 1).sum())} shared configs).")

    ds_color = dict(zip(DATASETS, S.CAT))
    base_style = {"Brute-force loop": dict(ls="-", marker="s"),
                  "Textbook": dict(ls="--", marker="^")}
    fig, axes = plt.subplots(2, len(DIMS), figsize=(4.4 * len(DIMS), 7.4), facecolor=S.SURFACE,
                             sharex=True)
    for c, d in enumerate(DIMS):
        for r, key in enumerate(["seconds", "extra_mb"]):
            ax = axes[r, c]
            S.style_axes(ax, grid_axis="both")
            for name in DATASETS:
                sub = df[(df.method == METHODS[0]) & (df.dataset == name) & (df.d == d)]
                sub = sub.sort_values("n")
                ax.plot(sub.n, sub[key].clip(lower=1e-3 if key == "seconds" else 0.1),
                        color=ds_color[name], lw=2.2, marker="o", ms=4.5, mec=S.SURFACE,
                        zorder=3)
            for method, st in base_style.items():
                sub = df[(df.method == method) & (df.dataset == "Gaussian blobs") & (df.d == d)]
                sub = sub.sort_values("n")
                ax.plot(sub.n, sub[key].clip(lower=1e-3 if key == "seconds" else 0.1),
                        color=S.MUTED, lw=1.6, ms=4.5, mec=S.SURFACE, zorder=2, **st)
            if key == "extra_mb":
                # The textbook matrix alone needs 8 n^2 bytes.
                ns = np.array(SIZES, dtype=float)
                ax.plot(ns, 8 * ns**2 / 2**20, color=S.MUTED, lw=1, ls=":", zorder=1)
                ax.annotate("n × n matrix: 7.3 TiB at 1M", xy=(ns[-1], 8 * ns[-1]**2 / 2**20),
                            xytext=(-4, -2), textcoords="offset points", ha="right", va="top",
                            fontsize=7.5, color=S.MUTED)
                ax.plot(ns, ns * d * 8 / 2**20, color=S.AXIS, lw=1, ls="-.", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(SIZES)
            ax.set_xticklabels(["1k", "10k", "100k", "1M"])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(f"d = {d}", fontsize=10, color=S.INK)
            if c == 0:
                ax.set_ylabel("Wall-clock time, s" if key == "seconds"
                              else "Peak memory above the data, MiB",
                              fontsize=9, color=S.INK_2)
            if r == 1:
                ax.set_xlabel("Samples n (Kennard-Stone selects 1%)", fontsize=8.5,
                              color=S.INK_2)
    handles = [plt.Line2D([], [], color=ds_color[n], lw=2.2, marker="o", ms=4.5, mec=S.SURFACE)
               for n in DATASETS]
    labels = [f"efficient-kennard-stone: {n}" for n in DATASETS]
    for method, st in base_style.items():
        handles.append(plt.Line2D([], [], color=S.MUTED, lw=1.6, ms=4.5, **st))
        labels.append(f"{method} (Gaussian blobs)")
    handles.append(plt.Line2D([], [], color=S.AXIS, lw=1, ls="-.") )
    labels.append("size of the data itself")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=S.INK, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "scaling.png", dpi=110, facecolor=S.SURFACE)
    plt.close(fig)

    # Where the time goes at the largest n.
    big = df[(df.method == METHODS[0]) & (df.n == max(SIZES))]
    stages = [("t_tree", "Build ball tree"), ("t_diameter", "Exact diameter"),
              ("t_select", "Select samples")]
    stage_color = [S.CAT[0], S.CAT[1], S.CAT[2]]
    fig, axes = plt.subplots(1, len(DIMS), figsize=(4.6 * len(DIMS), 3.2), facecolor=S.SURFACE,
                             sharey=True)
    for c, d in enumerate(DIMS):
        ax = axes[c]
        S.style_axes(ax, grid_axis="x")
        sub = big[big.d == d].set_index("dataset").reindex(DATASETS)
        total = sub[[col for col, _ in stages]].fillna(0).sum(axis=1).to_numpy()
        left = np.zeros(len(DATASETS))
        for (col, label), color in zip(stages, stage_color):
            vals = 100 * sub[col].fillna(0).to_numpy() / np.maximum(total, 1e-12)
            ax.barh(range(len(DATASETS)), vals, left=left, color=color, height=0.62,
                    label=label, edgecolor=S.SURFACE, linewidth=0.8)
            left += vals
        for y, tot in enumerate(sub.seconds):
            ax.text(101, y, f"{tot:,.1f} s", va="center", fontsize=8.5, color=S.INK)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS, fontsize=8.5, color=S.INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 122)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"d = {d}", fontsize=10, color=S.INK)
        ax.set_xlabel("Share of wall-clock time, %", fontsize=8.5, color=S.INK_2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=S.INK, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.875, f"n = {max(SIZES):,} samples, selecting {n_select(max(SIZES)):,}. "
             "Bars: share of time per stage; labels: total time. When nothing prunes, the "
             "exact diameter (O(n²) pairs) dominates.",
             ha="center", fontsize=9, color=S.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(OUT / "worst_case.png", dpi=110, facecolor=S.SURFACE)
    plt.close(fig)
    print(f"Wrote figures to {OUT}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worker", nargs=4, metavar=("METHOD", "DATASET", "N", "D"))
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.worker:
        method, name, n, d = a.worker
        worker(method, name, int(n), int(d))
        return
    if not a.plot_only:
        run_grid()
    plot()


if __name__ == "__main__":
    main()
