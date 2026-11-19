"""Coverage of Kennard-Stone subsets as the number of features grows.

Five anisotropic Gaussian clusters are generated in d = 2, 5, 10, 20 and 50
dimensions (20,000 samples each). Kennard-Stone (Euclidean and Mahalanobis) is
compared with uniform random subsets. Convex-hull volumes are impractical in
many dimensions, so the extent is measured along the principal axes instead.

Outputs, written to ``examples/figures/``:

* ``dimensions.png``: coverage against the kept fraction, one column per d.
* ``dimensions.csv``: the numbers behind it.

Run from the repository root::

    python examples/dimensions.py
"""
from pathlib import Path

import _style as S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

from efficient_kennard_stone import kennard_stone

OUT = Path(__file__).resolve().parent / "figures"
N = 20_000
DIMS = [2, 5, 10, 20, 50]
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 10

METHODS = {
    "Random": dict(color=S.MUTED, lw=2),
    "Kennard-Stone, Euclidean": dict(kw=dict(), color=S.CAT[0], lw=2.4),
    "Kennard-Stone, Mahalanobis": dict(kw=dict(metric="mahalanobis"), color=S.CAT[2], lw=2),
}


def make_data(d, seed=0):
    """Five Gaussian clusters, each stretched by its own random linear map."""
    rng = np.random.default_rng(seed)
    X, y = make_blobs(N, n_features=d, centers=5, center_box=(-12, 12), random_state=seed)
    for c in range(5):
        A = rng.normal(size=(d, d)) / np.sqrt(d) + 0.5 * np.eye(d)
        m = y == c
        X[m] = (X[m] - X[m].mean(0)) @ A + X[m].mean(0)
    return X


def coverage(X, idx, axes_full, ranges_full):
    """(extent %, mean gap, max gap) of the subset ``idx``.

    extent: range of the subset's projections onto the principal axes of the
    full data, as a % of the full range, averaged over axes. Gaps: distance
    from each sample to its nearest kept sample.
    """
    P = (X[idx] - X.mean(0)) @ axes_full
    extent = 100 * np.mean(np.ptp(P, axis=0) / ranges_full)
    _, gaps = pairwise_distances_argmin_min(X, X[idx])
    return extent, gaps.mean(), gaps.max()


def evaluate():
    rows = []
    for d in DIMS:
        X = make_data(d)
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        axes_full = Vt.T
        ranges_full = np.ptp(Xc @ axes_full, axis=0)
        n_max = int(round(max(FRACTIONS) * N))
        orders = {m: kennard_stone(X, n_max, **spec["kw"])
                  for m, spec in METHODS.items() if m != "Random"}
        for method in METHODS:
            for frac in FRACTIONS:
                k = int(round(frac * N))
                seeds = range(N_SEEDS) if method == "Random" else [0]
                for seed in seeds:
                    idx = (np.random.default_rng(seed).choice(N, k, replace=False)
                           if method == "Random" else orders[method][:k])
                    ext, mg, xg = coverage(X, idx, axes_full, ranges_full)
                    rows.append(dict(d=d, method=method, fraction=frac, seed=seed,
                                     extent=ext, mean_gap=mg, max_gap=xg))
        print(f"d={d} done", flush=True)
    return pd.DataFrame(rows)


def plot(df):
    agg = df.groupby(["d", "method", "fraction"], sort=False)[
        ["extent", "mean_gap", "max_gap"]].mean()
    metrics = [
        ("extent", "Extent kept, % of range\nalong principal axes (higher is better)"),
        ("max_gap", "Max gap to nearest kept point\n(edge coverage, lower is better)"),
        ("mean_gap", "Mean gap to nearest kept point\n(core coverage, lower is better)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(DIMS), figsize=(3.2 * len(DIMS), 3.0 * len(metrics)),
                             facecolor=S.SURFACE, sharex=True)
    for c, d in enumerate(DIMS):
        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            S.style_axes(ax)
            for method, spec in METHODS.items():
                series = agg.loc[(d, method)][key]
                ax.plot(series.index, series.values, color=spec["color"], lw=spec["lw"],
                        marker="o", ms=4, mec=S.SURFACE, mew=1,
                        zorder=2 if method == "Random" else 3)
            ax.set_xscale("log")
            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(f"d = {d}", fontsize=10, color=S.INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=S.INK_2)
            if r == len(metrics) - 1:
                ax.set_xlabel("Fraction of data kept", fontsize=8.5, color=S.INK_2)
    handles = [plt.Line2D([], [], color=s["color"], lw=s["lw"], marker="o", ms=4,
                          mec=S.SURFACE) for s in METHODS.values()]
    fig.legend(handles, list(METHODS), loc="upper center", ncol=len(METHODS),
               frameon=False, fontsize=9, labelcolor=S.INK, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.955, f"{N:,} samples from five anisotropic Gaussian clusters in d "
             f"dimensions. Random: mean of {N_SEEDS} seeds.",
             ha="center", fontsize=9, color=S.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "dimensions.png", dpi=110, facecolor=S.SURFACE)
    plt.close(fig)
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = evaluate()
    agg = plot(df)
    agg.round(4).to_csv(OUT / "dimensions.csv")
    at = df[df.fraction == 0.05].groupby(["d", "method"])[["extent", "mean_gap", "max_gap"]].mean()
    print("Coverage at 5% kept:")
    print(at.round(3).to_string())


if __name__ == "__main__":
    main()
