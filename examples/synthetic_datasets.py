"""Kennard-Stone subsets of scikit-learn's 2-D synthetic clustering datasets.

Compares a uniform random subset with Kennard-Stone selection (Euclidean,
Mahalanobis, and the cheaper "farthest from the mean" start) on the five 2-D
datasets of scikit-learn's clustering comparison. Kennard-Stone is
unsupervised; the generating labels are only used to colour the points and to
measure coverage per cluster.

A Kennard-Stone selection of k samples is the first k samples of the full
Kennard-Stone ordering, so each dataset needs only one run per method.

Outputs, written to ``examples/figures/``:

* ``subsets.png``: one 5% subset per dataset and method, drawn over the full data.
* ``metrics.png``: coverage of each cluster as the kept fraction varies.
* ``metrics.csv``: the numbers behind ``metrics.png``.

Run from the repository root::

    python examples/synthetic_datasets.py

Requires matplotlib and pandas in addition to the package dependencies.
"""
from pathlib import Path

import _style as S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from sklearn.datasets import make_blobs, make_circles, make_moons

from efficient_kennard_stone import kennard_stone

OUT = Path(__file__).resolve().parent / "figures"
N = 3000
SHOW_FRACTION = 0.05
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 30

# Random is the neutral reference line; Kennard-Stone variants take the
# categorical slots.
METHODS = {
    "Random": dict(color=S.MUTED, ls="-", lw=2),
    "Kennard-Stone, Euclidean": dict(kw=dict(), color=S.CAT[0], ls="-", lw=2.6),
    "Kennard-Stone, Mahalanobis": dict(kw=dict(metric="mahalanobis"), color=S.CAT[2],
                                       ls="-", lw=2),
    # Usually starts at a diameter endpoint, so it often coincides with the
    # classic start: drawn dashed on top so both stay visible.
    "Kennard-Stone, mean start": dict(kw=dict(init="mean"), color=S.CAT[1], ls="--",
                                      lw=1.6),
}
SHOWN_METHODS = ["Random", "Kennard-Stone, Euclidean", "Kennard-Stone, Mahalanobis"]


def make_datasets():
    """Return ``{name: (X, y)}``, the same family as scikit-learn's comparison."""
    out = {}
    X, y = make_blobs(N, centers=3, random_state=170)
    out["Isotropic blobs"] = (X, y)
    out["Anisotropic blobs"] = (X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y)
    out["Unequal variance blobs"] = make_blobs(N, centers=3, cluster_std=[1.0, 2.5, 0.5],
                                               random_state=170)
    out["Moons"] = make_moons(N, noise=0.05, random_state=0)
    out["Concentric circles"] = make_circles(N, factor=0.5, noise=0.04, random_state=0)
    return out


def subset_metrics(X, labels, idx):
    """Per-cluster coverage of the full data by the kept subset, averaged.

    * extent: convex-hull area of kept points / hull area of all points (%).
    * mean_gap, max_gap: distance from each original point to the nearest kept
      point of the same cluster (lower is better). ``max_gap`` measures how
      well the outer edge is covered, ``mean_gap`` how well the core is.
    """
    keep = np.zeros(len(X), bool)
    keep[idx] = True
    extent, mean_gap, max_gap = [], [], []
    for c in np.unique(labels):
        m = labels == c
        full, kept = X[m], X[m & keep]
        extent.append(100 * ConvexHull(kept).volume / ConvexHull(full).volume
                      if len(kept) >= 3 else 0.0)
        if len(kept) == 0:
            continue
        gaps, _ = cKDTree(kept).query(full)
        mean_gap.append(gaps.mean())
        max_gap.append(gaps.max())
    return np.mean(extent), np.mean(mean_gap), np.mean(max_gap)


def selections(X):
    """Kennard-Stone orderings (one per variant) and a function giving subsets."""
    n_max = int(round(max(FRACTIONS) * len(X)))
    orders = {m: kennard_stone(X, n_max, **spec["kw"])
              for m, spec in METHODS.items() if m != "Random"}

    def subset(method, fraction, seed):
        k = int(round(fraction * len(X)))
        if method == "Random":
            return np.random.default_rng(seed).choice(len(X), k, replace=False)
        return orders[method][:k]

    return subset


def evaluate(datasets):
    rows = []
    for name, (X, y) in datasets.items():
        subset = selections(X)
        for method in METHODS:
            seeds = range(N_SEEDS) if method == "Random" else [0]
            for frac in FRACTIONS:
                for seed in seeds:
                    ext, mg, xg = subset_metrics(X, y, subset(method, frac, seed))
                    rows.append(dict(dataset=name, method=method, fraction=frac,
                                     seed=seed, extent=ext, mean_gap=mg, max_gap=xg))
    return pd.DataFrame(rows)


def plot_subsets(datasets, summary):
    names = list(datasets)
    ncols = 1 + len(SHOWN_METHODS)
    fig, axes = plt.subplots(len(names), ncols, figsize=(3.1 * ncols, 3.0 * len(names)),
                             facecolor=S.SURFACE)
    pct = f"{SHOW_FRACTION:.0%}"
    col_titles = ["All data, coloured by cluster label"] + [
        f"{m} ({pct} kept)" for m in SHOWN_METHODS]
    for r, name in enumerate(names):
        X, y = datasets[name]
        subset = selections(X)
        colors = np.array(S.CAT[:3])[y]
        for c in range(ncols):
            ax = axes[r, c]
            S.style_scatter_axes(ax)
            if c == 0:
                ax.scatter(X[:, 0], X[:, 1], s=3, c=colors, linewidths=0, alpha=0.6)
                ax.set_ylabel(name, color=S.INK, fontsize=9)
            else:
                method = SHOWN_METHODS[c - 1]
                idx = subset(method, SHOW_FRACTION, seed=0)
                ax.scatter(X[:, 0], X[:, 1], s=3, c=S.GRID, linewidths=0)
                ax.scatter(X[idx, 0], X[idx, 1], s=12, c=colors[idx],
                           edgecolors=S.SURFACE, linewidths=0.4)
                if method != "Random":
                    # The two most distant samples start the selection.
                    ax.scatter(X[idx[:2], 0], X[idx[:2], 1], s=60, facecolors="none",
                               edgecolors=S.INK, linewidths=0.9)
                s = summary.loc[(name, method)]
                ax.set_xlabel(f"extent kept {s.extent:.0f}%   max gap {s.max_gap:.2f}   "
                              f"mean gap {s.mean_gap:.2f}", fontsize=7.5, color=S.INK_2)
            ax.margins(0.07)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9, color=S.INK)
    fig.suptitle(
        f"{pct} subsets of {N:,} points. Grey points are discarded; circled points are "
        f"the first two Kennard-Stone picks.\nRandom-panel metrics are averages over "
        f"{N_SEEDS} seeds; Kennard-Stone is deterministic.",
        fontsize=10, color=S.INK_2, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "subsets.png", dpi=110, facecolor=S.SURFACE)
    plt.close(fig)


def plot_metrics(df):
    agg = df.groupby(["dataset", "method", "fraction"], sort=False)[
        ["extent", "mean_gap", "max_gap"]].mean()
    datasets = list(dict.fromkeys(df.dataset))
    metrics = [
        ("extent", "Extent kept, % of hull area\n(higher is better)"),
        ("max_gap", "Max gap to nearest kept point\n(edge coverage, lower is better)"),
        ("mean_gap", "Mean gap to nearest kept point\n(core coverage, lower is better)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(datasets),
                             figsize=(3.2 * len(datasets), 3.0 * len(metrics)),
                             facecolor=S.SURFACE, sharex=True)
    for c, name in enumerate(datasets):
        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            S.style_axes(ax)
            for method, spec in METHODS.items():
                series = agg.loc[(name, method)][key]
                ax.plot(series.index, series.values, color=spec["color"], lw=spec["lw"],
                        ls=spec["ls"], marker="o", ms=4, mec=S.SURFACE, mew=1,
                        zorder=2 if method == "Random" else 3)
            ax.set_xscale("log")
            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(name, fontsize=10, color=S.INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=S.INK_2)
            if r == len(metrics) - 1:
                ax.set_xlabel("Fraction of data kept", fontsize=8.5, color=S.INK_2)
    handles = [plt.Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"], marker="o",
                          ms=4, mec=S.SURFACE) for s in METHODS.values()]
    fig.legend(handles, list(METHODS), loc="upper center", ncol=len(METHODS),
               frameon=False, fontsize=9, labelcolor=S.INK, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "metrics.png", dpi=110, facecolor=S.SURFACE)
    plt.close(fig)
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets()
    df = evaluate(datasets)
    agg = plot_metrics(df)
    agg.round(4).to_csv(OUT / "metrics.csv")
    summary = df[df.fraction == SHOW_FRACTION].groupby(["dataset", "method"])[
        ["extent", "mean_gap", "max_gap"]].mean()
    plot_subsets(datasets, summary)
    pd.set_option("display.width", 120)
    print(f"Metrics at {SHOW_FRACTION:.0%} kept (Random: mean of {N_SEEDS} seeds):")
    print(summary.round(3).to_string())
    print(f"\nWrote figures to {OUT}")


if __name__ == "__main__":
    main()
