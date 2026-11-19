"""Shared colours and axis styling for the example figures."""
import matplotlib

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots (validated for all-pairs contrast); at most four are used.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#8f5bd6"]


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def style_scatter_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_color(GRID)
