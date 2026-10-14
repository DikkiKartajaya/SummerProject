"""Shared matplotlib/seaborn theme for every figure the project produces.

One minimalist look everywhere: off-white surface, hairline gridlines, no
chart-junk (no top/right spines, no legend boxes for single-series plots),
and a fixed-order categorical palette so a given label always gets the same
color across figures. Call `apply()` once per process before plotting.
"""

from __future__ import annotations

import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Fixed hue order - never cycled, so "positive" (say) keeps its color across
# every figure it appears in regardless of how many categories are present.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Single-hue blue ramp for magnitude (heatmaps, confusion matrices).
_SEQUENTIAL_STEPS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_STEPS)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def palette(n: int) -> list[str]:
    """First n colors of the fixed categorical order."""
    return (CATEGORICAL * (n // len(CATEGORICAL) + 1))[:n]


def label_palette(labels) -> dict[str, str]:
    """Stable label -> color mapping (alphabetical -> fixed hue order), so a
    given label (e.g. 'positive') keeps the same color in every figure that
    plots it, across scripts."""
    ordered = sorted(set(labels))
    return {lab: CATEGORICAL[i % len(CATEGORICAL)] for i, lab in enumerate(ordered)}


def apply() -> None:
    """Set the shared theme. Idempotent - safe to call from every module."""
    sns.set_theme(style="white")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "medium",
        "axes.titlesize": 12,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
