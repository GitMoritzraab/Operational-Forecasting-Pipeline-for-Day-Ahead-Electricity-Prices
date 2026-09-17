"""
plot_anc_bar.py
===============

Plots Average Normalised Coefficients (ANC) as horizontal bar charts for
two electricity-price forecasting models:

* **Fundamental Model** -- features derived from fundamental market drivers
  (wind, load, solar, lagged prices, calendar variables).
* **EXAA-Enriched Model** -- same feature set augmented by the day-ahead
  EXAA auction price.

Each model produces one PDF figure saved to the ``output/`` directory
in the repository root.

Expected repository layout
--------------------------
repo/
├── output/              (created automatically if missing)
└── visualization/
    └── plot_anc_bar.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from experiment_config import load_experiment_config

# ============================================================
#  ANC VALUES
# ============================================================

features_fund = [
    "Wind $d$",
    "Load $d$",
    "Price $d-1$",
    "Solar $d$",
    "Price $d-7$",
    "Market dummy",
    "Price $d-2$",
    "Weekday",
    "Holiday",
]
anc_fundamental = np.array([43.4, 30.9, 28.5, 7.5, 3.2, 3.2, 2.0, 1.5, 0.0])

features_exaa = [
    "EXAA $d$",
    "Price $d-1$",
    "Wind $d$",
    "Load $d$",
    "Price $d-7$",
    "Market dummy",
    "Solar $d$",
    "Price $d-2$",
    "Weekday",
    "Holiday",
]
anc_exaa = np.array([90.3, 8.0, 6.9, 4.6, 3.1, 3.0, 2.3, 1.9, 0.9, 0.0])

# ============================================================
#  STYLE
# ============================================================

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
})

COLOR = "#1a5c8a"
BAR_H = 0.55

FEATURE_LABELS = {
    "Wind d": "Wind $d$",
    "Load d": "Load $d$",
    "Price d-1": "Price $d-1$",
    "Solar d": "Solar $d$",
    "Price d-7": "Price $d-7$",
    "15-min market dummy": "Market dummy",
    "Price d-2": "Price $d-2$",
    "Weekday": "Weekday",
    "Holiday": "Holiday",
    "EXAA d": "EXAA $d$",
}


def load_anc(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path).sort_values("ANC", ascending=False)
    return (
        [FEATURE_LABELS.get(value, value) for value in df["feature_group"]],
        df["ANC"].to_numpy(dtype=float),
    )


def draw_anc_bars(ax, features, anc_values, xmax):
    """
    Draw a ranked horizontal bar chart of ANC values onto an existing Axes.

    Each bar is annotated with its numeric value to the right of the bar.
    The y-axis is inverted so that the most important feature (highest ANC)
    appears at the top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which the chart is drawn.
    features : list of str
        Feature labels in rank order (index 0 = most important).
        Supports LaTeX math notation (e.g. ``"Wind $d$"``).
    anc_values : numpy.ndarray
        ANC scores corresponding to each entry in *features*.  Values
        should be non-negative and listed in the same order as *features*.
    xmax : float
        Upper limit of the x-axis.  Typically set to
        ``anc_values.max() * 1.18`` to leave room for the value labels.

    Returns
    -------
    None
    """
    n = len(features)
    y = np.arange(n)

    bars = ax.barh(y, anc_values, height=BAR_H, color=COLOR, zorder=3)

    for bar, val in zip(bars, anc_values):
        ax.text(bar.get_width() + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left",
                fontsize=10.0, color=COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=12)
    ax.set_xlabel("ANC", fontsize=13, labelpad=6)
    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="x", which="major", labelsize=12)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", axis="x", zorder=0)
    ax.invert_yaxis()


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    config = load_experiment_config(REPO_ROOT)
    anc_base = config.results_root / "lear_anc_results" / "era5" / "c5" / "d112"
    features_fund, anc_fundamental = load_anc(
        anc_base / "fundamental" / "anc_all_feature_results.csv"
    )
    features_exaa, anc_exaa = load_anc(
        anc_base / "exaa" / "anc_all_feature_results.csv"
    )
    _OUT  = config.output_root
    _OUT.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Fundamental Model ──────────────────────────────
    xmax_fund = anc_fundamental.max() * 1.18
    fig1, ax1 = plt.subplots(figsize=(5.0, 5.0))
    draw_anc_bars(ax1, features_fund, anc_fundamental, xmax_fund)
    fig1.tight_layout()
    fig1.savefig(_OUT / "anc_fundamental.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("Saved: anc_fundamental.pdf")

    # ── Figure 2: EXAA-Enriched Model ────────────────────────────
    xmax_exaa = anc_exaa.max() * 1.18
    fig2, ax2 = plt.subplots(figsize=(5.0, 5.0))
    draw_anc_bars(ax2, features_exaa, anc_exaa, xmax_exaa)
    fig2.tight_layout()
    fig2.savefig(_OUT / "anc_exaa.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: anc_exaa.pdf")
