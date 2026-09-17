"""
generate_comptime_tables.py
====================================
Generates colour-coded computation time heatmap tables for the LEAR and
SQRA models.

Each cell shows the mean computation time per forecasting day in bold
(in minutes) and the observed [min–max] range in smaller text beneath it.
Cells are shaded using a truncated RdYlGn_r colormap so that shorter
runtimes appear green and longer runtimes appear red.

Four tables are produced:
- LEAR Fundamental    (3 rows x 4 columns)
- LEAR EXAA Enriched  (3 rows x 4 columns)
- LEAR EXAA Only      (1 row  x 3 columns)
- SQRA                (6 rows x 1 column)

Expected repository layout
--------------------------
repo/
├── output/              (created automatically if missing)
└── visualization/
    └── generate_comptime_tables.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from experiment_config import load_experiment_config

# ============================================================
#  ENTER MEASUREMENTS HERE
#  Format per cell: (mean, min, max)  — values in minutes or seconds
#  Unknown values: (np.nan, np.nan, np.nan)
#
#  Rows:    C=1, C=5, C=25
#  Columns: ICON-D2 D=56, ERA5 D=56, ERA5 D=112, ERA5 D=364
# ============================================================

lear_fundamental = np.array([
    # ICON-D2 D=56             ERA5 D=56          ERA5 D=112      ERA5 D=364
    [(0.41, 0.37, 0.73), (0.55, 0.36, 2.93), (0.52, 0.45, 0.66), (8.36, 7.61, 9.15)],  # C=1
    [(1.07, 0.86, 3.67), (0.43, 0.40, 0.62), (0.70, 0.64, 1.46), (8.23, 7.44, 8.87)],  # C=5
    [(3.16, 3.10, 3.26), (0.96, 0.81, 2.21), (1.30, 1.23, 1.88), (18.27, 16.61, 19.89)], # C=25
], dtype=object)

lear_exaa_enriched= np.array([
    # ICON-D2 D=56             ERA5 D=56          ERA5 D=112      ERA5 D=364
    [(0.51, 0.43, 2.83), (0.59, 0.42, 1.97), (0.57, 0.52, 1.24), (4.27, 3.67, 6.89)],  # C=1
    [(1.24, 0.95, 3.13), (0.61, 0.45, 1.00), (0.64, 0.60, 1.01), (2.99, 2.51, 4.09)],  # C=5
    [(3.66, 3.11, 4.87), (0.86, 0.82, 1.07), (1.55, 1.24, 3.48), (12.00, 10.93, 14.06)],  # C=25
], dtype=object)

# EXAA-Scaled LEAR Model
# One row (no C, no ICON-D2), three columns: D=56, D=112, D=364
# p = 96 (only EXAA prices as features)
lear_exaa_only = np.array([
    # D=56                       D=112                     D=364
    [(0.0660, 0.0566, 0.0894), (0.2441, 0.2160, 0.2802), (0.1415, 0.1289, 0.1892)],
], dtype=object)

# Rows: 5 SQRA configurations  |  Column: D_SQRA=60
# Measured values from prob_metrics_all_configs.csv (unit: minutes)
# Order: ERA5_Fund, DWD_Fund, ERA5_EXAA, DWD_EXAA, EXAA_Naive
sqra = np.array([
    [(0.0380, 0.0359, 0.0464)],  # ERA5_Fundamental
    [(0.0379, 0.0338, 0.0701)],  # DWD_Fundamental
    [(0.0330, 0.0321, 0.0414)],  # ERA5_EXAA_Enriched
    [(0.0338, 0.0312, 0.0424)],  # DWD_EXAA_Enriched
    [(0.0343, 0.0313, 0.0557)],  # EXAA_Naive
    [(0.0324, 0.0308, 0.0393)],  # EXAA_Only
], dtype=object)

# Row labels and p-values for the SQRA table
sqra_row_labels = [
    (r"$\mathrm{SQRA}_{\mathrm{ERA5,\,Fundamental}}$",  "$p=3$"),
    (r"$\mathrm{SQRA}_{\mathrm{DWD,\,Fundamental}}$",   "$p=3$"),
    (r"$\mathrm{SQRA}_{\mathrm{ERA5,EXAA}}$",  "$p=3$"),
    (r"$\mathrm{SQRA}_{\mathrm{DWD,EXAA}}$",   "$p=3$"),
    (r"$\mathrm{SQRA}_{\mathrm{EXAA,Naive}}$",  "$p=1$"),
    (r"$\mathrm{SQRA}_{\mathrm{EXAA,Only}}$",  "$p=3$"),
]


def _runtime_metrics(path: Path, config) -> tuple[float, float, float]:
    """Return mean/min/max daily runtime in minutes for the evaluation sample."""
    df = pd.read_csv(path)
    df["forecast_day"] = pd.to_datetime(df["forecast_day"], utc=True).dt.tz_convert(config.timezone)
    expected_days = {
        day.date()
        for day in pd.date_range(
            config.evaluation_start, config.evaluation_end, freq="D", tz=config.timezone
        )
        if day.date() not in set(config.evaluation_skip_dates)
    }
    available_days = set(df["forecast_day"].dt.date)
    missing_days = sorted(expected_days - available_days)
    if missing_days:
        preview = ", ".join(map(str, missing_days[:8]))
        raise ValueError(f"Incomplete evaluation runtime coverage in {path}; missing: {preview}")
    dates = pd.Index(df["forecast_day"].dt.date)
    mask = (
        (dates >= config.evaluation_start)
        & (dates <= config.evaluation_end)
        & ~dates.isin(config.evaluation_skip_dates)
    )
    seconds_column = (
        "runtime_seconds" if "runtime_seconds" in df.columns else "computation_time_seconds"
    )
    minutes = df.loc[mask, seconds_column].dropna().astype(float) / 60.0
    if minutes.empty:
        raise ValueError(f"No evaluation runtimes in {path}")
    return float(minutes.mean()), float(minutes.min()), float(minutes.max())


def load_measurements(config):
    base = config.results_root
    point = base / "lear_op_results"
    fundamental = []
    enriched = []
    for clusters in (1, 5, 25):
        fundamental.append([
            _runtime_metrics(point / "dwd" / "d56" / f"c{clusters}" / "fundamental" / "runtime.csv", config),
            *[
                _runtime_metrics(point / "era5" / f"d{days}" / f"c{clusters}" / "fundamental" / "runtime.csv", config)
                for days in (56, 112, 364)
            ],
        ])
        enriched.append([
            _runtime_metrics(point / "dwd" / "d56" / f"c{clusters}" / "exaa" / "runtime.csv", config),
            *[
                _runtime_metrics(point / "era5" / f"d{days}" / f"c{clusters}" / "exaa" / "runtime.csv", config)
                for days in (56, 112, 364)
            ],
        ])
    only = [[
        _runtime_metrics(point / "exaa_only" / f"d{days}" / "runtime.csv", config)
        for days in (56, 112, 364)
    ]]
    sqra_values = [
        [_runtime_metrics(base / "sqra_results" / name / "runtime.csv", config)]
        for name in (
            "era5_fundamental",
            "dwd_fundamental",
            "era5_exaa_enriched",
            "dwd_exaa_enriched",
            "exaa_naive",
            "exaa_only",
        )
    ]
    return tuple(np.array(values, dtype=object) for values in (
        fundamental, enriched, only, sqra_values
    ))


# ============================================================
#  CONFIGURATION
# ============================================================

DPI          = 300
FIGSIZE_LEAR = (7, 3.2)
FIGSIZE_SQRA = (3.5, 2.0)

col_labels_lear = ["$D_{\\mathrm{LEAR}}=56$", "$D_{\\mathrm{LEAR}}=56$",
                   "$D_{\\mathrm{LEAR}}=112$", "$D_{\\mathrm{LEAR}}=364$"]
col_labels_sqra = ["$D_{\\mathrm{SQRA}}=60$"]

# p-values (baseline, without EXAA): ERA5 | ICON-D2, one per row (C=1, C=5, C=25)
# With EXAA, each row gains +96 additional columns (exaa_offset=96).
p_base_lear = [
    (441, 609),
    (633, 1473),
    (1593, 5793),
]
c_labels_lear = ["$C=1$", "$C=5$", "$C=25$"]

def fmt_thousands(n):
    """
    Format an integer as a LaTeX string with a thousands separator.

    Numbers below 1000 are returned as plain strings.  Numbers >= 1000
    use LaTeX's ``{{,}}`` thin-space separator (e.g. 1{,}473).

    Parameters
    ----------
    n : int
        Non-negative integer to format.

    Returns
    -------
    str
        LaTeX-compatible string representation of *n*.
    """
    if n >= 1000:
        thousands = n // 1000
        rest = n % 1000
        return f"{thousands}{{,}}{rest:03d}"
    return str(n)

def make_p_labels(offset=0):
    labels = []
    for era5, icond2 in p_base_lear:
        p1 = icond2 + offset
        p2 = era5   + offset
        labels.append(f"$p={fmt_thousands(p1)}\\,|\\,{fmt_thousands(p2)}$")
    return labels


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

CMAP_MINVAL = 0.3
CMAP_MAXVAL = 1.0

def truncated_cmap(cmap_name, minval=CMAP_MINVAL, maxval=CMAP_MAXVAL, n=256):
    """
    Return a colormap restricted to a sub-range of the original.

    Uses only the portion of the colormap between *minval* and *maxval*
    to dampen contrast and avoid hard colour extremes.

    Parameters
    ----------
    cmap_name : str
        Name of a registered Matplotlib colormap (e.g. ``"RdYlGn_r"``).
    minval : float, optional
        Lower bound of the sub-range, in [0, 1].  Default is
        ``CMAP_MINVAL``.
    maxval : float, optional
        Upper bound of the sub-range, in [0, 1].  Default is
        ``CMAP_MAXVAL``.
    n : int, optional
        Number of colour samples used to construct the new colormap.
        Default is 256.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        New colormap covering only [minval, maxval] of the original.
    """
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}", colors)

def extract(data):
    """Split object array of (mean, min, max) tuples into three float arrays."""
    mean = np.array([[c[0] for c in row] for row in data], dtype=float)
    lo   = np.array([[c[1] for c in row] for row in data], dtype=float)
    hi   = np.array([[c[2] for c in row] for row in data], dtype=float)
    return mean, lo, hi

def get_vmax(*datasets):
    """
    Compute the global mean-value maximum across one or more datasets.

    NaN values are ignored.  Returns 10.0 as a safe fallback when all
    values are NaN.

    Parameters
    ----------
    *datasets : array-like
        One or more object arrays of (mean, min, max) tuples, as
        accepted by :func:`extract`.

    Returns
    -------
    float
        Global maximum of all mean values.
    """
    vals = np.concatenate([extract(d)[0].flatten() for d in datasets])
    vals = vals[~np.isnan(vals)]
    return float(np.max(vals)) if len(vals) > 0 else 10.0

def fmt_mean(val, decimals=1):
    """
    Format a mean computation-time value as a fixed-point string.

    Parameters
    ----------
    val : float
        Mean value.  NaN is rendered as the placeholder ``"x.xx"``.
    decimals : int, optional
        Number of decimal places.  Default is 1.

    Returns
    -------
    str
        Fixed-point string, or ``"x.xx"`` for NaN.
    """
    if np.isnan(val):
        return "x.xx"
    return f"{val:.{decimals}f}"

def fmt_range(lo, hi, decimals=1):
    """
    Format a [min–max] range as a parenthesised string with an en-dash.

    Parameters
    ----------
    lo : float
        Minimum value.  NaN triggers the placeholder output.
    hi : float
        Maximum value.  NaN triggers the placeholder output.
    decimals : int, optional
        Number of decimal places.  Default is 1.

    Returns
    -------
    str
        String of the form ``"(lo–hi)"``, or ``"(x.x–x.x)"`` if either
        value is NaN.
    """
    if np.isnan(lo) or np.isnan(hi):
        return "(x.x\u2013x.x)"
    return f"({lo:.{decimals}f}\u2013{hi:.{decimals}f})"

def text_color(bg):
    """
    Choose black or white foreground for readability against a background colour.

    Uses the ITU-R BT.601 luminance formula to decide whether the
    background is perceived as dark (returns ``"white"``) or light
    (returns ``"black"``).

    Parameters
    ----------
    bg : sequence of float
        RGB colour as a sequence of three floats in [0, 1].

    Returns
    -------
    str
        ``"white"`` if the background is dark, ``"black"`` otherwise.
    """
    brightness = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    return "white" if brightness < 0.45 else "black"

def subtext_color(bg):
    """
    Choose a muted foreground colour for secondary text against a background.

    Uses the same luminance formula as :func:`text_color` but returns
    lighter or darker grey instead of pure black/white.

    Parameters
    ----------
    bg : sequence of float
        RGB colour as a sequence of three floats in [0, 1].

    Returns
    -------
    str
        ``"#cccccc"`` for dark backgrounds, ``"#555555"`` for light ones.
    """
    brightness = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    return "#cccccc" if brightness < 0.45 else "#555555"

def draw_cells(ax, mean_data, lo_data, hi_data, norm, cmap_obj, decimals=1):
    """
    Render coloured cells with mean and range annotations onto an Axes.

    Each cell is a filled rectangle coloured according to the mean value.
    The mean is printed in bold at the top of the cell; the [min–max]
    range in smaller text beneath it.  Text colours are chosen
    automatically for legibility.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target Axes.
    mean_data : numpy.ndarray of shape (n_rows, n_cols)
        Mean values; NaN cells use the colourmap minimum colour.
    lo_data : numpy.ndarray of shape (n_rows, n_cols)
        Minimum values displayed as part of the secondary text.
    hi_data : numpy.ndarray of shape (n_rows, n_cols)
        Maximum values displayed as part of the secondary text.
    norm : matplotlib.colors.Normalize
        Normalisation mapping mean values to [0, 1] for the colormap.
    cmap_obj : matplotlib.colors.Colormap
        Colormap used to map normalised values to colours.
    decimals : int, optional
        Number of decimal places for both the mean and range labels.
        Default is 1.

    Returns
    -------
    None
    """
    n_rows, n_cols = mean_data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = mean_data[i, j]
            color = cmap_obj(norm(val if not np.isnan(val) else 0))
            ax.add_patch(plt.Rectangle([j, n_rows - i - 1], 1, 1,
                                       color=color, ec="white", lw=1.5))
            ax.text(j + 0.5, n_rows - i - 0.38, fmt_mean(val, decimals),
                    ha="center", va="center", fontsize=10,
                    color=text_color(color), fontweight="bold")
            ax.text(j + 0.5, n_rows - i - 0.68, fmt_range(lo_data[i,j], hi_data[i,j], decimals),
                    ha="center", va="center", fontsize=8.5,
                    color=subtext_color(color))

def add_group_headers(fig, ax, n_cols, separator_after=3):
    """
    Add "ICON-D2" and "ERA5" group headers above the column labels.

    Draws two bold text labels and underlines spanning their respective
    column groups, placed just above the top edge of the Axes in figure
    coordinates.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure (needed for figure-coordinate transforms and
        :meth:`fig.text`).
    ax : matplotlib.axes.Axes
        Axes whose column layout determines label positions.
    n_cols : int
        Total number of data columns.
    separator_after : int, optional
        Column index after which the group boundary is drawn.  Columns
        ``[0, separator_after)`` belong to the first group ("ICON-D2")
        and ``[separator_after, n_cols)`` to the second ("ERA5").
        Default is 3.

    Returns
    -------
    None
    """
    fig.canvas.draw()
    ax_trans = ax.transData + fig.transFigure.inverted()
    col_edges = [ax_trans.transform((j, 0))[0] for j in range(n_cols + 1)]

    era5_left    = col_edges[0]
    sep_x        = col_edges[separator_after]
    icond2_right = col_edges[n_cols]
    era5_x       = (era5_left + sep_x) / 2
    icond2_x     = (sep_x + icond2_right) / 2

    header_y = ax.get_position().y1 + 0.06
    ul_y     = header_y - 0.025
    gap      = 0.008

    fig.text(era5_x,   header_y, "ICON-D2", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
    fig.text(icond2_x, header_y, "ERA5",    ha="center", va="bottom",
             fontsize=10, fontweight="bold")
    for x0, x1 in [(era5_left+gap, sep_x-gap), (sep_x+gap, icond2_right-gap)]:
        fig.add_artist(plt.Line2D([x0, x1], [ul_y, ul_y],
                                  transform=fig.transFigure,
                                  color="black", lw=0.8))


def make_lear_figure(data, filename, vmin, vmax, exaa_offset=0):
    """
    Produce and save a 3-row LEAR computation time heatmap table as a PDF.

    Renders a colour-coded table with ICON-D2 and ERA5 group headers,
    row labels showing cluster count and feature count p, and column
    labels for training window lengths.  A colorbar is added to the right.

    Parameters
    ----------
    data : numpy.ndarray of dtype object
        Array of (mean, min, max) tuples with shape (n_rows, n_cols).
    filename : Path or str
        Destination file path for the saved PDF.
    vmin : float
        Lower bound of the colormap normalisation.
    vmax : float
        Upper bound of the colormap normalisation (global maximum).
    exaa_offset : int, optional
        Number of additional EXAA feature columns to add to the base
        p-values when constructing row labels.  Default is 0.

    Returns
    -------
    None
    """
    mean, lo, hi = extract(data)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = truncated_cmap("RdYlGn_r")
    n_rows, n_cols = mean.shape

    fig, ax = plt.subplots(figsize=FIGSIZE_LEAR)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.72, bottom=0.18)

    draw_cells(ax, mean, lo, hi, norm, cmap_obj)
    ax.axvline(x=1, color="black", lw=2.5)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([j+0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels_lear, fontsize=9)
    ax.xaxis.set_tick_params(length=0)

    p_labels = make_p_labels(offset=exaa_offset)
    ax.set_yticks([])
    for i, label in enumerate(c_labels_lear):
        y = n_rows - i - 0.38
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=9,
                transform=ax.transData)
        ax.text(-0.15, y - 0.32, p_labels[i], ha="right", va="center",
                fontsize=8, color="#666666", transform=ax.transData)

    ax.set_frame_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Computation Time per\nForecasting Day (Minutes)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    add_group_headers(fig, ax, n_cols=n_cols, separator_after=1)
    fig.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def make_sqra_figure(data, filename, vmin, vmax):
    """
    Produce and save a multi-row SQRA computation time heatmap table as a PDF.

    Renders a colour-coded single-column table with model name and
    feature-count labels on the left, and a colorbar on the right.

    Parameters
    ----------
    data : numpy.ndarray of dtype object
        Array of (mean, min, max) tuples with shape (n_rows, 1).
    filename : Path or str
        Destination file path for the saved PDF.
    vmin : float
        Lower bound of the colormap normalisation.
    vmax : float
        Upper bound of the colormap normalisation.

    Returns
    -------
    None
    """
    mean, lo, hi = extract(data)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = truncated_cmap("RdYlGn_r")
    n_rows, n_cols = mean.shape

    fig_h = max(2.0, n_rows * 0.52 + 0.5)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))
    fig.subplots_adjust(left=0.05, right=0.78, top=0.88, bottom=0.08)

    draw_cells(ax, mean, lo, hi, norm, cmap_obj, decimals=2)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([j+0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels_sqra, fontsize=11)
    ax.xaxis.set_tick_params(length=0)

    ax.set_yticks([])
    for i, (label, p_label) in enumerate(sqra_row_labels):
        y = n_rows - i - 0.38
        ax.text(-0.08, y, label, ha="right", va="center", fontsize=11,
                transform=ax.transData)
        ax.text(-0.08, y - 0.32, p_label, ha="right", va="center",
                fontsize=9, color="#666666", transform=ax.transData)

    ax.set_frame_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.08, pad=0.04)
    cbar.set_label("Computation Time per\nForecasting Day (Minutes)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

def make_exaa_only_figure(data, filename, vmin, vmax):
    """
    Produce and save a single-row EXAA Only LEAR heatmap table as a PDF.

    Renders a compact one-row colour-coded table for the EXAA Only LEAR
    model (ERA5 only, no cluster dimension C, three training window lengths).
    A colorbar is added to the right.

    Parameters
    ----------
    data : numpy.ndarray of dtype object
        Array of (mean, min, max) tuples with shape (1, 3).
    filename : Path or str
        Destination file path for the saved PDF.
    vmin : float
        Lower bound of the colormap normalisation.
    vmax : float
        Upper bound of the colormap normalisation.

    Returns
    -------
    None
    """
    mean, lo, hi = extract(data)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = truncated_cmap("RdYlGn_r")
    n_rows, n_cols = mean.shape  # yields 1 x 3

    col_labels_scaled = ["$D_{\\mathrm{LEAR}}=56$", "$D_{\\mathrm{LEAR}}=112$",
                         "$D_{\\mathrm{LEAR}}=364$"]

    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    fig.subplots_adjust(left=0.16, right=0.68, top=0.72, bottom=0.28)

    draw_cells(ax, mean, lo, hi, norm, cmap_obj, decimals=2)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels_scaled, fontsize=9)
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticks([])

    # Row label: no C-value, only p=96
    ax.text(-0.08, 0.62, "$p=96$", ha="right", va="center",
            fontsize=10, color="#666666", transform=ax.transData)

    ax.set_frame_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("Computation Time per\nForecasting Day (Minutes)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(filename, dpi=DPI, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"Saved: {filename}")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    config = load_experiment_config(REPO_ROOT)
    lear_fundamental, lear_exaa_enriched, lear_exaa_only, sqra = load_measurements(config)
    _OUT  = config.output_root
    _OUT.mkdir(parents=True, exist_ok=True)

    VMIN_LEAR = 0
    VMAX_LEAR = get_vmax(lear_fundamental, lear_exaa_enriched, lear_exaa_only, sqra)  # shared scale
    VMIN_SQRA = VMIN_LEAR
    VMAX_SQRA = VMAX_LEAR

    make_lear_figure(lear_fundamental,   _OUT / "comptime_lear_fundamental.pdf",   VMIN_LEAR, VMAX_LEAR, exaa_offset=0)
    make_lear_figure(lear_exaa_enriched, _OUT / "comptime_lear_exaa_enriched.pdf", VMIN_LEAR, VMAX_LEAR, exaa_offset=96)
    make_exaa_only_figure(lear_exaa_only, _OUT / "comptime_lear_exaa_only.pdf", VMIN_LEAR, VMAX_LEAR)
    make_sqra_figure(sqra,           _OUT / "comptime_sqra.pdf",           VMIN_SQRA, VMAX_SQRA)

    print("\nDone! All four PDFs have been saved.")
