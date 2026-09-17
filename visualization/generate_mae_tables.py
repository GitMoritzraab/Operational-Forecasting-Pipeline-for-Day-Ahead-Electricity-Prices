"""
generate_mae_tables.py
======================

Generates colour-coded MAE/RMSE heatmap tables for LEAR model results.

Each cell shows the Mean Absolute Error (MAE) in bold and the Root Mean
Squared Error (RMSE) in smaller text beneath it. Cells are shaded using a
truncated RdYlGn_r colormap so that lower errors appear green and higher
errors appear red, with the colour range clipped to avoid overly saturated
extremes.

Three tables are produced:

- LEAR Fundamental    (3 rows x 4 columns)
- LEAR EXAA Enriched  (3 rows x 4 columns)
- LEAR EXAA Only      (1 row  x 3 columns)

Expected repository layout
--------------------------
repo/
├── output/              (created automatically if missing)
└── visualization/
    └── generate_mae_tables.py
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
from pipeline.delivery_index import (
    delivery_dates,
    read_forecast_csv,
    validate_delivery_index,
)

# ============================================================
#  ENTER MEASUREMENTS HERE
#  Format per cell: (mae, rmse)  in €/MWh
#  Unknown values: (np.nan, np.nan)
#
#  Rows:    C=1, C=5, C=25
#  Columns: ICON-D2 D=56, ERA5 D=56, ERA5 D=112, ERA5 D=364
# ============================================================

lear_fundamental = np.array([
    # ICON-D2 D=56       ERA5 D=56      ERA5 D=112      ERA5 D=364
    [(13.63, 21.45), (13.18, 20.55), (13.66, 21.07), (12.87, 19.55)],  # C=1
    [(14.00, 21.95), (13.01, 20.54), (12.59, 20.13), (11.65, 17.71)],  # C=5
    [(13.94, 25.05), (14.30, 31.08), (13.02, 20.07), (12.03, 18.14)],  # C=25
], dtype=object)

exaa_only = np.array([
    # D=56,                   D=112,             D=364
    [(7.7765, 12.3097), (7.2945, 11.4498), (7.2054, 11.2965)],  # EXAA_Scaled
], dtype=object)

lear_exaa_enriched = np.array([
    # ICON-D2 D=56    ERA5 D=56        ERA5 D=112    ERA5 D=364
    [(8.13, 12.84), (7.98, 12.57), (7.47, 11.64), (7.36, 11.60)],  # C=1
    [(8.40, 13.01), (8.31, 12.97), (7.64, 11.90), (7.43, 11.59)],  # C=5
    [(8.55, 13.38), (8.63, 13.52), (7.75, 12.09), (7.37, 11.34)],  # C=25
], dtype=object)


# ============================================================
#  CONFIGURATION
# ============================================================

DPI          = 300
FIGSIZE_LEAR = (7, 2.8)

# Muted colormap per table: use only the middle range (no hard red/green)
# Values between 0.0 (hard end) and 1.0 (full end)
CMAP_MINVAL_FUNDAMENTAL = 0.4
CMAP_MAXVAL_FUNDAMENTAL = 0.7

CMAP_MINVAL_EXAA    = 0.2
CMAP_MAXVAL_EXAA    = 0.35

row_labels_lear = ["$C=1$", "$C=5$", "$C=25$"]
col_labels_lear = ["$D_{\\mathrm{LEAR}}=56$", "$D_{\\mathrm{LEAR}}=56$",
                   "$D_{\\mathrm{LEAR}}=112$", "$D_{\\mathrm{LEAR}}=364$"]
col_labels_scaled = ["$D_{\\mathrm{LEAR}}=56$", "$D_{\\mathrm{LEAR}}=112$",
                     "$D_{\\mathrm{LEAR}}=364$"]


def _forecast_metrics(path: Path, config) -> tuple[float, float]:
    """Calculate MAE/RMSE over the centrally configured evaluation sample."""
    df = read_forecast_csv(
        path,
        required_columns=("y_pred", "y_true"),
        require_complete_days=True,
    )
    start = pd.Timestamp(config.evaluation_start).normalize()
    end = pd.Timestamp(config.evaluation_end).normalize()
    skipped = {
        pd.Timestamp(day).normalize() for day in config.evaluation_skip_dates
    }
    expected_days = [
        day
        for day in pd.date_range(start, end, freq="D")
        if day not in skipped
    ]
    dates = delivery_dates(df.index)
    mask = (
        (dates >= start)
        & (dates <= end)
        & ~dates.isin(skipped)
    )
    sample = df.loc[mask, ["y_pred", "y_true"]]
    validate_delivery_index(sample.index, expected_days=expected_days)
    if sample.isna().any().any():
        missing_columns = sample.columns[sample.isna().any()].tolist()
        raise ValueError(f"Missing evaluation values in {path}: {missing_columns}")
    if sample.empty:
        raise ValueError(f"No evaluation observations in {path}")
    error = sample["y_pred"] - sample["y_true"]
    return float(error.abs().mean()), float(np.sqrt(np.mean(np.square(error))))


def load_measurements(config):
    """Load the same table layout as the manuscript directly from forecasts."""
    base = config.results_root / "lear_op_results"
    fundamental = []
    enriched = []
    for clusters in (1, 5, 25):
        fundamental.append([
            _forecast_metrics(base / "dwd" / "d56" / f"c{clusters}" / "fundamental" / "forecast.csv", config),
            *[
                _forecast_metrics(base / "era5" / f"d{days}" / f"c{clusters}" / "fundamental" / "forecast.csv", config)
                for days in (56, 112, 364)
            ],
        ])
        enriched.append([
            _forecast_metrics(base / "dwd" / "d56" / f"c{clusters}" / "exaa" / "forecast.csv", config),
            *[
                _forecast_metrics(base / "era5" / f"d{days}" / f"c{clusters}" / "exaa" / "forecast.csv", config)
                for days in (56, 112, 364)
            ],
        ])
    only = [[
        _forecast_metrics(base / "exaa_only" / f"d{days}" / "forecast.csv", config)
        for days in (56, 112, 364)
    ]]
    return (
        np.array(fundamental, dtype=object),
        np.array(enriched, dtype=object),
        np.array(only, dtype=object),
    )


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def truncated_cmap(cmap_name, minval, maxval, n=256):
    """
    Return a colormap restricted to a sub-range of the original.

    Uses only the middle portion of the colormap to dampen contrast
    and avoid hard red/green extremes.

    Parameters
    ----------
    cmap_name : str
        Name of a registered Matplotlib colormap (e.g. ``"RdYlGn_r"``).
    minval : float
        Lower bound of the sub-range, in [0, 1].
    maxval : float
        Upper bound of the sub-range, in [0, 1].
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
    """Split object array of (mae, rmse) tuples into two float arrays."""
    mae  = np.array([[c[0] for c in row] for row in data], dtype=float)
    rmse = np.array([[c[1] for c in row] for row in data], dtype=float)
    return mae, rmse

def get_vmin_vmax(*datasets):
    """
    Compute the global MAE minimum and maximum across one or more datasets.

    NaN values are ignored.  If all values are NaN, returns (0.0, 10.0)
    as a safe fallback.

    Parameters
    ----------
    *datasets : array-like
        One or more object arrays of (mae, rmse) tuples, as accepted by
        :func:`extract`.

    Returns
    -------
    vmin : float
        Global minimum MAE.
    vmax : float
        Global maximum MAE.
    """
    vals = np.concatenate([extract(d)[0].flatten() for d in datasets])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0, 10.0
    return float(np.min(vals)), float(np.max(vals))

def fmt_mae(val):
    """
    Format a MAE value as a fixed-point string.

    Parameters
    ----------
    val : float
        MAE value.  NaN is rendered as the placeholder ``"x.xx"``.

    Returns
    -------
    str
        Two-decimal string representation, or ``"x.xx"`` for NaN.
    """
    return "x.xx" if np.isnan(val) else f"{val:.2f}"

def fmt_rmse(val):
    """
    Format an RMSE value as a parenthesised fixed-point string.

    Parameters
    ----------
    val : float
        RMSE value.  NaN is rendered as the placeholder ``"(x.xx)"``.

    Returns
    -------
    str
        Two-decimal string in parentheses, or ``"(x.xx)"`` for NaN.
    """
    return "(x.xx)" if np.isnan(val) else f"({val:.2f})"

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
        ``"#aaaaaa"`` for dark backgrounds, ``"#333333"`` for light ones.
    """
    brightness = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    return "#aaaaaa" if brightness < 0.45 else "#333333"

def draw_cells(ax, mae_data, rmse_data, norm, cmap_obj):
    """
    Render coloured cells with MAE and RMSE annotations onto an Axes.

    Each cell is a filled rectangle coloured according to the MAE value.
    The MAE is printed in bold at the top of the cell; the RMSE in smaller
    text beneath it.  Text colours are chosen automatically for legibility.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target Axes.
    mae_data : numpy.ndarray of shape (n_rows, n_cols)
        MAE values; NaN cells use the colormap maximum colour.
    rmse_data : numpy.ndarray of shape (n_rows, n_cols)
        RMSE values displayed as secondary text.
    norm : matplotlib.colors.Normalize
        Normalisation mapping MAE values to [0, 1] for the colormap.
    cmap_obj : matplotlib.colors.Colormap
        Colormap used to map normalised values to colours.

    Returns
    -------
    None
    """
    n_rows, n_cols = mae_data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = mae_data[i, j]
            color = cmap_obj(norm(val if not np.isnan(val) else norm.vmax))
            ax.add_patch(plt.Rectangle([j, n_rows - i - 1], 1, 1,
                                       color=color, ec="white", lw=1.5))
            ax.text(j + 0.5, n_rows - i - 0.38, fmt_mae(val),
                    ha="center", va="center", fontsize=11,
                    color=text_color(color), fontweight="bold")
            ax.text(j + 0.5, n_rows - i - 0.68, fmt_rmse(rmse_data[i, j]),
                    ha="center", va="center", fontsize=7.5,
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


def make_lear_figure(data, filename, vmin, vmax, cmap_minval, cmap_maxval):
    """
    Produce and save a 3-row LEAR heatmap table as a PDF.

    Renders a colour-coded table with ICON-D2 and ERA5 group headers,
    row labels for cluster counts, and column labels for training window
    lengths.  A colorbar is added to the right.

    Parameters
    ----------
    data : numpy.ndarray of dtype object
        Array of (mae, rmse) tuples with shape (n_rows, n_cols).
    filename : Path or str
        Destination file path for the saved PDF.
    vmin : float
        Lower bound of the colormap normalisation (global MAE minimum).
    vmax : float
        Upper bound of the colormap normalisation (global MAE maximum).
    cmap_minval : float
        Lower truncation bound for the colormap sub-range, in [0, 1].
    cmap_maxval : float
        Upper truncation bound for the colormap sub-range, in [0, 1].

    Returns
    -------
    None
    """
    mae, rmse = extract(data)
    cmap_obj = truncated_cmap("RdYlGn_r", cmap_minval, cmap_maxval)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    n_rows, n_cols = mae.shape

    fig, ax = plt.subplots(figsize=FIGSIZE_LEAR)
    fig.subplots_adjust(left=0.12, right=0.82, top=0.72, bottom=0.18)

    draw_cells(ax, mae, rmse, norm, cmap_obj)
    ax.axvline(x=1, color="black", lw=2.5)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([j+0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels_lear, fontsize=9)
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticks([n_rows-i-0.5 for i in range(n_rows)])
    ax.set_yticklabels(row_labels_lear, fontsize=9)
    ax.yaxis.set_tick_params(length=0)
    ax.set_frame_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("MAE (€/MWh)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    add_group_headers(fig, ax, n_cols=n_cols, separator_after=1)
    fig.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def make_exaa_only_figure(data, filename, vmin, vmax, cmap_minval, cmap_maxval):
    """
    Produce and save a single-row EXAA Only heatmap table as a PDF.

    Renders a compact one-row colour-coded table without group headers.
    A narrow colorbar is appended to the right using
    :class:`mpl_toolkits.axes_grid1.make_axes_locatable`.

    Parameters
    ----------
    data : numpy.ndarray of dtype object
        Array of (mae, rmse) tuples with shape (1, n_cols).
    filename : Path or str
        Destination file path for the saved PDF.
    vmin : float
        Lower bound of the colormap normalisation.
    vmax : float
        Upper bound of the colormap normalisation.
    cmap_minval : float
        Lower truncation bound for the colormap sub-range, in [0, 1].
    cmap_maxval : float
        Upper truncation bound for the colormap sub-range, in [0, 1].

    Returns
    -------
    None
    """
    mae, rmse = extract(data)
    cmap_obj = truncated_cmap("RdYlGn_r", cmap_minval, cmap_maxval)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    n_rows, n_cols = mae.shape  # yields (1, 3)

    fig, ax = plt.subplots(figsize=(6.5, 1.2))
    fig.subplots_adjust(left=0.01, right=0.75, top=0.60, bottom=0.25)

    draw_cells(ax, mae, rmse, norm, cmap_obj)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels_scaled, fontsize=9)
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticks([])
    ax.set_frame_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("MAE (€/MWh)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(filename, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved: {filename}")


# ============================================================
#  SEPARATE COLOUR SCALES & RUN
# ============================================================

if __name__ == "__main__":
    config = load_experiment_config(REPO_ROOT)
    lear_fundamental, lear_exaa_enriched, exaa_only = load_measurements(config)
    _OUT  = config.output_root
    _OUT.mkdir(parents=True, exist_ok=True)

    vmin_fundamental, vmax_fundamental = get_vmin_vmax(lear_fundamental)
    vmin_exaa,    vmax_exaa    = get_vmin_vmax(lear_exaa_enriched)

    make_lear_figure(lear_fundamental, _OUT / "mae_lear_fundamental.pdf",  vmin_fundamental, vmax_fundamental,
                     CMAP_MINVAL_FUNDAMENTAL, CMAP_MAXVAL_FUNDAMENTAL)
    make_lear_figure(lear_exaa_enriched,    _OUT / "mae_lear_exaa_enriched.pdf",    vmin_exaa,    vmax_exaa,
                     CMAP_MINVAL_EXAA,    CMAP_MAXVAL_EXAA)
    make_exaa_only_figure(exaa_only, _OUT / "mae_exaa_only.pdf", vmin_exaa, vmax_exaa,
                       CMAP_MINVAL_EXAA, CMAP_MAXVAL_EXAA)

    print("\nDone! All PDFs have been saved.")
