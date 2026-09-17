"""
plot_prob_forecast_example.py
=============================

Visualises a probabilistic electricity-price forecast as a publication-ready
figure. The script reads a quantile-forecast CSV produced by the SQRA
pipeline and renders shaded prediction intervals together with the median
forecast and the realised spot price.

Expected repository layout
--------------------------
repo/
├── results/sqra_results/    (input CSVs)
├── output/                  (created automatically if missing)
└── visualization/
    └── plot_prob_forecast_example.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional
import sys

# ============================================================
#  CONFIGURATION
# ============================================================

_HERE = Path(__file__).parent.parent  # points to repository root
sys.path.insert(0, str(_HERE))
from experiment_config import load_experiment_config
from pipeline.delivery_index import (
    delivery_clock_index,
    read_forecast_csv as read_canonical_forecast_csv,
)
EXPERIMENT = load_experiment_config(_HERE)

# Path to the forecast CSV file
FORECAST_PATH = EXPERIMENT.results_root / "sqra_results" / "era5_fundamental" / "forecast.csv"

# Export path for the saved plot (None = display only, do not save)
EXPORT_PATH = EXPERIMENT.output_root / "plot_prob_forecast_example.pdf"

# Time window for the plot (None = entire dataset)
START_DATE = "2025-12-05"
END_DATE   = "2025-12-09 23:45"

# Column name of the realised prices (None = do not plot)
Y_TRUE_COL = "y_true"

# ============================================================
#  LOAD DATA
# ============================================================

def load_forecast_csv(path: Path) -> pd.DataFrame:
    """
    Load a canonical quantile-forecast CSV for clock-based display.

    The CSV is expected to contain columns ``q0.100``, ``q0.250``,
    ``q0.500``, ``q0.750``, ``q0.900``, and optionally ``y_true``.
    The file must use the explicit ``delivery_date``/``mtu`` result index.
    For plotting only, those keys are converted to a timezone-naive local
    clock index.  This preserves all 96 normalized MTUs even on daylight-
    saving transition dates.  Column names are then normalised to the
    ``q_<level>`` convention used throughout this script
    (e.g. ``q0.100`` → ``q_0.1``).

    Parameters
    ----------
    path : Path
        Absolute or relative path to the forecast CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a timezone-naive local-clock ``DatetimeIndex`` and
        renamed quantile columns.
    """
    required_quantiles = (
        "q0.100",
        "q0.250",
        "q0.500",
        "q0.750",
        "q0.900",
    )
    df = read_canonical_forecast_csv(
        path,
        required_columns=required_quantiles,
        require_complete_days=True,
    )
    df.index = delivery_clock_index(df.index)

    # Rename columns: q0.100 -> q_0.1 etc. (uniform format)
    rename_map = {
        "q0.100": "q_0.1",
        "q0.250": "q_0.25",
        "q0.500": "q_0.5",
        "q0.750": "q_0.75",
        "q0.900": "q_0.9",
    }
    df = df.rename(columns=rename_map)

    return df


# ============================================================
#  PLOTTING FUNCTION
# ============================================================

def plot_prob_forecast_paper(
    df_forecast:   pd.DataFrame,
    quantile_cols: list,
    start_date:    Optional[str] = None,
    end_date:      Optional[str] = None,
    y_true_col:    Optional[str] = "y_true",
    save_path:     Optional[Path] = None,
    dpi:           int = 300,
):
    """
    Render a publication-ready probabilistic forecast plot.

    The figure shows two nested prediction intervals (outer 80 % and
    inner 50 %), the median forecast line, and—optionally—the realised
    price series.  Vertical dashed lines mark midnight boundaries, and
    the axes are styled for serif-font academic publication.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        DataFrame returned by :func:`load_forecast_csv`.  Must contain
        at least the columns listed in *quantile_cols* and have a
        timezone-naive local-clock ``DatetimeIndex``.
    quantile_cols : list of str
        Ordered list of column names representing quantile levels, from
        lowest to highest (e.g. ``["q_0.1", "q_0.25", "q_0.5",
        "q_0.75", "q_0.9"]``).  The first and last entries define the
        outer prediction interval; the second and second-to-last define
        the inner interval; the middle entry is used as the median.
    start_date : str, optional
        ISO-8601 string (e.g. ``"2026-02-02"``) for the left edge of
        the plot window.  If ``None``, the series starts at the first
        available timestamp.
    end_date : str, optional
        ISO-8601 string (e.g. ``"2026-02-05 23:45"``) for the right
        edge of the plot window.  If ``None``, the series ends at the
        last available timestamp.
    y_true_col : str or None, optional
        Column name of the realised price series.  Set to ``None`` to
        suppress this line entirely.  Default is ``"y_true"``.
    save_path : Path or None, optional
        Destination path for the saved figure.  Parent directories are
        created automatically.  If ``None``, the figure is only
        displayed and not written to disk.  Default is ``None``.
    dpi : int, optional
        Resolution in dots per inch used when saving the figure.
        Default is ``300``.

    Returns
    -------
    None
        The figure is displayed via :func:`matplotlib.pyplot.show` and,
        if *save_path* is provided, written to disk.

    Raises
    ------
    ValueError
        If the time window selected by *start_date* / *end_date*
        contains no rows.
    """
    # Select time window
    if start_date is not None and end_date is not None:
        df = df_forecast.loc[start_date:end_date].copy()
    elif start_date is not None:
        df = df_forecast.loc[start_date:].copy()
    elif end_date is not None:
        df = df_forecast.loc[:end_date].copy()
    else:
        df = df_forecast.copy()

    if df.empty:
        raise ValueError("The selected time window is empty. Please check START_DATE / END_DATE.")

    n            = len(quantile_cols)
    q_outer_low  = quantile_cols[0]
    q_outer_high = quantile_cols[-1]
    q_median     = quantile_cols[n // 2]
    q_inner_low  = quantile_cols[1] if n >= 4 else None
    q_inner_high = quantile_cols[-2] if n >= 4 else None

    # ── Style ────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset":  "cm",
        "font.size":         13,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.5,
        "grid.color":        "#cccccc",
        "grid.linestyle":    "--",
    })

    COLOR_PI_OUTER = "#d0e4f5"
    COLOR_PI_INNER = "#3d8fc4"
    COLOR_MEDIAN   = "#08306b"
    COLOR_ACTUAL   = "#c0392b"

    fig, ax = plt.subplots(figsize=(12.5, 5.5))

    # Outer PI (10–90 %)
    ax.fill_between(df.index, df[q_outer_low], df[q_outer_high],
                    color=COLOR_PI_OUTER, alpha=1.0,
                    label=r"80% PI (10%–90%)")

    # Inner PI (25–75 %)
    if q_inner_low and q_inner_high:
        ax.fill_between(df.index, df[q_inner_low], df[q_inner_high],
                        color=COLOR_PI_INNER, alpha=0.75,
                        label=r"50% PI (25%–75%)")

    # Median
    ax.plot(df.index, df[q_median],
            color=COLOR_MEDIAN, linewidth=1.8,
            label="Median forecast", zorder=3)

    # Realised price
    if y_true_col is not None and y_true_col in df.columns:
        ax.plot(df.index, df[y_true_col],
                color=COLOR_ACTUAL, linewidth=1.4,
                linestyle="-", label="Realized price", zorder=4)

    # Day separator lines
    days = pd.date_range(
        start=df.index[0].normalize() + pd.Timedelta(days=1),
        end=df.index[-1].normalize(), freq="D",
        tz=df.index.tz,
    )
    for day in days:
        ax.axvline(day, color="#888888", linewidth=0.6, linestyle="--", zorder=1)

    # ── Axis ───────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.tick_params(axis="x", which="major", labelsize=13, pad=4)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", labelsize=13)

    ax.set_xlabel("Date", fontsize=15, labelpad=6)
    ax.set_ylabel("Electricity Price (\u20ac/MWh)", fontsize=15, labelpad=6)
    ax.grid(True, which="major", axis="y")
    ax.set_xlim(df.index[0], df.index[-1])

    y_max = df[[q_outer_low, q_outer_high]].max().max()
    y_min = df[[q_outer_low, q_outer_high]].min().min()
    if y_true_col is not None and y_true_col in df.columns:
        y_max = max(y_max, df[y_true_col].max())
        y_min = min(y_min, df[y_true_col].min())
    y_range = y_max - y_min
    ax.set_ylim(bottom=y_min - 0.15 * y_range, top=y_max + 0.35 * y_range)

    ax.legend(loc="upper left", bbox_to_anchor=(0.08, 0.96),
              ncol=4, fontsize=13, framealpha=0.9,
              edgecolor="#cccccc", frameon=True)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    QUANTILE_COLS = ["q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9"]

    df = load_forecast_csv(FORECAST_PATH)

    plot_prob_forecast_paper(
        df_forecast   = df,
        quantile_cols = QUANTILE_COLS,
        start_date    = START_DATE,
        end_date      = END_DATE,
        y_true_col    = Y_TRUE_COL,
        save_path     = EXPORT_PATH,
        dpi           = 300,
    )
