#!/usr/bin/env python3
"""Plot monthly point-forecast MAE and negative EPEX-price counts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Mapping, Sequence, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402
from experiment_manifest import get_lear_run  # noqa: E402
from pipeline.delivery_index import (  # noqa: E402
    delivery_dates,
    read_forecast_csv,
    validate_delivery_index,
)


ANALYSIS_START = date(2025, 12, 1)
ANALYSIS_END = date(2026, 7, 31)
PDF_FILENAME = "monthly_point_forecast_mae.pdf"
CSV_FILENAME = "monthly_point_forecast_mae.csv"
NEGATIVE_PRICE_PDF_FILENAME = "monthly_epex_negative_price_mtus.pdf"
NEGATIVE_PRICE_CSV_FILENAME = "monthly_epex_negative_price_mtus.csv"


@dataclass(frozen=True)
class ModelSpecification:
    """One point-forecast series included in the monthly comparison."""

    run_name: str
    label: str
    color: str
    marker: str


MODEL_SPECIFICATIONS: Tuple[ModelSpecification, ...] = (
    ModelSpecification(
        "dwd_d56_c5_fundamental",
        r"Fundamental, ICON-D2 ($C=5$, $D_{\mathrm{LEAR}}=56$)",
        "#1a5c8a",
        "o",
    ),
    ModelSpecification(
        "era5_d364_c5_fundamental",
        r"Fundamental, ERA5 ($C=5$, $D_{\mathrm{LEAR}}=364$)",
        "#e07a1f",
        "s",
    ),
    ModelSpecification(
        "era5_d364_c1_exaa",
        r"EXAA-Enriched, ERA5 ($C=1$, $D_{\mathrm{LEAR}}=364$)",
        "#c0392b",
        "D",
    ),
    ModelSpecification(
        "exaa_only_d364",
        r"EXAA-Only ($D_{\mathrm{LEAR}}=364$)",
        "#2a8c82",
        "^",
    ),
)


def representative_forecast_files(results_root: Path) -> Mapping[str, Path]:
    """Return the four canonical point-forecast paths in display order."""
    return {
        specification.run_name: (
            get_lear_run(specification.run_name).output_dir(results_root)
            / "forecast.csv"
        )
        for specification in MODEL_SPECIFICATIONS
    }


def expected_days(
    start: date,
    end: date,
    skip_dates: Sequence[date],
) -> Tuple[pd.Timestamp, ...]:
    """Return the complete, inclusive evaluation-day sequence after skips."""
    skipped = {pd.Timestamp(day).normalize() for day in skip_dates}
    return tuple(
        day
        for day in pd.date_range(start=start, end=end, freq="D")
        if day not in skipped
    )


def load_evaluation_forecast(
    path: Path,
    start: date,
    end: date,
    skip_dates: Sequence[date],
) -> pd.DataFrame:
    """Load one point forecast and require the complete common evaluation grid."""
    frame = read_forecast_csv(
        path,
        required_columns=("y_pred", "y_true"),
        require_complete_days=True,
    )
    start_timestamp = pd.Timestamp(start).normalize()
    end_timestamp = pd.Timestamp(end).normalize()
    skipped = {pd.Timestamp(day).normalize() for day in skip_dates}
    dates = delivery_dates(frame.index)
    mask = (dates >= start_timestamp) & (dates <= end_timestamp)
    if skipped:
        mask &= ~dates.isin(skipped)
    sample = frame.loc[mask, ["y_pred", "y_true"]].copy()
    evaluation_days = expected_days(start, end, skip_dates)
    validate_delivery_index(
        sample.index,
        require_complete_days=True,
        expected_days=evaluation_days,
    )
    numeric = sample[["y_pred", "y_true"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        bad = (~np.isfinite(numeric)).sum(axis=0)
        raise ValueError(
            f"Non-finite point-forecast values in {path}: "
            f"y_pred={int(bad[0])}, y_true={int(bad[1])}."
        )
    return sample


def load_epex_prices(
    path: Path,
    start: date,
    end: date,
    timezone: str,
) -> pd.DataFrame:
    """Load the complete physical EPEX quarter-hour grid for the period."""
    if not path.is_file():
        raise FileNotFoundError(f"EPEX price cache not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    if "price_da" not in frame.columns:
        raise ValueError(f"EPEX price cache lacks column 'price_da': {path}")

    try:
        timestamps = pd.to_datetime(frame.index, utc=True).tz_convert(timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse EPEX timestamps in {path}.") from exc
    if not timestamps.is_unique:
        raise ValueError(f"EPEX price cache contains duplicate timestamps: {path}")

    start_timestamp = pd.Timestamp(start, tz=timezone)
    end_exclusive = pd.Timestamp(end + timedelta(days=1), tz=timezone)
    expected_timestamps = pd.date_range(
        start=start_timestamp,
        end=end_exclusive - pd.Timedelta(minutes=15),
        freq="15min",
    )
    prices = pd.Series(
        pd.to_numeric(frame["price_da"], errors="coerce").to_numpy(dtype=float),
        index=timestamps,
        name="epex_price",
    ).sort_index()
    prices = prices.loc[
        (prices.index >= start_timestamp) & (prices.index < end_exclusive)
    ]

    missing = expected_timestamps.difference(prices.index)
    extra = prices.index.difference(expected_timestamps)
    if len(missing) or len(extra):
        raise ValueError(
            "EPEX timestamps do not cover the complete physical quarter-hour grid. "
            f"Missing={len(missing)}, Extra={len(extra)}."
        )
    prices = prices.reindex(expected_timestamps)
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError(
            f"EPEX price cache contains {int((~np.isfinite(prices)).sum())} "
            f"non-finite prices in the requested period: {path}"
        )

    return pd.DataFrame(
        {
            "delivery_timestamp": expected_timestamps,
            "delivery_date": expected_timestamps.date,
            "epex_price": prices.to_numpy(dtype=float),
        }
    )


def calculate_monthly_negative_price_counts(
    epex_prices: pd.DataFrame,
    start: date = ANALYSIS_START,
    end: date = ANALYSIS_END,
) -> pd.DataFrame:
    """Count strictly negative EPEX prices for each calendar month."""
    required = {"delivery_date", "epex_price"}
    missing = required - set(epex_prices.columns)
    if missing:
        raise ValueError(f"EPEX price table is missing columns: {sorted(missing)}")

    sample = epex_prices.copy()
    sample["month"] = pd.to_datetime(sample["delivery_date"]).dt.to_period("M")
    sample["negative_price"] = sample["epex_price"] < 0.0
    expected_months = pd.period_range(start=start, end=end, freq="M")
    monthly = (
        sample.groupby("month", sort=True)
        .agg(
            negative_price_mtus=("negative_price", "sum"),
            total_mtus=("negative_price", "size"),
        )
        .reindex(expected_months)
    )
    if monthly.isna().any().any():
        missing_months = monthly.index[
            monthly["negative_price_mtus"].isna()
        ].astype(str).tolist()
        raise ValueError(f"Missing EPEX price months: {missing_months}")
    monthly["negative_price_mtus"] = monthly["negative_price_mtus"].astype(int)
    monthly["total_mtus"] = monthly["total_mtus"].astype(int)
    monthly["negative_price_share"] = (
        monthly["negative_price_mtus"] / monthly["total_mtus"]
    )
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    return monthly.reset_index()


def calculate_monthly_mae(
    forecasts: Mapping[str, pd.DataFrame],
    start: date = ANALYSIS_START,
    end: date = ANALYSIS_END,
) -> pd.DataFrame:
    """Calculate pooled quarter-hourly MAE for each model and calendar month."""
    expected_months = pd.period_range(start=start, end=end, freq="M")
    records = []
    reference = None
    for specification in MODEL_SPECIFICATIONS:
        if specification.run_name not in forecasts:
            raise ValueError(
                f"Missing representative forecast: {specification.run_name}"
            )
        forecast = forecasts[specification.run_name]
        if reference is None:
            reference = forecast[["y_true"]]
        else:
            if not forecast.index.equals(reference.index):
                raise ValueError(
                    f"Forecast index differs for {specification.run_name}."
                )
            if not np.allclose(
                forecast["y_true"].to_numpy(dtype=float),
                reference["y_true"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Realised prices differ for {specification.run_name}."
                )

        dates = delivery_dates(forecast.index)
        months = dates.to_period("M")
        absolute_error = (
            forecast["y_pred"].to_numpy(dtype=float)
            - forecast["y_true"].to_numpy(dtype=float)
        )
        data = pd.DataFrame(
            {
                "month": months,
                "delivery_date": dates,
                "absolute_error": np.abs(absolute_error),
            },
            index=forecast.index,
        )
        grouped = data.groupby("month", sort=True)
        monthly = grouped.agg(
            mae=("absolute_error", "mean"),
            observations=("absolute_error", "size"),
            delivery_days=("delivery_date", "nunique"),
        ).reindex(expected_months)
        if monthly.isna().any().any():
            missing_months = monthly.index[monthly["mae"].isna()].astype(str).tolist()
            raise ValueError(
                f"Missing monthly values for {specification.run_name}: "
                f"{missing_months}"
            )
        for month, row in monthly.iterrows():
            records.append(
                {
                    "month": str(month),
                    "model": specification.run_name,
                    "label": specification.label,
                    "mae": float(row["mae"]),
                    "observations": int(row["observations"]),
                    "delivery_days": int(row["delivery_days"]),
                }
            )
    return pd.DataFrame(records)


def plot_monthly_mae(monthly_mae: pd.DataFrame, output_path: Path) -> None:
    """Create and save the publication-quality monthly MAE line plot."""
    required = {"month", "model", "mae"}
    missing = required - set(monthly_mae.columns)
    if missing:
        raise ValueError(f"Monthly MAE table is missing columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
        }
    )

    months = pd.period_range(ANALYSIS_START, ANALYSIS_END, freq="M")
    x_positions = np.arange(len(months))
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    for specification in MODEL_SPECIFICATIONS:
        model_data = monthly_mae.loc[
            monthly_mae["model"] == specification.run_name
        ].set_index("month")
        values = model_data.reindex(months.astype(str))["mae"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite plotted MAE for {specification.run_name}."
            )
        axis.plot(
            x_positions,
            values,
            color=specification.color,
            marker=specification.marker,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=1.8,
            label=specification.label,
            zorder=3,
        )

    labels = [
        "Dec\n2025",
        "Jan\n2026",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
    ]
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("MAE (€/MWh)")
    axis.tick_params(axis="x", length=0)
    axis.yaxis.grid(True, zorder=0)
    axis.xaxis.grid(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.4,
        handlelength=2.4,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_monthly_negative_price_counts(
    monthly_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a publication-quality bar plot of negative EPEX-price MTUs."""
    required = {"month", "negative_price_mtus"}
    missing = required - set(monthly_counts.columns)
    if missing:
        raise ValueError(
            f"Monthly negative-price table is missing columns: {sorted(missing)}"
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
        }
    )

    months = pd.period_range(ANALYSIS_START, ANALYSIS_END, freq="M")
    values = (
        monthly_counts.set_index("month")
        .reindex(months.astype(str))["negative_price_mtus"]
        .to_numpy(dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite monthly negative-price counts cannot be plotted.")

    x_positions = np.arange(len(months))
    labels = ["Dec\n2025", "Jan\n2026", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(
        x_positions,
        values,
        width=0.62,
        color="#1a5c8a",
        zorder=3,
    )
    annotation_offset = max(float(values.max()) * 0.018, 0.5)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + annotation_offset,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Quarter-hour MTUs with negative EPEX prices")
    axis.tick_params(axis="x", length=0)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.grid(True, zorder=0)
    axis.xaxis.grid(False)
    axis.set_ylim(0, max(float(values.max()) + 4 * annotation_offset, 1.0))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_analysis(config: ExperimentConfig) -> pd.DataFrame:
    """Persist monthly point MAE and negative EPEX-price analyses."""
    paths = representative_forecast_files(config.results_root)
    forecasts = {
        name: load_evaluation_forecast(
            path,
            ANALYSIS_START,
            ANALYSIS_END,
            config.evaluation_skip_dates,
        )
        for name, path in paths.items()
    }
    monthly = calculate_monthly_mae(forecasts)
    epex_prices = load_epex_prices(
        config.entsoe_price_cache_dir / "prices_da.csv",
        ANALYSIS_START,
        ANALYSIS_END,
        config.timezone,
    )
    negative_prices = calculate_monthly_negative_price_counts(epex_prices)

    display = monthly.pivot(index="month", columns="label", values="mae")
    print("Monthly point-forecast MAE (€/MWh)")
    print(display.to_string(float_format=lambda value: f"{value:.3f}"))
    print()
    print("Monthly EPEX DE-LU quarter-hour MTUs with prices below zero")
    print(negative_prices.to_string(index=False))

    config.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_root / CSV_FILENAME
    pdf_path = config.output_root / PDF_FILENAME
    negative_csv_path = config.output_root / NEGATIVE_PRICE_CSV_FILENAME
    negative_pdf_path = config.output_root / NEGATIVE_PRICE_PDF_FILENAME
    monthly.to_csv(csv_path, index=False)
    plot_monthly_mae(monthly, pdf_path)
    negative_prices.to_csv(negative_csv_path, index=False)
    plot_monthly_negative_price_counts(negative_prices, negative_pdf_path)
    print(f"Saved: {csv_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {negative_csv_path}")
    print(f"Saved: {negative_pdf_path}")
    return monthly


def main() -> int:
    run_analysis(load_experiment_config(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
