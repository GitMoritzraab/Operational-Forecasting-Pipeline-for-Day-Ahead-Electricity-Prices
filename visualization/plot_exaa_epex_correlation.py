#!/usr/bin/env python3
"""Analyse and plot EXAA/EPEX DE-LU day-ahead price correlations.

Prices are matched by local delivery date and 15-minute market time unit
(MTU) over the inclusive evaluation horizon configured in ``.env``.  The
analysis deliberately uses all price delivery days and does not apply model
evaluation skip dates, because those exclusions concern other pipeline
inputs.  The spring daylight-saving day is therefore retained with its 92
available quarter-hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Callable, Sequence, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402


EPEX_COLUMN = "price_da"
EXAA_COLUMN = "price_exaa"
PDF_FILENAME = "exaa_epex_price_correlations.pdf"
CSV_FILENAME = "exaa_epex_price_correlations.csv"

GROUPS: Tuple[Tuple[str, Callable[[pd.DataFrame], pd.Series]], ...] = (
    ("All delivery days", lambda panel: pd.Series(True, index=panel.index)),
    ("Tuesday–Saturday", lambda panel: panel["weekday"].isin((1, 2, 3, 4, 5))),
    ("Sunday", lambda panel: panel["weekday"] == 6),
    ("Monday", lambda panel: panel["weekday"] == 0),
    ("Negative EPEX prices", lambda panel: panel["epex_price"] < 0.0),
)

PLOT_LABELS = {
    "All delivery days": "All delivery\ndays",
    "Tuesday–Saturday": "Tuesday–Saturday",
    "Sunday": "Sunday",
    "Monday": "Monday",
    "Negative EPEX prices": "Negative EPEX\nprices",
}


@dataclass(frozen=True)
class MatchDiagnostics:
    """Coverage counts for the price-pair matching step."""

    expected_mtus: int
    epex_missing_mtus: int
    exaa_missing_mtus: int
    matched_mtus: int
    expected_delivery_days: int
    matched_delivery_days: int


def _local_bounds(
    start: date,
    end: date,
    timezone: str,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if end < start:
        raise ValueError("The evaluation end date must not precede its start date.")
    return (
        pd.Timestamp(start, tz=timezone),
        pd.Timestamp(end + timedelta(days=1), tz=timezone),
    )


def _expected_delivery_grid(
    start: date,
    end: date,
    timezone: str,
) -> pd.DataFrame:
    """Return every physically occurring local quarter-hour in the period."""
    start_timestamp, end_exclusive = _local_bounds(start, end, timezone)
    timestamps = pd.date_range(
        start=start_timestamp,
        end=end_exclusive - pd.Timedelta(minutes=15),
        freq="15min",
    )
    return pd.DataFrame(
        {
            "delivery_timestamp": timestamps,
            "delivery_date": timestamps.date,
            "mtu": timestamps.hour * 4 + timestamps.minute // 15 + 1,
        }
    )


def load_price_cache(
    path: Path,
    price_column: str,
    output_column: str,
    start: date,
    end: date,
    timezone: str,
) -> pd.DataFrame:
    """Load one cached price series and derive its local delivery key."""
    if not path.is_file():
        raise FileNotFoundError(f"Price cache not found: {path}")

    frame = pd.read_csv(path, index_col=0)
    if price_column not in frame.columns:
        raise ValueError(
            f"Required column {price_column!r} is missing from price cache: {path}"
        )
    try:
        timestamps = pd.to_datetime(frame.index, utc=True).tz_convert(timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse timestamp index in price cache: {path}") from exc
    if not timestamps.is_unique:
        raise ValueError(f"Duplicate timestamps in price cache: {path}")

    start_timestamp, end_exclusive = _local_bounds(start, end, timezone)
    selected = (timestamps >= start_timestamp) & (timestamps < end_exclusive)
    result = pd.DataFrame(
        {
            "source_timestamp": timestamps[selected],
            "delivery_date": timestamps[selected].date,
            "mtu": (
                timestamps[selected].hour * 4
                + timestamps[selected].minute // 15
                + 1
            ),
            output_column: pd.to_numeric(
                frame.loc[selected, price_column], errors="coerce"
            ).to_numpy(),
        }
    )
    if result.empty:
        raise ValueError(
            f"No {output_column} observations between {start} and {end}: {path}"
        )
    duplicate_keys = result.duplicated(["delivery_date", "mtu"], keep=False)
    if duplicate_keys.any():
        sample = result.loc[duplicate_keys, ["delivery_date", "mtu"]].head(4)
        raise ValueError(
            "Duplicate local delivery-date/MTU keys in "
            f"{path}: {sample.to_dict('records')}"
        )
    return result


def build_matched_price_panel(
    epex_path: Path,
    exaa_path: Path,
    start: date,
    end: date,
    timezone: str,
) -> Tuple[pd.DataFrame, MatchDiagnostics]:
    """Match EPEX and EXAA prices by local delivery date and 15-minute MTU."""
    expected = _expected_delivery_grid(start, end, timezone)
    epex = load_price_cache(
        epex_path,
        EPEX_COLUMN,
        "epex_price",
        start,
        end,
        timezone,
    ).rename(columns={"source_timestamp": "epex_timestamp"})
    exaa = load_price_cache(
        exaa_path,
        EXAA_COLUMN,
        "exaa_price",
        start,
        end,
        timezone,
    ).rename(columns={"source_timestamp": "exaa_timestamp"})

    keys = ["delivery_date", "mtu"]
    combined = expected.merge(epex, on=keys, how="left", validate="one_to_one")
    combined = combined.merge(exaa, on=keys, how="left", validate="one_to_one")

    for source in ("epex", "exaa"):
        source_timestamp = combined[f"{source}_timestamp"]
        present = source_timestamp.notna()
        mismatched = present & (
            source_timestamp != combined["delivery_timestamp"]
        )
        if mismatched.any():
            sample = combined.loc[
                mismatched,
                ["delivery_date", "mtu", "delivery_timestamp", f"{source}_timestamp"],
            ].head(4)
            raise ValueError(
                f"{source.upper()} timestamps disagree with their local delivery keys: "
                f"{sample.to_dict('records')}"
            )

    epex_missing = combined["epex_price"].isna()
    exaa_missing = combined["exaa_price"].isna()
    matched = combined.loc[~epex_missing & ~exaa_missing].copy()
    if matched.empty:
        raise ValueError("No matched, finite EXAA/EPEX price pairs were found.")
    matched["weekday"] = pd.to_datetime(matched["delivery_date"]).dt.weekday
    matched = matched[
        [
            "delivery_timestamp",
            "delivery_date",
            "mtu",
            "weekday",
            "epex_price",
            "exaa_price",
        ]
    ].sort_values(["delivery_date", "mtu"])
    matched.reset_index(drop=True, inplace=True)

    expected_days = int(expected["delivery_date"].nunique())
    matched_days = int(matched["delivery_date"].nunique())
    if matched_days != expected_days:
        available = set(matched["delivery_date"])
        missing_days = sorted(set(expected["delivery_date"]) - available)
        preview = ", ".join(map(str, missing_days[:8]))
        raise ValueError(
            "At least one delivery day has no matched EXAA/EPEX observation. "
            f"Missing days: {preview}"
        )

    diagnostics = MatchDiagnostics(
        expected_mtus=len(expected),
        epex_missing_mtus=int(epex_missing.sum()),
        exaa_missing_mtus=int(exaa_missing.sum()),
        matched_mtus=len(matched),
        expected_delivery_days=expected_days,
        matched_delivery_days=matched_days,
    )
    return matched, diagnostics


def calculate_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate pooled Pearson and Spearman correlations for five groups."""
    required = ["delivery_date", "weekday", "epex_price", "exaa_price"]
    missing = set(required) - set(panel.columns)
    if missing:
        raise ValueError(f"Matched price panel is missing columns: {sorted(missing)}")

    records = []
    for group_name, selector in GROUPS:
        sample = panel.loc[selector(panel), required].copy()
        sample = sample.dropna(subset=["epex_price", "exaa_price"])
        if len(sample) < 2:
            raise ValueError(f"Fewer than two matched observations for {group_name}.")
        if sample["epex_price"].nunique() < 2 or sample["exaa_price"].nunique() < 2:
            raise ValueError(f"A price series is constant for {group_name}.")

        pearson = sample["epex_price"].corr(sample["exaa_price"])
        spearman = sample["epex_price"].rank(method="average").corr(
            sample["exaa_price"].rank(method="average")
        )
        if not np.isfinite(pearson) or not np.isfinite(spearman):
            raise ValueError(f"Non-finite correlation calculated for {group_name}.")
        records.append(
            {
                "group": group_name,
                "delivery_days": int(sample["delivery_date"].nunique()),
                "matched_mtus": len(sample),
                "pearson": float(pearson),
                "spearman": float(spearman),
            }
        )
    return pd.DataFrame(records)


def plot_correlations(
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the publication-quality grouped Pearson/Spearman bar plot."""
    required = {"group", "pearson", "spearman"}
    missing = required - set(correlations.columns)
    if missing:
        raise ValueError(f"Correlation table is missing columns: {sorted(missing)}")
    groups = correlations["group"].tolist()
    if len(groups) != len(set(groups)):
        raise ValueError("Correlation table contains duplicate groups.")
    unknown_groups = set(groups) - set(PLOT_LABELS)
    if unknown_groups:
        raise ValueError(f"Unknown correlation groups: {sorted(unknown_groups)}")

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

    x_positions = np.arange(len(correlations))
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    pearson_bars = axis.bar(
        x_positions - bar_width / 2,
        correlations["pearson"],
        width=bar_width,
        color="#1a5c8a",
        label="Pearson",
        zorder=3,
    )
    spearman_bars = axis.bar(
        x_positions + bar_width / 2,
        correlations["spearman"],
        width=bar_width,
        color="#c0392b",
        label="Spearman",
        zorder=3,
    )

    values = correlations[["pearson", "spearman"]].to_numpy(dtype=float)
    has_negative = bool((values < 0).any())
    axis.set_ylim((-1.05, 1.05) if has_negative else (0.0, 1.05))
    annotation_offset = 0.025 if has_negative else 0.012
    for bars in (pearson_bars, spearman_bars):
        for bar in bars:
            value = float(bar.get_height())
            vertical_alignment = "bottom" if value >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (annotation_offset if value >= 0 else -annotation_offset),
                f"{value:.3f}",
                ha="center",
                va=vertical_alignment,
                fontsize=9,
                color="#333333",
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([PLOT_LABELS[group] for group in groups], fontsize=10)
    axis.set_ylabel("Correlation coefficient", fontsize=11)
    axis.tick_params(axis="x", length=0)
    axis.tick_params(axis="y", labelsize=10)
    axis.yaxis.grid(True, zorder=0)
    axis.xaxis.grid(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_analysis(config: ExperimentConfig) -> pd.DataFrame:
    """Run the configured correlation analysis and persist its table and plot."""
    epex_path = config.entsoe_price_cache_dir / "prices_da.csv"
    exaa_path = config.exaa_price_cache_dir / "prices_exaa.csv"
    panel, diagnostics = build_matched_price_panel(
        epex_path,
        exaa_path,
        config.evaluation_start,
        config.evaluation_end,
        config.timezone,
    )
    correlations = calculate_correlations(panel)

    print(
        "EXAA/EPEX DE-LU descriptive price correlation analysis\n"
        f"Delivery period: {config.evaluation_start} through {config.evaluation_end}\n"
        f"Expected delivery days: {diagnostics.expected_delivery_days}\n"
        f"Matched delivery days: {diagnostics.matched_delivery_days}\n"
        f"Expected MTUs: {diagnostics.expected_mtus}\n"
        f"EPEX missing MTUs: {diagnostics.epex_missing_mtus}\n"
        f"EXAA missing MTUs: {diagnostics.exaa_missing_mtus}\n"
        f"Matched MTUs: {diagnostics.matched_mtus}"
    )
    print()
    print(
        correlations.to_string(
            index=False,
            formatters={
                "pearson": lambda value: f"{value:.4f}",
                "spearman": lambda value: f"{value:.4f}",
            },
        )
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_root / CSV_FILENAME
    pdf_path = config.output_root / PDF_FILENAME
    correlations.to_csv(csv_path, index=False)
    plot_correlations(
        correlations,
        pdf_path,
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {pdf_path}")
    return correlations


def main() -> int:
    run_analysis(load_experiment_config(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
