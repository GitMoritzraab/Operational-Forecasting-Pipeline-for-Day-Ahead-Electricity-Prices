#!/usr/bin/env python3
"""Generate independent persistence point and SQRA benchmark forecasts.

This runner is deliberately separate from ``run_full_experiment.py`` and the
main evaluation registry.  It writes four self-contained result directories:

* ``benchmarks/persistence_d1``
* ``benchmarks/persistence_d7``
* ``benchmarks/sqra_persistence_d1``
* ``benchmarks/sqra_persistence_d7``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


# Permit both ``python -m evaluation.run_benchmarks`` and direct execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from experiment_config import ExperimentConfig, load_experiment_config
from pipeline.delivery_index import (
    make_delivery_index,
    schema_metadata,
    validate_delivery_index,
)
from pipeline.lear.lear_model import (
    build_daily_mtu_matrix,
    save_experiment_outputs,
)
from pipeline.sqra.sqra_model import (
    DEFAULT_QUANTILES,
    evaluate_probabilistic_forecasts,
    generate_sqra_forecast,
    make_forecast_days,
)


POINT_BENCHMARKS = {
    "persistence_d1": 1,
    "persistence_d7": 7,
}


def load_epex_cache(config: ExperimentConfig) -> tuple[pd.DataFrame, Path]:
    """Load the cached DE-LU day-ahead price series without downloading data."""
    path = config.entsoe_price_cache_dir / "prices_da.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"EPEX price cache not found: {path}. Populate it before running benchmarks."
        )

    frame = pd.read_csv(path, index_col=0)
    if "price_da" not in frame.columns:
        raise ValueError(f"EPEX price cache lacks required column 'price_da': {path}")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(config.timezone)
    frame.index.name = "timestamp"
    if frame.index.has_duplicates:
        raise ValueError(f"EPEX price cache contains duplicate timestamps: {path}")
    frame["price_da"] = pd.to_numeric(frame["price_da"], errors="coerce")
    return frame.sort_index(), path


def _canonical_daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the manuscript-normalized 96-MTU EPEX matrix by local date."""
    daily = build_daily_mtu_matrix(prices["price_da"], "price_da")
    dates = pd.DatetimeIndex(daily.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    daily.index = dates.normalize()
    daily.index.name = "delivery_date"
    return daily.sort_index()


def build_persistence_forecast(
    prices: pd.DataFrame,
    forecast_days: Iterable[object],
    lag_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast each MTU with the realized price at the same MTU d-k days ago."""
    if lag_days <= 0:
        raise ValueError("lag_days must be positive.")

    days = pd.DatetimeIndex(pd.to_datetime(list(forecast_days))).tz_localize(None)
    days = days.normalize().sort_values()
    if days.empty:
        raise ValueError("No persistence forecast days were requested.")
    if days.has_duplicates:
        raise ValueError("Persistence forecast days must be unique.")

    daily = _canonical_daily_prices(prices)
    forecast_parts: list[pd.DataFrame] = []
    runtime_records: list[dict] = []
    for day in days:
        started = time.perf_counter()
        source_day = day - pd.DateOffset(days=lag_days)
        y_true = daily.reindex([day]).to_numpy(dtype=float).reshape(-1)
        y_pred = daily.reindex([source_day]).to_numpy(dtype=float).reshape(-1)
        day_frame = pd.DataFrame(
            {"y_pred": y_pred, "y_true": y_true},
            index=make_delivery_index([day]),
        )
        forecast_parts.append(day_frame)
        runtime_records.append(
            {
                "forecast_day": day.date(),
                "runtime_seconds": time.perf_counter() - started,
            }
        )

    forecast = pd.concat(forecast_parts).sort_index()
    validate_delivery_index(forecast.index, require_complete_days=True)
    missing_targets = int(forecast["y_true"].isna().sum())
    if missing_targets:
        raise ValueError(
            f"EPEX cache has {missing_targets} missing target MTUs in the requested "
            "persistence forecast period."
        )
    return forecast, pd.DataFrame(runtime_records)


def _point_days(config: ExperimentConfig) -> pd.DatetimeIndex:
    return make_forecast_days(
        config.point_forecast_start,
        config.evaluation_end,
        config.timezone,
        config.forecast_skip_dates,
    )


def _save_probabilistic_outputs(
    *,
    forecast: pd.DataFrame,
    runtime: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    quantiles = list(DEFAULT_QUANTILES)
    metrics = evaluate_probabilistic_forecasts(
        forecast,
        start_date=str(config.evaluation_start),
        end_date=str(config.evaluation_end),
        quantiles=quantiles,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_delivery_index(forecast.index, require_complete_days=True)
    forecast.to_csv(output_dir / "forecast.csv", index=True)
    runtime.to_csv(output_dir / "runtime.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved: forecast | runtime | metrics | config  ->  {output_dir}")


def run_benchmarks(
    experiment: Optional[ExperimentConfig] = None,
) -> dict[str, Path]:
    """Generate and save both point and both probabilistic benchmarks."""
    config = experiment or load_experiment_config(REPO_ROOT)
    prices, price_path = load_epex_cache(config)
    root = config.results_root / "benchmarks"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    point_days = _point_days(config)
    evaluation_days = make_forecast_days(
        config.evaluation_start,
        config.evaluation_end,
        config.timezone,
        config.evaluation_skip_dates,
    )
    quantiles = list(DEFAULT_QUANTILES)
    calibration_mode = (
        "mtu_specific" if config.sqra_mtu_specific else "pooled"
    )

    print(
        f"Persistence point period: {point_days.min().date()} -> "
        f"{point_days.max().date()} ({len(point_days)} days)"
    )
    print(
        f"SQRA evaluation period: {evaluation_days.min().date()} -> "
        f"{evaluation_days.max().date()} ({len(evaluation_days)} days)"
    )
    print(f"SQRA calibration: {calibration_mode}, {config.sqra_train_days} days")

    outputs: dict[str, Path] = {}
    for name, lag_days in POINT_BENCHMARKS.items():
        print(f"\nPoint benchmark: {name}")
        point_forecast, point_runtime = build_persistence_forecast(
            prices,
            point_days,
            lag_days,
        )
        point_dir = root / name
        point_metadata = {
            "experiment_name": name,
            "benchmark_type": "persistence_point_forecast",
            "lag_days": lag_days,
            "source": str(price_path),
            "test_start": str(config.point_forecast_start),
            "test_end": str(config.evaluation_end),
            "created_at": created_at,
            **schema_metadata(),
        }
        save_experiment_outputs(
            name=name,
            forecast_df=point_forecast,
            runtime_df=point_runtime,
            config=point_metadata,
            export_dir=point_dir,
            evaluation_start=config.evaluation_start,
            evaluation_end=config.evaluation_end,
            evaluation_skip_dates=config.evaluation_skip_dates,
        )
        outputs[name] = point_dir

        sqra_name = f"sqra_{name}"
        print(f"Probabilistic benchmark: {sqra_name}")
        panel = point_forecast.rename(columns={"y_pred": "prediction_p1"})
        sqra_forecast, sqra_runtime = generate_sqra_forecast(
            df=panel,
            forecast_days=evaluation_days,
            train_days=config.sqra_train_days,
            quantiles=quantiles,
            feature_cols=["prediction_p1"],
            mtu_specific=config.sqra_mtu_specific,
        )
        sqra_dir = root / sqra_name
        sqra_metadata = {
            "experiment_name": (
                f"{sqra_name}_d{config.sqra_train_days}_{calibration_mode}"
            ),
            "benchmark_type": "sqra_persistence",
            "lag_days": lag_days,
            "point_benchmark": name,
            "train_days": config.sqra_train_days,
            "mtu_specific": config.sqra_mtu_specific,
            "calibration_mode": calibration_mode,
            "quantiles": quantiles,
            "test_start": str(config.evaluation_start),
            "test_end": str(config.evaluation_end),
            "created_at": created_at,
            **schema_metadata(),
        }
        _save_probabilistic_outputs(
            forecast=sqra_forecast,
            runtime=sqra_runtime,
            metadata=sqra_metadata,
            output_dir=sqra_dir,
            config=config,
        )
        outputs[sqra_name] = sqra_dir

    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    run_benchmarks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
