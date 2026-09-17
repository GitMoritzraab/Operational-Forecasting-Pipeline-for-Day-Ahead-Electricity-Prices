#!/usr/bin/env python3
"""Run LEAR point forecasts, the EXAA baseline, or ANC as standard Python."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Support both ``python -m pipeline.lear.run_lear`` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)
from experiment_manifest import LearRun, get_lear_run, lear_runs
from market_cache import load_or_fetch_frame
from pipeline.delivery_index import schema_metadata, validate_delivery_index
from pipeline.lear.lear_model import (
    ICON_REQUIRED_RUN,
    build_dwd_features,
    build_era5_features,
    build_exaa_naive_forecast,
    build_load_features,
    build_price_features,
    build_temporal_features,
    build_y_matrix,
    compute_metrics,
    fetch_load_forecast,
    fetch_prices,
    fetch_prices_exaa,
    load_dwd,
    load_era5,
    merge_all_features,
    rolling_anc_feature_importance,
    rolling_point_forecast,
    save_experiment_outputs,
    summarize_anc,
)


ANC_WEATHER_SOURCE = "ERA5"
ANC_TRAIN_DAYS = 112
ANC_CLUSTERS = 5
ANC_LARS_START_DATE = "2025-12-01"


def _api_key() -> str:
    value = os.getenv("ENTSOE_API_KEY", "").strip()
    if not value or value == "your_api_key_here":
        raise ValueError("Environment variable 'ENTSOE_API_KEY' is not set.")
    return value


def _input_bounds(config: ExperimentConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(config.input_start, tz=config.timezone),
        pd.Timestamp(config.evaluation_end, tz=config.timezone),
    )


def load_market_inputs(
    config: ExperimentConfig,
    *,
    include_load: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load or download the shared SDAC, EXAA, and optional load inputs."""
    start, end = _input_bounds(config)
    api_key = _api_key()
    price_fetcher = partial(
        fetch_prices,
        api_key=api_key,
        target_tz=config.timezone,
    )
    exaa_fetcher = partial(
        fetch_prices_exaa,
        api_key=api_key,
        target_tz=config.timezone,
    )
    load_fetcher = partial(
        fetch_load_forecast,
        api_key=api_key,
        target_tz=config.timezone,
    )

    prices = load_or_fetch_frame(
        config.entsoe_price_cache_dir / "prices_da.csv",
        start,
        end,
        price_fetcher,
        timezone=config.timezone,
        refresh=config.refresh_market_cache,
    )
    exaa = load_or_fetch_frame(
        config.exaa_price_cache_dir / "prices_exaa.csv",
        start,
        end,
        exaa_fetcher,
        timezone=config.timezone,
        refresh=config.refresh_market_cache,
    )
    load = None
    if include_load:
        load = load_or_fetch_frame(
            config.entsoe_load_cache_dir / "load_forecast.csv",
            start,
            end,
            load_fetcher,
            timezone=config.timezone,
            refresh=config.refresh_market_cache,
        )
    return prices, exaa, load


def _era5_dirs(config: ExperimentConfig, clusters: int) -> List[Path]:
    """Return every ERA5 year directory required by the configured dates."""
    return [
        config.era5_year_dir(clusters, year)
        for year in range(
            config.weather_training_start.year,
            config.evaluation_end.year + 1,
        )
    ]


def _weather_features(
    config: ExperimentConfig,
    weather_source: str,
    clusters: int,
) -> pd.DataFrame:
    source = weather_source.upper()
    if source == "ERA5":
        era5 = load_era5(
            dirs=_era5_dirs(config, clusters),
            target_tz=config.timezone,
        )
        return build_era5_features(era5, target_tz=config.timezone)
    if source == "DWD":
        hourly, quarter_hourly = load_dwd(
            icon_dir=config.icon_cluster_dir(clusters),
            start_folder_date=config.dwd_preprocess_start_date,
            required_run=ICON_REQUIRED_RUN,
            skip_dates=set(config.dwd_folder_skip_dates),
            target_tz=config.timezone,
        )
        return build_dwd_features(
            hourly,
            quarter_hourly,
            target_tz=config.timezone,
        )
    raise ValueError(
        f"Unknown weather source: {weather_source}. Choose 'ERA5' or 'DWD'."
    )


def _assemble_operational_matrices(
    config: ExperimentConfig,
    run: LearRun,
    prices: pd.DataFrame,
    exaa: pd.DataFrame,
    load: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned daily X/Y matrices for one manifest run."""
    # EXAA-only intentionally bypasses every weather and load operation.
    if run.use_exaa_only:
        features = build_price_features(
            df_prices=prices,
            exaa_only=True,
            df_prices_exaa_15=exaa,
        )
        dropped = pd.DataFrame()
    else:
        if run.weather_source is None or run.clusters is None:
            raise ValueError(
                f"LEAR run {run.name} requires weather_source and clusters."
            )
        if load is None:
            raise ValueError(f"LEAR run {run.name} requires load forecast data.")
        daily_index = prices.index.normalize().unique().sort_values()
        weather = _weather_features(
            config,
            run.weather_source,
            run.clusters,
        )
        features, dropped = merge_all_features(
            df_weather_features=weather,
            df_price_features=build_price_features(
                df_prices=prices,
                exaa_vector=run.use_exaa,
                df_prices_exaa_15=exaa if run.use_exaa else None,
            ),
            df_load_features=build_load_features(load),
            df_time_features=build_temporal_features(daily_index),
        )

    target = build_y_matrix(prices, features.index)
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]
    return features, target, dropped


def _point_forecast_days(config: ExperimentConfig) -> pd.DatetimeIndex:
    days = pd.date_range(
        start=config.point_forecast_start,
        end=config.evaluation_end,
        freq="D",
        tz=config.timezone,
    )
    return pd.DatetimeIndex(
        [day for day in days if day.date() not in set(config.forecast_skip_dates)]
    )


def operational_experiment_name(run: LearRun) -> str:
    """Return the experiment-name layout written by the legacy notebook."""
    if run.use_exaa_only:
        return f"lear_exaa_only_d{run.train_days}"
    variant = "exaa" if run.use_exaa else "fundamental"
    return (
        f"lear_{str(run.weather_source).lower()}_{variant}"
        f"_c{run.clusters}_d{run.train_days}"
    )


def run_exaa_naive(config: ExperimentConfig) -> Path:
    """Create the deterministic EXAA-naive point-forecast baseline."""
    prices, exaa, _ = load_market_inputs(config, include_load=False)
    start = pd.Timestamp(config.point_forecast_start, tz=config.timezone)
    end = pd.Timestamp(config.evaluation_end, tz=config.timezone)
    forecast = build_exaa_naive_forecast(
        prices,
        exaa,
        start,
        end,
        config.forecast_skip_dates,
    )
    export_dir = config.results_root / "lear_op_results" / "exaa_naive"
    export_dir.mkdir(parents=True, exist_ok=True)
    validate_delivery_index(forecast.index, require_complete_days=True)
    forecast.to_csv(export_dir / "forecast.csv", index=True)
    baseline_config = {
        "experiment_name": "exaa_naive",
        "test_start": str(config.point_forecast_start),
        "test_end": str(config.evaluation_end),
        **schema_metadata(),
    }
    with (export_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_config, handle, indent=2)
    print(f"Saved: EXAA-naive forecast | config -> {export_dir}")
    return export_dir


def run_operational(config: ExperimentConfig, run: LearRun) -> Path:
    """Execute and save one canonical operational LEAR configuration."""
    prices, exaa, load = load_market_inputs(
        config,
        include_load=not run.use_exaa_only,
    )
    features, target, dropped = _assemble_operational_matrices(
        config,
        run,
        prices,
        exaa,
        load,
    )
    print(f"Number of dropped rows (days): {len(dropped)}")
    print(f"X shape    : {features.shape}")
    print(f"Y shape    : {target.shape}")
    if len(features):
        print(
            f"Date range : {features.index.min().date()} -> "
            f"{features.index.max().date()}"
        )

    use_vst = config.lear_use_vst
    lars_start = pd.Timestamp(run.lars_start_date, tz=config.timezone)
    forecast, runtime, _, _, _ = rolling_point_forecast(
        X=features,
        Y=target,
        forecast_days=_point_forecast_days(config),
        train_days=run.train_days,
        lars_start_date=lars_start,
        use_vst=use_vst,
    )
    metrics = compute_metrics(forecast, "full")
    invalid = int((~np.isfinite(forecast["y_pred"])).sum())
    print(f"{'Observations':<14}: {len(forecast)}")
    print(f"{'Invalid preds':<14}: {invalid}")
    print(f"{'MAE':<14}: {metrics['mae']:.3f}")
    print(f"{'RMSE':<14}: {metrics['rmse']:.3f}")
    print(f"{'Bias':<14}: {metrics['bias']:.3f}")

    experiment_name = operational_experiment_name(run)
    export_dir = run.output_dir(config.results_root)
    save_experiment_outputs(
        name=experiment_name,
        forecast_df=forecast,
        runtime_df=runtime,
        config={
            "experiment_name": experiment_name,
            "use_exaa": run.use_exaa,
            "use_exaa_only": run.use_exaa_only,
            "n_clusters": run.clusters,
            "weather_source": run.weather_source,
            "train_days": run.train_days,
            "use_vst": use_vst,
            "lars_start_date": str(lars_start.date()),
            "test_start": str(config.point_forecast_start),
            "test_end": str(config.evaluation_end),
            **schema_metadata(),
        },
        export_dir=export_dir,
        evaluation_start=getattr(config, "evaluation_start", None),
        evaluation_end=getattr(config, "evaluation_end", None),
        evaluation_skip_dates=getattr(config, "evaluation_skip_dates", ()),
    )
    return export_dir


def _evaluation_days(config: ExperimentConfig) -> pd.DatetimeIndex:
    days = pd.date_range(
        start=config.evaluation_start,
        end=config.evaluation_end,
        freq="D",
        tz=config.timezone,
    )
    return pd.DatetimeIndex(
        [day for day in days if day.date() not in set(config.evaluation_skip_dates)]
    )


def run_anc(config: ExperimentConfig, variant: str) -> Path:
    """Execute the fixed ERA5 C=5, D=112 ANC manuscript configuration."""
    if variant not in {"fundamental", "exaa"}:
        raise ValueError("ANC variant must be 'fundamental' or 'exaa'.")
    use_exaa = variant == "exaa"
    prices, exaa, load = load_market_inputs(config, include_load=True)
    assert load is not None
    weather = _weather_features(config, ANC_WEATHER_SOURCE, ANC_CLUSTERS)
    daily_index = prices.index.normalize().unique().sort_values()
    features, dropped = merge_all_features(
        df_weather_features=weather,
        df_price_features=build_price_features(
            df_prices=prices,
            exaa_vector=use_exaa,
            df_prices_exaa_15=exaa if use_exaa else None,
        ),
        df_load_features=build_load_features(load),
        df_time_features=build_temporal_features(daily_index),
    )
    target = build_y_matrix(prices, features.index)
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]
    print(f"Number of dropped rows (days): {len(dropped)}")
    print(f"X shape: {features.shape}")
    print(f"Y shape: {target.shape}")

    lars_start = pd.Timestamp(ANC_LARS_START_DATE, tz=config.timezone)
    anc = rolling_anc_feature_importance(
        X=features,
        Y=target,
        forecast_days=_evaluation_days(config),
        train_days=ANC_TRAIN_DAYS,
        lars_start_date=lars_start,
    )
    print(f"ANC records : {len(anc)}")
    overall, wind, solar = summarize_anc(anc)

    export_dir = (
        config.results_root
        / "lear_anc_results"
        / "era5"
        / f"c{ANC_CLUSTERS}"
        / f"d{ANC_TRAIN_DAYS}"
        / variant
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(export_dir / "anc_all_feature_results.csv", index=True)
    wind.to_csv(export_dir / "anc_wind_feature_results.csv", index=False)
    solar.to_csv(export_dir / "anc_solar_feature_results.csv", index=False)

    experiment_name = (
        f"anc_era5_{'exaa' if use_exaa else 'fundamental'}"
        f"_c{ANC_CLUSTERS}_d{ANC_TRAIN_DAYS}"
    )
    anc_config = {
        "experiment_name": experiment_name,
        "use_exaa": use_exaa,
        "use_exaa_only": False,
        "n_clusters": ANC_CLUSTERS,
        "weather_source": ANC_WEATHER_SOURCE,
        "train_days": ANC_TRAIN_DAYS,
        "lars_start_date": str(lars_start.date()),
        "test_start": str(config.evaluation_start),
        "test_end": str(config.evaluation_end),
        **schema_metadata(),
    }
    with (export_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(anc_config, handle, indent=2)
    print(f"Saved: anc_features | anc_wind | anc_solar | config -> {export_dir}")
    return export_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "exaa-naive",
        help="Generate the deterministic EXAA-naive point baseline.",
    )

    operational = subparsers.add_parser(
        "operational",
        help="Run one manifest-defined operational LEAR model.",
    )
    operational.add_argument(
        "--config",
        required=True,
        choices=[run.name for run in lear_runs()],
        help="Canonical LEAR configuration name.",
    )

    anc = subparsers.add_parser(
        "anc",
        help="Run one ANC feature-importance variant.",
    )
    anc.add_argument(
        "--variant",
        required=True,
        choices=("fundamental", "exaa"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(REPO_ROOT)
    if args.command == "exaa-naive":
        run_exaa_naive(config)
    elif args.command == "operational":
        run_operational(config, get_lear_run(args.config))
    elif args.command == "anc":
        run_anc(config, args.variant)
    else:  # pragma: no cover - argparse enforces the command choices.
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
