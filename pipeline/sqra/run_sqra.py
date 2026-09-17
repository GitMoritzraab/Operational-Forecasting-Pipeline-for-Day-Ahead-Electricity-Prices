"""Command-line entry point for one configured SQRA experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from experiment_config import ExperimentConfig, load_experiment_config
from experiment_manifest import SQRA_RUNS, get_sqra_run
from pipeline.delivery_index import (
    delivery_dates,
    schema_metadata,
    validate_delivery_index,
)
from pipeline.sqra.sqra_model import (
    DEFAULT_QUANTILES,
    build_sqra_panel,
    evaluate_probabilistic_forecasts,
    generate_sqra_forecast,
    make_forecast_days,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SqraArtifacts:
    """In-memory artifacts produced by one SQRA execution."""

    forecast: pd.DataFrame
    runtime: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict
    output_dir: Path


def run_sqra_experiment(
    config_name: str,
    experiment: Optional[ExperimentConfig] = None,
) -> SqraArtifacts:
    """Run one manifest-defined SQRA configuration and persist its artifacts."""
    config = experiment or load_experiment_config(REPO_ROOT)
    specification = get_sqra_run(config_name)
    input_paths = list(specification.inputs(config.results_root))
    output_dir = specification.output_dir(config.results_root)
    quantiles = list(DEFAULT_QUANTILES)
    calibration_mode = "mtu_specific" if config.sqra_mtu_specific else "pooled"
    experiment_name = (
        f"sqra_{config_name}_d{config.sqra_train_days}_{calibration_mode}"
    )

    print(f"SQRA configuration: {config_name}")
    print(f"Calibration mode: {calibration_mode}")
    for path in input_paths:
        print(f"Input: {path}")
    print(f"Output: {output_dir}")

    panel, feature_cols = build_sqra_panel(input_paths, config.timezone)
    forecast_days = make_forecast_days(
        config.evaluation_start,
        config.evaluation_end,
        config.timezone,
        config.evaluation_skip_dates,
    )
    if forecast_days.empty:
        raise ValueError("No SQRA forecast days remain after applying skip dates.")

    panel_days = delivery_dates(panel.index)
    print(f"Panel shape: {panel.shape}")
    print(f"Panel dates: {panel_days.min().date()} -> {panel_days.max().date()}")
    print(f"Forecast days: {len(forecast_days)}")
    forecast, runtime = generate_sqra_forecast(
        df=panel,
        forecast_days=forecast_days,
        train_days=config.sqra_train_days,
        quantiles=quantiles,
        feature_cols=feature_cols,
        mtu_specific=config.sqra_mtu_specific,
    )
    validate_delivery_index(
        forecast.index,
        require_complete_days=True,
        expected_days=forecast_days,
    )
    metrics = evaluate_probabilistic_forecasts(
        df=forecast,
        start_date=str(config.evaluation_start),
        end_date=str(config.evaluation_end),
        quantiles=quantiles,
        y_true_col="y_true",
    )

    metadata = {
        "experiment_name": experiment_name,
        "train_days": config.sqra_train_days,
        "mtu_specific": config.sqra_mtu_specific,
        "calibration_mode": calibration_mode,
        "quantiles": quantiles,
        "test_start": str(config.evaluation_start),
        "test_end": str(config.evaluation_end),
        "import_paths": [str(path) for path in input_paths],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **schema_metadata(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "forecast.csv", index=True)
    runtime.to_csv(output_dir / "runtime.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved SQRA outputs: {output_dir}")
    return SqraArtifacts(forecast, runtime, metrics, metadata, output_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=tuple(SQRA_RUNS),
        help="Manifest SQRA configuration to execute.",
    )
    args = parser.parse_args(argv)
    run_sqra_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
