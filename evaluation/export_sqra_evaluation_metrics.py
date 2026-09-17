#!/usr/bin/env python3
"""Export evaluation-period metrics for all six SQRA configurations to Excel."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluation_core import (  # noqa: E402
    FREQ,
    N_PERIODS,
    QUANTILES,
    QUANTILE_MODELS,
    Y_TRUE_COL,
    aggregated_pinball_score,
    check_index,
    check_missing,
    check_numeric,
    enforce_eval_window,
    quantile_model_files,
    read_forecast_csv,
)
from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402
from pipeline.delivery_index import delivery_dates  # noqa: E402


DEFAULT_FILENAME = "sqra_evaluation_metrics.xlsx"
QUANTILE_COLUMNS = tuple(f"q{quantile:.3f}" for quantile in QUANTILES)

CONFIGURATION_NAMES = {
    "fund_DWD": "dwd_fundamental",
    "fund_ERA5": "era5_fundamental",
    "exaa_naive": "exaa_naive",
    "exaa_DWD": "dwd_exaa_enriched",
    "exaa_ERA5": "era5_exaa_enriched",
    "exaa_only": "exaa_only",
}

DISPLAY_NAMES = {
    "fund_DWD": "SQRA (DWD, Fundamental)",
    "fund_ERA5": "SQRA (ERA5, Fundamental)",
    "exaa_naive": "SQRA (EXAA, Naive)",
    "exaa_DWD": "SQRA (DWD, EXAA)",
    "exaa_ERA5": "SQRA (ERA5, EXAA)",
    "exaa_only": "SQRA (EXAA, Only)",
}


def load_sqra_evaluation_forecasts(
    config: ExperimentConfig,
) -> Mapping[str, pd.DataFrame]:
    """Load and validate only the six SQRA outputs for the evaluation sample."""
    paths = quantile_model_files(config.results_root)
    start = pd.Timestamp(config.evaluation_start).normalize()
    end = pd.Timestamp(config.evaluation_end).normalize()
    skip_dates = {
        pd.Timestamp(day).normalize() for day in config.evaluation_skip_dates
    }
    required_columns = (Y_TRUE_COL, *QUANTILE_COLUMNS)
    forecasts = OrderedDict()
    reference: Optional[pd.DataFrame] = None

    for model_name in QUANTILE_MODELS:
        path = paths[model_name]
        frame = enforce_eval_window(
            read_forecast_csv(path, config.timezone),
            start,
            end,
            skip_dates,
        )
        missing_columns = set(required_columns) - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"[{model_name}] Missing columns {sorted(missing_columns)} in {path}."
            )
        frame = frame.loc[:, required_columns].copy()
        check_index(
            frame,
            model_name,
            start,
            end,
            FREQ,
            N_PERIODS,
            skip_dates,
        )
        check_missing(frame, model_name)
        check_numeric(frame, required_columns, model_name)
        values = frame.loc[:, required_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"[{model_name}] Forecast contains non-finite values.")
        quantile_values = frame.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
        if np.any(np.diff(quantile_values, axis=1) < -1e-12):
            raise ValueError(f"[{model_name}] Forecast contains quantile crossing.")

        if reference is None:
            reference = frame[[Y_TRUE_COL]]
        else:
            if not frame.index.equals(reference.index):
                raise ValueError(
                    f"[{model_name}] Evaluation index differs across SQRA models."
                )
            if not np.allclose(
                frame[Y_TRUE_COL].to_numpy(dtype=float),
                reference[Y_TRUE_COL].to_numpy(dtype=float),
            ):
                raise ValueError(
                    f"[{model_name}] Realised prices differ across SQRA models."
                )
        forecasts[model_name] = frame

    return forecasts


def calculate_sqra_evaluation_metrics(
    forecasts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return median MAE/RMSE and APS for all canonical SQRA models."""
    records = []
    for model_name in QUANTILE_MODELS:
        if model_name not in forecasts:
            raise ValueError(f"Missing SQRA forecast: {model_name}")
        frame = forecasts[model_name]
        required_columns = (Y_TRUE_COL, *QUANTILE_COLUMNS)
        missing_columns = set(required_columns) - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"[{model_name}] Missing columns: {sorted(missing_columns)}"
            )
        values = frame.loc[:, required_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"[{model_name}] Forecast contains non-finite values.")

        errors = (
            frame["q0.500"].to_numpy(dtype=float)
            - frame[Y_TRUE_COL].to_numpy(dtype=float)
        )
        records.append(
            {
                "Configuration": CONFIGURATION_NAMES[model_name],
                "Model": DISPLAY_NAMES[model_name],
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                "APS": aggregated_pinball_score(frame, QUANTILES, Y_TRUE_COL),
                "Observations": len(frame),
                "Delivery days": int(delivery_dates(frame.index).nunique()),
            }
        )
    return pd.DataFrame(records)


def write_metrics_workbook(
    metrics: pd.DataFrame,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    """Write metrics and calculation metadata to a formatted Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        {
            "Setting": [
                "Evaluation start",
                "Evaluation end",
                "Excluded delivery dates",
                "Quantiles",
                "Point metrics",
                "APS definition",
            ],
            "Value": [
                str(config.evaluation_start),
                str(config.evaluation_end),
                ", ".join(map(str, config.evaluation_skip_dates)) or "None",
                ", ".join(f"{quantile:.2f}" for quantile in QUANTILES),
                "MAE and RMSE of q0.500 (median forecast)",
                "Mean pinball loss across all five quantiles and all MTUs",
            ],
        }
    )

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="SQRA metrics", index=False)
            metadata.to_excel(writer, sheet_name="Metadata", index=False)

            metrics_sheet = writer.sheets["SQRA metrics"]
            metrics_sheet.freeze_panes = "A2"
            metrics_sheet.auto_filter.ref = metrics_sheet.dimensions
            widths = {
                "A": 24,
                "B": 30,
                "C": 14,
                "D": 14,
                "E": 14,
                "F": 14,
                "G": 16,
            }
            for column, width in widths.items():
                metrics_sheet.column_dimensions[column].width = width
            for row in metrics_sheet.iter_rows(
                min_row=2,
                min_col=3,
                max_col=5,
            ):
                for cell in row:
                    cell.number_format = "0.0000"

            metadata_sheet = writer.sheets["Metadata"]
            metadata_sheet.freeze_panes = "A2"
            metadata_sheet.column_dimensions["A"].width = 28
            metadata_sheet.column_dimensions["B"].width = 66
    except ImportError as exc:
        raise RuntimeError(
            "Writing .xlsx files requires openpyxl. Install it with: "
            "python -m pip install openpyxl"
        ) from exc


def run_export(
    config: ExperimentConfig,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Calculate, print, and export the SQRA evaluation metrics."""
    destination = output_path or (config.output_root / DEFAULT_FILENAME)
    if destination.suffix.lower() != ".xlsx":
        raise ValueError(f"Excel output path must end with .xlsx: {destination}")
    forecasts = load_sqra_evaluation_forecasts(config)
    metrics = calculate_sqra_evaluation_metrics(forecasts)
    print("SQRA evaluation-period metrics")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "MAE": lambda value: f"{value:.4f}",
                "RMSE": lambda value: f"{value:.4f}",
                "APS": lambda value: f"{value:.4f}",
            },
        )
    )
    write_metrics_workbook(metrics, config, destination)
    print(f"Saved: {destination}")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Optional .xlsx path (default: OUTPUT_ROOT/{DEFAULT_FILENAME}).",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve() if args.output is not None else None
    run_export(load_experiment_config(REPO_ROOT), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
