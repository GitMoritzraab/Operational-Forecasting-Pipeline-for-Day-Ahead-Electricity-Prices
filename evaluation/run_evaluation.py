#!/usr/bin/env python3
"""Evaluate all point and SQRA forecasts without executing a notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluation_core import (  # noqa: E402
    LABEL_MAP_POINT,
    LABEL_MAP_SQRA,
    EvaluationData,
    EvaluationResults,
    evaluate,
    load_evaluation_data,
    point_model_files,
    plot_gw_heatmap,
)
from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402
from pipeline.lear.lear_model import compute_metrics  # noqa: E402


def save_point_evaluation_metrics(
    config: ExperimentConfig,
    data: EvaluationData,
) -> None:
    """Upsert the current evaluation-period row in every point metrics file.

    This operates only on already-generated forecast files.  It never refits a
    point model, and it also creates ``metrics.csv`` for a point baseline that
    previously had only a forecast export.
    """
    for model_name, forecast_path in point_model_files(config.results_root).items():
        evaluation_forecast = data.point[[model_name, "y_true"]].rename(
            columns={model_name: "y_pred"}
        )
        row = compute_metrics(evaluation_forecast, "evaluation")
        metrics_path = forecast_path.parent / "metrics.csv"
        if metrics_path.is_file():
            metrics = pd.read_csv(metrics_path)
            if "period" not in metrics.columns:
                raise ValueError(
                    f"Point metrics file lacks required column 'period': {metrics_path}"
                )
            # Replace rather than duplicate the row.  This also refreshes the
            # values if EVALUATION_START_DATE/END_DATE changed since a prior run.
            metrics = metrics.loc[
                metrics["period"].astype(str).str.lower() != "evaluation"
            ].copy()
            metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
        else:
            metrics = pd.DataFrame([row])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        print(f"Saved evaluation metrics: {metrics_path}")


def print_evaluation_tables(results: EvaluationResults) -> None:
    """Print the tables that the interactive notebook displays."""
    print("\nPoint forecast metrics")
    print(results.point_metrics.to_string())
    print("\nProbabilistic median MAE")
    print(results.median_mae.to_string())
    print("\nAggregated Pinball Score")
    print(results.aps_summary.to_string())
    for label, table in results.coverage.items():
        print(f"\n{label}")
        print(table.to_string())


def save_evaluation_figures(
    results: EvaluationResults, output_directory: Path
) -> None:
    """Write the same four GW PDFs produced by the original evaluation."""
    output_directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            results.point_gw_fundamental,
            "gw_test_mae_fundamental.pdf",
            (8, 7),
            11,
            LABEL_MAP_POINT,
            0.05,
            0.02,
        ),
        (
            results.point_gw_exaa,
            "gw_test_mae_exaa.pdf",
            (9, 8),
            11,
            LABEL_MAP_POINT,
            0.047,
            0.02,
        ),
        (
            results.point_gw_all,
            "gw_test_mae_all.pdf",
            (13, 11),
            9,
            LABEL_MAP_POINT,
            0.03,
            0.01,
        ),
        (
            results.aps_gw,
            "gw_test_aps.pdf",
            (6, 4.5),
            13,
            LABEL_MAP_SQRA,
            0.03,
            0.01,
        ),
    )
    for p_values, filename, figsize, fontsize, labels, fraction, pad in specifications:
        figure = plot_gw_heatmap(
            p_values,
            title="",
            figsize=figsize,
            fontsize=fontsize,
            label_map=labels,
            fraction=fraction,
            pad=pad,
        )
        path = output_directory / filename
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {path}")


def run_evaluation(config: ExperimentConfig) -> EvaluationResults:
    """Load, validate, evaluate, report, and plot the configured experiment."""
    data = load_evaluation_data(config)
    print("All validation checks passed.")
    save_point_evaluation_metrics(config, data)
    results = evaluate(data)
    print_evaluation_tables(results)
    save_evaluation_figures(results, config.output_root)
    return results


def main() -> int:
    config = load_experiment_config(REPO_ROOT)
    run_evaluation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
