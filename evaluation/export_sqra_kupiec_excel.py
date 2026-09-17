#!/usr/bin/env python3
"""Export SQRA Kupiec MTU pass counts for two PIs and significance levels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluation_core import mtu_kupiec_test  # noqa: E402
from evaluation.export_sqra_evaluation_metrics import (  # noqa: E402
    load_sqra_evaluation_forecasts,
)
from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402


DEFAULT_FILENAME = "sqra_kupiec_mtu_counts.xlsx"

# Match the model order in manuscript Table D.5.
MODEL_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("fund_ERA5", "era5_fundamental", "SQRA (ERA5, Fundamental)"),
    ("fund_DWD", "dwd_fundamental", "SQRA (DWD, Fundamental)"),
    ("exaa_ERA5", "era5_exaa_enriched", "SQRA (ERA5, EXAA)"),
    ("exaa_DWD", "dwd_exaa_enriched", "SQRA (DWD, EXAA)"),
    ("exaa_naive", "exaa_naive", "SQRA (EXAA, Naive)"),
    ("exaa_only", "exaa_only", "SQRA (EXAA, Only)"),
)

TEST_COLUMNS: Tuple[Tuple[str, float, float, float], ...] = (
    ("50% PI - 1%", 0.25, 0.75, 0.01),
    ("50% PI - 5%", 0.25, 0.75, 0.05),
    ("80% PI - 1%", 0.10, 0.90, 0.01),
    ("80% PI - 5%", 0.10, 0.90, 0.05),
)


def calculate_kupiec_mtu_counts(
    forecasts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Count MTUs whose Kupiec null is not rejected in each configuration."""
    records = []
    for model_key, configuration, display_name in MODEL_ROWS:
        if model_key not in forecasts:
            raise ValueError(f"Missing SQRA forecast: {model_key}")
        frame = forecasts[model_key]
        record = {
            "Configuration": configuration,
            "Model": display_name,
        }
        for (
            column,
            lower_quantile,
            upper_quantile,
            significance_level,
        ) in TEST_COLUMNS:
            nominal_coverage = upper_quantile - lower_quantile
            record[column] = mtu_kupiec_test(
                frame,
                lower_quantile,
                upper_quantile,
                alpha=nominal_coverage,
                significance_level=significance_level,
            )
        records.append(record)
    return pd.DataFrame(records)


def write_kupiec_workbook(
    counts: pd.DataFrame,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    """Write the four Kupiec count columns and metadata to an Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        {
            "Setting": [
                "Evaluation start",
                "Evaluation end",
                "Excluded delivery dates",
                "MTUs per normalized delivery day",
                "Passing rule",
                "50% prediction interval",
                "80% prediction interval",
            ],
            "Value": [
                str(config.evaluation_start),
                str(config.evaluation_end),
                ", ".join(map(str, config.evaluation_skip_dates)) or "None",
                "96",
                "Null not rejected when Kupiec p-value >= significance level",
                "q0.250 to q0.750; nominal coverage 0.50",
                "q0.100 to q0.900; nominal coverage 0.80",
            ],
        }
    )

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            counts.to_excel(writer, sheet_name="Kupiec MTU counts", index=False)
            metadata.to_excel(writer, sheet_name="Metadata", index=False)

            counts_sheet = writer.sheets["Kupiec MTU counts"]
            counts_sheet.freeze_panes = "A2"
            counts_sheet.auto_filter.ref = counts_sheet.dimensions
            counts_sheet.column_dimensions["A"].width = 24
            counts_sheet.column_dimensions["B"].width = 30
            for column in ("C", "D", "E", "F"):
                counts_sheet.column_dimensions[column].width = 15

            metadata_sheet = writer.sheets["Metadata"]
            metadata_sheet.freeze_panes = "A2"
            metadata_sheet.column_dimensions["A"].width = 34
            metadata_sheet.column_dimensions["B"].width = 68
    except ImportError as exc:
        raise RuntimeError(
            "Writing .xlsx files requires openpyxl. Install it with: "
            "python -m pip install openpyxl"
        ) from exc


def run_export(
    config: ExperimentConfig,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load SQRA forecasts, calculate counts, and save the workbook."""
    destination = output_path or (config.output_root / DEFAULT_FILENAME)
    if destination.suffix.lower() != ".xlsx":
        raise ValueError(f"Excel output path must end with .xlsx: {destination}")

    forecasts = load_sqra_evaluation_forecasts(config)
    counts = calculate_kupiec_mtu_counts(forecasts)
    print("SQRA Kupiec MTUs for which the null hypothesis is not rejected")
    print(counts.to_string(index=False))
    write_kupiec_workbook(counts, config, destination)
    print(f"Saved: {destination}")
    return counts


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
