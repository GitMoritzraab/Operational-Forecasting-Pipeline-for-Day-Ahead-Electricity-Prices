#!/usr/bin/env python3
"""Check every canonical point forecast for missing delivery days and MTUs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402
from experiment_manifest import lear_runs  # noqa: E402
from pipeline.delivery_index import (  # noqa: E402
    DELIVERY_DATE_COLUMN,
    MTU_COLUMN,
    MTUS_PER_DAY,
    read_forecast_csv,
)


@dataclass(frozen=True)
class CoverageReport:
    """Coverage findings for one point-forecast file."""

    name: str
    path: Path
    row_count: int = 0
    missing_days: Tuple[date, ...] = ()
    partial_days: Mapping[date, Tuple[int, ...]] = None
    extra_days: Tuple[date, ...] = ()
    missing_evaluation_days: Tuple[date, ...] = ()
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.partial_days is None:
            object.__setattr__(self, "partial_days", {})

    @property
    def ok(self) -> bool:
        return not (
            self.error
            or self.missing_days
            or self.partial_days
            or self.extra_days
            or self.missing_evaluation_days
        )


def point_forecast_files(results_root: Path) -> Dict[str, Path]:
    """Return all 28 canonical point-forecast result paths."""
    paths = {
        run.name: run.output_dir(results_root) / "forecast.csv"
        for run in lear_runs()
    }
    paths["exaa_naive"] = (
        results_root / "lear_op_results" / "exaa_naive" / "forecast.csv"
    )
    return paths


def expected_delivery_days(
    start: date,
    end: date,
    skipped: Sequence[date],
) -> Tuple[date, ...]:
    """Build an inclusive sequence of expected, non-skipped calendar days."""
    skipped_set = set(skipped)
    return tuple(
        timestamp.date()
        for timestamp in pd.date_range(start=start, end=end, freq="D")
        if timestamp.date() not in skipped_set
    )


def inspect_point_forecast(
    name: str,
    path: Path,
    point_days: Sequence[date],
    evaluation_days: Sequence[date],
) -> CoverageReport:
    """Compare one canonical forecast against the configured date horizons."""
    if not path.is_file():
        return CoverageReport(name=name, path=path, error="file is missing")

    try:
        forecast = read_forecast_csv(
            path,
            required_columns=("y_pred", "y_true"),
            require_complete_days=False,
        )
    except Exception as exc:
        return CoverageReport(
            name=name,
            path=path,
            error=f"cannot read or validate canonical forecast: {exc}",
        )

    dates = pd.DatetimeIndex(
        forecast.index.get_level_values(DELIVERY_DATE_COLUMN)
    )
    mtus = forecast.index.get_level_values(MTU_COLUMN)
    actual_days = {timestamp.date() for timestamp in dates.unique()}
    expected_point = set(point_days)
    expected_evaluation = set(evaluation_days)

    partial_days = {}
    complete_days = set()
    expected_mtus = set(range(1, MTUS_PER_DAY + 1))
    for day in sorted(actual_days):
        day_mask = dates.date == day
        present_mtus = {int(value) for value in mtus[day_mask]}
        missing_mtus = tuple(sorted(expected_mtus - present_mtus))
        if missing_mtus:
            partial_days[day] = missing_mtus
        else:
            complete_days.add(day)

    missing_days = tuple(sorted(expected_point - actual_days))
    extra_days = tuple(sorted(actual_days - expected_point))
    missing_evaluation_days = tuple(
        sorted(expected_evaluation - complete_days)
    )
    return CoverageReport(
        name=name,
        path=path,
        row_count=len(forecast),
        missing_days=missing_days,
        partial_days=partial_days,
        extra_days=extra_days,
        missing_evaluation_days=missing_evaluation_days,
    )


def check_all_point_forecasts(
    config: ExperimentConfig,
) -> Tuple[CoverageReport, ...]:
    """Inspect all canonical LEAR and EXAA-naive point outputs."""
    point_days = expected_delivery_days(
        config.point_forecast_start,
        config.evaluation_end,
        config.forecast_skip_dates,
    )
    evaluation_days = expected_delivery_days(
        config.evaluation_start,
        config.evaluation_end,
        config.evaluation_skip_dates,
    )
    return tuple(
        inspect_point_forecast(
            name,
            path,
            point_days,
            evaluation_days,
        )
        for name, path in point_forecast_files(config.results_root).items()
    )


def _format_dates(days: Sequence[date]) -> str:
    return ", ".join(day.isoformat() for day in days)


def print_reports(
    config: ExperimentConfig,
    reports: Sequence[CoverageReport],
    *,
    show_ok: bool,
) -> None:
    """Print a concise human-readable coverage report."""
    point_days = expected_delivery_days(
        config.point_forecast_start,
        config.evaluation_end,
        config.forecast_skip_dates,
    )
    evaluation_days = expected_delivery_days(
        config.evaluation_start,
        config.evaluation_end,
        config.evaluation_skip_dates,
    )
    print("Point-forecast coverage check")
    print(
        f"  full point horizon: {config.point_forecast_start} through "
        f"{config.evaluation_end} ({len(point_days)} expected days)"
    )
    print(
        f"  evaluation horizon: {config.evaluation_start} through "
        f"{config.evaluation_end} ({len(evaluation_days)} expected days)"
    )
    print(f"  models: {len(reports)}")

    failures = [report for report in reports if not report.ok]
    if show_ok:
        for report in reports:
            if report.ok:
                print(f"\n[OK] {report.name}: {report.row_count} rows")

    for report in failures:
        print(f"\n[PROBLEM] {report.name}")
        print(f"  file: {report.path}")
        if report.error:
            print(f"  error: {report.error}")
            continue
        print(f"  rows: {report.row_count}")
        if report.missing_days:
            print(
                f"  entirely missing point-horizon days ({len(report.missing_days)}): "
                f"{_format_dates(report.missing_days)}"
            )
        if report.partial_days:
            print(f"  partial delivery days ({len(report.partial_days)}):")
            for day, missing_mtus in report.partial_days.items():
                print(
                    f"    {day.isoformat()}: missing MTUs "
                    + ",".join(map(str, missing_mtus))
                )
        if report.extra_days:
            print(
                f"  unexpected extra days ({len(report.extra_days)}): "
                f"{_format_dates(report.extra_days)}"
            )
        if report.missing_evaluation_days:
            print(
                "  MISSING FROM EVALUATION "
                f"({len(report.missing_evaluation_days)}): "
                f"{_format_dates(report.missing_evaluation_days)}"
            )

    print("\nSummary")
    print(f"  complete models: {len(reports) - len(failures)}/{len(reports)}")
    print(f"  models with coverage problems: {len(failures)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Print a separate line for every complete model as well as problems.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(REPO_ROOT)
    reports = check_all_point_forecasts(config)
    print_reports(config, reports, show_ok=args.show_ok)
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
