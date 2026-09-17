#!/usr/bin/env python3
"""Run all historical weather preprocessing required by the experiment.

The evaluation dates in ``.env`` determine the ERA5 years and the final
ICON-D2 date.  Raw-data locations are read from ``ERA5_RAW_ARCHIVE`` and
``DWD_RAW_ARCHIVE``.  Each existing aggregation script is executed in a
separate process using the currently active Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
sys.path.insert(0, str(REPO_ROOT))

from experiment_config import ExperimentConfig, load_experiment_config
from pipeline.dwd_history import migrate_daily_outputs
from pipeline.dwd_raw import discover_issue_dates, raw_run_available


DEFAULT_CLUSTERS = (1, 5, 25)
ERA5_VARIABLES = ("u10", "v10", "t2m", "sp", "sd", "ssrd", "fdir")
ERA5_COMPLETION_FILE = ".preprocess_historic.json"
DWD_FOLDER_OFFSET_DATE = date(2025, 10, 26)


def _parse_clusters(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, positive list of cluster counts."""
    try:
        clusters = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--clusters must be a comma-separated list of integers, e.g. 1,5,25."
        ) from exc
    if not clusters or any(cluster <= 0 for cluster in clusters):
        raise argparse.ArgumentTypeError("Every cluster count must be positive.")
    return clusters


def _parse_icon_folder_date(value: str) -> date:
    """Parse an explicit ICON raw-folder date in ISO format."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--date must use YYYY-MM-DD format, e.g. 2026-06-12."
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess all historical ERA5 and/or ICON-D2 data required by "
            "the evaluation dates configured in .env. With neither source "
            "flag, both sources are processed."
        )
    )
    parser.add_argument(
        "--era5",
        action="store_true",
        help="Preprocess ERA5 for every required year.",
    )
    parser.add_argument(
        "--icon",
        action="store_true",
        help="Preprocess historical DWD ICON-D2 forecasts.",
    )
    parser.add_argument(
        "--clusters",
        type=_parse_clusters,
        default=DEFAULT_CLUSTERS,
        metavar="N[,N...]",
        help="Cluster counts to generate (default: 1,5,25).",
    )
    parser.add_argument(
        "--date",
        type=_parse_icon_folder_date,
        nargs="+",
        metavar="YYYY-MM-DD",
        help=(
            "Process only these DWD ICON raw-folder dates, for example "
            "--date 2026-06-12 2026-06-18. Each value selects one "
            "dwd_icon_daily_YYYYMMDD directory or "
            "dwd_icon_archived_YYYYMMDD ZIP set. This option implies --icon "
            "when no source flag is given."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate outputs that already appear complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the work plan without processing data.",
    )
    return parser


def _required_archive(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("/path/to/"):
        raise ValueError(f"{name} must point to the raw-data folder in .env.")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist or is unavailable: {path}")
    return path


def _required_years(config: ExperimentConfig) -> tuple[int, ...]:
    return tuple(range(config.weather_training_start.year, config.evaluation_end.year + 1))


def _run_child(label: str, script: Path, overrides: dict[str, str], dry_run: bool) -> None:
    command = [sys.executable, str(script)]
    print(f"\n{label}", flush=True)
    print(f"  {subprocess.list2cmdline(command)}", flush=True)
    if dry_run:
        return

    environment = os.environ.copy()
    environment.update(overrides)
    try:
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}.") from exc


def _era5_expected_dates(config: ExperimentConfig, year: int) -> tuple[date, date]:
    return (
        max(date(year, 1, 1), config.weather_training_start),
        min(date(year, 12, 31), config.evaluation_end),
    )


def _csv_date_bounds(path: Path) -> tuple[date, date] | None:
    """Return the first and last data dates from an aggregated CSV."""
    first: date | None = None
    last: date | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("timestamp,"):
                continue
            raw_timestamp = line.split(",", 1)[0].strip()
            try:
                current = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00")).date()
            except ValueError:
                return None
            if first is None:
                first = current
            last = current
    return None if first is None or last is None else (first, last)


def _era5_files_cover_expected_dates(
    config: ExperimentConfig, clusters: int, year: int
) -> bool:
    output_dir = config.era5_year_dir(clusters, year)
    if not output_dir.is_dir():
        return False
    expected_start, expected_end = _era5_expected_dates(config, year)
    for variable in ERA5_VARIABLES:
        covered = False
        for path in output_dir.glob(f"{variable}_*.csv"):
            bounds = _csv_date_bounds(path)
            if bounds and bounds[0] <= expected_start and bounds[1] >= expected_end:
                covered = True
                break
        if not covered:
            return False
    return True


def _era5_completion_payload(
    config: ExperimentConfig,
    clusters: int,
    year: int,
    main_file: Path,
    solar_file: Path,
) -> dict[str, object]:
    expected_start, expected_end = _era5_expected_dates(config, year)

    def signature(path: Path) -> dict[str, object]:
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    return {
        "version": 1,
        "clusters": clusters,
        "year": year,
        "expected_start": expected_start.isoformat(),
        "expected_end": expected_end.isoformat(),
        "main_file": signature(main_file),
        "solar_file": signature(solar_file),
    }


def _era5_output_complete(
    config: ExperimentConfig,
    clusters: int,
    year: int,
    main_file: Path,
    solar_file: Path,
) -> bool:
    marker = config.era5_year_dir(clusters, year) / ERA5_COMPLETION_FILE
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        saved == _era5_completion_payload(config, clusters, year, main_file, solar_file)
        and _era5_files_cover_expected_dates(config, clusters, year)
    )


def _mark_era5_complete(
    config: ExperimentConfig,
    clusters: int,
    year: int,
    main_file: Path,
    solar_file: Path,
) -> None:
    if not _era5_files_cover_expected_dates(config, clusters, year):
        expected_start, expected_end = _era5_expected_dates(config, year)
        raise RuntimeError(
            f"ERA5 year={year}, clusters={clusters} did not produce all seven variables "
            f"covering {expected_start} through {expected_end}."
        )
    output_dir = config.era5_year_dir(clusters, year)
    marker = output_dir / ERA5_COMPLETION_FILE
    marker.write_text(
        json.dumps(
            _era5_completion_payload(config, clusters, year, main_file, solar_file),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _preprocess_era5(
    config: ExperimentConfig,
    clusters: tuple[int, ...],
    *,
    force: bool,
    dry_run: bool,
) -> None:
    raw_archive = _required_archive("ERA5_RAW_ARCHIVE")
    years = _required_years(config)
    script = PREPROCESSING_DIR / "aggregate_era5.py"

    tasks: list[tuple[int, int, Path, Path]] = []
    missing: list[Path] = []
    for cluster_count in clusters:
        cluster_file = REPO_ROOT / "data" / "clustering" / f"icon_d2_clustering_c{cluster_count}.parquet"
        if not cluster_file.is_file():
            missing.append(cluster_file)
        for year in years:
            main_file = raw_archive / f"era5_main_{year}.grib"
            solar_file = raw_archive / f"era5_solar_{year}.grib"
            for path in (main_file, solar_file):
                if not path.is_file():
                    missing.append(path)
            tasks.append((cluster_count, year, main_file, solar_file))

    if missing:
        details = "\n".join(f"  - {path}" for path in dict.fromkeys(missing))
        raise FileNotFoundError(f"Missing ERA5 preprocessing inputs:\n{details}")

    print(
        f"\nERA5: years {years[0]}-{years[-1]}, clusters "
        f"{','.join(map(str, clusters))}",
        flush=True,
    )
    for position, (cluster_count, year, main_file, solar_file) in enumerate(tasks, start=1):
        label = f"[ERA5 {position:02d}/{len(tasks):02d}] year={year}, clusters={cluster_count}"
        if not force and _era5_output_complete(
            config, cluster_count, year, main_file, solar_file
        ):
            print(f"\n{label}: existing output complete, skipping", flush=True)
            continue
        _run_child(
            label,
            script,
            {
                "ERA5_N_CLUSTERS": str(cluster_count),
                "ERA5_PREPROCESS_YEAR": str(year),
                "ERA5_MAIN_FILE": str(main_file),
                "ERA5_SOLAR_FILE": str(solar_file),
            },
            dry_run,
        )
        if not dry_run:
            _mark_era5_complete(config, cluster_count, year, main_file, solar_file)


def _available_icon_forecast_dates(
    raw_archive: Path,
    run_hour: str,
    skipped_folder_dates: tuple[date, ...],
) -> set[date]:
    available: set[date] = set()
    skipped = set(skipped_folder_dates)
    for folder_date in discover_issue_dates(raw_archive):
        if not raw_run_available(raw_archive, folder_date, run_hour):
            continue
        if folder_date in skipped:
            continue
        forecast_date = (
            folder_date
            if folder_date < DWD_FOLDER_OFFSET_DATE
            else folder_date + timedelta(days=1)
        )
        available.add(forecast_date)
    return available


def _missing_icon_evaluation_dates(
    config: ExperimentConfig, raw_archive: Path, run_hour: str
) -> tuple[date, ...]:
    available = _available_icon_forecast_dates(
        raw_archive, run_hour, config.dwd_folder_skip_dates
    )
    required: set[date] = set()
    current = config.evaluation_start
    skipped = set(config.forecast_skip_dates)
    while current <= config.evaluation_end:
        if current not in skipped:
            required.add(current)
        current += timedelta(days=1)
    return tuple(sorted(required - available))


def _preprocess_icon(
    config: ExperimentConfig,
    clusters: tuple[int, ...],
    *,
    force: bool,
    dry_run: bool,
    folder_dates: tuple[date, ...] = (),
) -> None:
    raw_archive = _required_archive("DWD_RAW_ARCHIVE")
    missing_cluster_files = [
        REPO_ROOT / "data" / "clustering" / f"icon_d2_clustering_c{cluster}.parquet"
        for cluster in clusters
        if not (
            REPO_ROOT / "data" / "clustering" / f"icon_d2_clustering_c{cluster}.parquet"
        ).is_file()
    ]
    if missing_cluster_files:
        details = "\n".join(f"  - {path}" for path in missing_cluster_files)
        raise FileNotFoundError(f"Missing ICON-D2 clustering inputs:\n{details}")

    run_hour = "09"
    if not folder_dates:
        start_date = os.getenv("DWD_PREPROCESS_START_DATE", "2025-08-01")
        missing_dates = _missing_icon_evaluation_dates(config, raw_archive, run_hour)
        if missing_dates:
            details = ", ".join(day.isoformat() for day in missing_dates)
            raise FileNotFoundError(
                "The ICON-D2 archive has no usable run 09 input for these required "
                f"evaluation dates: {details}. Restore the raw folder(s), or explicitly "
                "add the dates to FORECAST_SKIP_DATES and EVALUATION_SKIP_DATES in .env."
            )
        date_description = f"{start_date} through {config.evaluation_end}"
        selected_dates: tuple[date | None, ...] = (None,)
    else:
        selected_dates = tuple(dict.fromkeys(folder_dates))
        missing_run_sources = []
        for selected_date in selected_dates:
            only_day = selected_date.strftime("%Y%m%d")
            if not raw_run_available(raw_archive, selected_date, run_hour):
                missing_run_sources.append(
                    f"{raw_archive} / "
                    f"{{dwd_icon_daily_{only_day}, dwd_icon_archived_{only_day}}} "
                    f"(run {run_hour})"
                )
            if selected_date > config.evaluation_end:
                raise ValueError(
                    f"Selected ICON folder date {selected_date} is after the "
                    f"configured evaluation end {config.evaluation_end}."
                )
        if missing_run_sources:
            details = "\n".join(f"  - {path}" for path in missing_run_sources)
            raise FileNotFoundError(
                "The selected ICON-D2 raw source(s) do not contain run 09:\n"
                f"{details}"
            )
        date_description = "raw folder dates " + ",".join(
            selected_date.isoformat() for selected_date in selected_dates
        )

    script = PREPROCESSING_DIR / "aggregate_icon_d2.py"
    print(
        f"\nICON-D2: {date_description}, clusters "
        f"{','.join(map(str, clusters))}",
        flush=True,
    )
    tasks = [
        (cluster_count, selected_date)
        for selected_date in selected_dates
        for cluster_count in clusters
    ]
    for position, (cluster_count, selected_date) in enumerate(tasks, start=1):
        only_day = "" if selected_date is None else selected_date.strftime("%Y%m%d")
        task_start_date = (
            start_date if selected_date is None else selected_date.isoformat()
        )
        date_label = "all dates" if selected_date is None else str(selected_date)
        _run_child(
            f"[ICON {position:02d}/{len(tasks):02d}] date={date_label}, "
            f"clusters={cluster_count}",
            script,
            {
                "DWD_N_CLUSTERS": str(cluster_count),
                "DWD_PREPROCESS_START_DATE": task_start_date,
                "DWD_SKIP_EXISTING_OUTPUT": "false" if force else "true",
                "DWD_ONLY_DAY": only_day,
                "DWD_RUN_HOUR": run_hour,
            },
            dry_run,
        )
    if not dry_run and callable(getattr(config, "icon_cluster_dir", None)):
        for cluster_count in clusters:
            migrated = migrate_daily_outputs(
                config.icon_cluster_dir(cluster_count),
                timezone=config.timezone,
                force=True,
            )
            print(
                f"ICON-D2 C={cluster_count}: consolidated {migrated} valid "
                "daily outputs into variable histories.",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    folder_dates = tuple(args.date or ())
    if folder_dates and args.era5:
        parser.error("--date is an ICON-D2 folder selector and cannot be used with --era5.")
    config = load_experiment_config(REPO_ROOT)

    run_both = not args.era5 and not args.icon and not folder_dates
    run_icon = args.icon or run_both or bool(folder_dates)
    run_era5 = args.era5 or run_both

    plan_lines = [
        "Historical preprocessing plan:",
        f"  evaluation: {config.evaluation_start} through {config.evaluation_end}",
        f"  weather history starts: {config.weather_training_start}",
        "  sources: "
        + ", ".join(
            name
            for name, enabled in (("ICON-D2", run_icon), ("ERA5", run_era5))
            if enabled
        ),
        f"  clusters: {','.join(map(str, args.clusters))}",
    ]
    if folder_dates:
        plan_lines.append(
            "  ICON raw folder dates: "
            + ",".join(folder_date.isoformat() for folder_date in folder_dates)
        )
    plan_lines.append(f"  mode: {'dry run' if args.dry_run else 'execute'}")
    print("\n".join(plan_lines), flush=True)

    try:
        if run_icon:
            _preprocess_icon(
                config,
                args.clusters,
                force=args.force,
                dry_run=args.dry_run,
                folder_dates=folder_dates,
            )
        if run_era5:
            _preprocess_era5(
                config,
                args.clusters,
                force=args.force,
                dry_run=args.dry_run,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    message = (
        "Historical preprocessing dry run completed successfully."
        if args.dry_run
        else "Historical preprocessing completed successfully."
    )
    print(f"\n{message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
