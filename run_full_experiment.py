#!/usr/bin/env python3
"""Run selected point forecasts and optional SQRA forecasts.

All automated stages are ordinary Python modules or scripts.  The notebooks
in the repository are optional interactive views and are never executed here.
Evaluation, tables, and figures are handled by ``run_full_evaluation.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from experiment_config import ExperimentConfig, load_experiment_config
from experiment_manifest import SQRA_RUNS, LearRun, lear_runs
from pipeline.forecast_schema import FORECAST_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parent


def _execution_python() -> Path:
    """Return the interpreter used for every automated pipeline stage."""
    configured = os.getenv("FORECAST_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"FORECAST_PYTHON does not exist: {candidate}")
        return candidate.resolve()

    local_candidate = (
        REPO_ROOT / "forecast" / "Scripts" / "python.exe"
        if os.name == "nt"
        else REPO_ROOT / "forecast" / "bin" / "python"
    )
    if local_candidate.is_file():
        return local_candidate.resolve()
    return Path(sys.executable).resolve()


def _date_summary(config: ExperimentConfig) -> str:
    return (
        f"evaluation={config.evaluation_start}..{config.evaluation_end}, "
        f"point_forecasts={config.point_forecast_start}..{config.evaluation_end}, "
        f"market_inputs={config.input_start}..{config.evaluation_end}"
    )


POINT_SOURCES = frozenset({"era5", "dwd", "exaa"})
TRAIN_DAYS = (56, 112, 364)


def _select_lear_runs(
    sources: set[str],
    train_days: int | None = None,
    cluster: int | None = None,
    fundamental: bool | None = None,
) -> tuple[LearRun, ...]:
    """Return canonical fitted point runs for the requested source categories."""
    unknown = sources - POINT_SOURCES
    if unknown:
        raise ValueError(f"Unknown point-forecast sources: {', '.join(sorted(unknown))}")

    selected: list[LearRun] = []
    for run in lear_runs():
        source = (
            "exaa"
            if run.use_exaa_only
            else str(run.weather_source).lower()
        )
        if source not in sources:
            continue
        if train_days is not None and run.train_days != train_days:
            continue
        # Cluster and fundamental/EXAA-enriched are properties only of the
        # weather-based ERA5 and DWD runs.  Explicit EXAA-only selections are
        # intentionally left unchanged when combined with a weather source.
        if not run.use_exaa_only:
            if cluster is not None and run.clusters != cluster:
                continue
            if fundamental is not None and (not run.use_exaa) != fundamental:
                continue
        selected.append(run)
    return tuple(selected)


def _parse_boolean(value: str) -> bool:
    """Parse an explicit true/false command-line value."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _preflight(
    config: ExperimentConfig,
    point_runs: Sequence[LearRun],
    *,
    include_exaa_naive: bool,
    run_sqra: bool,
) -> list[str]:
    """Check only the inputs needed by the selected forecast plan."""
    errors: list[str] = []
    if point_runs or include_exaa_naive:
        api_key = os.getenv("ENTSOE_API_KEY", "")
        if not api_key or api_key == "your_api_key_here":
            errors.append("ENTSOE_API_KEY is missing from .env.")

        era5_clusters = sorted(
            {
                run.clusters
                for run in point_runs
                if run.weather_source == "ERA5" and run.clusters is not None
            }
        )
        if era5_clusters:
            years = range(
                config.weather_training_start.year,
                config.evaluation_end.year + 1,
            )
            for clusters in era5_clusters:
                for year in years:
                    directory = config.era5_year_dir(clusters, year)
                    if not directory.is_dir() or not any(directory.glob("*.csv")):
                        errors.append(
                            f"Missing ERA5 C={clusters}, year={year}: {directory}"
                        )

        dwd_clusters = sorted(
            {
                run.clusters
                for run in point_runs
                if run.weather_source == "DWD" and run.clusters is not None
            }
        )
        for clusters in dwd_clusters:
            icon_dir = config.icon_cluster_dir(clusters)
            if not icon_dir.is_dir() or not any(icon_dir.rglob("*.csv")):
                errors.append(f"Missing DWD ICON-D2 C={clusters}: {icon_dir}")

    if run_sqra:
        scheduled_point_outputs = {
            run.output_dir(config.results_root) / "forecast.csv" for run in point_runs
        }
        if include_exaa_naive:
            scheduled_point_outputs.add(
                config.results_root
                / "lear_op_results"
                / "exaa_naive"
                / "forecast.csv"
            )
        sqra_inputs = sorted(
            {
                path
                for run in SQRA_RUNS.values()
                for path in run.inputs(config.results_root)
            },
            key=str,
        )
        for path in sqra_inputs:
            # A combined point+SQRA command will create or validate these
            # selected point outputs before starting SQRA.
            if path in scheduled_point_outputs:
                continue
            if not path.is_file():
                errors.append(f"Missing SQRA point forecast: {path}")
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    header = handle.readline().strip().split(",")
            except OSError as exc:
                errors.append(f"Unreadable SQRA point forecast: {path} ({exc})")
                continue
            if header[:2] != ["delivery_date", "mtu"]:
                errors.append(f"Incompatible SQRA point forecast schema: {path}")
    return errors


def _result_complete(run: LearRun, config: ExperimentConfig) -> bool:
    config_path = run.output_dir(config.results_root) / "config.json"
    forecast_path = run.output_dir(config.results_root) / "forecast.csv"
    if not config_path.exists() or not forecast_path.exists():
        return False
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        with forecast_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
    except (json.JSONDecodeError, OSError):
        return False
    return (
        header[:2] == ["delivery_date", "mtu"]
        and saved.get("test_start") == str(config.point_forecast_start)
        and saved.get("test_end") == str(config.evaluation_end)
        and saved.get("train_days") == run.train_days
        and saved.get("use_vst") == config.lear_use_vst
        and saved.get("forecast_schema_version") == FORECAST_SCHEMA_VERSION
    )


def _exaa_naive_result_complete(config: ExperimentConfig) -> bool:
    """Return whether the date-matched EXAA-naive baseline can be resumed."""
    output_dir = config.results_root / "lear_op_results" / "exaa_naive"
    config_path = output_dir / "config.json"
    forecast_path = output_dir / "forecast.csv"
    if not config_path.exists() or not forecast_path.exists():
        return False
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        with forecast_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
    except (json.JSONDecodeError, OSError):
        return False
    return (
        header[:2] == ["delivery_date", "mtu"]
        and saved.get("test_start") == str(config.point_forecast_start)
        and saved.get("test_end") == str(config.evaluation_end)
        and saved.get("forecast_schema_version") == FORECAST_SCHEMA_VERSION
    )


def _safe_log_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def _run_logged_command(
    command: Sequence[object],
    config: ExperimentConfig,
    *,
    stage: str,
    name: str,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Stream one child process to the terminal and a deterministic UTF-8 log."""
    normalized = [str(part) for part in command]
    log_path = config.output_root / "logs" / stage / f"{_safe_log_name(name)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(environment or {})
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    started = datetime.now().astimezone()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        header = (
            f"started={started.isoformat(timespec='seconds')}\n"
            f"cwd={REPO_ROOT}\n"
            f"command={' '.join(normalized)}\n"
        )
        log.write(header)
        log.flush()
        process = None
        try:
            process = subprocess.Popen(
                normalized,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
                raise RuntimeError("Could not capture child-process output.")
            with process.stdout:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            return_code = process.wait()
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finished = datetime.now().astimezone()
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            log.write(
                f"finished={finished.isoformat(timespec='seconds')}\n"
                f"status={status}\nerror={type(exc).__name__}: {exc}\n"
            )
            raise

        finished = datetime.now().astimezone()
        status = "success" if return_code == 0 else f"failed ({return_code})"
        log.write(
            f"finished={finished.isoformat(timespec='seconds')}\n"
            f"status={status}\n"
        )

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, normalized)


def _run_module(
    config: ExperimentConfig,
    stage: str,
    name: str,
    module: str,
    *arguments: str,
) -> None:
    _run_logged_command(
        [_execution_python(), "-m", module, *arguments],
        config,
        stage=stage,
        name=name,
    )


def _run_lear(
    config: ExperimentConfig,
    force: bool,
    runs: Sequence[LearRun],
    *,
    include_exaa_naive: bool,
) -> None:
    """Run one selected point-forecast batch in canonical manifest order."""
    if include_exaa_naive:
        if not force and _exaa_naive_result_complete(config):
            print("resume: exaa_naive")
        else:
            print("LEAR baseline: EXAA-naive", flush=True)
            _run_module(
                config,
                "lear",
                "exaa_naive",
                "pipeline.lear.run_lear",
                "exaa-naive",
            )

    for position, run in enumerate(runs, start=1):
        if not force and _result_complete(run, config):
            print(f"[{position:02d}/{len(runs):02d}] resume: {run.name}")
            continue
        print(f"[{position:02d}/{len(runs):02d}] LEAR: {run.name}", flush=True)
        _run_module(
            config,
            "lear",
            run.name,
            "pipeline.lear.run_lear",
            "operational",
            "--config",
            run.name,
        )


def _run_sqra(config: ExperimentConfig) -> None:
    for position, name in enumerate(SQRA_RUNS, start=1):
        print(f"[{position}/{len(SQRA_RUNS)}] SQRA: {name}", flush=True)
        _run_module(
            config,
            "sqra",
            name,
            "pipeline.sqra.run_sqra",
            "--config",
            name,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--era5",
        action="store_true",
        help="Run ERA5 point forecasts (fundamental and EXAA-enriched).",
    )
    parser.add_argument(
        "--dwd",
        action="store_true",
        help="Run DWD ICON-D2 point forecasts (fundamental and EXAA-enriched).",
    )
    parser.add_argument(
        "--exaa",
        action="store_true",
        help="Run EXAA-only fitted forecasts and the EXAA-naive baseline.",
    )
    parser.add_argument(
        "--d",
        type=int,
        choices=TRAIN_DAYS,
        metavar="{56,112,364}",
        help="Restrict selected fitted point forecasts to one training window.",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        choices=(1, 5, 25),
        metavar="{1,5,25}",
        help=(
            "Restrict selected ERA5/DWD point forecasts to one weather-cluster "
            "configuration. Requires --era5 or --dwd."
        ),
    )
    parser.add_argument(
        "--fundamental",
        type=_parse_boolean,
        metavar="{true,false}",
        help=(
            "Restrict selected ERA5/DWD runs: true selects the fundamental "
            "variant and false selects the EXAA-enriched variant. If omitted, "
            "both variants run. Requires --era5 or --dwd."
        ),
    )
    parser.add_argument(
        "--sqra",
        action="store_true",
        help=(
            "Run all six SQRA configurations. By itself this runs SQRA only; "
            "with source flags it runs those point forecasts first."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only check configuration and input coverage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived plan without executing it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun complete selected point outputs instead of resuming.",
    )
    args = parser.parse_args()

    config = load_experiment_config(REPO_ROOT)
    explicit_sources = {
        source for source in POINT_SOURCES if getattr(args, source)
    }
    weather_source_selected = bool(explicit_sources & {"era5", "dwd"})
    weather_filter_used = args.cluster is not None or args.fundamental is not None
    if weather_filter_used and not weather_source_selected:
        parser.error("--cluster and --fundamental require --era5 or --dwd")
    if args.sqra and not explicit_sources and (
        args.d is not None or weather_filter_used
    ):
        parser.error(
            "--d, --cluster, and --fundamental require a point source when "
            "used together with --sqra"
        )

    run_points = not args.sqra or bool(explicit_sources)
    selected_sources = explicit_sources or (set(POINT_SOURCES) if run_points else set())
    point_runs = (
        _select_lear_runs(
            selected_sources,
            args.d,
            cluster=args.cluster,
            fundamental=args.fundamental,
        )
        if run_points
        else ()
    )
    include_exaa_naive = run_points and "exaa" in selected_sources
    if run_points and not point_runs and not include_exaa_naive:
        selected = ", ".join(sorted(selected_sources))
        parser.error(
            "No fitted point forecasts match "
            f"sources={selected}, D={args.d}, cluster={args.cluster}, "
            f"fundamental={args.fundamental}"
        )

    print(_date_summary(config))
    print(f"Execution Python={_execution_python()}")
    selected_point_count = len(point_runs) + int(include_exaa_naive)
    print(
        f"Selected point forecasts={selected_point_count}, "
        f"SQRA configurations={len(SQRA_RUNS) if args.sqra else 0}"
    )
    print(
        "SQRA calibration="
        + ("MTU-specific" if config.sqra_mtu_specific else "pooled across MTUs")
    )
    if args.dry_run:
        if run_points:
            if include_exaa_naive:
                print("LEAR  exaa_naive")
            for run in point_runs:
                print(f"LEAR  {run.name} -> {run.output_dir(config.results_root)}")
        if args.sqra:
            for name, run in SQRA_RUNS.items():
                inputs = ", ".join(map(str, run.inputs(config.results_root)))
                print(f"SQRA  {name} <- {inputs}")
        return 0

    errors = _preflight(
        config,
        point_runs,
        include_exaa_naive=include_exaa_naive,
        run_sqra=args.sqra,
    )
    if errors:
        print("Preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    if args.preflight:
        print("Preflight passed.")
        return 0

    if run_points:
        _run_lear(
            config,
            args.force,
            point_runs,
            include_exaa_naive=include_exaa_naive,
        )
    if args.sqra:
        _run_sqra(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
