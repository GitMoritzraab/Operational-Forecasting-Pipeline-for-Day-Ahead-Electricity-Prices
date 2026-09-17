#!/usr/bin/env python3
"""Evaluate completed forecasts and generate the paper tables and figures.

This post-processing runner is intentionally separate from
``run_full_experiment.py``.  By default it never fits a LEAR or SQRA model
and can therefore be rerun cheaply after the required point and
probabilistic forecasts have been produced.  The optional ``--anc`` switch
runs the two manuscript feature-importance variants before post-processing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Optional, Sequence

from experiment_config import ExperimentConfig, load_experiment_config
from experiment_manifest import SQRA_RUNS, lear_runs


REPO_ROOT = Path(__file__).resolve().parent
VALID_STAGES = ("evaluation", "plots")
PLOT_SCRIPTS = (
    "generate_mae_tables.py",
    "generate_comptime_tables.py",
    "plot_prob_forecast_example.py",
    "plot_exaa_epex_correlation.py",
)
POINT_FORECAST_COLUMNS = frozenset(("delivery_date", "mtu", "y_pred", "y_true"))
SQRA_FORECAST_COLUMNS = frozenset(
    (
        "delivery_date",
        "mtu",
        "y_true",
        "q0.100",
        "q0.250",
        "q0.500",
        "q0.750",
        "q0.900",
    )
)


@dataclass(frozen=True)
class CsvRequirement:
    """One input artifact that must exist before post-processing starts."""

    label: str
    path: Path
    required_columns: frozenset[str]
    canonical_index: bool = False


def _execution_python() -> Path:
    """Return the interpreter selected for all post-processing commands."""
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
        f"results={config.results_root}, output={config.output_root}"
    )


def _safe_log_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def _run_logged_command(
    command: Sequence[object],
    config: ExperimentConfig,
    *,
    stage: str,
    name: str,
    environment: Optional[Mapping[str, str]] = None,
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
        log.write(
            f"started={started.isoformat(timespec='seconds')}\n"
            f"cwd={REPO_ROOT}\n"
            f"command={' '.join(normalized)}\n"
        )
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


def _point_forecast_requirements(config: ExperimentConfig) -> list[CsvRequirement]:
    requirements = [
        CsvRequirement(
            f"point forecast {run.name}",
            run.output_dir(config.results_root) / "forecast.csv",
            POINT_FORECAST_COLUMNS,
            canonical_index=True,
        )
        for run in lear_runs()
    ]
    requirements.append(
        CsvRequirement(
            "point forecast exaa_naive",
            config.results_root / "lear_op_results" / "exaa_naive" / "forecast.csv",
            POINT_FORECAST_COLUMNS,
            canonical_index=True,
        )
    )
    return requirements


def _sqra_forecast_requirements(config: ExperimentConfig) -> list[CsvRequirement]:
    return [
        CsvRequirement(
            f"SQRA forecast {name}",
            run.output_dir(config.results_root) / "forecast.csv",
            SQRA_FORECAST_COLUMNS,
            canonical_index=True,
        )
        for name, run in SQRA_RUNS.items()
    ]


def _runtime_requirements(config: ExperimentConfig) -> list[CsvRequirement]:
    runtime_columns = frozenset(("forecast_day",))
    requirements = [
        CsvRequirement(
            f"point runtime {run.name}",
            run.output_dir(config.results_root) / "runtime.csv",
            runtime_columns,
        )
        for run in lear_runs()
    ]
    requirements.extend(
        CsvRequirement(
            f"SQRA runtime {name}",
            run.output_dir(config.results_root) / "runtime.csv",
            runtime_columns,
        )
        for name, run in SQRA_RUNS.items()
    )
    return requirements


def _price_requirements(config: ExperimentConfig) -> list[CsvRequirement]:
    return [
        CsvRequirement(
            "EPEX DE-LU price cache",
            config.entsoe_price_cache_dir / "prices_da.csv",
            frozenset(("price_da",)),
        ),
        CsvRequirement(
            "EXAA price cache",
            config.exaa_price_cache_dir / "prices_exaa.csv",
            frozenset(("price_exaa",)),
        ),
    ]


def _requirements(config: ExperimentConfig, stages: set[str]) -> list[CsvRequirement]:
    requirements: list[CsvRequirement] = []
    if "evaluation" in stages:
        requirements.extend(_point_forecast_requirements(config))
        requirements.extend(_sqra_forecast_requirements(config))
    if "plots" in stages:
        requirements.extend(_point_forecast_requirements(config))
        requirements.extend(_sqra_forecast_requirements(config))
        requirements.extend(_runtime_requirements(config))
        requirements.extend(_price_requirements(config))

    # Evaluation and plots share most inputs.  Validate each physical file once.
    unique: dict[Path, CsvRequirement] = {}
    for requirement in requirements:
        existing = unique.get(requirement.path)
        if existing is None:
            unique[requirement.path] = requirement
        else:
            unique[requirement.path] = CsvRequirement(
                existing.label,
                existing.path,
                existing.required_columns | requirement.required_columns,
                existing.canonical_index or requirement.canonical_index,
            )
    return list(unique.values())


def _validate_csv(requirement: CsvRequirement) -> list[str]:
    path = requirement.path
    if not path.is_file():
        return [f"Missing {requirement.label}: {path}"]
    try:
        if path.stat().st_size == 0:
            return [f"Empty {requirement.label}: {path}"]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except OSError as exc:
        return [f"Cannot read {requirement.label}: {path} ({exc})"]
    if not header:
        return [f"Missing CSV header in {requirement.label}: {path}"]

    columns = set(header)
    missing = sorted(requirement.required_columns - columns)
    errors = []
    if missing:
        errors.append(
            f"Invalid {requirement.label}; missing columns {missing}: {path}"
        )
    if requirement.canonical_index and header[:2] != ["delivery_date", "mtu"]:
        errors.append(
            f"Invalid {requirement.label}; first columns must be "
            f"delivery_date,mtu: {path}"
        )
    if "forecast_day" in columns and not (
        {"runtime_seconds", "computation_time_seconds"} & columns
    ):
        errors.append(
            f"Invalid {requirement.label}; missing runtime-seconds column: {path}"
        )
    return errors


def _preflight(
    config: ExperimentConfig,
    stages: set[str],
    *,
    include_anc: bool = False,
) -> list[str]:
    """Return all obvious missing/incompatible post-processing inputs."""
    errors: list[str] = []
    for requirement in _requirements(config, stages):
        errors.extend(_validate_csv(requirement))
    if include_anc:
        api_key = os.getenv("ENTSOE_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            errors.append("ENTSOE_API_KEY is missing from .env (required by ANC).")
        for year in range(
            config.weather_training_start.year,
            config.evaluation_end.year + 1,
        ):
            directory = config.era5_year_dir(5, year)
            if not directory.is_dir() or not any(directory.glob("*.csv")):
                errors.append(f"Missing ERA5 C=5, year={year} for ANC: {directory}")
    return errors


def _run_evaluation(config: ExperimentConfig) -> None:
    _run_module(config, "evaluation", "evaluation", "evaluation.run_evaluation")


def _run_anc(config: ExperimentConfig) -> None:
    print("ANC: ERA5 C=5, D=112, fundamental and EXAA-enriched importance", flush=True)
    for variant in ("fundamental", "exaa"):
        _run_module(
            config,
            "anc",
            variant,
            "pipeline.lear.run_lear",
            "anc",
            "--variant",
            variant,
        )


def _anc_outputs_exist(config: ExperimentConfig) -> bool:
    anc_base = config.results_root / "lear_anc_results" / "era5" / "c5" / "d112"
    return (
        (anc_base / "fundamental" / "anc_all_feature_results.csv").is_file()
        and (anc_base / "exaa" / "anc_all_feature_results.csv").is_file()
    )


def _plot_scripts(
    config: ExperimentConfig,
    *,
    include_anc: bool = False,
) -> list[str]:
    scripts = list(PLOT_SCRIPTS)
    if include_anc or _anc_outputs_exist(config):
        scripts.append("plot_anc_bar.py")
    return scripts


def _run_plots(config: ExperimentConfig) -> None:
    for script in _plot_scripts(config):
        _run_logged_command(
            [_execution_python(), REPO_ROOT / "visualization" / script],
            config,
            stage="plots",
            name=Path(script).stem,
        )


def _parse_stages(value: str, parser: argparse.ArgumentParser) -> set[str]:
    stages = {part.strip().lower() for part in value.split(",") if part.strip()}
    if not stages:
        parser.error("At least one stage is required.")
    unknown = stages - set(VALID_STAGES)
    if unknown:
        parser.error(f"Unknown stages: {', '.join(sorted(unknown))}")
    return stages


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="evaluation,plots",
        help="Comma-separated post-processing stages: evaluation,plots",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only check that all selected-stage result inputs are present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the post-processing plan without validating or executing it.",
    )
    parser.add_argument(
        "--anc",
        action="store_true",
        help=(
            "Run both ERA5 C=5, D=112 ANC variants before the selected "
            "evaluation/plot stages."
        ),
    )
    args = parser.parse_args(argv)

    config = load_experiment_config(REPO_ROOT)
    stages = _parse_stages(args.stages, parser)
    print(_date_summary(config))
    print(f"Execution Python={_execution_python()}")

    if args.dry_run:
        if args.anc:
            print("ANC   ERA5 C=5 D=112 fundamental + EXAA-enriched")
        if "evaluation" in stages:
            print("EVAL  28 point models + 6 SQRA models")
        if "plots" in stages:
            for script in _plot_scripts(config, include_anc=args.anc):
                print(f"PLOT  visualization/{script}")
        return 0

    errors = _preflight(config, stages, include_anc=args.anc)
    if errors:
        print("Preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    if args.preflight:
        print(
            f"Preflight passed ({len(_requirements(config, stages))} result/input files)."
        )
        return 0

    if args.anc:
        _run_anc(config)
    if "evaluation" in stages:
        _run_evaluation(config)
    if "plots" in stages:
        _run_plots(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
