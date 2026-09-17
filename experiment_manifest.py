"""Canonical LEAR and SQRA experiment definitions used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LearRun:
    name: str
    weather_source: str | None
    use_exaa: bool
    use_exaa_only: bool
    train_days: int
    clusters: int | None
    lars_start_date: str

    def output_dir(self, results_root: Path) -> Path:
        base = results_root / "lear_op_results"
        if self.use_exaa_only:
            return base / "exaa_only" / f"d{self.train_days}"
        variant = "exaa" if self.use_exaa else "fundamental"
        return (
            base
            / str(self.weather_source).lower()
            / f"d{self.train_days}"
            / f"c{self.clusters}"
            / variant
        )


def lear_runs() -> tuple[LearRun, ...]:
    runs: list[LearRun] = []
    for weather, days_values in (("ERA5", (56, 112, 364)), ("DWD", (56,))):
        for days in days_values:
            for clusters in (1, 5, 25):
                for use_exaa in (False, True):
                    variant = "exaa" if use_exaa else "fundamental"
                    runs.append(
                        LearRun(
                            name=f"{weather.lower()}_d{days}_c{clusters}_{variant}",
                            weather_source=weather,
                            use_exaa=use_exaa,
                            use_exaa_only=False,
                            train_days=days,
                            clusters=clusters,
                            # Preserve the solver rule recorded by the submitted runs.
                            lars_start_date="2026-12-01" if days == 364 else "2025-12-01",
                        )
                    )
    for days in (56, 112, 364):
        runs.append(
            LearRun(
                name=f"exaa_only_d{days}",
                weather_source=None,
                use_exaa=True,
                use_exaa_only=True,
                train_days=days,
                clusters=None,
                lars_start_date="2026-12-01" if days == 364 else "2025-12-01",
            )
        )
    return tuple(runs)


def get_lear_run(name: str) -> LearRun:
    """Return one canonical LEAR experiment by its command-line name."""
    for run in lear_runs():
        if run.name == name:
            return run
    choices = ", ".join(run.name for run in lear_runs())
    raise ValueError(f"Unknown LEAR configuration {name!r}. Choose one of: {choices}")


@dataclass(frozen=True)
class SqraRun:
    name: str
    member_paths: tuple[str, ...]

    def inputs(self, results_root: Path) -> tuple[Path, ...]:
        return tuple(results_root / path for path in self.member_paths)

    def output_dir(self, results_root: Path) -> Path:
        return results_root / "sqra_results" / self.name


SQRA_RUNS: dict[str, SqraRun] = {
    "era5_fundamental": SqraRun(
        "era5_fundamental",
        (
            "lear_op_results/era5/d364/c1/fundamental/forecast.csv",
            "lear_op_results/era5/d364/c5/fundamental/forecast.csv",
            "lear_op_results/era5/d364/c25/fundamental/forecast.csv",
        ),
    ),
    "dwd_fundamental": SqraRun(
        "dwd_fundamental",
        (
            "lear_op_results/dwd/d56/c1/fundamental/forecast.csv",
            "lear_op_results/dwd/d56/c5/fundamental/forecast.csv",
            "lear_op_results/dwd/d56/c25/fundamental/forecast.csv",
        ),
    ),
    "era5_exaa_enriched": SqraRun(
        "era5_exaa_enriched",
        (
            "lear_op_results/era5/d364/c1/exaa/forecast.csv",
            "lear_op_results/era5/d364/c5/exaa/forecast.csv",
            "lear_op_results/era5/d364/c25/exaa/forecast.csv",
        ),
    ),
    "dwd_exaa_enriched": SqraRun(
        "dwd_exaa_enriched",
        (
            "lear_op_results/dwd/d56/c1/exaa/forecast.csv",
            "lear_op_results/dwd/d56/c5/exaa/forecast.csv",
            "lear_op_results/dwd/d56/c25/exaa/forecast.csv",
        ),
    ),
    "exaa_naive": SqraRun(
        "exaa_naive", ("lear_op_results/exaa_naive/forecast.csv",)
    ),
    "exaa_only": SqraRun(
        "exaa_only",
        (
            "lear_op_results/exaa_only/d56/forecast.csv",
            "lear_op_results/exaa_only/d112/forecast.csv",
            "lear_op_results/exaa_only/d364/forecast.csv",
        ),
    ),
}


def get_sqra_run(name: str) -> SqraRun:
    try:
        return SQRA_RUNS[name]
    except KeyError as exc:
        choices = ", ".join(SQRA_RUNS)
        raise ValueError(f"Unknown SQRA_CONFIG_NAME={name!r}. Choose one of: {choices}") from exc
