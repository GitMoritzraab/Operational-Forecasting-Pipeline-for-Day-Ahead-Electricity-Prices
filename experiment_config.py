"""Shared, environment-driven configuration for the full experiment pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


def _find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".env.example").exists() and (candidate / "pipeline").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current directory.")


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got {value!r}.")


def env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def env_date(name: str, default: str) -> date:
    value = os.getenv(name, default)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD, got {value!r}.") from exc


def env_dates(name: str, default: str = "") -> tuple[date, ...]:
    values = [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]
    try:
        return tuple(sorted({date.fromisoformat(value) for value in values}))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of YYYY-MM-DD dates.") from exc


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    evaluation_start: date
    evaluation_end: date
    timezone: str
    sqra_train_days: int
    sqra_mtu_specific: bool
    lear_use_vst: bool
    max_lear_train_days: int
    price_lag_days: int
    point_history_buffer_days: int
    evaluation_skip_dates: tuple[date, ...]
    forecast_skip_dates: tuple[date, ...]
    dwd_folder_skip_dates: tuple[date, ...]
    dwd_preprocess_start_date: date
    results_root: Path
    output_root: Path
    entsoe_price_cache_dir: Path
    exaa_price_cache_dir: Path
    entsoe_load_cache_dir: Path
    era5_data_root: Path
    icon_data_root: Path
    refresh_market_cache: bool

    @property
    def point_forecast_start(self) -> date:
        """First point-forecast day, including SQRA calibration history."""
        return self.evaluation_start - timedelta(
            days=self.sqra_train_days + self.point_history_buffer_days
        )

    @property
    def input_start(self) -> date:
        """Earliest market input needed by the longest LEAR window and price lags."""
        return self.point_forecast_start - timedelta(
            days=self.max_lear_train_days + self.price_lag_days
        )

    @property
    def weather_training_start(self) -> date:
        return self.point_forecast_start - timedelta(days=self.max_lear_train_days)

    def era5_year_dir(self, clusters: int, year: int) -> Path:
        return self.era5_data_root / f"c{clusters}" / str(year)

    def icon_cluster_dir(self, clusters: int) -> Path:
        return self.icon_data_root / f"c{clusters}"


def load_experiment_config(repo_root: Path | str | None = None) -> ExperimentConfig:
    root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
    _load_env_file(root / ".env")

    evaluation_start = env_date("EVALUATION_START_DATE", "2025-12-01")
    evaluation_end = env_date("EVALUATION_END_DATE", "2026-07-31")
    if evaluation_end < evaluation_start:
        raise ValueError("EVALUATION_END_DATE must not be before EVALUATION_START_DATE.")

    sqra_days = env_int("SQRA_TRAIN_DAYS", 60)
    max_lear_days = env_int("MAX_LEAR_TRAIN_DAYS", 364)
    lag_days = env_int("PRICE_LAG_DAYS", 7)
    buffer_days = env_int("POINT_HISTORY_BUFFER_DAYS", 1)

    return ExperimentConfig(
        repo_root=root,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        timezone=os.getenv("TARGET_TIMEZONE", "Europe/Berlin"),
        sqra_train_days=sqra_days,
        sqra_mtu_specific=env_bool("SQRA_MTU_SPECIFIC", False),
        lear_use_vst=env_bool("LEAR_USE_VST", True),
        max_lear_train_days=max_lear_days,
        price_lag_days=lag_days,
        point_history_buffer_days=buffer_days,
        evaluation_skip_dates=env_dates(
            "EVALUATION_SKIP_DATES", "2026-01-22,2026-06-12"
        ),
        forecast_skip_dates=env_dates(
            "FORECAST_SKIP_DATES", "2026-01-22,2026-06-12"
        ),
        dwd_folder_skip_dates=env_dates(
            "DWD_FOLDER_SKIP_DATES", "2025-10-26,2025-10-27,2025-10-28"
        ),
        dwd_preprocess_start_date=env_date(
            "DWD_PREPROCESS_START_DATE", "2025-08-01"
        ),
        results_root=_resolve(
            root,
            os.getenv("RESULTS_ROOT", "results_extended_2025-12-01_2026-07-31"),
        ),
        output_root=_resolve(
            root,
            os.getenv("OUTPUT_ROOT", "output_extended_2025-12-01_2026-07-31"),
        ),
        entsoe_price_cache_dir=_resolve(
            root,
            os.getenv(
                "ENTSOE_DE_LU_CACHE_DIR",
                os.getenv("MARKET_DATA_CACHE_DIR", "data/cache/entsoe"),
            ),
        ),
        exaa_price_cache_dir=_resolve(
            root,
            os.getenv(
                "EXAA_CACHE_DIR",
                os.getenv("MARKET_DATA_CACHE_DIR", "data/cache/entsoe"),
            ),
        ),
        entsoe_load_cache_dir=_resolve(
            root,
            os.getenv(
                "ENTSOE_LOAD_FORECAST_CACHE_DIR",
                os.getenv("MARKET_DATA_CACHE_DIR", "data/cache/entsoe"),
            ),
        ),
        era5_data_root=_resolve(root, os.getenv("ERA5_DATA_ROOT", "data/era5")),
        icon_data_root=_resolve(root, os.getenv("ICON_DATA_ROOT", "data/icon")),
        refresh_market_cache=env_bool("REFRESH_MARKET_DATA", False),
    )
