"""Environment-driven configuration for the daily operational pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from experiment_config import env_bool, env_int, load_experiment_config


MODEL_VARIANTS = ("fundamental", "exaa_enriched", "exaa_only")
SQRA_VARIANTS = ("auto", *MODEL_VARIANTS)
WEATHER_CLUSTERS = (1, 5, 25)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower().replace("-", "_")
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")
    return value


def _optional_path(root: Path, name: str) -> Optional[Path]:
    value = os.getenv(name, "").strip()
    return _resolve(root, value) if value else None


def _download_exaa_mode() -> str:
    value = os.getenv("DOWNLOAD_EXAA", "auto").strip().lower()
    if value not in {"auto", "true", "false"}:
        raise ValueError("DOWNLOAD_EXAA must be auto, true, or false.")
    return value


@dataclass(frozen=True)
class OperationalConfig:
    repo_root: Path
    timezone: str
    entsoe_api_key: str
    arena_api_base_url: str
    arena_point_challenge_id: str
    arena_quantile_challenge_id: str
    submit_to_arena: bool
    point_variant: str
    point_clusters: int
    sqra_variant: str
    sqra_train_days: int
    sqra_mtu_specific: bool
    lear_use_vst: bool
    download_dwd: bool
    delete_dwd_raw_after_preprocess: bool
    download_exaa: str
    dwd_open_data_url: str
    dwd_operational_run_hour: str
    dwd_history_run_hour: str
    allow_dwd_history_fallback: bool
    dwd_download_attempts: int
    dwd_retry_seconds: int
    dwd_request_timeout_seconds: int
    dwd_raw_archive: Path
    icon_data_root: Path
    entsoe_price_cache_dir: Path
    exaa_price_cache_dir: Path
    entsoe_load_cache_dir: Path
    results_root: Path
    output_root: Path

    @property
    def resolved_sqra_variant(self) -> str:
        return self.point_variant if self.sqra_variant == "auto" else self.sqra_variant

    @property
    def needs_dwd(self) -> bool:
        return self.point_variant != "exaa_only" or self.resolved_sqra_variant != "exaa_only"

    @property
    def needs_exaa(self) -> bool:
        return (
            self.point_variant in {"exaa_enriched", "exaa_only"}
            or self.resolved_sqra_variant in {"exaa_enriched", "exaa_only"}
        )

    @property
    def needs_load(self) -> bool:
        return self.needs_dwd

    @property
    def required_dwd_clusters(self) -> tuple[int, ...]:
        clusters = {self.point_clusters}
        if self.resolved_sqra_variant in {"fundamental", "exaa_enriched"}:
            clusters.update({1, 5, 25})
        return tuple(sorted(clusters))

    def icon_cluster_dir(self, clusters: int) -> Path:
        return self.icon_data_root / f"c{clusters}"


def load_operational_config(repo_root: Path | str | None = None) -> OperationalConfig:
    """Load daily-pipeline settings after loading the repository's ``.env``."""
    root = Path(repo_root or Path.cwd()).resolve()
    historical = load_experiment_config(root)

    point_variant = _choice("POINT_FORECAST_VARIANT", "fundamental", MODEL_VARIANTS)
    sqra_variant = _choice("SQRA_FORECAST_VARIANT", "auto", SQRA_VARIANTS)
    point_clusters = env_int("POINT_WEATHER_CLUSTERS", 5)
    if point_clusters not in WEATHER_CLUSTERS:
        raise ValueError("POINT_WEATHER_CLUSTERS must be 1, 5, or 25.")

    # Operational downloads are intentionally separate from the long-term
    # historical raw archive.  The fallback preserves compatibility with old
    # .env files that only define DWD_RAW_ARCHIVE.
    raw_archive = _optional_path(root, "DWD_OPERATIONAL_RAW_ROOT")
    if raw_archive is None:
        raw_archive = _optional_path(root, "DWD_RAW_ARCHIVE")
    if raw_archive is None:
        raw_archive = root / "data" / "dwd_raw"

    market_root = _optional_path(root, "OPERATIONAL_MARKET_DATA_ROOT")
    if market_root is None:
        market_root = root / "data" / "market"
    operational_price_cache = _optional_path(
        root, "OPERATIONAL_ENTSOE_PRICE_CACHE_DIR"
    ) or (market_root / "entsoe")
    operational_load_cache = _optional_path(
        root, "OPERATIONAL_ENTSOE_LOAD_CACHE_DIR"
    ) or (market_root / "entsoe")
    operational_exaa_cache = _optional_path(
        root, "OPERATIONAL_EXAA_CACHE_DIR"
    ) or (market_root / "exaa")

    attempts = env_int("DWD_DOWNLOAD_MAX_ATTEMPTS", 5)
    if attempts < 1:
        raise ValueError("DWD_DOWNLOAD_MAX_ATTEMPTS must be at least 1.")

    return OperationalConfig(
        repo_root=root,
        timezone=os.getenv("TARGET_TIMEZONE", "Europe/Berlin").strip(),
        entsoe_api_key=os.getenv("ENTSOE_API_KEY", "").strip(),
        arena_api_base_url=os.getenv(
            "ENERGY_ARENA_API_BASE_URL", "https://api.energy-arena.org"
        ).strip().rstrip("/"),
        arena_point_challenge_id=os.getenv(
            "ENERGY_ARENA_POINT_CHALLENGE_ID", "2"
        ).strip(),
        arena_quantile_challenge_id=os.getenv(
            "ENERGY_ARENA_QUANTILE_CHALLENGE_ID", "8"
        ).strip(),
        submit_to_arena=env_bool("SUBMIT_TO_ENERGY_ARENA", True),
        point_variant=point_variant,
        point_clusters=point_clusters,
        sqra_variant=sqra_variant,
        sqra_train_days=env_int("SQRA_TRAIN_DAYS", 60),
        sqra_mtu_specific=env_bool("SQRA_MTU_SPECIFIC", False),
        lear_use_vst=env_bool("LEAR_USE_VST", True),
        download_dwd=env_bool("DOWNLOAD_DWD", True),
        delete_dwd_raw_after_preprocess=env_bool(
            "DELETE_DWD_RAW_AFTER_PREPROCESS", True
        ),
        download_exaa=_download_exaa_mode(),
        dwd_open_data_url=os.getenv(
            "DWD_OPEN_DATA_URL",
            "https://opendata.dwd.de/weather/nwp/icon-d2/grib",
        ).strip().rstrip("/"),
        dwd_operational_run_hour=os.getenv("DWD_OPERATIONAL_RUN_HOUR", "06").zfill(2),
        dwd_history_run_hour=os.getenv("DWD_HISTORY_FALLBACK_RUN_HOUR", "09").zfill(2),
        allow_dwd_history_fallback=env_bool("ALLOW_DWD_HISTORY_RUN_FALLBACK", True),
        dwd_download_attempts=attempts,
        dwd_retry_seconds=env_int("DWD_DOWNLOAD_RETRY_SECONDS", 30),
        dwd_request_timeout_seconds=env_int("DWD_REQUEST_TIMEOUT_SECONDS", 60),
        dwd_raw_archive=raw_archive,
        icon_data_root=historical.icon_data_root,
        entsoe_price_cache_dir=operational_price_cache,
        exaa_price_cache_dir=operational_exaa_cache,
        entsoe_load_cache_dir=operational_load_cache,
        results_root=_resolve(
            root,
            os.getenv("OPERATIONAL_RESULTS_ROOT", "data/operational/results"),
        ),
        output_root=_resolve(
            root,
            os.getenv("OPERATIONAL_OUTPUT_ROOT", "data/operational/output"),
        ),
    )
