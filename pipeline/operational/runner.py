"""End-to-end daily forecasting workflow used by :mod:`run_pipeline`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from contextlib import ExitStack
from dataclasses import replace
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from experiment_manifest import LearRun, get_lear_run, get_sqra_run
from market_cache import load_or_fetch_frame
from pipeline.delivery_index import (
    delivery_dates,
    read_forecast_csv,
    schema_metadata,
    validate_delivery_index,
)
from pipeline.lear.lear_model import (
    build_dwd_features,
    build_load_features,
    build_price_features,
    build_temporal_features,
    build_y_matrix,
    compute_metrics,
    fetch_load_forecast,
    fetch_prices,
    fetch_prices_exaa,
    load_dwd,
    merge_all_features,
    rolling_point_forecast,
)
from pipeline.operational.config import (
    MODEL_VARIANTS,
    OperationalConfig,
    load_operational_config,
)
from pipeline.operational.prepare_dwd import prepare_target_dwd
from pipeline.operational.locking import interprocess_lock
from pipeline.operational.energy_arena import (
    ArenaTargets,
    atomic_write_json,
    build_point_payload,
    build_quantile_payload,
    resolve_targets,
    submission_record_path,
    submit_with_record,
)
from pipeline.sqra.sqra_model import build_sqra_panel, generate_sqra_forecast


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _redacted_command(arguments: Sequence[str]) -> str:
    # Secrets are read exclusively from .env and therefore never appear in the
    # command line that is persisted in the operational log.
    return " ".join(
        [sys.executable, str(Path(sys.argv[0]).name), *arguments]
    )


def _point_name(variant: str, clusters: int) -> str:
    if variant == "exaa_only":
        return "exaa_only_d364"
    suffix = "fundamental" if variant == "fundamental" else "exaa"
    return f"dwd_d56_c{clusters}_{suffix}"


def _sqra_name(variant: str) -> str:
    return {
        "fundamental": "dwd_fundamental",
        "exaa_enriched": "dwd_exaa_enriched",
        "exaa_only": "exaa_only",
    }[variant]


def resolve_model_plan(
    config: OperationalConfig,
) -> tuple[tuple[LearRun, ...], LearRun, str, frozenset[str]]:
    """Return runs, submitted point run, SQRA name, and SQRA member names."""
    sqra_name = _sqra_name(config.resolved_sqra_variant)
    sqra_spec = get_sqra_run(sqra_name)
    member_names: list[str] = []
    for member in sqra_spec.member_paths:
        parts = Path(member).parts
        if sqra_name.startswith("dwd_"):
            variant = "fundamental" if parts[-2] == "fundamental" else "exaa"
            name = f"dwd_d56_{parts[-3]}_{variant}"
        else:
            name = f"exaa_only_{parts[-2]}"
        if name not in member_names:
            member_names.append(name)

    point_run = get_lear_run(_point_name(config.point_variant, config.point_clusters))
    run_names = [*member_names]
    if point_run.name not in run_names:
        run_names.append(point_run.name)
    runs = tuple(get_lear_run(name) for name in run_names)
    return runs, point_run, sqra_name, frozenset(member_names)


def _lock_token(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def pipeline_lock_paths(config: OperationalConfig) -> tuple[Path, ...]:
    """Return locks for exactly the outputs touched by one pipeline plan.

    Different model families may therefore train in parallel. Runs which would
    update the same LEAR/SQRA history remain mutually exclusive. The common
    Energy Arena account is locked separately only during submission.
    """
    runs, _, sqra_name, _ = resolve_model_plan(config)
    resources = {
        *(f"lear_{run.name}" for run in runs),
        f"sqra_{sqra_name}",
    }
    lock_root = config.output_root / "locks"
    return tuple(
        lock_root / f"{_lock_token(resource)}.lock"
        for resource in sorted(resources)
    )


def arena_lock_path(config: OperationalConfig) -> Path:
    """Return the single submission lock for the configured Arena account."""
    return config.output_root / "locks" / "arena_default.lock"


def _submission_stream(config: OperationalConfig) -> str:
    """Name the stable local artifacts for one information-cutoff stream."""
    # Preserve the existing Fundamental output paths while keeping the later
    # EXAA submission's local payload and receipt separate.
    return (
        "default"
        if config.point_variant == "fundamental"
        else config.point_variant
    )


def pipeline_log_path(config: OperationalConfig) -> Path:
    """Return the replace-on-each-run log for one operational model variant."""
    return config.output_root / "logs" / f"{_lock_token(config.point_variant)}.log"


def _timestamp(value: date, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=timezone)


def _fetch_retry(label: str, operation, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(f"{label} attempt {attempt} failed: {exc}; retrying", flush=True)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}")


def _read_cache_only(path: Path, start: date, end: date, timezone: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required cache does not exist: {path}")
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(timezone)
    start_ts = _timestamp(start, timezone)
    end_exclusive = _timestamp(end + timedelta(days=1), timezone)
    expected = pd.date_range(start_ts, end_exclusive, freq="15min", inclusive="left")
    selected = frame.reindex(expected)
    if selected.empty or selected.isna().to_numpy().any():
        raise ValueError(
            f"Cache {path} is incomplete for {start.isoformat()} through {end.isoformat()}."
        )
    selected.index.name = "timestamp"
    return selected


def load_market_data(
    config: OperationalConfig,
    *,
    start_date: date,
    target_date: date,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load exactly the market inputs available before the target is realized."""
    if not config.entsoe_api_key or config.entsoe_api_key == "your_api_key_here":
        raise ValueError("ENTSOE_API_KEY is missing from .env.")
    start = _timestamp(start_date, config.timezone)
    realized_end_date = target_date - timedelta(days=1)
    realized_end = _timestamp(realized_end_date, config.timezone)
    target = _timestamp(target_date, config.timezone)

    prices = _fetch_retry(
        "ENTSO-E prices",
        lambda: load_or_fetch_frame(
            config.entsoe_price_cache_dir / "prices_da.csv",
            start,
            realized_end,
            partial(
                fetch_prices,
                api_key=config.entsoe_api_key,
                target_tz=config.timezone,
            ),
            timezone=config.timezone,
            allow_incomplete=True,
        ),
    )

    exaa: Optional[pd.DataFrame] = None
    should_load_exaa = config.needs_exaa or config.download_exaa == "true"
    if should_load_exaa:
        exaa_path = config.exaa_price_cache_dir / "prices_exaa.csv"
        if config.download_exaa == "false":
            exaa = _read_cache_only(
                exaa_path, start_date, target_date, config.timezone
            )
        else:
            exaa = _fetch_retry(
                "EXAA prices",
                lambda: load_or_fetch_frame(
                    exaa_path,
                    start,
                    target,
                    partial(
                        fetch_prices_exaa,
                        api_key=config.entsoe_api_key,
                        target_tz=config.timezone,
                    ),
                    timezone=config.timezone,
                ),
            )

    load: Optional[pd.DataFrame] = None
    if config.needs_load:
        load = _fetch_retry(
            "ENTSO-E load forecast",
            lambda: load_or_fetch_frame(
                config.entsoe_load_cache_dir / "load_forecast.csv",
                start,
                target,
                partial(
                    fetch_load_forecast,
                    api_key=config.entsoe_api_key,
                    target_tz=config.timezone,
                ),
                timezone=config.timezone,
            ),
        )
    return prices, exaa, load


def _load_weather_features(
    config: OperationalConfig,
    clusters: int,
    first_delivery_date: date,
    target_date: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    run_hours = [config.dwd_history_run_hour, config.dwd_operational_run_hour]
    if not config.allow_dwd_history_fallback:
        run_hours = [config.dwd_operational_run_hour]
    for run_hour in dict.fromkeys(run_hours):
        try:
            hourly, quarter_hourly = load_dwd(
                icon_dir=config.icon_cluster_dir(clusters),
                start_folder_date=first_delivery_date - timedelta(days=1),
                end_folder_date=target_date - timedelta(days=1),
                required_run=run_hour,
                skip_dates=set(),
                target_tz=config.timezone,
            )
        except ValueError as exc:
            if "No hourly DWD data loaded" in str(exc):
                continue
            raise
        frames.append(
            build_dwd_features(hourly, quarter_hourly, target_tz=config.timezone)
        )
    if not frames:
        raise ValueError(f"No processed DWD features found for C={clusters}.")
    combined = pd.concat(frames).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined


def _atomic_frame(path: Path, frame: pd.DataFrame, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _requested_days(
    target_date: date,
    train_days: int,
    is_sqra_member: bool,
) -> pd.DatetimeIndex:
    first = target_date - timedelta(days=train_days if is_sqra_member else 0)
    return pd.date_range(first, target_date, freq="D")


def _assemble_matrices(
    config: OperationalConfig,
    run: LearRun,
    *,
    prices: pd.DataFrame,
    exaa: Optional[pd.DataFrame],
    load: Optional[pd.DataFrame],
    weather: dict[int, pd.DataFrame],
    first_feature_date: date,
    target_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_index = pd.date_range(
        first_feature_date,
        target_date,
        freq="D",
        tz=config.timezone,
    )
    if run.use_exaa_only:
        if exaa is None:
            raise ValueError(f"{run.name} requires EXAA prices.")
        features = build_price_features(
            prices,
            df_prices_exaa_15=exaa,
            exaa_only=True,
            daily_index=daily_index,
        ).reindex(daily_index)
    else:
        if run.clusters is None or run.clusters not in weather:
            raise ValueError(f"No DWD features loaded for {run.name}.")
        if load is None:
            raise ValueError(f"{run.name} requires target-day load forecasts.")
        if run.use_exaa and exaa is None:
            raise ValueError(f"{run.name} requires EXAA prices.")
        features, dropped = merge_all_features(
            weather[run.clusters],
            build_price_features(
                prices,
                df_prices_exaa_15=exaa if run.use_exaa else None,
                exaa_vector=run.use_exaa,
                daily_index=daily_index,
            ),
            build_load_features(load),
            build_temporal_features(daily_index),
            dropna=False,
        )
        if not dropped.empty:
            relevant = dropped.loc[
                pd.to_datetime(dropped["date"]).dt.date.between(
                    first_feature_date, target_date
                )
            ]
            if not relevant.empty:
                print(
                    f"{run.name}: incomplete feature days detected="
                    f"{len(relevant)}; operational price policy will be applied",
                    flush=True,
                )
    target = build_y_matrix(prices, features.index)
    return features, target


def _validate_training(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: pd.DatetimeIndex,
    train_days: int,
    *,
    allow_incomplete_prices: bool = False,
) -> None:
    for raw_day in forecast_days:
        day = raw_day if raw_day.tzinfo else raw_day.tz_localize(X.index.tz)
        start = day - pd.DateOffset(days=train_days)
        mask = (X.index >= start) & (X.index < day)
        available = X.index[mask]
        if len(available) != train_days:
            expected = pd.date_range(start, day - pd.DateOffset(days=1), freq="D")
            missing = [str(value.date()) for value in expected.difference(available)]
            raise ValueError(
                f"Incomplete {train_days}-day LEAR window for {day.date()}; "
                f"missing feature days: {missing}."
            )
        if not allow_incomplete_prices:
            if not np.isfinite(Y.loc[mask].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Training targets contain missing values before {day.date()}."
                )
            continue

        if day not in X.index:
            raise ValueError(f"No feature row is available for {day.date()}.")
        test_row = X.loc[[day]]
        unavailable = [
            column
            for column in X.columns
            if not np.isfinite(test_row[column].to_numpy(dtype=float)).all()
        ]
        unsupported = [
            column for column in unavailable if not column.startswith("price_d")
        ]
        if unsupported:
            raise ValueError(
                f"Forecast features contain unavailable non-price inputs for "
                f"{day.date()}: {unsupported}."
            )
        active_columns = [column for column in X.columns if column not in unavailable]
        if not active_columns:
            raise ValueError(f"No usable forecast features remain for {day.date()}.")
        feature_valid = np.isfinite(
            X.loc[mask, active_columns].to_numpy(dtype=float)
        ).all(axis=1)
        target_valid = np.isfinite(Y.loc[mask].to_numpy(dtype=float))
        valid_counts = (target_valid & feature_valid[:, None]).sum(axis=0)
        if int(valid_counts.min()) < 5:
            raise ValueError(
                f"Fewer than five complete training observations remain before "
                f"{day.date()}."
            )


def _backfill_truth(frame: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for day in delivery_dates(result.index).unique():
        localized = pd.Timestamp(day, tz=target.index.tz)
        if localized not in target.index:
            continue
        values = target.loc[localized].to_numpy(dtype=float)
        result.loc[day, "y_true"] = values
    return result


def _normalize_runtime_days(runtime: pd.DataFrame) -> pd.DataFrame:
    result = runtime.copy()
    if not result.empty and "forecast_day" in result.columns:
        result["forecast_day"] = pd.to_datetime(
            result["forecast_day"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    return result


def run_point_model(
    config: OperationalConfig,
    run: LearRun,
    *,
    requested_days: pd.DatetimeIndex,
    prices: pd.DataFrame,
    exaa: Optional[pd.DataFrame],
    load: Optional[pd.DataFrame],
    weather: dict[int, pd.DataFrame],
    force: bool,
) -> pd.DataFrame:
    first_forecast = requested_days.min().date()
    first_feature = first_forecast - timedelta(days=run.train_days)
    X, Y = _assemble_matrices(
        config,
        run,
        prices=prices,
        exaa=exaa,
        load=load,
        weather=weather,
        first_feature_date=first_feature,
        target_date=requested_days.max().date(),
    )
    output_dir = run.output_dir(config.results_root)
    forecast_path = output_dir / "forecast.csv"
    existing: Optional[pd.DataFrame] = None
    if forecast_path.is_file():
        existing = read_forecast_csv(
            forecast_path,
            required_columns=("y_pred", "y_true"),
            require_complete_days=True,
        )
        existing = _backfill_truth(existing, Y)

    canonical_days = requested_days.tz_localize(None).normalize()
    existing_days = (
        set(delivery_dates(existing.index).unique()) if existing is not None else set()
    )
    missing_days = [
        day
        for day, canonical in zip(requested_days, canonical_days)
        if force or canonical not in existing_days
    ]
    runtime = pd.DataFrame()
    if missing_days:
        _validate_training(
            X,
            Y,
            pd.DatetimeIndex(missing_days),
            run.train_days,
            allow_incomplete_prices=True,
        )
        lars_start = (
            pd.Timestamp("1900-01-01", tz=config.timezone)
            if run.train_days in {56, 112}
            else pd.Timestamp("2200-01-01", tz=config.timezone)
        )
        generated, runtime, _, _, _ = rolling_point_forecast(
            X=X,
            Y=Y,
            forecast_days=missing_days,
            train_days=run.train_days,
            lars_start_date=lars_start,
            use_vst=config.lear_use_vst,
            allow_incomplete_prices=True,
        )
        combined = generated if existing is None else pd.concat([existing, generated])
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    else:
        assert existing is not None
        combined = existing

    requested_index = combined.loc[
        combined.index.get_level_values("delivery_date").isin(canonical_days)
    ]
    if requested_index.index.get_level_values("delivery_date").nunique() != len(canonical_days):
        raise RuntimeError(f"{run.name} did not produce every requested forecast day.")
    if not np.isfinite(requested_index["y_pred"]).all():
        raise RuntimeError(f"{run.name} produced NaN or infinite predictions.")

    _atomic_frame(forecast_path, combined, index=True)
    if not runtime.empty:
        runtime = _normalize_runtime_days(runtime)
        runtime_path = output_dir / "runtime.csv"
        if runtime_path.is_file():
            previous_runtime = _normalize_runtime_days(pd.read_csv(runtime_path))
            runtime = pd.concat([previous_runtime, runtime], ignore_index=True)
            runtime = runtime.drop_duplicates("forecast_day", keep="last")
        _atomic_frame(runtime_path, runtime, index=False)
    realized = combined.loc[combined["y_true"].notna()]
    if not realized.empty:
        _atomic_frame(
            output_dir / "metrics.csv",
            pd.DataFrame([compute_metrics(realized, "realized")]),
            index=False,
        )
    metadata = {
        "mode": "daily_operational",
        "model": run.name,
        "train_days": run.train_days,
        "weather_source": run.weather_source,
        "n_clusters": run.clusters,
        "use_exaa": run.use_exaa,
        "use_exaa_only": run.use_exaa_only,
        "use_vst": config.lear_use_vst,
        "dwd_operational_run_hour": config.dwd_operational_run_hour,
        "dwd_history_run_hour": config.dwd_history_run_hour,
        "missing_price_policy": (
            "skip incomplete training rows and suppress unavailable target-day "
            "EPEX lag columns"
        ),
        **schema_metadata(),
    }
    atomic_write_json(output_dir / "config.json", metadata)
    return combined


def run_sqra_model(
    config: OperationalConfig,
    sqra_name: str,
    target_date: date,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    specification = get_sqra_run(sqra_name)
    panel, feature_columns = build_sqra_panel(
        specification.inputs(config.results_root), config.timezone
    )
    required_days = pd.date_range(
        target_date - timedelta(days=config.sqra_train_days),
        target_date,
        freq="D",
    )
    panel_days = delivery_dates(panel.index).unique()
    missing = required_days.difference(panel_days)
    if len(missing):
        raise ValueError(
            "SQRA calibration panel is missing delivery days: "
            + ", ".join(str(day.date()) for day in missing)
        )
    forecast, runtime = generate_sqra_forecast(
        df=panel,
        forecast_days=[pd.Timestamp(target_date)],
        train_days=config.sqra_train_days,
        quantiles=quantiles,
        feature_cols=feature_columns,
        mtu_specific=config.sqra_mtu_specific,
    )
    validate_delivery_index(forecast.index, require_complete_days=True)
    output_dir = specification.output_dir(config.results_root)
    forecast_path = output_dir / "forecast.csv"
    if forecast_path.is_file():
        previous = read_forecast_csv(forecast_path, require_complete_days=True)
        common_truth = previous.index.intersection(panel.index)
        if len(common_truth) and "y_true" in previous.columns:
            previous.loc[common_truth, "y_true"] = panel.loc[
                common_truth, "y_true"
            ].to_numpy()
        combined = pd.concat([previous, forecast])
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = forecast
    _atomic_frame(forecast_path, combined, index=True)
    runtime = _normalize_runtime_days(runtime)
    runtime_path = output_dir / "runtime.csv"
    if runtime_path.is_file():
        previous_runtime = _normalize_runtime_days(pd.read_csv(runtime_path))
        runtime = pd.concat([previous_runtime, runtime], ignore_index=True)
        runtime = runtime.drop_duplicates("forecast_day", keep="last")
    _atomic_frame(runtime_path, runtime, index=False)
    atomic_write_json(
        output_dir / "config.json",
        {
            "mode": "daily_operational",
            "configuration": sqra_name,
            "train_days": config.sqra_train_days,
            "mtu_specific": config.sqra_mtu_specific,
            "quantiles": list(quantiles),
            "import_paths": [str(path) for path in specification.inputs(config.results_root)],
            **schema_metadata(),
        },
    )
    return combined


def _payload_path(
    config: OperationalConfig,
    submission_stream: str,
    challenge_id: str,
    target_start: datetime,
) -> Path:
    # Keep the target in the interface alongside the challenge metadata, while
    # retaining only the latest validated payload for each model stream/challenge.
    del target_start
    return (
        config.output_root
        / "payloads"
        / submission_stream
        / challenge_id
        / "latest.json"
    )


def _required_input_start(
    target_date: date,
    runs: Sequence[LearRun],
    sqra_members: frozenset[str],
    sqra_train_days: int,
) -> tuple[date, date]:
    market_starts: list[date] = []
    weather_starts: list[date] = []
    for run in runs:
        earliest_forecast = target_date - timedelta(
            days=sqra_train_days if run.name in sqra_members else 0
        )
        feature_start = earliest_forecast - timedelta(days=run.train_days)
        market_starts.append(feature_start - timedelta(days=7))
        if not run.use_exaa_only:
            weather_starts.append(feature_start)
    return min(market_starts), min(weather_starts) if weather_starts else target_date


def execute_pipeline(
    config: OperationalConfig,
    targets: ArenaTargets,
    *,
    submit: bool,
    force_download: bool = False,
    force_forecast: bool = False,
    force_submit: bool = False,
) -> dict[str, object]:
    runs, point_run, sqra_name, sqra_members = resolve_model_plan(config)
    target_date = targets.target_date
    market_start, weather_start = _required_input_start(
        target_date, runs, sqra_members, config.sqra_train_days
    )
    print(f"Target: {targets.target_start.isoformat()}")
    print(f"Submitted point model: {point_run.name}")
    print(f"SQRA configuration: {sqra_name}")
    print("Arena quantiles: " + ", ".join(f"{q:g}" for q in targets.quantile.quantiles))
    print("Point runs: " + ", ".join(run.name for run in runs))

    if config.needs_dwd:
        prepare_target_dwd(
            config,
            target_date,
            force_download=force_download,
            delete_raw=config.delete_dwd_raw_after_preprocess,
        )

    prices, exaa, load = load_market_data(
        config, start_date=market_start, target_date=target_date
    )
    weather = {
        clusters: _load_weather_features(
            config, clusters, weather_start, target_date
        )
        for clusters in config.required_dwd_clusters
    } if config.needs_dwd else {}

    forecasts: dict[str, pd.DataFrame] = {}
    for run in runs:
        days = _requested_days(
            target_date,
            config.sqra_train_days,
            run.name in sqra_members,
        ).tz_localize(config.timezone)
        print(f"Point forecast {run.name}: {days[0].date()}..{days[-1].date()}")
        forecasts[run.name] = run_point_model(
            config,
            run,
            requested_days=days,
            prices=prices,
            exaa=exaa,
            load=load,
            weather=weather,
            force=force_forecast,
        )

    sqra = run_sqra_model(
        config,
        sqra_name,
        target_date,
        targets.quantile.quantiles,
    )
    point_payload = build_point_payload(forecasts[point_run.name], targets.point)
    quantile_payload = build_quantile_payload(sqra, targets.quantile)
    submission_stream = _submission_stream(config)
    point_payload_path = _payload_path(
        config,
        submission_stream,
        targets.point.challenge_id,
        targets.point.target_start,
    )
    quantile_payload_path = _payload_path(
        config,
        submission_stream,
        targets.quantile.challenge_id,
        targets.quantile.target_start,
    )
    atomic_write_json(point_payload_path, point_payload)
    atomic_write_json(quantile_payload_path, quantile_payload)
    print(f"Point payload: {point_payload_path}")
    print(f"Quantile payload: {quantile_payload_path}")

    submissions: dict[str, object] = {}
    if submit:
        # Model fitting may overlap across information-cutoff streams. Only
        # the short write-to-one-account section is serialized.
        with interprocess_lock(
            arena_lock_path(config),
            timeout_seconds=15 * 60,
            description="Energy Arena submission account",
        ):
            now = datetime.now().astimezone(targets.point.deadline.tzinfo)
            deadline = min(targets.point.deadline, targets.quantile.deadline)
            if now > deadline:
                raise RuntimeError(
                    f"Energy Arena deadline has passed: {deadline.isoformat()}"
                )
            for label, challenge, payload in (
                ("point", targets.point, point_payload),
                ("quantile", targets.quantile, quantile_payload),
            ):
                record_path = submission_record_path(
                    config.output_root,
                    submission_stream,
                    challenge.challenge_id,
                    challenge.target_start,
                )
                submissions[label] = submit_with_record(
                    api_base=config.arena_api_base_url,
                    api_key=config.arena_api_key,
                    payload=payload,
                    record_path=record_path,
                    force=force_submit,
                )
                print(f"Energy Arena {label}: {submissions[label]}")
    else:
        print("Submission disabled; validated payloads were saved locally.")
    return {
        "target_start": targets.target_start.isoformat(),
        "point_model": point_run.name,
        "sqra_model": sqra_name,
        "point_payload": str(point_payload_path),
        "quantile_payload": str(quantile_payload_path),
        "submissions": submissions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--model", choices=MODEL_VARIANTS)
    variants.add_argument("--exaa-only", "--exaa_only", action="store_true")
    variants.add_argument("--exaa-enriched", "--exaa_enriched", action="store_true")
    variants.add_argument("--fundamental", action="store_true")
    parser.add_argument("--cluster", type=int, choices=(1, 5, 25))
    target_overrides = parser.add_mutually_exclusive_group()
    target_overrides.add_argument("--target-date", type=date.fromisoformat)
    target_overrides.add_argument(
        "--d-1",
        dest="delivery_day_minus_one",
        action="store_true",
        help=(
            "Test the delivery day one calendar day before Energy Arena's live "
            "target; requires --no-submit."
        ),
    )
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--check-setup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-forecast", action="store_true")
    parser.add_argument("--force-submit", action="store_true")
    return parser


def _apply_cli(config: OperationalConfig, args: argparse.Namespace) -> OperationalConfig:
    variant = args.model
    if args.exaa_only:
        variant = "exaa_only"
    elif args.exaa_enriched:
        variant = "exaa_enriched"
    elif args.fundamental:
        variant = "fundamental"
    changes = {}
    if variant:
        changes.update(point_variant=variant, sqra_variant=variant)
    if args.cluster:
        changes["point_clusters"] = args.cluster
    return replace(config, **changes) if changes else config


def _check_setup(
    config: OperationalConfig,
    *,
    submission_requested: bool,
) -> list[str]:
    errors: list[str] = []
    if not config.entsoe_api_key or config.entsoe_api_key == "your_api_key_here":
        errors.append("ENTSOE_API_KEY is missing.")
    if submission_requested and not config.arena_api_key:
        errors.append("ENERGY_ARENA_API_KEY is missing.")
    if config.needs_dwd:
        for clusters in config.required_dwd_clusters:
            cluster_file = (
                config.repo_root
                / "data"
                / "clustering"
                / f"icon_d2_clustering_c{clusters}.parquet"
            )
            if not cluster_file.is_file():
                errors.append(f"Missing cluster assignment: {cluster_file}")
        if not config.download_dwd and not config.dwd_raw_archive.is_dir():
            errors.append(
                "DWD_OPERATIONAL_RAW_ROOT is unavailable: "
                f"{config.dwd_raw_archive}"
            )
    return errors


def main(argv: Optional[Sequence[str]] = None, repo_root: Optional[Path] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delivery_day_minus_one and not args.no_submit:
        parser.error("--d-1 is a testing override and requires --no-submit")
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    config = _apply_cli(load_operational_config(root), args)
    submission_requested = config.submit_to_arena and not args.no_submit
    errors = _check_setup(
        config,
        submission_requested=submission_requested,
    )
    if errors:
        print("Setup failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    targets = resolve_targets(
        api_base=config.arena_api_base_url,
        point_challenge_id=config.arena_point_challenge_id,
        quantile_challenge_id=config.arena_quantile_challenge_id,
        api_key=config.arena_api_key,
        requested_target_date=args.target_date,
        target_date_offset_days=-1 if args.delivery_day_minus_one else 0,
    )
    runs, point_run, sqra_name, _ = resolve_model_plan(config)
    if args.check_setup:
        print("Setup passed.")
        print(f"Next target: {targets.target_start.isoformat()}")
        print(f"Point model: {point_run.name}")
        print(f"SQRA model: {sqra_name}")
        return 0
    if args.dry_run:
        print(f"Target: {targets.target_start.isoformat()}")
        print(f"Point submission: {point_run.name}")
        print("Required point runs: " + ", ".join(run.name for run in runs))
        print(f"SQRA submission: {sqra_name}")
        print(
            "Arena quantiles: "
            + ", ".join(f"{q:g}" for q in targets.quantile.quantiles)
        )
        print(f"DWD clusters: {config.required_dwd_clusters if config.needs_dwd else 'none'}")
        print(f"Submit: {submission_requested}")
        return 0

    with ExitStack() as locks:
        for lock_path in pipeline_lock_paths(config):
            locks.enter_context(
                interprocess_lock(
                    lock_path,
                    description=f"pipeline output {lock_path.stem}",
                )
            )

        config.output_root.mkdir(parents=True, exist_ok=True)
        log_path = pipeline_log_path(config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Pipeline log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(original_stdout, log)
            sys.stderr = _Tee(original_stderr, log)
            try:
                print(f"started_at={datetime.now().astimezone().isoformat()}")
                print(f"command={_redacted_command(list(argv or sys.argv[1:]))}")
                summary = execute_pipeline(
                    config,
                    targets,
                    submit=submission_requested,
                    force_download=args.force_download,
                    force_forecast=args.force_forecast,
                    force_submit=args.force_submit,
                )
                print(json.dumps(summary, indent=2))
                print(f"completed_at={datetime.now().astimezone().isoformat()}")
                print("status=success")
            except BaseException:
                print(f"failed_at={datetime.now().astimezone().isoformat()}")
                print("status=failed")
                traceback.print_exc()
                raise
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    return 0
