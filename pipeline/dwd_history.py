"""Consolidated, appendable ICON-D2 histories used by LEAR."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.dwd_processed import read_processed_output


HISTORY_FORMAT_VERSION = 1
HISTORY_MANIFEST = "dwd_history.json"
DWD_FOLDER_OFFSET_DATE = date(2025, 10, 26)
FORECAST_FIELDS = ("u10", "v10", "ASWDIR_S", "ASWDIFD_S")
SOLAR_FIELDS = frozenset(("ASWDIR_S", "ASWDIFD_S"))
METADATA_COLUMNS = ("delivery_date", "issue_date", "run_hour", "timestamp")
DAILY_FOLDER_PATTERN = re.compile(r"^dwd_icon_daily_(\d{8})_(\d{2})$")


def history_path(cluster_dir: Path, field: str) -> Path:
    if field not in FORECAST_FIELDS:
        raise ValueError(f"Unsupported DWD history field: {field}")
    return cluster_dir / f"{field}.parquet"


def delivery_date_for_issue(issue_date: date) -> date:
    """Preserve the repository's historical folder-to-delivery convention."""
    return (
        issue_date
        if issue_date < DWD_FOLDER_OFFSET_DATE
        else issue_date + timedelta(days=1)
    )


def _cluster_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame.columns if column.startswith("cluster_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )


def _timestamp_column(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        values = frame["timestamp"]
    elif isinstance(frame.index, pd.DatetimeIndex):
        values = pd.Series(frame.index, index=frame.index)
    else:
        raise ValueError("Processed DWD frame has no timestamp column or index.")
    return pd.to_datetime(values, utc=True, errors="raise")


def _collapse_duplicate_timestamps(
    field: str,
    frame: pd.DataFrame,
    cluster_columns: Sequence[str],
) -> pd.DataFrame:
    """Collapse repeated valid times before validating a delivery-day slice.

    A few archived ICON-D2 files expose the same valid time more than once.
    Keeping an arbitrary occurrence would make the result depend on archive
    member order, so numeric cluster values are averaged for each repeated
    timestamp.  Identical duplicates are therefore unchanged, while genuinely
    conflicting duplicates are handled deterministically.
    """
    duplicated = frame["timestamp"].duplicated(keep=False)
    if not duplicated.any():
        return frame

    duplicate_rows = int(duplicated.sum())
    duplicate_times = int(frame.loc[duplicated, "timestamp"].nunique())
    before = len(frame)
    collapsed = (
        frame.loc[:, ["timestamp", *cluster_columns]]
        .groupby("timestamp", as_index=False, sort=True)[list(cluster_columns)]
        .mean()
    )
    print(
        f"DWD {field}: collapsed {before - len(collapsed)} duplicate row(s) "
        f"across {duplicate_times} valid timestamp(s) "
        f"({duplicate_rows} rows involved).",
        flush=True,
    )
    return collapsed


def delivery_slice(
    field: str,
    frame: pd.DataFrame,
    *,
    issue_date: date,
    run_hour: str,
    delivery_date: date,
    timezone: str,
) -> pd.DataFrame:
    """Normalize one processed forecast to the timestamps used for one delivery day."""
    cluster_columns = _cluster_columns(frame)
    if not cluster_columns:
        raise ValueError(f"{field} has no cluster columns.")
    timestamps = _timestamp_column(frame)
    if field in SOLAR_FIELDS:
        timestamps = timestamps - pd.Timedelta(minutes=15)
    local = timestamps.dt.tz_convert(timezone)
    start = pd.Timestamp(delivery_date, tz=timezone)
    end = start + pd.DateOffset(days=1)
    selected = (local >= start) & (local < end)
    values = frame.loc[selected.to_numpy(), cluster_columns].reset_index(drop=True)
    selected_timestamps = timestamps.loc[selected].reset_index(drop=True)
    result = values.copy()
    result.insert(0, "timestamp", selected_timestamps)
    result = _collapse_duplicate_timestamps(field, result, cluster_columns)
    result.insert(0, "run_hour", str(run_hour).zfill(2))
    result.insert(0, "issue_date", issue_date.isoformat())
    result.insert(0, "delivery_date", delivery_date.isoformat())
    return result.loc[:, [*METADATA_COLUMNS, *cluster_columns]]


def _expected_timestamps(delivery_date: date, frequency: str, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(delivery_date, tz=timezone)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq=frequency, inclusive="left").tz_convert("UTC")


def validate_delivery_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    delivery_date: date,
    run_hour: str,
    timezone: str,
) -> None:
    """Require complete paired wind and solar tables for one delivery day."""
    missing_fields = sorted(set(FORECAST_FIELDS) - set(tables))
    if missing_fields:
        raise ValueError(f"Missing DWD history fields: {missing_fields}")
    expected_by_field = {
        "u10": _expected_timestamps(delivery_date, "1h", timezone),
        "v10": _expected_timestamps(delivery_date, "1h", timezone),
        "ASWDIR_S": _expected_timestamps(delivery_date, "15min", timezone),
        "ASWDIFD_S": _expected_timestamps(delivery_date, "15min", timezone),
    }
    actual_by_field: dict[str, pd.DatetimeIndex] = {}
    for field in FORECAST_FIELDS:
        frame = tables[field]
        required = set(METADATA_COLUMNS)
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{field} history lacks columns {sorted(required - set(frame.columns))}."
            )
        selected = frame.loc[
            (frame["delivery_date"].astype(str) == delivery_date.isoformat())
            & (frame["run_hour"].astype(str).str.zfill(2) == str(run_hour).zfill(2))
        ]
        actual = pd.DatetimeIndex(
            pd.to_datetime(selected["timestamp"], utc=True, errors="raise")
        ).sort_values()
        expected = expected_by_field[field]
        if not actual.equals(expected):
            missing = expected.difference(actual)
            extra = actual.difference(expected)
            raise ValueError(
                f"Incomplete {field} history for delivery {delivery_date}, run "
                f"{run_hour}: rows={len(actual)}/{len(expected)}, "
                f"missing={len(missing)}, extra={len(extra)}."
            )
        cluster_columns = _cluster_columns(selected)
        if not cluster_columns or not np.isfinite(
            selected[cluster_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{field} history contains missing/non-finite cluster values for "
                f"delivery {delivery_date}, run {run_hour}."
            )
        actual_by_field[field] = actual
    if not actual_by_field["u10"].equals(actual_by_field["v10"]):
        raise ValueError("u10 and v10 timestamps do not match.")
    if not actual_by_field["ASWDIR_S"].equals(actual_by_field["ASWDIFD_S"]):
        raise ValueError("Direct and diffuse solar timestamps do not match.")


def _read_history(cluster_dir: Path, field: str) -> pd.DataFrame:
    path = history_path(cluster_dir, field)
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path, engine="pyarrow")
    frame["delivery_date"] = frame["delivery_date"].astype(str)
    frame["issue_date"] = frame["issue_date"].astype(str)
    frame["run_hour"] = frame["run_hour"].astype(str).str.zfill(2)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame


def read_history_tables(cluster_dir: Path) -> dict[str, pd.DataFrame]:
    return {field: _read_history(cluster_dir, field) for field in FORECAST_FIELDS}


def history_is_complete(cluster_dir: Path) -> bool:
    manifest_path = cluster_dir / HISTORY_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != HISTORY_FORMAT_VERSION:
            return False
        for field in FORECAST_FIELDS:
            path = history_path(cluster_dir, field)
            detail = manifest["files"][field]
            if not path.is_file() or path.stat().st_size != detail["size"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _merge_field(existing: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = additions.copy()
    else:
        replacement_keys = set(
            zip(additions["delivery_date"].astype(str), additions["run_hour"].astype(str))
        )
        existing_keys = list(
            zip(existing["delivery_date"].astype(str), existing["run_hour"].astype(str))
        )
        keep = [key not in replacement_keys for key in existing_keys]
        combined = pd.concat([existing.loc[keep], additions], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return combined.sort_values(
        ["delivery_date", "run_hour", "timestamp"]
    ).reset_index(drop=True)


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_history(
    cluster_dir: Path,
    additions: Mapping[str, pd.DataFrame],
    *,
    timezone: str,
) -> None:
    """Atomically stage and then replace all four variable history files."""
    if set(additions) != set(FORECAST_FIELDS):
        raise ValueError("A DWD history update must contain all forecast fields.")
    cluster_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        field: _merge_field(_read_history(cluster_dir, field), additions[field])
        for field in FORECAST_FIELDS
    }
    for field, frame in merged.items():
        if frame.empty or frame.duplicated(
            ["delivery_date", "run_hour", "timestamp"]
        ).any():
            raise ValueError(f"Invalid duplicate/empty consolidated {field} history.")

    with tempfile.TemporaryDirectory(prefix=".dwd_history_", dir=str(cluster_dir)) as tmp:
        stage = Path(tmp)
        staged_paths: dict[str, Path] = {}
        for field, frame in merged.items():
            staged = stage / f"{field}.parquet"
            frame.to_parquet(staged, index=False, engine="pyarrow", compression="zstd")
            staged_paths[field] = staged

        backups: dict[str, Path] = {}
        replaced: list[str] = []
        try:
            for field in FORECAST_FIELDS:
                destination = history_path(cluster_dir, field)
                backup = stage / f"{field}.previous.parquet"
                if destination.exists():
                    os.replace(destination, backup)
                    backups[field] = backup
                os.replace(staged_paths[field], destination)
                replaced.append(field)
        except BaseException:
            for field in reversed(replaced):
                history_path(cluster_dir, field).unlink(missing_ok=True)
            for field, backup in backups.items():
                if backup.exists():
                    os.replace(backup, history_path(cluster_dir, field))
            raise

    all_dates = sorted(
        {
            value
            for frame in merged.values()
            for value in frame["delivery_date"].astype(str).unique()
        }
    )
    manifest = {
        "format": "dwd_consolidated_history",
        "format_version": HISTORY_FORMAT_VERSION,
        "variables": list(FORECAST_FIELDS),
        "first_delivery_date": all_dates[0],
        "last_delivery_date": all_dates[-1],
        "files": {
            field: {
                "filename": history_path(cluster_dir, field).name,
                "rows": len(frame),
                "size": history_path(cluster_dir, field).stat().st_size,
            }
            for field, frame in merged.items()
        },
        "timezone": timezone,
    }
    _atomic_json(cluster_dir / HISTORY_MANIFEST, manifest)


def _read_daily_csvs(folder: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*.csv")):
        name = path.name
        field = next(
            (candidate for candidate in FORECAST_FIELDS if name.startswith(candidate + "_")),
            None,
        )
        if field is None:
            continue
        frames[field] = pd.read_csv(path, comment="#")
    return frames


def read_daily_output(folder: Path) -> dict[str, pd.DataFrame]:
    frames = read_processed_output(folder)
    if frames is None:
        frames = _read_daily_csvs(folder)
    missing = sorted(set(FORECAST_FIELDS) - set(frames))
    if missing:
        raise ValueError(f"{folder} lacks processed DWD fields {missing}.")
    return {field: frames[field] for field in FORECAST_FIELDS}


def append_daily_output(
    cluster_dir: Path,
    daily_folder: Path,
    *,
    issue_date: date,
    run_hour: str,
    delivery_date: date,
    timezone: str,
) -> None:
    frames = read_daily_output(daily_folder)
    additions = {
        field: delivery_slice(
            field,
            frame,
            issue_date=issue_date,
            run_hour=run_hour,
            delivery_date=delivery_date,
            timezone=timezone,
        )
        for field, frame in frames.items()
    }
    validate_delivery_tables(
        additions,
        delivery_date=delivery_date,
        run_hour=run_hour,
        timezone=timezone,
    )
    update_history(cluster_dir, additions, timezone=timezone)


def migrate_daily_outputs(
    cluster_dir: Path,
    *,
    timezone: str,
    force: bool = False,
) -> int:
    """Build consolidated histories from existing daily CSV/Parquet folders."""
    if history_is_complete(cluster_dir) and not force:
        return 0
    additions: dict[str, list[pd.DataFrame]] = {field: [] for field in FORECAST_FIELDS}
    migrated = 0
    for folder in sorted(cluster_dir.glob("dwd_icon_daily_*_*")):
        if not folder.is_dir():
            continue
        match = DAILY_FOLDER_PATTERN.match(folder.name)
        if match is None:
            continue
        issue_date = date.fromisoformat(
            f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
        )
        run_hour = match.group(2)
        delivery_date = delivery_date_for_issue(issue_date)
        try:
            frames = read_daily_output(folder)
            normalized = {
                field: delivery_slice(
                    field,
                    frame,
                    issue_date=issue_date,
                    run_hour=run_hour,
                    delivery_date=delivery_date,
                    timezone=timezone,
                )
                for field, frame in frames.items()
            }
            validate_delivery_tables(
                normalized,
                delivery_date=delivery_date,
                run_hour=run_hour,
                timezone=timezone,
            )
        except (OSError, ValueError) as exc:
            print(f"DWD history migration skipped {folder.name}: {exc}", flush=True)
            continue
        for field, frame in normalized.items():
            additions[field].append(frame)
        migrated += 1
    if not migrated:
        return 0
    update_history(
        cluster_dir,
        {field: pd.concat(frames, ignore_index=True) for field, frames in additions.items()},
        timezone=timezone,
    )
    return migrated


def consolidated_delivery_available(
    cluster_dir: Path,
    *,
    delivery_date: date,
    run_hour: str,
    timezone: str,
) -> bool:
    if not history_is_complete(cluster_dir):
        return False
    try:
        validate_delivery_tables(
            read_history_tables(cluster_dir),
            delivery_date=delivery_date,
            run_hour=run_hour,
            timezone=timezone,
        )
        return True
    except ValueError:
        return False


def load_consolidated_dwd(
    cluster_dir: Path,
    *,
    start_issue_date: date,
    end_issue_date: date | None,
    required_run: str,
    skipped_issue_dates: Sequence[date],
    target_tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not history_is_complete(cluster_dir):
        raise FileNotFoundError(f"No complete consolidated DWD history in {cluster_dir}.")
    selected: dict[str, pd.DataFrame] = {}
    end_value = end_issue_date or date.max
    skipped = {value.isoformat() for value in skipped_issue_dates}
    for field, frame in read_history_tables(cluster_dir).items():
        subset = frame.loc[
            frame["issue_date"].between(start_issue_date.isoformat(), end_value.isoformat())
            & (frame["run_hour"].astype(str).str.zfill(2) == str(required_run).zfill(2))
            & ~frame["issue_date"].isin(skipped)
        ].copy()
        cluster_columns = _cluster_columns(subset)
        prefix = {"ASWDIR_S": "ASWDIR", "ASWDIFD_S": "ASWDIFD"}.get(field, field)
        subset = subset.rename(
            columns={column: f"{prefix}_{column}" for column in cluster_columns}
        )
        subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True).dt.tz_convert(
            target_tz
        )
        selected[field] = subset[["timestamp", *[f"{prefix}_{c}" for c in cluster_columns]]]

    def merge(fields: tuple[str, str]) -> pd.DataFrame:
        result = selected[fields[0]]
        result = result.merge(selected[fields[1]], on="timestamp", how="inner")
        if result.empty:
            raise ValueError("No hourly DWD data loaded." if fields[0] == "u10" else "No quarter-hourly DWD data loaded.")
        return result.sort_values("timestamp").set_index("timestamp")

    hourly = merge(("u10", "v10"))
    quarter_hourly = merge(("ASWDIR_S", "ASWDIFD_S"))
    hourly.index.name = "timestamp"
    quarter_hourly.index.name = "timestamp"
    return hourly, quarter_hourly
