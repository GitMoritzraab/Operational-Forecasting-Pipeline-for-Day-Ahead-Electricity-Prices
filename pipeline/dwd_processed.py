"""Read and write compact processed ICON-D2 containers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile

import pandas as pd


PARQUET_FILENAME = "icon_d2_aggregated.parquet"
LEGACY_WORKBOOK_FILENAME = "icon_d2_aggregated.xlsx"
COMPLETE_FILENAME = ".complete"
FORMAT_VERSION = 2
VARIABLE_COLUMN = "variable"
TIMESTAMP_COLUMN = "timestamp"


def _paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / PARQUET_FILENAME, output_dir / COMPLETE_FILENAME


def _safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    result.index.name = TIMESTAMP_COLUMN
    return result


def _to_long_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for variable, frame in frames.items():
        part = _safe_frame(frame).reset_index()
        part.insert(0, VARIABLE_COLUMN, variable)
        parts.append(part)
    if not parts:
        raise ValueError("Cannot write an empty DWD processed Parquet file.")
    return pd.concat(parts, ignore_index=True, sort=False)


def _from_long_table(
    table: pd.DataFrame,
    variable_order: list[str],
) -> dict[str, pd.DataFrame]:
    required = {VARIABLE_COLUMN, TIMESTAMP_COLUMN}
    if not required.issubset(table.columns):
        raise ValueError(
            f"Processed DWD Parquet file lacks columns "
            f"{sorted(required - set(table.columns))}."
        )
    available = set(table[VARIABLE_COLUMN].dropna().astype(str))
    if available != set(variable_order):
        raise ValueError(
            "Processed DWD Parquet variables do not match the completion marker."
        )
    frames: dict[str, pd.DataFrame] = {}
    value_columns = [
        column
        for column in table.columns
        if column not in {VARIABLE_COLUMN, TIMESTAMP_COLUMN}
    ]
    for variable in variable_order:
        frame = table.loc[
            table[VARIABLE_COLUMN].astype(str) == variable,
            [TIMESTAMP_COLUMN, *value_columns],
        ].copy().reset_index(drop=True)
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN], errors="raise"
        )
        frames[variable] = frame
    return frames


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_processed_parquet(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    *,
    variable_metadata: Mapping[str, Mapping[str, Any]],
    issue_date: str,
    run_hour: str,
) -> tuple[Path, Path]:
    """Atomically write one Parquet file and one completion marker."""
    if set(frames) != set(variable_metadata):
        raise ValueError("DWD Parquet frames and metadata have different variables.")
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, marker_path = _paths(output_dir)
    table = _to_long_table(frames)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{parquet_path.stem}.",
        suffix=parquet_path.suffix,
        dir=str(output_dir),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        os.replace(temporary, parquet_path)
    finally:
        temporary.unlink(missing_ok=True)

    marker_payload = {
        "format": "icon_d2_processed_parquet",
        "format_version": FORMAT_VERSION,
        "parquet": PARQUET_FILENAME,
        "parquet_size": parquet_path.stat().st_size,
        "issue_date": issue_date,
        "run_hour": run_hour,
        "variables_order": list(frames),
        "variables": variable_metadata,
    }
    _atomic_json(marker_path, marker_payload)
    return parquet_path, marker_path


def parquet_is_complete(output_dir: Path) -> bool:
    """Return whether the marker matches a readable Parquet container."""
    parquet_path, marker_path = _paths(output_dir)
    if not parquet_path.is_file() or not marker_path.is_file():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            return False
        if payload.get("parquet") != PARQUET_FILENAME:
            return False
        if payload.get("parquet_size") != parquet_path.stat().st_size:
            return False
        variables = list(payload.get("variables_order") or [])
        if not variables:
            return False
        table = pd.read_parquet(
            parquet_path,
            columns=[VARIABLE_COLUMN, TIMESTAMP_COLUMN],
            engine="pyarrow",
        )
        return set(table[VARIABLE_COLUMN].dropna().astype(str)) == set(variables)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _legacy_workbook_is_complete(output_dir: Path) -> bool:
    workbook_path = output_dir / LEGACY_WORKBOOK_FILENAME
    marker_path = output_dir / COMPLETE_FILENAME
    if not workbook_path.is_file() or not marker_path.is_file():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            return False
        if payload.get("workbook") != LEGACY_WORKBOOK_FILENAME:
            return False
        if payload.get("workbook_size") != workbook_path.stat().st_size:
            return False
        expected_sheets = list(payload.get("sheets") or [])
        with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
            return bool(expected_sheets) and workbook.sheet_names == expected_sheets
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, json.JSONDecodeError):
        return False


def read_processed_output(output_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Read Parquet, the short-lived XLSX format, or return None for CSV folders."""
    parquet_path, marker_path = _paths(output_dir)
    if parquet_path.exists() or (
        marker_path.exists()
        and not (output_dir / LEGACY_WORKBOOK_FILENAME).exists()
    ):
        if not parquet_is_complete(output_dir):
            raise ValueError(f"Incomplete or corrupt DWD Parquet file: {parquet_path}")
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        return _from_long_table(
            pd.read_parquet(parquet_path, engine="pyarrow"),
            list(payload["variables_order"]),
        )

    workbook_path = output_dir / LEGACY_WORKBOOK_FILENAME
    if workbook_path.exists() or marker_path.exists():
        if not _legacy_workbook_is_complete(output_dir):
            raise ValueError(f"Incomplete or corrupt DWD workbook: {workbook_path}")
        return pd.read_excel(
            workbook_path,
            sheet_name=None,
            engine="openpyxl",
        )
    return None


def read_completed_legacy_outputs(
    output_dir: Path,
    source_folders: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]] | None:
    """Load a complete set of old CSV/marker pairs for fast migration."""
    if not source_folders:
        return None
    frames: dict[str, pd.DataFrame] = {}
    metadata_by_variable: dict[str, dict[str, Any]] = {}
    for source_folder in source_folders:
        marker = output_dir / f".{source_folder}.complete"
        if not marker.is_file():
            return None
        former_name = marker.read_text(encoding="utf-8").strip().splitlines()
        if not former_name:
            return None
        csv_path = output_dir / former_name[0]
        if not csv_path.is_file():
            return None

        metadata: dict[str, Any] = {}
        with csv_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("#"):
                    break
                key, separator, value = line[1:].partition(":")
                if separator:
                    metadata[key.strip().lower().replace(" ", "_")] = value.strip()
        field_name = str(metadata.get("variable") or csv_path.name.split("_", 1)[0])
        if field_name in frames:
            raise ValueError(
                f"Duplicate DWD variable {field_name!r} in legacy folder {output_dir}."
            )
        frame = pd.read_csv(csv_path, comment="#")
        if TIMESTAMP_COLUMN not in frame.columns:
            raise ValueError(f"Missing timestamp column in legacy DWD file: {csv_path}")
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN], errors="raise"
        )
        frame = frame.set_index(TIMESTAMP_COLUMN)
        frame.index.name = TIMESTAMP_COLUMN
        metadata.update(
            {
                "source_folder": source_folder,
                "field_name": field_name,
                "former_csv_name": csv_path.name,
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            }
        )
        frames[field_name] = frame
        metadata_by_variable[field_name] = metadata
    return frames, metadata_by_variable


def completed_xlsx_for_migration(
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]] | None:
    """Read a complete version-1 workbook so it can be replaced by Parquet."""
    if not _legacy_workbook_is_complete(output_dir):
        return None
    marker = json.loads(
        (output_dir / COMPLETE_FILENAME).read_text(encoding="utf-8")
    )
    frames = pd.read_excel(
        output_dir / LEGACY_WORKBOOK_FILENAME,
        sheet_name=None,
        index_col=TIMESTAMP_COLUMN,
        engine="openpyxl",
    )
    metadata = marker.get("variables") or {
        name: {
            "field_name": name,
            "rows": len(frame),
            "columns": len(frame.columns),
        }
        for name, frame in frames.items()
    }
    return frames, metadata


def remove_superseded_outputs(output_dir: Path) -> None:
    """Remove only superseded per-variable CSVs/markers and the XLSX container."""
    for path in output_dir.glob("*.csv"):
        path.unlink()
    for path in output_dir.glob(".*.complete"):
        path.unlink()
    (output_dir / LEGACY_WORKBOOK_FILENAME).unlink(missing_ok=True)
