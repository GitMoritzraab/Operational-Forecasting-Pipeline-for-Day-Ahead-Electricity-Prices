#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_icon_d2.py
====================

Aggregates DWD ICON-D2 weather forecast data to spatial clusters and saves
all variables for a run in one Parquet file plus one completion marker.

For each forecast day and model run, the script loads the raw GRIB2 files,
maps each grid point to the nearest ICON-D2 cluster centroid, and computes
cluster-averaged values for wind, temperature, and solar radiation variables.

Note: raw ICON-D2 data is not included in the repository due to file size.
Adjust LSDF_BASE and OUTPUT_PARENT in the CONFIG section to your local paths.
"""

from __future__ import annotations

# ====================================================================
# IMPORTS
# ====================================================================
import os
import sys
import bz2
import glob
import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from datetime import datetime

# Optional plotting imports (only used if plot=True)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Do not let optional Unicode status symbols abort long Windows runs when the
# active console uses a legacy encoding such as cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# ====================================================================
# CONFIG
# ====================================================================

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from experiment_config import env_bool, load_experiment_config
from pipeline.dwd_processed import (
    PARQUET_FILENAME,
    completed_xlsx_for_migration,
    read_completed_legacy_outputs,
    remove_superseded_outputs,
    parquet_is_complete,
    write_processed_parquet,
)
from pipeline.dwd_raw import (
    available_run_hours,
    available_variables,
    discover_issue_dates,
    materialize_raw_run,
)

EXPERIMENT = load_experiment_config(BASE_DIR)

# Path to the raw ICON-D2 data source (SMB mount or local copy).
# Must be adjusted by the user to point to the correct location.
LSDF_BASE = Path(os.getenv(
    "DWD_RAW_ARCHIVE",
    "/Volumes/iip-projects/energy/climateData/icon_by_Max_Kleinebrahm",
))
# Output directory for aggregated Parquet files.
# Note: aggregated data is stored outside the repository due to file size.
# Adjust to your preferred output location.
N_CLUSTERS = int(os.getenv("DWD_N_CLUSTERS", "5"))
OUTPUT_PARENT = str(EXPERIMENT.icon_cluster_dir(N_CLUSTERS))
CLUSTER_FILE = BASE_DIR / "data" / "clustering" / f"icon_d2_clustering_c{N_CLUSTERS}.parquet"

# Path to the Natural Earth 10m admin-0 shapefile.
# The shapefile is included in the repository under data/shapefile/.
SHAPEFILE_PATH = str(Path(__file__).parent.parent / "data" / "shapefile" / "ne_10m_admin_0_countries.shp")

BUFFER_KM = 50
PLOT_CLUSTERS = False          # keep False for RAM friendliness
SKIP_EXISTING_OUTPUT = env_bool("DWD_SKIP_EXISTING_OUTPUT", True)

# Optional: restrict to one day for testing (set e.g. "20250801" or None)
ONLY_DAY = os.getenv("DWD_ONLY_DAY") or None

# Optional: restrict to one run hour (e.g. "12") or None
ONLY_RUN_HOUR = os.getenv("DWD_RUN_HOUR", "09") or None

START_DATE = datetime.fromisoformat(os.getenv("DWD_PREPROCESS_START_DATE", "2025-08-01"))
END_DATE = datetime.combine(EXPERIMENT.evaluation_end, datetime.min.time())

# These are the four ICON-D2 variables actually consumed by the LEAR model.
VARIABLES = ["u_10m", "v_10m", "aswdir_s", "aswdifd_s"]
REQUIRED_OUTPUT_FIELDS = {"u10", "v10", "ASWDIR_S", "ASWDIFD_S"}


# ====================================================================
# HELPERS
# ====================================================================

def compute_step_flux(timestamps, flux_values):
    """Convert cumulative mean/avg flux into instantaneous flux [W/m²]."""
    t = pd.to_datetime(timestamps)
    Fbar = np.asarray(flux_values, dtype=float)
    if len(Fbar) < 2:
        return Fbar

    tsec = np.asarray((t - t[0]) / np.timedelta64(1, "s"), dtype=float)
    E = Fbar * tsec
    F_inst = np.full_like(Fbar, np.nan, dtype=float)

    dt = np.diff(tsec)
    valid = dt != 0
    if np.any(valid):
        F_inst[1:][valid] = np.diff(E)[valid] / dt[valid]
    return np.clip(F_inst, 0, None)


def deduplicate_timesteps(timestamps, values, *, variable_name="variable"):
    """Average rows with the same valid time before further processing.

    Some archived ICON-D2 runs contain multiple GRIB messages resolving to the
    same ``valid_time``.  Deduplicating here is especially important for the
    cumulative solar variables: their instantaneous flux must be calculated
    from a strictly increasing timestamp sequence.
    """
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    matrix = np.asarray(values, dtype=float)
    if len(index) != len(matrix):
        raise ValueError("DWD timestamps and aggregated values have different lengths.")
    duplicated = index.duplicated(keep=False)
    if not duplicated.any():
        return index, matrix

    duplicate_rows = int(duplicated.sum())
    duplicate_times = int(index[duplicated].nunique())
    table = pd.DataFrame(matrix, index=index)
    collapsed = table.groupby(level=0, sort=True).mean()
    print(
        f"DWD {variable_name}: collapsed {len(index) - len(collapsed)} "
        f"duplicate row(s) across {duplicate_times} valid timestamp(s) "
        f"({duplicate_rows} rows involved)."
    )
    return pd.DatetimeIndex(collapsed.index), collapsed.to_numpy(dtype=float)


def safe_unit_for_filename(units: str) -> str:
    """Make unit string filename-safe."""
    if units is None:
        units = ""
    return (
        str(units)
        .replace("/", "-per-")
        .replace("*", "")
        .replace(" ", "_")
        .replace("²", "2")
        .replace("°", "")
    )


def filter_points_in_germany(latitudes, longitudes, shapefile_path, buffer_km=50):
    """Return boolean mask for points inside Germany (+buffer)."""
    world = gpd.read_file(shapefile_path)
    germany = world[world["NAME"].str.lower() == "germany"]
    if germany.empty:
        raise ValueError("Germany not found in shapefile!")

    germany_m = germany.to_crs(epsg=3035)
    germany_buffered = gpd.GeoSeries(
        germany_m.buffer(buffer_km * 1000), crs=germany_m.crs
    ).to_crs(epsg=4326)

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    points_flat = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points_flat, crs="EPSG:4326")

    mask = points_gdf.within(germany_buffered.unary_union)
    mask_grid = mask.values.reshape(lat_grid.shape)

    print(f"✅ {mask.sum():,} of {mask.size:,} grid points inside Germany (+{buffer_km} km)")
    return mask_grid


def cluster_german_coordinates(latitudes, longitudes, mask_grid, shapefile_path,
                              n_clusters=50, random_state=42, plot=False):
    """Cluster German grid points using KMeans and optionally plot."""
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords_germany = np.column_stack([lon_grid[mask_grid], lat_grid[mask_grid]])
    print(f"🟢 Clustering {len(coords_germany):,} German grid points into {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(coords_germany)
    centroids = km.cluster_centers_

    # Sort by latitude (north→south)
    sort_idx = np.argsort(centroids[:, 1])[::-1]
    centroids_sorted = centroids[sort_idx]
    label_mapping = {old: new for new, old in enumerate(sort_idx)}
    labels_ordered = np.array([label_mapping[l] for l in labels])

    print("✅ KMeans completed and ordered by latitude.")
    print(f"Cluster sizes: {np.bincount(labels_ordered)}")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        world = gpd.read_file(shapefile_path)
        world[world["NAME"] == "Germany"].plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

        sc = ax.scatter(
            coords_germany[:, 0], coords_germany[:, 1],
            c=labels_ordered, cmap="tab20", s=4, alpha=0.6
        )

        for i, (cx, cy) in enumerate(centroids_sorted):
            ax.text(cx, cy, str(i), fontsize=8, fontweight="bold", color="red",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="circle,pad=0.3"))

        plt.colorbar(sc, ax=ax, label="Cluster ID (north→south)")
        plt.title(f"KMeans Clustering of German Grid Points ({n_clusters} clusters)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return coords_germany, labels_ordered, centroids_sorted


def load_cluster_assignment(latitudes, longitudes, cluster_file):
    """Map the raw ICON grid to the repository's fixed clustering parquet."""
    cluster_df = pd.read_parquet(cluster_file)
    required_columns = {"lon", "lat", "cluster_id"}
    if not required_columns.issubset(cluster_df.columns):
        raise ValueError(
            f"{cluster_file} must contain columns {sorted(required_columns)}."
        )

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    precision = 8
    selected_index = pd.MultiIndex.from_arrays([
        cluster_df["lon"].round(precision),
        cluster_df["lat"].round(precision),
    ])
    if not selected_index.is_unique:
        raise ValueError(f"Duplicate lon/lat rows in clustering file: {cluster_file}")
    grid_index = pd.MultiIndex.from_arrays([
        np.round(lon_grid.ravel(), precision),
        np.round(lat_grid.ravel(), precision),
    ])
    positions = selected_index.get_indexer(grid_index)
    mask_flat = positions >= 0
    if int(mask_flat.sum()) != len(cluster_df):
        raise ValueError(
            f"ICON grid and {cluster_file.name} do not match: found "
            f"{int(mask_flat.sum()):,} of {len(cluster_df):,} clustered points."
        )

    cluster_ids = cluster_df["cluster_id"].to_numpy(dtype=int)
    labels = cluster_ids[positions[mask_flat]]
    mask_grid = mask_flat.reshape(lon_grid.shape)
    print(
        f"✅ Loaded fixed C={N_CLUSTERS} assignment for "
        f"{len(labels):,} ICON-D2 grid points."
    )
    return mask_grid, labels


def find_first_grib2_bz2(icon_run_dir: str) -> str | None:
    """Find any first GRIB2.bz2 file inside a run directory to read grid info."""
    var_dirs = sorted([d for d in glob.glob(os.path.join(icon_run_dir, "*")) if os.path.isdir(d)])
    for vdir in var_dirs:
        files = sorted(glob.glob(os.path.join(vdir, "icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2")))
        if files:
            return files[0]
    return None


def read_grid_from_one_file(grib2_bz2_path: str):
    """Read latitude/longitude arrays from a single compressed GRIB2 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_grib = os.path.join(tmpdir, os.path.basename(grib2_bz2_path).replace(".bz2", ""))
        with bz2.open(grib2_bz2_path, "rb") as src, open(tmp_grib, "wb") as dst:
            dst.write(src.read())

        ds = xr.open_dataset(tmp_grib, engine="cfgrib")  # NOTE: no .load()
        lat = ds.latitude.values
        lon = ds.longitude.values
        ds.close()

    return np.array(lat), np.array(lon)


# ====================================================================
# STREAMING VARIABLE PROCESSING
# ====================================================================

def process_variable_streaming(
    var_dir: str,
    mask_germany: np.ndarray,
    cluster_labels: np.ndarray,
):
    """
    Stream one variable folder and return its compact aggregated table.

    Iterates over all compressed GRIB2 files in *var_dir*, decompresses
    each file in a temporary directory, reads one field, aggregates
    immediately to cluster means, and discards the full grid.  Only the
    compact ``(timestamps, cluster_means)`` arrays are held in memory
    across iterations.  Cumulative-type variables (stepType avg / acc /
    mean) are converted to instantaneous flux before writing.

    Parameters
    ----------
    var_dir : str
        Path to the variable folder containing
        ``icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2`` files.
    mask_germany : numpy.ndarray of shape (lat, lon), dtype bool
        Boolean mask selecting ICON-D2 grid points inside Germany
        (plus buffer), as returned by :func:`filter_points_in_germany`.
    cluster_labels : numpy.ndarray of shape (n_germany_points,), dtype int
        Cluster ID (0-based) for every ``True`` point in the flattened
        *mask_germany*, as returned by :func:`cluster_german_coordinates`.
    Returns
    -------
    tuple or None
        ``(field_name, dataframe, metadata)`` for a populated variable
        folder, otherwise ``None``.
    """
    folder_name = os.path.basename(var_dir)
    pattern = os.path.join(var_dir, "icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    # For aggregation
    mask_flat = mask_germany.ravel()
    labels = cluster_labels
    n_clusters = int(np.max(labels)) + 1

    timestamps_list: list[pd.Timestamp] = []
    cluster_series: list[np.ndarray] = []

    # Metadata from first file
    field_name = None
    units = ""
    long_name = ""
    step_type = ""

    print(f"📦 Streaming variable folder: {folder_name} ({len(files)} files)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            tmp_grib = os.path.join(tmpdir, os.path.basename(f).replace(".bz2", ""))
            with bz2.open(f, "rb") as src, open(tmp_grib, "wb") as dst:
                dst.write(src.read())

            ds = xr.open_dataset(tmp_grib, engine="cfgrib")  # no .load()
            fn = list(ds.data_vars)[0]
            if field_name is None:
                field_name = fn
                attrs = ds[field_name].attrs
                long_name = attrs.get("long_name", "") or ""
                units = attrs.get("units", "") or ""
                step_type = (attrs.get("GRIB_stepType", "") or "").lower()

            arr = ds[field_name].values  # loads the data array only
            vt = ds.get("valid_time", None)

            # valid_time can be scalar or array-like
            if vt is None:
                # fallback: try time coordinate
                tvals = ds.coords.get("time", None)
                if tvals is None:
                    raise ValueError(f"No valid_time/time coordinate in {f}")
                tvals = np.atleast_1d(tvals.values)
            else:
                tvals = np.atleast_1d(vt.values)

            ds.close()

            # arr can be 2D (lat, lon) or 3D (time, lat, lon)
            if arr.ndim == 2:
                arr_3d = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr_3d = arr
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} in {f}")

            if arr_3d.shape[0] != len(tvals):
                # Some cfgrib configs may yield mismatch; handle conservatively
                # Use min length to stay consistent.
                min_len = min(arr_3d.shape[0], len(tvals))
                arr_3d = arr_3d[:min_len, ...]
                tvals = tvals[:min_len]

            # Aggregate each timestep slice
            for i in range(arr_3d.shape[0]):
                ts = pd.to_datetime(tvals[i])
                timestamps_list.append(ts)

                slice_2d = arr_3d[i, :, :]
                slice_flat_germany = slice_2d.reshape(-1)[mask_flat]  # only Germany points

                means = np.full(n_clusters, np.nan, dtype=float)
                for c in range(n_clusters):
                    cmask = labels == c
                    if np.any(cmask):
                        means[c] = np.nanmean(slice_flat_germany[cmask])

                cluster_series.append(means)

                # per-timestep cleanup
                del slice_2d, slice_flat_germany, means
                gc.collect()

            # per-file cleanup
            del arr, arr_3d, tvals
            gc.collect()

    # Sort by timestamp to be safe
    ts = pd.to_datetime(timestamps_list)
    values = np.vstack(cluster_series)  # shape (T, n_clusters)

    order = np.argsort(ts.values)
    ts = ts.values[order]
    values = values[order, :]

    ts = pd.to_datetime(ts)

    # Archived inputs can contain repeated valid times.  Collapse them before
    # converting cumulative/average radiation to instantaneous flux so the
    # conversion always sees a strictly increasing time axis.
    ts, values = deduplicate_timesteps(
        ts,
        values,
        variable_name=field_name or folder_name,
    )

    # Compute instantaneous flux if needed
    if any(x in step_type for x in ["avg", "acc", "mean"]):
        print(f"⚡ Computing instantaneous flux for {field_name} (stepType={step_type})")
        values = np.apply_along_axis(lambda y: compute_step_flux(ts, y), 0, values)
        flux_suffix = "_instantaneous"
    else:
        flux_suffix = "_raw"

    unit_safe = safe_unit_for_filename(units)
    first_ts = ts[0].strftime("%Y%m%d%H") if len(ts) else "unknown"

    outname = f"{field_name}_{unit_safe}_{first_ts}{flux_suffix}.csv"
    df = pd.DataFrame(values, index=ts, columns=[f"cluster_{i}" for i in range(n_clusters)])
    df.index.name = "timestamp"
    metadata = {
        "source_folder": folder_name,
        "field_name": field_name,
        "long_name": long_name,
        "units": units,
        "step_type": step_type,
        "transformation": (
            "instantaneous flux" if flux_suffix == "_instantaneous" else "raw"
        ),
        "clusters": n_clusters,
        "first_timestamp": first_ts,
        "former_csv_name": outname,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }
    print(
        f"✅ Aggregated {field_name} "
        f"({df.shape[0]} timesteps × {df.shape[1]} clusters)"
    )
    return field_name, df, metadata


def _migrate_completed_output(
    output_dir: Path,
    issue_date,
    run_hour: str,
    source_folders: list[str],
) -> bool:
    """Convert a complete CSV/XLSX output without touching raw GRIB files."""
    legacy = completed_xlsx_for_migration(output_dir)
    if legacy is None:
        legacy = read_completed_legacy_outputs(output_dir, source_folders)
    if legacy is None or not REQUIRED_OUTPUT_FIELDS.issubset(legacy[0]):
        return False
    run_frames, run_metadata = legacy
    parquet_path, marker_path = write_processed_parquet(
        output_dir,
        run_frames,
        variable_metadata=run_metadata,
        issue_date=issue_date.strftime("%Y%m%d"),
        run_hour=run_hour,
    )
    remove_superseded_outputs(output_dir)
    print(
        f"✅ Migrated legacy output to {parquet_path.name} with "
        f"{len(run_frames)} variables and {marker_path.name}"
    )
    return True


def _process_run(
    issue_date,
    run_hour: str,
    mask_germany,
    cluster_labels,
) -> tuple[object, object, list[str]]:
    day_name = f"dwd_icon_daily_{issue_date:%Y%m%d}"
    output_dir = Path(OUTPUT_PARENT) / f"{day_name}_{run_hour}"
    if SKIP_EXISTING_OUTPUT and parquet_is_complete(output_dir):
        print(
            f"↩️  Skipping completed DWD Parquet file: "
            f"{output_dir / PARQUET_FILENAME}"
        )
        return mask_germany, cluster_labels, []

    source_folders = available_variables(
        LSDF_BASE,
        issue_date,
        run_hour,
        VARIABLES,
    )
    print(
        f"📁 Found {len(source_folders)} variable sources to process in "
        f"{day_name}/{run_hour}."
    )
    if SKIP_EXISTING_OUTPUT and _migrate_completed_output(
        output_dir, issue_date, run_hour, source_folders
    ):
        return mask_germany, cluster_labels, []

    run_errors: list[str] = []
    with materialize_raw_run(
        LSDF_BASE,
        issue_date,
        run_hour,
        VARIABLES,
    ) as icon_dir:
        if mask_germany is None or cluster_labels is None:
            first_file = find_first_grib2_bz2(str(icon_dir))
            if first_file is None:
                error = f"{day_name}/{run_hour}: no regular-grid GRIB2.bz2 files"
                print(f"❌ {error}")
                return mask_germany, cluster_labels, [error]
            print("🌍 Reading grid from first available file (one-time init)...")
            latitude, longitude = read_grid_from_one_file(first_file)
            print("🧩 Loading fixed repository cluster assignment (one-time init)...")
            mask_germany, cluster_labels = load_cluster_assignment(
                latitude, longitude, CLUSTER_FILE
            )
        else:
            print("♻️ Reusing existing grid, mask, and clusters.")

        var_dirs = [
            icon_dir / variable
            for variable in VARIABLES
            if (icon_dir / variable).is_dir()
        ]
        run_frames = {}
        run_metadata = {}
        for var_dir in var_dirs:
            try:
                result = process_variable_streaming(
                    var_dir=str(var_dir),
                    mask_germany=mask_germany,
                    cluster_labels=cluster_labels,
                )
                if result is None:
                    continue
                field_name, frame, metadata = result
                run_frames[field_name] = frame
                run_metadata[field_name] = metadata
            except Exception as exc:
                print(f"❌ Error processing variable '{var_dir.name}': {exc}")
                run_errors.append(f"{day_name}/{run_hour}/{var_dir.name}: {exc}")
            finally:
                gc.collect()

    missing_required = sorted(REQUIRED_OUTPUT_FIELDS - set(run_frames))
    if missing_required:
        run_errors.append(
            f"{day_name}/{run_hour}: missing required processed variables "
            f"{missing_required}"
        )
    if run_errors:
        print(
            f"❌ Parquet file not written for {day_name}/{run_hour} because "
            f"{len(run_errors)} variable(s) failed."
        )
        return mask_germany, cluster_labels, run_errors

    parquet_path, marker_path = write_processed_parquet(
        output_dir,
        run_frames,
        variable_metadata=run_metadata,
        issue_date=issue_date.strftime("%Y%m%d"),
        run_hour=run_hour,
    )
    remove_superseded_outputs(output_dir)
    print(
        f"✅ Saved {parquet_path.name} with {len(run_frames)} variables "
        f"and {marker_path.name}"
    )
    print(f"✅ Finished processing {day_name} — {run_hour}")
    return mask_germany, cluster_labels, []


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    issue_dates = [
        value
        for value in discover_issue_dates(LSDF_BASE)
        if START_DATE.date() <= value <= END_DATE.date()
        and (ONLY_DAY is None or value.strftime("%Y%m%d") == ONLY_DAY)
    ]
    print(f"\n📅 Found {len(issue_dates)} daily ICON sources.")
    if not issue_dates:
        raise FileNotFoundError(
            f"No directory-based or ZIP-based ICON-D2 sources found for "
            f"{START_DATE.date()} through {END_DATE.date()} below {LSDF_BASE}."
        )

    mask_germany = None
    cluster_labels = None
    processing_errors: list[str] = []
    for issue_date in issue_dates:
        day_name = f"dwd_icon_daily_{issue_date:%Y%m%d}"
        print(f"\n{'='*80}\n🗓️  Processing day: {day_name}\n{'='*80}")
        run_hours = available_run_hours(LSDF_BASE, issue_date)
        if ONLY_RUN_HOUR is not None:
            run_hours = [hour for hour in run_hours if hour == ONLY_RUN_HOUR]
        if not run_hours:
            print(f"⚠️ No requested ICON run found for {issue_date}, skipping.")
            continue
        for run_hour in run_hours:
            print(f"\n🕐 Forecast run hour: {run_hour}")
            mask_germany, cluster_labels, errors = _process_run(
                issue_date,
                run_hour,
                mask_germany,
                cluster_labels,
            )
            processing_errors.extend(errors)
            gc.collect()

    if processing_errors:
        details = "\n".join(f"  - {error}" for error in processing_errors)
        raise RuntimeError(
            f"ICON-D2 preprocessing failed for {len(processing_errors)} variable folder(s):\n{details}"
        )

    print("\n🎉 All requested ICON-D2 directories processed successfully (streaming mode).")
