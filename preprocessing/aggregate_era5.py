#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_era5.py
=================

Aggregates ERA5 weather data to ICON-D2 spatial clusters and saves
the results as CSV files.

Solar handling
--------------
ERA5 ssrd / fdir have dims (time, step, lat, lon) where each step
represents one hourly interval energy [J/m²]. The (time, step)
dimensions are flattened to a true hourly time axis via valid_time,
and values are converted from J/m² to W/m² by division by 3600.
No differencing, no accumulation, no resets.
"""

# ====================================================================
# IMPORTS
# ====================================================================
import gc
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


# Do not let optional Unicode status symbols abort long Windows runs when the
# active console uses a legacy encoding such as cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


# ====================================================================
# CONFIG
# ====================================================================
BASE_DIR = Path(__file__).parent.parent  # points to repository root
sys.path.insert(0, str(BASE_DIR))
from experiment_config import load_experiment_config

EXPERIMENT = load_experiment_config(BASE_DIR)

# Output year and cluster count.  The historical launcher overrides both for
# each required combination.
ERA5_PREPROCESS_YEAR = int(os.getenv("ERA5_PREPROCESS_YEAR", "2026"))
N_CLUSTERS = int(os.getenv("ERA5_N_CLUSTERS", "25"))

# Path to the raw ERA5 GRIB files.
# These files are not included in the repository due to file size.
# By default they are resolved from ERA5_RAW_ARCHIVE and the processing year.
ERA5_RAW_ARCHIVE = Path(os.getenv("ERA5_RAW_ARCHIVE", "/path/to/raw/ERA5"))
ERA5_MAIN_FILE = Path(os.getenv(
    "ERA5_MAIN_FILE", str(ERA5_RAW_ARCHIVE / f"era5_main_{ERA5_PREPROCESS_YEAR}.grib")
))
ERA5_SOLAR_FILE = Path(os.getenv(
    "ERA5_SOLAR_FILE", str(ERA5_RAW_ARCHIVE / f"era5_solar_{ERA5_PREPROCESS_YEAR}.grib")
))

CLUSTER_FILE = BASE_DIR / "data" / "clustering" / f"icon_d2_clustering_c{N_CLUSTERS}.parquet"

# Output directory for aggregated CSVs.
# Note: aggregated data is stored outside the repository due to file size.
# Adjust to your preferred output location.
OUTPUT_PARENT = EXPERIMENT.era5_year_dir(N_CLUSTERS, ERA5_PREPROCESS_YEAR)

# Adjust as needed:
MAIN_VARIABLES  = ["u10", "v10", "t2m", "sp", "sd"]
SOLAR_VARIABLES = ["ssrd", "fdir"]

# Adjust as needed:
_year_start = date(ERA5_PREPROCESS_YEAR, 1, 1)
_year_end = date(ERA5_PREPROCESS_YEAR, 12, 31)
START_TIME = pd.Timestamp(max(_year_start, EXPERIMENT.weather_training_start))
END_TIME = pd.Timestamp(min(_year_end, EXPERIMENT.evaluation_end)) + pd.Timedelta(hours=23, minutes=45)

# ====================================================================
# HELPERS
# ====================================================================
def safe_unit_for_filename(units: str) -> str:
    """
    Convert a physical unit string into a filesystem-safe representation.

    Replaces characters that are illegal or awkward in file names with
    safe alternatives (e.g. ``"/"`` → ``"-per-"``, ``"²"`` → ``"2"``).

    Parameters
    ----------
    units : str
        Unit string as returned by xarray variable attributes
        (e.g. ``"J m-2"``, ``"W/m²"``).

    Returns
    -------
    str
        Sanitised string suitable for use in a file name.
    """
    return (
        str(units)
        .replace("/", "-per-")
        .replace(" ", "_")
        .replace("*", "")
        .replace("²", "2")
    )


def build_era5_cluster_grid(cluster_df, lats, lons):
    """
    Map every ERA5 grid point to the nearest ICON-D2 cluster centroid.

    Uses a KD-tree nearest-neighbour search in lon/lat space to assign
    each ERA5 (lat, lon) cell to a cluster ID from the ICON-D2 clustering.

    Parameters
    ----------
    cluster_df : pandas.DataFrame
        DataFrame with columns ``"lon"``, ``"lat"``, and ``"cluster_id"``
        representing the ICON-D2 cluster centroids.
    lats : numpy.ndarray of shape (Y,)
        Latitude values of the ERA5 grid.
    lons : numpy.ndarray of shape (X,)
        Longitude values of the ERA5 grid.

    Returns
    -------
    numpy.ndarray of shape (Y, X)
        Integer cluster IDs, one per ERA5 grid cell.
    """
    icon_coords = cluster_df[["lon", "lat"]].to_numpy()
    icon_ids = cluster_df["cluster_id"].to_numpy()
    tree = cKDTree(icon_coords)

    lon_g, lat_g = np.meshgrid(lons, lats)
    era5_coords = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    _, idx = tree.query(era5_coords)

    return icon_ids[idx].reshape(lat_g.shape)


def extract_main_values_and_times(da: xr.DataArray):
    """
    Extract raw values and timestamps from a standard ERA5 DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        ERA5 variable with a single ``"time"`` dimension
        (shape: ``(T, lat, lon)``).

    Returns
    -------
    values : numpy.ndarray of shape (T, lat, lon)
        Raw variable values.
    times : pandas.DatetimeIndex
        Timestamps corresponding to the time dimension.
    """
    return da.values, pd.to_datetime(da.time.values)


def extract_solar_values_and_times(ds: xr.Dataset, da: xr.DataArray):
    """
    Flatten the (time, step) dimensions of an ERA5 solar DataArray.

    ERA5 solar variables (ssrd, fdir) have shape ``(time, step, lat, lon)``
    where each step represents one hourly interval energy in J/m².  This
    function merges the two leading dimensions into a single true hourly
    time axis derived from ``valid_time``.

    Parameters
    ----------
    ds : xarray.Dataset
        Parent dataset containing the ``"valid_time"`` coordinate with
        shape ``(T, S)``.
    da : xarray.DataArray
        Solar variable with shape ``(T, S, lat, lon)``.

    Returns
    -------
    values : numpy.ndarray of shape (T*S, lat, lon)
        Flattened interval energies in J/m².
    times : pandas.DatetimeIndex of length T*S
        True hourly timestamps, shifted back by one hour so that each
        value is labelled with the start of its accumulation interval.
    """
    vals = da.values  # (T, S, lat, lon)
    T, S, Y, X = vals.shape

    values = vals.reshape(T * S, Y, X)
    # Implementation for: solar_rate (t) = solar_energy( t to t+1) / 3600
    times = pd.to_datetime(ds["valid_time"].values.reshape(T * S)) - pd.Timedelta(hours=1)

    return values, times


def aggregate_to_clusters(values_tll, cluster_grid, n_clusters):
    """
    Average ERA5 grid-point values within each cluster for every timestep.

    Parameters
    ----------
    values_tll : numpy.ndarray of shape (T, lat, lon)
        ERA5 variable values on the full grid.
    cluster_grid : numpy.ndarray of shape (lat, lon)
        Integer cluster ID for each ERA5 grid cell, as returned by
        :func:`build_era5_cluster_grid`.
    n_clusters : int
        Total number of clusters (``max(cluster_id) + 1``).

    Returns
    -------
    numpy.ndarray of shape (T, n_clusters)
        Spatially averaged values per cluster and timestep.  Clusters
        with no grid points remain ``NaN``.
    """
    T = values_tll.shape[0]
    out = np.full((T, n_clusters), np.nan)

    for c in range(n_clusters):
        mask = cluster_grid == c
        if np.any(mask):
            out[:, c] = np.nanmean(values_tll[:, mask], axis=1)

    return out


def write_icon_style_csv(outpath, times, out, var, long_name, units, n_clusters, transformation):
    """
    Write cluster-aggregated ERA5 data to a CSV file with a metadata header.

    The file begins with comment lines (prefixed ``"#"``) describing the
    source, variable, units, and transformation applied.  The data block
    follows in standard CSV format with a ``"timestamp"`` index column and
    one column per cluster (``"cluster_0"``, ``"cluster_1"``, …).

    Parameters
    ----------
    outpath : pathlib.Path
        Destination file path.  Parent directories are created
        automatically if they do not exist.
    times : pandas.DatetimeIndex
        Timestamps for each row of *out*.
    out : numpy.ndarray of shape (T, n_clusters)
        Aggregated values to write.
    var : str
        Short ERA5 variable name (e.g. ``"u10"``).
    long_name : str
        Human-readable variable description from xarray attributes.
    units : str
        Physical unit string (after any transformation).
    n_clusters : int
        Number of clusters, used to generate column names.
    transformation : str
        Description of any transformation applied to the raw ERA5 values
        (e.g. ``"raw"`` or ``"hourly mean flux (ERA5 interval energy)"``).

    Returns
    -------
    None
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(out, index=times, columns=[f"cluster_{i}" for i in range(n_clusters)])
    df.index.name = "timestamp"

    header = [
        "# Source: ERA5",
        f"# Variable: {var}",
        f"# Long name: {long_name}",
        f"# Units: {units}",
        f"# Transformation: {transformation}",
        f"# Clusters: {n_clusters}",
    ]

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f)

    print(f"✅ Saved {outpath.name} ({df.shape[0]} × {df.shape[1]})")


# ====================================================================
# PROCESSING
# ====================================================================
def process_dataset(ds, variables, cluster_df, n_clusters, solar_mode, start_time, end_time):
    """
    Process one ERA5 dataset: aggregate all variables to clusters and save CSVs.

    For each variable in *variables*, the function extracts values and
    timestamps, spatially aggregates them to cluster means, filters to the
    configured time window, applies any solar unit conversion, and writes
    the result to a CSV file in ``OUTPUT_PARENT``.

    Parameters
    ----------
    ds : xarray.Dataset
        Opened ERA5 dataset (main or solar).
    variables : list of str
        ERA5 variable names to process (e.g. ``["u10", "v10"]``).
    cluster_df : pandas.DataFrame
        ICON-D2 cluster centroids with columns ``"lon"``, ``"lat"``,
        ``"cluster_id"``.
    n_clusters : int
        Total number of clusters.
    solar_mode : bool
        If ``True``, treat the dataset as a solar radiation dataset:
        flatten the ``(time, step)`` dimensions and convert J/m² to W/m².
        If ``False``, use the raw ``time`` dimension directly.
    start_time : pandas.Timestamp
        Inclusive start of the output time window.
    end_time : pandas.Timestamp
        Inclusive end of the output time window.

    Returns
    -------
    None
    """
    lats = ds.latitude.values
    lons = ds.longitude.values
    cluster_grid = build_era5_cluster_grid(cluster_df, lats, lons)

    for var in variables:
        print(f"\n📦 Variable: {var}")
        da = ds[var]

        if solar_mode:
            values_tll, times = extract_solar_values_and_times(ds, da)
        else:
            values_tll, times = extract_main_values_and_times(da)

        out = aggregate_to_clusters(values_tll, cluster_grid, n_clusters)

        mask = (times >= start_time) & (times <= end_time)
        
        if not np.any(mask):
            raise RuntimeError(
                f"No timestamps >= {start_time} found in variable {var}."
            )
        times = times[mask]
        out = out[mask]

        if solar_mode:
            out = np.nan_to_num(out) / 3600.0
            units = "W m-2"
            transformation = "hourly mean flux (ERA5 interval energy)"
        else:
            units = da.attrs.get("units", "")
            transformation = "raw"

        outname = f"{var}_{safe_unit_for_filename(units)}_{times[0].strftime('%Y%m%d%H')}.csv"
        write_icon_style_csv(
            OUTPUT_PARENT / outname,
            times,
            out,
            var,
            da.attrs.get("long_name", var),
            units,
            n_clusters,
            transformation,
        )

        del values_tll, out
        gc.collect()


# ====================================================================
# MAIN
# ====================================================================
def main():
    """
    Entry point: load inputs, run processing for main and solar datasets.

    Creates the output directory if it does not exist, reads the ICON-D2
    cluster parquet file to determine the number of clusters, then
    processes the main ERA5 dataset (wind, temperature, pressure, snow)
    followed by the solar ERA5 dataset (ssrd, fdir).

    Returns
    -------
    None
    """
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)

    cluster_df = pd.read_parquet(CLUSTER_FILE)
    n_clusters = int(cluster_df["cluster_id"].max()) + 1

    ds_main = xr.open_dataset(
        ERA5_MAIN_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""}
    )
    process_dataset(ds_main, MAIN_VARIABLES, cluster_df, n_clusters, solar_mode=False,
                    start_time=START_TIME, end_time=END_TIME)
    ds_main.close()

    ds_solar = xr.open_dataset(
        ERA5_SOLAR_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""}
    )
    process_dataset(ds_solar, SOLAR_VARIABLES, cluster_df, n_clusters, solar_mode=True,
                    start_time=START_TIME, end_time=END_TIME)
    ds_solar.close()

    print("\n🎉 ERA5 → ICON aggregation finished (hourly & correct).")


if __name__ == "__main__":
    main()
