#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANC Spatial Heatmap
===================

Visualizes the Average Normalized Contribution (ANC) of ERA5 weather clusters
as a choropleth map over Germany -- separately for Wind and Solar.

Cluster polygons are reconstructed from the KMeans clustering parquet file
(identical logic to Clustering_Extraction_final.py), then colored by ANC value.

Inputs
------
- data/clustering/icon_d2_clustering_c5.parquet       cluster grid points (lon, lat, cluster_id)
- results/lear_anc_results/.../anc_wind_feature_results.csv   ANC per wind cluster
- results/lear_anc_results/.../anc_solar_feature_results.csv  ANC per solar cluster
- data/shapefile/ne_10m_admin_0_countries.shp         Germany outline

Output
------
- output/anc_heatmap_wind.pdf
- output/anc_heatmap_solar.pdf
"""

# ===============================================================
# IMPORTS
# ===============================================================
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import box
import sys

# ===============================================================
# CONFIGURATION
# ===============================================================

BASE_DIR = Path(__file__).parent.parent  # points to repository root
sys.path.insert(0, str(BASE_DIR))
from experiment_config import load_experiment_config
EXPERIMENT = load_experiment_config(BASE_DIR)

# Input paths
PARQUET_PATH   = BASE_DIR / "data" / "clustering" / "icon_d2_clustering_c5.parquet"
ANC_WIND_PATH  = EXPERIMENT.results_root / "lear_anc_results" / "era5" / "c5" / "d112" / "fundamental" / "anc_wind_feature_results.csv"
ANC_SOLAR_PATH = EXPERIMENT.results_root / "lear_anc_results" / "era5" / "c5" / "d112" / "fundamental" / "anc_solar_feature_results.csv"
SHAPEFILE_PATH = BASE_DIR / "data" / "shapefile" / "ne_10m_admin_0_countries.shp"

# Output paths
OUTPUT_DIR        = EXPERIMENT.output_root
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_WIND_PATH  = OUTPUT_DIR / "anc_heatmap_wind.pdf"
OUTPUT_SOLAR_PATH = OUTPUT_DIR / "anc_heatmap_solar.pdf"

# Plot configuration
TRAIN_DAYS = 112
CMAP_WIND  = "YlOrRd"
CMAP_SOLAR = "YlOrRd"
DPI        = 300


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def build_cluster_polygons(parquet_path: Path) -> gpd.GeoDataFrame:
    """
    Load cluster grid points from parquet and reconstruct cluster polygons.

    Identical logic to Clustering_Extraction_final.py:
    - Grid cells are built in native lon/lat space (EPSG:4326)
    - buffer(0.001).buffer(-0.001) merges shared boundaries in WGS84
      to prevent MultiPolygon split artefacts after reprojection
    - Final geometries are in EPSG:3035

    Returns a GeoDataFrame in EPSG:3035 with columns:
        cluster_id, geometry
    """
    df = pd.read_parquet(parquet_path)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    # Build grid cells in native lon/lat space
    lon_vals = np.sort(df["lon"].unique())
    lat_vals = np.sort(df["lat"].unique())

    dlon = np.median(np.diff(lon_vals))
    dlat = np.median(np.diff(lat_vals))

    half_dlon = dlon / 2
    half_dlat = dlat / 2

    gdf["geometry"] = gdf.apply(
        lambda row: box(
            row["lon"] - half_dlon,
            row["lat"] - half_dlat,
            row["lon"] + half_dlon,
            row["lat"] + half_dlat,
        ),
        axis=1,
    )

    # Dissolve in WGS84 with buffer fix, then project to EPSG:3035
    cluster_gdf = (
        gdf
        .dissolve(by="cluster_id", as_index=False)
        [["cluster_id", "geometry"]]
    )

    cluster_gdf["geometry"] = cluster_gdf["geometry"].buffer(0.001).buffer(-0.001)
    cluster_gdf = gpd.GeoDataFrame(cluster_gdf, geometry="geometry", crs="EPSG:4326")
    cluster_gdf = cluster_gdf.to_crs(epsg=3035)
    cluster_gdf["geometry"] = cluster_gdf["geometry"].buffer(0)

    return cluster_gdf


def load_anc(csv_path: Path, train_days: int) -> pd.DataFrame:
    """
    Load ANC summary CSV and filter to the specified training window.

    Expected columns: train_days, cluster_group, cluster_id, ANC
    Returns DataFrame with columns: cluster_id, cluster_group, ANC
    """
    df = pd.read_csv(csv_path)
    df = df[df["train_days"] == train_days].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for train_days={train_days} in {csv_path}. "
            f"Available: {pd.read_csv(csv_path)['train_days'].unique()}"
        )

    return df[["cluster_id", "cluster_group", "ANC"]].reset_index(drop=True)


def load_germany(shapefile_path: Path, target_crs) -> gpd.GeoDataFrame:
    """
    Load Germany outline from Natural Earth shapefile and reproject.
    """
    world = gpd.read_file(shapefile_path)
    germany = world[world["NAME"].str.lower() == "germany"].to_crs(target_crs)

    if germany.empty:
        raise RuntimeError("Germany not found in shapefile.")

    return germany


def plot_anc_heatmap(
    cluster_gdf: gpd.GeoDataFrame,
    anc_df: pd.DataFrame,
    germany: gpd.GeoDataFrame,
    title: str,
    output_path: Path,
    cmap: str = "YlOrRd",
    dpi: int = 300,
) -> None:
    """
    Plot ANC choropleth map over Germany and save as PDF.

    Parameters
    ----------
    cluster_gdf : GeoDataFrame
        Cluster polygons with column 'cluster_id'.
    anc_df : DataFrame
        ANC values with columns 'cluster_id', 'cluster_group', 'ANC'.
    germany : GeoDataFrame
        Germany outline for geographic reference.
    title : str
        Plot title.
    output_path : Path
        Output file path (.pdf).
    cmap : str
        Matplotlib colormap name.
    dpi : int
        Output resolution.
    """

    # --------------------------------------------------
    # Join ANC values onto cluster polygons
    # --------------------------------------------------
    gdf = cluster_gdf.merge(anc_df, on="cluster_id", how="left")

    if gdf["ANC"].isna().any():
        missing = gdf[gdf["ANC"].isna()]["cluster_id"].tolist()
        raise ValueError(f"Missing ANC values for cluster_ids: {missing}")

    # --------------------------------------------------
    # Colormap setup
    # --------------------------------------------------
    vmin = gdf["ANC"].min()
    vmax = gdf["ANC"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.solid_capstyle"] = "round"

    fig, ax = plt.subplots(figsize=(5.5, 7))

    # --------------------------------------------------
    # Plot cluster polygons colored by ANC
    # --------------------------------------------------
    gdf.explode(index_parts=False).plot(
        ax=ax,
        column="ANC",
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    # --------------------------------------------------
    # Plot cluster boundaries
    # --------------------------------------------------
    gdf_exploded = gdf.explode(index_parts=False)
    outer = gdf_exploded.dissolve(by="cluster_id").boundary

    outer.plot(
        ax=ax,
        edgecolor="#4d4d4d",
        linewidth=0.5,
        zorder=3,
    )

    # --------------------------------------------------
    # Cluster labels (bold ANC value in white circle)
    # --------------------------------------------------
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        label = f"{row['ANC']:.2f}"
        ax.annotate(
            label,
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=5,
            bbox=dict(
                boxstyle="circle,pad=0.4",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    # --------------------------------------------------
    # Germany border overlay
    # --------------------------------------------------
    germany.boundary.plot(
        ax=ax,
        linewidth=1.3,
        edgecolor="black",
        zorder=4,
    )

    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ANC", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # --------------------------------------------------
    # Title & cleanup
    # --------------------------------------------------
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


# ===============================================================
# MAIN
# ===============================================================

def main():

    print("📂 Loading cluster polygons ...")
    cluster_gdf = build_cluster_polygons(PARQUET_PATH)

    print("🗺️  Loading Germany outline ...")
    germany = load_germany(SHAPEFILE_PATH, target_crs=cluster_gdf.crs)

    # --------------------------------------------------
    # Wind Heatmap
    # --------------------------------------------------
    print(f"\n💨 Wind ANC Heatmap (train_days={TRAIN_DAYS}) ...")
    anc_wind = load_anc(ANC_WIND_PATH, train_days=TRAIN_DAYS)
    print(anc_wind.to_string(index=False))

    plot_anc_heatmap(
        cluster_gdf=cluster_gdf,
        anc_df=anc_wind,
        germany=germany,
        title=f"Wind Cluster Importance by ANC (train_days={TRAIN_DAYS})",
        output_path=OUTPUT_WIND_PATH,
        cmap=CMAP_WIND,
        dpi=DPI,
    )

    # --------------------------------------------------
    # Solar Heatmap
    # --------------------------------------------------
    print(f"\n☀️  Solar ANC Heatmap (train_days={TRAIN_DAYS}) ...")
    anc_solar = load_anc(ANC_SOLAR_PATH, train_days=TRAIN_DAYS)
    print(anc_solar.to_string(index=False))

    plot_anc_heatmap(
        cluster_gdf=cluster_gdf,
        anc_df=anc_solar,
        germany=germany,
        title=f"Solar Cluster Importance by ANC (train_days={TRAIN_DAYS})",
        output_path=OUTPUT_SOLAR_PATH,
        cmap=CMAP_SOLAR,
        dpi=DPI,
    )

    print("\n✅ All heatmaps completed.")


if __name__ == "__main__":
    main()
