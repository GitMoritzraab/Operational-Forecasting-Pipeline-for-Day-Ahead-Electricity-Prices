"""Reusable data, feature-engineering, LEAR, and ANC model functions.

The functions in this module are the script-friendly implementation of the
original interactive calculations. Configuration-dependent
values are passed explicitly so importing the module has no filesystem,
network, or experiment-execution side effects.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import holidays
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.parsers import parse_prices
from scipy.stats import norm
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from pipeline.delivery_index import (
    INDEX_COLUMNS,
    MTUS_PER_DAY,
    delivery_dates,
    make_delivery_index,
    validate_delivery_index,
)
from pipeline.dwd_processed import read_processed_output
from pipeline.dwd_history import history_is_complete, load_consolidated_dwd


COUNTRY_CODE_ENTSOE = "DE_LU"
DEFAULT_TARGET_TZ = "Europe/Berlin"
POST_REGIME_START = date(2025, 10, 1)
DWD_FOLDER_OFFSET_DATE = date(2025, 10, 26)
ICON_REQUIRED_RUN = "09"
Z_075 = norm.ppf(0.75)


def fetch_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    api_key: str,
    target_tz: str = DEFAULT_TARGET_TZ,
    country_code: str = COUNTRY_CODE_ENTSOE,
) -> pd.DataFrame:
    """Fetch DE-LU SDAC prices and return a 15-minute aligned series."""
    client = EntsoePandasClient(api_key=api_key)
    query_start = start_day.tz_convert(target_tz)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz)

    raw_series = client.query_day_ahead_prices(
        country_code,
        start=query_start,
        end=query_end,
    )
    if raw_series.empty:
        raise ValueError("No day-ahead price data returned by the ENTSO-E API.")

    raw_series = raw_series.tz_convert(target_tz).rename("price_da")
    full_index = pd.date_range(
        start=raw_series.index.min().normalize(),
        end=(
            raw_series.index.max().normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=15)
        ),
        freq="15min",
        tz=target_tz,
    )
    series_15 = raw_series.reindex(full_index).ffill(limit=3)

    expected_counts = pd.Series(1, index=full_index).groupby(
        full_index.normalize()
    ).transform("count")
    actual_counts = series_15.groupby(series_15.index.normalize()).transform("count")
    series_15 = series_15.where(actual_counts >= expected_counts)

    start_cut = start_day.tz_convert(target_tz).normalize()
    end_cut = (
        end_day.tz_convert(target_tz).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(minutes=15)
    )
    return (
        series_15.loc[start_cut:end_cut]
        .to_frame(name="price_da")
        .rename_axis("timestamp")
        .sort_index()
    )


def fetch_prices_exaa(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    api_key: str,
    target_tz: str = DEFAULT_TARGET_TZ,
    country_code: str = COUNTRY_CODE_ENTSOE,
) -> pd.DataFrame:
    """Fetch EXAA prices (ENTSO-E sequence 2) in 90-day chunks."""
    client = EntsoeRawClient(api_key=api_key)
    chunk_days = 90
    all_series: List[pd.Series] = []
    current_start = start_day.normalize()

    while current_start <= end_day:
        current_end = min(
            current_start + pd.Timedelta(days=chunk_days - 1), end_day
        )
        query_start = current_start.tz_convert(target_tz)
        query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)
        xml = client.query_day_ahead_prices(
            country_code,
            start=query_start,
            end=query_end,
            sequence=2,
        )
        parsed = parse_prices(xml)

        if isinstance(parsed, dict):
            chunk_series = (
                next(iter(parsed.values()))
                if len(parsed) == 1
                else max(parsed.values(), key=len)
            )
        else:
            chunk_series = parsed

        if chunk_series is None or len(chunk_series) == 0:
            current_start = current_end + pd.Timedelta(days=1)
            continue
        if getattr(chunk_series.index, "tz", None) is None:
            chunk_series = chunk_series.tz_localize("UTC")

        all_series.append(
            chunk_series.tz_convert(target_tz).rename("price_exaa")
        )
        current_start = current_end + pd.Timedelta(days=1)

    if not all_series:
        raise ValueError("No EXAA price data returned by the ENTSO-E API.")

    series = pd.concat(all_series).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    full_index = pd.date_range(
        start=series.index.min().normalize(),
        end=(
            series.index.max().normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=15)
        ),
        freq="15min",
        tz=target_tz,
    )
    series_15 = series.reindex(full_index).ffill(limit=3)

    expected_counts = pd.Series(1, index=full_index).groupby(
        full_index.normalize()
    ).transform("count")
    actual_counts = series_15.groupby(series_15.index.normalize()).transform("count")
    series_15 = series_15.where(actual_counts >= expected_counts)

    start_cut = start_day.tz_convert(target_tz).normalize()
    end_cut = (
        end_day.tz_convert(target_tz).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(minutes=15)
    )
    return (
        series_15.loc[start_cut:end_cut]
        .to_frame(name="price_exaa")
        .rename_axis("timestamp")
        .sort_index()
    )


def fetch_load_forecast(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    api_key: str,
    target_tz: str = DEFAULT_TARGET_TZ,
    country_code: str = COUNTRY_CODE_ENTSOE,
) -> pd.DataFrame:
    """Fetch the DE-LU day-ahead load forecast at 15-minute resolution."""
    client = EntsoePandasClient(api_key=api_key)
    query_start = start_day.tz_convert(target_tz)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz)
    obj = client.query_load_forecast(
        country_code,
        start=query_start,
        end=query_end,
    )

    if isinstance(obj, pd.DataFrame):
        numeric_cols = obj.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "No numeric columns found in load forecast response: "
                f"{obj.columns.tolist()}"
            )
        series = obj[numeric_cols[0]].copy()
    else:
        series = obj.copy()

    series = series.tz_convert(target_tz)
    series.name = "load_fc"
    full_index = pd.date_range(
        start=series.index.min().normalize(),
        end=(
            series.index.max().normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=15)
        ),
        freq="15min",
        tz=target_tz,
    )
    series_15 = series.reindex(full_index).ffill(limit=3)

    expected_counts = pd.Series(1, index=full_index).groupby(
        full_index.normalize()
    ).transform("count")
    actual_counts = series_15.groupby(series_15.index.normalize()).transform("count")
    series_15 = series_15.where(actual_counts >= expected_counts)

    start_cut = start_day.tz_convert(target_tz).normalize()
    end_cut = (
        end_day.tz_convert(target_tz).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(minutes=15)
    )
    return (
        series_15.loc[start_cut:end_cut]
        .to_frame(name="load_fc")
        .rename_axis("timestamp")
        .sort_index()
    )


def load_era5(
    dirs: Sequence[Path],
    target_tz: str = DEFAULT_TARGET_TZ,
) -> pd.DataFrame:
    """Load and merge clustered ERA5 CSVs from multiple yearly folders."""
    yearly_dfs: List[pd.DataFrame] = []
    for base_dir in dirs:
        csv_files = sorted(
            filename for filename in os.listdir(base_dir) if filename.endswith(".csv")
        )
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {base_dir}")

        variable_dfs: List[pd.DataFrame] = []
        for filename in csv_files:
            var_name = filename.split("_")[0]
            df_var = pd.read_csv(
                base_dir / filename,
                comment="#",
                parse_dates=["timestamp"],
            ).set_index("timestamp")
            df_var = df_var.rename(
                columns={col: f"{var_name}_{col}" for col in df_var.columns}
            )
            variable_dfs.append(df_var)

        df_year = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_year = df_year.join(df_next, how="inner")
        yearly_dfs.append(df_year)

    df_era5 = pd.concat(yearly_dfs).sort_index()
    df_era5 = df_era5.loc[~df_era5.index.duplicated(keep="first")]
    if df_era5.index.tz is None:
        df_era5.index = df_era5.index.tz_localize("UTC")
    df_era5.index = df_era5.index.tz_convert(target_tz)
    df_era5.index.name = "timestamp"
    return df_era5


def load_dwd(
    icon_dir: Path,
    start_folder_date: date,
    required_run: str,
    skip_dates: Optional[Set[date]] = None,
    folder_offset_date: date = DWD_FOLDER_OFFSET_DATE,
    target_tz: str = DEFAULT_TARGET_TZ,
    end_folder_date: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed ICON-D2 hourly wind and quarter-hourly solar data."""
    if not icon_dir.exists():
        raise FileNotFoundError(f"ICON directory not found: {icon_dir}")
    skipped = skip_dates or set()
    if history_is_complete(icon_dir):
        return load_consolidated_dwd(
            icon_dir,
            start_issue_date=start_folder_date,
            end_issue_date=end_folder_date,
            required_run=required_run,
            skipped_issue_dates=tuple(skipped),
            target_tz=target_tz,
        )
    hourly_dfs: List[pd.DataFrame] = []
    qh_dfs: List[pd.DataFrame] = []

    for folder_name in sorted(os.listdir(icon_dir)):
        folder_path = icon_dir / folder_name
        if not folder_path.is_dir():
            continue
        try:
            date_str = folder_name.split("_")[3]
            folder_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue
        folder_run = folder_name.rsplit("_", 1)[-1]
        if folder_run.isdigit() and len(folder_run) == 2 and folder_run != required_run:
            continue
        if (
            folder_date < start_folder_date
            or (end_folder_date is not None and folder_date > end_folder_date)
            or folder_date in skipped
        ):
            continue

        forecast_date = (
            folder_date
            if folder_date < folder_offset_date
            else folder_date + timedelta(days=1)
        )
        variable_dfs: List[pd.DataFrame] = []
        processed_frames = read_processed_output(folder_path)
        if processed_frames is not None:
            variable_inputs = list(processed_frames.items())
        else:
            variable_inputs = []
            for filename in sorted(os.listdir(folder_path)):
                if not filename.endswith(".csv"):
                    continue
                parts = filename.replace(".csv", "").split("_")
                run_hour = next(
                    (
                        token[-2:]
                        for token in parts
                        if token.isdigit() and len(token) == 10
                    ),
                    "",
                )
                if run_hour != required_run:
                    continue
                variable_inputs.append(
                    (
                        parts[0],
                        pd.read_csv(
                            folder_path / filename,
                            comment="#",
                            sep=",",
                            engine="python",
                        ),
                    )
                )

        for var_name, df_var in variable_inputs:
            if "timestamp" not in df_var.columns:
                raise ValueError(
                    f"Missing 'timestamp' column for {var_name} in {folder_path}"
                )
            feature_prefix = {
                "ASWDIR_S": "ASWDIR",
                "ASWDIFD_S": "ASWDIFD",
            }.get(var_name, var_name)
            df_var = df_var.rename(
                columns={
                    col: f"{feature_prefix}_{col}"
                    for col in df_var.columns
                    if col.startswith("cluster_")
                }
            )
            df_var["timestamp"] = pd.to_datetime(
                df_var["timestamp"], utc=True
            ).dt.tz_convert(target_tz)
            variable_dfs.append(df_var)

        # A cluster directory can legitimately contain outputs for several
        # initialization hours.  Ignore folders belonging to another run;
        # callers validate the requested run's delivery-date coverage after
        # loading.
        if not variable_dfs:
            continue

        df_day = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_day = df_day.merge(df_next, on="timestamp", how="outer")
        df_day = df_day.sort_values("timestamp")

        start = pd.Timestamp(forecast_date, tz=target_tz)
        end = start + pd.Timedelta(days=1)
        hourly_cols = ["timestamp"] + [
            col for col in df_day.columns if col.startswith(("t2m_", "u10_", "v10_"))
        ]
        qh_cols = ["timestamp"] + [
            col
            for col in df_day.columns
            if col.startswith(("ASWDIR_", "ASWDIFD_"))
        ]

        df_hourly_day = (
            df_day[hourly_cols]
            .loc[
                (df_day["timestamp"] >= start)
                & (df_day["timestamp"] < end)
                & (df_day["timestamp"].dt.minute == 0)
            ]
            .copy()
        )
        df_qh_day = df_day[qh_cols].copy()
        df_qh_day["timestamp"] = df_qh_day["timestamp"] - pd.Timedelta(minutes=15)
        df_qh_day = df_qh_day.loc[
            (df_qh_day["timestamp"] >= start) & (df_qh_day["timestamp"] < end)
        ]
        hourly_dfs.append(df_hourly_day)
        qh_dfs.append(df_qh_day)

    if not hourly_dfs:
        raise ValueError("No hourly DWD data loaded.")
    if not qh_dfs:
        raise ValueError("No quarter-hourly DWD data loaded.")

    df_hourly = (
        pd.concat(hourly_dfs, ignore_index=True)
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    df_qh = (
        pd.concat(qh_dfs, ignore_index=True)
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    df_hourly.index.name = "timestamp"
    df_qh.index.name = "timestamp"
    return df_hourly, df_qh


def build_era5_features(
    df_era5: pd.DataFrame,
    target_tz: str = DEFAULT_TARGET_TZ,
) -> pd.DataFrame:
    """Build one daily row of hourly solar and wind features from ERA5."""
    df = df_era5.copy()
    ts_local = df.index.tz_convert(target_tz)
    df["date"] = ts_local.date
    df["hour"] = ts_local.hour

    u_cols = [col for col in df.columns if col.startswith("u10_cluster_")]
    for u_col in u_cols:
        cluster_id = u_col.split("_")[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col in df.columns:
            df[f"wind_speed_cluster_{cluster_id}"] = np.sqrt(
                df[u_col] ** 2 + df[v_col] ** 2
            )

    feature_cols = [
        col for col in df.columns if col.startswith("ssrd_cluster_")
    ] + [col for col in df.columns if col.startswith("wind_speed_cluster_")]
    df_features = df.pivot_table(
        index="date",
        columns="hour",
        values=feature_cols,
        aggfunc="mean",
    )
    df_features.columns = [
        f"{col_name}_h{int(hour):02d}"
        for col_name, hour in df_features.columns
    ]
    df_features = df_features.interpolate(axis=1, limit=1, limit_area="inside")
    df_features.index = pd.to_datetime(df_features.index).tz_localize(target_tz)
    df_features.index.name = "date"
    return df_features


def build_dwd_features(
    df_hourly: pd.DataFrame,
    df_qh: pd.DataFrame,
    target_tz: str = DEFAULT_TARGET_TZ,
) -> pd.DataFrame:
    """Build daily hourly wind and 15-minute solar ICON-D2 features."""
    df_w = df_hourly.copy()
    ts_loc_w = df_w.index.tz_convert(target_tz)
    df_w = df_w.assign(date=ts_loc_w.date, hour=ts_loc_w.hour)

    wind_data: Dict[str, np.ndarray] = {}
    for u_col in [c for c in df_w.columns if c.startswith("u10_cluster_")]:
        cluster_id = u_col.split("_")[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col in df_w.columns:
            wind_data[f"wind_speed_cluster_{cluster_id}"] = np.sqrt(
                df_w[u_col].to_numpy() ** 2 + df_w[v_col].to_numpy() ** 2
            )
    if not wind_data:
        raise ValueError(
            "Function build_dwd_features: no matching u10/v10 cluster pairs found."
        )

    df_w_base = df_w[["date", "hour"]].copy()
    df_w_speed = pd.DataFrame(wind_data, index=df_w.index)
    df_w_full = pd.concat([df_w_base, df_w_speed], axis=1)
    wind_cols = list(df_w_speed.columns)
    df_wind_pivot = df_w_full.pivot_table(
        index="date",
        columns="hour",
        values=wind_cols,
        aggfunc="mean",
    )
    df_wind_pivot.columns = [
        f"{column}_h{int(hour):02d}" for column, hour in df_wind_pivot.columns
    ]
    df_wind_pivot = df_wind_pivot.interpolate(
        axis=1, limit=1, limit_area="inside"
    )

    df_s = df_qh.copy()
    ts_loc_s = df_s.index.tz_convert(target_tz)
    df_s = df_s.assign(
        date=ts_loc_s.date,
        mtu=ts_loc_s.hour * 4 + ts_loc_s.minute // 15,
    )

    solar_data: Dict[str, np.ndarray] = {}
    for dir_col in [c for c in df_s.columns if c.startswith("ASWDIR_cluster_")]:
        cluster_id = dir_col.split("_")[-1]
        dif_col = f"ASWDIFD_cluster_{cluster_id}"
        if dif_col in df_s.columns:
            solar_data[f"sw_dir_cluster_{cluster_id}"] = df_s[dir_col].to_numpy()
            solar_data[f"sw_dif_cluster_{cluster_id}"] = df_s[dif_col].to_numpy()
    if not solar_data:
        raise ValueError(
            "Function build_dwd_features: no matching ASWDIR/ASWDIFD cluster pairs "
            "found."
        )

    df_s_base = df_s[["date", "mtu"]].copy()
    df_solar_vals = pd.DataFrame(solar_data, index=df_s.index)
    df_s_full = pd.concat([df_s_base, df_solar_vals], axis=1)
    solar_cols = list(df_solar_vals.columns)
    df_solar_pivot = df_s_full.pivot_table(
        index="date",
        columns="mtu",
        values=solar_cols,
        aggfunc="mean",
    )
    df_solar_pivot.columns = [
        f"{column}_mtu{int(mtu):02d}"
        for column, mtu in df_solar_pivot.columns
    ]
    df_solar_pivot = df_solar_pivot.interpolate(
        axis=1, limit=4, limit_area="inside"
    )

    df_pivot = pd.concat([df_wind_pivot, df_solar_pivot], axis=1)
    full_index = pd.date_range(
        start=df_pivot.index.min(),
        end=df_pivot.index.max(),
        freq="D",
    )
    df_pivot = df_pivot.reindex(full_index)
    df_pivot.index = pd.to_datetime(df_pivot.index).tz_localize(target_tz)
    df_pivot.index.name = "date"
    return df_pivot


def build_daily_mtu_matrix(
    series: pd.Series,
    value_name: str,
) -> pd.DataFrame:
    """Normalize a timestamped series to one row of 96 local-clock MTUs per day.

    The local clock defines the MTU, rather than elapsed time from midnight.  A
    repeated MTU on the autumn clock-change day is averaged by ``pivot_table``.
    The four absent MTUs on the spring clock-change day are linearly
    interpolated between their neighboring observations, matching the existing
    feature/target preprocessing used by LEAR.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError("series index must be timezone-aware.")

    frame = series.rename(value_name).to_frame().sort_index()
    frame["date_local"] = frame.index.normalize()
    frame["mtu"] = frame.index.hour * 4 + frame.index.minute // 15
    daily = (
        frame.pivot_table(
            index="date_local",
            columns="mtu",
            values=value_name,
            aggfunc="mean",
        )
        .reindex(columns=range(MTUS_PER_DAY))
        .interpolate(axis=1, limit=4, limit_area="inside")
    )
    daily.index.name = "date"
    return daily


def build_price_features(
    df_prices: pd.DataFrame,
    df_prices_exaa_15: Optional[pd.DataFrame] = None,
    exaa_vector: bool = False,
    exaa_only: bool = False,
    daily_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Build lagged SDAC and optional target-day EXAA daily vectors."""
    if exaa_only:
        exaa_vector = True
    if exaa_vector and df_prices_exaa_15 is None:
        raise ValueError(
            "df_prices_exaa_15 is required when exaa_vector=True or exaa_only=True."
        )

    df = df_prices.copy().sort_index()
    df_utc = df.tz_convert("UTC")
    df_utc["price_lag1d"] = df_utc["price_da"].shift(freq="1D")
    df_utc["price_lag2d"] = df_utc["price_da"].shift(freq="2D")
    df_utc["price_lag7d"] = df_utc["price_da"].shift(freq="7D")
    df["price_lag1d"] = df_utc["price_lag1d"].values
    df["price_lag2d"] = df_utc["price_lag2d"].values
    df["price_lag7d"] = df_utc["price_lag7d"].values

    daily_vector_dfs: List[pd.DataFrame] = []
    if not exaa_only:
        daily_matrix_da = build_daily_mtu_matrix(df["price_da"], "price_da")
        if daily_index is not None:
            requested = pd.DatetimeIndex(daily_index)
            if requested.tz is None:
                requested = requested.tz_localize(df.index.tz)
            else:
                requested = requested.tz_convert(df.index.tz)
            daily_matrix_da = daily_matrix_da.reindex(
                daily_matrix_da.index.union(requested).sort_values()
            )
        for lag, prefix in ((1, "price_d1"), (2, "price_d2"), (7, "price_d7")):
            daily_matrix = daily_matrix_da.shift(lag)
            daily_matrix.columns = [
                f"{prefix}_mtu_{int(column):02d}"
                for column in daily_matrix.columns
            ]
            daily_vector_dfs.append(daily_matrix)

    if exaa_vector:
        assert df_prices_exaa_15 is not None
        daily_matrix_exaa = build_daily_mtu_matrix(
            df_prices_exaa_15["price_exaa"], "price_exaa"
        )
        if daily_index is not None:
            requested = pd.DatetimeIndex(daily_index)
            if requested.tz is None:
                requested = requested.tz_localize(df_prices_exaa_15.index.tz)
            else:
                requested = requested.tz_convert(df_prices_exaa_15.index.tz)
            daily_matrix_exaa = daily_matrix_exaa.reindex(
                daily_matrix_exaa.index.union(requested).sort_values()
            )
        daily_matrix_exaa.columns = [
            f"exaa_d0_mtu_{int(column):02d}"
            for column in daily_matrix_exaa.columns
        ]
        daily_vector_dfs.append(daily_matrix_exaa)

    result = pd.concat(daily_vector_dfs, axis=1, sort=False)
    result.index.name = "date"
    return result


def build_load_features(df_load_fc: pd.DataFrame) -> pd.DataFrame:
    """Build the target-day 96-MTU load-forecast vector."""
    df = df_load_fc.copy().sort_index()
    daily_matrix = build_daily_mtu_matrix(df["load_fc"], "load_fc")
    daily_matrix.columns = [
        f"load_d0_mtu_{int(column):02d}" for column in daily_matrix.columns
    ]
    daily_matrix.index.name = "date"
    return daily_matrix


def build_temporal_features(
    daily_index: pd.DatetimeIndex,
    post_regime_start: date = POST_REGIME_START,
) -> pd.DataFrame:
    """Build market-regime, weekday, and German-holiday features."""
    if not isinstance(daily_index, pd.DatetimeIndex):
        raise TypeError("daily_index must be a DatetimeIndex.")
    if daily_index.tz is None:
        raise ValueError("daily_index must be timezone-aware.")

    regime_ts = pd.Timestamp(post_regime_start, tz=daily_index.tz)
    df_out = pd.DataFrame(index=daily_index)
    df_out["is_15min_market"] = (daily_index >= regime_ts).astype(int)

    weekday_oh = pd.get_dummies(
        daily_index.weekday, prefix="weekday", dtype=int
    )
    weekday_oh.index = daily_index
    df_out = pd.concat([df_out, weekday_oh], axis=1)

    de_holidays = holidays.Germany(years=daily_index.year.unique())
    df_out["is_holiday"] = pd.Series(
        daily_index.date, index=daily_index
    ).isin(set(de_holidays.keys())).astype(int)
    df_out.index.name = "date"
    return df_out


def merge_all_features(
    df_weather_features: pd.DataFrame,
    df_price_features: pd.DataFrame,
    df_load_features: pd.DataFrame,
    df_time_features: pd.DataFrame,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join daily feature blocks and report rows containing missing values."""
    features = (
        df_weather_features.join(df_price_features, how="inner")
        .join(df_load_features, how="inner")
        .join(df_time_features, how="inner")
        .sort_index()
    )
    nan_mask = features.isna().any(axis=1)
    dropped_info = pd.DataFrame(
        {
            "date": features.index[nan_mask],
            "nan_columns": features.loc[nan_mask]
            .isna()
            .apply(lambda row: row.index[row].tolist(), axis=1)
            .values,
        }
    )
    if dropna:
        features = features.loc[~nan_mask]
    features.index.name = "date"
    return features, dropped_info


def build_y_matrix(
    df_prices_15: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the daily 96-MTU SDAC target matrix."""
    target = build_daily_mtu_matrix(
        df_prices_15["price_da"].sort_index(), "price_da"
    )
    target.columns = range(MTUS_PER_DAY)
    return target.reindex(daily_index)


def build_exaa_naive_forecast(
    df_prices_15: pd.DataFrame,
    df_prices_exaa_15: pd.DataFrame,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    skip_dates: Iterable[date] = (),
) -> pd.DataFrame:
    """Return EXAA as a canonical 96-MTU deterministic point baseline.

    EPEX and EXAA are normalized separately before they are aligned.  This is
    essential on clock-change days: the spring day gains four interpolated
    local-clock MTUs, while repeated autumn MTUs are averaged.
    """
    y_pred_daily = build_daily_mtu_matrix(
        df_prices_exaa_15["price_exaa"], "price_exaa"
    )
    y_true_daily = build_daily_mtu_matrix(
        df_prices_15["price_da"], "price_da"
    )

    requested_days = pd.date_range(
        start=start_day.normalize(),
        end=end_day.normalize(),
        freq="D",
    )
    skipped = set(skip_dates)
    requested_days = pd.DatetimeIndex(
        [day for day in requested_days if day.date() not in skipped]
    )
    canonical_days = requested_days.tz_localize(None).normalize()
    index = make_delivery_index(canonical_days)
    return pd.DataFrame(
        {
            "y_pred": y_pred_daily.reindex(requested_days).to_numpy().reshape(-1),
            "y_true": y_true_daily.reindex(requested_days).to_numpy().reshape(-1),
        },
        index=index,
    )


def robust_params(x: Sequence[float], eps: float = 1e-12) -> Tuple[float, float]:
    """Return a median and MAD-based robust scale."""
    values = np.asarray(x, dtype=float)
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    scale = mad / Z_075
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
    return float(center), float(scale)


def inverse_vst_bias_corrected(
    y_hat_trans: Sequence[float],
    residuals_trans: Sequence[float],
    center: float,
    scale: float,
) -> np.ndarray:
    """Invert the arcsinh target transform with empirical bias correction."""
    y_hat = np.asarray(y_hat_trans, dtype=float).reshape(-1)
    residuals = np.asarray(residuals_trans, dtype=float).reshape(-1)
    if residuals.size == 0:
        return center + scale * np.sinh(y_hat)
    return center + scale * np.mean(
        np.sinh(y_hat[:, None] + residuals[None, :]),
        axis=1,
    )


def scale_fold_point(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    y_tr: pd.Series,
    y_va: Optional[pd.Series],
    use_vst: bool = True,
    ssrd_filter_min_range: float = 20.0,
    ssrd_filter_min_pos_share: float = 0.50,
    ssrd_filter_min_iqr: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray], dict]:
    """Apply the notebook's fold-wise point-model scaling."""
    cols = X_tr.columns.tolist()
    price_cols = [c for c in cols if c.startswith(("price_d", "exaa_d"))]
    time_cols = [
        c
        for c in cols
        if c.startswith("weekday_") or c in ["is_15min_market", "is_holiday"]
    ]
    ssrd_cols = [c for c in cols if "ssrd" in c.lower()]
    sw_dir_cols = [c for c in cols if "sw_dir" in c.lower()]
    sw_dif_cols = [c for c in cols if "sw_dif" in c.lower()]
    wind_cols = [c for c in cols if "wind_speed" in c.lower()]
    load_cols = [c for c in cols if "load" in c.lower()]
    categorized = (
        price_cols
        + time_cols
        + ssrd_cols
        + sw_dir_cols
        + sw_dif_cols
        + wind_cols
        + load_cols
    )
    other_cols = [c for c in cols if c not in categorized]

    X_tr_work = X_tr.copy()
    X_va_work = X_va.copy()
    degenerate_ssrd_cols: List[str] = []
    for col in ssrd_cols + sw_dir_cols + sw_dif_cols:
        values = X_tr_work[col].astype(float)
        col_range = values.max() - values.min()
        pos_share = (values > 0).mean()
        is_degenerate = (col_range < ssrd_filter_min_range) or (
            pos_share < ssrd_filter_min_pos_share
        )
        if ssrd_filter_min_iqr is not None:
            col_iqr = values.quantile(0.75) - values.quantile(0.25)
            is_degenerate = is_degenerate or (col_iqr < ssrd_filter_min_iqr)
        if is_degenerate:
            degenerate_ssrd_cols.append(col)

    if degenerate_ssrd_cols:
        X_tr_work.loc[:, degenerate_ssrd_cols] = 0.0
        X_va_work.loc[:, degenerate_ssrd_cols] = 0.0

    y_center, y_scale = robust_params(y_tr.values)
    y_tr_scaled = (y_tr.values - y_center) / y_scale
    y_va_scaled = (
        (y_va.values - y_center) / y_scale if y_va is not None else None
    )
    if use_vst:
        y_tr_scaled = np.arcsinh(y_tr_scaled)
        if y_va_scaled is not None:
            y_va_scaled = np.arcsinh(y_va_scaled)

    y_params = {
        "y_center": float(y_center),
        "y_scale": float(y_scale),
        "use_vst": bool(use_vst),
        "degenerate_ssrd_cols": degenerate_ssrd_cols,
    }

    X_tr_scaled = X_tr_work.copy()
    X_va_scaled = X_va_work.copy()
    for col in price_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = np.arcsinh(
            (X_tr_work[col].values - center) / scale
        )
        X_va_scaled[col] = np.arcsinh(
            (X_va_work[col].values - center) / scale
        )
    for col in wind_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = (X_tr_work[col].values - center) / scale
        X_va_scaled[col] = (X_va_work[col].values - center) / scale
    for col in load_cols:
        center, scale = robust_params(X_tr_work[col].values)
        X_tr_scaled[col] = (X_tr_work[col].values - center) / scale
        X_va_scaled[col] = (X_va_work[col].values - center) / scale

    cont_cols = ssrd_cols + sw_dir_cols + sw_dif_cols + other_cols
    if cont_cols:
        scaler = MinMaxScaler()
        X_tr_scaled[cont_cols] = scaler.fit_transform(X_tr_work[cont_cols])
        X_va_scaled[cont_cols] = scaler.transform(X_va_work[cont_cols])

    return (
        X_tr_scaled,
        X_va_scaled,
        y_tr_scaled,
        y_va_scaled,
        y_params,
    )


def rolling_point_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[pd.Timestamp],
    train_days: int,
    lars_start_date: pd.Timestamp,
    use_vst: bool = True,
    progress: bool = True,
    allow_incomplete_prices: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling MTU-specific LEAR point forecasts."""
    forecast_records: List[dict] = []
    runtime_records: List[dict] = []
    coef_records: List[dict] = []
    intercept_records: List[dict] = []
    degenerate_ssrd_records: List[dict] = []

    for day in forecast_days:
        day_start = time.perf_counter()
        use_lars = day >= lars_start_date
        # Calendar offsets preserve local midnight across DST transitions.
        train_start = day - pd.DateOffset(days=train_days)
        train_end = day - pd.DateOffset(days=1)
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask = X.index == day
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        active_columns = X.columns.tolist()
        if allow_incomplete_prices:
            X_test_day = X.loc[test_mask]
            unavailable_columns = [
                column
                for column in X.columns
                if not np.isfinite(
                    X_test_day[column].to_numpy(dtype=float)
                ).all()
            ]
            unsupported_columns = [
                column
                for column in unavailable_columns
                if not column.startswith("price_d")
            ]
            if unsupported_columns:
                raise ValueError(
                    f"Forecast features contain unavailable non-price inputs for "
                    f"{day.date()}: {unsupported_columns}."
                )
            active_columns = [
                column for column in X.columns if column not in unavailable_columns
            ]
            if not active_columns:
                raise ValueError(f"No usable forecast features remain for {day.date()}.")
            if unavailable_columns and progress:
                blocks = sorted(
                    {
                        column.split("_mtu_", 1)[0]
                        for column in unavailable_columns
                    }
                )
                print(
                    f"  {day.date()} unavailable EPEX lag block(s): "
                    + ", ".join(blocks)
                    + "; forecasting with remaining variables",
                    flush=True,
                )

        delivery_date = day.tz_localize(None).normalize()
        for mtu in range(MTUS_PER_DAY):
            X_tr = X.loc[train_mask, active_columns]
            y_tr = Y.loc[train_mask, mtu]
            X_te = X.loc[test_mask, active_columns]
            if allow_incomplete_prices:
                valid_training = np.isfinite(
                    X_tr.to_numpy(dtype=float)
                ).all(axis=1) & np.isfinite(y_tr.to_numpy(dtype=float))
                X_tr = X_tr.loc[valid_training]
                y_tr = y_tr.loc[valid_training]
                if len(y_tr) < 5:
                    raise ValueError(
                        f"Fewer than five complete training observations remain "
                        f"for {day.date()}, MTU {mtu + 1}."
                    )
            X_tr_s, X_te_s, y_tr_s, _, y_params = scale_fold_point(
                X_tr=X_tr,
                X_va=X_te,
                y_tr=y_tr,
                y_va=None,
                use_vst=use_vst,
            )

            if use_lars:
                model = LassoLarsCV(cv=5, max_iter=1000, n_jobs=1)
            else:
                model = LassoCV(cv=5, tol=1e-3, max_iter=10_000, n_jobs=1)
            model.fit(X_tr_s.values, y_tr_s)
            y_pred_s = model.predict(X_te_s.values)

            if use_vst:
                y_fit_s = model.predict(X_tr_s.values)
                residuals_s = y_tr_s - y_fit_s
                y_pred = inverse_vst_bias_corrected(
                    y_hat_trans=y_pred_s,
                    residuals_trans=residuals_s,
                    center=y_params["y_center"],
                    scale=y_params["y_scale"],
                )
            else:
                y_pred = (
                    y_params["y_center"] + y_params["y_scale"] * y_pred_s
                )

            forecast_records.append(
                {
                    "delivery_date": delivery_date,
                    "mtu": mtu + 1,
                    "y_pred": float(np.asarray(y_pred).ravel()[0]),
                    "y_true": float(Y.loc[day, mtu]),
                }
            )

            nonzero_mask = model.coef_ != 0
            coef_records.append(
                {
                    "forecast_day": day.date(),
                    "mtu": mtu + 1,
                    "alpha": model.alpha_,
                    "n_nonzero": int(nonzero_mask.sum()),
                    "nonzero_cols": X_tr.columns[nonzero_mask].tolist(),
                    "nonzero_vals": model.coef_[nonzero_mask].tolist(),
                    "use_lars": use_lars,
                }
            )
            intercept_records.append(
                {
                    "forecast_day": day.date(),
                    "mtu": mtu + 1,
                    "intercept": float(model.intercept_),
                }
            )
            degenerate_ssrd_records.append(
                {
                    "forecast_day": day.date(),
                    "mtu": mtu + 1,
                    "n_degenerate": len(y_params["degenerate_ssrd_cols"]),
                    "degenerate_cols": y_params["degenerate_ssrd_cols"],
                }
            )

        day_runtime = time.perf_counter() - day_start
        runtime_records.append(
            {
                "forecast_day": day,
                "train_days": train_days,
                "use_vst": use_vst,
                "use_lars": use_lars,
                "runtime_seconds": day_runtime,
            }
        )
        if progress:
            print(
                f"  {day.date()}  {'LARS' if use_lars else 'LassoCV'}  "
                f"{day_runtime:.1f}s",
                flush=True,
            )

    if not forecast_records:
        raise ValueError("No LEAR forecasts were produced for the requested days.")
    forecast_df = (
        pd.DataFrame(forecast_records)
        .set_index(list(INDEX_COLUMNS))
        .sort_index()
    )
    validate_delivery_index(forecast_df.index, require_complete_days=True)
    return (
        forecast_df,
        pd.DataFrame(runtime_records),
        pd.DataFrame(coef_records),
        pd.DataFrame(intercept_records),
        pd.DataFrame(degenerate_ssrd_records),
    )


def compute_metrics(forecast: pd.DataFrame, label: str) -> dict:
    """Compute the point-forecast metrics written by the original notebook."""
    n_inf_nan = int((~np.isfinite(forecast["y_pred"])).sum())
    valid = forecast[np.isfinite(forecast["y_pred"])]
    mae = (
        mean_absolute_error(valid["y_true"], valid["y_pred"])
        if len(valid) > 0
        else np.nan
    )
    rmse = (
        np.sqrt(((valid["y_true"] - valid["y_pred"]) ** 2).mean())
        if len(valid) > 0
        else np.nan
    )
    bias = (
        (valid["y_pred"] - valid["y_true"]).mean()
        if len(valid) > 0
        else np.nan
    )
    return {
        "period": label,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "n_obs": len(forecast),
        "n_inf_nan": n_inf_nan,
    }


def select_evaluation_period(
    forecast: pd.DataFrame,
    start_date,
    end_date,
    skip_dates: Iterable[date] = (),
) -> pd.DataFrame:
    """Select the inclusive configured evaluation days from a point forecast."""
    validate_delivery_index(forecast.index)

    def normalize(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.normalize()

    start = normalize(start_date)
    end = normalize(end_date)
    if end < start:
        raise ValueError("Evaluation end date must not precede its start date.")
    excluded = {normalize(value) for value in skip_dates}
    dates = delivery_dates(forecast.index)
    mask = (dates >= start) & (dates <= end)
    if excluded:
        mask &= ~dates.isin(excluded)
    selected = forecast.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            f"Point forecast has no observations in evaluation period "
            f"{start.date()} through {end.date()}."
        )
    return selected


def save_experiment_outputs(
    name: str,
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: dict,
    export_dir: Path,
    *,
    evaluation_start=None,
    evaluation_end=None,
    evaluation_skip_dates: Iterable[date] = (),
) -> None:
    """Save forecast, runtime, monthly metrics, and configuration files."""
    del name  # Kept in the interface for notebook and historical compatibility.
    export_dir.mkdir(parents=True, exist_ok=True)
    validate_delivery_index(forecast_df.index, require_complete_days=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)

    rows = [compute_metrics(forecast_df, "full")]
    delivery_months = delivery_dates(forecast_df.index).to_period("M")
    months = delivery_months.unique()
    for period in months:
        mask = delivery_months == period
        rows.append(compute_metrics(forecast_df[mask], str(period)))
    if (evaluation_start is None) != (evaluation_end is None):
        raise ValueError(
            "evaluation_start and evaluation_end must either both be set or both omitted."
        )
    if evaluation_start is not None:
        evaluation_forecast = select_evaluation_period(
            forecast_df,
            evaluation_start,
            evaluation_end,
            evaluation_skip_dates,
        )
        # Keep the evaluation row last so it is immediately visible and can be
        # compared across point-model metrics files.
        rows.append(compute_metrics(evaluation_forecast, "evaluation"))
    pd.DataFrame(rows).to_csv(export_dir / "metrics.csv", index=False)
    with (export_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Saved: forecast | runtime | metrics | config  ->  {export_dir}")


def scale_fold_anc(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    ssrd_filter_min_range: float = 20.0,
    ssrd_filter_min_pos_share: float = 0.50,
    ssrd_filter_min_iqr: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply fold-wise min-max scaling for ANC re-estimation."""
    cols = X_tr.columns.tolist()
    calendar_cols = [
        c
        for c in cols
        if c.startswith("weekday_") or c in ["is_15min_market", "is_holiday"]
    ]
    ssrd_cols = [c for c in cols if "ssrd" in c.lower()]
    sw_dir_cols = [c for c in cols if "sw_dir" in c.lower()]
    sw_dif_cols = [c for c in cols if "sw_dif" in c.lower()]
    scale_cols = [c for c in cols if c not in calendar_cols]

    X_tr_work = X_tr.copy()
    X_te_work = X_te.copy()
    degenerate_ssrd_cols: List[str] = []
    for col in ssrd_cols + sw_dir_cols + sw_dif_cols:
        values = X_tr_work[col].astype(float)
        col_range = values.max() - values.min()
        pos_share = (values > 0).mean()
        is_degenerate = (col_range < ssrd_filter_min_range) or (
            pos_share < ssrd_filter_min_pos_share
        )
        if ssrd_filter_min_iqr is not None:
            col_iqr = values.quantile(0.75) - values.quantile(0.25)
            is_degenerate = is_degenerate or (col_iqr < ssrd_filter_min_iqr)
        if is_degenerate:
            degenerate_ssrd_cols.append(col)

    if degenerate_ssrd_cols:
        X_tr_work.loc[:, degenerate_ssrd_cols] = 0.0
        X_te_work.loc[:, degenerate_ssrd_cols] = 0.0

    X_tr_scaled = X_tr_work.copy()
    X_te_scaled = X_te_work.copy()
    scaling_params = {"degenerate_ssrd_cols": degenerate_ssrd_cols}
    for col in scale_cols:
        col_min = float(X_tr_work[col].min())
        col_max = float(X_tr_work[col].max())
        col_range = col_max - col_min
        if not np.isfinite(col_range) or col_range <= 1e-12:
            X_tr_scaled[col] = 0.0
            X_te_scaled[col] = 0.0
            col_range = 1.0
        else:
            X_tr_scaled[col] = (X_tr_work[col] - col_min) / col_range
            X_te_scaled[col] = (X_te_work[col] - col_min) / col_range
        scaling_params[col] = {
            "min": col_min,
            "max": col_max,
            "range": float(col_range),
        }
    return X_tr_scaled, X_te_scaled, scaling_params


def rolling_anc_feature_importance(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[pd.Timestamp],
    train_days: int,
    lars_start_date: pd.Timestamp,
    progress: bool = True,
) -> pd.DataFrame:
    """Run rolling MTU-specific ANC contribution estimation."""
    anc_records: List[dict] = []
    for day in forecast_days:
        use_lars = day >= lars_start_date
        # Calendar offsets preserve local midnight across DST transitions.
        train_start = day - pd.DateOffset(days=train_days)
        train_end = day - pd.DateOffset(days=1)
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        test_mask = X.index == day
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        delivery_date = day.tz_localize(None).normalize()
        for mtu in range(MTUS_PER_DAY):
            X_tr = X.loc[train_mask]
            y_tr = Y.loc[train_mask, mtu]
            X_te = X.loc[test_mask]
            X_tr_s, X_te_s, _ = scale_fold_anc(X_tr=X_tr, X_te=X_te)
            if use_lars:
                model = LassoLarsCV(cv=5, max_iter=1000, n_jobs=1)
            else:
                model = LassoCV(cv=5, tol=1e-3, max_iter=10_000, n_jobs=1)
            model.fit(X_tr_s.values, y_tr.values)

            beta_series = pd.Series(
                model.coef_, index=X_tr_s.columns, dtype=float
            )
            x_row = X_te_s.iloc[0]
            for feature in X_te_s.columns:
                feature_value = float(x_row[feature])
                beta = float(beta_series[feature])
                anc_records.append(
                    {
                        "delivery_date": delivery_date,
                        "mtu": mtu + 1,
                        "train_days": train_days,
                        "use_lars": use_lars,
                        "feature": feature,
                        "feature_value": feature_value,
                        "beta": beta,
                        "contribution": feature_value * beta,
                    }
                )
        if progress:
            print(
                f"  {day.date()}  {'LARS' if use_lars else 'LassoCV'}",
                flush=True,
            )
    return pd.DataFrame(anc_records)


def map_feature_to_group(feature: str) -> str:
    """Map a raw LEAR regressor to the manuscript's feature groups."""
    value = feature.lower()
    if "exaa" in value:
        return "EXAA d"
    if "price_d1" in value:
        return "Price d-1"
    if "price_d2" in value:
        return "Price d-2"
    if "price_d7" in value:
        return "Price d-7"
    if "load_d0" in value:
        return "Load d"
    if "wind" in value:
        return "Wind d"
    if "ssrd" in value or "sw_dir" in value or "sw_dif" in value:
        return "Solar d"
    if value.startswith("weekday_"):
        return "Weekday"
    if "is_holiday" in value:
        return "Holiday"
    if "is_15min_market" in value:
        return "15-min market dummy"
    return "Other"


def map_wind_feature_to_cluster(feature: str) -> str:
    """Map an hourly wind feature to its cluster label."""
    match = re.search(r"wind_speed_cluster_(\d+)_h\d+", feature.lower())
    return f"Wind Cluster {match.group(1)}" if match else "Other"


def map_solar_feature_to_cluster(feature: str) -> str:
    """Map an hourly solar feature to its cluster label."""
    value = feature.lower()
    if not any(token in value for token in ("ssrd", "sw_dir", "sw_dif")):
        return "Other"
    match = re.search(r"cluster_(\d+)_h\d+", value)
    return f"Solar Cluster {match.group(1)}" if match else "Other"


def summarize_anc(
    anc_df: pd.DataFrame,
    wind_mtu_window: Iterable[int] = range(1, MTUS_PER_DAY + 1),
    solar_mtu_window: Iterable[int] = range(1, MTUS_PER_DAY + 1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the overall, wind-cluster, and solar-cluster ANC exports."""
    analysis = anc_df.copy()
    analysis["feature_group"] = analysis["feature"].apply(map_feature_to_group)
    grouped = (
        analysis.groupby(
            [
                "delivery_date",
                "mtu",
                "train_days",
                "feature_group",
            ],
            as_index=False,
        )["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    grouped["abs_group_contribution"] = grouped["group_contribution"].abs()
    overall = (
        grouped.groupby(["train_days", "feature_group"], as_index=False)[
            "abs_group_contribution"
        ]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values(["train_days", "ANC"], ascending=[True, False])
        .reset_index(drop=True)
    )

    wind = anc_df.copy()
    wind["cluster_group"] = wind["feature"].apply(map_wind_feature_to_cluster)
    wind = wind.loc[
        (wind["cluster_group"] != "Other")
        & wind["mtu"].isin(tuple(wind_mtu_window))
    ].copy()
    wind_grouped = (
        wind.groupby(
            [
                "delivery_date",
                "mtu",
                "train_days",
                "cluster_group",
            ],
            as_index=False,
        )["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    wind_grouped["abs_group_contribution"] = wind_grouped[
        "group_contribution"
    ].abs()
    wind_summary = (
        wind_grouped.groupby(["train_days", "cluster_group"], as_index=False)[
            "abs_group_contribution"
        ]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values("ANC", ascending=False)
        .reset_index(drop=True)
    )
    wind_export = wind_summary.copy()
    wind_export["cluster_id"] = (
        wind_export["cluster_group"]
        .str.extract(r"Wind Cluster (\d+)")
        .astype(int)
    )

    solar = anc_df.copy()
    solar["cluster_group"] = solar["feature"].apply(map_solar_feature_to_cluster)
    solar = solar.loc[
        (solar["cluster_group"] != "Other")
        & solar["mtu"].isin(tuple(solar_mtu_window))
    ].copy()
    solar_grouped = (
        solar.groupby(
            [
                "delivery_date",
                "mtu",
                "train_days",
                "cluster_group",
            ],
            as_index=False,
        )["contribution"]
        .sum()
        .rename(columns={"contribution": "group_contribution"})
    )
    solar_grouped["abs_group_contribution"] = solar_grouped[
        "group_contribution"
    ].abs()
    solar_summary = (
        solar_grouped.groupby(["train_days", "cluster_group"], as_index=False)[
            "abs_group_contribution"
        ]
        .mean()
        .rename(columns={"abs_group_contribution": "ANC"})
        .sort_values("ANC", ascending=False)
        .reset_index(drop=True)
    )
    solar_export = solar_summary.copy()
    solar_export["cluster_id"] = (
        solar_export["cluster_group"]
        .str.extract(r"Solar Cluster (\d+)")
        .astype(int)
    )
    return overall, wind_export, solar_export
