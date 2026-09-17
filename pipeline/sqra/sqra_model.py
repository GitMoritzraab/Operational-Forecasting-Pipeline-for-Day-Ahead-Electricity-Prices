"""Reusable model, panel-building, and metric functions for SQRA forecasts."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from remodels.qra import SQRA
from sklearn.metrics import mean_absolute_error

from pipeline.delivery_index import (
    INDEX_COLUMNS,
    MTUS_PER_DAY,
    delivery_dates,
    read_forecast_csv,
    validate_delivery_index,
)


DEFAULT_QUANTILES: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


def _normalize_delivery_day(value) -> pd.Timestamp:
    """Return one timezone-naive, normalized delivery-day key."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = pd.Timestamp(timestamp.date())
    return timestamp.normalize()


def _delivery_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract canonical delivery dates from a forecast/panel MultiIndex."""
    return delivery_dates(df.index)


def _mtus(df: pd.DataFrame) -> pd.Index:
    """Extract canonical one-based MTU numbers from a forecast/panel."""
    validate_delivery_index(df.index)
    return df.index.get_level_values(INDEX_COLUMNS[1])


def load_forecast(path: Path, timezone: str) -> pd.DataFrame:
    """Load one canonical point-forecast CSV keyed by delivery date and MTU.

    ``timezone`` remains in the public signature for compatibility with callers,
    but canonical delivery keys are deliberately timezone independent.  A local
    delivery day always contains MTUs 1 through 96, including DST transition
    days after their inputs have been normalized by LEAR.
    """
    del timezone
    return read_forecast_csv(
        Path(path),
        required_columns=("y_pred", "y_true"),
        require_complete_days=True,
    )[["y_pred", "y_true"]]


def build_sqra_panel(
    paths: Sequence[Path],
    timezone: str,
) -> Tuple[pd.DataFrame, list[str]]:
    """Load and align the point forecasts used as SQRA regressors."""
    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("At least one point forecast is required for SQRA.")

    forecasts = [load_forecast(path, timezone) for path in input_paths]
    common_index = forecasts[0].index
    for forecast in forecasts[1:]:
        common_index = common_index.intersection(forecast.index)
    common_index = common_index.sort_values()
    if common_index.empty:
        raise ValueError("SQRA point-forecast panel has no common delivery keys.")
    validate_delivery_index(common_index, require_complete_days=True)

    panel = pd.DataFrame(index=common_index)
    feature_cols = [f"prediction_p{i + 1}" for i in range(len(forecasts))]
    reference_y = forecasts[0].loc[common_index, "y_true"]
    for i, (path, forecast) in enumerate(zip(input_paths, forecasts)):
        aligned = forecast.loc[common_index]
        panel[feature_cols[i]] = aligned["y_pred"]
        if i and not np.allclose(
            aligned["y_true"], reference_y, equal_nan=True
        ):
            raise ValueError(f"y_true differs across SQRA panel member: {path}")

    panel["y_true"] = reference_y
    return panel, feature_cols


def fit_sqra_for_mtu_and_quantile(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    mtu: int,
    quantile: float,
    feature_cols: list[str],
) -> Optional[SQRA]:
    """Fit one SQRA model for one MTU and one quantile."""
    idx = train_mask & (_mtus(df) == mtu)
    if idx.sum() == 0:
        return None

    X_train = df.loc[idx, feature_cols]
    y_train = df.loc[idx, "y_true"]
    valid = X_train.notna().all(axis=1) & y_train.notna()
    if valid.sum() == 0:
        return None

    model = SQRA(quantile=quantile, fit_intercept=True)
    try:
        model.fit(
            X_train.loc[valid].to_numpy(),
            y_train.loc[valid].to_numpy(),
        )
    except Exception:
        return None
    return model


def predict_sqra_for_mtu(
    df: pd.DataFrame,
    test_mask: np.ndarray,
    mtu: int,
    model: Optional[SQRA],
    feature_cols: list[str],
) -> pd.Series:
    """Predict one quantile for one MTU, preserving invalid rows as NaN."""
    idx = test_mask & (_mtus(df) == mtu)
    if not idx.any():
        return pd.Series(dtype=float)

    index = df.loc[idx].index
    if model is None:
        return pd.Series(index=index, dtype=float)

    X_test = df.loc[idx, feature_cols]
    valid = X_test.notna().all(axis=1)
    if not valid.any():
        return pd.Series(index=index, dtype=float)

    predictions = pd.Series(index=index, dtype=float)
    predictions.loc[valid] = model.predict(X_test.loc[valid].to_numpy())
    return predictions


def rolling_sqra_forecast_mtu(
    df: pd.DataFrame,
    forecast_days: Sequence,
    train_days: int,
    quantiles: Sequence[float],
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate rolling forecasts with one model per MTU and quantile."""
    all_days = []
    runtime_records = []
    delivery_dates = _delivery_dates(df)

    for day in forecast_days:
        delivery_day = _normalize_delivery_day(day)
        print(delivery_day.date())
        start_time = time.perf_counter()
        train_start = delivery_day - pd.Timedelta(days=train_days)
        # Calendar-day comparison is intentional: the target day is excluded
        # even when it represents a normalized 23- or 25-hour DST day.
        train_mask = (delivery_dates >= train_start) & (
            delivery_dates < delivery_day
        )
        test_mask = delivery_dates == delivery_day

        day_df = pd.DataFrame(index=df.loc[test_mask].index)
        for quantile in quantiles:
            mtu_predictions = []
            for mtu in range(1, MTUS_PER_DAY + 1):
                model = fit_sqra_for_mtu_and_quantile(
                    df=df,
                    train_mask=train_mask,
                    mtu=mtu,
                    quantile=quantile,
                    feature_cols=feature_cols,
                )
                mtu_predictions.append(
                    predict_sqra_for_mtu(
                        df=df,
                        test_mask=test_mask,
                        mtu=mtu,
                        model=model,
                        feature_cols=feature_cols,
                    )
                )
            day_df[f"q{quantile:.3f}"] = pd.concat(mtu_predictions).sort_index()

        day_df["y_true"] = df.loc[test_mask, "y_true"]
        all_days.append(day_df)
        runtime_records.append(
            {
                "forecast_day": delivery_day.date(),
                "computation_time_seconds": time.perf_counter() - start_time,
            }
        )

    forecast_df = pd.concat(all_days).sort_index() if all_days else pd.DataFrame()
    return forecast_df, pd.DataFrame(runtime_records)


def fit_sqra_for_quantile_pooled(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    quantile: float,
    feature_cols: list[str],
) -> Optional[SQRA]:
    """Fit one SQRA model using every valid MTU in the training window."""
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "y_true"]
    valid = X_train.notna().all(axis=1) & y_train.notna()
    if valid.sum() == 0:
        return None

    model = SQRA(quantile=quantile, fit_intercept=True)
    try:
        model.fit(
            X_train.loc[valid].to_numpy(),
            y_train.loc[valid].to_numpy(),
        )
    except Exception:
        return None
    return model


def predict_sqra_pooled(
    df: pd.DataFrame,
    test_mask: np.ndarray,
    model: Optional[SQRA],
    feature_cols: list[str],
) -> pd.Series:
    """Apply one pooled SQRA model to every valid MTU of a forecast day."""
    index = df.loc[test_mask].index
    predictions = pd.Series(index=index, dtype=float)
    if model is None or len(index) == 0:
        return predictions

    X_test = df.loc[test_mask, feature_cols]
    valid = X_test.notna().all(axis=1)
    if valid.any():
        predictions.loc[valid] = model.predict(X_test.loc[valid].to_numpy())
    return predictions


def rolling_sqra_forecast_pooled(
    df: pd.DataFrame,
    forecast_days: Sequence,
    train_days: int,
    quantiles: Sequence[float],
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate rolling forecasts with one shared model per day and quantile."""
    all_days = []
    runtime_records = []
    delivery_dates = _delivery_dates(df)

    for day in forecast_days:
        delivery_day = _normalize_delivery_day(day)
        print(delivery_day.date())
        start_time = time.perf_counter()
        train_start = delivery_day - pd.Timedelta(days=train_days)
        train_mask = (delivery_dates >= train_start) & (
            delivery_dates < delivery_day
        )
        test_mask = delivery_dates == delivery_day

        day_df = pd.DataFrame(index=df.loc[test_mask].index)
        for quantile in quantiles:
            model = fit_sqra_for_quantile_pooled(
                df=df,
                train_mask=train_mask,
                quantile=quantile,
                feature_cols=feature_cols,
            )
            day_df[f"q{quantile:.3f}"] = predict_sqra_pooled(
                df=df,
                test_mask=test_mask,
                model=model,
                feature_cols=feature_cols,
            )

        day_df["y_true"] = df.loc[test_mask, "y_true"]
        all_days.append(day_df)
        runtime_records.append(
            {
                "forecast_day": delivery_day.date(),
                "computation_time_seconds": time.perf_counter() - start_time,
            }
        )

    forecast_df = pd.concat(all_days).sort_index() if all_days else pd.DataFrame()
    return forecast_df, pd.DataFrame(runtime_records)


def sort_quantiles(df: pd.DataFrame, quantile_cols: list[str]) -> pd.DataFrame:
    """Sort quantile values row-wise to enforce monotonicity."""
    missing_cols = [column for column in quantile_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing quantile columns: {missing_cols}")

    sorted_df = df.copy()
    sorted_df[quantile_cols] = np.sort(
        sorted_df[quantile_cols].to_numpy(), axis=1
    )
    return sorted_df


def generate_sqra_forecast(
    df: pd.DataFrame,
    forecast_days: Sequence,
    train_days: int,
    quantiles: Sequence[float],
    feature_cols: list[str],
    mtu_specific: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the selected rolling calibration mode and sort its quantiles."""
    rolling_forecast = (
        rolling_sqra_forecast_mtu if mtu_specific else rolling_sqra_forecast_pooled
    )
    forecast, runtime = rolling_forecast(
        df=df,
        forecast_days=forecast_days,
        train_days=train_days,
        quantiles=quantiles,
        feature_cols=feature_cols,
    )
    quantile_cols = [f"q{quantile:.3f}" for quantile in quantiles]
    return sort_quantiles(forecast, quantile_cols), runtime


def mae_median(
    df: pd.DataFrame,
    q_median: float = 0.5,
    y_true_col: str = "y_true",
) -> float:
    """Compute MAE between realized values and the median quantile."""
    return mean_absolute_error(df[y_true_col], df[f"q{q_median:.3f}"])


def empirical_coverage(
    df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    y_true_col: str = "y_true",
) -> float:
    """Compute empirical coverage of one prediction interval."""
    inside = (df[y_true_col] >= df[f"q{lower_q:.3f}"]) & (
        df[y_true_col] <= df[f"q{upper_q:.3f}"]
    )
    return float(inside.mean())


def pinball_score(y_true, y_q, quantile: float) -> float:
    """Compute mean pinball loss for one quantile."""
    difference = y_true - y_q
    return float(
        np.mean(
            np.maximum(
                quantile * difference,
                (quantile - 1) * difference,
            )
        )
    )


def aggregate_pinball_score(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    y_true_col: str = "y_true",
) -> Tuple[dict[float, float], float]:
    """Compute pinball scores and their average across quantiles."""
    y_true = df[y_true_col]
    scores = {
        quantile: pinball_score(y_true, df[f"q{quantile:.3f}"], quantile)
        for quantile in quantiles
    }
    return scores, float(np.mean(list(scores.values())))


def evaluate_probabilistic_forecasts(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    quantiles: Sequence[float],
    y_true_col: str = "y_true",
) -> pd.DataFrame:
    """Calculate the notebook's per-day SQRA metrics and summary row."""
    sorted_quantiles = sorted(quantiles)
    if 0.5 not in sorted_quantiles:
        raise ValueError("Quantiles must contain 0.5 for MAE evaluation.")
    if len(sorted_quantiles) < 3:
        raise ValueError("Need at least 3 quantiles to form prediction intervals.")

    middle = sorted_quantiles.index(0.5)
    if middle == 0 or middle == len(sorted_quantiles) - 1:
        raise ValueError("Quantiles must be symmetric around 0.5.")

    inner_low = sorted_quantiles[middle - 1]
    inner_high = sorted_quantiles[middle + 1]
    outer_low = sorted_quantiles[0]
    outer_high = sorted_quantiles[-1]
    delivery_dates = _delivery_dates(df)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    results = []
    for day in date_range:
        day_df = df.loc[delivery_dates == day]
        if day_df.empty:
            continue
        _, average_pinball = aggregate_pinball_score(
            day_df, sorted_quantiles, y_true_col
        )
        results.append(
            {
                "Target Day": str(day.date()),
                "MAE (median)": mae_median(day_df, 0.5, y_true_col),
                f"Coverage {inner_low}-{inner_high}": empirical_coverage(
                    day_df, inner_low, inner_high, y_true_col
                ),
                f"Coverage {outer_low}-{outer_high}": empirical_coverage(
                    day_df, outer_low, outer_high, y_true_col
                ),
                "APS": average_pinball,
            }
        )

    result_df = pd.DataFrame(results)
    summary = result_df.mean(numeric_only=True)
    summary["Target Day"] = "Mean over all days"
    return pd.concat([result_df, pd.DataFrame([summary])], ignore_index=True)


def make_forecast_days(
    start: date,
    end: date,
    timezone: str,
    skip_dates: Sequence[date],
) -> pd.DatetimeIndex:
    """Build timezone-independent canonical delivery days.

    The retained ``timezone`` argument keeps the entry-point API stable.  It is
    intentionally not applied: DST is represented by normalization to 96 MTUs,
    not by elapsed-time arithmetic on timezone-aware midnights.
    """
    del timezone
    days = pd.date_range(start=start, end=end, freq="D")
    skipped = set(skip_dates)
    return pd.DatetimeIndex([day for day in days if day.date() not in skipped])
