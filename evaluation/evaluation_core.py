"""Reusable calculations for point and probabilistic forecast evaluation.

This module contains the numerical work from the original interactive
evaluation. It deliberately performs no file writes and never
shows a Matplotlib window, which makes the same calculations usable from the
automated runner, tests, and the retained interactive notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats

from experiment_config import ExperimentConfig
from experiment_manifest import lear_runs
from pipeline.delivery_index import (
    MTUS_PER_DAY,
    delivery_dates,
    make_delivery_index,
    read_forecast_csv as read_canonical_forecast_csv,
    validate_delivery_index,
)


FREQ = "15min"
N_PERIODS = MTUS_PER_DAY
QUANTILES: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
PREDICTION_INTERVALS: Tuple[Tuple[float, float], ...] = (
    (0.10, 0.90),
    (0.25, 0.75),
)
SIGNIFICANCE_LEVEL = 0.05
Y_TRUE_COL = "y_true"

QUANTILE_MODELS: Tuple[str, ...] = (
    "fund_DWD",
    "fund_ERA5",
    "exaa_naive",
    "exaa_DWD",
    "exaa_ERA5",
    "exaa_only",
)

LABEL_MAP_SQRA = {
    "fund_DWD": r"$\mathrm{SQRA}_{\mathrm{DWD,\,Fundamental}}$",
    "fund_ERA5": r"$\mathrm{SQRA}_{\mathrm{ERA5,\,Fundamental}}$",
    "exaa_naive": r"$\mathrm{SQRA}_{\mathrm{EXAA,\,Naive}}$",
    "exaa_DWD": r"$\mathrm{SQRA}_{\mathrm{DWD,\,EXAA}}$",
    "exaa_ERA5": r"$\mathrm{SQRA}_{\mathrm{ERA5,\,EXAA}}$",
    "exaa_only": r"$\mathrm{SQRA}_{\mathrm{EXAA,\,Only}}$",
}


def _point_model_display_label(run) -> str:
    """Return the manuscript display name for one canonical LEAR run."""
    if run.use_exaa_only:
        return f"EXAA-Only_d{run.train_days}"
    prefix = "EXAA-Enriched" if run.use_exaa else "Fundamental"
    return (
        f"{prefix}_{run.weather_source}_d{run.train_days}_c{run.clusters}"
    )


LABEL_MAP_POINT = {
    run.name: _point_model_display_label(run)
    for run in lear_runs()
}
LABEL_MAP_POINT["exaa_naive"] = "EXAA-Naive"


@dataclass(frozen=True)
class EvaluationData:
    """Validated point and quantile panels for the configured sample."""

    point: pd.DataFrame
    quantile: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResults:
    """All tables and GW matrices produced by the evaluation stage."""

    point_metrics: pd.DataFrame
    point_gw_fundamental: pd.DataFrame
    point_gw_exaa: pd.DataFrame
    point_gw_all: pd.DataFrame
    median_mae: pd.DataFrame
    aps_per_timestamp: pd.DataFrame
    aps_summary: pd.DataFrame
    coverage: Mapping[str, pd.DataFrame]
    aps_gw: pd.DataFrame


def point_model_files(results_root: Path) -> Dict[str, Path]:
    """Return the 28 point-model inputs in the notebook's canonical order."""
    files = {
        run.name: run.output_dir(results_root) / "forecast.csv"
        for run in lear_runs()
    }
    files["exaa_naive"] = (
        results_root / "lear_op_results" / "exaa_naive" / "forecast.csv"
    )
    return files


def quantile_model_files(results_root: Path) -> Dict[str, Path]:
    """Return the six SQRA inputs using the notebook's display keys/order."""
    base = results_root / "sqra_results"
    return {
        "fund_ERA5": base / "era5_fundamental" / "forecast.csv",
        "fund_DWD": base / "dwd_fundamental" / "forecast.csv",
        "exaa_ERA5": base / "era5_exaa_enriched" / "forecast.csv",
        "exaa_DWD": base / "dwd_exaa_enriched" / "forecast.csv",
        "exaa_naive": base / "exaa_naive" / "forecast.csv",
        "exaa_only": base / "exaa_only" / "forecast.csv",
    }


def point_model_groups() -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Return fundamental, EXAA, and combined point-model plot groups."""
    clusters = (1, 5, 25)
    era5_days = (56, 112, 364)
    fundamental = tuple(
        [f"dwd_d56_c{clusters_value}_fundamental" for clusters_value in clusters]
        + [
            f"era5_d{days}_c{clusters_value}_fundamental"
            for days in era5_days
            for clusters_value in clusters
        ]
    )
    exaa = tuple(
        ["exaa_naive"]
        + [f"dwd_d56_c{clusters_value}_exaa" for clusters_value in clusters]
        + [
            f"era5_d{days}_c{clusters_value}_exaa"
            for days in era5_days
            for clusters_value in clusters
        ]
        + [f"exaa_only_d{days}" for days in era5_days]
    )
    return fundamental, exaa, fundamental + exaa


def _calendar_day(value: Union[pd.Timestamp, str]) -> pd.Timestamp:
    """Return a timezone-naive, normalized delivery date."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def read_forecast_csv(path: Path, timezone: str) -> pd.DataFrame:
    """Read a canonical ``delivery_date``/``mtu`` forecast export.

    ``timezone`` is retained in the public signature for compatibility with
    callers.  Delivery dates deliberately have no timezone: they describe a
    local market calendar day, while MTU identifies one of its 96 normalized
    quarter-hour positions.
    """
    del timezone
    return read_canonical_forecast_csv(path, require_complete_days=True)


def enforce_eval_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    skip_dates: Set[pd.Timestamp],
) -> pd.DataFrame:
    """Cut a panel to the inclusive evaluation window and excluded dates."""
    validate_delivery_index(frame.index)
    start_day = _calendar_day(start)
    end_day = _calendar_day(end)
    excluded = {_calendar_day(day) for day in skip_dates}
    dates = delivery_dates(frame.index)
    mask = (dates >= start_day) & (dates <= end_day)
    if excluded:
        mask &= ~dates.isin(excluded)
    return frame.loc[mask].copy()


def model_view(
    quantile_frame: pd.DataFrame,
    model_name: str,
    quantiles: Sequence[float] = QUANTILES,
    y_true_col: str = Y_TRUE_COL,
) -> pd.DataFrame:
    """Extract one SQRA model with unprefixed quantile column names."""
    model_columns = [f"{model_name}_q{quantile:.3f}" for quantile in quantiles]
    renamed = {
        f"{model_name}_q{quantile:.3f}": f"q{quantile:.3f}"
        for quantile in quantiles
    }
    return pd.concat(
        [quantile_frame[[y_true_col]], quantile_frame[model_columns].rename(columns=renamed)],
        axis=1,
    )


def check_index(
    frame: pd.DataFrame,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    n_periods: int,
    skip_dates: Set[pd.Timestamp],
) -> None:
    """Verify complete delivery-day/MTU coverage for the evaluation sample."""
    try:
        validate_delivery_index(frame.index)
    except ValueError as exc:
        raise ValueError(f"[{name}] {exc}") from exc

    del frequency
    start_day = _calendar_day(start)
    end_day = _calendar_day(end)
    excluded = {_calendar_day(day) for day in skip_dates}
    delivery_days = pd.date_range(start_day, end_day, freq="D")
    if excluded:
        delivery_days = delivery_days[~delivery_days.isin(excluded)]
    if n_periods != MTUS_PER_DAY:
        raise ValueError(
            f"[{name}] Canonical forecasts require n_periods={MTUS_PER_DAY}, "
            f"got {n_periods}."
        )
    expected = make_delivery_index(delivery_days)
    missing = expected.difference(frame.index)
    extra = frame.index.difference(expected)
    if len(missing) or len(extra):
        first_missing = missing[:3].tolist() if len(missing) else None
        raise ValueError(
            f"[{name}] Index does not match the expected canonical delivery grid. "
            f"Missing={len(missing)}, Extra={len(extra)}. "
            f"First missing: {first_missing}."
        )


def check_missing(frame: pd.DataFrame, name: str) -> None:
    """Reject panels containing missing values."""
    missing_columns = frame.columns[frame.isna().any()].tolist()
    if missing_columns:
        raise ValueError(f"[{name}] Missing values in columns: {missing_columns}.")


def check_numeric(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Reject non-numeric forecast or target columns."""
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"[{name}] Non-numeric columns: {non_numeric}.")


def check_quantile_monotonicity(
    frame: pd.DataFrame,
    model_name: str,
    quantiles: Sequence[float] = QUANTILES,
) -> None:
    """Reject quantile crossing for one SQRA model."""
    columns = [f"{model_name}_q{quantile:.3f}" for quantile in quantiles]
    values = frame[columns].to_numpy()
    differences = np.diff(values, axis=1)
    if not np.all(differences >= -1e-12):
        row = int(np.where(np.any(differences < -1e-12, axis=1))[0][0])
        raise ValueError(
            f"[{model_name}] Quantiles not monotone at row {row}, "
            f"time={frame.index[row]}, values={frame.iloc[row][columns].to_dict()}."
        )


def load_evaluation_data(config: ExperimentConfig) -> EvaluationData:
    """Load, align, and validate every forecast required by evaluation."""
    timezone = config.timezone
    start = pd.Timestamp(config.evaluation_start).normalize()
    end = pd.Timestamp(config.evaluation_end).normalize()
    skip_dates = {
        pd.Timestamp(day).normalize()
        for day in config.evaluation_skip_dates
    }

    y_true = None
    point_forecasts = []
    for model_name, path in point_model_files(config.results_root).items():
        frame = enforce_eval_window(
            read_forecast_csv(path, timezone), start, end, skip_dates
        )
        if y_true is None and Y_TRUE_COL in frame.columns:
            y_true = frame[[Y_TRUE_COL]]
        if "y_pred" not in frame.columns:
            raise ValueError(f"[{model_name}] Missing 'y_pred' column in {path}.")
        point_forecasts.append(
            frame[["y_pred"]].rename(columns={"y_pred": model_name})
        )
    if y_true is None:
        raise ValueError("Could not find y_true in any point forecast file.")
    point = pd.concat([y_true] + point_forecasts, axis=1).sort_index()

    quantile_columns = [f"q{quantile:.3f}" for quantile in QUANTILES]
    y_true_quantile = None
    quantile_forecasts = []
    for model_name, path in quantile_model_files(config.results_root).items():
        frame = enforce_eval_window(
            read_forecast_csv(path, timezone), start, end, skip_dates
        )
        if y_true_quantile is None and Y_TRUE_COL in frame.columns:
            y_true_quantile = frame[[Y_TRUE_COL]]
        missing = set(quantile_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"[{model_name}] Missing quantile columns: {missing}.")
        quantile_forecasts.append(
            frame[quantile_columns].rename(
                columns={
                    f"q{quantile:.3f}": f"{model_name}_q{quantile:.3f}"
                    for quantile in QUANTILES
                }
            )
        )
    if y_true_quantile is None:
        raise ValueError("Could not find y_true in any quantile forecast file.")
    quantile = pd.concat(
        [y_true_quantile] + quantile_forecasts, axis=1
    ).sort_index()

    check_index(point, "POINT", start, end, FREQ, N_PERIODS, skip_dates)
    check_index(quantile, "QUANTILE", start, end, FREQ, N_PERIODS, skip_dates)
    check_missing(point, "POINT")
    check_missing(quantile, "QUANTILE")
    check_numeric(point, point.columns.tolist(), "POINT")
    check_numeric(quantile, [Y_TRUE_COL], "QUANTILE")
    for model_name in quantile_model_files(config.results_root):
        columns = [
            f"{model_name}_q{quantile_level:.3f}"
            for quantile_level in QUANTILES
        ]
        check_numeric(quantile, columns, f"QUANTILE::{model_name}")
        check_quantile_monotonicity(quantile, model_name, QUANTILES)
    return EvaluationData(point=point, quantile=quantile)


def mae_point(
    frame: pd.DataFrame, model: str, y_true_col: str = Y_TRUE_COL
) -> float:
    """Return mean absolute point-forecast error."""
    if model not in frame.columns:
        raise ValueError(f"Column '{model}' not found in DataFrame.")
    if y_true_col not in frame.columns:
        raise ValueError(f"Column '{y_true_col}' not found in DataFrame.")
    return float(np.mean(np.abs(frame[y_true_col].values - frame[model].values)))


def rmse_point(
    frame: pd.DataFrame, model: str, y_true_col: str = Y_TRUE_COL
) -> float:
    """Return root mean squared point-forecast error."""
    if model not in frame.columns:
        raise ValueError(f"Column '{model}' not found in DataFrame.")
    if y_true_col not in frame.columns:
        raise ValueError(f"Column '{y_true_col}' not found in DataFrame.")
    error = frame[y_true_col].values - frame[model].values
    return float(np.sqrt(np.mean(error ** 2)))


def bias_point(
    frame: pd.DataFrame, model: str, y_true_col: str = Y_TRUE_COL
) -> float:
    """Return mean signed forecast error (forecast minus realised)."""
    if model not in frame.columns:
        raise ValueError(f"Column '{model}' not found in DataFrame.")
    if y_true_col not in frame.columns:
        raise ValueError(f"Column '{y_true_col}' not found in DataFrame.")
    return float(np.mean(frame[model].values - frame[y_true_col].values))


def GW(
    p_real: np.ndarray,
    p_pred_1: np.ndarray,
    p_pred_2: np.ndarray,
    norm: int = 1,
    version: str = "multivariate",
) -> Union[float, np.ndarray]:
    """One-sided Giacomini-White conditional predictive-accuracy test."""
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError("All three arrays must have the same shape.")
    if p_real.ndim == 1 or (p_real.ndim == 2 and p_real.shape[1] == 1):
        raise ValueError("Arrays must have shape (n_days, H) with H > 1.")

    loss_1 = p_real - p_pred_1
    loss_2 = p_real - p_pred_2
    differential = (
        np.abs(loss_1) - np.abs(loss_2)
        if norm == 1
        else loss_1 ** 2 - loss_2 ** 2
    )
    tau = 1
    n_days, n_periods = differential.shape

    if version == "univariate":
        statistic = np.full(n_periods, np.inf)
        for period in range(n_periods):
            instruments = np.stack(
                [
                    np.ones_like(differential[:-tau, period]),
                    differential[:-tau, period],
                ]
            )
            current = differential[tau:, period]
            sample_size = n_days - tau
            regression = np.array(instruments, ndmin=2) * current
            betas = np.linalg.lstsq(
                regression.T, np.ones(sample_size), rcond=None
            )[0]
            error = np.ones((sample_size, 1)) - regression.T @ betas
            statistic[period] = sample_size * (1 - np.mean(error ** 2))
        regression_for_df = regression
        sign_data = differential
    elif version == "multivariate":
        averaged = differential.mean(axis=1)
        instruments = np.stack([np.ones_like(averaged[:-tau]), averaged[:-tau]])
        current = averaged[tau:]
        sample_size = n_days - tau
        regression_for_df = np.array(instruments, ndmin=2) * current
        betas = np.linalg.lstsq(
            regression_for_df.T, np.ones(sample_size), rcond=None
        )[0]
        error = np.ones((sample_size, 1)) - regression_for_df.T @ betas
        statistic = sample_size * (1 - np.mean(error ** 2))
        sign_data = current
    else:
        raise ValueError("version must be 'univariate' or 'multivariate'.")

    statistic *= np.sign(np.mean(sign_data, axis=0))
    degrees_of_freedom = regression_for_df.shape[0]
    return 1 - scipy.stats.chi2.cdf(statistic, degrees_of_freedom)


def gw_test_point(
    frame: pd.DataFrame,
    model_1: str,
    model_2: str,
    y_true_col: str = Y_TRUE_COL,
    n_periods: int = N_PERIODS,
    norm: int = 1,
    version: str = "multivariate",
) -> Union[float, np.ndarray]:
    """Apply the GW test to flat point-forecast columns."""
    realised = frame[y_true_col].values.reshape(-1, n_periods)
    prediction_1 = frame[model_1].values.reshape(-1, n_periods)
    prediction_2 = frame[model_2].values.reshape(-1, n_periods)
    return GW(realised, prediction_1, prediction_2, norm=norm, version=version)


def pairwise_point_gw(
    frame: pd.DataFrame,
    model_columns: Sequence[str],
    y_true_col: str = Y_TRUE_COL,
    n_periods: int = N_PERIODS,
    norm: int = 1,
) -> pd.DataFrame:
    """Return the pairwise multivariate GW matrix for point forecasts."""
    if len(frame) % n_periods:
        raise ValueError(
            f"len(df)={len(frame)} is not divisible by n_periods={n_periods}."
        )
    models = list(model_columns)
    p_values = pd.DataFrame(1.0, index=models, columns=models)
    for model_1 in models:
        for model_2 in models:
            if model_1 != model_2:
                p_values.loc[model_1, model_2] = gw_test_point(
                    frame,
                    model_1,
                    model_2,
                    y_true_col=y_true_col,
                    n_periods=n_periods,
                    norm=norm,
                    version="multivariate",
                )
    return p_values


def mae_for_median(
    frame: pd.DataFrame,
    y_true_col: str = Y_TRUE_COL,
    q_median: float = 0.5,
) -> float:
    """Return MAE for the median quantile forecast."""
    quantile_column = f"q{q_median:.3f}"
    if y_true_col not in frame.columns:
        raise ValueError(f"Missing column '{y_true_col}'.")
    if quantile_column not in frame.columns:
        raise ValueError(f"Missing median quantile column '{quantile_column}'.")
    return float(
        np.mean(
            np.abs(
                frame[y_true_col].to_numpy()
                - frame[quantile_column].to_numpy()
            )
        )
    )


def pinball_score(y: np.ndarray, prediction: np.ndarray, tau: float) -> np.ndarray:
    """Return element-wise pinball loss."""
    difference = y - prediction
    return np.maximum(tau * difference, (tau - 1) * difference)


def aps_loss_per_timestamp(
    frame: pd.DataFrame,
    quantiles: Sequence[float] = QUANTILES,
    y_true_col: str = Y_TRUE_COL,
) -> pd.Series:
    """Return pinball loss averaged over quantiles for each timestamp."""
    realised = frame[y_true_col].to_numpy()
    losses = sum(
        pinball_score(
            realised,
            frame[f"q{quantile:.3f}"].to_numpy(),
            quantile,
        )
        for quantile in quantiles
    )
    return pd.Series(losses / len(quantiles), index=frame.index, name="APS")


def aps_loss_matrix(
    frame: pd.DataFrame,
    quantiles: Sequence[float] = QUANTILES,
    y_true_col: str = Y_TRUE_COL,
    n_periods: int = N_PERIODS,
) -> np.ndarray:
    """Return APS as a matrix with days in rows and MTUs in columns."""
    losses = aps_loss_per_timestamp(frame, quantiles, y_true_col).to_numpy()
    if len(losses) % n_periods:
        raise ValueError(
            f"len(df)={len(losses)} is not divisible by n_periods={n_periods}."
        )
    return losses.reshape(len(losses) // n_periods, n_periods)


def aggregated_pinball_score(
    frame: pd.DataFrame,
    quantiles: Sequence[float] = QUANTILES,
    y_true_col: str = Y_TRUE_COL,
) -> float:
    """Return pinball loss averaged over quantiles and timestamps."""
    return float(aps_loss_per_timestamp(frame, quantiles, y_true_col).mean())


def coverage_indicator(
    frame: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    y_true_col: str = Y_TRUE_COL,
) -> pd.Series:
    """Return whether each realised value lies within the interval."""
    lower_column = f"q{lower_quantile:.3f}"
    upper_column = f"q{upper_quantile:.3f}"
    if lower_column not in frame.columns or upper_column not in frame.columns:
        raise ValueError(
            f"Missing quantile column(s): '{lower_column}', '{upper_column}'."
        )
    return (
        (frame[y_true_col] >= frame[lower_column])
        & (frame[y_true_col] <= frame[upper_column])
    ).astype(int)


def empirical_coverage(
    frame: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    y_true_col: str = Y_TRUE_COL,
) -> float:
    """Return empirical prediction-interval coverage."""
    return float(
        coverage_indicator(
            frame, lower_quantile, upper_quantile, y_true_col
        ).mean()
    )


def kupiec_test(indicators: np.ndarray, alpha: float) -> float:
    """Return the Kupiec unconditional-coverage test p-value."""
    n_observations = len(indicators)
    n_covered = indicators.sum()
    if n_covered == 0 or n_covered == n_observations:
        return 0.0
    empirical = n_covered / n_observations
    statistic = -2 * (
        (n_observations - n_covered)
        * np.log((1 - alpha) / (1 - empirical))
        + n_covered * np.log(alpha / empirical)
    )
    return float(1 - scipy.stats.chi2.cdf(statistic, df=1))


def mtu_kupiec_test(
    frame: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    alpha: float,
    y_true_col: str = Y_TRUE_COL,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> int:
    """Return the number of MTUs passing the Kupiec coverage test."""
    validate_delivery_index(frame.index)
    indicators = coverage_indicator(
        frame, lower_quantile, upper_quantile, y_true_col
    ).to_numpy()
    mtu_index = np.asarray(frame.index.get_level_values("mtu"), dtype=int)
    return sum(
        kupiec_test(indicators[mtu_index == mtu], alpha) >= significance_level
        for mtu in range(1, N_PERIODS + 1)
        if (mtu_index == mtu).any()
    )


def GW_loss(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    version: str = "multivariate",
) -> Union[float, np.ndarray]:
    """Apply the GW test to precomputed loss matrices."""
    if loss_1.shape != loss_2.shape:
        raise ValueError("loss_1 and loss_2 must have the same shape.")
    if loss_1.ndim != 2:
        raise ValueError("Loss matrices must have shape (n_days, n_periods).")
    differential = loss_1 - loss_2
    tau = 1
    n_days, n_periods = differential.shape
    sample_size = n_days - tau

    if version == "univariate":
        statistic = np.full(n_periods, np.nan)
        for period in range(n_periods):
            current = differential[tau:, period]
            instruments = np.vstack(
                [np.ones(sample_size), differential[:-tau, period]]
            )
            regression = instruments * current
            betas = np.linalg.lstsq(
                regression.T, np.ones(sample_size), rcond=None
            )[0]
            error = np.ones(sample_size) - regression.T @ betas
            statistic[period] = sample_size * (1.0 - np.mean(error ** 2))
        statistic *= np.sign(np.mean(differential[tau:], axis=0))
        return 1.0 - scipy.stats.chi2.cdf(statistic, df=2)
    if version == "multivariate":
        averaged = differential.mean(axis=1)
        current = averaged[tau:]
        instruments = np.vstack([np.ones(sample_size), averaged[:-tau]])
        regression = instruments * current
        betas = np.linalg.lstsq(
            regression.T, np.ones(sample_size), rcond=None
        )[0]
        error = np.ones(sample_size) - regression.T @ betas
        statistic = sample_size * (1.0 - np.mean(error ** 2))
        statistic *= np.sign(current.mean())
        return float(1.0 - scipy.stats.chi2.cdf(statistic, df=2))
    raise ValueError("version must be 'univariate' or 'multivariate'.")


def pairwise_loss_gw(losses: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return the pairwise multivariate GW matrix for arbitrary losses."""
    models = list(losses)
    p_values = pd.DataFrame(1.0, index=models, columns=models)
    for model_1 in models:
        for model_2 in models:
            if model_1 != model_2:
                p_values.loc[model_1, model_2] = GW_loss(
                    losses[model_1], losses[model_2], version="multivariate"
                )
    return p_values


def evaluate(data: EvaluationData) -> EvaluationResults:
    """Compute all notebook metrics and statistical comparison matrices."""
    point_models = [
        column for column in data.point.columns if column != Y_TRUE_COL
    ]
    point_metrics = pd.DataFrame(
        {
            model: {
                "MAE": mae_point(data.point, model),
                "RMSE": rmse_point(data.point, model),
                "Bias": bias_point(data.point, model),
            }
            for model in point_models
        }
    ).T.sort_values("MAE")

    fundamental, exaa, all_models = point_model_groups()
    point_gw_fundamental = pairwise_point_gw(data.point, fundamental)
    point_gw_exaa = pairwise_point_gw(data.point, exaa)
    point_gw_all = pairwise_point_gw(data.point, all_models)

    median_mae = pd.DataFrame(
        {
            model_name: {
                "MAE (Median)": mae_for_median(
                    model_view(data.quantile, model_name)
                )
            }
            for model_name in QUANTILE_MODELS
        }
    ).T.sort_values("MAE (Median)")

    aps_per_timestamp = pd.DataFrame(
        {
            model_name: aps_loss_per_timestamp(
                model_view(data.quantile, model_name)
            )
            for model_name in QUANTILE_MODELS
        }
    ).sort_index()
    aps_summary = (
        aps_per_timestamp.mean()
        .rename("APS")
        .to_frame()
        .sort_values("APS")
    )

    coverage_tables: Dict[str, pd.DataFrame] = {}
    for lower_quantile, upper_quantile in PREDICTION_INTERVALS:
        nominal = upper_quantile - lower_quantile
        label = f"PI_{int(lower_quantile * 100)}_{int(upper_quantile * 100)}"
        coverage_tables[label] = pd.DataFrame(
            {
                model_name: {
                    "Empirical Coverage": empirical_coverage(
                        model_view(data.quantile, model_name),
                        lower_quantile,
                        upper_quantile,
                    ),
                    "Nominal Coverage": nominal,
                    "Kupiec MTUs passed": mtu_kupiec_test(
                        model_view(data.quantile, model_name),
                        lower_quantile,
                        upper_quantile,
                        alpha=nominal,
                    ),
                }
                for model_name in QUANTILE_MODELS
            }
        ).T

    loss_matrices = {
        model_name: aps_loss_matrix(model_view(data.quantile, model_name))
        for model_name in QUANTILE_MODELS
    }
    aps_gw = pairwise_loss_gw(loss_matrices)
    return EvaluationResults(
        point_metrics=point_metrics,
        point_gw_fundamental=point_gw_fundamental,
        point_gw_exaa=point_gw_exaa,
        point_gw_all=point_gw_all,
        median_mae=median_mae,
        aps_per_timestamp=aps_per_timestamp,
        aps_summary=aps_summary,
        coverage=coverage_tables,
        aps_gw=aps_gw,
    )


def plot_gw_heatmap(
    p_values: pd.DataFrame,
    title: str = "GW test",
    figsize: Tuple[float, float] = (12, 9),
    fontsize: int = 10,
    label_map: Optional[Mapping[str, str]] = None,
    fraction: float = 0.03,
    pad: float = 0.01,
):
    """Build the notebook's red/green GW heatmap and return its figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    diagonal_grey = 0.12
    red = np.concatenate(
        [np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [diagonal_grey]]
    )
    green = np.concatenate(
        [np.linspace(0.5, 1, 50), np.zeros(49), [diagonal_grey]]
    )
    blue = np.concatenate([np.zeros(99), [diagonal_grey]])
    color_map = mpl.colors.ListedColormap(
        np.stack([red, green, blue], axis=1)
    )
    models = list(p_values.columns)
    labels = (
        [label_map.get(model, model) for model in models]
        if label_map is not None
        else models
    )
    n_models = len(models)
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        p_values.astype(float).values,
        cmap=color_map,
        vmin=0,
        vmax=0.1,
        interpolation="nearest",
    )
    lines = []
    for position in range(n_models + 1):
        lines.append(
            [(position - 0.5, -0.5), (position - 0.5, n_models - 0.5)]
        )
        lines.append(
            [(-0.5, position - 0.5), (n_models - 0.5, position - 0.5)]
        )
    axis.add_collection(
        LineCollection(lines, colors="black", linewidths=1.4, zorder=10)
    )
    axis.set_xlim(-0.5, n_models - 0.5)
    axis.set_ylim(n_models - 0.5, -0.5)
    axis.set_xticks(range(n_models))
    axis.set_yticks(range(n_models))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    axis.set_yticklabels(labels, fontsize=fontsize)
    axis.plot(range(n_models), range(n_models), "wx", markersize=8, zorder=11)
    colorbar = figure.colorbar(image, ax=axis, fraction=fraction, pad=pad)
    colorbar.set_label("p-value", fontsize=fontsize + 1)
    colorbar.ax.tick_params(labelsize=fontsize)
    if title:
        axis.set_title(title, fontsize=fontsize + 1)
    figure.tight_layout()
    return figure
