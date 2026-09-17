"""Canonical local delivery-day/MTU indexing for forecast result files.

Physical timezone-aware timestamps contain 92 or 100 quarter-hours on European
daylight-saving transition days.  The forecasting models, however, use the
manuscript's normalized daily representation with 96 local-clock MTUs.  Result
files therefore use the explicit key ``(delivery_date, mtu)`` instead of
pretending that every canonical MTU has a physical timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline.forecast_schema import (
    DELIVERY_DATE_COLUMN,
    FORECAST_INDEX_NAME,
    FORECAST_SCHEMA_VERSION,
    INDEX_COLUMNS,
    MTU_COLUMN,
    MTUS_PER_DAY,
)


def _naive_normalized_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def make_delivery_index(days: Iterable[object]) -> pd.MultiIndex:
    """Return all 96 canonical MTUs for each supplied local delivery date."""
    dates = _naive_normalized_dates(days).sort_values()
    if dates.has_duplicates:
        raise ValueError("Delivery days must be unique.")
    return pd.MultiIndex.from_product(
        [dates, range(1, MTUS_PER_DAY + 1)],
        names=list(INDEX_COLUMNS),
    )


def validate_delivery_index(
    index: pd.Index,
    *,
    require_complete_days: bool = False,
    expected_days: Sequence[object] | None = None,
) -> None:
    """Validate the canonical, sorted, unique delivery-date/MTU contract."""
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != INDEX_COLUMNS:
        raise ValueError(
            "Forecast index must be a MultiIndex named "
            f"{INDEX_COLUMNS}, got {getattr(index, 'names', None)}."
        )
    if not index.is_unique:
        raise ValueError("Forecast index contains duplicate delivery-date/MTU keys.")
    if not index.is_monotonic_increasing:
        raise ValueError("Forecast index must be sorted by delivery_date and mtu.")

    dates = pd.DatetimeIndex(index.get_level_values(DELIVERY_DATE_COLUMN))
    if dates.tz is not None:
        raise ValueError("delivery_date must be timezone-naive.")
    if not dates.equals(dates.normalize()):
        raise ValueError("delivery_date values must be normalized to midnight.")

    raw_mtus = np.asarray(index.get_level_values(MTU_COLUMN))
    try:
        numeric_mtus = raw_mtus.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("mtu values must be integers from 1 through 96.") from exc
    if (
        not np.isfinite(numeric_mtus).all()
        or not np.equal(numeric_mtus, np.floor(numeric_mtus)).all()
        or not ((numeric_mtus >= 1) & (numeric_mtus <= MTUS_PER_DAY)).all()
    ):
        raise ValueError("mtu values must be integers from 1 through 96.")

    if require_complete_days:
        unique_days = dates.unique().sort_values()
        expected = make_delivery_index(unique_days)
        if not index.equals(expected):
            missing = expected.difference(index)
            extra = index.difference(expected)
            raise ValueError(
                "Each delivery day must contain MTUs 1 through 96 exactly once. "
                f"Missing={len(missing)}, Extra={len(extra)}."
            )

    if expected_days is not None:
        expected = make_delivery_index(expected_days)
        missing = expected.difference(index)
        extra = index.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(
                "Forecast delivery index does not match the expected days. "
                f"Missing={len(missing)}, Extra={len(extra)}."
            )


def read_forecast_csv(
    path: Path,
    *,
    required_columns: Sequence[str] = (),
    require_complete_days: bool = False,
) -> pd.DataFrame:
    """Read and validate a schema-v2 canonical forecast CSV."""
    frame = pd.read_csv(path)
    missing_index = set(INDEX_COLUMNS) - set(frame.columns)
    if missing_index:
        raise ValueError(
            f"Forecast file lacks canonical index columns {sorted(missing_index)}: {path}"
        )
    missing_values = set(required_columns) - set(frame.columns)
    if missing_values:
        raise ValueError(
            f"Forecast file lacks required columns {sorted(missing_values)}: {path}"
        )

    frame[DELIVERY_DATE_COLUMN] = pd.to_datetime(
        frame[DELIVERY_DATE_COLUMN], errors="raise"
    ).dt.normalize()
    numeric_mtu = pd.to_numeric(frame[MTU_COLUMN], errors="raise")
    if not np.equal(numeric_mtu, np.floor(numeric_mtu)).all():
        raise ValueError(f"Forecast MTUs must be integers: {path}")
    frame[MTU_COLUMN] = numeric_mtu.astype(int)
    frame = frame.set_index(list(INDEX_COLUMNS))
    validate_delivery_index(frame.index, require_complete_days=require_complete_days)
    return frame


def delivery_dates(index: pd.MultiIndex) -> pd.DatetimeIndex:
    """Return the delivery-date level as a DatetimeIndex after validation."""
    validate_delivery_index(index)
    return pd.DatetimeIndex(index.get_level_values(DELIVERY_DATE_COLUMN))


def delivery_clock_index(index: pd.MultiIndex) -> pd.DatetimeIndex:
    """Create timezone-naive local-clock labels for display only."""
    validate_delivery_index(index)
    dates = pd.DatetimeIndex(index.get_level_values(DELIVERY_DATE_COLUMN))
    mtus = np.asarray(index.get_level_values(MTU_COLUMN), dtype=int)
    return pd.DatetimeIndex(
        dates + pd.to_timedelta((mtus - 1) * 15, unit="min"),
        name="delivery_time",
    )


def schema_metadata() -> dict[str, object]:
    """Return metadata persisted in LEAR and SQRA configuration files."""
    return {
        "forecast_schema_version": FORECAST_SCHEMA_VERSION,
        "forecast_index": FORECAST_INDEX_NAME,
        "mtus_per_delivery_day": MTUS_PER_DAY,
    }
