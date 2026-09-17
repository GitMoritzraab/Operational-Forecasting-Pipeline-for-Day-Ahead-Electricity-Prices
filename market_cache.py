"""Persistent cache helper for the ENTSO-E inputs used by LEAR."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import tempfile
from datetime import timedelta

import pandas as pd

from pipeline.operational.locking import cache_lock_path, interprocess_lock


def _read(path: Path, timezone: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(timezone)
    frame.index.name = "timestamp"
    return frame.sort_index()


def load_or_fetch_frame(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetcher: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    *,
    timezone: str,
    refresh: bool = False,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    """Return a cached interval, fetching and merging when necessary.

    ``allow_incomplete`` is intended for operational inputs whose upstream
    source can occasionally omit a historical delivery day.  The cache is
    still refreshed first; unresolved cells are returned as ``NaN`` so the
    caller can apply an explicit model-level missing-data policy.
    """
    with interprocess_lock(
        cache_lock_path(path),
        timeout_seconds=15 * 60,
        description=f"market cache {path.name}",
    ):
        return _load_or_fetch_frame_unlocked(
            path,
            start,
            end,
            fetcher,
            timezone=timezone,
            refresh=refresh,
            allow_incomplete=allow_incomplete,
        )


def _load_or_fetch_frame_unlocked(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetcher: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    *,
    timezone: str,
    refresh: bool = False,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    """Implementation executed while the cache-specific lock is held."""
    start_cut = start.tz_convert(timezone).normalize()
    end_day = end.tz_convert(timezone).date()
    end_exclusive = pd.Timestamp(end_day + timedelta(days=1), tz=timezone)
    end_cut = end_exclusive - pd.Timedelta(minutes=15)
    expected_index = pd.date_range(
        start=start_cut,
        end=end_exclusive,
        freq="15min",
        inclusive="left",
    )

    def report_incomplete(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.reindex(expected_index).copy()
        missing_mask = result.isna().any(axis=1)
        missing_days = sorted(
            {timestamp.date().isoformat() for timestamp in result.index[missing_mask]}
        )
        print(
            f"WARNING: {path.name} remains incomplete for "
            f"{int(missing_mask.sum())} timestamp(s) on delivery day(s): "
            + ", ".join(missing_days),
            flush=True,
        )
        return result

    def complete(frame: pd.DataFrame) -> bool:
        if frame.empty or not expected_index.isin(frame.index).all():
            return False
        selected = frame.reindex(expected_index)
        return not selected.isna().to_numpy().any()

    cached: pd.DataFrame | None = None
    if path.exists() and not refresh:
        cached = _read(path, timezone)
        if complete(cached):
            return cached.reindex(expected_index).copy()

    fetch_start = start
    fetch_end = end
    if cached is not None:
        cached_requested = cached.reindex(expected_index)
        missing_mask = cached_requested.isna().any(axis=1)
        missing_index = expected_index[missing_mask]
        if len(missing_index):
            # Fetch complete delivery days around only the unresolved/new part
            # of the interval instead of repeatedly downloading all history.
            fetch_start = missing_index.min().normalize()
            fetch_end = missing_index.max().normalize()
    print(
        f"Updating {path.name}: {fetch_start.date()} through {fetch_end.date()}",
        flush=True,
    )
    fresh = fetcher(fetch_start, fetch_end)
    if fresh.empty:
        if allow_incomplete and cached is not None:
            # The API can return no rows at all for one unpublished historical
            # delivery day.  Preserve the known cache gap and let the caller's
            # explicit missing-data policy handle it.
            return report_incomplete(cached)
        raise ValueError(f"No data returned while refreshing {path.name}.")

    if cached is None:
        combined = fresh
    else:
        # Prefer newly published values, but never let a partial API response
        # replace a valid cached value with NaN.
        combined = fresh.combine_first(cached)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        combined.to_csv(temporary_path, index=True)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    result = combined.reindex(expected_index).copy()
    if not complete(combined):
        if allow_incomplete:
            return report_incomplete(combined)
        raise ValueError(
            f"{path.name} does not completely cover the requested interval "
            f"{start_cut.date()} through {end_cut.date()}."
        )
    return result
