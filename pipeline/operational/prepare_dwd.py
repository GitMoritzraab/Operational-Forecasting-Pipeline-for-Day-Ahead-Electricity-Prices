"""Download, consolidate, verify, and clean up operational ICON-D2 data."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from pipeline.dwd_history import (
    append_daily_output,
    consolidated_delivery_available,
    migrate_daily_outputs,
)
from pipeline.operational.config import OperationalConfig, WEATHER_CLUSTERS
from pipeline.operational.dwd import cleanup_raw_run, download_run
from pipeline.operational.locking import interprocess_lock


@contextmanager
def _preparation_lock(config: OperationalConfig):
    path = config.output_root / "prepare_dwd_data.lock"
    with interprocess_lock(
        path,
        timeout_seconds=30 * 60,
        description="DWD preparation",
    ):
        yield


def migrate_existing_processed(
    config: OperationalConfig,
    *,
    clusters: Sequence[int] = WEATHER_CLUSTERS,
    force: bool = False,
) -> dict[int, int]:
    """Create consolidated histories from the existing daily output folders."""
    counts: dict[int, int] = {}
    for cluster in clusters:
        cluster_dir = config.icon_cluster_dir(cluster)
        count = migrate_daily_outputs(
            cluster_dir,
            timezone=config.timezone,
            force=force,
        )
        counts[cluster] = count
        if count:
            print(
                f"DWD C={cluster}: consolidated {count} existing daily outputs.",
                flush=True,
            )
    return counts


def _target_ready(
    config: OperationalConfig,
    target_date: date,
    clusters: Sequence[int],
) -> bool:
    return all(
        consolidated_delivery_available(
            config.icon_cluster_dir(cluster),
            delivery_date=target_date,
            run_hour=config.dwd_operational_run_hour,
            timezone=config.timezone,
        )
        for cluster in clusters
    )


def _aggregate_target(
    config: OperationalConfig,
    *,
    issue_date: date,
    target_date: date,
    clusters: Sequence[int],
) -> None:
    script = config.repo_root / "preprocessing" / "aggregate_icon_d2.py"
    with tempfile.TemporaryDirectory(prefix="dwd_processed_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for cluster in clusters:
            environment = os.environ.copy()
            environment.update(
                {
                    "DWD_RAW_ARCHIVE": str(config.dwd_raw_archive),
                    "ICON_DATA_ROOT": str(temporary_root),
                    "DWD_N_CLUSTERS": str(cluster),
                    "DWD_PREPROCESS_START_DATE": issue_date.isoformat(),
                    "DWD_RUN_HOUR": config.dwd_operational_run_hour,
                    "DWD_ONLY_DAY": issue_date.strftime("%Y%m%d"),
                    "DWD_SKIP_EXISTING_OUTPUT": "false",
                    "EVALUATION_END_DATE": issue_date.isoformat(),
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                }
            )
            print(
                f"DWD preprocess issue={issue_date}, run="
                f"{config.dwd_operational_run_hour}, C={cluster}",
                flush=True,
            )
            subprocess.run(
                [sys.executable, str(script)],
                cwd=config.repo_root,
                env=environment,
                check=True,
            )
            daily_folder = (
                temporary_root
                / f"c{cluster}"
                / (
                    f"dwd_icon_daily_{issue_date:%Y%m%d}_"
                    f"{config.dwd_operational_run_hour}"
                )
            )
            if not daily_folder.is_dir():
                raise RuntimeError(
                    f"DWD aggregation did not create its expected output: {daily_folder}"
                )
            append_daily_output(
                config.icon_cluster_dir(cluster),
                daily_folder,
                issue_date=issue_date,
                run_hour=config.dwd_operational_run_hour,
                delivery_date=target_date,
                timezone=config.timezone,
            )


def _prepare_target_dwd_unlocked(
    config: OperationalConfig,
    target_date: date,
    *,
    force_download: bool = False,
    force_migration: bool = False,
    delete_raw: bool = True,
    clusters: Sequence[int] = WEATHER_CLUSTERS,
) -> dict[str, object]:
    """Prepare all cluster histories for one delivery day, idempotently."""
    if config.dwd_operational_run_hour != "06":
        raise ValueError(
            "Operational DWD preparation is restricted to the 06 UTC ICON-D2 run."
        )
    selected_clusters = tuple(sorted(set(int(value) for value in clusters)))
    invalid = sorted(set(selected_clusters) - set(WEATHER_CLUSTERS))
    if invalid:
        raise ValueError(f"Unsupported DWD cluster counts: {invalid}")
    migrate_existing_processed(
        config,
        clusters=selected_clusters,
        force=force_migration,
    )
    issue_date = target_date - timedelta(days=1)
    if force_download or not _target_ready(config, target_date, selected_clusters):
        download_run(config, issue_date, force=force_download)
        _aggregate_target(
            config,
            issue_date=issue_date,
            target_date=target_date,
            clusters=selected_clusters,
        )
    if not _target_ready(config, target_date, selected_clusters):
        raise RuntimeError(
            f"DWD histories failed final validation for delivery {target_date}."
        )

    removed: list[str] = []
    if delete_raw:
        removed = [
            str(path)
            for path in cleanup_raw_run(
                config.dwd_raw_archive,
                issue_date,
                config.dwd_operational_run_hour,
            )
        ]
        if removed:
            print(
                f"DWD cleanup: removed {len(removed)} consumed raw item(s) for "
                f"{issue_date} {config.dwd_operational_run_hour}.",
                flush=True,
            )
    print(
        f"DWD histories ready for delivery {target_date}: "
        + ", ".join(f"C={value}" for value in selected_clusters),
        flush=True,
    )
    return {
        "target_date": target_date.isoformat(),
        "issue_date": issue_date.isoformat(),
        "run_hour": config.dwd_operational_run_hour,
        "clusters": list(selected_clusters),
        "raw_removed": removed,
    }


def prepare_target_dwd(
    config: OperationalConfig,
    target_date: date,
    *,
    force_download: bool = False,
    force_migration: bool = False,
    delete_raw: bool = True,
    clusters: Sequence[int] = WEATHER_CLUSTERS,
) -> dict[str, object]:
    """Run target preparation under a lock shared with the forecast pipeline."""
    with _preparation_lock(config):
        return _prepare_target_dwd_unlocked(
            config,
            target_date,
            force_download=force_download,
            force_migration=force_migration,
            delete_raw=delete_raw,
            clusters=clusters,
        )
