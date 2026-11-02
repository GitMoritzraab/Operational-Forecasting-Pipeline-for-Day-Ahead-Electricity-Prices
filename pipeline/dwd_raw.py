"""Compatibility helpers for directory-based and ZIP-based ICON-D2 archives."""

from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence
from zipfile import ZipFile


REGULAR_GRID_TOKEN = "regular-lat-lon"
ARCHIVED_PREFIX = "dwd_icon_archived_"
LEGACY_PREFIX = "dwd_icon_daily_"
TIMESTAMPED_RUN_PATTERN = re.compile(r"^(\d{8})_([01]\d|2[0-3])$")
EXPECTED_LEADS = frozenset(range(49))


@dataclass(frozen=True)
class RawRunLocation:
    issue_date: date
    run_hour: str
    kind: str
    path: Path


def legacy_day_dir(root: Path, issue_date: date) -> Path:
    return root / f"{LEGACY_PREFIX}{issue_date:%Y%m%d}"


def legacy_run_dir(root: Path, issue_date: date, run_hour: str) -> Path:
    return legacy_day_dir(root, issue_date) / "icon-d2" / run_hour


def archived_day_dir(root: Path, issue_date: date) -> Path:
    return root / f"{ARCHIVED_PREFIX}{issue_date:%Y%m%d}"


def timestamped_run_dir(root: Path, issue_date: date, run_hour: str) -> Path:
    return root / f"{issue_date:%Y%m%d}_{run_hour}"


def _timestamped_runs(
    root: Path, issue_date: date | None = None
) -> Iterator[tuple[date, str, Path]]:
    """Discover valid YYYYMMDD_HH folders without treating JSON files as runs."""
    pattern = (
        f"{issue_date:%Y%m%d}_[0-9][0-9]"
        if issue_date is not None
        else "[0-9]*_[0-9][0-9]"
    )
    for path in root.glob(pattern):
        match = TIMESTAMPED_RUN_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        try:
            folder_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        yield folder_date, match.group(2), path


def variable_archive_path(
    root: Path,
    issue_date: date,
    run_hour: str,
    variable: str,
) -> Path:
    return archived_day_dir(root, issue_date) / variable_archive_filename(
        run_hour, variable
    )


def variable_archive_filename(run_hour: str, variable: str) -> str:
    return f"icon-d2__{run_hour}__{variable}.zip"


def lead_from_filename(
    filename: str,
    issue_token: str,
    variable: str,
    *,
    grid_token: str | None = REGULAR_GRID_TOKEN,
) -> int | None:
    lower = Path(filename).name.lower()
    if (
        (grid_token is not None and grid_token.lower() not in lower)
        or issue_token not in lower
        or not lower.endswith(".grib2.bz2")
        or f"_{variable.lower()}.grib2.bz2" not in lower
    ):
        return None
    match = re.search(rf"_{re.escape(issue_token)}_(\d{{3}})_", lower)
    return int(match.group(1)) if match else None


def zip_regular_members(
    archive_path: Path,
    issue_date: date,
    run_hour: str,
    variable: str,
) -> dict[int, str]:
    """Return regular-grid members keyed by forecast lead."""
    issue_token = f"{issue_date:%Y%m%d}{run_hour}"
    members: dict[int, str] = {}
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            lead = lead_from_filename(
                info.filename,
                issue_token,
                variable,
                grid_token=REGULAR_GRID_TOKEN,
            )
            if lead is not None:
                if lead in members:
                    raise ValueError(
                        f"Duplicate lead {lead:03d} in DWD archive {archive_path}."
                    )
                members[lead] = info.filename
    return members


def locate_raw_run(root: Path, issue_date: date, run_hour: str) -> RawRunLocation | None:
    """Locate ZIP sets in daily, archived, or YYYYMMDD_HH containers."""
    for archive_dir in (
        archived_day_dir(root, issue_date),
        legacy_day_dir(root, issue_date),
        timestamped_run_dir(root, issue_date, run_hour),
    ):
        if archive_dir.is_dir() and any(
            archive_dir.glob(f"icon-d2__{run_hour}__*.zip")
        ):
            return RawRunLocation(issue_date, run_hour, "zip", archive_dir)
    run_dir = legacy_run_dir(root, issue_date, run_hour)
    if run_dir.is_dir():
        return RawRunLocation(issue_date, run_hour, "directory", run_dir)
    return None


def raw_run_available(root: Path, issue_date: date, run_hour: str) -> bool:
    return locate_raw_run(root, issue_date, run_hour) is not None


def available_variables(
    root: Path,
    issue_date: date,
    run_hour: str,
    allowed_variables: Sequence[str],
) -> list[str]:
    location = locate_raw_run(root, issue_date, run_hour)
    if location is None:
        return []
    if location.kind == "zip":
        return [
            variable
            for variable in allowed_variables
            if (
                location.path / variable_archive_filename(run_hour, variable)
            ).is_file()
        ]
    return [
        variable
        for variable in allowed_variables
        if (location.path / variable).is_dir()
    ]


def discover_issue_dates(root: Path) -> list[date]:
    """Discover issue dates from all supported top-level folder names."""
    dates: set[date] = set()
    for prefix in (LEGACY_PREFIX, ARCHIVED_PREFIX):
        for path in root.glob(f"{prefix}*"):
            if not path.is_dir():
                continue
            token = path.name[len(prefix) :]
            try:
                dates.add(datetime.strptime(token, "%Y%m%d").date())
            except ValueError:
                continue
    dates.update(issue_date for issue_date, _, _ in _timestamped_runs(root))
    return sorted(dates)


def available_run_hours(root: Path, issue_date: date) -> list[str]:
    hours: set[str] = set()
    legacy = legacy_day_dir(root, issue_date) / "icon-d2"
    if legacy.is_dir():
        hours.update(
            path.name
            for path in legacy.iterdir()
            if path.is_dir() and len(path.name) == 2 and path.name.isdigit()
        )
    for archive_dir in (
        archived_day_dir(root, issue_date),
        legacy_day_dir(root, issue_date),
    ):
        if not archive_dir.is_dir():
            continue
        pattern = re.compile(r"^icon-d2__(\d{2})__.+\.zip$", re.IGNORECASE)
        for path in archive_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                hours.add(match.group(1))
    for _, run_hour, archive_dir in _timestamped_runs(root, issue_date):
        if any(archive_dir.glob(f"icon-d2__{run_hour}__*.zip")):
            hours.add(run_hour)
    return sorted(hours)


@contextmanager
def materialize_raw_run(
    root: Path,
    issue_date: date,
    run_hour: str,
    variables: Sequence[str],
) -> Iterator[Path]:
    """Yield a legacy run directory or a temporary regular-grid ZIP extraction."""
    location = locate_raw_run(root, issue_date, run_hour)
    if location is None:
        raise FileNotFoundError(
            f"No directory or ZIP DWD run found for {issue_date} {run_hour}."
        )
    if location.kind == "directory":
        yield location.path
        return

    with tempfile.TemporaryDirectory(prefix="dwd_icon_zip_") as temporary_directory:
        run_dir = Path(temporary_directory) / "icon-d2" / run_hour
        for variable in variables:
            archive_path = location.path / variable_archive_filename(
                run_hour, variable
            )
            if not archive_path.is_file():
                continue
            members = zip_regular_members(
                archive_path, issue_date, run_hour, variable
            )
            missing = sorted(EXPECTED_LEADS - set(members))
            if missing:
                raise ValueError(
                    f"DWD archive {archive_path} is incomplete; missing "
                    f"regular-grid leads {missing}."
                )
            variable_dir = run_dir / variable
            variable_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path) as archive:
                for member_name in members.values():
                    destination = variable_dir / Path(member_name).name
                    with archive.open(member_name) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
        yield run_dir
