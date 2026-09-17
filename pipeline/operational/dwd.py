"""Download, verify, and preprocess daily DWD ICON-D2 inputs."""

from __future__ import annotations

import bz2
import json
import os
import shutil
import tempfile
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import requests

from dwd_icon_downloader import download_file
from pipeline.operational.config import OperationalConfig
from pipeline.dwd_raw import (
    EXPECTED_LEADS,
    REGULAR_GRID_TOKEN,
    archived_day_dir,
    lead_from_filename,
    legacy_run_dir,
    locate_raw_run,
    variable_archive_path,
    variable_archive_filename,
    zip_regular_members,
)


DWD_MODEL_VARIABLES = (
    "u_10m",
    "v_10m",
    "aswdir_s",
    "aswdifd_s",
)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.values.append(str(value))


def raw_run_dir(root: Path, issue_date: date, run_hour: str) -> Path:
    """Return the legacy directory location (kept for compatibility)."""
    return legacy_run_dir(root, issue_date, run_hour)


def _lead_from_name(filename: str, issue_token: str, variable: str) -> int | None:
    return lead_from_filename(
        filename,
        issue_token,
        variable,
        grid_token=REGULAR_GRID_TOKEN,
    )


def _remote_files(
    config: OperationalConfig,
    issue_date: date,
    run_hour: str,
    variable: str,
) -> dict[int, str]:
    directory_url = f"{config.dwd_open_data_url}/{run_hour}/{variable}/"
    response = requests.get(directory_url, timeout=config.dwd_request_timeout_seconds)
    response.raise_for_status()
    parser = _Links()
    parser.feed(response.text)
    issue_token = f"{issue_date:%Y%m%d}{run_hour}"
    files: dict[int, str] = {}
    for href in parser.values:
        filename = href.rsplit("/", 1)[-1]
        lead = _lead_from_name(filename, issue_token, variable)
        if lead is not None:
            files[lead] = urljoin(directory_url, href)
    missing = sorted(EXPECTED_LEADS - set(files))
    if missing:
        raise RuntimeError(
            f"DWD {issue_token} {variable} is incomplete; missing leads {missing}."
        )
    return files


def _verify_bz2_grib_stream(compressed, label: str) -> None:
    first = b""
    try:
        with bz2.BZ2File(compressed, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                if not first:
                    first = block[:4]
    except (OSError, EOFError) as exc:
        raise ValueError(f"Invalid BZip2 data in {label}: {exc}") from exc
    if first != b"GRIB":
        raise ValueError(
            f"Decompressed file does not begin with a GRIB message: {label}"
        )


def verify_bz2_grib(path: Path) -> None:
    """Read the full BZip2 stream so CRC errors cannot pass unnoticed."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty DWD file: {path}")
    _verify_bz2_grib_stream(path, str(path))


def _download_one(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_file(
        url,
        destination,
        request_timeout=(15, timeout),
        max_attempts=1,
    )
    verify_bz2_grib(destination)


def verify_raw_run(
    run_dir: Path,
    issue_date: date,
    run_hour: str,
    variables: Iterable[str] = DWD_MODEL_VARIABLES,
) -> None:
    issue_token = f"{issue_date:%Y%m%d}{run_hour}"
    errors: list[str] = []
    for variable in variables:
        by_lead: dict[int, Path] = {}
        variable_dir = run_dir / variable
        for path in variable_dir.glob("*.grib2.bz2"):
            lead = _lead_from_name(path.name, issue_token, variable)
            if lead is not None:
                by_lead[lead] = path
        missing = sorted(EXPECTED_LEADS - set(by_lead))
        if missing:
            errors.append(f"{variable}: missing leads {missing}")
            continue
        for lead, path in sorted(by_lead.items()):
            try:
                verify_bz2_grib(path)
            except ValueError as exc:
                errors.append(f"{variable} lead {lead:03d}: {exc}")
    if errors:
        raise RuntimeError("Invalid DWD raw run:\n  - " + "\n  - ".join(errors))


def verify_variable_archive(
    archive_path: Path,
    issue_date: date,
    run_hour: str,
    variable: str,
) -> None:
    """Verify ZIP integrity and every regular-grid nested BZip2/GRIB member."""
    try:
        members = zip_regular_members(
            archive_path, issue_date, run_hour, variable
        )
        missing = sorted(EXPECTED_LEADS - set(members))
        if missing:
            raise RuntimeError(
                f"{variable}: archive missing regular-grid leads {missing}"
            )
        with ZipFile(archive_path) as archive:
            for lead, member_name in sorted(members.items()):
                with archive.open(member_name) as compressed:
                    _verify_bz2_grib_stream(
                        compressed,
                        f"{archive_path}!{member_name} (lead {lead:03d})",
                    )
    except (BadZipFile, OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid DWD ZIP archive {archive_path}: {exc}") from exc


def verify_archived_run(
    root: Path,
    issue_date: date,
    run_hour: str,
    variables: Iterable[str] = DWD_MODEL_VARIABLES,
    *,
    archive_dir: Path | None = None,
) -> None:
    source_dir = archive_dir or archived_day_dir(root, issue_date)
    errors: list[str] = []
    for variable in variables:
        archive_path = source_dir / variable_archive_filename(
            run_hour, variable
        )
        if not archive_path.is_file():
            errors.append(f"{variable}: missing archive {archive_path.name}")
            continue
        try:
            verify_variable_archive(archive_path, issue_date, run_hour, variable)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("Invalid archived DWD raw run:\n  - " + "\n  - ".join(errors))


def verify_available_run(
    root: Path,
    issue_date: date,
    run_hour: str,
    variables: Iterable[str] = DWD_MODEL_VARIABLES,
) -> Path:
    location = locate_raw_run(root, issue_date, run_hour)
    if location is None:
        raise RuntimeError(
            f"No DWD raw run found for {issue_date} {run_hour} below {root}."
        )
    if location.kind == "zip":
        verify_archived_run(
            root,
            issue_date,
            run_hour,
            variables,
            archive_dir=location.path,
        )
    else:
        verify_raw_run(location.path, issue_date, run_hour, variables)
    return location.path


def _download_variable_archive(
    config: OperationalConfig,
    issue_date: date,
    run_hour: str,
    variable: str,
    remote: dict[int, str],
) -> Path:
    archive_path = variable_archive_path(
        config.dwd_raw_archive, issue_date, run_hour, variable
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.part")
    temporary_archive.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"dwd_{variable}_"
        ) as temporary_directory:
            with ZipFile(temporary_archive, "w", compression=ZIP_STORED) as archive:
                for lead, url in sorted(remote.items()):
                    filename = url.rsplit("/", 1)[-1]
                    downloaded = Path(temporary_directory) / filename
                    print(f"DWD download {variable} lead={lead:03d}", flush=True)
                    _download_one(
                        url,
                        downloaded,
                        config.dwd_request_timeout_seconds,
                    )
                    archive.write(
                        downloaded,
                        arcname=f"icon-d2/{run_hour}/{variable}/{filename}",
                    )
                    downloaded.unlink()
        verify_variable_archive(
            temporary_archive, issue_date, run_hour, variable
        )
        os.replace(temporary_archive, archive_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
    return archive_path


def download_run(
    config: OperationalConfig,
    issue_date: date,
    *,
    force: bool = False,
) -> Path:
    """Download the configured run, retrying incomplete or corrupt data."""
    run_hour = config.dwd_operational_run_hour
    if not config.download_dwd:
        return verify_available_run(config.dwd_raw_archive, issue_date, run_hour)

    if not force:
        try:
            return verify_available_run(
                config.dwd_raw_archive, issue_date, run_hour
            )
        except RuntimeError:
            # Missing or invalid local input is repaired below by downloading
            # a complete canonical dwd_icon_archived_* ZIP set.
            pass

    last_error: Exception | None = None
    for attempt in range(1, config.dwd_download_attempts + 1):
        try:
            for variable in DWD_MODEL_VARIABLES:
                archive_path = variable_archive_path(
                    config.dwd_raw_archive, issue_date, run_hour, variable
                )
                valid = False
                if archive_path.exists() and not force:
                    try:
                        verify_variable_archive(
                            archive_path, issue_date, run_hour, variable
                        )
                        valid = True
                    except RuntimeError:
                        valid = False
                if not valid:
                    remote = _remote_files(
                        config, issue_date, run_hour, variable
                    )
                    _download_variable_archive(
                        config,
                        issue_date,
                        run_hour,
                        variable,
                        remote,
                    )
            verify_archived_run(config.dwd_raw_archive, issue_date, run_hour)
            manifest = {
                "format_version": 1,
                "issue_date": issue_date.isoformat(),
                "run_hour": run_hour,
                "variables": list(DWD_MODEL_VARIABLES),
                "leads": [min(EXPECTED_LEADS), max(EXPECTED_LEADS)],
                "grid": REGULAR_GRID_TOKEN,
            }
            archive_dir = archived_day_dir(config.dwd_raw_archive, issue_date)
            archive_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = archive_dir / "manifest.json"
            temporary_manifest = manifest_path.with_name(".manifest.json.part")
            temporary_manifest.unlink(missing_ok=True)
            try:
                temporary_manifest.write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_manifest, manifest_path)
            finally:
                temporary_manifest.unlink(missing_ok=True)
            return archive_dir
        except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
            last_error = exc
            if attempt == config.dwd_download_attempts:
                break
            wait_seconds = min(60, config.dwd_retry_seconds * (2 ** (attempt - 1)))
            print(
                f"DWD verification/download attempt {attempt} failed: {exc}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"DWD run {issue_date} {run_hour} failed after "
        f"{config.dwd_download_attempts} attempts: {last_error}"
    )


def cleanup_raw_run(
    root: Path,
    issue_date: date,
    run_hour: str,
    variables: Iterable[str] = DWD_MODEL_VARIABLES,
) -> list[Path]:
    """Remove only the verified run/variable inputs consumed by preprocessing."""
    location = locate_raw_run(root, issue_date, run_hour)
    if location is None:
        return []
    removed: list[Path] = []
    if location.kind == "zip":
        for variable in variables:
            path = location.path / variable_archive_filename(run_hour, variable)
            if path.is_file():
                path.unlink()
                removed.append(path)
        manifest = location.path / "manifest.json"
        if manifest.is_file():
            manifest.unlink()
            removed.append(manifest)
        try:
            location.path.rmdir()
        except OSError:
            pass
        return removed

    for variable in variables:
        variable_dir = location.path / variable
        if variable_dir.is_dir():
            shutil.rmtree(variable_dir)
            removed.append(variable_dir)
    for directory in (
        location.path,
        location.path.parent,
        location.path.parent.parent,
    ):
        try:
            directory.rmdir()
        except OSError:
            break
    return removed
