from __future__ import annotations

import argparse
import errno
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URLS = {
    "icon-d2": "https://opendata.dwd.de/weather/nwp/icon-d2/grib/",
    "icon-d2-eps": "https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/",
}

VARIABLES_ICON_D2 = [
    "aswdir_s",
    "aswdifd_s",
    "u_10m",
    "v_10m",
    "vmax_10m",
    "t_2m",
    "td_2m",
    "p",
    "u",
    "h_snow",
    "tot_prec",
    "snow_gsp",
]

VARIABLES_ICON_D2_EPS = ["aswdir_s", "aswdifd_s", "u_10m", "v_10m"]
ONLY_RUN_HOURS = ["00", "03", "06", "09", "12", "15", "18", "21"]

P_LEVELS_TO_KEEP = {1, 2, 3, 4, 5}
REGULAR_U_LEVELS_TO_KEEP = {62, 63}

P_FILE_REGEX = re.compile(
    r"_model-level_\d{10}_\d{3}_(\d+)_p\.grib2(?:\.bz2)?$"
)
REGULAR_U_FILE_REGEX = re.compile(
    r"^icon-d2_germany_regular-lat-lon_model-level_\d{10}_\d{3}_(\d+)_u\.grib2(?:\.bz2)?$"
)

USER_AGENT = "dwd-icon-downloader/2.0"
REQUEST_TIMEOUT = (15, 120)
MAX_ATTEMPTS = 4
CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 1000
MIN_INITIAL_FREE_BYTES = 250 * 1024**3
ARCHIVE_FREE_MARGIN_BYTES = 10 * 1024**3

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


@dataclass
class DownloadStats:
    candidates: int = 0
    downloaded: int = 0
    skipped: int = 0
    pending: int = 0
    failed: int = 0


class FatalLocalError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def disk_free_bytes(path: Path) -> int:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free


def format_gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def fetch_text(url: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = 2 ** (attempt - 1)
            print(
                f"[WARN] Request failed (attempt {attempt}/{MAX_ATTEMPTS}): "
                f"{url}: {exc}. Retrying in {delay}s."
            )
            time.sleep(delay)

    raise RuntimeError(f"Request failed after {MAX_ATTEMPTS} attempts: {url}: {last_error}")


def filter_files_by_levels(
    files: list[str], file_regex: re.Pattern[str], levels_to_keep: set[int]
) -> list[str]:
    kept: list[str] = []
    for file_name in files:
        match = file_regex.search(file_name)
        if match and int(match.group(1)) in levels_to_keep:
            kept.append(file_name)
    return kept


def get_run_folders(base_url: str) -> list[str]:
    soup = BeautifulSoup(fetch_text(base_url), "html.parser")
    folders = sorted(
        {
            link["href"].strip("/")
            for link in soup.find_all("a", href=True)
            if link["href"].endswith("/")
            and link["href"].strip("/").isdigit()
            and link["href"].strip("/") in ONLY_RUN_HOURS
        }
    )
    if not folders:
        raise RuntimeError(f"No expected run-hour folders found at {base_url}")
    missing_folders = sorted(set(ONLY_RUN_HOURS) - set(folders))
    if missing_folders:
        raise RuntimeError(
            f"Missing expected run-hour folders at {base_url}: {missing_folders}"
        )
    return folders


def get_files_from_variable_folder(
    var_url: str,
    file_regex: re.Pattern[str] | None = None,
    levels_to_keep: set[int] | None = None,
) -> list[str]:
    soup = BeautifulSoup(fetch_text(var_url), "html.parser")
    files = sorted(
        {
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].endswith((".grib2", ".grib2.bz2"))
        }
    )

    if file_regex is not None and levels_to_keep is not None:
        files = filter_files_by_levels(files, file_regex, levels_to_keep)

    if not files:
        raise RuntimeError(f"No matching files found at {var_url}")
    return files


def is_completed_download(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if path.name.endswith(".bz2"):
        try:
            with path.open("rb") as handle:
                return handle.read(3) == b"BZh"
        except OSError:
            return False
    return True


def download_file(
    file_url: str,
    destination: Path,
    *,
    request_timeout=REQUEST_TIMEOUT,
    max_attempts: int = MAX_ATTEMPTS,
) -> bool:
    """Download atomically. Return True when downloaded, False when already complete."""
    if is_completed_download(destination):
        return False

    if destination.exists():
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    part_path.unlink(missing_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        bytes_written = 0
        try:
            with SESSION.get(
                file_url,
                stream=True,
                timeout=request_timeout,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                expected_size = int(content_length) if content_length else None

                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written <= 0:
                raise RuntimeError("server returned an empty file")
            if expected_size is not None and bytes_written != expected_size:
                raise RuntimeError(
                    f"size mismatch: expected {expected_size}, received {bytes_written}"
                )
            if destination.name.endswith(".bz2"):
                with part_path.open("rb") as handle:
                    if handle.read(3) != b"BZh":
                        raise RuntimeError("download does not have a BZip2 header")

            part_path.replace(destination)
            return True
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            part_path.unlink(missing_ok=True)
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                raise FatalLocalError(
                    f"Local disk is full while downloading {file_url}. "
                    "The partial file was removed."
                ) from exc
            if attempt == max_attempts:
                break
            delay = 2 ** (attempt - 1)
            print(
                f"[WARN] Download failed (attempt {attempt}/{max_attempts}): "
                f"{file_url}: {exc}. Retrying in {delay}s."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Download failed after {max_attempts} attempts: {file_url}: {last_error}"
    )


def archive_dataset(staging_dir: Path, outgoing_dir: Path) -> dict:
    variable_dirs = sorted(path for path in staging_dir.glob("*/*/*") if path.is_dir())
    if not variable_dirs:
        raise RuntimeError(f"No variable directories found in {staging_dir}")

    incomplete_files = list(staging_dir.rglob("*.part"))
    if incomplete_files:
        raise RuntimeError(
            f"Refusing to archive while {len(incomplete_files)} partial downloads remain"
        )

    source_bytes = sum(
        path.stat().st_size
        for path in staging_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".part")
    )
    free_bytes = disk_free_bytes(outgoing_dir)
    required_bytes = source_bytes + ARCHIVE_FREE_MARGIN_BYTES
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient free space to build archives: "
            f"available={format_gib(free_bytes)}, required={format_gib(required_bytes)}"
        )
    print(
        f"[INFO] Archive disk-space check: source={format_gib(source_bytes)} "
        f"available={format_gib(free_bytes)}"
    )

    build_dir = outgoing_dir.with_name(f"{outgoing_dir.name}.part")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    manifest: dict = {
        "version": 1,
        "created_at_utc": utc_now_iso(),
        "format": "ZIP_STORED",
        "description": (
            "One non-recompressing ZIP archive per model/run-hour/variable. "
            "Member paths are relative to the original staging root."
        ),
        "archives": [],
    }

    try:
        for index, variable_dir in enumerate(variable_dirs, start=1):
            relative_parts = variable_dir.relative_to(staging_dir).parts
            if len(relative_parts) != 3:
                raise RuntimeError(f"Unexpected variable directory: {variable_dir}")
            model_name, run_hour, variable = relative_parts
            files = sorted(
                path
                for path in variable_dir.iterdir()
                if path.is_file() and not path.name.endswith(".part")
            )
            if not files:
                raise RuntimeError(f"No completed files to archive in {variable_dir}")

            archive_name = f"{model_name}__{run_hour}__{variable}.zip"
            final_archive = build_dir / archive_name
            temp_archive = build_dir / f"{archive_name}.part"
            entries: list[dict] = []

            with zipfile.ZipFile(
                temp_archive,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                for source_file in files:
                    member_name = source_file.relative_to(staging_dir).as_posix()
                    archive.write(source_file, arcname=member_name)
                    info = archive.getinfo(member_name)
                    entries.append(
                        {
                            "path": member_name,
                            "size_bytes": info.file_size,
                            "crc32": f"{info.CRC:08x}",
                        }
                    )

            temp_archive.replace(final_archive)
            with zipfile.ZipFile(final_archive, mode="r") as verification_archive:
                if len(verification_archive.infolist()) != len(files):
                    raise RuntimeError(f"Archive member-count mismatch: {final_archive}")

            manifest["archives"].append(
                {
                    "filename": archive_name,
                    "model": model_name,
                    "run_hour": run_hour,
                    "variable": variable,
                    "member_count": len(entries),
                    "uncompressed_bytes": sum(entry["size_bytes"] for entry in entries),
                    "archive_bytes": final_archive.stat().st_size,
                    "entries": entries,
                }
            )
            print(
                f"[ARCHIVE] {index}/{len(variable_dirs)} {archive_name}: "
                f"{len(files)} members"
            )

        manifest["archive_count"] = len(manifest["archives"])
        manifest["member_count"] = sum(
            item["member_count"] for item in manifest["archives"]
        )
        manifest["source_bytes"] = sum(
            item["uncompressed_bytes"] for item in manifest["archives"]
        )
        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if outgoing_dir.exists():
            shutil.rmtree(outgoing_dir)
        build_dir.replace(outgoing_dir)
        return manifest
    except Exception:
        # Keep the raw staging data. The incomplete build is removed on the next run.
        raise


def load_completed_archive_set(outgoing_dir: Path) -> dict | None:
    manifest_path = outgoing_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archives = manifest["archives"]
        if not archives:
            return None
        for archive in archives:
            archive_path = outgoing_dir / archive["filename"]
            if (
                not archive_path.is_file()
                or archive_path.stat().st_size != archive["archive_bytes"]
            ):
                return None
        return manifest
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a complete DWD ICON snapshot and package it into inode-friendly "
            "stored ZIP archives."
        )
    )
    parser.add_argument(
        "--collection-date",
        default=datetime.now().strftime("%Y%m%d"),
        help="Collection date used for isolated staging, in YYYYMMDD format.",
    )
    parser.add_argument("--staging-root", default="staging")
    parser.add_argument("--outgoing-root", default="outgoing")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and count candidates without downloading or creating archives.",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}", args.collection_date):
        parser.error("--collection-date must use YYYYMMDD format")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    staging_dir = Path(args.staging_root) / args.collection_date
    outgoing_dir = Path(args.outgoing_root) / args.collection_date
    stats = DownloadStats()
    errors: list[str] = []
    processed = 0

    if not args.dry_run:
        free_bytes = disk_free_bytes(staging_dir)
        print(f"[INFO] Initial free disk space: {format_gib(free_bytes)}")
        if free_bytes < MIN_INITIAL_FREE_BYTES:
            print(
                "[ERROR] Refusing to start: less than "
                f"{format_gib(MIN_INITIAL_FREE_BYTES)} is available for raw files "
                "and archive creation."
            )
            return 1

    for model_name, base_url in BASE_URLS.items():
        variables = (
            VARIABLES_ICON_D2
            if model_name == "icon-d2"
            else VARIABLES_ICON_D2_EPS
        )
        try:
            run_folders = get_run_folders(base_url)
        except RuntimeError as exc:
            errors.append(str(exc))
            print(f"[ERROR] {exc}")
            continue

        print(f"[INFO] {model_name}: run folders={run_folders}")
        for run_hour in run_folders:
            for variable in variables:
                var_url = urljoin(base_url, f"{run_hour}/{variable}/")
                file_regex = None
                levels_to_keep = None
                if model_name == "icon-d2" and variable == "p":
                    file_regex = P_FILE_REGEX
                    levels_to_keep = P_LEVELS_TO_KEEP
                elif model_name == "icon-d2" and variable == "u":
                    file_regex = REGULAR_U_FILE_REGEX
                    levels_to_keep = REGULAR_U_LEVELS_TO_KEEP

                try:
                    files = get_files_from_variable_folder(
                        var_url,
                        file_regex=file_regex,
                        levels_to_keep=levels_to_keep,
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
                    print(f"[ERROR] {exc}")
                    continue

                before_downloaded = stats.downloaded
                before_skipped = stats.skipped
                before_pending = stats.pending
                before_failed = stats.failed
                stats.candidates += len(files)

                for filename in files:
                    destination = staging_dir / model_name / run_hour / variable / filename
                    file_url = urljoin(var_url, filename)
                    processed += 1
                    if args.dry_run:
                        if is_completed_download(destination):
                            stats.skipped += 1
                        else:
                            stats.pending += 1
                        continue

                    try:
                        if download_file(file_url, destination):
                            stats.downloaded += 1
                        else:
                            stats.skipped += 1
                    except FatalLocalError as exc:
                        print(f"[ERROR] {exc}")
                        print("[ERROR] Aborting immediately; completed files are retained.")
                        return 1
                    except RuntimeError as exc:
                        stats.failed += 1
                        errors.append(str(exc))
                        print(f"[ERROR] {exc}")

                    if processed % PROGRESS_INTERVAL == 0:
                        print(
                            f"[PROGRESS] processed={processed} downloaded={stats.downloaded} "
                            f"skipped={stats.skipped} failed={stats.failed}"
                        )

                print(
                    f"[INFO] {model_name}/{run_hour}/{variable}: candidates={len(files)} "
                    f"downloaded={stats.downloaded - before_downloaded} "
                    f"skipped={stats.skipped - before_skipped} "
                    f"pending={stats.pending - before_pending} "
                    f"failed={stats.failed - before_failed}"
                )

    print(
        f"[SUMMARY] candidates={stats.candidates} downloaded={stats.downloaded} "
        f"skipped={stats.skipped} pending={stats.pending} failed={stats.failed} "
        f"listing_errors={len(errors) - stats.failed}"
    )

    if errors:
        print(
            f"[ERROR] Dataset is incomplete ({len(errors)} errors). "
            "Raw completed files are retained for a retry."
        )
        return 1
    if args.dry_run:
        print("[INFO] Dry run completed; no files were written.")
        return 0
    if stats.candidates == 0:
        print("[ERROR] No candidate files were discovered.")
        return 1

    if stats.downloaded == 0:
        existing_manifest = load_completed_archive_set(outgoing_dir)
        if existing_manifest is not None:
            print(
                f"[DONE] Reusing {existing_manifest['archive_count']} verified archives "
                f"in {outgoing_dir}."
            )
            return 0

    try:
        manifest = archive_dataset(staging_dir, outgoing_dir)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"[ERROR] Archive creation failed: {exc}")
        return 1

    print(
        f"[DONE] Created {manifest['archive_count']} archives containing "
        f"{manifest['member_count']} files in {outgoing_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
