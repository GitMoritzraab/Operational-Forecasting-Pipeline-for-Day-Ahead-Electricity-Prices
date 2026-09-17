from __future__ import annotations

import bz2
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

from pipeline import dwd_raw
from pipeline.operational import dwd


ISSUE_DAY = date(2026, 9, 15)
RUN_HOUR = "06"
VARIABLE = "u_10m"


def _filename(lead: int, grid: str = "regular-lat-lon") -> str:
    return (
        f"icon-d2_germany_{grid}_single-level_"
        f"2026091506_{lead:03d}_2d_u_10m.grib2.bz2"
    )


def _make_archive(
    root: Path,
    include_native: bool = True,
    *,
    daily_container: bool = False,
) -> Path:
    if daily_container:
        path = dwd_raw.legacy_day_dir(root, ISSUE_DAY) / (
            dwd_raw.variable_archive_filename(RUN_HOUR, VARIABLE)
        )
    else:
        path = dwd_raw.variable_archive_path(root, ISSUE_DAY, RUN_HOUR, VARIABLE)
    path.parent.mkdir(parents=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for lead in range(49):
            archive.writestr(
                f"icon-d2/{RUN_HOUR}/{VARIABLE}/{_filename(lead)}",
                bz2.compress(b"GRIB" + bytes([lead])),
            )
            if include_native:
                archive.writestr(
                    f"icon-d2/{RUN_HOUR}/{VARIABLE}/"
                    f"{_filename(lead, 'icosahedral')}",
                    bz2.compress(b"GRIB-native" + bytes([lead])),
                )
    return path


class DwdRawArchiveTests(unittest.TestCase):
    def test_zip_reader_selects_only_regular_grid_members(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = _make_archive(root)
            members = dwd_raw.zip_regular_members(
                archive, ISSUE_DAY, RUN_HOUR, VARIABLE
            )

        self.assertEqual(set(members), set(range(49)))
        self.assertTrue(all("regular-lat-lon" in name for name in members.values()))

    def test_archived_run_is_discovered_and_materialized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_archive(root)
            self.assertEqual(dwd_raw.discover_issue_dates(root), [ISSUE_DAY])
            self.assertEqual(dwd_raw.available_run_hours(root, ISSUE_DAY), [RUN_HOUR])
            with dwd_raw.materialize_raw_run(
                root, ISSUE_DAY, RUN_HOUR, [VARIABLE]
            ) as run_dir:
                files = list((run_dir / VARIABLE).glob("*.grib2.bz2"))
                self.assertEqual(len(files), 49)
                self.assertTrue(all("regular-lat-lon" in path.name for path in files))

    def test_zip_set_inside_daily_folder_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = _make_archive(root, daily_container=True)

            location = dwd_raw.locate_raw_run(root, ISSUE_DAY, RUN_HOUR)
            self.assertIsNotNone(location)
            self.assertEqual(location.kind, "zip")
            self.assertEqual(location.path, archive.parent)
            self.assertEqual(dwd_raw.available_run_hours(root, ISSUE_DAY), [RUN_HOUR])
            with dwd_raw.materialize_raw_run(
                root, ISSUE_DAY, RUN_HOUR, [VARIABLE]
            ) as run_dir:
                self.assertEqual(
                    len(list((run_dir / VARIABLE).glob("*.grib2.bz2"))),
                    49,
                )
            self.assertEqual(
                dwd.verify_available_run(
                    root,
                    ISSUE_DAY,
                    RUN_HOUR,
                    variables=[VARIABLE],
                ),
                archive.parent,
            )

    def test_materialization_rejects_incomplete_archives(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = _make_archive(root, include_native=False)
            replacement = archive.with_suffix(".replacement")
            with ZipFile(archive) as source, ZipFile(
                replacement, "w", compression=ZIP_STORED
            ) as target:
                for info in source.infolist()[:-1]:
                    target.writestr(info, source.read(info.filename))
            replacement.replace(archive)

            with self.assertRaisesRegex(ValueError, "missing regular-grid leads"):
                with dwd_raw.materialize_raw_run(
                    root, ISSUE_DAY, RUN_HOUR, [VARIABLE]
                ):
                    pass

    def test_nested_zip_and_bzip_integrity_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = _make_archive(root)
            dwd.verify_variable_archive(
                archive, ISSUE_DAY, RUN_HOUR, VARIABLE
            )

    def test_operational_download_writes_atomic_variable_zip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = SimpleNamespace(
                dwd_raw_archive=root,
                dwd_request_timeout_seconds=30,
            )
            remote = {
                lead: f"https://example.test/{_filename(lead)}"
                for lead in range(49)
            }

            def fake_download(url, destination, _timeout):
                destination.write_bytes(bz2.compress(b"GRIB" + url.encode()))

            with patch.object(dwd, "_download_one", side_effect=fake_download):
                archive = dwd._download_variable_archive(
                    config,
                    ISSUE_DAY,
                    RUN_HOUR,
                    VARIABLE,
                    remote,
                )

            self.assertTrue(archive.is_file())
            self.assertFalse(archive.with_name(f".{archive.name}.part").exists())
            self.assertFalse(any(archive.parent.glob("*.grib2.bz2")))
            with ZipFile(archive) as zipped:
                self.assertEqual(len(zipped.infolist()), 49)
            dwd.verify_variable_archive(
                archive, ISSUE_DAY, RUN_HOUR, VARIABLE
            )

    def test_cleanup_removes_only_consumed_run_variable_archives(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = _make_archive(root, include_native=False, daily_container=True)
            unrelated = archive.parent / "icon-d2__09__u_10m.zip"
            unrelated.write_bytes(b"unrelated")
            (archive.parent / "manifest.json").write_text("{}", encoding="utf-8")

            removed = dwd.cleanup_raw_run(
                root,
                ISSUE_DAY,
                RUN_HOUR,
                variables=[VARIABLE],
            )

            self.assertIn(archive, removed)
            self.assertFalse(archive.exists())
            self.assertFalse((archive.parent / "manifest.json").exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(archive.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
