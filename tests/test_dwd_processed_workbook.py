from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.dwd_processed import (
    COMPLETE_FILENAME,
    PARQUET_FILENAME,
    parquet_is_complete,
    read_processed_output,
    read_completed_legacy_outputs,
    remove_superseded_outputs,
    write_processed_parquet,
)
from pipeline.dwd_history import (
    FORECAST_FIELDS,
    HISTORY_MANIFEST,
    consolidated_delivery_available,
    delivery_slice,
    migrate_daily_outputs,
)
from pipeline.lear.lear_model import load_dwd
from pipeline.lear.lear_model import build_dwd_features


def _metadata(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "field_name": name,
            "rows": len(frame),
            "columns": len(frame.columns),
        }
        for name, frame in frames.items()
    }


def _weather_frames() -> dict[str, pd.DataFrame]:
    hourly_index = pd.date_range("2026-01-01 23:00", periods=24, freq="h")
    solar_index = pd.date_range("2026-01-01 23:15", periods=96, freq="15min")
    return {
        "u10": pd.DataFrame({"cluster_0": np.arange(24)}, index=hourly_index),
        "v10": pd.DataFrame({"cluster_0": np.arange(24) + 1}, index=hourly_index),
        "ASWDIR_S": pd.DataFrame(
            {"cluster_0": np.arange(96)}, index=solar_index
        ),
        "ASWDIFD_S": pd.DataFrame(
            {"cluster_0": np.arange(96) + 1}, index=solar_index
        ),
    }


class DwdProcessedParquetTests(unittest.TestCase):
    def test_delivery_slice_collapses_duplicate_valid_times_by_mean(self):
        timestamps = pd.date_range(
            "2026-09-14 22:00", periods=24, freq="h", tz="UTC"
        )
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cluster_0": np.arange(24, dtype=float),
            }
        )
        duplicate = frame.iloc[[1]].copy()
        duplicate["cluster_0"] = 5.0
        frame = pd.concat([frame, duplicate], ignore_index=True)

        normalized = delivery_slice(
            "u10",
            frame,
            issue_date=date(2026, 9, 14),
            run_hour="06",
            delivery_date=date(2026, 9, 15),
            timezone="Europe/Berlin",
        )

        self.assertEqual(len(normalized), 24)
        self.assertFalse(normalized["timestamp"].duplicated().any())
        averaged = normalized.loc[
            normalized["timestamp"] == timestamps[1], "cluster_0"
        ].item()
        self.assertEqual(averaged, 3.0)

    def test_one_parquet_and_one_marker_round_trip_all_variables(self):
        frames = _weather_frames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_processed_parquet(
                output,
                frames,
                variable_metadata=_metadata(frames),
                issue_date="20260101",
                run_hour="09",
            )
            files = {path.name for path in output.iterdir()}
            loaded = read_processed_output(output)

        self.assertEqual(files, {PARQUET_FILENAME, COMPLETE_FILENAME})
        self.assertIsNotNone(loaded)
        self.assertEqual(list(loaded), list(frames))
        for variable, expected in frames.items():
            expected_table = expected.rename_axis("timestamp").reset_index()
            pd.testing.assert_frame_equal(
                loaded[variable], expected_table, check_freq=False
            )

    def test_all_nan_cluster_column_is_preserved(self):
        frames = _weather_frames()
        for frame in frames.values():
            frame["cluster_1"] = np.nan
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_processed_parquet(
                output,
                frames,
                variable_metadata=_metadata(frames),
                issue_date="20260101",
                run_hour="09",
            )
            loaded = read_processed_output(output)

        self.assertIsNotNone(loaded)
        self.assertEqual(
            list(loaded["u10"].columns),
            ["timestamp", "cluster_0", "cluster_1"],
        )
        self.assertTrue(loaded["u10"]["cluster_1"].isna().all())

    def test_lear_loader_reads_new_parquet_format(self):
        frames = _weather_frames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            icon_dir = Path(temporary_directory)
            output = icon_dir / "dwd_icon_daily_20260101_09"
            write_processed_parquet(
                output,
                frames,
                variable_metadata=_metadata(frames),
                issue_date="20260101",
                run_hour="09",
            )
            hourly, quarter_hourly = load_dwd(
                icon_dir,
                start_folder_date=date(2026, 1, 1),
                end_folder_date=date(2026, 1, 1),
                required_run="09",
            )

        self.assertEqual(len(hourly), 24)
        self.assertEqual(len(quarter_hourly), 96)
        self.assertIn("u10_cluster_0", hourly.columns)
        self.assertIn("v10_cluster_0", hourly.columns)
        self.assertIn("ASWDIR_cluster_0", quarter_hourly.columns)
        self.assertIn("ASWDIFD_cluster_0", quarter_hourly.columns)
        features = build_dwd_features(hourly, quarter_hourly)
        self.assertEqual(features.shape[0], 1)
        self.assertTrue(any(column.startswith("sw_dir_cluster_0") for column in features))

    def test_daily_outputs_migrate_to_four_consolidated_variable_histories(self):
        frames = _weather_frames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            icon_dir = Path(temporary_directory)
            daily = icon_dir / "dwd_icon_daily_20260101_09"
            write_processed_parquet(
                daily,
                frames,
                variable_metadata=_metadata(frames),
                issue_date="20260101",
                run_hour="09",
            )

            migrated = migrate_daily_outputs(
                icon_dir,
                timezone="Europe/Berlin",
            )
            hourly, quarter_hourly = load_dwd(
                icon_dir,
                start_folder_date=date(2026, 1, 1),
                end_folder_date=date(2026, 1, 1),
                required_run="09",
            )

            self.assertEqual(migrated, 1)
            self.assertEqual(
                {path.name for path in icon_dir.glob("*.parquet")},
                {f"{field}.parquet" for field in FORECAST_FIELDS},
            )
            self.assertTrue((icon_dir / HISTORY_MANIFEST).is_file())
            self.assertTrue(
                consolidated_delivery_available(
                    icon_dir,
                    delivery_date=date(2026, 1, 2),
                    run_hour="09",
                    timezone="Europe/Berlin",
                )
            )

        self.assertEqual(len(hourly), 24)
        self.assertEqual(len(quarter_hourly), 96)

    def test_lear_loader_still_reads_historical_csv_format(self):
        frames = _weather_frames()
        filenames = {
            "u10": "u10_m_s-1_2026010109_raw.csv",
            "v10": "v10_m_s-1_2026010109_raw.csv",
            "ASWDIR_S": "ASWDIR_S_W_m-2_2026010109_instantaneous.csv",
            "ASWDIFD_S": "ASWDIFD_S_W_m-2_2026010109_instantaneous.csv",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            icon_dir = Path(temporary_directory)
            output = icon_dir / "dwd_icon_daily_20260101_09"
            output.mkdir()
            for name, frame in frames.items():
                frame.rename_axis("timestamp").to_csv(output / filenames[name])
            hourly, quarter_hourly = load_dwd(
                icon_dir,
                start_folder_date=date(2026, 1, 1),
                end_folder_date=date(2026, 1, 1),
                required_run="09",
            )

        self.assertEqual(len(hourly), 24)
        self.assertEqual(len(quarter_hourly), 96)

    def test_cleanup_removes_only_legacy_outputs(self):
        frames = _weather_frames()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_processed_parquet(
                output,
                frames,
                variable_metadata=_metadata(frames),
                issue_date="20260101",
                run_hour="09",
            )
            (output / "u10_old.csv").write_text("old", encoding="utf-8")
            (output / ".u_10m.complete").write_text("old", encoding="utf-8")
            remove_superseded_outputs(output)

            self.assertTrue(parquet_is_complete(output))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {PARQUET_FILENAME, COMPLETE_FILENAME},
            )

    def test_completed_legacy_csvs_can_be_migrated_without_grib(self):
        frame = pd.DataFrame(
            {"cluster_0": [1.0, 2.0]},
            index=pd.date_range("2026-01-01 09:00", periods=2, freq="h"),
        )
        frame.index.name = "timestamp"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            csv_path = output / "u10_m_s-1_2026010109_raw.csv"
            with csv_path.open("w", encoding="utf-8") as handle:
                handle.write("# Folder: u_10m\n# Variable: u10\n")
                frame.to_csv(handle)
            (output / ".u_10m.complete").write_text(
                csv_path.name + "\n", encoding="utf-8"
            )

            migrated = read_completed_legacy_outputs(output, ["u_10m"])

        self.assertIsNotNone(migrated)
        migrated_frames, metadata = migrated
        pd.testing.assert_frame_equal(
            migrated_frames["u10"], frame, check_freq=False
        )
        self.assertEqual(metadata["u10"]["former_csv_name"], csv_path.name)


if __name__ == "__main__":
    unittest.main()
