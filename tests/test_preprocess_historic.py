import tempfile
import unittest
from zipfile import ZipFile
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from preprocessing import preprocess_historic


class HistoricIconDateSelectionTests(unittest.TestCase):
    def test_parser_accepts_iso_icon_folder_date(self):
        args = preprocess_historic._parser().parse_args(
            ["--icon", "--date", "2026-06-12", "--clusters", "1,5"]
        )

        self.assertEqual(args.date, [date(2026, 6, 12)])
        self.assertEqual(args.clusters, (1, 5))

    def test_parser_accepts_multiple_icon_folder_dates(self):
        args = preprocess_historic._parser().parse_args(
            [
                "--date",
                "2026-06-12",
                "2026-06-18",
                "--clusters",
                "1",
            ]
        )

        self.assertEqual(
            args.date,
            [date(2026, 6, 12), date(2026, 6, 18)],
        )

    def test_parser_rejects_non_iso_date(self):
        with self.assertRaises(SystemExit):
            preprocess_historic._parser().parse_args(
                ["--date", "12.06.2026"]
            )

    def test_selected_date_is_forwarded_to_icon_aggregator(self):
        selected = date(2026, 6, 12)
        config = SimpleNamespace(evaluation_end=date(2026, 7, 31))

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_archive = Path(temp_dir)
            (raw_archive / "dwd_icon_daily_20260612" / "icon-d2" / "09").mkdir(
                parents=True
            )

            with (
                patch.object(
                    preprocess_historic,
                    "_required_archive",
                    return_value=raw_archive,
                ),
                patch.object(preprocess_historic, "_run_child") as run_child,
            ):
                preprocess_historic._preprocess_icon(
                    config,
                    (1, 5),
                    force=True,
                    dry_run=False,
                    folder_dates=(selected,),
                )

        self.assertEqual(run_child.call_count, 2)
        for call in run_child.call_args_list:
            overrides = call.args[2]
            self.assertEqual(overrides["DWD_ONLY_DAY"], "20260612")
            self.assertEqual(overrides["DWD_PREPROCESS_START_DATE"], "2026-06-12")
            self.assertEqual(overrides["DWD_SKIP_EXISTING_OUTPUT"], "false")

    def test_selected_date_requires_the_raw_run_folder(self):
        config = SimpleNamespace(evaluation_end=date(2026, 7, 31))
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                preprocess_historic,
                "_required_archive",
                return_value=Path(temp_dir),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "run 09"):
                    preprocess_historic._preprocess_icon(
                        config,
                        (1,),
                        force=True,
                        dry_run=True,
                        folder_dates=(date(2026, 6, 12),),
                    )

    def test_selected_date_accepts_new_zip_archive_layout(self):
        selected = date(2026, 6, 12)
        config = SimpleNamespace(evaluation_end=date(2026, 7, 31))
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_archive = Path(temp_dir)
            archived = raw_archive / "dwd_icon_archived_20260612"
            archived.mkdir()
            with ZipFile(archived / "icon-d2__09__u_10m.zip", "w"):
                pass
            with (
                patch.object(
                    preprocess_historic,
                    "_required_archive",
                    return_value=raw_archive,
                ),
                patch.object(preprocess_historic, "_run_child") as run_child,
            ):
                preprocess_historic._preprocess_icon(
                    config,
                    (1,),
                    force=False,
                    dry_run=False,
                    folder_dates=(selected,),
                )

        run_child.assert_called_once()

    def test_multiple_dates_create_one_task_per_date_and_cluster(self):
        selected = (date(2026, 6, 12), date(2026, 6, 18))
        config = SimpleNamespace(evaluation_end=date(2026, 7, 31))

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_archive = Path(temp_dir)
            for folder_date in selected:
                day = folder_date.strftime("%Y%m%d")
                (
                    raw_archive / f"dwd_icon_daily_{day}" / "icon-d2" / "09"
                ).mkdir(parents=True)

            with (
                patch.object(
                    preprocess_historic,
                    "_required_archive",
                    return_value=raw_archive,
                ),
                patch.object(preprocess_historic, "_run_child") as run_child,
            ):
                preprocess_historic._preprocess_icon(
                    config,
                    (1, 5),
                    force=True,
                    dry_run=False,
                    folder_dates=selected,
                )

        self.assertEqual(run_child.call_count, 4)
        only_days = [call.args[2]["DWD_ONLY_DAY"] for call in run_child.call_args_list]
        self.assertEqual(
            only_days,
            ["20260612", "20260612", "20260618", "20260618"],
        )


if __name__ == "__main__":
    unittest.main()
