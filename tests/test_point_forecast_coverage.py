import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from evaluation import check_point_forecast_coverage as coverage
from pipeline.delivery_index import make_delivery_index


def _write_forecast(path: Path, days, *, drop_keys=()):
    index = make_delivery_index(days)
    if drop_keys:
        index = index.difference(
            pd.MultiIndex.from_tuples(
                list(drop_keys), names=["delivery_date", "mtu"]
            )
        )
    frame = pd.DataFrame(
        {"y_pred": 1.0, "y_true": 1.0},
        index=index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


class PointForecastCoverageTests(unittest.TestCase):
    def test_complete_forecast_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "forecast.csv"
            days = [date(2026, 1, 1), date(2026, 1, 2)]
            _write_forecast(path, days)

            report = coverage.inspect_point_forecast(
                "model", path, days, days
            )

        self.assertTrue(report.ok)
        self.assertEqual(report.row_count, 192)

    def test_reports_missing_complete_day_and_evaluation_impact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "forecast.csv"
            _write_forecast(path, [date(2026, 1, 1)])

            report = coverage.inspect_point_forecast(
                "model",
                path,
                [date(2026, 1, 1), date(2026, 1, 2)],
                [date(2026, 1, 2)],
            )

        self.assertFalse(report.ok)
        self.assertEqual(report.missing_days, (date(2026, 1, 2),))
        self.assertEqual(
            report.missing_evaluation_days, (date(2026, 1, 2),)
        )

    def test_reports_partial_day_and_missing_mtus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "forecast.csv"
            day = date(2026, 1, 1)
            _write_forecast(
                path,
                [day],
                drop_keys=((pd.Timestamp(day), 3), (pd.Timestamp(day), 96)),
            )

            report = coverage.inspect_point_forecast(
                "model", path, [day], [day]
            )

        self.assertEqual(report.missing_days, ())
        self.assertEqual(report.partial_days, {day: (3, 96)})
        self.assertEqual(report.missing_evaluation_days, (day,))

    def test_registry_contains_all_28_point_models(self):
        paths = coverage.point_forecast_files(Path("results"))
        self.assertEqual(len(paths), 28)
        self.assertIn("exaa_naive", paths)
        self.assertIn("dwd_d56_c1_fundamental", paths)

    def test_configured_skip_dates_are_not_expected(self):
        config = SimpleNamespace(
            point_forecast_start=date(2026, 1, 1),
            evaluation_start=date(2026, 1, 2),
            evaluation_end=date(2026, 1, 3),
            forecast_skip_dates=(date(2026, 1, 2),),
            evaluation_skip_dates=(date(2026, 1, 2),),
        )

        point_days = coverage.expected_delivery_days(
            config.point_forecast_start,
            config.evaluation_end,
            config.forecast_skip_dates,
        )
        evaluation_days = coverage.expected_delivery_days(
            config.evaluation_start,
            config.evaluation_end,
            config.evaluation_skip_dates,
        )

        self.assertEqual(point_days, (date(2026, 1, 1), date(2026, 1, 3)))
        self.assertEqual(evaluation_days, (date(2026, 1, 3),))


if __name__ == "__main__":
    unittest.main()
