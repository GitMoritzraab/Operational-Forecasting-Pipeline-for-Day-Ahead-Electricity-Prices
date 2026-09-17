from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.plot_monthly_point_mae import (
    ANALYSIS_END,
    ANALYSIS_START,
    MODEL_SPECIFICATIONS,
    calculate_monthly_mae,
    calculate_monthly_negative_price_counts,
    load_epex_prices,
    load_evaluation_forecast,
    plot_monthly_mae,
    plot_monthly_negative_price_counts,
    representative_forecast_files,
)
from pipeline.delivery_index import make_delivery_index


def _representative_frames():
    days = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D")
    index = make_delivery_index(days)
    realised = np.linspace(-50.0, 200.0, len(index))
    return {
        specification.run_name: pd.DataFrame(
            {
                "y_pred": realised + offset,
                "y_true": realised,
            },
            index=index,
        )
        for offset, specification in enumerate(MODEL_SPECIFICATIONS, start=1)
    }


class MonthlyPointMaeTests(unittest.TestCase):
    def test_registry_uses_requested_four_point_forecasts(self):
        paths = representative_forecast_files(Path("results"))
        self.assertEqual(
            list(paths),
            [
                "dwd_d56_c5_fundamental",
                "era5_d364_c5_fundamental",
                "era5_d364_c1_exaa",
                "exaa_only_d364",
            ],
        )

    def test_monthly_mae_is_pooled_over_all_monthly_mtus(self):
        monthly = calculate_monthly_mae(_representative_frames())

        self.assertEqual(len(monthly), 4 * 8)
        self.assertEqual(
            monthly["month"].drop_duplicates().tolist(),
            [
                "2025-12",
                "2026-01",
                "2026-02",
                "2026-03",
                "2026-04",
                "2026-05",
                "2026-06",
                "2026-07",
            ],
        )
        for offset, specification in enumerate(MODEL_SPECIFICATIONS, start=1):
            values = monthly.loc[
                monthly["model"] == specification.run_name, "mae"
            ]
            np.testing.assert_allclose(values, float(offset))

    def test_loader_respects_configured_skip_days_and_requires_coverage(self):
        days = pd.date_range("2025-12-01", "2025-12-03", freq="D")
        index = make_delivery_index(days)
        frame = pd.DataFrame(
            {"y_pred": 2.0, "y_true": 1.0},
            index=index,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            loaded = load_evaluation_forecast(
                path,
                date(2025, 12, 1),
                date(2025, 12, 3),
                (date(2025, 12, 2),),
            )

        self.assertEqual(len(loaded), 2 * 96)
        self.assertNotIn(
            pd.Timestamp("2025-12-02"),
            loaded.index.get_level_values("delivery_date"),
        )

    def test_loader_rejects_a_missing_delivery_day(self):
        index = make_delivery_index(["2025-12-01", "2025-12-03"])
        frame = pd.DataFrame(
            {"y_pred": 2.0, "y_true": 1.0},
            index=index,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            with self.assertRaisesRegex(ValueError, "expected days"):
                load_evaluation_forecast(
                    path,
                    date(2025, 12, 1),
                    date(2025, 12, 3),
                    (),
                )

    def test_publication_plot_is_written(self):
        monthly = calculate_monthly_mae(_representative_frames())
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "monthly.pdf"
            plot_monthly_mae(monthly, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_negative_price_count_is_strictly_below_zero(self):
        prices = pd.DataFrame(
            {
                "delivery_date": [
                    date(2025, 12, 1),
                    date(2025, 12, 1),
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                ],
                "epex_price": [-1.0, 0.0, -2.0, -3.0],
            }
        )
        counts = calculate_monthly_negative_price_counts(
            prices,
            date(2025, 12, 1),
            date(2026, 1, 31),
        )

        self.assertEqual(counts["negative_price_mtus"].tolist(), [1, 2])
        self.assertEqual(counts["total_mtus"].tolist(), [2, 2])

    def test_epex_loader_preserves_physical_spring_dst_grid(self):
        local = pd.date_range(
            "2026-03-29 00:00",
            "2026-03-29 23:45",
            freq="15min",
            tz="Europe/Berlin",
        )
        frame = pd.DataFrame(
            {"price_da": np.arange(len(local), dtype=float)},
            index=local.tz_convert("UTC"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices_da.csv"
            frame.to_csv(path)
            loaded = load_epex_prices(
                path,
                date(2026, 3, 29),
                date(2026, 3, 29),
                "Europe/Berlin",
            )

        self.assertEqual(len(loaded), 92)

    def test_negative_price_bar_plot_is_written(self):
        counts = pd.DataFrame(
            {
                "month": [
                    "2025-12",
                    "2026-01",
                    "2026-02",
                    "2026-03",
                    "2026-04",
                    "2026-05",
                    "2026-06",
                    "2026-07",
                ],
                "negative_price_mtus": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "negative.pdf"
            plot_monthly_negative_price_counts(counts, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
