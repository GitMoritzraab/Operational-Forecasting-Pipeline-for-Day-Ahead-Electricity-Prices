from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from visualization.plot_exaa_epex_correlation import (
    build_matched_price_panel,
    calculate_correlations,
    load_price_cache,
    plot_correlations,
)


class PriceCorrelationAnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_week_of_prices(root: Path):
        index = pd.date_range(
            "2025-12-01",
            "2025-12-07 23:45",
            freq="15min",
            tz="Europe/Berlin",
        ).rename("timestamp")
        epex_values = np.arange(len(index), dtype=float) - 100.0
        exaa_values = 2.0 * epex_values + 5.0
        epex_path = root / "prices_da.csv"
        exaa_path = root / "prices_exaa.csv"
        pd.DataFrame({"price_da": epex_values}, index=index).to_csv(epex_path)
        pd.DataFrame({"price_exaa": exaa_values}, index=index).to_csv(exaa_path)
        return epex_path, exaa_path

    def test_matching_and_all_five_correlation_groups(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            epex_path, exaa_path = self._write_week_of_prices(root)
            panel, diagnostics = build_matched_price_panel(
                epex_path,
                exaa_path,
                date(2025, 12, 1),
                date(2025, 12, 7),
                "Europe/Berlin",
            )
            correlations = calculate_correlations(panel)

        self.assertEqual(len(panel), 7 * 96)
        self.assertEqual(diagnostics.expected_mtus, 7 * 96)
        self.assertEqual(diagnostics.matched_mtus, 7 * 96)
        self.assertEqual(diagnostics.epex_missing_mtus, 0)
        self.assertEqual(diagnostics.exaa_missing_mtus, 0)
        self.assertEqual(
            correlations["group"].tolist(),
            [
                "All delivery days",
                "Tuesday–Saturday",
                "Sunday",
                "Monday",
                "Negative EPEX prices",
            ],
        )
        self.assertEqual(correlations["delivery_days"].tolist(), [7, 5, 1, 1, 2])
        self.assertEqual(
            correlations["matched_mtus"].tolist(),
            [7 * 96, 5 * 96, 96, 96, 100],
        )
        np.testing.assert_allclose(correlations["pearson"], 1.0)
        np.testing.assert_allclose(correlations["spearman"], 1.0)

    def test_duplicate_cache_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicates.csv"
            timestamp = pd.Timestamp("2025-12-01", tz="Europe/Berlin")
            pd.DataFrame(
                {"price_da": [1.0, 2.0]},
                index=pd.DatetimeIndex([timestamp, timestamp], name="timestamp"),
            ).to_csv(path)
            with self.assertRaisesRegex(ValueError, "Duplicate timestamps"):
                load_price_cache(
                    path,
                    "price_da",
                    "epex_price",
                    date(2025, 12, 1),
                    date(2025, 12, 1),
                    "Europe/Berlin",
                )

    def test_spring_dst_delivery_day_retains_92_physical_mtus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index = pd.date_range(
                "2026-03-29",
                "2026-03-29 23:45",
                freq="15min",
                tz="Europe/Berlin",
            ).rename("timestamp")
            values = np.arange(len(index), dtype=float)
            epex_path = root / "prices_da.csv"
            exaa_path = root / "prices_exaa.csv"
            pd.DataFrame({"price_da": values}, index=index).to_csv(epex_path)
            pd.DataFrame({"price_exaa": values + 1.0}, index=index).to_csv(exaa_path)
            panel, diagnostics = build_matched_price_panel(
                epex_path,
                exaa_path,
                date(2026, 3, 29),
                date(2026, 3, 29),
                "Europe/Berlin",
            )

        self.assertEqual(len(panel), 92)
        self.assertEqual(diagnostics.expected_mtus, 92)
        self.assertEqual(diagnostics.matched_mtus, 92)
        self.assertTrue((panel["weekday"] == 6).all())

    def test_one_sided_missing_price_is_reported_and_excluded_pairwise(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            epex_path, exaa_path = self._write_week_of_prices(root)
            epex = pd.read_csv(epex_path, index_col=0)
            epex.iloc[10, epex.columns.get_loc("price_da")] = np.nan
            epex.to_csv(epex_path)

            panel, diagnostics = build_matched_price_panel(
                epex_path,
                exaa_path,
                date(2025, 12, 1),
                date(2025, 12, 7),
                "Europe/Berlin",
            )
            correlations = calculate_correlations(panel)

        self.assertEqual(diagnostics.epex_missing_mtus, 1)
        self.assertEqual(diagnostics.exaa_missing_mtus, 0)
        self.assertEqual(diagnostics.matched_mtus, 7 * 96 - 1)
        self.assertEqual(correlations.iloc[0]["matched_mtus"], 7 * 96 - 1)

    def test_missing_required_price_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices.csv"
            index = pd.date_range(
                "2025-12-01", periods=2, freq="15min", tz="Europe/Berlin"
            ).rename("timestamp")
            pd.DataFrame({"wrong_column": [1.0, 2.0]}, index=index).to_csv(path)

            with self.assertRaisesRegex(ValueError, "Required column 'price_da'"):
                load_price_cache(
                    path,
                    "price_da",
                    "epex_price",
                    date(2025, 12, 1),
                    date(2025, 12, 1),
                    "Europe/Berlin",
                )

    def test_publication_plot_is_written_as_nonempty_pdf(self):
        correlations = pd.DataFrame(
            {
                "group": [
                    "All delivery days",
                    "Tuesday–Saturday",
                    "Sunday",
                    "Monday",
                    "Negative EPEX prices",
                ],
                "delivery_days": [7, 5, 1, 1, 2],
                "matched_mtus": [672, 480, 96, 96, 100],
                "pearson": [0.95, 0.96, 0.91, 0.93, 0.89],
                "spearman": [0.94, 0.95, 0.90, 0.92, 0.88],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "correlations.pdf"
            plot_correlations(
                correlations,
                output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
