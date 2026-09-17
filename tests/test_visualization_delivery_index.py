from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from visualization.generate_mae_tables import _forecast_metrics
from visualization.plot_prob_forecast_example import load_forecast_csv


class VisualizationDeliveryIndexTests(unittest.TestCase):
    def test_mae_metrics_include_normalized_spring_dst_day(self):
        index = make_delivery_index(
            pd.date_range("2026-03-28", "2026-03-30", freq="D")
        )
        realised = np.linspace(-10.0, 100.0, len(index))
        frame = pd.DataFrame(
            {"y_pred": realised + 2.0, "y_true": realised}, index=index
        )
        config = SimpleNamespace(
            evaluation_start=date(2026, 3, 28),
            evaluation_end=date(2026, 3, 30),
            evaluation_skip_dates=(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            mae, rmse = _forecast_metrics(path, config)
        self.assertAlmostEqual(mae, 2.0)
        self.assertAlmostEqual(rmse, 2.0)

    def test_probabilistic_plot_loader_builds_naive_clock_index(self):
        index = make_delivery_index(["2026-03-29"])
        frame = pd.DataFrame(index=index)
        for quantile, offset in zip(
            (0.10, 0.25, 0.50, 0.75, 0.90),
            (-4.0, -2.0, 0.0, 2.0, 4.0),
        ):
            frame[f"q{quantile:.3f}"] = np.arange(96, dtype=float) + offset
        frame["y_true"] = np.arange(96, dtype=float)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            loaded = load_forecast_csv(path)

        self.assertIsInstance(loaded.index, pd.DatetimeIndex)
        self.assertIsNone(loaded.index.tz)
        self.assertEqual(loaded.index[0], pd.Timestamp("2026-03-29 00:00"))
        self.assertEqual(loaded.index[8], pd.Timestamp("2026-03-29 02:00"))
        self.assertEqual(loaded.index[-1], pd.Timestamp("2026-03-29 23:45"))
        self.assertEqual(len(loaded), 96)


if __name__ == "__main__":
    unittest.main()
