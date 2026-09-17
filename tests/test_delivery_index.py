from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.delivery_index import (
    delivery_clock_index,
    make_delivery_index,
    read_forecast_csv,
    validate_delivery_index,
)


class DeliveryIndexTests(unittest.TestCase):
    def test_spring_dst_day_has_96_unique_canonical_mtus(self):
        index = make_delivery_index([pd.Timestamp("2026-03-29", tz="Europe/Berlin")])

        self.assertEqual(index.names, ["delivery_date", "mtu"])
        self.assertEqual(len(index), 96)
        self.assertTrue(index.is_unique)
        self.assertEqual(index[0], (pd.Timestamp("2026-03-29"), 1))
        self.assertEqual(index[-1], (pd.Timestamp("2026-03-29"), 96))
        clock = delivery_clock_index(index)
        self.assertEqual(clock[8], pd.Timestamp("2026-03-29 02:00"))
        self.assertEqual(clock[-1], pd.Timestamp("2026-03-29 23:45"))

    def test_schema_v2_csv_round_trip(self):
        index = make_delivery_index(["2026-03-28", "2026-03-29"])
        frame = pd.DataFrame(
            {"y_pred": np.arange(len(index)), "y_true": np.arange(len(index))},
            index=index,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            loaded = read_forecast_csv(
                path,
                required_columns=("y_pred", "y_true"),
                require_complete_days=True,
            )

        pd.testing.assert_frame_equal(loaded, frame)

    def test_incomplete_day_is_rejected_when_required(self):
        index = make_delivery_index(["2026-03-29"]).delete(8)
        with self.assertRaisesRegex(ValueError, "MTUs 1 through 96"):
            validate_delivery_index(index, require_complete_days=True)

    def test_unsorted_persisted_keys_are_rejected(self):
        index = make_delivery_index(["2026-03-29"])
        frame = pd.DataFrame(
            {"y_pred": np.arange(96), "y_true": np.arange(96)}, index=index
        ).iloc[::-1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forecast.csv"
            frame.to_csv(path)
            with self.assertRaisesRegex(ValueError, "must be sorted"):
                read_forecast_csv(path)


if __name__ == "__main__":
    unittest.main()
