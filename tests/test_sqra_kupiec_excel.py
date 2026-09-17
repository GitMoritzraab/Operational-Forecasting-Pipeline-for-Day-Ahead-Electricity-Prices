from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from evaluation.export_sqra_kupiec_excel import (
    MODEL_ROWS,
    TEST_COLUMNS,
    calculate_kupiec_mtu_counts,
)


class SqraKupiecExcelTests(unittest.TestCase):
    def test_all_models_intervals_and_significance_levels_are_exported(self):
        forecasts = {
            model_key: pd.DataFrame({"marker": [position]})
            for position, (model_key, _, _) in enumerate(MODEL_ROWS)
        }

        def fake_kupiec(frame, lower, upper, alpha, significance_level):
            self.assertAlmostEqual(alpha, upper - lower)
            return int(frame.iloc[0, 0] * 10 + significance_level * 100)

        with patch(
            "evaluation.export_sqra_kupiec_excel.mtu_kupiec_test",
            side_effect=fake_kupiec,
        ) as mocked:
            result = calculate_kupiec_mtu_counts(forecasts)

        self.assertEqual(mocked.call_count, len(MODEL_ROWS) * len(TEST_COLUMNS))
        self.assertEqual(
            result.columns.tolist(),
            [
                "Configuration",
                "Model",
                "50% PI - 1%",
                "50% PI - 5%",
                "80% PI - 1%",
                "80% PI - 5%",
            ],
        )
        self.assertEqual(
            result["Configuration"].tolist(),
            [configuration for _, configuration, _ in MODEL_ROWS],
        )


if __name__ == "__main__":
    unittest.main()
