from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.export_sqra_evaluation_metrics import (
    QUANTILE_COLUMNS,
    calculate_sqra_evaluation_metrics,
)
from evaluation.evaluation_core import QUANTILE_MODELS
from pipeline.delivery_index import make_delivery_index


class SqraMetricsExcelTests(unittest.TestCase):
    def test_all_six_metrics_use_the_median_and_repository_aps(self):
        index = make_delivery_index(["2025-12-01", "2025-12-02"])
        realised = np.linspace(-20.0, 100.0, len(index))
        offsets = (-4.0, -2.0, 1.0, 3.0, 5.0)
        forecasts = {}
        for model_position, model_name in enumerate(QUANTILE_MODELS):
            shift = float(model_position)
            frame = pd.DataFrame({"y_true": realised}, index=index)
            for column, offset in zip(QUANTILE_COLUMNS, offsets):
                frame[column] = realised + offset + shift
            forecasts[model_name] = frame

        metrics = calculate_sqra_evaluation_metrics(forecasts)

        self.assertEqual(len(metrics), 6)
        self.assertEqual(
            metrics["Configuration"].tolist(),
            [
                "dwd_fundamental",
                "era5_fundamental",
                "exaa_naive",
                "dwd_exaa_enriched",
                "era5_exaa_enriched",
                "exaa_only",
            ],
        )
        np.testing.assert_allclose(metrics["MAE"], [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(metrics["RMSE"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(np.isfinite(metrics["APS"]).all())
        self.assertEqual(metrics["Observations"].tolist(), [192] * 6)
        self.assertEqual(metrics["Delivery days"].tolist(), [2] * 6)


if __name__ == "__main__":
    unittest.main()
