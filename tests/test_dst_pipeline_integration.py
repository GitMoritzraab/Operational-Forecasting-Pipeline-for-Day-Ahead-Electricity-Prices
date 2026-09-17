from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation.evaluation_core import check_index
from pipeline.delivery_index import validate_delivery_index
from pipeline.lear.lear_model import build_exaa_naive_forecast
from pipeline.sqra.sqra_model import (
    DEFAULT_QUANTILES,
    build_sqra_panel,
    generate_sqra_forecast,
)


class RecordingSqra:
    fitted_targets = []

    def __init__(self, quantile, fit_intercept=True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        del X
        values = np.asarray(y, dtype=float)
        self.fitted_targets.append(values.copy())
        self.value = float(np.quantile(values, self.quantile))
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class DstPipelineIntegrationTests(unittest.TestCase):
    def test_physical_spring_day_flows_as_96_mtus_through_sqra_and_evaluation(self):
        timezone = "Europe/Berlin"
        physical_index = pd.date_range(
            pd.Timestamp("2026-01-28", tz=timezone),
            pd.Timestamp("2026-03-30", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        local_dates = pd.DatetimeIndex(physical_index.normalize())
        day_number = pd.factorize(local_dates)[0]
        mtu = physical_index.hour * 4 + physical_index.minute // 15
        actual = day_number * 100.0 + mtu
        epex = pd.DataFrame({"price_da": actual}, index=physical_index)
        exaa = pd.DataFrame({"price_exaa": actual + 2.0}, index=physical_index)

        point = build_exaa_naive_forecast(
            epex,
            exaa,
            pd.Timestamp("2026-01-28", tz=timezone),
            pd.Timestamp("2026-03-29", tz=timezone),
        )
        validate_delivery_index(point.index, require_complete_days=True)
        self.assertEqual(len(point), 61 * 96)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            point.to_csv(first)
            alternative = point.copy()
            alternative["y_pred"] += 1.0
            alternative.to_csv(second)
            panel, feature_columns = build_sqra_panel(
                [first, second], timezone
            )

        target_day = pd.Timestamp("2026-03-29")
        RecordingSqra.fitted_targets.clear()
        with patch("pipeline.sqra.sqra_model.SQRA", RecordingSqra):
            forecast, _ = generate_sqra_forecast(
                panel,
                [target_day],
                train_days=60,
                quantiles=DEFAULT_QUANTILES,
                feature_cols=feature_columns,
                mtu_specific=False,
            )

        validate_delivery_index(
            forecast.index,
            require_complete_days=True,
            expected_days=[target_day],
        )
        self.assertEqual(len(forecast), 96)
        self.assertEqual(len(RecordingSqra.fitted_targets), 5)
        self.assertTrue(
            all(len(values) == 60 * 96 for values in RecordingSqra.fitted_targets)
        )
        check_index(
            forecast,
            "DST integration",
            target_day,
            target_day,
            "15min",
            96,
            set(),
        )


if __name__ == "__main__":
    unittest.main()
