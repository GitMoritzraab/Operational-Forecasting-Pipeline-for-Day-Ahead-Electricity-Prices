from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation.run_benchmarks import (
    build_persistence_forecast,
    run_benchmarks,
)


class RecordingSQRA:
    def __init__(self, quantile=0.5, fit_intercept=False):
        self.quantile = quantile

    def fit(self, X, y):
        self.offset = float(np.nanquantile(y - X[:, 0], self.quantile))
        return self

    def predict(self, X):
        return X[:, 0] + self.offset


def synthetic_prices(start="2025-09-20", end="2025-12-02"):
    index = pd.date_range(start, f"{end} 23:45", freq="15min", tz="Europe/Berlin")
    local_day = (index.tz_localize(None).normalize() - pd.Timestamp(start)).days
    mtu = index.hour * 4 + index.minute // 15
    return pd.DataFrame(
        {"price_da": local_day * 1000.0 + mtu.astype(float)},
        index=index,
    )


class PersistenceBenchmarkTests(unittest.TestCase):
    def test_lags_use_same_mtu_on_calendar_delivery_day(self):
        prices = synthetic_prices(end="2025-10-10")
        days = pd.date_range("2025-10-08", "2025-10-09", freq="D")
        d1, _ = build_persistence_forecast(prices, days, 1)
        d7, _ = build_persistence_forecast(prices, days, 7)

        for day in days:
            for mtu in (1, 48, 96):
                key = (day, mtu)
                self.assertEqual(
                    d1.loc[key, "y_pred"], d1.loc[key, "y_true"] - 1000.0
                )
                self.assertEqual(
                    d7.loc[key, "y_pred"], d7.loc[key, "y_true"] - 7000.0
                )

    def test_runner_writes_four_compatible_result_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            cache_dir = temp / "entsoe"
            cache_dir.mkdir()
            prices = synthetic_prices()
            # Cache timestamps follow the repository's UTC serialization contract.
            prices.tz_convert("UTC").to_csv(cache_dir / "prices_da.csv")
            config = SimpleNamespace(
                entsoe_price_cache_dir=cache_dir,
                timezone="Europe/Berlin",
                point_forecast_start=date(2025, 10, 1),
                evaluation_start=date(2025, 12, 1),
                evaluation_end=date(2025, 12, 2),
                forecast_skip_dates=(),
                evaluation_skip_dates=(),
                sqra_train_days=60,
                sqra_mtu_specific=False,
                results_root=temp / "results",
            )

            with patch("pipeline.sqra.sqra_model.SQRA", RecordingSQRA):
                outputs = run_benchmarks(config)

            expected = {
                "persistence_d1",
                "persistence_d7",
                "sqra_persistence_d1",
                "sqra_persistence_d7",
            }
            self.assertEqual(set(outputs), expected)
            for name in expected:
                folder = config.results_root / "benchmarks" / name
                self.assertEqual(
                    {path.name for path in folder.iterdir()},
                    {"config.json", "forecast.csv", "metrics.csv", "runtime.csv"},
                )
                with (folder / "config.json").open(encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.assertEqual(metadata["forecast_schema_version"], 2)

            point = pd.read_csv(outputs["persistence_d1"] / "forecast.csv")
            point_metrics = pd.read_csv(
                outputs["persistence_d1"] / "metrics.csv"
            )
            probabilistic = pd.read_csv(
                outputs["sqra_persistence_d1"] / "forecast.csv"
            )
            self.assertEqual(
                list(point.columns),
                ["delivery_date", "mtu", "y_pred", "y_true"],
            )
            self.assertEqual(
                list(probabilistic.columns),
                [
                    "delivery_date",
                    "mtu",
                    "q0.100",
                    "q0.250",
                    "q0.500",
                    "q0.750",
                    "q0.900",
                    "y_true",
                ],
            )
            self.assertEqual(point_metrics.iloc[-1]["period"], "evaluation")
            self.assertEqual(int(point_metrics.iloc[-1]["n_obs"]), 2 * 96)


if __name__ == "__main__":
    unittest.main()
