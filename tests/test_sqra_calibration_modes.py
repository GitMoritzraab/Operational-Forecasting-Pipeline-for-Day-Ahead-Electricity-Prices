import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from remodels.qra import SQRA as RealSQRA

from experiment_config import env_bool, load_experiment_config
from experiment_manifest import SQRA_RUNS
from pipeline.delivery_index import (
    FORECAST_SCHEMA_VERSION,
    INDEX_COLUMNS,
    make_delivery_index,
    validate_delivery_index,
)
from pipeline.sqra import sqra_model
from pipeline.sqra.run_sqra import run_sqra_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


class RecordingSQRA:
    fit_records = []

    def __init__(self, quantile, fit_intercept=True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.fit_records.append((self.quantile, X.copy(), y.copy()))
        return self

    def predict(self, X):
        return X[:, 0] + self.quantile


def synthetic_panel():
    days = pd.date_range("2025-01-01", periods=61, freq="D")
    index = make_delivery_index(days)
    values = np.arange(len(index), dtype=float)
    panel = pd.DataFrame(
        {
            "prediction_p1": values,
            "y_true": values + 0.5,
        },
        index=index,
    )
    forecast_day = days[60]
    panel.iloc[0, panel.columns.get_loc("prediction_p1")] = np.nan
    panel.iloc[1, panel.columns.get_loc("y_true")] = np.nan
    target_or_later = (
        panel.index.get_level_values("delivery_date") >= forecast_day
    )
    panel.loc[target_or_later, "prediction_p1"] += 1_000_000
    panel.loc[(forecast_day, 1), "prediction_p1"] = np.nan
    return panel, forecast_day


class SqraConfigurationTests(unittest.TestCase):
    def test_false_is_default_and_true_restores_mtu_specific_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_bool("SQRA_MTU_SPECIFIC", False))
            self.assertFalse(load_experiment_config(REPO_ROOT).sqra_mtu_specific)
        with patch.dict(os.environ, {"SQRA_MTU_SPECIFIC": "true"}):
            self.assertTrue(load_experiment_config(REPO_ROOT).sqra_mtu_specific)

    def test_manifest_uses_revised_members_without_renaming_outputs(self):
        expected = {
            "era5_fundamental": (
                "lear_op_results/era5/d364/c1/fundamental/forecast.csv",
                "lear_op_results/era5/d364/c5/fundamental/forecast.csv",
                "lear_op_results/era5/d364/c25/fundamental/forecast.csv",
            ),
            "dwd_fundamental": (
                "lear_op_results/dwd/d56/c1/fundamental/forecast.csv",
                "lear_op_results/dwd/d56/c5/fundamental/forecast.csv",
                "lear_op_results/dwd/d56/c25/fundamental/forecast.csv",
            ),
            "era5_exaa_enriched": (
                "lear_op_results/era5/d364/c1/exaa/forecast.csv",
                "lear_op_results/era5/d364/c5/exaa/forecast.csv",
                "lear_op_results/era5/d364/c25/exaa/forecast.csv",
            ),
            "dwd_exaa_enriched": (
                "lear_op_results/dwd/d56/c1/exaa/forecast.csv",
                "lear_op_results/dwd/d56/c5/exaa/forecast.csv",
                "lear_op_results/dwd/d56/c25/exaa/forecast.csv",
            ),
            "exaa_only": (
                "lear_op_results/exaa_only/d56/forecast.csv",
                "lear_op_results/exaa_only/d112/forecast.csv",
                "lear_op_results/exaa_only/d364/forecast.csv",
            ),
            "exaa_naive": ("lear_op_results/exaa_naive/forecast.csv",),
        }
        self.assertEqual(set(SQRA_RUNS), set(expected))
        for name, member_paths in expected.items():
            self.assertEqual(SQRA_RUNS[name].member_paths, member_paths)
            self.assertEqual(
                SQRA_RUNS[name].output_dir(Path("results")),
                Path("results") / "sqra_results" / name,
            )


class SqraRollingModeTests(unittest.TestCase):
    def setUp(self):
        RecordingSQRA.fit_records.clear()
        self.panel, self.forecast_day = synthetic_panel()
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    def assert_output_contract(self, forecast, runtime):
        self.assertEqual(
            list(forecast.columns),
            ["q0.100", "q0.250", "q0.500", "q0.750", "q0.900", "y_true"],
        )
        self.assertEqual(len(forecast), 96)
        self.assertTrue(forecast.index.is_monotonic_increasing)
        self.assertEqual(tuple(forecast.index.names), INDEX_COLUMNS)
        validate_delivery_index(forecast.index, require_complete_days=True)
        self.assertEqual(len(runtime), 1)
        first_mtu = forecast.loc[(self.forecast_day, 1)]
        self.assertTrue(first_mtu.filter(like="q").isna().all())

    def test_pooled_mode_fits_five_models_without_target_day_leakage(self):
        with patch.object(sqra_model, "SQRA", RecordingSQRA):
            forecast, runtime = sqra_model.rolling_sqra_forecast_pooled(
                self.panel,
                [self.forecast_day],
                60,
                self.quantiles,
                ["prediction_p1"],
            )
        self.assertEqual(len(RecordingSQRA.fit_records), 5)
        self.assertTrue(
            all(len(X) == 60 * 96 - 2 for _, X, _ in RecordingSQRA.fit_records)
        )
        self.assertTrue(
            all(np.nanmax(X) < 1_000_000 for _, X, _ in RecordingSQRA.fit_records)
        )
        self.assert_output_contract(forecast, runtime)

    def test_mtu_specific_mode_retains_480_fits(self):
        with patch.object(sqra_model, "SQRA", RecordingSQRA):
            forecast, runtime = sqra_model.rolling_sqra_forecast_mtu(
                self.panel,
                [self.forecast_day],
                60,
                self.quantiles,
                ["prediction_p1"],
            )
        self.assertEqual(len(RecordingSQRA.fit_records), 96 * 5)
        self.assertTrue(
            all(np.nanmax(X) < 1_000_000 for _, X, _ in RecordingSQRA.fit_records)
        )
        self.assert_output_contract(forecast, runtime)

    def test_real_sqra_pooled_smoke(self):
        with patch.object(sqra_model, "SQRA", RealSQRA):
            forecast, runtime = sqra_model.rolling_sqra_forecast_pooled(
                self.panel,
                [self.forecast_day],
                2,
                [0.50],
                ["prediction_p1"],
            )
        self.assertEqual(list(forecast.columns), ["q0.500", "y_true"])
        self.assertEqual(len(forecast), 96)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(forecast["q0.500"].isna().sum(), 1)

    def test_spring_dst_target_day_uses_96_mtus_and_calendar_training(self):
        days = pd.date_range("2026-03-27", "2026-03-29", freq="D")
        index = make_delivery_index(days)
        values = np.arange(len(index), dtype=float)
        panel = pd.DataFrame(
            {"prediction_p1": values, "y_true": values + 0.25},
            index=index,
        )

        RecordingSQRA.fit_records.clear()
        with patch.object(sqra_model, "SQRA", RecordingSQRA):
            forecast, runtime = sqra_model.rolling_sqra_forecast_pooled(
                panel,
                [pd.Timestamp("2026-03-29")],
                2,
                [0.50],
                ["prediction_p1"],
            )

        self.assertEqual(len(RecordingSQRA.fit_records), 1)
        _, X_train, _ = RecordingSQRA.fit_records[0]
        self.assertEqual(len(X_train), 2 * 96)
        self.assertLess(X_train.max(), panel.loc[pd.Timestamp("2026-03-29")].iloc[0, 0])
        self.assertEqual(len(forecast), 96)
        self.assertEqual(
            forecast.index.get_level_values("mtu").tolist(), list(range(1, 97))
        )
        self.assertEqual(len(runtime), 1)


class SqraPanelTests(unittest.TestCase):
    @staticmethod
    def write_forecast(path, days, y_pred, y_true):
        index = make_delivery_index(pd.DatetimeIndex(days))
        pd.DataFrame(
            {"y_pred": y_pred, "y_true": y_true}, index=index
        ).to_csv(path)

    def test_panel_intersects_members_on_complete_canonical_days(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first_days = pd.date_range("2025-01-01", periods=2, freq="D")
            second_days = pd.date_range("2025-01-02", periods=2, freq="D")
            first_truth = np.arange(2 * 96, dtype=float)
            second_truth = np.arange(96, 3 * 96, dtype=float)
            self.write_forecast(
                base / "first.csv", first_days, first_truth + 1, first_truth
            )
            self.write_forecast(
                base / "second.csv", second_days, second_truth + 2, second_truth
            )

            panel, features = sqra_model.build_sqra_panel(
                [base / "first.csv", base / "second.csv"], "Europe/Berlin"
            )

        self.assertEqual(features, ["prediction_p1", "prediction_p2"])
        self.assertEqual(len(panel), 96)
        self.assertEqual(tuple(panel.index.names), INDEX_COLUMNS)
        self.assertEqual(panel.index.get_level_values("delivery_date").nunique(), 1)
        np.testing.assert_array_equal(panel["y_true"], np.arange(96, 192))

    def test_panel_rejects_different_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            day = pd.DatetimeIndex([pd.Timestamp("2025-01-01")])
            truth = np.arange(96, dtype=float)
            self.write_forecast(base / "first.csv", day, truth + 1, truth)
            changed_truth = truth.copy()
            changed_truth[-1] += 1
            self.write_forecast(
                base / "second.csv", day, truth + 2, changed_truth
            )
            with self.assertRaisesRegex(ValueError, "y_true differs"):
                sqra_model.build_sqra_panel(
                    [base / "first.csv", base / "second.csv"], "Europe/Berlin"
                )

    def test_panel_rejects_duplicates_and_disjoint_members(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            duplicated_index = pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2025-01-01"), 1),
                    (pd.Timestamp("2025-01-01"), 1),
                ],
                names=list(INDEX_COLUMNS),
            )
            pd.DataFrame(
                {"y_pred": [1.0, 2.0], "y_true": [3.0, 3.0]},
                index=duplicated_index,
            ).to_csv(base / "duplicates.csv")
            with self.assertRaisesRegex(ValueError, "duplicate delivery-date/MTU"):
                sqra_model.build_sqra_panel(
                    [base / "duplicates.csv"], "Europe/Berlin"
                )

            for filename, delivery_day in (
                ("first.csv", "2025-01-01"),
                ("second.csv", "2025-01-02"),
            ):
                self.write_forecast(
                    base / filename,
                    pd.DatetimeIndex([pd.Timestamp(delivery_day)]),
                    np.ones(96),
                    np.full(96, 2.0),
                )
            with self.assertRaisesRegex(ValueError, "no common delivery keys"):
                sqra_model.build_sqra_panel(
                    [base / "first.csv", base / "second.csv"], "Europe/Berlin"
                )

    def test_panel_keeps_all_96_mtu_keys_on_spring_dst_day(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "spring.csv"
            spring_day = pd.Timestamp("2026-03-29")
            self.write_forecast(
                path,
                pd.DatetimeIndex([spring_day]),
                np.arange(96, dtype=float),
                np.arange(96, dtype=float) + 0.5,
            )
            panel, _ = sqra_model.build_sqra_panel([path], "Europe/Berlin")

        self.assertEqual(len(panel), 96)
        self.assertEqual(
            panel.index.get_level_values("mtu").tolist(), list(range(1, 97))
        )
        self.assertEqual(
            panel.index.get_level_values("delivery_date").unique().tolist(),
            [spring_day],
        )

    def test_quantile_sorting_preserves_schema_and_nan_behavior(self):
        frame = pd.DataFrame(
            {
                "q0.100": [3.0, np.nan],
                "q0.500": [1.0, 2.0],
                "q0.900": [2.0, 1.0],
                "y_true": [0.0, 0.0],
            }
        )
        sorted_frame = sqra_model.sort_quantiles(
            frame, ["q0.100", "q0.500", "q0.900"]
        )
        self.assertEqual(list(sorted_frame.columns), list(frame.columns))
        np.testing.assert_allclose(
            sorted_frame.iloc[0, :3].to_numpy(), [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(
            sorted_frame.iloc[1, :3].to_numpy(), [1.0, 2.0, np.nan], equal_nan=True
        )


class SqraRunnerIntegrationTests(unittest.TestCase):
    def test_runner_writes_the_existing_four_file_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            panel, forecast_day = synthetic_panel()
            panel["prediction_p1"] = np.arange(len(panel), dtype=float)
            member_forecast = pd.DataFrame(
                {"y_pred": panel["prediction_p1"], "y_true": panel["y_true"]}
            )
            for clusters in (1, 5, 25):
                input_path = (
                    results_root
                    / "lear_op_results"
                    / "dwd"
                    / "d56"
                    / f"c{clusters}"
                    / "exaa"
                    / "forecast.csv"
                )
                input_path.parent.mkdir(parents=True)
                member_forecast.to_csv(input_path)

            config = replace(
                load_experiment_config(REPO_ROOT),
                evaluation_start=forecast_day.date(),
                evaluation_end=forecast_day.date(),
                sqra_train_days=60,
                sqra_mtu_specific=False,
                evaluation_skip_dates=(),
                results_root=results_root,
            )
            RecordingSQRA.fit_records.clear()
            with patch.object(sqra_model, "SQRA", RecordingSQRA):
                artifacts = run_sqra_experiment("dwd_exaa_enriched", config)

            output = results_root / "sqra_results" / "dwd_exaa_enriched"
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"forecast.csv", "runtime.csv", "metrics.csv", "config.json"},
            )
            self.assertEqual(
                list(artifacts.forecast.columns),
                ["q0.100", "q0.250", "q0.500", "q0.750", "q0.900", "y_true"],
            )
            self.assertEqual(tuple(artifacts.forecast.index.names), INDEX_COLUMNS)
            validate_delivery_index(
                artifacts.forecast.index,
                require_complete_days=True,
                expected_days=[forecast_day],
            )
            forecast_columns = pd.read_csv(output / "forecast.csv").columns.tolist()
            self.assertEqual(
                forecast_columns,
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
            self.assertEqual(
                list(artifacts.runtime.columns),
                ["forecast_day", "computation_time_seconds"],
            )
            self.assertEqual(
                list(artifacts.metrics.columns),
                [
                    "Target Day",
                    "MAE (median)",
                    "Coverage 0.25-0.75",
                    "Coverage 0.1-0.9",
                    "APS",
                ],
            )
            metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["calibration_mode"], "pooled")
            self.assertFalse(metadata["mtu_specific"])
            self.assertEqual(metadata["test_start"], str(forecast_day.date()))
            self.assertEqual(
                metadata["import_paths"],
                [
                    str(path)
                    for path in SQRA_RUNS["dwd_exaa_enriched"].inputs(results_root)
                ],
            )
            self.assertEqual(
                metadata["forecast_schema_version"], FORECAST_SCHEMA_VERSION
            )
            self.assertEqual(metadata["forecast_index"], "delivery_date_mtu")


if __name__ == "__main__":
    unittest.main()
