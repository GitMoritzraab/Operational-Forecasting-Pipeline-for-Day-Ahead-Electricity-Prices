import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiment_manifest import get_lear_run
from pipeline.delivery_index import make_delivery_index
from pipeline.lear import lear_model
from pipeline.lear import run_lear


class RecordingLasso:
    fit_inputs = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fit_inputs.append((X.copy(), np.asarray(y).copy()))
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.alpha_ = 0.0
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class LearManifestAndCliTests(unittest.TestCase):
    def test_manifest_and_cli_resolve_operational_run(self):
        expected = get_lear_run("era5_d56_c1_fundamental")
        args = run_lear.build_parser().parse_args(
            ["operational", "--config", expected.name]
        )
        self.assertEqual(args.command, "operational")
        self.assertEqual(get_lear_run(args.config), expected)

    def test_legacy_experiment_names_are_preserved(self):
        self.assertEqual(
            run_lear.operational_experiment_name(
                get_lear_run("era5_d56_c1_fundamental")
            ),
            "lear_era5_fundamental_c1_d56",
        )
        self.assertEqual(
            run_lear.operational_experiment_name(get_lear_run("exaa_only_d364")),
            "lear_exaa_only_d364",
        )


class LearExaaOnlyTests(unittest.TestCase):
    def test_market_loader_skips_load_cache(self):
        frame = pd.DataFrame()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SimpleNamespace(
                input_start=date(2025, 1, 1),
                evaluation_end=date(2025, 1, 2),
                timezone="Europe/Berlin",
                entsoe_price_cache_dir=root / "prices",
                exaa_price_cache_dir=root / "exaa",
                entsoe_load_cache_dir=root / "load",
                refresh_market_cache=False,
            )
            with (
                patch.object(run_lear, "_api_key", return_value="test-key"),
                patch.object(
                    run_lear,
                    "load_or_fetch_frame",
                    return_value=frame,
                ) as cache,
            ):
                _, _, load = run_lear.load_market_inputs(
                    config, include_load=False
                )

        self.assertIsNone(load)
        self.assertEqual(cache.call_count, 2)
        requested = [call.args[0] for call in cache.call_args_list]
        self.assertNotIn(config.entsoe_load_cache_dir / "load_forecast.csv", requested)

    def test_operational_exaa_only_requests_no_load_input(self):
        run = get_lear_run("exaa_only_d56")
        timestamp = pd.Timestamp("2025-01-01", tz="Europe/Berlin")
        forecast = pd.DataFrame(
            {"y_pred": [1.0], "y_true": [1.0]},
            index=pd.DatetimeIndex([timestamp], name="timestamp"),
        )
        runtime = pd.DataFrame(
            columns=[
                "forecast_day",
                "train_days",
                "use_vst",
                "use_lars",
                "runtime_seconds",
            ]
        )
        features = pd.DataFrame(
            {"exaa_d0_mtu_00": [1.0]},
            index=pd.DatetimeIndex([timestamp], name="date"),
        )
        target = pd.DataFrame(
            np.ones((1, 96)), index=features.index, columns=range(96)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                timezone="Europe/Berlin",
                point_forecast_start=date(2025, 1, 1),
                evaluation_end=date(2025, 1, 1),
                forecast_skip_dates=(),
                lear_use_vst=True,
                results_root=Path(temp_dir),
            )
            with (
                patch.object(
                    run_lear,
                    "load_market_inputs",
                    return_value=(pd.DataFrame(), pd.DataFrame(), None),
                ) as market,
                patch.object(
                    run_lear,
                    "_assemble_operational_matrices",
                    return_value=(features, target, pd.DataFrame()),
                ),
                patch.object(
                    run_lear,
                    "rolling_point_forecast",
                    return_value=(
                        forecast,
                        runtime,
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    ),
                ),
                patch.object(run_lear, "save_experiment_outputs") as save,
            ):
                run_lear.run_operational(config, run)

        market.assert_called_once_with(config, include_load=False)
        saved_config = save.call_args.kwargs["config"]
        self.assertEqual(saved_config["experiment_name"], "lear_exaa_only_d56")
        self.assertIsNone(saved_config["weather_source"])
        self.assertIsNone(saved_config["n_clusters"])
        self.assertEqual(saved_config["forecast_schema_version"], 2)
        self.assertEqual(saved_config["forecast_index"], "delivery_date_mtu")

    def test_exaa_only_matrix_assembly_never_loads_weather(self):
        index = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-01", tz="Europe/Berlin")], name="date"
        )
        features = pd.DataFrame({"exaa_d0_mtu_00": [10.0]}, index=index)
        target = pd.DataFrame(
            np.ones((1, 96)), index=index, columns=range(96)
        )
        run = get_lear_run("exaa_only_d56")
        with (
            patch.object(
                run_lear, "build_price_features", return_value=features
            ),
            patch.object(run_lear, "build_y_matrix", return_value=target),
            patch.object(run_lear, "_weather_features") as weather,
        ):
            actual_features, actual_target, dropped = (
                run_lear._assemble_operational_matrices(
                    SimpleNamespace(),
                    run,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                )
            )

        weather.assert_not_called()
        pd.testing.assert_frame_equal(actual_features, features)
        pd.testing.assert_frame_equal(actual_target, target)
        self.assertTrue(dropped.empty)


class LearDstNormalizationTests(unittest.TestCase):
    @staticmethod
    def _physical_day(day: str) -> pd.DatetimeIndex:
        start = pd.Timestamp(day, tz="Europe/Berlin")
        end = pd.Timestamp(start.date() + pd.Timedelta(days=1), tz=start.tz)
        return pd.date_range(start, end, freq="15min", inclusive="left")

    def test_spring_day_is_interpolated_to_96_local_mtus(self):
        index = self._physical_day("2026-03-29")
        self.assertEqual(len(index), 92)
        local_mtu = index.hour * 4 + index.minute // 15
        matrix = lear_model.build_daily_mtu_matrix(
            pd.Series(local_mtu.astype(float), index=index), "price"
        )

        self.assertEqual(matrix.shape, (1, 96))
        np.testing.assert_allclose(matrix.iloc[0].to_numpy(), np.arange(96))

    def test_autumn_repeated_mtus_are_averaged(self):
        index = self._physical_day("2026-10-25")
        self.assertEqual(len(index), 100)
        local_mtu = index.hour * 4 + index.minute // 15
        values = np.asarray(local_mtu, dtype=float).copy()
        second_hour = np.array(
            [
                timestamp.hour == 2
                and timestamp.utcoffset().total_seconds() == 3600
                for timestamp in index
            ]
        )
        values[second_hour] += 20.0
        matrix = lear_model.build_daily_mtu_matrix(
            pd.Series(values, index=index), "price"
        )

        np.testing.assert_allclose(matrix.iloc[0, 8:12], np.arange(8, 12) + 10)
        self.assertEqual(matrix.shape, (1, 96))

    def test_exaa_naive_uses_canonical_delivery_date_and_mtu(self):
        index = self._physical_day("2026-03-29")
        local_mtu = (index.hour * 4 + index.minute // 15).astype(float)
        epex = pd.DataFrame({"price_da": local_mtu}, index=index)
        exaa = pd.DataFrame({"price_exaa": local_mtu + 100.0}, index=index)

        forecast = lear_model.build_exaa_naive_forecast(
            epex,
            exaa,
            index[0],
            index[0],
        )

        self.assertEqual(forecast.index.names, ["delivery_date", "mtu"])
        self.assertEqual(len(forecast), 96)
        self.assertEqual(
            forecast.index.get_level_values("delivery_date").unique().tolist(),
            [pd.Timestamp("2026-03-29")],
        )
        self.assertEqual(
            forecast.index.get_level_values("mtu").tolist(),
            list(range(1, 97)),
        )
        np.testing.assert_allclose(forecast["y_true"], np.arange(96))
        np.testing.assert_allclose(forecast["y_pred"], np.arange(96) + 100)


class LearRollingTests(unittest.TestCase):
    def test_operational_missing_price_policy_skips_bad_training_target_and_lag(self):
        RecordingLasso.fit_inputs.clear()
        days = pd.date_range(
            "2026-09-10", periods=8, freq="D", tz="Europe/Berlin"
        )
        features = pd.DataFrame(
            {
                "price_d1_mtu_00": np.arange(8, dtype=float),
                "price_d2_mtu_00": np.arange(8, dtype=float) + 10.0,
                "load_d0_mtu_00": np.arange(8, dtype=float) + 20.0,
            },
            index=days,
        )
        # The operational day has no previous-day EPEX prices, so the complete
        # d-1 block must be removed from this fit.
        features.loc[days[-1], "price_d1_mtu_00"] = np.nan
        target = pd.DataFrame(
            np.arange(8 * 96, dtype=float).reshape(8, 96),
            index=days,
            columns=range(96),
        )
        # One unavailable historical price day is not used as a training target.
        target.loc[days[3], :] = np.nan

        with patch.object(lear_model, "LassoCV", RecordingLasso):
            forecast, _, _, _, _ = lear_model.rolling_point_forecast(
                X=features,
                Y=target,
                forecast_days=[days[-1]],
                train_days=6,
                lars_start_date=pd.Timestamp(
                    "2027-01-01", tz="Europe/Berlin"
                ),
                use_vst=False,
                progress=False,
                allow_incomplete_prices=True,
            )

        self.assertEqual(len(forecast), 96)
        self.assertEqual(len(RecordingLasso.fit_inputs), 96)
        self.assertTrue(
            all(X.shape == (5, 2) for X, _ in RecordingLasso.fit_inputs)
        )
        self.assertTrue(
            all(np.isfinite(X).all() and np.isfinite(y).all()
                for X, y in RecordingLasso.fit_inputs)
        )

    def test_rolling_point_uses_prior_days_and_returns_96_mtus(self):
        RecordingLasso.fit_inputs.clear()
        days = pd.date_range(
            "2025-01-01", periods=7, freq="D", tz="Europe/Berlin"
        )
        features = pd.DataFrame(
            {"is_holiday": [0, 1, 2, 3, 4, 5, 1_000_000]}, index=days
        )
        target_values = np.arange(7 * 96, dtype=float).reshape(7, 96)
        target = pd.DataFrame(target_values, index=days, columns=range(96))
        forecast_day = days[-1]

        with patch.object(lear_model, "LassoCV", RecordingLasso):
            forecast, runtime, coefficients, intercepts, degenerate = (
                lear_model.rolling_point_forecast(
                    X=features,
                    Y=target,
                    forecast_days=[forecast_day],
                    train_days=6,
                    lars_start_date=pd.Timestamp(
                        "2026-01-01", tz="Europe/Berlin"
                    ),
                    use_vst=False,
                    progress=False,
                )
            )

        self.assertEqual(len(RecordingLasso.fit_inputs), 96)
        self.assertTrue(
            all(len(X) == 6 for X, _ in RecordingLasso.fit_inputs)
        )
        self.assertTrue(
            all(np.max(X[:, 0]) <= 5 for X, _ in RecordingLasso.fit_inputs)
        )
        self.assertEqual(list(forecast.columns), ["y_pred", "y_true"])
        self.assertEqual(forecast.index.names, ["delivery_date", "mtu"])
        self.assertEqual(len(forecast), 96)
        self.assertIsNone(
            pd.DatetimeIndex(
                forecast.index.get_level_values("delivery_date")
            ).tz
        )
        self.assertEqual(
            forecast.index.get_level_values("mtu").tolist(),
            list(range(1, 97)),
        )
        self.assertTrue(forecast.index.is_monotonic_increasing)
        self.assertEqual(
            list(runtime.columns),
            [
                "forecast_day",
                "train_days",
                "use_vst",
                "use_lars",
                "runtime_seconds",
            ],
        )
        self.assertEqual(len(coefficients), 96)
        self.assertEqual(len(intercepts), 96)
        self.assertEqual(len(degenerate), 96)

    def test_anc_records_use_delivery_date_and_one_based_mtu(self):
        RecordingLasso.fit_inputs.clear()
        days = pd.date_range(
            "2026-03-23", periods=7, freq="D", tz="Europe/Berlin"
        )
        features = pd.DataFrame(
            {"is_holiday": np.arange(7, dtype=float)}, index=days
        )
        target = pd.DataFrame(
            np.arange(7 * 96, dtype=float).reshape(7, 96),
            index=days,
            columns=range(96),
        )

        with patch.object(lear_model, "LassoCV", RecordingLasso):
            anc = lear_model.rolling_anc_feature_importance(
                X=features,
                Y=target,
                forecast_days=[days[-1]],
                train_days=6,
                lars_start_date=pd.Timestamp(
                    "2027-01-01", tz="Europe/Berlin"
                ),
                progress=False,
            )

        self.assertNotIn("timestamp", anc.columns)
        self.assertNotIn("forecast_day", anc.columns)
        self.assertEqual(anc["delivery_date"].nunique(), 1)
        self.assertEqual(
            pd.Timestamp(anc["delivery_date"].iloc[0]),
            pd.Timestamp("2026-03-29"),
        )
        self.assertEqual(anc["mtu"].tolist(), list(range(1, 97)))


class LearOutputTests(unittest.TestCase):
    def test_operational_output_file_schemas(self):
        index = make_delivery_index([pd.Timestamp("2025-01-01")])
        forecast = pd.DataFrame(
            {
                "y_pred": np.arange(96, dtype=float),
                "y_true": np.full(96, 1.5),
            },
            index=index,
        )
        runtime = pd.DataFrame(
            [
                {
                    "forecast_day": pd.Timestamp("2025-01-01"),
                    "train_days": 56,
                    "use_vst": True,
                    "use_lars": False,
                    "runtime_seconds": 1.0,
                }
            ]
        )
        metadata = {
            "experiment_name": "lear_era5_fundamental_c1_d56",
            "test_start": "2025-01-01",
            "test_end": "2025-01-01",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            lear_model.save_experiment_outputs(
                metadata["experiment_name"],
                forecast,
                runtime,
                metadata,
                output,
                evaluation_start="2025-01-01",
                evaluation_end="2025-01-01",
            )
            forecast_columns = pd.read_csv(output / "forecast.csv").columns.tolist()
            runtime_columns = pd.read_csv(output / "runtime.csv").columns.tolist()
            metrics_columns = pd.read_csv(output / "metrics.csv").columns.tolist()
            saved_metadata = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            saved_metrics = pd.read_csv(output / "metrics.csv")

        self.assertEqual(
            forecast_columns,
            ["delivery_date", "mtu", "y_pred", "y_true"],
        )
        self.assertEqual(
            runtime_columns,
            [
                "forecast_day",
                "train_days",
                "use_vst",
                "use_lars",
                "runtime_seconds",
            ],
        )
        self.assertEqual(
            metrics_columns,
            ["period", "mae", "rmse", "bias", "n_obs", "n_inf_nan"],
        )
        self.assertEqual(saved_metadata, metadata)
        self.assertEqual(saved_metrics.iloc[-1]["period"], "evaluation")
        self.assertEqual(int(saved_metrics.iloc[-1]["n_obs"]), 96)

    def test_anc_summary_schemas(self):
        delivery_date = pd.Timestamp("2025-01-01")
        records = []
        for feature, contribution in (
            ("wind_speed_cluster_0_h00", 2.0),
            ("ssrd_cluster_1_h00", 3.0),
            ("price_d1_mtu_00", 4.0),
        ):
            records.append(
                {
                    "delivery_date": delivery_date,
                    "mtu": 1,
                    "train_days": 112,
                    "use_lars": True,
                    "feature": feature,
                    "feature_value": 1.0,
                    "beta": contribution,
                    "contribution": contribution,
                }
            )
        overall, wind, solar = lear_model.summarize_anc(pd.DataFrame(records))
        self.assertEqual(
            list(overall.columns), ["train_days", "feature_group", "ANC"]
        )
        self.assertEqual(
            list(wind.columns),
            ["train_days", "cluster_group", "ANC", "cluster_id"],
        )
        self.assertEqual(
            list(solar.columns),
            ["train_days", "cluster_group", "ANC", "cluster_id"],
        )
        self.assertEqual(wind["cluster_id"].tolist(), [0])
        self.assertEqual(solar["cluster_id"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
