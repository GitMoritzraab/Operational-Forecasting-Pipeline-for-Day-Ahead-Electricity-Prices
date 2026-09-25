from __future__ import annotations

import tempfile
import unittest
import bz2
import json
import os
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_cache import load_or_fetch_frame
from experiment_manifest import get_sqra_run
from pipeline.delivery_index import make_delivery_index
from pipeline.operational.config import OperationalConfig, load_operational_config
from pipeline.operational import dwd
from pipeline.operational.energy_arena import (
    ArenaChallenge,
    ArenaTargets,
    build_point_payload,
    build_quantile_payload,
    build_sqra_median_point_payload,
    physical_mtu_numbers,
    resolve_targets,
    submission_record_path,
)
from pipeline.operational.runner import (
    _assemble_matrices,
    _display_path,
    _fetch_retry,
    _payload_path,
    _submission_stream,
    arena_lock_path,
    build_operational_persistence_fallback,
    build_operational_persistence_quantile_fallback,
    build_parser,
    execute_pipeline,
    load_market_data,
    main,
    pipeline_log_path,
    pipeline_lock_paths,
    resolve_model_plan,
    run_point_model,
    run_sqra_model,
)


class _RecordingLasso:
    fit_shapes = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fit_shapes.append(X.shape)
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.alpha_ = 0.0
        self.intercept_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.intercept_, dtype=float)


class _RecordingSqra:
    fit_shapes = []

    def __init__(self, quantile, fit_intercept=True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.fit_shapes.append(X.shape)
        self.offset = float(np.quantile(y - X.mean(axis=1), self.quantile))
        return self

    def predict(self, X):
        return X.mean(axis=1) + self.offset


def _config(root: Path, **overrides) -> OperationalConfig:
    values = {
        "repo_root": root,
        "timezone": "Europe/Berlin",
        "entsoe_api_key": "entsoe-test",
        "arena_api_key": "arena-test",
        "arena_api_base_url": "https://api.energy-arena.org",
        "arena_point_challenge_id": "2",
        "arena_quantile_challenge_id": "8",
        "submit_to_arena": True,
        "point_variant": "fundamental",
        "point_clusters": 5,
        "sqra_variant": "auto",
        "sqra_train_days": 60,
        "sqra_mtu_specific": False,
        "lear_use_vst": True,
        "download_dwd": True,
        "delete_dwd_raw_after_preprocess": True,
        "download_exaa": "auto",
        "market_data_retry_seconds": 300,
        "exaa_only_download_attempts": 8,
        "fundamental_load_download_attempts": 4,
        "dwd_open_data_url": "https://opendata.dwd.de/weather/nwp/icon-d2/grib",
        "dwd_operational_run_hour": "06",
        "dwd_history_run_hour": "09",
        "allow_dwd_history_fallback": True,
        "dwd_download_attempts": 5,
        "dwd_retry_seconds": 0,
        "dwd_request_timeout_seconds": 30,
        "dwd_raw_archive": root / "raw",
        "icon_data_root": root / "processed",
        "entsoe_price_cache_dir": root / "prices",
        "exaa_price_cache_dir": root / "exaa",
        "entsoe_load_cache_dir": root / "load",
        "results_root": root / "results",
        "output_root": root / "output",
    }
    values.update(overrides)
    return OperationalConfig(**values)


def _challenge(objective: str, day: date) -> ArenaChallenge:
    timezone = "Europe/Berlin"
    return ArenaChallenge(
        challenge_id="2" if objective == "point" else "8",
        target_start=datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(timezone)),
        deadline=datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(timezone)),
        objective=objective,
        area="DE_LU",
        timezone=timezone,
        quantiles=(0.025, 0.25, 0.5, 0.75, 0.975) if objective == "quantile" else (),
        precision_decimals=2,
        detail={},
    )


class OperationalModelPlanTests(unittest.TestCase):
    def test_repository_paths_are_displayed_without_machine_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "DA_Price_Forecasting"
            config = _config(root)
            path = root / "results" / "operational" / "forecast.csv"

            displayed = _display_path(config, path)

        self.assertEqual(
            displayed,
            str(
                Path("DA_Price_Forecasting")
                / "results"
                / "operational"
                / "forecast.csv"
            ),
        )

    def test_operational_dwd_root_is_separate_from_historical_archive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.dict(
                os.environ,
                {
                    "DWD_OPERATIONAL_RAW_ROOT": "data/dwd_raw",
                    "DWD_RAW_ARCHIVE": "historical/dwd",
                    "OPERATIONAL_MARKET_DATA_ROOT": "data/market",
                },
            ):
                config = load_operational_config(root)

        self.assertEqual(config.dwd_raw_archive, root / "data" / "dwd_raw")
        self.assertEqual(
            config.entsoe_price_cache_dir, root / "data" / "market" / "entsoe"
        )
        self.assertEqual(
            config.entsoe_load_cache_dir, root / "data" / "market" / "entsoe"
        )
        self.assertEqual(
            config.exaa_price_cache_dir, root / "data" / "market" / "exaa"
        )

    def test_default_dwd_plan_keeps_c5_reference_and_all_sqra_members(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
            runs, point_run, sqra_name, members = resolve_model_plan(config)

        expected = {
            "dwd_d56_c1_fundamental",
            "dwd_d56_c5_fundamental",
            "dwd_d56_c25_fundamental",
        }
        self.assertEqual({run.name for run in runs}, expected)
        self.assertEqual(
            tuple(run.name for run in runs),
            (
                "dwd_d56_c1_fundamental",
                "dwd_d56_c5_fundamental",
                "dwd_d56_c25_fundamental",
            ),
        )
        self.assertEqual(members, expected)
        self.assertEqual(point_run.name, "dwd_d56_c5_fundamental")
        self.assertEqual(sqra_name, "dwd_fundamental")
        self.assertEqual(config.required_dwd_clusters, (1, 5, 25))

    def test_exaa_only_keeps_d364_reference_and_three_sqra_members(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                Path(temporary_directory),
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            runs, point_run, sqra_name, members = resolve_model_plan(config)

        expected = {"exaa_only_d56", "exaa_only_d112", "exaa_only_d364"}
        self.assertEqual({run.name for run in runs}, expected)
        self.assertEqual(
            tuple(run.name for run in runs),
            ("exaa_only_d56", "exaa_only_d112", "exaa_only_d364"),
        )
        self.assertEqual(members, expected)
        self.assertEqual(point_run.name, "exaa_only_d364")
        self.assertEqual(sqra_name, "exaa_only")
        self.assertFalse(config.needs_dwd)

    def test_fundamental_and_exaa_only_use_disjoint_model_locks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fundamental = _config(root)
            exaa_only = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            fundamental_locks = set(pipeline_lock_paths(fundamental))
            exaa_locks = set(pipeline_lock_paths(exaa_only))

        self.assertTrue(fundamental_locks)
        self.assertTrue(exaa_locks)
        self.assertTrue(fundamental_locks.isdisjoint(exaa_locks))

    def test_all_variants_share_one_short_lived_arena_submission_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fundamental = _config(root)
            exaa_only = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            fundamental_lock = arena_lock_path(fundamental)
            exaa_lock = arena_lock_path(exaa_only)

        self.assertEqual(fundamental_lock, exaa_lock)
        self.assertEqual(
            fundamental_lock,
            root / "output" / "locks" / "arena_default.lock",
        )

    def test_each_model_variant_has_one_stable_log_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fundamental = _config(root)
            exaa_only = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )

        self.assertEqual(
            pipeline_log_path(fundamental),
            root / "output" / "logs" / "fundamental.log",
        )
        self.assertEqual(
            pipeline_log_path(exaa_only),
            root / "output" / "logs" / "exaa_only.log",
        )

    def test_payload_path_is_stable_across_target_days(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
            first = _payload_path(
                config,
                "default",
                "2",
                datetime(2026, 9, 17, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            second = _payload_path(
                config,
                "default",
                "2",
                datetime(2026, 9, 18, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            config.output_root / "payloads" / "default" / "2" / "latest.json",
        )

    def test_information_cutoffs_keep_separate_stable_local_streams(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fundamental = _config(root)
            exaa_only = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )

        self.assertEqual(_submission_stream(fundamental), "default")
        self.assertEqual(_submission_stream(exaa_only), "exaa_only")

    def test_cli_accepts_requested_exaa_alias(self):
        args = build_parser().parse_args(["--exaa_only", "--no-submit"])
        self.assertTrue(args.exaa_only)
        self.assertTrue(args.no_submit)
        self.assertFalse(hasattr(args, "arena_profile"))
        self.assertFalse(hasattr(args, "arena_api_key"))

    def test_cli_accepts_safe_previous_delivery_day_override(self):
        args = build_parser().parse_args(["--no-submit", "--d-1"])
        self.assertTrue(args.no_submit)
        self.assertTrue(args.delivery_day_minus_one)

    def test_previous_delivery_day_override_cannot_submit(self):
        with self.assertRaises(SystemExit) as error:
            main(["--d-1"])
        self.assertEqual(error.exception.code, 2)

    def test_configuration_uses_only_the_primary_arena_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data" / "clustering").mkdir(parents=True)
            with patch.dict(
                os.environ,
                {"ENERGY_ARENA_API_KEY": "primary-key"},
            ):
                config = load_operational_config(root)
        self.assertEqual(config.arena_api_key, "primary-key")


class EnergyArenaPayloadTests(unittest.TestCase):
    def _forecasts(self, day: date):
        index = make_delivery_index([day])
        point = pd.DataFrame(
            {"y_pred": np.arange(1, 97, dtype=float), "y_true": np.nan},
            index=index,
        )
        quantile_values = np.column_stack(
            [np.arange(1, 97, dtype=float) + offset for offset in range(5)]
        )
        quantile = pd.DataFrame(
            quantile_values,
            index=index,
            columns=["q0.025", "q0.250", "q0.500", "q0.750", "q0.975"],
        )
        quantile["y_true"] = np.nan
        return point, quantile

    def test_submission_receipt_path_is_stable_across_target_days(self):
        root = Path("output")
        first = submission_record_path(
            root,
            "exaa_only",
            "8",
            datetime(2026, 9, 17, tzinfo=ZoneInfo("Europe/Berlin")),
        )
        second = submission_record_path(
            root,
            "exaa_only",
            "8",
            datetime(2026, 9, 18, tzinfo=ZoneInfo("Europe/Berlin")),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            root / "submissions" / "exaa_only" / "8" / "latest.json",
        )

    def test_regular_day_payload_shapes_and_quantile_order(self):
        day = date(2026, 9, 16)
        point, quantile = self._forecasts(day)
        point_payload = build_point_payload(point, _challenge("point", day))
        median_payload = build_sqra_median_point_payload(
            quantile, _challenge("point", day)
        )
        quantile_payload = build_quantile_payload(
            quantile, _challenge("quantile", day)
        )

        self.assertEqual(len(point_payload["values"]), 96)
        self.assertEqual(len(median_payload["values"]), 96)
        self.assertEqual(median_payload["values"][:3], [3, 4, 5])
        self.assertNotEqual(point_payload["values"], median_payload["values"])
        self.assertEqual(np.asarray(quantile_payload["values"]).shape, (96, 5))
        self.assertEqual(quantile_payload["values"][0], [1, 2, 3, 4, 5])
        self.assertEqual(
            median_payload["values"],
            np.asarray(quantile_payload["values"])[:, 2].tolist(),
        )

    def test_sqra_point_payload_requires_the_median_quantile(self):
        day = date(2026, 9, 16)
        _, quantile = self._forecasts(day)
        with self.assertRaisesRegex(ValueError, "q0.500"):
            build_sqra_median_point_payload(
                quantile.drop(columns="q0.500"),
                _challenge("point", day),
            )

    def test_previous_delivery_day_is_derived_from_live_arena_target(self):
        catalog = {
            "active_challenges": [
                {
                    "challenge_id": challenge_id,
                    "next_target_start": "2026-09-17T00:00:00+02:00",
                    "next_submission_deadline": "2026-09-16T12:00:00+02:00",
                }
                for challenge_id in ("2", "8")
            ]
        }

        def response(url, _api_key, _timeout):
            if url.endswith("/open"):
                return catalog
            challenge_id = url.rsplit("/", 1)[-1]
            return {
                "code": challenge_id,
                "forecast_objective": "point" if challenge_id == "2" else "quantile",
                "areas": ["DE_LU"],
                "target_code": "day_ahead_price",
                "target_period": {"timezone": "Europe/Berlin"},
                "probabilistic_forecast": (
                    {"quantiles": [0.025, 0.25, 0.5, 0.75, 0.975]}
                    if challenge_id == "8"
                    else {}
                ),
                "constraints": {"precision_decimals": 2},
            }

        with patch(
            "pipeline.operational.energy_arena._json_get", side_effect=response
        ):
            targets = resolve_targets(
                api_base="https://api.energy-arena.org",
                point_challenge_id="2",
                quantile_challenge_id="8",
                target_date_offset_days=-1,
            )

        self.assertEqual(targets.target_date, date(2026, 9, 16))
        self.assertEqual(targets.point.target_start, targets.quantile.target_start)

    def test_spring_and_autumn_payloads_follow_physical_dst_grid(self):
        spring = date(2026, 3, 29)
        autumn = date(2026, 10, 25)
        spring_mtus = physical_mtu_numbers(spring, "Europe/Berlin")
        autumn_mtus = physical_mtu_numbers(autumn, "Europe/Berlin")

        self.assertEqual(len(spring_mtus), 92)
        self.assertTrue(set(range(9, 13)).isdisjoint(spring_mtus))
        self.assertEqual(len(autumn_mtus), 100)
        for mtu in range(9, 13):
            self.assertEqual(int((autumn_mtus == mtu).sum()), 2)

        spring_point, spring_quantile = self._forecasts(spring)
        autumn_point, autumn_quantile = self._forecasts(autumn)
        self.assertEqual(
            len(build_point_payload(spring_point, _challenge("point", spring))["values"]),
            92,
        )
        self.assertEqual(
            len(
                build_quantile_payload(
                    autumn_quantile, _challenge("quantile", autumn)
                )["values"]
            ),
            100,
        )


class OperationalPersistenceFallbackTests(unittest.TestCase):
    @staticmethod
    def _prices(target_date: date, timezone: str) -> pd.DataFrame:
        start = pd.Timestamp(target_date - pd.Timedelta(days=67), tz=timezone)
        end = pd.Timestamp(target_date, tz=timezone)
        index = pd.date_range(start, end, freq="15min", inclusive="left")
        values = np.asarray(
            [
                timestamp.day * 1000.0
                + timestamp.hour * 4
                + timestamp.minute // 15
                for timestamp in index
            ],
            dtype=float,
        )
        return pd.DataFrame({"price_da": values}, index=index)

    def test_persistence_fallback_prefers_complete_d1(self):
        target = date(2026, 9, 18)
        prices = self._prices(target, "Europe/Berlin")

        forecast, lag, source_date = build_operational_persistence_fallback(
            prices,
            target,
        )

        self.assertEqual(lag, 1)
        self.assertEqual(source_date, date(2026, 9, 17))
        self.assertEqual(forecast.loc[(pd.Timestamp(target), 1), "y_pred"], 17000.0)
        self.assertEqual(forecast.loc[(pd.Timestamp(target), 96), "y_pred"], 17095.0)
        self.assertTrue(forecast["y_true"].isna().all())

    def test_persistence_fallback_uses_d7_when_d1_is_incomplete(self):
        target = date(2026, 9, 18)
        prices = self._prices(target, "Europe/Berlin")
        missing_d1 = prices.index.date == date(2026, 9, 17)
        prices.loc[missing_d1, "price_da"] = np.nan

        forecast, lag, source_date = build_operational_persistence_fallback(
            prices,
            target,
        )

        self.assertEqual(lag, 7)
        self.assertEqual(source_date, date(2026, 9, 11))
        self.assertEqual(forecast.loc[(pd.Timestamp(target), 1), "y_pred"], 11000.0)
        quantile_forecast, metadata = (
            build_operational_persistence_quantile_fallback(
                prices,
                target,
                lag=lag,
                point_forecast=forecast,
                quantiles=(0.025, 0.25, 0.5, 0.75, 0.975),
            )
        )
        self.assertEqual(metadata["lag_days"], 7)
        np.testing.assert_allclose(
            quantile_forecast["q0.500"],
            forecast["y_pred"],
        )

    def test_quantile_fallback_is_pooled_non_crossing_and_median_centered(self):
        target = date(2026, 9, 18)
        prices = self._prices(target, "Europe/Berlin")
        point, lag, _ = build_operational_persistence_fallback(prices, target)

        forecast, metadata = build_operational_persistence_quantile_fallback(
            prices,
            target,
            lag=lag,
            point_forecast=point,
            quantiles=(0.025, 0.25, 0.5, 0.75, 0.975),
        )

        quantile_columns = [
            "q0.025",
            "q0.250",
            "q0.500",
            "q0.750",
            "q0.975",
        ]
        np.testing.assert_allclose(forecast["q0.500"], point["y_pred"])
        self.assertTrue(
            (
                np.diff(
                    forecast[quantile_columns].to_numpy(dtype=float),
                    axis=1,
                )
                >= 0
            ).all()
        )
        self.assertEqual(metadata["complete_calibration_days"], 60)
        self.assertEqual(metadata["calibration_observations"], 60 * 96)
        self.assertTrue(metadata["target_excluded_from_calibration"])
        self.assertTrue(metadata["median_centered"])

    def test_fundamental_dwd_failure_saves_both_fallback_payloads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            target = date(2026, 9, 18)
            targets = ArenaTargets(
                point=_challenge("point", target),
                quantile=_challenge("quantile", target),
            )
            prices = self._prices(target, config.timezone)

            with (
                patch(
                    "pipeline.operational.runner.prepare_target_dwd",
                    side_effect=RuntimeError("06 run unavailable"),
                ),
                patch(
                    "pipeline.operational.runner._target_dwd_histories_ready",
                    return_value=False,
                ),
                patch(
                    "pipeline.operational.runner._load_entsoe_prices",
                    return_value=prices,
                ) as load_prices,
                patch(
                    "pipeline.operational.runner.load_market_data"
                ) as load_all_market,
            ):
                summary = execute_pipeline(
                    config,
                    targets,
                    submit=False,
                )

            point_path = (
                config.output_root
                / "payloads"
                / "default"
                / targets.point.challenge_id
                / "latest.json"
            )
            quantile_path = (
                config.output_root
                / "payloads"
                / "default"
                / targets.quantile.challenge_id
                / "latest.json"
            )
            point_payload = json.loads(point_path.read_text(encoding="utf-8"))
            quantile_payload = json.loads(
                quantile_path.read_text(encoding="utf-8")
            )
            results_created = config.results_root.exists()

        self.assertTrue(summary["fallback"])
        self.assertEqual(summary["point_model"], "persistence_d1_dwd_fallback")
        self.assertEqual(
            summary["quantile_model"],
            "pooled_residual_persistence_d1_dwd_fallback",
        )
        self.assertEqual(summary["quantile_status"], "generated")
        self.assertEqual(
            summary["quantile_fallback"]["calibration_observations"],
            60 * 96,
        )
        self.assertFalse(summary["model_histories_updated"])
        self.assertEqual(len(point_payload["values"]), 96)
        self.assertEqual(len(quantile_payload["values"]), 96)
        self.assertEqual(
            [row[2] for row in quantile_payload["values"]],
            point_payload["values"],
        )
        self.assertFalse(results_created)
        load_prices.assert_called_once()
        load_all_market.assert_not_called()

    def test_complete_fallback_submits_point_and_quantile_payloads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            target = date(2099, 9, 18)
            targets = ArenaTargets(
                point=_challenge("point", target),
                quantile=_challenge("quantile", target),
            )
            prices = self._prices(target, config.timezone)

            with (
                patch(
                    "pipeline.operational.runner.prepare_target_dwd",
                    side_effect=RuntimeError("06 run unavailable"),
                ),
                patch(
                    "pipeline.operational.runner._target_dwd_histories_ready",
                    return_value=False,
                ),
                patch(
                    "pipeline.operational.runner._load_entsoe_prices",
                    return_value=prices,
                ),
                patch(
                    "pipeline.operational.runner.submit_with_record",
                    return_value={"status": "submitted"},
                ) as submit_payload,
            ):
                summary = execute_pipeline(config, targets, submit=True)

        self.assertEqual(submit_payload.call_count, 2)
        submitted_challenges = {
            call.kwargs["payload"]["challenge_id"]
            for call in submit_payload.call_args_list
        }
        self.assertEqual(
            submitted_challenges,
            {
                targets.point.challenge_id,
                targets.quantile.challenge_id,
            },
        )
        self.assertEqual(set(summary["submissions"]), {"point", "quantile"})

    def test_insufficient_quantile_history_keeps_point_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            target = date(2026, 9, 18)
            targets = ArenaTargets(
                point=_challenge("point", target),
                quantile=_challenge("quantile", target),
            )
            prices = self._prices(target, config.timezone)
            cutoff = pd.Timestamp(target - pd.Timedelta(days=7), tz=config.timezone)
            prices = prices.loc[prices.index >= cutoff]

            with (
                patch(
                    "pipeline.operational.runner.prepare_target_dwd",
                    side_effect=RuntimeError("06 run unavailable"),
                ),
                patch(
                    "pipeline.operational.runner._target_dwd_histories_ready",
                    return_value=False,
                ),
                patch(
                    "pipeline.operational.runner._load_entsoe_prices",
                    return_value=prices,
                ),
            ):
                summary = execute_pipeline(config, targets, submit=False)

            point_path = (
                config.output_root
                / "payloads"
                / "default"
                / targets.point.challenge_id
                / "latest.json"
            )
            quantile_path = (
                config.output_root
                / "payloads"
                / "default"
                / targets.quantile.challenge_id
                / "latest.json"
            )
            point_exists = point_path.is_file()
            quantile_exists = quantile_path.is_file()

        self.assertEqual(
            summary["quantile_status"],
            "skipped_insufficient_history",
        )
        self.assertIsNone(summary["quantile_model"])
        self.assertTrue(point_exists)
        self.assertFalse(quantile_exists)
        self.assertFalse(summary["model_histories_updated"])

    def test_non_fundamental_dwd_failure_is_not_masked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                Path(temporary_directory),
                point_variant="exaa_enriched",
                sqra_variant="exaa_enriched",
            )
            target = date(2026, 9, 18)
            targets = ArenaTargets(
                point=_challenge("point", target),
                quantile=_challenge("quantile", target),
            )

            with patch(
                "pipeline.operational.runner.prepare_target_dwd",
                side_effect=RuntimeError("06 run unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "06 run unavailable"):
                    execute_pipeline(config, targets, submit=False)


class FuturePointForecastTests(unittest.TestCase):
    def test_point_runner_applies_operational_missing_price_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2026-01-01", periods=57, freq="D", tz=config.timezone
            )
            target_day = pd.DatetimeIndex([days[-1]])
            features = pd.DataFrame(
                {
                    "price_d1_mtu_00": np.arange(len(days), dtype=float),
                    "load_d0_mtu_00": np.arange(len(days), dtype=float),
                },
                index=days,
            )
            features.loc[days[-1], "price_d1_mtu_00"] = np.nan
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )
            target.loc[days[20], :] = np.nan
            target.loc[days[-1], :] = np.nan

            _RecordingLasso.fit_shapes = []
            with (
                patch(
                    "pipeline.operational.runner._assemble_matrices",
                    return_value=(features, target),
                ),
                patch(
                    "pipeline.lear.lear_model.LassoLarsCV",
                    _RecordingLasso,
                ),
                patch(
                    "pipeline.lear.lear_model.LassoCV",
                    _RecordingLasso,
                ),
            ):
                forecast = run_point_model(
                    config,
                    run,
                    requested_days=target_day,
                    prices=pd.DataFrame(),
                    exaa=None,
                    load=pd.DataFrame(),
                    weather={
                        5: pd.DataFrame(
                            {"weather": np.arange(len(days), dtype=float)},
                            index=days,
                        )
                    },
                    force=False,
                )

        target_rows = forecast.loc[pd.Timestamp(days[-1].date())]
        self.assertTrue(np.isfinite(target_rows["y_pred"]).all())
        self.assertTrue(target_rows["y_true"].isna().all())
        self.assertEqual(_RecordingLasso.fit_shapes, [(55, 1)] * 96)

    def test_target_can_be_forecast_before_its_epex_truth_exists(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2026-01-01", periods=57, freq="D", tz=config.timezone
            )
            target_day = pd.DatetimeIndex([days[-1]])
            features = pd.DataFrame(
                {"feature": np.arange(len(days), dtype=float)}, index=days
            )
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )
            target.loc[days[-1], :] = np.nan

            with (
                patch(
                    "pipeline.operational.runner._assemble_matrices",
                    return_value=(features, target),
                ),
                patch(
                    "pipeline.lear.lear_model.LassoLarsCV",
                    _RecordingLasso,
                ),
                patch(
                    "pipeline.lear.lear_model.LassoCV",
                    _RecordingLasso,
                ),
            ):
                forecast = run_point_model(
                    config,
                    run,
                    requested_days=target_day,
                    prices=pd.DataFrame(),
                    exaa=None,
                    load=pd.DataFrame(),
                    weather={
                        5: pd.DataFrame(
                            {"weather": np.arange(len(days), dtype=float)},
                            index=days,
                        )
                    },
                    force=False,
                )

        target_rows = forecast.loc[pd.Timestamp(days[-1].date())]
        self.assertEqual(len(target_rows), 96)
        self.assertTrue(np.isfinite(target_rows["y_pred"]).all())
        self.assertTrue(target_rows["y_true"].isna().all())

    def test_missing_historical_dwd_day_is_skipped_for_point_forecasts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2026-01-01", periods=57, freq="D", tz=config.timezone
            )
            missing_weather_day = days[20]
            requested_days = pd.DatetimeIndex(
                [missing_weather_day, days[-1]]
            )
            features = pd.DataFrame(
                {
                    "weather": np.arange(len(days), dtype=float),
                    "price_d1_mtu_00": np.arange(len(days), dtype=float),
                },
                index=days,
            )
            features.loc[missing_weather_day, "weather"] = np.nan
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )
            target.loc[days[-1], :] = np.nan
            weather = pd.DataFrame(
                {"weather": np.arange(len(days), dtype=float)}, index=days
            ).drop(index=missing_weather_day)

            _RecordingLasso.fit_shapes = []
            with (
                patch(
                    "pipeline.operational.runner._assemble_matrices",
                    return_value=(features, target),
                ),
                patch(
                    "pipeline.lear.lear_model.LassoLarsCV",
                    _RecordingLasso,
                ),
                patch(
                    "pipeline.lear.lear_model.LassoCV",
                    _RecordingLasso,
                ),
            ):
                forecast = run_point_model(
                    config,
                    run,
                    requested_days=requested_days,
                    prices=pd.DataFrame(),
                    exaa=None,
                    load=pd.DataFrame(),
                    weather={5: weather},
                    force=False,
                )

            metadata = json.loads(
                (run.output_dir(config.results_root) / "config.json").read_text(
                    encoding="utf-8"
                )
            )

        forecast_days = forecast.index.get_level_values("delivery_date").unique()
        self.assertEqual(forecast_days.tolist(), [pd.Timestamp(days[-1].date())])
        self.assertEqual(_RecordingLasso.fit_shapes, [(55, 2)] * 96)
        expected_day = missing_weather_day.date().isoformat()
        self.assertIn(expected_day, metadata["skipped_historical_weather_days"])
        self.assertIn(expected_day, metadata["skipped_point_forecast_days"])

    def test_missing_historical_exaa_day_is_skipped_for_point_forecasts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2025-01-01", periods=365, freq="D", tz=config.timezone
            )
            missing_exaa_day = days[100]
            requested_days = pd.DatetimeIndex(
                [missing_exaa_day, days[-1]]
            )
            features = pd.DataFrame(
                {"exaa_d0_mtu_00": np.arange(len(days), dtype=float)},
                index=days,
            )
            features.loc[missing_exaa_day, "exaa_d0_mtu_00"] = np.nan
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )
            target.loc[days[-1], :] = np.nan

            _RecordingLasso.fit_shapes = []
            with (
                patch(
                    "pipeline.operational.runner._assemble_matrices",
                    return_value=(features, target),
                ),
                patch(
                    "pipeline.lear.lear_model.LassoLarsCV",
                    _RecordingLasso,
                ),
                patch(
                    "pipeline.lear.lear_model.LassoCV",
                    _RecordingLasso,
                ),
            ):
                forecast = run_point_model(
                    config,
                    run,
                    requested_days=requested_days,
                    prices=pd.DataFrame(),
                    exaa=pd.DataFrame(),
                    load=None,
                    weather={},
                    force=False,
                )

            metadata = json.loads(
                (run.output_dir(config.results_root) / "config.json").read_text(
                    encoding="utf-8"
                )
            )

        forecast_days = forecast.index.get_level_values("delivery_date").unique()
        self.assertEqual(forecast_days.tolist(), [pd.Timestamp(days[-1].date())])
        self.assertEqual(_RecordingLasso.fit_shapes, [(363, 1)] * 96)
        expected_day = missing_exaa_day.date().isoformat()
        self.assertIn(expected_day, metadata["skipped_historical_exaa_days"])
        self.assertIn(expected_day, metadata["skipped_point_forecast_days"])

    def test_missing_target_day_exaa_remains_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2025-01-01", periods=365, freq="D", tz=config.timezone
            )
            features = pd.DataFrame(
                {"exaa_d0_mtu_00": np.arange(len(days), dtype=float)},
                index=days,
            )
            features.loc[days[-1], "exaa_d0_mtu_00"] = np.nan
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )

            with patch(
                "pipeline.operational.runner._assemble_matrices",
                return_value=(features, target),
            ):
                with self.assertRaisesRegex(RuntimeError, "target-day EXAA"):
                    run_point_model(
                        config,
                        run,
                        requested_days=pd.DatetimeIndex([days[-1]]),
                        prices=pd.DataFrame(),
                        exaa=pd.DataFrame(),
                        load=None,
                        weather={},
                        force=False,
                    )

    def test_missing_target_day_dwd_data_remains_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            run = resolve_model_plan(config)[1]
            days = pd.date_range(
                "2026-01-01", periods=57, freq="D", tz=config.timezone
            )
            target_day = days[-1]
            features = pd.DataFrame(
                {
                    "weather": np.arange(len(days), dtype=float),
                    "price_d1_mtu_00": np.arange(len(days), dtype=float),
                },
                index=days,
            )
            features.loc[target_day, "weather"] = np.nan
            target = pd.DataFrame(
                np.tile(np.arange(96, dtype=float), (len(days), 1)),
                index=days,
                columns=range(96),
            )
            weather = pd.DataFrame(
                {"weather": np.arange(len(days), dtype=float)}, index=days
            ).drop(index=target_day)

            with patch(
                "pipeline.operational.runner._assemble_matrices",
                return_value=(features, target),
            ):
                with self.assertRaisesRegex(RuntimeError, "target-day DWD"):
                    run_point_model(
                        config,
                        run,
                        requested_days=pd.DatetimeIndex([target_day]),
                        prices=pd.DataFrame(),
                        exaa=None,
                        load=pd.DataFrame(),
                        weather={5: weather},
                        force=False,
                    )

    def test_dwd_sqra_skips_missing_historical_delivery_day(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            days = pd.date_range("2026-01-01", periods=61, freq="D")
            missing_day = days[20]
            available_days = days.delete(20)
            index = make_delivery_index(available_days)
            truth = np.arange(len(index), dtype=float)
            truth[-96:] = np.nan
            specification = get_sqra_run("dwd_fundamental")
            for member_number, member_path in enumerate(
                specification.inputs(config.results_root), start=1
            ):
                member_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "y_pred": np.arange(len(index), dtype=float)
                        + member_number,
                        "y_true": truth,
                    },
                    index=index,
                ).to_csv(member_path)

            _RecordingSqra.fit_shapes.clear()
            with patch("pipeline.sqra.sqra_model.SQRA", _RecordingSqra):
                forecast = run_sqra_model(
                    config,
                    "dwd_fundamental",
                    days[-1].date(),
                    (0.025, 0.25, 0.5, 0.75, 0.975),
                )

            metadata = json.loads(
                (
                    specification.output_dir(config.results_root)
                    / "config.json"
                ).read_text(encoding="utf-8")
            )

        target_rows = forecast.loc[days[-1]]
        self.assertEqual(len(target_rows), 96)
        self.assertTrue(np.isfinite(target_rows.filter(like="q").to_numpy()).all())
        self.assertEqual(_RecordingSqra.fit_shapes, [(59 * 96, 3)] * 5)
        self.assertEqual(
            metadata["skipped_historical_delivery_days"],
            [missing_day.date().isoformat()],
        )

    def test_dwd_sqra_uses_three_cluster_members_for_unrealized_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            days = pd.date_range("2026-01-01", periods=61, freq="D")
            index = make_delivery_index(days)
            truth = np.arange(len(index), dtype=float)
            truth[-96:] = np.nan
            specification = get_sqra_run("dwd_fundamental")
            for member_number, path in enumerate(
                specification.inputs(config.results_root), start=1
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "y_pred": np.arange(len(index), dtype=float) + member_number,
                        "y_true": truth,
                    },
                    index=index,
                ).to_csv(path)

            _RecordingSqra.fit_shapes.clear()
            with patch("pipeline.sqra.sqra_model.SQRA", _RecordingSqra):
                forecast = run_sqra_model(
                    config,
                    "dwd_fundamental",
                    days[-1].date(),
                    (0.025, 0.25, 0.5, 0.75, 0.975),
                )

        target_rows = forecast.loc[days[-1]]
        self.assertEqual(len(target_rows), 96)
        self.assertTrue(
            np.isfinite(target_rows.filter(like="q").to_numpy()).all()
        )
        self.assertTrue(target_rows["y_true"].isna().all())
        self.assertEqual(_RecordingSqra.fit_shapes, [(60 * 96, 3)] * 5)


class DwdAcquisitionTests(unittest.TestCase):
    def test_operational_filename_is_mapped_to_its_lead(self):
        filename = (
            "icon-d2_germany_regular-lat-lon_single-level_"
            "2026091506_048_2d_u_10m.grib2.bz2"
        )
        self.assertEqual(dwd._lead_from_name(filename, "2026091506", "u_10m"), 48)
        self.assertIsNone(dwd._lead_from_name(filename, "2026091506", "v_10m"))

    def test_bzip_verification_reads_the_stream_and_checks_grib_magic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid = root / "valid.grib2.bz2"
            invalid = root / "invalid.grib2.bz2"
            valid.write_bytes(bz2.compress(b"GRIB" + b"test-data"))
            invalid.write_bytes(bz2.compress(b"NOT-A-GRIB"))

            dwd.verify_bz2_grib(valid)
            with self.assertRaisesRegex(ValueError, "GRIB message"):
                dwd.verify_bz2_grib(invalid)

    def test_failed_run_verification_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(
                root,
                dwd_download_attempts=2,
                dwd_retry_seconds=0,
            )
            issue_day = date(2026, 9, 15)
            with (
                patch.object(dwd, "_remote_files", return_value={}),
                patch.object(
                    dwd,
                    "verify_archived_run",
                    side_effect=[RuntimeError("corrupt"), None],
                ) as verify,
                patch.object(dwd, "_download_variable_archive"),
                patch.object(dwd.time, "sleep") as sleep,
            ):
                result = dwd.download_run(config, issue_day)

            self.assertEqual(
                result,
                dwd.archived_day_dir(root / "raw", issue_day),
            )
            self.assertEqual(verify.call_count, 2)
            sleep.assert_called_once_with(0)


class MarketAcquisitionPolicyTests(unittest.TestCase):
    @staticmethod
    def _price_frame(timezone: str) -> pd.DataFrame:
        index = pd.DatetimeIndex([pd.Timestamp("2026-09-24", tz=timezone)])
        return pd.DataFrame({"price_da": [100.0]}, index=index)

    def test_exaa_only_uses_eight_total_attempts_then_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                Path(temporary_directory),
                point_variant="exaa_only",
                sqra_variant="exaa_only",
                market_data_retry_seconds=0,
                exaa_only_download_attempts=8,
            )
            requested_files = []

            def loader(path, *args, **kwargs):
                requested_files.append(Path(path).name)
                if Path(path).name == "prices_da.csv":
                    return self._price_frame(config.timezone)
                raise RuntimeError("EXAA not published")

            with (
                patch(
                    "pipeline.operational.runner.load_or_fetch_frame",
                    side_effect=loader,
                ),
                patch("pipeline.operational.runner.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "after 8 attempts"):
                    load_market_data(
                        config,
                        start_date=date(2026, 9, 20),
                        target_date=date(2026, 9, 25),
                    )

        self.assertEqual(requested_files.count("prices_exaa.csv"), 8)
        self.assertEqual(sleep.call_count, 7)
        self.assertEqual(
            [item.args[0] for item in sleep.call_args_list],
            [0] * 7,
        )

    def test_historical_exaa_gap_does_not_trigger_target_retry_loop(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                Path(temporary_directory),
                point_variant="exaa_only",
                sqra_variant="exaa_only",
                market_data_retry_seconds=0,
            )
            index = pd.date_range(
                pd.Timestamp("2026-09-20", tz=config.timezone),
                pd.Timestamp("2026-09-26", tz=config.timezone),
                freq="15min",
                inclusive="left",
            )
            exaa_frame = pd.DataFrame({"price_exaa": 50.0}, index=index)
            historical_gap = exaa_frame.index.date == date(2026, 9, 22)
            exaa_frame.loc[historical_gap, "price_exaa"] = np.nan
            requested_files = []

            def loader(path, *args, **kwargs):
                requested_files.append(Path(path).name)
                if Path(path).name == "prices_da.csv":
                    return self._price_frame(config.timezone)
                return exaa_frame

            with (
                patch(
                    "pipeline.operational.runner.load_or_fetch_frame",
                    side_effect=loader,
                ),
                patch("pipeline.operational.runner.time.sleep") as sleep,
            ):
                _, exaa, load = load_market_data(
                    config,
                    start_date=date(2026, 9, 20),
                    target_date=date(2026, 9, 25),
                )

        self.assertIsNotNone(exaa)
        self.assertIsNone(load)
        self.assertTrue(exaa.loc[historical_gap, "price_exaa"].isna().all())
        self.assertTrue(
            exaa.loc[exaa.index.date == date(2026, 9, 25), "price_exaa"]
            .notna()
            .all()
        )
        self.assertEqual(requested_files.count("prices_exaa.csv"), 1)
        sleep.assert_not_called()

    def test_fundamental_omits_load_after_four_failed_attempts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                Path(temporary_directory),
                market_data_retry_seconds=0,
                fundamental_load_download_attempts=4,
            )
            requested_files = []

            def loader(path, *args, **kwargs):
                requested_files.append(Path(path).name)
                if Path(path).name == "prices_da.csv":
                    return self._price_frame(config.timezone)
                raise RuntimeError("load forecast not published")

            with (
                patch(
                    "pipeline.operational.runner.load_or_fetch_frame",
                    side_effect=loader,
                ),
                patch("pipeline.operational.runner.time.sleep") as sleep,
            ):
                prices, exaa, load = load_market_data(
                    config,
                    start_date=date(2026, 9, 20),
                    target_date=date(2026, 9, 25),
                )

        self.assertFalse(prices.empty)
        self.assertIsNone(exaa)
        self.assertIsNone(load)
        self.assertEqual(requested_files.count("load_forecast.csv"), 4)
        self.assertEqual(sleep.call_count, 3)

    def test_fundamental_matrix_can_omit_the_load_block(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
            lear_run = resolve_model_plan(config)[1]
            dates = pd.date_range(
                "2026-09-23", periods=2, freq="D", tz=config.timezone
            )
            weather = pd.DataFrame({"weather": [1.0, 2.0]}, index=dates)
            price_features = pd.DataFrame(
                {"price_d1_mtu_00": [50.0, 51.0]}, index=dates
            )
            time_features = pd.DataFrame(
                {"is_holiday": [0, 0]}, index=dates
            )
            target = pd.DataFrame({0: [45.0, np.nan]}, index=dates)

            with (
                patch(
                    "pipeline.operational.runner.build_price_features",
                    return_value=price_features,
                ),
                patch(
                    "pipeline.operational.runner.build_temporal_features",
                    return_value=time_features,
                ),
                patch(
                    "pipeline.operational.runner.build_y_matrix",
                    return_value=target,
                ),
            ):
                features, result_target = _assemble_matrices(
                    config,
                    lear_run,
                    prices=pd.DataFrame(),
                    exaa=None,
                    load=None,
                    weather={5: weather},
                    first_feature_date=date(2026, 9, 23),
                    target_date=date(2026, 9, 24),
                )

        self.assertEqual(
            features.columns.tolist(),
            ["weather", "price_d1_mtu_00", "is_holiday"],
        )
        self.assertFalse(any(column.startswith("load_") for column in features))
        pd.testing.assert_frame_equal(result_target, target)


class MarketCacheTests(unittest.TestCase):
    def test_market_retry_waits_between_attempts(self):
        calls = []

        def operation():
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise RuntimeError("not published yet")
            return "available"

        with patch("pipeline.operational.runner.time.sleep") as sleep:
            result = _fetch_retry(
                "EXAA prices",
                operation,
                attempts=3,
                retry_seconds=300,
            )

        self.assertEqual(result, "available")
        self.assertEqual(calls, [1, 2])
        sleep.assert_called_once_with(300)

    def test_empty_refresh_for_known_operational_gap_is_retained_as_missing(self):
        timezone = "Europe/Berlin"
        start = pd.Timestamp("2026-09-12", tz=timezone)
        end = pd.Timestamp("2026-09-14", tz=timezone)
        index = pd.date_range(
            start,
            pd.Timestamp("2026-09-15", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        cached = pd.DataFrame(
            {"price_da": np.arange(len(index), dtype=float)}, index=index
        )
        missing_day = cached.index.date == date(2026, 9, 13)
        cached.loc[missing_day, "price_da"] = np.nan
        calls = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices.csv"
            cached.to_csv(path)

            def empty_fetch(fetch_start, fetch_end):
                calls.append((fetch_start, fetch_end))
                return pd.DataFrame(columns=["price_da"])

            result = load_or_fetch_frame(
                path,
                start,
                end,
                empty_fetch,
                timezone=timezone,
                allow_incomplete=True,
            )

        missing = result.index.date == date(2026, 9, 13)
        self.assertEqual(
            calls,
            [
                (
                    pd.Timestamp("2026-09-13", tz=timezone),
                    pd.Timestamp("2026-09-13", tz=timezone),
                )
            ],
        )
        self.assertEqual(int(missing.sum()), 96)
        self.assertTrue(result.loc[missing, "price_da"].isna().all())
        self.assertTrue(result.loc[~missing, "price_da"].notna().all())

    def test_unresolved_historical_gap_is_retried_on_every_cache_load(self):
        timezone = "Europe/Berlin"
        start = pd.Timestamp("2026-09-20", tz=timezone)
        end = pd.Timestamp("2026-09-22", tz=timezone)
        index = pd.date_range(
            start,
            pd.Timestamp("2026-09-23", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        cached = pd.DataFrame({"price_exaa": 50.0}, index=index)
        gap = cached.index.date == date(2026, 9, 21)
        cached.loc[gap, "price_exaa"] = np.nan
        calls = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices_exaa.csv"
            cached.to_csv(path)

            def eventually_available_fetch(fetch_start, fetch_end):
                calls.append((fetch_start, fetch_end))
                if len(calls) == 1:
                    return pd.DataFrame(columns=["price_exaa"])
                return pd.DataFrame(
                    {"price_exaa": 75.0},
                    index=cached.index[gap],
                )

            first = load_or_fetch_frame(
                path,
                start,
                end,
                eventually_available_fetch,
                timezone=timezone,
                allow_incomplete=True,
            )
            second = load_or_fetch_frame(
                path,
                start,
                end,
                eventually_available_fetch,
                timezone=timezone,
                allow_incomplete=True,
            )

        expected_call = (
            pd.Timestamp("2026-09-21", tz=timezone),
            pd.Timestamp("2026-09-21", tz=timezone),
        )
        self.assertEqual(calls, [expected_call, expected_call])
        self.assertTrue(first.loc[gap, "price_exaa"].isna().all())
        self.assertTrue((second.loc[gap, "price_exaa"] == 75.0).all())

    def test_cache_fetches_only_the_new_daily_extension(self):
        timezone = "Europe/Berlin"
        first_day = pd.Timestamp("2026-09-15", tz=timezone)
        second_day = pd.Timestamp("2026-09-16", tz=timezone)
        first_index = pd.date_range(
            first_day,
            second_day,
            freq="15min",
            inclusive="left",
        )
        second_index = pd.date_range(
            second_day,
            pd.Timestamp("2026-09-17", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        cached = pd.DataFrame(
            {"price_da": np.arange(len(first_index), dtype=float)},
            index=first_index,
        )
        fresh = pd.DataFrame(
            {"price_da": np.arange(len(second_index), dtype=float) + 100.0},
            index=second_index,
        )
        calls = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices.csv"
            cached.to_csv(path)

            def fetcher(fetch_start, fetch_end):
                calls.append((fetch_start, fetch_end))
                return fresh

            result = load_or_fetch_frame(
                path,
                first_day,
                second_day,
                fetcher,
                timezone=timezone,
            )

        self.assertEqual(calls, [(second_day, second_day)])
        self.assertEqual(len(result), 192)
        self.assertFalse(result.isna().to_numpy().any())

    def test_operational_cache_can_return_unresolved_gap_without_losing_valid_cache(self):
        timezone = "Europe/Berlin"
        start = pd.Timestamp("2026-09-13", tz=timezone)
        end = pd.Timestamp("2026-09-13", tz=timezone)
        index = pd.date_range(
            start,
            pd.Timestamp("2026-09-14", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        cached = pd.DataFrame(
            {"price_da": np.arange(len(index), dtype=float)}, index=index
        )
        cached.loc[index[20], "price_da"] = np.nan
        fresh = cached.copy()
        fresh.loc[index[0], "price_da"] = np.nan
        fresh.loc[index[1], "price_da"] = 999.0

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices.csv"
            cached.to_csv(path)
            result = load_or_fetch_frame(
                path,
                start,
                end,
                lambda _start, _end: fresh,
                timezone=timezone,
                allow_incomplete=True,
            )

        self.assertEqual(result.loc[index[0], "price_da"], 0.0)
        self.assertEqual(result.loc[index[1], "price_da"], 999.0)
        self.assertTrue(pd.isna(result.loc[index[20], "price_da"]))

    def test_internal_cache_gap_triggers_refetch_and_is_repaired(self):
        timezone = "Europe/Berlin"
        start = pd.Timestamp("2026-09-10", tz=timezone)
        end = pd.Timestamp("2026-09-10", tz=timezone)
        index = pd.date_range(
            start,
            pd.Timestamp("2026-09-11", tz=timezone),
            freq="15min",
            inclusive="left",
        )
        complete = pd.DataFrame({"price_da": np.arange(len(index))}, index=index)
        complete.index.name = "timestamp"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prices.csv"
            complete.drop(index=index[20]).to_csv(path)
            fetch_calls = []

            def fetcher(fetch_start, fetch_end):
                fetch_calls.append((fetch_start, fetch_end))
                return complete

            result = load_or_fetch_frame(
                path,
                start,
                end,
                fetcher,
                timezone=timezone,
            )

        self.assertEqual(len(fetch_calls), 1)
        self.assertEqual(len(result), 96)
        self.assertFalse(result.isna().to_numpy().any())


if __name__ == "__main__":
    unittest.main()
