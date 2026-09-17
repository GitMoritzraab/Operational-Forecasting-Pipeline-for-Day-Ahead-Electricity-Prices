from __future__ import annotations

import tempfile
import unittest
import bz2
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
    build_point_payload,
    build_quantile_payload,
    physical_mtu_numbers,
    resolve_targets,
    submission_record_path,
)
from pipeline.operational.runner import (
    _redacted_command,
    build_parser,
    main,
    pipeline_log_path,
    pipeline_lock_paths,
    resolve_arena_key,
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

    def test_default_dwd_plan_submits_c5_and_uses_all_clusters_for_sqra(self):
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

    def test_exaa_only_submits_d364_and_combines_three_training_windows(self):
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
            fundamental_locks = set(
                pipeline_lock_paths(fundamental, "default")
            )
            exaa_locks = set(
                pipeline_lock_paths(exaa_only, "exaa_only")
            )

        self.assertTrue(fundamental_locks)
        self.assertTrue(exaa_locks)
        self.assertTrue(fundamental_locks.isdisjoint(exaa_locks))

    def test_same_arena_account_retains_a_shared_payload_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fundamental = _config(root)
            exaa_only = _config(
                root,
                point_variant="exaa_only",
                sqra_variant="exaa_only",
            )
            fundamental_locks = set(
                pipeline_lock_paths(fundamental, "default")
            )
            exaa_locks = set(
                pipeline_lock_paths(exaa_only, "default")
            )

        self.assertEqual(
            fundamental_locks.intersection(exaa_locks),
            {root / "output" / "locks" / "arena_default.lock"},
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

    def test_cli_accepts_requested_exaa_and_arena_aliases(self):
        args = build_parser().parse_args(
            ["--exaa_only", "--energy-arena", "account-key", "--no-submit"]
        )
        self.assertTrue(args.exaa_only)
        self.assertEqual(args.arena_api_key, "account-key")
        self.assertTrue(args.no_submit)

    def test_cli_accepts_safe_previous_delivery_day_override(self):
        args = build_parser().parse_args(["--no-submit", "--d-1"])
        self.assertTrue(args.no_submit)
        self.assertTrue(args.delivery_day_minus_one)

    def test_previous_delivery_day_override_cannot_submit(self):
        with self.assertRaises(SystemExit) as error:
            main(["--d-1"])
        self.assertEqual(error.exception.code, 2)

    def test_named_arena_profile_does_not_expose_or_mix_keys(self):
        with patch.dict(
            "os.environ",
            {"ENERGY_ARENA_API_KEY_EXAA_ONLY": "profile-key"},
            clear=False,
        ):
            key, profile = resolve_arena_key("exaa-only", "")
        self.assertEqual(key, "profile-key")
        self.assertEqual(profile, "exaa-only")

    def test_command_log_redacts_explicit_arena_key(self):
        command = _redacted_command(
            ["--exaa_only", "--energy-arena", "very-secret-key"]
        )
        self.assertNotIn("very-secret-key", command)
        self.assertIn("--energy-arena <redacted>", command)
        equals_command = _redacted_command(
            ["--energy-arena-api-key=another-secret-key"]
        )
        self.assertNotIn("another-secret-key", equals_command)
        self.assertIn("--energy-arena-api-key=<redacted>", equals_command)


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
        quantile_payload = build_quantile_payload(
            quantile, _challenge("quantile", day)
        )

        self.assertEqual(len(point_payload["values"]), 96)
        self.assertEqual(np.asarray(quantile_payload["values"]).shape, (96, 5))
        self.assertEqual(quantile_payload["values"][0], [1, 2, 3, 4, 5])

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
                    weather={},
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
                    weather={},
                    force=False,
                )

        target_rows = forecast.loc[pd.Timestamp(days[-1].date())]
        self.assertEqual(len(target_rows), 96)
        self.assertTrue(np.isfinite(target_rows["y_pred"]).all())
        self.assertTrue(target_rows["y_true"].isna().all())

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


class MarketCacheTests(unittest.TestCase):
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
