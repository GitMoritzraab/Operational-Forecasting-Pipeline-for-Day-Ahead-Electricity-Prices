import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import run_full_experiment as runner
from experiment_manifest import SQRA_RUNS, lear_runs


def _main_config(root=Path("unused")):
    return SimpleNamespace(
        evaluation_start=date(2025, 12, 1),
        evaluation_end=date(2026, 7, 31),
        point_forecast_start=date(2025, 10, 1),
        input_start=date(2024, 9, 25),
        sqra_mtu_specific=False,
        results_root=root,
    )


class RunnerSelectionTests(unittest.TestCase):
    def test_source_categories_match_manifest(self):
        era5 = runner._select_lear_runs({"era5"})
        dwd = runner._select_lear_runs({"dwd"})
        exaa = runner._select_lear_runs({"exaa"})

        self.assertEqual(len(era5), 18)
        self.assertEqual(len(dwd), 6)
        self.assertEqual(len(exaa), 3)
        self.assertTrue(all(run.weather_source == "ERA5" for run in era5))
        self.assertTrue(all(run.weather_source == "DWD" for run in dwd))
        self.assertTrue(all(run.use_exaa_only for run in exaa))

    def test_training_window_filters_fitted_runs_only(self):
        era5_d56 = runner._select_lear_runs({"era5"}, 56)
        all_d112 = runner._select_lear_runs(set(runner.POINT_SOURCES), 112)

        self.assertEqual(len(era5_d56), 6)
        self.assertTrue(all(run.train_days == 56 for run in era5_d56))
        # Six ERA5 variants plus one EXAA-only run; DWD has no D=112 run.
        self.assertEqual(len(all_d112), 7)
        self.assertTrue(all(run.train_days == 112 for run in all_d112))

    def test_cluster_and_variant_filters_can_select_one_era5_run(self):
        both = runner._select_lear_runs({"era5"}, 364, cluster=1)
        fundamental = runner._select_lear_runs(
            {"era5"}, 364, cluster=1, fundamental=True
        )
        exaa = runner._select_lear_runs(
            {"era5"}, 364, cluster=1, fundamental=False
        )

        self.assertEqual(
            tuple(run.name for run in both),
            ("era5_d364_c1_fundamental", "era5_d364_c1_exaa"),
        )
        self.assertEqual(
            tuple(run.name for run in fundamental),
            ("era5_d364_c1_fundamental",),
        )
        self.assertEqual(
            tuple(run.name for run in exaa),
            ("era5_d364_c1_exaa",),
        )

    def test_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown point-forecast sources"):
            runner._select_lear_runs({"unknown"})


class RunnerExecutionTests(unittest.TestCase):
    def test_module_command_uses_selected_python(self):
        config = SimpleNamespace(output_root=Path("unused"))
        selected_python = Path("forecast") / "Scripts" / "python.exe"
        with (
            patch.object(runner, "_execution_python", return_value=selected_python),
            patch.object(runner, "_run_logged_command") as run_logged,
        ):
            runner._run_module(
                config,
                "sqra",
                "era5_fundamental",
                "pipeline.sqra.run_sqra",
                "--config",
                "era5_fundamental",
            )

        command = run_logged.call_args.args[0]
        self.assertEqual(
            command,
            [
                selected_python,
                "-m",
                "pipeline.sqra.run_sqra",
                "--config",
                "era5_fundamental",
            ],
        )
        self.assertEqual(run_logged.call_args.kwargs["stage"], "sqra")

    def test_logged_command_streams_and_persists_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            config = SimpleNamespace(output_root=output_root)
            runner._run_logged_command(
                [sys.executable, "-c", "print('pipeline-smoke')"],
                config,
                stage="tests",
                name="smoke",
            )

            log = (output_root / "logs" / "tests" / "smoke.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("pipeline-smoke", log)
            self.assertIn("status=success", log)
            self.assertIn(f"cwd={runner.REPO_ROOT}", log)

    def test_logged_command_preserves_failure_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            config = SimpleNamespace(output_root=output_root)
            with self.assertRaises(subprocess.CalledProcessError):
                runner._run_logged_command(
                    [sys.executable, "-c", "print('before-failure'); raise SystemExit(7)"],
                    config,
                    stage="tests",
                    name="failure",
                )

            log = (output_root / "logs" / "tests" / "failure.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("before-failure", log)
            self.assertIn("status=failed (7)", log)

    def test_selected_lear_batch_runs_baseline_then_manifest_entries(self):
        config = SimpleNamespace()
        runs = runner._select_lear_runs({"exaa"}, 56)
        with (
            patch.object(runner, "_exaa_naive_result_complete", return_value=False),
            patch.object(runner, "_result_complete", return_value=False),
            patch.object(runner, "_run_module") as run_module,
        ):
            runner._run_lear(
                config,
                force=False,
                runs=runs,
                include_exaa_naive=True,
            )

        self.assertEqual(run_module.call_count, 2)
        self.assertEqual(
            run_module.call_args_list[0],
            call(
                config,
                "lear",
                "exaa_naive",
                "pipeline.lear.run_lear",
                "exaa-naive",
            ),
        )
        self.assertEqual(
            run_module.call_args_list[1],
            call(
                config,
                "lear",
                "exaa_only_d56",
                "pipeline.lear.run_lear",
                "operational",
                "--config",
                "exaa_only_d56",
            ),
        )

    def test_complete_baseline_and_fitted_run_are_resumed(self):
        config = SimpleNamespace()
        runs = runner._select_lear_runs({"era5"}, 56)[:1]
        with (
            patch.object(runner, "_exaa_naive_result_complete", return_value=True),
            patch.object(runner, "_result_complete", return_value=True),
            patch.object(runner, "_run_module") as run_module,
        ):
            runner._run_lear(
                config,
                force=False,
                runs=runs,
                include_exaa_naive=True,
            )
        run_module.assert_not_called()

    def test_force_reruns_complete_baseline_and_fitted_run(self):
        config = SimpleNamespace()
        runs = runner._select_lear_runs({"era5"}, 56)[:1]
        with (
            patch.object(runner, "_exaa_naive_result_complete", return_value=True),
            patch.object(runner, "_result_complete", return_value=True),
            patch.object(runner, "_run_module") as run_module,
        ):
            runner._run_lear(
                config,
                force=True,
                runs=runs,
                include_exaa_naive=True,
            )
        self.assertEqual(run_module.call_count, 2)

    def _run_main(self, arguments):
        config = _main_config()
        events = []
        with (
            patch.object(sys, "argv", ["run_full_experiment.py", *arguments]),
            patch.object(runner, "load_experiment_config", return_value=config),
            patch.object(runner, "_execution_python", return_value=Path(sys.executable)),
            patch.object(runner, "_preflight", return_value=[]),
            patch.object(
                runner,
                "_run_lear",
                side_effect=lambda config, force, runs, **kwargs: events.append(
                    ("point", tuple(run.name for run in runs), kwargs)
                ),
            ),
            patch.object(
                runner, "_run_sqra", side_effect=lambda *_: events.append(("sqra",))
            ),
        ):
            return runner.main(), events

    def test_default_runs_all_point_forecasts_only(self):
        status, events = self._run_main([])

        self.assertEqual(status, 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "point")
        self.assertEqual(len(events[0][1]), len(lear_runs()))
        self.assertTrue(events[0][2]["include_exaa_naive"])

    def test_sqra_alone_runs_only_sqra(self):
        status, events = self._run_main(["--sqra"])

        self.assertEqual(status, 0)
        self.assertEqual(events, [("sqra",)])

    def test_era5_d56_runs_six_point_forecasts(self):
        status, events = self._run_main(["--era5", "--d", "56"])

        self.assertEqual(status, 0)
        self.assertEqual(len(events[0][1]), 6)
        self.assertTrue(all("era5_d56" in name for name in events[0][1]))
        self.assertFalse(events[0][2]["include_exaa_naive"])

    def test_era5_d364_cluster_and_fundamental_select_one_run(self):
        status, events = self._run_main(
            [
                "--era5",
                "--d",
                "364",
                "--cluster",
                "5",
                "--fundamental",
                "true",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(events[0][1], ("era5_d364_c5_fundamental",))

    def test_era5_d364_cluster_without_variant_selects_both(self):
        status, events = self._run_main(
            ["--era5", "--d", "364", "--cluster", "25"]
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            events[0][1],
            ("era5_d364_c25_fundamental", "era5_d364_c25_exaa"),
        )

    def test_false_fundamental_selects_exaa_enriched(self):
        status, events = self._run_main(
            [
                "--era5",
                "--d",
                "364",
                "--cluster",
                "1",
                "--fundamental",
                "false",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(events[0][1], ("era5_d364_c1_exaa",))

    def test_weather_filters_without_weather_source_are_rejected(self):
        with patch.object(
            sys,
            "argv",
            ["run_full_experiment.py", "--cluster", "1"],
        ):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_d56_without_source_filters_all_source_categories(self):
        status, events = self._run_main(["--d", "56"])

        self.assertEqual(status, 0)
        # Six ERA5 + six DWD + one EXAA-only fitted run, plus EXAA-naive.
        self.assertEqual(len(events[0][1]), 13)
        self.assertTrue(events[0][2]["include_exaa_naive"])
        self.assertTrue(all("d56" in name for name in events[0][1]))

    def test_exaa_d112_includes_naive_and_one_fitted_run(self):
        status, events = self._run_main(["--exaa", "--d", "112"])

        self.assertEqual(status, 0)
        self.assertEqual(events[0][1], ("exaa_only_d112",))
        self.assertTrue(events[0][2]["include_exaa_naive"])

    def test_source_then_sqra_preserves_execution_order(self):
        status, events = self._run_main(["--dwd", "--sqra"])

        self.assertEqual(status, 0)
        self.assertEqual([event[0] for event in events], ["point", "sqra"])
        self.assertEqual(len(events[0][1]), 6)

    def test_d_filter_with_sqra_alone_is_rejected(self):
        with patch.object(
            sys, "argv", ["run_full_experiment.py", "--sqra", "--d", "56"]
        ):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_dwd_has_no_d112_configuration(self):
        with (
            patch.object(
                sys, "argv", ["run_full_experiment.py", "--dwd", "--d", "112"]
            ),
            patch.object(runner, "load_experiment_config", return_value=_main_config()),
        ):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_complete_lear_result_is_resumed(self):
        run = lear_runs()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SimpleNamespace(
                results_root=root,
                point_forecast_start=date(2025, 10, 1),
                evaluation_end=date(2026, 7, 31),
                lear_use_vst=True,
            )
            output = run.output_dir(root)
            output.mkdir(parents=True)
            (output / "forecast.csv").write_text(
                "delivery_date,mtu,y_pred,y_true\n", encoding="utf-8"
            )
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "test_start": str(config.point_forecast_start),
                        "test_end": str(config.evaluation_end),
                        "train_days": run.train_days,
                        "use_vst": True,
                        "forecast_schema_version": 2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(runner._result_complete(run, config))

            (output / "forecast.csv").write_text(
                "timestamp,y_pred,y_true\n", encoding="utf-8"
            )
            self.assertFalse(runner._result_complete(run, config))

    def test_complete_exaa_naive_result_is_resumed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SimpleNamespace(
                results_root=root,
                point_forecast_start=date(2025, 10, 1),
                evaluation_end=date(2026, 7, 31),
            )
            output = root / "lear_op_results" / "exaa_naive"
            output.mkdir(parents=True)
            (output / "forecast.csv").write_text(
                "delivery_date,mtu,y_pred,y_true\n", encoding="utf-8"
            )
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "test_start": str(config.point_forecast_start),
                        "test_end": str(config.evaluation_end),
                        "forecast_schema_version": 2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(runner._exaa_naive_result_complete(config))

            config.evaluation_end = date(2026, 7, 30)
            self.assertFalse(runner._exaa_naive_result_complete(config))

    def test_sqra_preflight_checks_member_results_not_weather_or_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(results_root=Path(temp_dir))
            with patch.dict(os.environ, {}, clear=True):
                errors = runner._preflight(
                    config,
                    (),
                    include_exaa_naive=False,
                    run_sqra=True,
                )

        expected = {
            path
            for sqra_run in SQRA_RUNS.values()
            for path in sqra_run.inputs(config.results_root)
        }
        self.assertEqual(len(errors), len(expected))
        self.assertTrue(all(error.startswith("Missing SQRA point forecast:") for error in errors))
        self.assertFalse(any("ENTSOE_API_KEY" in error for error in errors))

    def test_sqra_preflight_rejects_legacy_timestamp_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(results_root=Path(temp_dir))
            inputs = sorted(
                {
                    path
                    for sqra_run in SQRA_RUNS.values()
                    for path in sqra_run.inputs(config.results_root)
                },
                key=str,
            )
            for path in inputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "delivery_date,mtu,y_pred,y_true\n",
                    encoding="utf-8",
                )
            inputs[0].write_text(
                "timestamp,y_pred,y_true\n",
                encoding="utf-8",
            )

            errors = runner._preflight(
                config,
                (),
                include_exaa_naive=False,
                run_sqra=True,
            )

        self.assertEqual(
            errors,
            [f"Incompatible SQRA point forecast schema: {inputs[0]}"],
        )

    def test_sqra_preflight_allows_point_outputs_scheduled_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(results_root=Path(temp_dir))
            exaa_runs = runner._select_lear_runs({"exaa"})
            scheduled = {
                run.output_dir(config.results_root) / "forecast.csv"
                for run in exaa_runs
            }
            scheduled.add(
                config.results_root
                / "lear_op_results"
                / "exaa_naive"
                / "forecast.csv"
            )
            all_inputs = {
                path
                for sqra_run in SQRA_RUNS.values()
                for path in sqra_run.inputs(config.results_root)
            }
            with patch.dict(
                os.environ,
                {"ENTSOE_API_KEY": "test-key"},
                clear=True,
            ):
                errors = runner._preflight(
                    config,
                    exaa_runs,
                    include_exaa_naive=True,
                    run_sqra=True,
                )

        self.assertEqual(len(errors), len(all_inputs - scheduled))
        self.assertFalse(any(str(path) in "\n".join(errors) for path in scheduled))

    def test_exaa_preflight_does_not_check_weather(self):
        config = SimpleNamespace()
        with patch.dict(os.environ, {"ENTSOE_API_KEY": "test-key"}, clear=True):
            errors = runner._preflight(
                config,
                runner._select_lear_runs({"exaa"}),
                include_exaa_naive=True,
                run_sqra=False,
            )
        self.assertEqual(errors, [])

    def test_automated_stages_have_no_notebook_runtime_dependency(self):
        automated_files = (
            Path(runner.__file__),
            runner.REPO_ROOT / "pipeline" / "lear" / "run_lear.py",
            runner.REPO_ROOT / "pipeline" / "sqra" / "run_sqra.py",
            runner.REPO_ROOT / "evaluation" / "run_evaluation.py",
        )
        forbidden_terms = (
            ".ipynb",
            "jupyter",
            "nbconvert",
            "ipykernel",
            "kernel_name",
            "executed_notebooks",
        )
        for path in automated_files:
            source = path.read_text(encoding="utf-8").lower()
            for forbidden in forbidden_terms:
                self.assertNotIn(forbidden, source, f"{forbidden!r} in {path}")


if __name__ == "__main__":
    unittest.main()
