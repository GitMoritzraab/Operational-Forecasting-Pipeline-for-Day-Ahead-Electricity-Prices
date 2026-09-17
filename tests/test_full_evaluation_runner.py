import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import run_full_evaluation as runner
from experiment_manifest import SQRA_RUNS, lear_runs


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation_start=date(2025, 12, 1),
        evaluation_end=date(2026, 7, 31),
        results_root=root / "results",
        output_root=root / "output",
        entsoe_price_cache_dir=root / "epex",
        exaa_price_cache_dir=root / "exaa",
        weather_training_start=date(2024, 10, 2),
        era5_year_dir=lambda clusters, year: root / "era5" / f"c{clusters}" / str(year),
    )


def _write_requirement(requirement: runner.CsvRequirement) -> None:
    requirement.path.parent.mkdir(parents=True, exist_ok=True)
    columns = set(requirement.required_columns)
    if "forecast_day" in columns:
        columns.add("runtime_seconds")
    leading = ["delivery_date", "mtu"] if requirement.canonical_index else []
    trailing = sorted(columns - set(leading))
    requirement.path.write_text(
        ",".join(leading + trailing) + "\n",
        encoding="utf-8",
    )


class FullEvaluationRunnerTests(unittest.TestCase):
    def test_evaluation_preflight_requires_all_34_forecasts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            requirements = runner._requirements(config, {"evaluation"})
            errors = runner._preflight(config, {"evaluation"})

        self.assertEqual(len(requirements), len(lear_runs()) + 1 + len(SQRA_RUNS))
        self.assertEqual(len(requirements), 34)
        self.assertEqual(len(errors), 34)
        self.assertTrue(all(message.startswith("Missing ") for message in errors))

    def test_full_preflight_accepts_forecasts_runtimes_and_price_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            requirements = runner._requirements(config, {"evaluation", "plots"})
            for requirement in requirements:
                _write_requirement(requirement)

            self.assertEqual(runner._preflight(config, {"evaluation", "plots"}), [])
            self.assertEqual(len(requirements), 69)

    def test_preflight_rejects_legacy_forecast_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            requirement = runner._point_forecast_requirements(config)[0]
            requirement.path.parent.mkdir(parents=True)
            requirement.path.write_text(
                "timestamp,y_pred,y_true\n",
                encoding="utf-8",
            )

            errors = runner._validate_csv(requirement)

        self.assertTrue(any("missing columns" in message for message in errors))
        self.assertTrue(any("first columns must be" in message for message in errors))

    def test_logged_command_streams_and_persists_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            runner._run_logged_command(
                [sys.executable, "-c", "print('evaluation-smoke')"],
                config,
                stage="tests",
                name="smoke",
            )

            log = (config.output_root / "logs" / "tests" / "smoke.log").read_text(
                encoding="utf-8"
            )
        self.assertIn("evaluation-smoke", log)
        self.assertIn("status=success", log)
        self.assertIn(f"cwd={runner.REPO_ROOT}", log)

    def test_logged_command_retains_failure_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            with self.assertRaises(subprocess.CalledProcessError):
                runner._run_logged_command(
                    [sys.executable, "-c", "print('failed-child'); raise SystemExit(9)"],
                    config,
                    stage="tests",
                    name="failure",
                )
            log = (config.output_root / "logs" / "tests" / "failure.log").read_text(
                encoding="utf-8"
            )
        self.assertIn("failed-child", log)
        self.assertIn("status=failed (9)", log)

    def test_default_order_is_evaluation_then_plots(self):
        config = _config(Path("unused"))
        events = []
        with (
            patch.object(runner, "load_experiment_config", return_value=config),
            patch.object(runner, "_execution_python", return_value=Path(sys.executable)),
            patch.object(runner, "_preflight", return_value=[]),
            patch.object(
                runner, "_run_evaluation", side_effect=lambda *_: events.append("evaluation")
            ),
            patch.object(runner, "_run_plots", side_effect=lambda *_: events.append("plots")),
            patch.object(runner, "_run_anc", side_effect=lambda *_: events.append("anc")),
        ):
            self.assertEqual(runner.main([]), 0)

        self.assertEqual(events, ["evaluation", "plots"])

    def test_anc_switch_runs_both_variants_before_postprocessing(self):
        config = _config(Path("unused"))
        events = []
        with (
            patch.object(runner, "load_experiment_config", return_value=config),
            patch.object(runner, "_execution_python", return_value=Path(sys.executable)),
            patch.object(runner, "_preflight", return_value=[]),
            patch.object(runner, "_run_anc", side_effect=lambda *_: events.append("anc")),
            patch.object(
                runner, "_run_evaluation", side_effect=lambda *_: events.append("evaluation")
            ),
            patch.object(runner, "_run_plots", side_effect=lambda *_: events.append("plots")),
        ):
            self.assertEqual(runner.main(["--anc"]), 0)

        self.assertEqual(events, ["anc", "evaluation", "plots"])

    def test_evaluation_only_selection_does_not_run_plots(self):
        config = _config(Path("unused"))
        events = []
        with (
            patch.object(runner, "load_experiment_config", return_value=config),
            patch.object(runner, "_execution_python", return_value=Path(sys.executable)),
            patch.object(runner, "_preflight", return_value=[]),
            patch.object(
                runner, "_run_evaluation", side_effect=lambda *_: events.append("evaluation")
            ),
            patch.object(runner, "_run_plots", side_effect=lambda *_: events.append("plots")),
        ):
            self.assertEqual(runner.main(["--stages", "evaluation"]), 0)

        self.assertEqual(events, ["evaluation"])

    def test_anc_dispatches_two_logged_modules(self):
        config = _config(Path("unused"))
        with patch.object(runner, "_run_module") as run_module:
            runner._run_anc(config)

        self.assertEqual(
            run_module.call_args_list,
            [
                call(
                    config,
                    "anc",
                    "fundamental",
                    "pipeline.lear.run_lear",
                    "anc",
                    "--variant",
                    "fundamental",
                ),
                call(
                    config,
                    "anc",
                    "exaa",
                    "pipeline.lear.run_lear",
                    "anc",
                    "--variant",
                    "exaa",
                ),
            ],
        )

    def test_new_runner_has_no_notebook_dependency(self):
        source = Path(runner.__file__).read_text(encoding="utf-8").lower()
        for forbidden in (
            ".ipynb",
            "jupyter",
            "nbconvert",
            "ipykernel",
            "executed_notebooks",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
