from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation.evaluation_core import (
    EvaluationData,
    EvaluationResults,
    LABEL_MAP_POINT,
    LABEL_MAP_SQRA,
    QUANTILE_MODELS,
    aps_loss_matrix,
    check_index,
    check_missing,
    check_numeric,
    check_quantile_monotonicity,
    evaluate,
    load_evaluation_data,
    point_model_files,
    point_model_groups,
    quantile_model_files,
)
from evaluation.run_evaluation import (
    save_evaluation_figures,
    save_point_evaluation_metrics,
)


class EvaluationRegistryTests(unittest.TestCase):
    def test_legacy_registries_and_group_order_are_preserved(self):
        root = Path("results")
        self.assertEqual(len(point_model_files(root)), 28)
        self.assertEqual(len(quantile_model_files(root)), 6)
        fundamental, exaa, all_models = point_model_groups()
        self.assertEqual((len(fundamental), len(exaa), len(all_models)), (12, 16, 28))
        self.assertEqual(fundamental[0], "dwd_d56_c1_fundamental")
        self.assertEqual(exaa[0], "exaa_naive")

    def test_gw_display_names_match_the_manuscript_convention(self):
        _, _, point_models = point_model_groups()
        self.assertEqual(set(LABEL_MAP_POINT), set(point_models))
        self.assertEqual(
            LABEL_MAP_POINT["dwd_d56_c1_fundamental"],
            "Fundamental_DWD_d56_c1",
        )
        self.assertEqual(
            LABEL_MAP_POINT["era5_d364_c25_exaa"],
            "EXAA-Enriched_ERA5_d364_c25",
        )
        self.assertEqual(LABEL_MAP_POINT["exaa_naive"], "EXAA-Naive")
        self.assertEqual(LABEL_MAP_POINT["exaa_only_d112"], "EXAA-Only_d112")
        self.assertEqual(
            LABEL_MAP_SQRA["fund_DWD"],
            r"$\mathrm{SQRA}_{\mathrm{DWD,\,Fundamental}}$",
        )


class EvaluationCalculationTests(unittest.TestCase):
    @staticmethod
    def canonical_index(start: str, days: int) -> pd.MultiIndex:
        delivery_dates = pd.date_range(start, periods=days, freq="D")
        return pd.MultiIndex.from_product(
            [delivery_dates, range(1, 97)],
            names=["delivery_date", "mtu"],
        )

    @staticmethod
    def synthetic_data() -> EvaluationData:
        index = EvaluationCalculationTests.canonical_index("2025-01-01", 4)
        realised = np.linspace(-25.0, 150.0, len(index))
        fundamental, exaa, all_models = point_model_groups()
        point = pd.DataFrame({"y_true": realised}, index=index)
        for position, model in enumerate(all_models):
            point[model] = realised + (position + 1) * 0.1

        quantile_frame = pd.DataFrame({"y_true": realised}, index=index)
        offsets = (-8.0, -3.0, 0.0, 3.0, 8.0)
        for model_position, model in enumerate(QUANTILE_MODELS):
            for quantile_level, offset in zip(
                (0.10, 0.25, 0.50, 0.75, 0.90), offsets
            ):
                quantile_frame[f"{model}_q{quantile_level:.3f}"] = (
                    realised + offset + model_position * 0.05
                )
        return EvaluationData(point=point, quantile=quantile_frame)

    def test_complete_evaluation_contract(self):
        results = evaluate(self.synthetic_data())
        self.assertEqual(results.point_metrics.shape, (28, 3))
        self.assertEqual(results.point_gw_fundamental.shape, (12, 12))
        self.assertEqual(results.point_gw_exaa.shape, (16, 16))
        self.assertEqual(results.point_gw_all.shape, (28, 28))
        self.assertEqual(results.median_mae.shape, (6, 1))
        self.assertEqual(results.aps_summary.shape, (6, 1))
        self.assertEqual(set(results.coverage), {"PI_10_90", "PI_25_75"})
        self.assertEqual(results.aps_gw.shape, (6, 6))
        np.testing.assert_allclose(np.diag(results.point_gw_all), 1.0)

    def test_evaluation_backfills_point_metrics_without_model_execution(self):
        data = self.synthetic_data()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            files = point_model_files(root)
            first_path = next(iter(files.values()))
            first_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "period": "full",
                        "mae": 99.0,
                        "rmse": 99.0,
                        "bias": 99.0,
                        "n_obs": 1,
                        "n_inf_nan": 0,
                    },
                    {
                        "period": "evaluation",
                        "mae": 999.0,
                        "rmse": 999.0,
                        "bias": 999.0,
                        "n_obs": 1,
                        "n_inf_nan": 0,
                    },
                ]
            ).to_csv(first_path.parent / "metrics.csv", index=False)
            config = SimpleNamespace(results_root=root)

            save_point_evaluation_metrics(config, data)

            for model_name, forecast_path in files.items():
                metrics = pd.read_csv(forecast_path.parent / "metrics.csv")
                self.assertEqual(metrics.iloc[-1]["period"], "evaluation")
                self.assertEqual(
                    int((metrics["period"] == "evaluation").sum()), 1
                )
                self.assertEqual(
                    int(metrics.iloc[-1]["n_obs"]), len(data.point)
                )
                expected_mae = float(
                    np.abs(data.point[model_name] - data.point["y_true"]).mean()
                )
                self.assertAlmostEqual(metrics.iloc[-1]["mae"], expected_mae)

    def test_aps_matrix_retains_96_mtu_day_shape(self):
        data = self.synthetic_data()
        model = QUANTILE_MODELS[0]
        columns = {
            f"{model}_q{quantile:.3f}": f"q{quantile:.3f}"
            for quantile in (0.10, 0.25, 0.50, 0.75, 0.90)
        }
        frame = pd.concat(
            [data.quantile[["y_true"]], data.quantile[list(columns)].rename(columns=columns)],
            axis=1,
        )
        self.assertEqual(aps_loss_matrix(frame).shape, (4, 96))

    def test_loader_aligns_the_complete_file_registry(self):
        index = self.canonical_index("2025-01-01", 2)
        realised = np.linspace(10.0, 20.0, len(index))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for position, path in enumerate(point_model_files(root).values()):
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "y_pred": realised + position * 0.01,
                        "y_true": realised,
                    },
                    index=index,
                ).to_csv(path)
            for position, path in enumerate(quantile_model_files(root).values()):
                path.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.DataFrame({"y_true": realised}, index=index)
                for quantile_level, offset in zip(
                    (0.10, 0.25, 0.50, 0.75, 0.90),
                    (-4.0, -2.0, 0.0, 2.0, 4.0),
                ):
                    frame[f"q{quantile_level:.3f}"] = (
                        realised + offset + position * 0.01
                    )
                frame.to_csv(path)
            config = SimpleNamespace(
                timezone="Europe/Berlin",
                evaluation_start=date(2025, 1, 1),
                evaluation_end=date(2025, 1, 2),
                evaluation_skip_dates=(),
                results_root=root,
            )
            data = load_evaluation_data(config)
            self.assertEqual(data.point.shape, (192, 29))
            self.assertEqual(data.quantile.shape, (192, 31))
            self.assertEqual(
                data.point.index.names, ["delivery_date", "mtu"]
            )
            self.assertEqual(data.point.index.get_level_values("mtu").min(), 1)
            self.assertEqual(data.point.index.get_level_values("mtu").max(), 96)

    def test_exact_four_pdf_contract(self):
        models = ["a", "b"]
        square = pd.DataFrame([[1.0, 0.05], [0.95, 1.0]], index=models, columns=models)
        quantile_models = list(QUANTILE_MODELS)
        aps_square = pd.DataFrame(
            np.eye(6), index=quantile_models, columns=quantile_models
        )
        empty = pd.DataFrame()
        results = EvaluationResults(
            point_metrics=empty,
            point_gw_fundamental=square,
            point_gw_exaa=square,
            point_gw_all=square,
            median_mae=empty,
            aps_per_timestamp=empty,
            aps_summary=empty,
            coverage={},
            aps_gw=aps_square,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            save_evaluation_figures(results, output)
            self.assertEqual(
                {path.name for path in output.glob("*.pdf")},
                {
                    "gw_test_mae_fundamental.pdf",
                    "gw_test_mae_exaa.pdf",
                    "gw_test_mae_all.pdf",
                    "gw_test_aps.pdf",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in output.glob("*.pdf")))


class EvaluationValidationTests(unittest.TestCase):
    def test_exact_grid_accepts_normalized_spring_dst_day(self):
        start = pd.Timestamp("2026-03-28")
        end = pd.Timestamp("2026-03-30")
        index = pd.MultiIndex.from_product(
            [pd.date_range(start, end, freq="D"), range(1, 97)],
            names=["delivery_date", "mtu"],
        )
        frame = pd.DataFrame({"y_true": np.ones(len(index))}, index=index)
        check_index(frame, "DST", start, end, "15min", 96, set())
        self.assertEqual(len(frame), 3 * 96)

    def test_missing_canonical_mtu_is_rejected(self):
        start = pd.Timestamp("2026-03-29")
        index = pd.MultiIndex.from_product(
            [[start], range(1, 97)], names=["delivery_date", "mtu"]
        ).delete(8)
        frame = pd.DataFrame({"y_true": np.ones(len(index))}, index=index)
        with self.assertRaisesRegex(ValueError, "Missing=1"):
            check_index(frame, "MISSING", start, start, "15min", 96, set())

    def test_missing_non_numeric_and_crossing_quantiles_are_rejected(self):
        index = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
        with self.assertRaisesRegex(ValueError, "Missing values"):
            check_missing(pd.DataFrame({"value": [1.0, np.nan]}, index=index), "X")
        with self.assertRaisesRegex(TypeError, "Non-numeric"):
            check_numeric(pd.DataFrame({"value": ["a", "b"]}, index=index), ["value"], "X")

        frame = pd.DataFrame(index=index)
        for quantile, values in (
            (0.10, [1.0, 1.0]),
            (0.25, [2.0, 2.0]),
            (0.50, [3.0, 3.0]),
            (0.75, [4.0, 0.0]),
            (0.90, [5.0, 5.0]),
        ):
            frame[f"model_q{quantile:.3f}"] = values
        with self.assertRaisesRegex(ValueError, "Quantiles not monotone"):
            check_quantile_monotonicity(frame, "model")

    def test_duplicate_evaluation_keys_are_rejected(self):
        delivery_date = pd.Timestamp("2025-01-01")
        index = pd.MultiIndex.from_tuples(
            [(delivery_date, 1), (delivery_date, 1)],
            names=["delivery_date", "mtu"],
        )
        frame = pd.DataFrame(
            {"y_true": [1.0, 1.0]},
            index=index,
        )
        with self.assertRaisesRegex(ValueError, "duplicate delivery-date/MTU"):
            check_index(
                frame,
                "DUPLICATE",
                delivery_date,
                delivery_date,
                "15min",
                96,
                set(),
            )


if __name__ == "__main__":
    unittest.main()
