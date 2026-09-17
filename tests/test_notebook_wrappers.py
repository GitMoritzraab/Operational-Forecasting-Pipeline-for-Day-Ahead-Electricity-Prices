import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class NotebookWrapperTests(unittest.TestCase):
    def test_notebooks_are_valid_thin_guarded_wrappers(self):
        notebooks = {
            REPO_ROOT / "pipeline" / "lear" / "lear_pipeline.ipynb": (
                "pipeline.lear.run_lear",
                "RUN_OPERATIONAL = False",
            ),
            REPO_ROOT / "pipeline" / "sqra" / "sqra_pipeline.ipynb": (
                "pipeline.sqra.run_sqra",
                "RUN_EXPERIMENT = False",
            ),
            REPO_ROOT / "evaluation" / "evaluation.ipynb": (
                "evaluation.run_evaluation",
                "RUN_EVALUATION = False",
            ),
        }
        for path, (module_name, guard) in notebooks.items():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertLessEqual(len(notebook["cells"]), 8)
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
            )
            self.assertIn(module_name, source)
            self.assertIn(guard, source)
            self.assertNotIn("def ", source)
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
