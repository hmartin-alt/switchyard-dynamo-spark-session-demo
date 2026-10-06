"""Offline checks for the retained large-data workflow."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "shopping_large", ROOT / "scripts/shopping-mmlu/tiered-large.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class ShoppingWorkflowTests(unittest.TestCase):
    def test_split_sizes(self):
        quotas = list(dataset.EASY_QUOTAS.values()) + list(dataset.HARD_QUOTAS.values())
        self.assertEqual(
            {split: sum(q[split] for q in quotas) for split in ("train", "validation", "test")},
            {"train": 3500, "validation": 500, "test": 500})

    def test_exposed_questions_stay_in_training(self):
        rows = [{"source_question": f"Question {i}", "example_id": str(i)} for i in range(12)]
        selected, _ = dataset.select_task(
            rows, "example", {"train": 4, "validation": 2, "test": 2}, {"question 0"})
        self.assertEqual(len({r["example_id"] for r in selected}), 8)
        self.assertTrue(any(r["previously_inspected_in_pilot"] for r in selected))
        self.assertFalse(any(r["previously_inspected_in_pilot"] for r in selected if r["split"] != "train"))

    def test_no_archived_script_dependencies(self):
        for path in (ROOT / "scripts/shopping-mmlu").glob("tiered-large*.py"):
            source = path.read_text()
            self.assertNotIn("from esci_common", source)
            compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
