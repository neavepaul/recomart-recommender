import tempfile
import unittest
from pathlib import Path

from src.core import connect
from src.metadata import (
    finish_pipeline_run, latest_lineage, record_dataset, start_pipeline_run,
)
from src.model_tracking import numeric_metrics


class MetadataTests(unittest.TestCase):
    def test_pipeline_and_dataset_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "metadata.db"
            source = root / "source.csv"
            source.write_text("id\n1\n", encoding="utf-8")
            db = connect(db_path)
            db.execute("CREATE TABLE bronze_example(id INTEGER)")
            db.execute("INSERT INTO bronze_example VALUES (1)")
            db.commit()
            db.close()

            start_pipeline_run(db_path, "run-1", "test-flow", {"limit": 1})
            record_dataset(
                db_path, "run-1", "bronze_example", "bronze",
                "Load example CSV without transformation.", [],
                "source.csv", source,
            )
            finish_pipeline_run(db_path, "run-1", "COMPLETED")

            lineage = latest_lineage(db_path)
            self.assertEqual(lineage[0]["row_count"], 1)
            self.assertTrue(lineage[0]["source_version"].startswith("sha256:"))
            db = connect(db_path)
            self.assertEqual(
                db.execute(
                    "SELECT status FROM metadata_pipeline_runs WHERE run_id='run-1'"
                ).fetchone()[0],
                "COMPLETED",
            )
            db.close()

    def test_mlflow_metric_names_are_flattened_and_valid(self):
        metrics = numeric_metrics({"precision@10": 0.2, "warm": {"users": 3}})
        self.assertEqual(metrics, {"precision_at_10": 0.2, "warm.users": 3.0})


if __name__ == "__main__":
    unittest.main()
