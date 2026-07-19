import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import recomart
from src.evaluation import evaluate_popularity
from src.ingestion.item_property_csv import read_item_properties
from src.modeling import evaluate_models, prepare_model_data, profile_gold, train_models
from src.pipelines.gold import build_gold


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.db = root / "test.db"
        self.write("events.csv", ["timestamp", "visitorid", "event", "itemid", "transactionid"], [
            [1000, 1, "view", 10, ""], [2000, 1, "addtocart", 10, ""],
            [3000, 1, "transaction", 10, 99], [4000, 2, "invalid", 10, ""],
        ])
        props = [[1000, 10, "categoryid", "7"], [1000, 10, "available", "0"],
                 [2000, 10, "available", "1"], [1000, 10, "400", "n1"]]
        self.write("item_properties_part1.csv", ["timestamp", "itemid", "property", "value"], props)
        self.write("item_properties_part2.csv", ["timestamp", "itemid", "property", "value"], [])
        self.write("category_tree.csv", ["categoryid", "parentid"], [[7, 3]])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, header, rows):
        with (self.raw / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh); writer.writerow(header); writer.writerows(rows)

    def test_end_to_end(self):
        with patch.object(recomart, "RAW", self.raw):
            recomart.ingest_categories(self.db)
            recomart.replay_events(self.db)
            server = recomart.make_server("127.0.0.1", 0)
            import threading
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                recomart.ingest_products(self.db, f"http://127.0.0.1:{server.server_address[1]}", 2)
            finally:
                server.shutdown(); server.server_close(); thread.join()
            recomart.transform(self.db, 16)
            report = recomart.validate(self.db)
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["silver_user_events"], 3)
        db = recomart.connect(self.db)
        self.assertEqual(db.execute("SELECT available FROM silver_products").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT interaction_score FROM gold_user_item_features").fetchone()[0], 9)
        db.close()

    def test_property_reader_preserves_quoted_and_unquoted_commas(self):
        path = self.raw / "comma_values.csv"
        path.write_text(
            'timestamp,itemid,property,value\n'
            '1000,10,description,"red, large"\n'
            '2000,11,description,blue,small,sale\n',
            encoding="utf-8",
        )
        rows = list(read_item_properties(path))
        self.assertEqual(rows[0]["value"], "red, large")
        self.assertEqual(rows[1]["value"], "blue,small,sale")

    def test_time_based_popularity_evaluation(self):
        db = recomart.connect(self.db)
        db.executescript("""
            CREATE TABLE silver_user_events (
                event_timestamp TEXT, visitor_id INTEGER, item_id INTEGER,
                event_type TEXT, transaction_id INTEGER,
                event_timestamp_ms INTEGER
            );
            CREATE TABLE silver_products (
                item_id INTEGER PRIMARY KEY, category_id INTEGER,
                available INTEGER, encoded_properties TEXT
            );
            INSERT INTO silver_products VALUES
                (10, 1, 1, '[]'), (20, 1, 1, '[]'), (30, 1, 0, '[]');
            INSERT INTO silver_user_events VALUES
                ('1970-01-01', 1, 10, 'view', NULL, 1000),
                ('1970-01-01', 2, 20, 'transaction', 1, 2000),
                ('1970-01-01', 3, 20, 'view', NULL, 3000),
                ('1970-01-01', 1, 20, 'transaction', 2, 9000);
        """)
        db.commit()
        db.close()
        report = evaluate_popularity(
            self.db, k=1, target="transaction", cutoff_ms=5000
        )
        self.assertEqual(report["eligible_users"], 1)
        self.assertEqual(report["metrics"]["precision@1"], 1.0)
        self.assertEqual(report["metrics"]["recall@1"], 1.0)

    def test_profile_split_train_and_compare_models(self):
        model_db = Path(self.tmp.name) / "models.db"
        db = recomart.connect(model_db)
        db.executescript("""
            CREATE TABLE silver_user_events (
                event_timestamp TEXT,visitor_id INTEGER,item_id INTEGER,
                event_type TEXT,transaction_id INTEGER,event_timestamp_ms INTEGER
            );
            CREATE TABLE silver_products (
                item_id INTEGER PRIMARY KEY,category_id INTEGER,
                available INTEGER,encoded_properties TEXT
            );
            CREATE TABLE silver_category_hierarchy (
                category_id INTEGER,parent_category_id INTEGER
            );
            INSERT INTO silver_category_hierarchy VALUES (1,NULL);
            INSERT INTO silver_products VALUES
                (10,1,1,'[]'),(20,1,1,'[]'),(30,1,1,'[]');
            INSERT INTO silver_user_events VALUES
                ('1970-01-01',1,10,'view',NULL,1000),
                ('1970-01-01',1,20,'transaction',1,1100),
                ('1970-01-01',2,10,'view',NULL,1200),
                ('1970-01-01',2,20,'transaction',2,1300),
                ('1970-01-01',3,10,'view',NULL,1400),
                ('1970-01-01',4,30,'transaction',3,1500),
                ('1970-01-01',4,30,'transaction',4,1600),
                ('1970-01-01',4,30,'transaction',5,1700),
                ('1970-01-01',3,20,'transaction',6,9000);
        """)
        db.commit()
        build_gold(db, 16)
        db.close()

        profile = profile_gold(model_db, 2)
        self.assertEqual(profile["size"]["users"], 4)
        split = prepare_model_data(
            model_db, target="transaction", cutoff_ms=5000
        )
        self.assertEqual(split["test"]["eligible_users"], 1)
        trained = train_models(
            model_db, max_history=10, min_cooccurrence=1, neighbors=5
        )
        self.assertGreater(
            trained["collaborative_filtering"]["stored_directed_similarities"], 0
        )
        report = evaluate_models(model_db, k=1)
        self.assertEqual(
            report["models"]["weighted_popularity"]["precision@1"], 0.0
        )
        self.assertEqual(
            report["models"]["item_collaborative_filtering"]["precision@1"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
