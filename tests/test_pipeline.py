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

    def test_build_features_produces_feature_tables(self):
        from src.pipelines.features import build_features

        features_db = Path(self.tmp.name) / "features.db"
        db = recomart.connect(features_db)
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
                ('1970-01-01',2,20,'transaction',2,1300);
        """)
        db.commit()
        build_gold(db, 16)
        build_features(db, neighbors=5, min_cooccurrence=1, max_history=10)

        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM feature_user_activity").fetchone()[0], 2
        )
        self.assertEqual(
            db.execute(
                "SELECT avg_interaction_score FROM feature_user_activity WHERE visitor_id=1"
            ).fetchone()[0], 3.0
        )
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM feature_item_popularity").fetchone()[0],
            db.execute("SELECT COUNT(*) FROM gold_item_features").fetchone()[0],
        )
        self.assertEqual(
            db.execute(
                "SELECT item_id FROM feature_item_popularity WHERE popularity_rank=1"
            ).fetchone()[0], 20
        )
        self.assertEqual(
            db.execute(
                "SELECT avg_interaction_score FROM feature_item_popularity WHERE item_id=20"
            ).fetchone()[0], 5.0
        )
        neighbours = db.execute(
            "SELECT source_item_id,similar_item_id FROM feature_item_cooccurrence "
            "ORDER BY source_item_id"
        ).fetchall()
        db.close()
        self.assertIn((10, 20), neighbours)
        self.assertIn((20, 10), neighbours)

    def test_generate_eda_plots_writes_files(self):
        from src.modeling import generate_eda_plots
        plots_db = Path(self.tmp.name) / "plots.db"
        db = recomart.connect(plots_db)
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
                ('1970-01-01',3,10,'view',NULL,1400);
        """)
        db.commit()
        build_gold(db, 16)
        db.close()

        out_dir = Path(self.tmp.name) / "eda"
        result = generate_eda_plots(plots_db, out_dir=out_dir, top_n=2)
        self.assertEqual(result["output_dir"], str(out_dir))
        self.assertTrue(result["plots"])
        for path in result["plots"].values():
            self.assertTrue(Path(path).exists())


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "features.db"
        db = recomart.connect(self.db)
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
                ('1970-01-01',2,20,'transaction',2,1300);
        """)
        db.commit()
        build_gold(db, 16)
        from src.pipelines.features import build_features
        build_features(db, neighbors=5, min_cooccurrence=1, max_history=10)
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_register_creates_registry_and_manifest(self):
        from src import feature_store

        manifest_path = Path(self.tmp.name) / "manifest.json"
        manifest = feature_store.register_features(
            self.db, run_id="run-1", manifest_path=manifest_path
        )
        self.assertEqual(manifest["feature_version"], "run-1")
        self.assertTrue(manifest_path.exists())
        registry = feature_store.list_registry(self.db)
        self.assertEqual(
            {r["feature_view"] for r in registry},
            {"user_activity", "item_popularity", "item_cooccurrence"},
        )
        popularity = next(
            r for r in registry if r["feature_view"] == "item_popularity"
        )
        self.assertEqual(popularity["feature_version"], "run-1")
        self.assertIn("conversion_rate", popularity["feature_columns"])
        self.assertNotIn("item_id", popularity["feature_columns"])

    def test_online_retrieval_returns_latest_version(self):
        from src import feature_store

        feature_store.register_features(self.db, run_id="run-1", manifest_path=None)
        feature_store.register_features(self.db, run_id="run-2", manifest_path=None)
        online = feature_store.get_online_features(
            self.db, "item_popularity", [20]
        )
        self.assertEqual(online["feature_version"], "run-2")
        self.assertEqual(online["mode"], "inference")
        self.assertEqual(len(online["rows"]), 1)
        self.assertEqual(online["rows"][0]["item_id"], 20)

    def test_training_retrieval_uses_requested_snapshot(self):
        from src import feature_store

        feature_store.register_features(self.db, run_id="run-1", manifest_path=None)
        feature_store.register_features(self.db, run_id="run-2", manifest_path=None)
        training = feature_store.get_training_features(
            self.db, "user_activity", [1], version="run-1"
        )
        self.assertEqual(training["feature_version"], "run-1")
        self.assertEqual(training["mode"], "training")
        self.assertEqual(training["rows"][0]["visitor_id"], 1)

    def test_retention_prunes_old_versions(self):
        from src import feature_store

        for index in range(4):
            feature_store.register_features(
                self.db, run_id=f"run-{index}", retention=2, manifest_path=None
            )
        db = recomart.connect(self.db)
        versions = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT feature_version FROM feature_user_activity_versions"
            )
        }
        registry_versions = {
            row[0]
            for row in db.execute(
                "SELECT feature_version FROM feature_registry "
                "WHERE feature_view='user_activity'"
            )
        }
        db.close()
        self.assertEqual(versions, {"run-2", "run-3"})
        self.assertEqual(registry_versions, {"run-2", "run-3"})

    def test_registered_versions_are_immutable(self):
        from src import feature_store

        feature_store.register_features(self.db, run_id="run-1", manifest_path=None)
        db = recomart.connect(self.db)
        db.execute(
            "UPDATE feature_user_activity SET total_events=999 WHERE visitor_id=1"
        )
        db.commit()
        db.close()

        with self.assertRaisesRegex(RuntimeError, "already registered"):
            feature_store.register_features(
                self.db, run_id="run-1", manifest_path=None
            )
        training = feature_store.get_training_features(
            self.db, "user_activity", [1], version="run-1"
        )
        self.assertNotEqual(training["rows"][0]["total_events"], 999)

    def test_unknown_version_raises(self):
        from src import feature_store

        feature_store.register_features(self.db, run_id="run-1", manifest_path=None)
        with self.assertRaises(RuntimeError):
            feature_store.get_training_features(
                self.db, "item_popularity", [10], version="missing"
            )


if __name__ == "__main__":
    unittest.main()
