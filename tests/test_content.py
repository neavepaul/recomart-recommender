import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.core import connect
from src.modeling.content import build_content_model
from src.modeling.evaluate import evaluate_models
from src.modeling.inference import recommend
from src.modeling.tune import tune_hybrid


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
    "Content-model dependencies are not installed",
)
class ContentModelTests(unittest.TestCase):
    def test_build_recommend_and_hybrid_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "content.db"
            db = connect(db_path)
            db.executescript("""
                CREATE TABLE gold_item_features (
                    item_id INTEGER PRIMARY KEY,category_id INTEGER,
                    parent_category_id INTEGER,available INTEGER,
                    item_feature_vector TEXT NOT NULL
                );
                INSERT INTO gold_item_features VALUES
                    (10,1,100,1,'{"0":1.0,"1":1.0}'),
                    (20,1,100,1,'{"0":1.0,"1":1.0}'),
                    (30,2,200,1,'{"2":1.0}');
                CREATE TABLE gold_user_item_features (
                    visitor_id INTEGER,item_id INTEGER,view_count INTEGER,
                    cart_count INTEGER,purchase_count INTEGER,
                    interaction_score INTEGER,last_interaction_timestamp TEXT
                );
                INSERT INTO gold_user_item_features VALUES
                    (1,10,1,0,0,1,'1970-01-01 00:00:01');
                CREATE TABLE model_split_metadata (
                    cutoff_ms INTEGER,target TEXT,minimum_event_ms INTEGER,
                    maximum_event_ms INTEGER,created_at TEXT
                );
                INSERT INTO model_split_metadata VALUES
                    (5000,'transaction',1000,9000,CURRENT_TIMESTAMP);
                CREATE TABLE model_train_user_items (
                    visitor_id INTEGER,item_id INTEGER,view_count INTEGER,
                    cart_count INTEGER,purchase_count INTEGER,
                    interaction_score INTEGER,last_interaction_timestamp_ms INTEGER
                );
                INSERT INTO model_train_user_items VALUES
                    (1,10,1,0,0,1,1000);
                CREATE TABLE model_test_targets (
                    visitor_id INTEGER,item_id INTEGER,target_events INTEGER
                );
                INSERT INTO model_test_targets VALUES (1,20,1);
                CREATE TABLE silver_products (
                    item_id INTEGER PRIMARY KEY,category_id INTEGER,
                    available INTEGER,encoded_properties TEXT
                );
                INSERT INTO silver_products VALUES
                    (10,1,1,'[]'),(20,1,1,'[]'),(30,2,1,'[]');
                CREATE TABLE silver_user_events (
                    event_timestamp TEXT,visitor_id INTEGER,item_id INTEGER,
                    event_type TEXT,transaction_id INTEGER,event_timestamp_ms INTEGER
                );
                INSERT INTO silver_user_events VALUES
                    ('1970-01-01',1,10,'view',NULL,1000),
                    ('1970-01-01',2,10,'view',NULL,1100),
                    ('1970-01-01',2,20,'transaction',1,1200),
                    ('1970-01-01',1,20,'transaction',2,4000);
                CREATE TABLE model_popularity(item_id INTEGER,score REAL,rank INTEGER);
                INSERT INTO model_popularity VALUES (30,10,1),(20,5,2),(10,1,3);
                CREATE TABLE model_item_similarity (
                    source_item_id INTEGER,similar_item_id INTEGER,
                    similarity REAL,cooccurring_users INTEGER,neighbor_rank INTEGER
                );
            """)
            db.commit()
            db.close()

            model_dir = root / "model"
            built = build_content_model(db_path, model_dir=model_dir, vector_size=4)
            self.assertEqual(built["items"], 3)
            self.assertTrue((model_dir / "features.npz").exists())

            tuned = tune_hybrid(
                db_path, content_model_dir=model_dir,
                validation_cutoff_ms=3000, k=1, max_history=10,
                min_cooccurrence=1, neighbors=5,
            )
            self.assertEqual(tuned["warm_validation_users"], 1)
            self.assertEqual(tuned["configurations_tested"], 55)
            db = connect(db_path)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM model_hybrid_config").fetchone()[0],
                1,
            )
            db.close()

            report = evaluate_models(
                db_path, k=1, content_model_dir=model_dir
            )
            self.assertEqual(
                report["models"]["content_similarity"]["precision@1"], 1.0
            )
            self.assertEqual(
                report["models"]["item_cf_content_hybrid"]["precision@1"], 1.0
            )
            served = recommend(
                db_path, visitor_id=1, k=1, content_model_dir=model_dir
            )
            self.assertEqual(served["recommendations"][0]["item_id"], 20)
            self.assertNotIn(10, [row["item_id"] for row in served["recommendations"]])
            self.assertTrue(served["content_model_loaded"])
            cold = recommend(
                db_path, visitor_id=999, k=1, content_model_dir=model_dir
            )
            self.assertEqual(cold["recommendations"][0]["item_id"], 30)
            self.assertEqual(cold["recommendations"][0]["sources"], ["popularity"])
            self.assertEqual(cold["model"], "weighted_popularity_fallback")


if __name__ == "__main__":
    unittest.main()
