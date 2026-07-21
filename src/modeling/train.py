"""Popularity and item-based collaborative-filtering model training."""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Any

from src.cooccurrence import accumulate_item_pairs, calculate_similarities
from src.core import connect
from src.progress import sqlite_activity

logger = logging.getLogger(__name__)


def train_models(db_path: Path, max_history: int = 30,
                 min_cooccurrence: int = 2,
                 neighbors: int = 50) -> dict[str, Any]:
    if max_history < 2:
        raise ValueError("max_history must be at least 2")
    if min_cooccurrence < 1:
        raise ValueError("min_cooccurrence must be at least 1")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    db = connect(db_path)
    try:
        _require_split(db)
        logger.info("Training weighted popularity baseline")
        with sqlite_activity(db, "Popularity model"):
            _train_popularity(db)
        logger.info("Training item collaborative-filtering co-occurrences")
        pair_events, contributing_users = accumulate_item_pairs(
            db, source_table="model_train_user_items",
            order_column="last_interaction_timestamp_ms",
            pair_table="model_item_pair_stats", max_history=max_history,
        )
        logger.info("Calculating item cosine similarities")
        with sqlite_activity(db, "Collaborative similarities"):
            calculate_similarities(
                db, source_table="model_train_user_items",
                pair_table="model_item_pair_stats",
                similarity_table="model_item_similarity",
                min_cooccurrence=min_cooccurrence, neighbors=neighbors,
            )
        return {
            "popularity_items": db.execute("SELECT COUNT(*) FROM model_popularity").fetchone()[0],
            "collaborative_filtering": {
                "algorithm": "weighted-item-cosine",
                "max_history_per_user": max_history,
                "minimum_cooccurrence": min_cooccurrence,
                "neighbors_per_item": neighbors,
                "contributing_users": contributing_users,
                "pair_contributions": pair_events,
                "distinct_item_pairs": db.execute("SELECT COUNT(*) FROM model_item_pair_stats").fetchone()[0],
                "stored_directed_similarities": db.execute("SELECT COUNT(*) FROM model_item_similarity").fetchone()[0],
            },
        }
    finally:
        db.close()


def _require_split(db: sqlite3.Connection) -> None:
    required = {"model_split_metadata", "model_train_user_items", "model_test_targets"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        raise RuntimeError("Missing model split tables; run prepare-model-data first: " + ", ".join(sorted(missing)))


def _train_popularity(db: sqlite3.Connection) -> None:
    db.execute("DROP TABLE IF EXISTS model_popularity")
    db.execute("""CREATE TABLE model_popularity AS
        SELECT t.item_id,SUM(t.interaction_score) score,
               ROW_NUMBER() OVER (ORDER BY SUM(t.interaction_score) DESC,t.item_id) rank
        FROM model_train_user_items t JOIN silver_products p
        ON p.item_id=t.item_id AND p.available=1 GROUP BY t.item_id""")
    db.execute("CREATE UNIQUE INDEX ix_model_popularity_item ON model_popularity(item_id)")
    db.execute("CREATE UNIQUE INDEX ix_model_popularity_rank ON model_popularity(rank)")
    db.commit()
