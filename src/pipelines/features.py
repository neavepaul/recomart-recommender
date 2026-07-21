"""Gold-to-feature transformation for recommendation algorithms.

Materialises reusable feature tables from the Gold layer:

* ``feature_user_activity``    — per-user activity frequency and average score
* ``feature_item_popularity``  — per-item popularity, average score, conversion
* ``feature_item_cooccurrence`` — item-item cosine-similarity neighbours

All tables live in the same SQLite store as the Gold layer so downstream
training and the feature store (Task 7) can retrieve them directly.
"""

from __future__ import annotations

import logging
import sqlite3
import time

from src.cooccurrence import accumulate_item_pairs, calculate_similarities
from src.progress import sqlite_activity

logger = logging.getLogger(__name__)


def build_features(
    db: sqlite3.Connection,
    neighbors: int = 50,
    min_cooccurrence: int = 2,
    max_history: int = 30,
) -> None:
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if min_cooccurrence < 1:
        raise ValueError("min_cooccurrence must be at least 1")
    if max_history < 2:
        raise ValueError("max_history must be at least 2")
    _require_gold(db)
    started = time.monotonic()
    logger.info("Feature engineering: per-user activity aggregates")
    with sqlite_activity(db, "Feature user activity"):
        _build_user_activity(db)
    logger.info("Feature engineering: per-item popularity aggregates")
    with sqlite_activity(db, "Feature item popularity"):
        _build_item_popularity(db)
    logger.info("Feature engineering: item co-occurrence similarities")
    accumulate_item_pairs(
        db, source_table="gold_user_item_features",
        order_column="last_interaction_timestamp",
        pair_table="feature_item_pair_stats", max_history=max_history,
    )
    with sqlite_activity(db, "Feature item similarities"):
        calculate_similarities(
            db, source_table="gold_user_item_features",
            pair_table="feature_item_pair_stats",
            similarity_table="feature_item_cooccurrence",
            min_cooccurrence=min_cooccurrence, neighbors=neighbors,
        )
    db.commit()
    logger.info("Feature pipeline finished in %.1f seconds", time.monotonic() - started)


def _require_gold(db: sqlite3.Connection) -> None:
    required = {"gold_user_item_features", "gold_item_features"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        raise RuntimeError("Missing Gold tables; run build-gold first: " + ", ".join(sorted(missing)))


def _build_user_activity(db: sqlite3.Connection) -> None:
    db.executescript("""
    DROP TABLE IF EXISTS feature_user_activity;
    CREATE TABLE feature_user_activity AS
      SELECT visitor_id,
        COUNT(*) AS distinct_items,
        SUM(view_count + cart_count + purchase_count) AS total_events,
        SUM(view_count) AS view_count,
        SUM(cart_count) AS cart_count,
        SUM(purchase_count) AS purchase_count,
        SUM(interaction_score) AS total_interaction_score,
        SUM(interaction_score) * 1.0 / COUNT(*) AS avg_interaction_score,
        COALESCE(SUM(purchase_count) * 1.0
          / NULLIF(SUM(view_count + cart_count + purchase_count), 0), 0.0)
          AS purchase_rate,
        MAX(last_interaction_timestamp) AS last_active_timestamp
      FROM gold_user_item_features GROUP BY visitor_id;
    CREATE UNIQUE INDEX ix_feature_user_activity ON feature_user_activity(visitor_id);
    """)


def _build_item_popularity(db: sqlite3.Connection) -> None:
    db.executescript("""
    DROP TABLE IF EXISTS feature_item_popularity;
    CREATE TABLE feature_item_popularity AS
      SELECT f.item_id, f.category_id, f.parent_category_id, f.available,
        COALESCE(a.distinct_users, 0) AS distinct_users,
        COALESCE(a.view_count, 0) AS view_count,
        COALESCE(a.cart_count, 0) AS cart_count,
        COALESCE(a.purchase_count, 0) AS purchase_count,
        COALESCE(a.total_interaction_score, 0) AS total_interaction_score,
        COALESCE(a.total_interaction_score * 1.0
          / NULLIF(a.distinct_users, 0), 0.0) AS avg_interaction_score,
        COALESCE(a.purchase_count * 1.0
          / NULLIF(a.view_count, 0), 0.0) AS conversion_rate,
        ROW_NUMBER() OVER (
          ORDER BY COALESCE(a.total_interaction_score, 0) DESC, f.item_id
        ) AS popularity_rank
      FROM gold_item_features f
      LEFT JOIN (
        SELECT item_id,
          COUNT(DISTINCT visitor_id) AS distinct_users,
          SUM(view_count) AS view_count,
          SUM(cart_count) AS cart_count,
          SUM(purchase_count) AS purchase_count,
          SUM(interaction_score) AS total_interaction_score
        FROM gold_user_item_features GROUP BY item_id
      ) a ON a.item_id = f.item_id;
    CREATE UNIQUE INDEX ix_feature_item_popularity ON feature_item_popularity(item_id);
    CREATE UNIQUE INDEX ix_feature_item_popularity_rank ON feature_item_popularity(popularity_rank);
    """)
