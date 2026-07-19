"""Popularity and item-based collaborative-filtering model training."""

from __future__ import annotations

import itertools
import math
import sqlite3
from pathlib import Path
from typing import Any

from src.core import connect


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
        _train_popularity(db)
        pair_events, contributing_users = _accumulate_item_pairs(db, max_history)
        _calculate_similarities(db, min_cooccurrence, neighbors)
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


def _accumulate_item_pairs(db: sqlite3.Connection, max_history: int) -> tuple[int, int]:
    db.execute("DROP TABLE IF EXISTS model_item_pair_stats")
    db.execute("""CREATE TABLE model_item_pair_stats (
        item_i INTEGER NOT NULL,item_j INTEGER NOT NULL,dot_product REAL NOT NULL,
        cooccurring_users INTEGER NOT NULL,PRIMARY KEY(item_i,item_j))""")
    upsert = """INSERT INTO model_item_pair_stats VALUES (?,?,?,1)
        ON CONFLICT(item_i,item_j) DO UPDATE SET
        dot_product=dot_product+excluded.dot_product,
        cooccurring_users=cooccurring_users+1"""
    query = """SELECT visitor_id,item_id,interaction_score FROM model_train_user_items
               ORDER BY visitor_id,interaction_score DESC,last_interaction_timestamp_ms DESC"""
    current_user = None
    history: list[tuple[int, float]] = []
    batch: list[tuple[int, int, float]] = []
    pair_events = contributing_users = 0

    def add_history(items: list[tuple[int, float]]) -> None:
        nonlocal pair_events, contributing_users, batch
        selected = items[:max_history]
        if len(selected) < 2:
            return
        contributing_users += 1
        selected.sort(key=lambda value: value[0])
        for (item_i, score_i), (item_j, score_j) in itertools.combinations(selected, 2):
            batch.append((item_i, item_j, score_i * score_j))
            pair_events += 1
            if len(batch) >= 50_000:
                db.executemany(upsert, batch)
                db.commit()
                batch = []

    for visitor_id, item_id, score in db.execute(query):
        if current_user is not None and visitor_id != current_user:
            add_history(history)
            history = []
        current_user = visitor_id
        history.append((item_id, float(score)))
    if current_user is not None:
        add_history(history)
    if batch:
        db.executemany(upsert, batch)
    db.commit()
    return pair_events, contributing_users


def _calculate_similarities(db: sqlite3.Connection,
                            min_cooccurrence: int, neighbors: int) -> None:
    db.execute("DROP TABLE IF EXISTS model_item_similarity")
    db.execute("""CREATE TABLE model_item_similarity (
        source_item_id INTEGER NOT NULL,similar_item_id INTEGER NOT NULL,
        similarity REAL NOT NULL,cooccurring_users INTEGER NOT NULL,
        neighbor_rank INTEGER NOT NULL,PRIMARY KEY(source_item_id,similar_item_id))""")
    db.create_function("SQRT", 1, math.sqrt)
    db.execute("DROP TABLE IF EXISTS temp.model_item_norms")
    db.execute("""CREATE TEMP TABLE model_item_norms AS
        SELECT item_id,SUM(interaction_score*interaction_score) norm_squared
        FROM model_train_user_items GROUP BY item_id""")
    db.execute("CREATE UNIQUE INDEX temp.ix_model_item_norms ON model_item_norms(item_id)")
    db.execute("DROP TABLE IF EXISTS temp.model_similarity_candidates")
    db.execute("""CREATE TEMP TABLE model_similarity_candidates AS
        SELECT p.item_i source_item_id,p.item_j similar_item_id,
               p.dot_product/SQRT(ni.norm_squared*nj.norm_squared) similarity,
               p.cooccurring_users
        FROM model_item_pair_stats p
        JOIN model_item_norms ni ON ni.item_id=p.item_i
        JOIN model_item_norms nj ON nj.item_id=p.item_j
        WHERE p.cooccurring_users>=?
        UNION ALL
        SELECT p.item_j,p.item_i,
               p.dot_product/SQRT(ni.norm_squared*nj.norm_squared),
               p.cooccurring_users
        FROM model_item_pair_stats p
        JOIN model_item_norms ni ON ni.item_id=p.item_i
        JOIN model_item_norms nj ON nj.item_id=p.item_j
        WHERE p.cooccurring_users>=?""", (min_cooccurrence, min_cooccurrence))
    db.execute("""INSERT INTO model_item_similarity
        SELECT source_item_id,similar_item_id,similarity,cooccurring_users,neighbor_rank
        FROM (
          SELECT *,ROW_NUMBER() OVER (
            PARTITION BY source_item_id
            ORDER BY similarity DESC,cooccurring_users DESC,similar_item_id
          ) neighbor_rank
          FROM model_similarity_candidates
        ) WHERE neighbor_rank<=?""", (neighbors,))
    db.execute("CREATE INDEX ix_model_similarity_source ON model_item_similarity(source_item_id,neighbor_rank)")
    db.commit()
