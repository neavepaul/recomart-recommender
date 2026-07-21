"""Shared item co-occurrence and cosine-similarity computation.

Used by both temporal model training (``src/modeling/train.py``) and the Gold
feature-engineering layer (``src/pipelines/features.py``). The logic accumulates
weighted item-pair co-occurrences per user, then derives ranked cosine-similarity
neighbours. Parametrising the source table and output tables keeps a single,
tested implementation for both call sites.
"""

from __future__ import annotations

import itertools
import math
import sqlite3

from src.progress import Progress


def accumulate_item_pairs(
    db: sqlite3.Connection,
    *,
    source_table: str,
    order_column: str,
    pair_table: str,
    max_history: int,
) -> tuple[int, int]:
    """Accumulate weighted item-pair dot products and co-occurrence counts.

    Reads ``(visitor_id, item_id, interaction_score)`` from ``source_table`` and,
    for each user's top ``max_history`` items, records the product of interaction
    scores for every unordered item pair into ``pair_table``.
    """
    db.execute(f"DROP TABLE IF EXISTS {pair_table}")
    db.execute(
        f"""CREATE TABLE {pair_table} (
        item_i INTEGER NOT NULL,item_j INTEGER NOT NULL,dot_product REAL NOT NULL,
        cooccurring_users INTEGER NOT NULL,PRIMARY KEY(item_i,item_j))"""
    )
    upsert = f"""INSERT INTO {pair_table} VALUES (?,?,?,1)
        ON CONFLICT(item_i,item_j) DO UPDATE SET
        dot_product=dot_product+excluded.dot_product,
        cooccurring_users=cooccurring_users+1"""
    query = f"""SELECT visitor_id,item_id,interaction_score FROM {source_table}
               ORDER BY visitor_id,interaction_score DESC,{order_column} DESC"""
    current_user = None
    history: list[tuple[int, float]] = []
    batch: list[tuple[int, int, float]] = []
    pair_events = contributing_users = 0
    progress = Progress("Item pair contributions", unit="pairs")

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
                progress.update(pair_events)

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
    progress.close(pair_events)
    return pair_events, contributing_users


def calculate_similarities(
    db: sqlite3.Connection,
    *,
    source_table: str,
    pair_table: str,
    similarity_table: str,
    min_cooccurrence: int,
    neighbors: int,
) -> None:
    """Derive ranked cosine-similarity neighbours from accumulated pair stats."""
    db.execute(f"DROP TABLE IF EXISTS {similarity_table}")
    db.execute(
        f"""CREATE TABLE {similarity_table} (
        source_item_id INTEGER NOT NULL,similar_item_id INTEGER NOT NULL,
        similarity REAL NOT NULL,cooccurring_users INTEGER NOT NULL,
        neighbor_rank INTEGER NOT NULL,PRIMARY KEY(source_item_id,similar_item_id))"""
    )
    db.create_function("SQRT", 1, math.sqrt)
    db.execute("DROP TABLE IF EXISTS temp.item_cooccurrence_norms")
    db.execute(
        f"""CREATE TEMP TABLE item_cooccurrence_norms AS
        SELECT item_id,SUM(interaction_score*interaction_score) norm_squared
        FROM {source_table} GROUP BY item_id"""
    )
    db.execute(
        "CREATE UNIQUE INDEX temp.ix_item_cooccurrence_norms ON item_cooccurrence_norms(item_id)"
    )
    db.execute("DROP TABLE IF EXISTS temp.item_cooccurrence_candidates")
    db.execute(
        f"""CREATE TEMP TABLE item_cooccurrence_candidates AS
        SELECT p.item_i source_item_id,p.item_j similar_item_id,
               p.dot_product/SQRT(ni.norm_squared*nj.norm_squared) similarity,
               p.cooccurring_users
        FROM {pair_table} p
        JOIN item_cooccurrence_norms ni ON ni.item_id=p.item_i
        JOIN item_cooccurrence_norms nj ON nj.item_id=p.item_j
        WHERE p.cooccurring_users>=?
        UNION ALL
        SELECT p.item_j,p.item_i,
               p.dot_product/SQRT(ni.norm_squared*nj.norm_squared),
               p.cooccurring_users
        FROM {pair_table} p
        JOIN item_cooccurrence_norms ni ON ni.item_id=p.item_i
        JOIN item_cooccurrence_norms nj ON nj.item_id=p.item_j
        WHERE p.cooccurring_users>=?""",
        (min_cooccurrence, min_cooccurrence),
    )
    db.execute(
        f"""INSERT INTO {similarity_table}
        SELECT source_item_id,similar_item_id,similarity,cooccurring_users,neighbor_rank
        FROM (
          SELECT *,ROW_NUMBER() OVER (
            PARTITION BY source_item_id
            ORDER BY similarity DESC,cooccurring_users DESC,similar_item_id
          ) neighbor_rank
          FROM item_cooccurrence_candidates
        ) WHERE neighbor_rank<=?""",
        (neighbors,),
    )
    db.execute(
        f"CREATE INDEX ix_{similarity_table}_source ON {similarity_table}(source_item_id,neighbor_rank)"
    )
    db.commit()
