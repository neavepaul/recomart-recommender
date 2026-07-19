"""Silver-to-Gold model feature generation."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time

from src.progress import Progress, sqlite_activity

logger = logging.getLogger(__name__)


def build_gold(db: sqlite3.Connection, vector_size: int = 256) -> None:
    started = time.monotonic()
    logger.info("Gold interactions: aggregating user/item behavior")
    with sqlite_activity(db, "Gold user-item aggregation"):
        _build_interaction_features(db)
    logger.info("Gold items: generating sparse vectors with %s dimensions", f"{vector_size:,}")
    _build_item_features(db, vector_size)
    db.commit()
    logger.info("Gold pipeline finished in %.1f seconds", time.monotonic() - started)


def _build_interaction_features(db: sqlite3.Connection) -> None:
    db.executescript("""
    DROP TABLE IF EXISTS gold_user_item_features;
    CREATE TABLE gold_user_item_features AS
      SELECT visitor_id, item_id,
        SUM(event_type='view') AS view_count,
        SUM(event_type='addtocart') AS cart_count,
        SUM(event_type='transaction') AS purchase_count,
        SUM(CASE event_type WHEN 'view' THEN 1 WHEN 'addtocart' THEN 3 ELSE 5 END)
          AS interaction_score,
        datetime(MAX(event_timestamp_ms) / 1000, 'unixepoch')
          AS last_interaction_timestamp
      FROM silver_user_events GROUP BY visitor_id, item_id;
    CREATE UNIQUE INDEX ix_gold_user_item
      ON gold_user_item_features(visitor_id,item_id);
    """)


def _build_item_features(db: sqlite3.Connection, vector_size: int) -> None:
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    db.execute("DROP TABLE IF EXISTS gold_item_features")
    db.execute("""CREATE TABLE gold_item_features (
      item_id INTEGER PRIMARY KEY, category_id INTEGER, parent_category_id INTEGER,
      available INTEGER, item_feature_vector TEXT NOT NULL)""")
    rows = []
    item_count = 0
    progress = Progress("Gold item vectors", unit="items")
    query = """SELECT p.item_id,p.category_id,c.parent_category_id,
                      p.available,p.encoded_properties
               FROM silver_products p LEFT JOIN silver_category_hierarchy c
               ON p.category_id=c.category_id"""
    for item_id, category, parent, available, encoded in db.execute(query):
        tokens = json.loads(encoded)
        if category is not None:
            tokens.append(f"category:{category}")
        if parent is not None:
            tokens.append(f"parent_category:{parent}")
        vector: dict[int, float] = {}
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % vector_size
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[bucket] = vector.get(bucket, 0.0) + sign
        rows.append((item_id, category, parent, available, json.dumps(vector, separators=(",", ":"))))
        item_count += 1
        if len(rows) >= 5_000:
            db.executemany("INSERT INTO gold_item_features VALUES (?,?,?,?,?)", rows)
            rows = []
            progress.update(item_count)
    if rows:
        db.executemany("INSERT INTO gold_item_features VALUES (?,?,?,?,?)", rows)
    progress.close(item_count)
    logger.info("Gold items complete: %s vectors", f"{item_count:,}")
