"""Bronze-to-Silver cleaning and product consolidation."""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from src.progress import Progress, sqlite_activity

logger = logging.getLogger(__name__)


def build_silver(db: sqlite3.Connection) -> None:
    started = time.monotonic()
    logger.info("Silver events: cleaning and validating Bronze events")
    with sqlite_activity(db, "Silver user events"):
        db.executescript("""
    BEGIN;
    DROP TABLE IF EXISTS silver_user_events;
    CREATE TABLE silver_user_events AS
      SELECT datetime(timestamp / 1000, 'unixepoch') AS event_timestamp,
             visitorid AS visitor_id, itemid AS item_id, event AS event_type,
             transactionid AS transaction_id, timestamp AS event_timestamp_ms
      FROM bronze_events
      WHERE event IN ('view','addtocart','transaction')
        AND visitorid >= 0 AND itemid >= 0 AND timestamp > 0;
    CREATE INDEX ix_silver_events_user_item ON silver_user_events(visitor_id,item_id);

    COMMIT;
        """)
    logger.info("Silver categories: structuring category hierarchy")
    with sqlite_activity(db, "Silver category hierarchy"):
        db.executescript("""
    BEGIN;

    DROP TABLE IF EXISTS silver_category_hierarchy;
    CREATE TABLE silver_category_hierarchy AS
      SELECT categoryid AS category_id, parentid AS parent_category_id
      FROM bronze_category_tree;
    CREATE UNIQUE INDEX ix_silver_category ON silver_category_hierarchy(category_id);

    COMMIT;
        """)
    logger.info("Silver products: selecting latest value for each item/property")
    with sqlite_activity(db, "Silver latest product properties"):
        db.executescript("""
    BEGIN;

    DROP TABLE IF EXISTS silver_latest_properties;
    CREATE TEMP TABLE silver_latest_properties AS
      SELECT itemid, property, value, timestamp FROM (
        SELECT *, row_number() OVER (
          PARTITION BY itemid, property ORDER BY timestamp DESC, rowid DESC
        ) AS rn FROM bronze_item_properties
      ) WHERE rn = 1;

    DROP TABLE IF EXISTS silver_products;
    CREATE TABLE silver_products (
      item_id INTEGER PRIMARY KEY, category_id INTEGER, available INTEGER,
      encoded_properties TEXT NOT NULL
    );
    COMMIT;
        """)
    logger.info("Silver products: consolidating property rows into products")
    _consolidate_products(db)
    db.commit()
    logger.info("Silver pipeline finished in %.1f seconds", time.monotonic() - started)


def _consolidate_products(db: sqlite3.Connection) -> None:
    current_item = category = available = None
    properties: list[str] = []
    rows = []
    product_count = 0
    progress = Progress("Silver products", unit="products")
    query = "SELECT itemid, property, value FROM silver_latest_properties ORDER BY itemid, property"
    for item_id, prop, value in db.execute(query):
        if current_item is not None and item_id != current_item:
            rows.append((current_item, category, available, json.dumps(properties, separators=(",", ":"))))
            product_count += 1
            if len(rows) >= 5_000:
                db.executemany("INSERT INTO silver_products VALUES (?,?,?,?)", rows)
                rows = []
                progress.update(product_count)
            category, available, properties = None, None, []
        current_item = item_id
        if prop == "categoryid":
            try:
                category = int(value)
            except ValueError:
                category = None
        elif prop == "available":
            try:
                available = int(value)
            except ValueError:
                available = None
        else:
            properties.append(f"{prop}_{value}")
    if current_item is not None:
        rows.append((current_item, category, available, json.dumps(properties, separators=(",", ":"))))
        product_count += 1
    if rows:
        db.executemany("INSERT INTO silver_products VALUES (?,?,?,?)", rows)
    progress.close(product_count)
    logger.info("Silver products complete: %s products", f"{product_count:,}")
