"""REST client ingestion for product metadata."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from src.core import connect, rebuilt_table
from src.ingestion.product_api import make_server


def ingest_products(
    db_path: Path,
    api_url: str,
    page_size: int = 5_000,
    limit: int | None = None,
) -> int:
    db = connect(db_path)
    ddl = """CREATE TABLE bronze_item_properties (
        timestamp INTEGER NOT NULL, itemid INTEGER NOT NULL,
        property TEXT NOT NULL, value TEXT NOT NULL
    )"""
    count = 0
    try:
        with rebuilt_table(db, "bronze_item_properties", ddl):
            offset = 0
            while limit is None or count < limit:
                request_size = page_size if limit is None else min(page_size, limit - count)
                url = f"{api_url.rstrip('/')}/item-properties?offset={offset}&limit={request_size}"
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = json.load(response)
                items = payload["items"]
                if not items:
                    break
                rows = [
                    (int(row["timestamp"]), int(row["itemid"]), row["property"], row["value"])
                    for row in items
                ]
                db.executemany("INSERT INTO bronze_item_properties VALUES (?,?,?,?)", rows)
                count += len(rows)
                offset = int(payload["next_offset"])
                if not payload["has_more"]:
                    break
            db.execute(
                "CREATE INDEX ix_bronze_props_item "
                "ON bronze_item_properties(itemid,timestamp)"
            )
        return count
    finally:
        db.close()

