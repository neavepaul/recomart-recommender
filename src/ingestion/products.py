"""REST client ingestion for product metadata."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

from src.core import connect, rebuilt_table
from src.ingestion.product_api import make_server
from src.progress import Progress, sqlite_activity

logger = logging.getLogger(__name__)


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
    started = time.monotonic()
    logger.info(
        "Bronze products: ingesting REST pages of %s rows%s",
        f"{page_size:,}",
        f" (limit {limit:,})" if limit is not None else "",
    )
    try:
        progress = Progress("Bronze product properties", total=limit, unit="rows")
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
                progress.update(count)
                offset = int(payload["next_offset"])
                if not payload["has_more"]:
                    break
            logger.info("Bronze products: building item/timestamp index")
            with sqlite_activity(db, "Bronze product index"):
                db.execute(
                    "CREATE INDEX ix_bronze_props_item "
                    "ON bronze_item_properties(itemid,timestamp)"
                )
        progress.close(count)
        logger.info(
            "Bronze products complete: %s rows in %.1f seconds",
            f"{count:,}", time.monotonic() - started,
        )
        return count
    finally:
        db.close()
