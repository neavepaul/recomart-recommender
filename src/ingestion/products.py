"""REST client ingestion for product metadata."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.core import connect, rebuilt_table
from src.ingestion.product_api import MAX_API_PAGE_SIZE, make_server
from src.progress import Progress, sqlite_activity

logger = logging.getLogger(__name__)


def _request_page(url: str, attempts: int = 3) -> dict:
    """Fetch one API page, retrying transient timeouts with the same offset."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return json.load(response)
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == attempts:
                raise
            logger.warning(
                "Product API request failed (%s); retrying page (%d/%d)",
                error, attempt + 1, attempts,
            )
            time.sleep(attempt)
    raise RuntimeError("unreachable")


def ingest_products(
    db_path: Path,
    api_url: str,
    page_size: int = 50_000,
    limit: int | None = None,
) -> int:
    if not 1 <= page_size <= MAX_API_PAGE_SIZE:
        raise ValueError(
            f"api page size must be between 1 and {MAX_API_PAGE_SIZE:,} rows"
        )
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
                payload = _request_page(url)
                effective_page_size = int(payload.get("page_size", request_size))
                if effective_page_size != request_size:
                    raise RuntimeError(
                        f"Product API accepted {effective_page_size:,} rows but "
                        f"{request_size:,} were requested"
                    )
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
                    "ON bronze_item_properties(itemid,property,timestamp DESC)"
                )
        progress.close(count)
        logger.info(
            "Bronze products complete: %s rows in %.1f seconds",
            f"{count:,}", time.monotonic() - started,
        )
        return count
    finally:
        db.close()
