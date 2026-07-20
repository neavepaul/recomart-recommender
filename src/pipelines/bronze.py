"""Bronze pipeline orchestration across the three source adapters."""

from __future__ import annotations

import threading
import logging
from pathlib import Path

from src.ingestion.categories import ingest_categories
from src.ingestion.events import replay_events
from src.ingestion.products import ingest_products, make_server

logger = logging.getLogger(__name__)


def build_bronze(
    db_path: Path,
    raw_dir: Path,
    speed: float = 0,
    limit: int | None = None,
    api_page_size: int = 50_000,
) -> dict[str, int]:
    """Run batch, clickstream, and REST ingestion into Bronze tables."""
    logger.info("Bronze pipeline started")
    result = {
        "bronze_category_tree": ingest_categories(db_path, raw_dir),
        "bronze_events": replay_events(db_path, raw_dir, speed, limit),
    }
    server = make_server("127.0.0.1", 0, raw_dir)
    thread = threading.Thread(
        target=server.serve_forever,
        name="recomart-product-api",
        daemon=False,
    )
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        result["bronze_item_properties"] = ingest_products(
            db_path, url, api_page_size, limit
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    logger.info("Bronze pipeline complete: %s", result)
    return result
