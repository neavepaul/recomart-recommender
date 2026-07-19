"""Bronze pipeline orchestration across the three source adapters."""

from __future__ import annotations

import threading
from pathlib import Path

from src.ingestion.categories import ingest_categories
from src.ingestion.events import replay_events
from src.ingestion.products import ingest_products, make_server


def build_bronze(
    db_path: Path,
    raw_dir: Path,
    speed: float = 0,
    limit: int | None = None,
    api_page_size: int = 5_000,
) -> dict[str, int]:
    """Run batch, clickstream, and REST ingestion into Bronze tables."""
    result = {
        "bronze_category_tree": ingest_categories(db_path, raw_dir),
        "bronze_events": replay_events(db_path, raw_dir, speed, limit),
    }
    server = make_server("127.0.0.1", 0, raw_dir)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
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
    return result

