"""Layer orchestration without source-ingestion concerns."""

from __future__ import annotations

from pathlib import Path
import logging

from src.core import connect, table_counts
from src.pipelines.gold import build_gold
from src.pipelines.silver import build_silver

logger = logging.getLogger(__name__)


def transform(db_path: Path, vector_size: int = 256) -> dict[str, int]:
    db = connect(db_path)
    required = {"bronze_events", "bronze_item_properties", "bronze_category_tree"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        db.close()
        raise RuntimeError(f"Missing Bronze tables: {', '.join(sorted(missing))}")
    try:
        logger.info("Silver pipeline started")
        build_silver(db)
        logger.info("Silver pipeline complete")
        logger.info("Gold pipeline started (vector size %s)", f"{vector_size:,}")
        build_gold(db, vector_size)
        logger.info("Gold pipeline complete")
        return table_counts(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
