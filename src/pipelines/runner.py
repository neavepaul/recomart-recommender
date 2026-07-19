"""Layer orchestration without source-ingestion concerns."""

from __future__ import annotations

from pathlib import Path

from src.core import connect, table_counts
from src.pipelines.gold import build_gold
from src.pipelines.silver import build_silver


def transform(db_path: Path, vector_size: int = 256) -> dict[str, int]:
    db = connect(db_path)
    required = {"bronze_events", "bronze_item_properties", "bronze_category_tree"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        db.close()
        raise RuntimeError(f"Missing Bronze tables: {', '.join(sorted(missing))}")
    try:
        build_silver(db)
        build_gold(db, vector_size)
        return table_counts(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

