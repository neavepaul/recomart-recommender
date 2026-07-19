"""Batch ingestion for the category hierarchy reference data."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.core import connect, nullable_int, rebuilt_table

logger = logging.getLogger(__name__)


def ingest_categories(db_path: Path, raw_dir: Path) -> int:
    logger.info("Bronze categories: loading category_tree.csv")
    db = connect(db_path)
    ddl = "CREATE TABLE bronze_category_tree (categoryid INTEGER PRIMARY KEY, parentid INTEGER)"
    try:
        with (raw_dir / "category_tree.csv").open(newline="", encoding="utf-8") as stream:
            rows = [
                (int(row["categoryid"]), nullable_int(row.get("parentid")))
                for row in csv.DictReader(stream)
            ]
        with rebuilt_table(db, "bronze_category_tree", ddl):
            db.executemany("INSERT INTO bronze_category_tree VALUES (?,?)", rows)
        logger.info("Bronze categories complete: %s rows", f"{len(rows):,}")
        return len(rows)
    finally:
        db.close()
