"""Batch ingestion for the category hierarchy reference data."""

from __future__ import annotations

import csv
from pathlib import Path

from src.core import connect, nullable_int, rebuilt_table


def ingest_categories(db_path: Path, raw_dir: Path) -> int:
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
        return len(rows)
    finally:
        db.close()

