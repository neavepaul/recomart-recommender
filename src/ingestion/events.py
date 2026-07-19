"""Simulated near-real-time event clickstream ingestion."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.core import batched, connect, nullable_int, rebuilt_table


def _rows(path: Path, limit: int | None = None):
    with path.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if limit is not None and index >= limit:
                break
            yield (
                int(row["timestamp"]),
                int(row["visitorid"]),
                row["event"],
                int(row["itemid"]),
                nullable_int(row.get("transactionid")),
            )


def replay_events(db_path: Path, raw_dir: Path, speed: float = 0, limit: int | None = None) -> int:
    """Replay events in source order into the Bronze event table."""
    db = connect(db_path)
    count = 0
    ddl = """CREATE TABLE bronze_events (
        timestamp INTEGER NOT NULL, visitorid INTEGER NOT NULL,
        event TEXT NOT NULL, itemid INTEGER NOT NULL, transactionid INTEGER
    )"""
    try:
        with rebuilt_table(db, "bronze_events", ddl):
            previous: int | None = None
            for batch in batched(_rows(raw_dir / "events.csv", limit), 5_000):
                if speed > 0:
                    current = batch[0][0]
                    if previous is not None:
                        time.sleep(min(max(current - previous, 0) / 1000 / speed, 1.0))
                    previous = batch[-1][0]
                db.executemany("INSERT INTO bronze_events VALUES (?,?,?,?,?)", batch)
                count += len(batch)
            db.execute(
                "CREATE INDEX ix_bronze_events_user_item "
                "ON bronze_events(visitorid,itemid)"
            )
        return count
    finally:
        db.close()

