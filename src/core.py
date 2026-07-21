"""Shared configuration and SQLite utilities for RecoMart."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_DB = ROOT / "data" / "recomart.db"
EVENT_TYPES = {"view", "addtocart", "transaction"}
WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA temp_store=FILE")
    # Keep a bounded working set in RAM and memory-map reads. These settings
    # remain comfortably below an 8 GB machine while helping the 20M-row sort.
    db.execute("PRAGMA cache_size=-262144")
    db.execute("PRAGMA mmap_size=536870912")
    return db


@contextmanager
def rebuilt_table(db: sqlite3.Connection, name: str, ddl: str) -> Iterator[None]:
    """Replace one stage table transactionally."""
    db.execute("BEGIN")
    try:
        db.execute(f"DROP TABLE IF EXISTS {name}")
        db.execute(ddl)
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise


def batched(rows: Iterable[tuple[Any, ...]], size: int = 10_000):
    batch: list[tuple[Any, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def nullable_int(value: str | None) -> int | None:
    return None if value is None or value.strip() == "" else int(value)


def table_counts(db: sqlite3.Connection) -> dict[str, int]:
    tables = [
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name LIKE 'bronze_%' OR name LIKE 'silver_%' OR name LIKE 'gold_%' "
            "OR name LIKE 'feature_%') "
            "ORDER BY name"
        )
    ]
    return {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in tables}
