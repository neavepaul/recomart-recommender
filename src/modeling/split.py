"""Leakage-safe temporal training and test-table construction."""

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import connect
from src.progress import sqlite_activity

logger = logging.getLogger(__name__)

TARGET_EVENTS = {"transaction": ("transaction",),
                 "high-intent": ("addtocart", "transaction")}


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def prepare_model_data(db_path: Path, target: str = "transaction",
                       test_days: int = 14,
                       cutoff_ms: int | None = None) -> dict[str, Any]:
    if target not in TARGET_EVENTS:
        raise ValueError(f"target must be one of: {', '.join(TARGET_EVENTS)}")
    if test_days <= 0:
        raise ValueError("test_days must be positive")
    db = connect(db_path)
    try:
        _require_silver(db)
        minimum, maximum = db.execute(
            "SELECT MIN(event_timestamp_ms),MAX(event_timestamp_ms) FROM silver_user_events"
        ).fetchone()
        if minimum is None:
            raise RuntimeError("silver_user_events is empty")
        cutoff = cutoff_ms if cutoff_ms is not None else maximum - test_days * 86_400_000
        if cutoff <= minimum or cutoff > maximum:
            raise ValueError(f"cutoff must be after {_iso(minimum)} and no later than {_iso(maximum)}")
        events = TARGET_EVENTS[target]
        logger.info("Preparing temporal split at %s with target '%s'", _iso(cutoff), target)
        placeholders = ",".join("?" for _ in events)
        db.execute("BEGIN")
        for table in ("model_split_metadata", "model_train_user_items",
                      "model_test_targets", "model_popularity",
                      "model_item_pair_stats", "model_item_similarity",
                      "model_hybrid_config"):
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.execute("""CREATE TABLE model_split_metadata (
            cutoff_ms INTEGER NOT NULL,target TEXT NOT NULL,
            minimum_event_ms INTEGER NOT NULL,maximum_event_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        db.execute("INSERT INTO model_split_metadata(cutoff_ms,target,minimum_event_ms,maximum_event_ms) VALUES (?,?,?,?)",
                   (cutoff, target, minimum, maximum))
        with sqlite_activity(db, "Temporal split training features"):
            db.execute("""CREATE TABLE model_train_user_items AS
                SELECT visitor_id,item_id,SUM(event_type='view') view_count,
                       SUM(event_type='addtocart') cart_count,
                       SUM(event_type='transaction') purchase_count,
                       SUM(CASE event_type WHEN 'view' THEN 1 WHEN 'addtocart' THEN 3 ELSE 5 END) interaction_score,
                       MAX(event_timestamp_ms) last_interaction_timestamp_ms
                FROM silver_user_events WHERE event_timestamp_ms<?
                GROUP BY visitor_id,item_id""", (cutoff,))
        db.execute("CREATE UNIQUE INDEX ix_model_train_user_item ON model_train_user_items(visitor_id,item_id)")
        db.execute("CREATE INDEX ix_model_train_item ON model_train_user_items(item_id)")
        with sqlite_activity(db, "Temporal split future targets"):
            db.execute(f"""CREATE TABLE model_test_targets AS
                SELECT e.visitor_id,e.item_id,COUNT(*) target_events
                FROM silver_user_events e JOIN silver_products p
                ON p.item_id=e.item_id AND p.available=1
                WHERE e.event_timestamp_ms>=? AND e.event_type IN ({placeholders})
                  AND NOT EXISTS (SELECT 1 FROM model_train_user_items t
                      WHERE t.visitor_id=e.visitor_id AND t.item_id=e.item_id)
                GROUP BY e.visitor_id,e.item_id""", (cutoff, *events))
        db.execute("CREATE UNIQUE INDEX ix_model_test_user_item ON model_test_targets(visitor_id,item_id)")
        db.commit()
        train_pairs, train_users, train_items = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT visitor_id),COUNT(DISTINCT item_id) FROM model_train_user_items"
        ).fetchone()
        test_pairs, test_users = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT visitor_id) FROM model_test_targets"
        ).fetchone()
        logger.info("Temporal split complete: %s train pairs, %s users with test targets",
                    f"{train_pairs:,}", f"{test_users:,}")
        return {"target": target, "cutoff_timestamp": _iso(cutoff),
                "train_period": {"start": _iso(minimum), "end_exclusive": _iso(cutoff)},
                "test_period": {"start_inclusive": _iso(cutoff), "end": _iso(maximum)},
                "train": {"users": train_users, "items": train_items,
                          "user_item_pairs": train_pairs},
                "test": {"eligible_users": test_users,
                         "novel_available_targets": test_pairs}}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _require_silver(db: sqlite3.Connection) -> None:
    required = {"silver_user_events", "silver_products"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        raise RuntimeError(f"Missing Silver tables: {', '.join(sorted(missing))}")
