"""Time-based offline evaluation for recommendation baselines."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import connect

TARGET_EVENTS = {
    "transaction": ("transaction",),
    "high-intent": ("addtocart", "transaction"),
}


def _timestamp(value_ms: int) -> str:
    return datetime.fromtimestamp(value_ms / 1000, tz=timezone.utc).isoformat()


def _event_placeholders(events: tuple[str, ...]) -> str:
    return ",".join("?" for _ in events)


def _ndcg(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(recommended[:k])
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def evaluate_popularity(
    db_path: Path,
    k: int = 10,
    target: str = "transaction",
    test_days: int = 14,
    cutoff_ms: int | None = None,
) -> dict[str, Any]:
    """Evaluate a train-period popularity model on future implicit targets.

    Training uses weighted interactions strictly before the cutoff. Relevant
    targets are selected events at or after the cutoff. Items already seen by a
    user during training are excluded from recommendations and test targets.
    Only products whose latest Silver availability is 1 are eligible.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if target not in TARGET_EVENTS:
        raise ValueError(f"target must be one of: {', '.join(TARGET_EVENTS)}")
    if test_days <= 0:
        raise ValueError("test_days must be positive")

    db = connect(db_path)
    try:
        _require_tables(db)
        minimum, maximum = db.execute(
            "SELECT MIN(event_timestamp_ms), MAX(event_timestamp_ms) "
            "FROM silver_user_events"
        ).fetchone()
        if minimum is None or maximum is None:
            raise RuntimeError("silver_user_events is empty")
        cutoff = cutoff_ms if cutoff_ms is not None else maximum - test_days * 86_400_000
        if cutoff <= minimum or cutoff > maximum:
            raise ValueError(
                f"cutoff must be after {_timestamp(minimum)} and no later than "
                f"{_timestamp(maximum)}"
            )

        available_items = {
            row[0] for row in db.execute(
                "SELECT item_id FROM silver_products WHERE available = 1"
            )
        }
        if not available_items:
            raise RuntimeError("No available products exist in silver_products")

        popularity = _train_popularity(db, cutoff, available_items)
        if not popularity:
            raise RuntimeError("No available items have interactions before the cutoff")

        events = TARGET_EVENTS[target]
        targets = _test_targets(db, cutoff, events, available_items)
        if not targets:
            raise RuntimeError(f"No future {target} targets exist after the cutoff")

        seen = _training_history(db, cutoff, set(targets))
        eligible_targets: dict[int, set[int]] = {}
        for visitor_id, item_ids in targets.items():
            novel = item_ids - seen.get(visitor_id, set())
            if novel:
                eligible_targets[visitor_id] = novel
        if not eligible_targets:
            raise RuntimeError("No users have novel, available targets in the test period")

        totals = defaultdict(float)
        recommended_catalog: set[int] = set()
        users_with_hits = 0
        for visitor_id, relevant in eligible_targets.items():
            recommendations = []
            visitor_seen = seen.get(visitor_id, set())
            for item_id, _score in popularity:
                if item_id not in visitor_seen:
                    recommendations.append(item_id)
                    if len(recommendations) == k:
                        break
            hits = len(set(recommendations) & relevant)
            totals["precision"] += hits / k
            totals["recall"] += hits / len(relevant)
            totals["ndcg"] += _ndcg(recommendations, relevant, k)
            if hits:
                users_with_hits += 1
            recommended_catalog.update(recommendations)

        user_count = len(eligible_targets)
        return {
            "model": "weighted-popularity",
            "target": target,
            "k": k,
            "cutoff_timestamp": _timestamp(cutoff),
            "train_period": {
                "start": _timestamp(minimum),
                "end_exclusive": _timestamp(cutoff),
            },
            "test_period": {
                "start_inclusive": _timestamp(cutoff),
                "end": _timestamp(maximum),
            },
            "eligible_users": user_count,
            "relevant_test_items": sum(map(len, eligible_targets.values())),
            "metrics": {
                f"precision@{k}": totals["precision"] / user_count,
                f"recall@{k}": totals["recall"] / user_count,
                f"ndcg@{k}": totals["ndcg"] / user_count,
                f"hit_rate@{k}": users_with_hits / user_count,
                f"catalog_coverage@{k}": len(recommended_catalog) / len(available_items),
            },
            "catalog": {
                "available_items": len(available_items),
                "train_items_with_interactions": len(popularity),
                "distinct_items_recommended": len(recommended_catalog),
            },
            "evaluation_rules": {
                "exclude_training_items": True,
                "available_items_only": True,
                "train_events": "view=1, addtocart=3, transaction=5",
            },
        }
    finally:
        db.close()


def _require_tables(db: sqlite3.Connection) -> None:
    required = {"silver_user_events", "silver_products"}
    existing = {
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = required - existing
    if missing:
        raise RuntimeError(f"Missing Silver tables: {', '.join(sorted(missing))}")


def _train_popularity(
    db: sqlite3.Connection,
    cutoff: int,
    available_items: set[int],
) -> list[tuple[int, int]]:
    scores = db.execute(
        """SELECT e.item_id,
                  SUM(CASE e.event_type
                      WHEN 'view' THEN 1
                      WHEN 'addtocart' THEN 3
                      WHEN 'transaction' THEN 5 ELSE 0 END) AS score
           FROM silver_user_events e
           JOIN silver_products p ON p.item_id = e.item_id AND p.available = 1
           WHERE e.event_timestamp_ms < ?
           GROUP BY e.item_id
           ORDER BY score DESC, e.item_id ASC""",
        (cutoff,),
    ).fetchall()
    return [(item_id, score) for item_id, score in scores if item_id in available_items]


def _test_targets(
    db: sqlite3.Connection,
    cutoff: int,
    events: tuple[str, ...],
    available_items: set[int],
) -> dict[int, set[int]]:
    placeholders = _event_placeholders(events)
    rows = db.execute(
        f"""SELECT DISTINCT visitor_id, item_id
            FROM silver_user_events
            WHERE event_timestamp_ms >= ? AND event_type IN ({placeholders})""",
        (cutoff, *events),
    )
    targets: dict[int, set[int]] = defaultdict(set)
    for visitor_id, item_id in rows:
        if item_id in available_items:
            targets[visitor_id].add(item_id)
    return targets


def _training_history(
    db: sqlite3.Connection,
    cutoff: int,
    visitors: set[int],
) -> dict[int, set[int]]:
    # A temporary table avoids constructing a potentially huge SQL IN clause.
    db.execute("DROP TABLE IF EXISTS temp.evaluation_visitors")
    db.execute("CREATE TEMP TABLE evaluation_visitors (visitor_id INTEGER PRIMARY KEY)")
    db.executemany(
        "INSERT INTO evaluation_visitors VALUES (?)",
        ((visitor_id,) for visitor_id in visitors),
    )
    rows = db.execute(
        """SELECT DISTINCT e.visitor_id, e.item_id
           FROM silver_user_events e
           JOIN evaluation_visitors v ON v.visitor_id = e.visitor_id
           WHERE e.event_timestamp_ms < ?""",
        (cutoff,),
    )
    seen: dict[int, set[int]] = defaultdict(set)
    for visitor_id, item_id in rows:
        seen[visitor_id].add(item_id)
    return seen

