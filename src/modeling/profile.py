"""Descriptive profiling for Gold recommendation data."""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Any

from src.core import connect

logger = logging.getLogger(__name__)


def _distribution(db: sqlite3.Connection, entity: str) -> dict[str, float | int]:
    if entity not in {"visitor_id", "item_id"}:
        raise ValueError("unsupported profile entity")
    histogram = db.execute(
        f"""SELECT interactions, COUNT(*) FROM (
                SELECT {entity}, COUNT(*) AS interactions
                FROM gold_user_item_features GROUP BY {entity}
            ) GROUP BY interactions ORDER BY interactions"""
    ).fetchall()
    total = sum(frequency for _, frequency in histogram)
    weighted = sum(value * frequency for value, frequency in histogram)

    def percentile(fraction: float) -> int:
        threshold = max(1, int(total * fraction + 0.999999))
        cumulative = 0
        for value, frequency in histogram:
            cumulative += frequency
            if cumulative >= threshold:
                return value
        return histogram[-1][0]

    return {
        "min": histogram[0][0], "mean": weighted / total,
        "p50": percentile(0.50), "p90": percentile(0.90),
        "p99": percentile(0.99), "max": histogram[-1][0],
    }


def profile_gold(db_path: Path, top_n: int = 10) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    db = connect(db_path)
    try:
        logger.info("Profiling Gold tables")
        required = {"gold_user_item_features", "gold_item_features"}
        existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = required - existing
        if missing:
            raise RuntimeError(f"Missing Gold tables: {', '.join(sorted(missing))}")
        pairs, users, interacted_items = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT visitor_id),COUNT(DISTINCT item_id) FROM gold_user_item_features"
        ).fetchone()
        catalog_items = db.execute("SELECT COUNT(*) FROM gold_item_features").fetchone()[0]
        possible = users * interacted_items
        density = pairs / possible if possible else 0.0
        totals = db.execute(
            """SELECT COALESCE(SUM(view_count),0),COALESCE(SUM(cart_count),0),
                      COALESCE(SUM(purchase_count),0),COALESCE(SUM(interaction_score),0)
               FROM gold_user_item_features"""
        ).fetchone()
        availability = {
            ("unknown" if value is None else str(value)): count
            for value, count in db.execute("SELECT available,COUNT(*) FROM gold_item_features GROUP BY available")
        }
        active_users = [
            {"visitor_id": visitor, "distinct_items": items, "interaction_score": score}
            for visitor, items, score in db.execute(
                """SELECT visitor_id,COUNT(*) items,SUM(interaction_score) score
                   FROM gold_user_item_features GROUP BY visitor_id
                   ORDER BY score DESC,items DESC LIMIT ?""", (top_n,)
            )
        ]
        popular_items = [
            {"item_id": item, "users": item_users, "interaction_score": score,
             "category_id": category, "available": available}
            for item, item_users, score, category, available in db.execute(
                """SELECT ui.item_id,COUNT(*) users,SUM(ui.interaction_score) score,
                          f.category_id,f.available
                   FROM gold_user_item_features ui LEFT JOIN gold_item_features f
                   ON f.item_id=ui.item_id GROUP BY ui.item_id
                   ORDER BY score DESC,users DESC LIMIT ?""", (top_n,)
            )
        ]
        categories = [
            {"category_id": category, "items": items, "users": category_users,
             "interaction_score": score}
            for category, items, category_users, score in db.execute(
                """SELECT f.category_id,COUNT(DISTINCT ui.item_id),
                          COUNT(DISTINCT ui.visitor_id),SUM(ui.interaction_score) score
                   FROM gold_user_item_features ui JOIN gold_item_features f
                   ON f.item_id=ui.item_id GROUP BY f.category_id
                   ORDER BY score DESC LIMIT ?""", (top_n,)
            )
        ]
        result = {
            "size": {"users": users, "interacted_items": interacted_items,
                     "catalog_items": catalog_items, "user_item_pairs": pairs},
            "interaction_totals": {"views": totals[0], "add_to_carts": totals[1],
                                   "purchases": totals[2], "weighted_score": totals[3]},
            "sparsity": {"possible_user_item_pairs": possible,
                         "observed_density": density, "sparsity": 1.0 - density},
            "availability": availability,
            "items_per_user": _distribution(db, "visitor_id"),
            "users_per_item": _distribution(db, "item_id"),
            "most_active_users": active_users,
            "most_common_items": popular_items,
            "most_common_categories": categories,
        }
        logger.info("Gold profile complete: %s users, %s interacted items",
                    f"{users:,}", f"{interacted_items:,}")
        return result
    finally:
        db.close()
