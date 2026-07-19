"""Cross-layer data-quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core import connect, table_counts


def validate(db_path: Path) -> dict[str, Any]:
    db = connect(db_path)
    try:
        result: dict[str, Any] = {"counts": table_counts(db), "checks": {}}
        checks = result["checks"]
        checks["valid_event_types"] = db.execute(
            "SELECT COUNT(*)=0 FROM silver_user_events "
            "WHERE event_type NOT IN ('view','addtocart','transaction')"
        ).fetchone()[0] == 1
        checks["unique_products"] = db.execute(
            "SELECT COUNT(*)=COUNT(DISTINCT item_id) FROM silver_products"
        ).fetchone()[0] == 1
        checks["unique_user_items"] = db.execute(
            "SELECT COUNT(*)=COUNT(DISTINCT visitor_id||':'||item_id) "
            "FROM gold_user_item_features"
        ).fetchone()[0] == 1
        checks["scores_match"] = db.execute(
            "SELECT COUNT(*)=0 FROM gold_user_item_features WHERE interaction_score "
            "!= view_count+3*cart_count+5*purchase_count"
        ).fetchone()[0] == 1
        checks["vectors_present"] = db.execute(
            "SELECT COUNT(*)=0 FROM gold_item_features "
            "WHERE item_feature_vector IS NULL OR item_feature_vector=''"
        ).fetchone()[0] == 1
        result["ok"] = all(checks.values())
        return result
    finally:
        db.close()

