"""Online-style recommendation inference from persisted RecoMart models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.core import connect
from src.modeling.content import DEFAULT_CONTENT_DIR, recommend_content
from src.modeling.evaluate import _hybrid_config


def recommend(
    db_path: Path,
    visitor_id: int,
    k: int = 10,
    content_model_dir: Path = DEFAULT_CONTENT_DIR,
    max_history: int = 30,
) -> dict[str, Any]:
    """Return unseen, available hybrid recommendations for one visitor."""
    if visitor_id < 0:
        raise ValueError("visitor_id must be non-negative")
    if k < 1:
        raise ValueError("k must be positive")
    if max_history < 1:
        raise ValueError("max_history must be positive")
    content_model_dir = content_model_dir.resolve()
    db = connect(db_path)
    try:
        existing = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "gold_user_item_features", "silver_products",
            "model_popularity", "model_item_similarity",
        }
        missing = sorted(required - existing)
        if missing:
            raise RuntimeError("Missing inference tables: " + ", ".join(missing))
        history = {
            int(item): float(score)
            for item, score in db.execute(
                """SELECT item_id,interaction_score
                   FROM gold_user_item_features WHERE visitor_id=?
                   ORDER BY last_interaction_timestamp DESC,
                            interaction_score DESC,item_id
                   LIMIT ?""",
                (visitor_id, max_history),
            )
        }
        available = {
            int(row[0]) for row in db.execute(
                "SELECT item_id FROM silver_products WHERE available=1"
            )
        }
        popularity = [
            int(row[0]) for row in db.execute(
                """SELECT p.item_id FROM model_popularity p
                   JOIN silver_products s ON s.item_id=p.item_id
                   WHERE s.available=1 ORDER BY p.rank"""
            )
        ]
        candidate_pool = max(100, k * 10)
        cf_scores: dict[int, float] = defaultdict(float)
        if history:
            placeholders = ",".join("?" for _ in history)
            for source, candidate, similarity in db.execute(
                f"""SELECT source_item_id,similar_item_id,similarity
                    FROM model_item_similarity
                    WHERE source_item_id IN ({placeholders})
                    ORDER BY source_item_id,neighbor_rank""",
                list(history),
            ):
                candidate = int(candidate)
                if candidate not in history and candidate in available:
                    cf_scores[candidate] += history[int(source)] * float(similarity)
        cf_ranking = [
            item for item, _ in sorted(
                cf_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )[:candidate_pool]
        ]
        config = _hybrid_config(db)
        content_ranking: list[int] = []
        content_enabled = (
            "model_content_metadata" in existing
            and (content_model_dir / "features.npz").exists()
            and (content_model_dir / "catalog.npz").exists()
        )
        if content_enabled and history:
            _, raw, _ = recommend_content(
                {visitor_id: set()}, {visitor_id: history}, popularity,
                available, k, content_model_dir, candidate_pool,
                config["vector_weight"], config["category_weight"],
                config["parent_weight"],
            )
            content_ranking = raw.get(visitor_id, [])
        fused: dict[int, float] = defaultdict(float)
        sources: dict[int, set[str]] = defaultdict(set)
        for rank, item in enumerate(cf_ranking, 1):
            fused[item] += config["item_cf_weight"] / (
                config["rrf_constant"] + rank
            )
            sources[item].add("item_cf")
        for rank, item in enumerate(content_ranking, 1):
            fused[item] += config["content_weight"] / (
                config["rrf_constant"] + rank
            )
            sources[item].add("content")
        ranked = [
            item for item, _ in sorted(
                fused.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if item not in history and item in available
        ][:k]
        fallback_used = len(ranked) < k
        used = set(ranked)
        for item in popularity:
            if len(ranked) == k:
                break
            if item not in history and item not in used:
                ranked.append(item)
                used.add(item)
                sources[item].add("popularity")
        metadata: dict[int, tuple[int | None, int | None]] = {}
        if ranked and "gold_item_features" in existing:
            placeholders = ",".join("?" for _ in ranked)
            metadata = {
                int(item): (category, parent)
                for item, category, parent in db.execute(
                    f"""SELECT item_id,category_id,parent_category_id
                        FROM gold_item_features
                        WHERE item_id IN ({placeholders})""",
                    ranked,
                )
            }
        recommendations = []
        for rank, item in enumerate(ranked, 1):
            category, parent = metadata.get(item, (None, None))
            recommendations.append({
                "rank": rank,
                "item_id": item,
                "score": fused.get(item, 0.0),
                "sources": sorted(sources[item]),
                "category_id": category,
                "parent_category_id": parent,
            })
        return {
            "visitor_id": visitor_id,
            "k": k,
            "history_items_used": len(history),
            "model": (
                "weighted_popularity_fallback" if not history
                else "item_cf_content_hybrid" if content_enabled
                else "item_cf_with_popularity_fallback"
            ),
            "content_model_loaded": content_enabled,
            "fallback_used": fallback_used,
            "weights": config,
            "recommendations": recommendations,
        }
    finally:
        db.close()
