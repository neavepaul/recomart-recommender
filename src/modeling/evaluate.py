"""Comparable offline evaluation for persisted recommendation models."""

from __future__ import annotations

import math
import sqlite3
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.core import connect

logger = logging.getLogger(__name__)


def _metrics(recommendations: dict[int, list[int]], targets: dict[int, set[int]],
             k: int, available_count: int) -> dict[str, float | int]:
    precision = recall = ndcg = 0.0
    hit_users = 0
    catalog: set[int] = set()
    for visitor, relevant in targets.items():
        ranked = recommendations.get(visitor, [])[:k]
        hits = len(set(ranked) & relevant)
        precision += hits / k
        recall += hits / len(relevant)
        dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(ranked) if item in relevant)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
        ndcg += dcg / ideal if ideal else 0
        hit_users += int(hits > 0)
        catalog.update(ranked)
    users = len(targets)
    if not users:
        return {"evaluated_users": 0, f"precision@{k}": 0.0,
                f"recall@{k}": 0.0, f"ndcg@{k}": 0.0,
                f"hit_rate@{k}": 0.0, f"catalog_coverage@{k}": 0.0,
                "distinct_items_recommended": 0}
    return {"evaluated_users": users,
            f"precision@{k}": precision / users, f"recall@{k}": recall / users,
            f"ndcg@{k}": ndcg / users, f"hit_rate@{k}": hit_users / users,
            f"catalog_coverage@{k}": len(catalog) / available_count,
            "distinct_items_recommended": len(catalog)}


def _segmented_metrics(recommendations: dict[int, list[int]],
                       targets: dict[int, set[int]],
                       histories: dict[int, dict[int, float]],
                       k: int, available_count: int) -> dict[str, Any]:
    warm = {user: items for user, items in targets.items() if histories.get(user)}
    cold = {user: items for user, items in targets.items() if not histories.get(user)}
    overall = _metrics(recommendations, targets, k, available_count)
    overall["segments"] = {
        "warm_users": _metrics(recommendations, warm, k, available_count),
        "cold_start_users": _metrics(recommendations, cold, k, available_count),
    }
    return overall


def evaluate_models(db_path: Path, k: int = 10,
                    content_model_dir: Path = Path("models/content")) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    db = connect(db_path)
    try:
        _require_models(db)
        logger.info("Evaluating popularity and collaborative models at K=%d", k)
        targets: dict[int, set[int]] = defaultdict(set)
        for visitor, item in db.execute("SELECT visitor_id,item_id FROM model_test_targets"):
            targets[visitor].add(item)
        if not targets:
            raise RuntimeError("No test targets; prepare a split with a longer test period")
        available = {row[0] for row in db.execute("SELECT item_id FROM silver_products WHERE available=1")}
        histories = _histories(db, set(targets))
        popularity = [row[0] for row in db.execute("SELECT item_id FROM model_popularity ORDER BY rank")]
        baseline = {visitor: [item for item in popularity if item not in histories.get(visitor, {})][:k]
                    for visitor in targets}
        neighbor_map = _needed_neighbors(db, histories)
        collaborative: dict[int, list[int]] = {}
        collaborative_raw: dict[int, list[int]] = {}
        fallback_users = 0
        candidate_pool = max(100, k * 10)
        for visitor in targets:
            history = histories.get(visitor, {})
            scores: dict[int, float] = defaultdict(float)
            for seed, weight in history.items():
                for candidate, similarity in neighbor_map.get(seed, []):
                    if candidate not in history and candidate in available:
                        scores[candidate] += weight * similarity
            ranked = [item for item, _ in sorted(scores.items(), key=lambda value: (-value[1], value[0]))]
            collaborative_raw[visitor] = ranked[:candidate_pool]
            if len(ranked) < k:
                fallback_users += 1
                ranked_set = set(ranked)
                for item in popularity:
                    if item not in history and item not in ranked_set:
                        ranked.append(item)
                        ranked_set.add(item)
                        if len(ranked) == k:
                            break
            collaborative[visitor] = ranked[:k]
        baseline_metrics = _segmented_metrics(
            baseline, targets, histories, k, len(available)
        )
        cf_metrics = _segmented_metrics(
            collaborative, targets, histories, k, len(available)
        )
        models: dict[str, Any] = {
            "weighted_popularity": baseline_metrics,
            "item_collaborative_filtering": {
                **cf_metrics,
                "users_requiring_popularity_fallback": fallback_users,
            },
        }
        existing_tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        content_lift = hybrid_lift = None
        if "model_content_metadata" in existing_tables:
            logger.info("Evaluating content and item-CF/content hybrid models")
            from src.modeling.content import recommend_content
            content_recommendations, content_raw, content_details = recommend_content(
                targets, histories, popularity, available, k,
                content_model_dir, candidate_pool,
            )
            content_metrics = _segmented_metrics(
                content_recommendations, targets, histories, k, len(available)
            )
            models["content_similarity"] = {**content_metrics, **content_details}
            hybrid: dict[int, list[int]] = {}
            hybrid_fallback = 0
            for visitor in targets:
                scores: dict[int, float] = defaultdict(float)
                for rank, item in enumerate(collaborative_raw.get(visitor, []), 1):
                    scores[item] += 0.60 / (20 + rank)
                for rank, item in enumerate(content_raw.get(visitor, []), 1):
                    scores[item] += 0.40 / (20 + rank)
                history = histories.get(visitor, {})
                ranked = [
                    item for item, _ in sorted(
                        scores.items(), key=lambda value: (-value[1], value[0])
                    ) if item not in history and item in available
                ][:k]
                if len(ranked) < k:
                    hybrid_fallback += 1
                    used = set(ranked)
                    for item in popularity:
                        if item not in history and item not in used:
                            ranked.append(item)
                            used.add(item)
                            if len(ranked) == k:
                                break
                hybrid[visitor] = ranked
            hybrid_metrics = _segmented_metrics(
                hybrid, targets, histories, k, len(available)
            )
            models["item_cf_content_hybrid"] = {
                **hybrid_metrics,
                "item_cf_weight": 0.60,
                "content_weight": 0.40,
                "fusion": "reciprocal-rank",
                "users_requiring_popularity_fallback": hybrid_fallback,
            }
            content_lift = {
                f"precision@{k}": content_metrics[f"precision@{k}"] - baseline_metrics[f"precision@{k}"],
                f"recall@{k}": content_metrics[f"recall@{k}"] - baseline_metrics[f"recall@{k}"],
                f"ndcg@{k}": content_metrics[f"ndcg@{k}"] - baseline_metrics[f"ndcg@{k}"],
            }
            hybrid_lift = {
                f"precision@{k}": hybrid_metrics[f"precision@{k}"] - baseline_metrics[f"precision@{k}"],
                f"recall@{k}": hybrid_metrics[f"recall@{k}"] - baseline_metrics[f"recall@{k}"],
                f"ndcg@{k}": hybrid_metrics[f"ndcg@{k}"] - baseline_metrics[f"ndcg@{k}"],
            }
        target_name, cutoff = db.execute("SELECT target,cutoff_ms FROM model_split_metadata LIMIT 1").fetchone()
        logger.info("Model evaluation complete for %s eligible users", f"{len(targets):,}")
        return {
            "target": target_name,
            "cutoff_ms": cutoff,
            "k": k,
            "eligible_users": len(targets),
            "warm_users": sum(bool(histories.get(user)) for user in targets),
            "cold_start_users": sum(not histories.get(user) for user in targets),
            "relevant_test_items": sum(map(len, targets.values())),
            "models": models,
            "lift_over_popularity": {
                f"precision@{k}": cf_metrics[f"precision@{k}"] - baseline_metrics[f"precision@{k}"],
                f"recall@{k}": cf_metrics[f"recall@{k}"] - baseline_metrics[f"recall@{k}"],
                f"ndcg@{k}": cf_metrics[f"ndcg@{k}"] - baseline_metrics[f"ndcg@{k}"],
            },
            "content_lift_over_popularity": content_lift,
            "hybrid_lift_over_popularity": hybrid_lift,
        }
    finally:
        db.close()


def _require_models(db: sqlite3.Connection) -> None:
    required = {"model_split_metadata", "model_train_user_items", "model_test_targets",
                "model_popularity", "model_item_similarity"}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - existing
    if missing:
        raise RuntimeError("Missing trained model tables: " + ", ".join(sorted(missing)))


def _histories(db: sqlite3.Connection, visitors: set[int]) -> dict[int, dict[int, float]]:
    db.execute("DROP TABLE IF EXISTS temp.model_evaluation_visitors")
    db.execute("CREATE TEMP TABLE model_evaluation_visitors(visitor_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO model_evaluation_visitors VALUES (?)", ((v,) for v in visitors))
    histories: dict[int, dict[int, float]] = defaultdict(dict)
    for visitor, item, score in db.execute(
        """SELECT t.visitor_id,t.item_id,t.interaction_score
           FROM model_train_user_items t JOIN model_evaluation_visitors v
           ON v.visitor_id=t.visitor_id"""):
        histories[visitor][item] = float(score)
    return histories


def _needed_neighbors(db: sqlite3.Connection,
                      histories: dict[int, dict[int, float]]) -> dict[int, list[tuple[int, float]]]:
    seeds = {item for history in histories.values() for item in history}
    db.execute("DROP TABLE IF EXISTS temp.model_evaluation_seeds")
    db.execute("CREATE TEMP TABLE model_evaluation_seeds(item_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO model_evaluation_seeds VALUES (?)", ((item,) for item in seeds))
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for source, target, similarity in db.execute(
        """SELECT s.source_item_id,s.similar_item_id,s.similarity
           FROM model_item_similarity s JOIN model_evaluation_seeds e
           ON e.item_id=s.source_item_id ORDER BY s.source_item_id,s.neighbor_rank"""):
        neighbors[source].append((target, similarity))
    return neighbors
