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
    return {f"precision@{k}": precision / users, f"recall@{k}": recall / users,
            f"ndcg@{k}": ndcg / users, f"hit_rate@{k}": hit_users / users,
            f"catalog_coverage@{k}": len(catalog) / available_count,
            "distinct_items_recommended": len(catalog)}


def evaluate_models(db_path: Path, k: int = 10) -> dict[str, Any]:
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
        fallback_users = 0
        for visitor in targets:
            history = histories.get(visitor, {})
            scores: dict[int, float] = defaultdict(float)
            for seed, weight in history.items():
                for candidate, similarity in neighbor_map.get(seed, []):
                    if candidate not in history and candidate in available:
                        scores[candidate] += weight * similarity
            ranked = [item for item, _ in sorted(scores.items(), key=lambda value: (-value[1], value[0]))]
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
        baseline_metrics = _metrics(baseline, targets, k, len(available))
        cf_metrics = _metrics(collaborative, targets, k, len(available))
        target_name, cutoff = db.execute("SELECT target,cutoff_ms FROM model_split_metadata LIMIT 1").fetchone()
        logger.info("Model evaluation complete for %s eligible users", f"{len(targets):,}")
        return {"target": target_name, "cutoff_ms": cutoff, "k": k,
                "eligible_users": len(targets),
                "warm_users": sum(bool(histories.get(user)) for user in targets),
                "cold_start_users": sum(not histories.get(user) for user in targets),
                "relevant_test_items": sum(map(len, targets.values())),
                "models": {"weighted_popularity": baseline_metrics,
                           "item_collaborative_filtering": {
                               **cf_metrics,
                               "users_requiring_popularity_fallback": fallback_users}},
                "lift_over_popularity": {
                    f"precision@{k}": cf_metrics[f"precision@{k}"] - baseline_metrics[f"precision@{k}"],
                    f"recall@{k}": cf_metrics[f"recall@{k}"] - baseline_metrics[f"recall@{k}"],
                    f"ndcg@{k}": cf_metrics[f"ndcg@{k}"] - baseline_metrics[f"ndcg@{k}"]}}
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
