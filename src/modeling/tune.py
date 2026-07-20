"""Leakage-safe validation tuning for content and hybrid ranking weights."""

from __future__ import annotations

import itertools
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import connect
from src.modeling.content import recommend_content
from src.modeling.evaluate import _metrics
from src.modeling.split import TARGET_EVENTS, _iso
from src.progress import Progress, sqlite_activity

logger = logging.getLogger(__name__)
DAY_MS = 86_400_000
CONTENT_GRID = (
    (1.00, 0.00, 0.00),
    (0.90, 0.075, 0.025),
    (0.80, 0.15, 0.05),
    (0.70, 0.20, 0.10),
    (0.60, 0.25, 0.15),
)
ITEM_CF_GRID = tuple(value / 10 for value in range(11))
RRF_CONSTANT = 20.0


def tune_hybrid(
    db_path: Path,
    content_model_dir: Path = Path("models/content"),
    validation_days: int = 14,
    validation_cutoff_ms: int | None = None,
    k: int = 10,
    max_history: int = 30,
    min_cooccurrence: int = 2,
    neighbors: int = 50,
) -> dict[str, Any]:
    """Tune ranking weights before the final cutoff and persist the winner."""
    if validation_days <= 0 or k <= 0:
        raise ValueError("validation_days and k must be positive")
    if max_history < 2 or min_cooccurrence < 1 or neighbors < 1:
        raise ValueError("invalid collaborative-filtering training parameters")
    db = connect(db_path)
    try:
        row = db.execute(
            "SELECT cutoff_ms,target,minimum_event_ms FROM model_split_metadata LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Missing temporal split; run prepare-model-data first")
        final_cutoff, target, minimum = row
        validation_cutoff = (
            validation_cutoff_ms
            if validation_cutoff_ms is not None
            else final_cutoff - validation_days * DAY_MS
        )
        if not minimum < validation_cutoff < final_cutoff:
            raise ValueError(
                "validation cutoff must be inside the training period: "
                f"{_iso(minimum)} < cutoff < {_iso(final_cutoff)}"
            )
        logger.info(
            "Tuning hybrid on validation window %s to %s",
            _iso(validation_cutoff), _iso(final_cutoff),
        )
        _build_validation_split(db, target, validation_cutoff, final_cutoff)
        _train_validation_cf(db, max_history, min_cooccurrence, neighbors)
        targets = _targets(db)
        histories = _histories(db, set(targets))
        warm_targets = {
            visitor: items for visitor, items in targets.items() if histories.get(visitor)
        }
        if not warm_targets:
            raise RuntimeError("Validation window contains no warm users with novel targets")
        available = {
            row[0] for row in db.execute(
                "SELECT item_id FROM silver_products WHERE available=1"
            )
        }
        popularity = [
            row[0] for row in db.execute(
                "SELECT item_id FROM temp.tuning_popularity ORDER BY rank"
            )
        ]
        candidate_pool = max(100, k * 10)
        collaborative_raw = _collaborative_rankings(
            db, targets, histories, available, candidate_pool
        )
        best: dict[str, Any] | None = None
        trials: list[dict[str, Any]] = []
        for vector_weight, category_weight, parent_weight in CONTENT_GRID:
            _, content_raw, _ = recommend_content(
                targets, histories, popularity, available, k,
                content_model_dir, candidate_pool, vector_weight,
                category_weight, parent_weight,
            )
            for item_cf_weight in ITEM_CF_GRID:
                content_weight = round(1.0 - item_cf_weight, 10)
                recommendations = _fuse(
                    warm_targets, histories, collaborative_raw, content_raw,
                    popularity, available, k, item_cf_weight, content_weight,
                )
                metrics = _metrics(
                    recommendations, warm_targets, k, len(available)
                )
                trial = {
                    "vector_weight": vector_weight,
                    "category_weight": category_weight,
                    "parent_weight": parent_weight,
                    "item_cf_weight": item_cf_weight,
                    "content_weight": content_weight,
                    f"precision@{k}": metrics[f"precision@{k}"],
                    f"recall@{k}": metrics[f"recall@{k}"],
                    f"ndcg@{k}": metrics[f"ndcg@{k}"],
                    f"hit_rate@{k}": metrics[f"hit_rate@{k}"],
                }
                trials.append(trial)
                score = (
                    trial[f"ndcg@{k}"], trial[f"recall@{k}"],
                    trial[f"precision@{k}"], -abs(item_cf_weight - 0.5),
                )
                if best is None or score > best["_score"]:
                    best = {**trial, "_score": score}
        assert best is not None
        best.pop("_score")
        created_at = datetime.now(timezone.utc).isoformat()
        _persist_config(
            db, best, RRF_CONSTANT, k, validation_cutoff,
            final_cutoff, len(warm_targets), created_at,
        )
        result = {
            "selection_metric": f"warm_user_ndcg@{k}",
            "validation_period": {
                "start_inclusive": _iso(validation_cutoff),
                "end_exclusive": _iso(final_cutoff),
            },
            "validation_users": len(targets),
            "warm_validation_users": len(warm_targets),
            "validation_targets": sum(map(len, targets.values())),
            "configurations_tested": len(trials),
            "best": {**best, "rrf_constant": RRF_CONSTANT},
            "top_configurations": sorted(
                trials,
                key=lambda value: (
                    -value[f"ndcg@{k}"], -value[f"recall@{k}"],
                    -value[f"precision@{k}"],
                ),
            )[:5],
        }
        model_dir = content_model_dir.resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "tuning.json").write_text(
            json.dumps({**result, "created_at": created_at}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Hybrid tuning complete: validation NDCG@%d %.6f, item-CF/content %.1f/%.1f",
            k, best[f"ndcg@{k}"], best["item_cf_weight"],
            best["content_weight"],
        )
        return result
    finally:
        db.close()


def _build_validation_split(db, target: str, start: int, end: int) -> None:
    events = TARGET_EVENTS[target]
    placeholders = ",".join("?" for _ in events)
    db.executescript("""
        DROP TABLE IF EXISTS temp.tuning_train;
        DROP TABLE IF EXISTS temp.tuning_targets;
        DROP TABLE IF EXISTS temp.tuning_popularity;
        DROP TABLE IF EXISTS temp.tuning_pairs;
        DROP TABLE IF EXISTS temp.tuning_similarity;
    """)
    with sqlite_activity(db, "Validation training features"):
        db.execute("""CREATE TEMP TABLE tuning_train AS
            SELECT visitor_id,item_id,SUM(event_type='view') view_count,
                   SUM(event_type='addtocart') cart_count,
                   SUM(event_type='transaction') purchase_count,
                   SUM(CASE event_type WHEN 'view' THEN 1
                       WHEN 'addtocart' THEN 3 ELSE 5 END) interaction_score,
                   MAX(event_timestamp_ms) last_interaction_timestamp_ms
            FROM silver_user_events WHERE event_timestamp_ms<?
            GROUP BY visitor_id,item_id""", (start,))
    db.execute(
        "CREATE UNIQUE INDEX temp.ix_tuning_train_user_item "
        "ON tuning_train(visitor_id,item_id)"
    )
    with sqlite_activity(db, "Validation future targets"):
        db.execute(f"""CREATE TEMP TABLE tuning_targets AS
            SELECT e.visitor_id,e.item_id,COUNT(*) target_events
            FROM silver_user_events e JOIN silver_products p
              ON p.item_id=e.item_id AND p.available=1
            WHERE e.event_timestamp_ms>=? AND e.event_timestamp_ms<?
              AND e.event_type IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM tuning_train t
                WHERE t.visitor_id=e.visitor_id AND t.item_id=e.item_id
              )
            GROUP BY e.visitor_id,e.item_id""", (start, end, *events))
    db.execute(
        "CREATE UNIQUE INDEX temp.ix_tuning_targets_user_item "
        "ON tuning_targets(visitor_id,item_id)"
    )


def _train_validation_cf(
    db, max_history: int, min_cooccurrence: int, neighbors: int
) -> None:
    db.execute("""CREATE TEMP TABLE tuning_popularity AS
        SELECT t.item_id,SUM(t.interaction_score) score,
               ROW_NUMBER() OVER (
                 ORDER BY SUM(t.interaction_score) DESC,t.item_id
               ) rank
        FROM tuning_train t JOIN silver_products p
          ON p.item_id=t.item_id AND p.available=1
        GROUP BY t.item_id""")
    db.execute("""CREATE TEMP TABLE tuning_pairs (
        item_i INTEGER NOT NULL,item_j INTEGER NOT NULL,dot_product REAL NOT NULL,
        cooccurring_users INTEGER NOT NULL,PRIMARY KEY(item_i,item_j))""")
    upsert = """INSERT INTO tuning_pairs VALUES (?,?,?,1)
        ON CONFLICT(item_i,item_j) DO UPDATE SET
        dot_product=dot_product+excluded.dot_product,
        cooccurring_users=cooccurring_users+1"""
    current_user = None
    history: list[tuple[int, float]] = []
    batch: list[tuple[int, int, float]] = []
    contributions = 0
    progress = Progress("Validation collaborative pairs", unit="pairs")

    def add_history(items: list[tuple[int, float]]) -> None:
        nonlocal contributions, batch
        selected = sorted(items[:max_history])
        for (item_i, score_i), (item_j, score_j) in itertools.combinations(selected, 2):
            batch.append((item_i, item_j, score_i * score_j))
            contributions += 1
            if len(batch) >= 50_000:
                db.executemany(upsert, batch)
                batch = []
                progress.update(contributions)

    query = """SELECT visitor_id,item_id,interaction_score FROM tuning_train
               ORDER BY visitor_id,interaction_score DESC,last_interaction_timestamp_ms DESC"""
    for visitor, item, score in db.execute(query):
        if current_user is not None and visitor != current_user:
            add_history(history)
            history = []
        current_user = visitor
        history.append((item, float(score)))
    if current_user is not None:
        add_history(history)
    if batch:
        db.executemany(upsert, batch)
    progress.close(contributions)
    db.create_function("SQRT", 1, math.sqrt)
    with sqlite_activity(db, "Validation collaborative similarities"):
        db.execute("""CREATE TEMP TABLE tuning_norms AS
            SELECT item_id,SUM(interaction_score*interaction_score) norm_squared
            FROM tuning_train GROUP BY item_id""")
        db.execute(
            "CREATE UNIQUE INDEX temp.ix_tuning_norms_item ON tuning_norms(item_id)"
        )
        db.execute("""CREATE TEMP TABLE tuning_similarity AS
            WITH candidates AS (
              SELECT p.item_i source_item_id,p.item_j similar_item_id,
                     p.dot_product/SQRT(ni.norm_squared*nj.norm_squared) similarity,
                     p.cooccurring_users
              FROM tuning_pairs p JOIN tuning_norms ni ON ni.item_id=p.item_i
              JOIN tuning_norms nj ON nj.item_id=p.item_j
              WHERE p.cooccurring_users>=?
              UNION ALL
              SELECT p.item_j,p.item_i,
                     p.dot_product/SQRT(ni.norm_squared*nj.norm_squared),
                     p.cooccurring_users
              FROM tuning_pairs p JOIN tuning_norms ni ON ni.item_id=p.item_i
              JOIN tuning_norms nj ON nj.item_id=p.item_j
              WHERE p.cooccurring_users>=?
            )
            SELECT source_item_id,similar_item_id,similarity,cooccurring_users,
                   neighbor_rank FROM (
              SELECT *,ROW_NUMBER() OVER (
                PARTITION BY source_item_id
                ORDER BY similarity DESC,cooccurring_users DESC,similar_item_id
              ) neighbor_rank FROM candidates
            ) WHERE neighbor_rank<=?""",
            (min_cooccurrence, min_cooccurrence, neighbors),
        )
        db.execute(
            "CREATE INDEX temp.ix_tuning_similarity_source "
            "ON tuning_similarity(source_item_id,neighbor_rank)"
        )


def _targets(db) -> dict[int, set[int]]:
    targets: dict[int, set[int]] = defaultdict(set)
    for visitor, item in db.execute("SELECT visitor_id,item_id FROM temp.tuning_targets"):
        targets[visitor].add(item)
    return targets


def _histories(db, visitors: set[int]) -> dict[int, dict[int, float]]:
    db.execute("CREATE TEMP TABLE tuning_visitors(visitor_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO tuning_visitors VALUES (?)", ((v,) for v in visitors))
    histories: dict[int, dict[int, float]] = defaultdict(dict)
    for visitor, item, score in db.execute("""
        SELECT t.visitor_id,t.item_id,t.interaction_score
        FROM tuning_train t JOIN tuning_visitors v ON v.visitor_id=t.visitor_id
    """):
        histories[visitor][item] = float(score)
    return histories


def _collaborative_rankings(
    db, targets, histories, available: set[int], candidate_pool: int
) -> dict[int, list[int]]:
    seeds = {item for history in histories.values() for item in history}
    db.execute("CREATE TEMP TABLE tuning_seeds(item_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO tuning_seeds VALUES (?)", ((item,) for item in seeds))
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for source, candidate, similarity in db.execute("""
        SELECT s.source_item_id,s.similar_item_id,s.similarity
        FROM tuning_similarity s JOIN tuning_seeds e
          ON e.item_id=s.source_item_id
        ORDER BY s.source_item_id,s.neighbor_rank
    """):
        neighbors[source].append((candidate, similarity))
    rankings: dict[int, list[int]] = {}
    for visitor in targets:
        history = histories.get(visitor, {})
        scores: dict[int, float] = defaultdict(float)
        for seed, weight in history.items():
            for candidate, similarity in neighbors.get(seed, []):
                if candidate not in history and candidate in available:
                    scores[candidate] += weight * similarity
        rankings[visitor] = [
            item for item, _ in sorted(
                scores.items(), key=lambda value: (-value[1], value[0])
            )[:candidate_pool]
        ]
    return rankings


def _fuse(
    targets, histories, collaborative, content, popularity, available,
    k: int, item_cf_weight: float, content_weight: float,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    for visitor in targets:
        scores: dict[int, float] = defaultdict(float)
        for rank, item in enumerate(collaborative.get(visitor, []), 1):
            scores[item] += item_cf_weight / (RRF_CONSTANT + rank)
        for rank, item in enumerate(content.get(visitor, []), 1):
            scores[item] += content_weight / (RRF_CONSTANT + rank)
        history = histories.get(visitor, {})
        ranked = [
            item for item, _ in sorted(
                scores.items(), key=lambda value: (-value[1], value[0])
            ) if item not in history and item in available
        ][:k]
        used = set(ranked)
        for item in popularity:
            if len(ranked) == k:
                break
            if item not in history and item not in used:
                ranked.append(item)
                used.add(item)
        recommendations[visitor] = ranked
    return recommendations


def _persist_config(
    db, best, rrf_constant: float, k: int, validation_cutoff: int,
    final_cutoff: int, users: int, created_at: str,
) -> None:
    db.execute("DROP TABLE IF EXISTS model_hybrid_config")
    db.execute("""CREATE TABLE model_hybrid_config (
        vector_weight REAL NOT NULL,category_weight REAL NOT NULL,
        parent_weight REAL NOT NULL,item_cf_weight REAL NOT NULL,
        content_weight REAL NOT NULL,rrf_constant REAL NOT NULL,
        validation_k INTEGER NOT NULL,validation_cutoff_ms INTEGER NOT NULL,
        final_cutoff_ms INTEGER NOT NULL,warm_validation_users INTEGER NOT NULL,
        validation_ndcg REAL NOT NULL,created_at TEXT NOT NULL)""")
    db.execute(
        "INSERT INTO model_hybrid_config VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            best["vector_weight"], best["category_weight"],
            best["parent_weight"], best["item_cf_weight"],
            best["content_weight"], rrf_constant, k, validation_cutoff,
            final_cutoff, users, best[f"ndcg@{k}"], created_at,
        ),
    )
    db.commit()
