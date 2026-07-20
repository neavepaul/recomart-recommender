"""Sparse product-content model built from Gold metadata vectors."""

from __future__ import annotations

import json
import logging
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import ROOT, connect
from src.progress import Progress

logger = logging.getLogger(__name__)
DEFAULT_CONTENT_DIR = ROOT / "models" / "content"


def _dependencies():
    try:
        import numpy as np
        import scipy.sparse as sparse
    except ImportError as error:
        raise RuntimeError("Content modeling requires numpy and scipy") from error
    return np, sparse


def build_content_model(
    db_path: Path,
    model_dir: Path = DEFAULT_CONTENT_DIR,
    vector_size: int = 256,
) -> dict[str, Any]:
    """Persist a normalized sparse item-feature matrix and catalog metadata."""
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    np, sparse = _dependencies()
    db = connect(db_path)
    try:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gold_item_features'"
        ).fetchone()
        if not exists:
            raise RuntimeError("Missing gold_item_features; build Gold first")
        item_ids = array("q")
        categories = array("q")
        parents = array("q")
        availability = array("b")
        rows = array("I")
        columns = array("I")
        values = array("f")
        count = nnz = 0
        progress = Progress("Content feature matrix", unit="items")
        for item_id, category, parent, available, encoded in db.execute(
            """SELECT item_id,category_id,parent_category_id,available,item_feature_vector
               FROM gold_item_features ORDER BY item_id"""
        ):
            item_ids.append(item_id)
            categories.append(-1 if category is None else category)
            parents.append(-1 if parent is None else parent)
            availability.append(1 if available == 1 else 0)
            for bucket, value in json.loads(encoded).items():
                bucket_index = int(bucket)
                if bucket_index >= vector_size:
                    raise ValueError(
                        f"Gold vector bucket {bucket_index} exceeds vector size {vector_size}"
                    )
                rows.append(count)
                columns.append(bucket_index)
                values.append(float(value))
                nnz += 1
            count += 1
            if count % 10_000 == 0:
                progress.update(count)
        progress.close(count)
        matrix = sparse.csr_matrix(
            (
                np.frombuffer(values, dtype=np.float32),
                (np.frombuffer(rows, dtype=np.uint32),
                 np.frombuffer(columns, dtype=np.uint32)),
            ),
            shape=(count, vector_size),
            dtype=np.float32,
        )
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        matrix = sparse.diags((1.0 / norms).astype(np.float32)) @ matrix
        model_dir = model_dir.resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(model_dir / "features.npz", matrix, compressed=True)
        np.savez_compressed(
            model_dir / "catalog.npz",
            item_ids=np.frombuffer(item_ids, dtype=np.int64),
            categories=np.frombuffer(categories, dtype=np.int64),
            parents=np.frombuffer(parents, dtype=np.int64),
            available=np.frombuffer(availability, dtype=np.int8),
        )
        metadata = {
            "algorithm": "sparse-content-cosine",
            "items": count,
            "vector_size": vector_size,
            "nonzero_features": nnz,
            "category_weight": 0.15,
            "parent_category_weight": 0.05,
            "vector_weight": 0.80,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        db.execute("DROP TABLE IF EXISTS model_content_metadata")
        db.execute("""CREATE TABLE model_content_metadata(
            model_dir TEXT NOT NULL,vector_size INTEGER NOT NULL,
            items INTEGER NOT NULL,built_at TEXT NOT NULL)""")
        db.execute(
            "INSERT INTO model_content_metadata VALUES (?,?,?,?)",
            (str(model_dir), vector_size, count, metadata["built_at"]),
        )
        db.commit()
        logger.info("Content model saved to %s", model_dir)
        return metadata
    finally:
        db.close()


def recommend_content(
    targets: dict[int, set[int]],
    histories: dict[int, dict[int, float]],
    popularity: list[int],
    available_items: set[int],
    k: int,
    model_dir: Path = DEFAULT_CONTENT_DIR,
    candidate_pool: int = 100,
    vector_weight: float = 0.80,
    category_weight: float = 0.15,
    parent_weight: float = 0.05,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[str, int]]:
    """Recommend by weighted user content profile and category affinity."""
    if min(vector_weight, category_weight, parent_weight) < 0:
        raise ValueError("content weights cannot be negative")
    if vector_weight + category_weight + parent_weight <= 0:
        raise ValueError("at least one content weight must be positive")
    np, sparse = _dependencies()
    model_dir = model_dir.resolve()
    feature_path = model_dir / "features.npz"
    catalog_path = model_dir / "catalog.npz"
    if not feature_path.exists() or not catalog_path.exists():
        raise RuntimeError(f"Content model artifacts not found under {model_dir}")
    features = sparse.load_npz(feature_path).tocsr()
    catalog = np.load(catalog_path)
    item_ids = catalog["item_ids"]
    categories = catalog["categories"]
    parents = catalog["parents"]
    available_mask = catalog["available"] == 1
    item_to_row = {int(item): row for row, item in enumerate(item_ids)}
    candidate_rows = np.asarray(
        [row for row, item in enumerate(item_ids)
         if available_mask[row] and int(item) in available_items],
        dtype=np.int32,
    )
    candidate_features = features[candidate_rows]
    candidate_ids = item_ids[candidate_rows]
    candidate_categories = categories[candidate_rows]
    candidate_parents = parents[candidate_rows]
    recommendations: dict[int, list[int]] = {}
    raw_rankings: dict[int, list[int]] = {}
    fallback_users = 0
    progress = Progress("Content recommendations", total=len(targets), unit="users")
    for completed, visitor in enumerate(targets, 1):
        history = histories.get(visitor, {})
        history_rows = [item_to_row[item] for item in history if item in item_to_row]
        ranked: list[int] = []
        if history_rows:
            weights = np.asarray(
                [math_log_weight(history[int(item_ids[row])]) for row in history_rows],
                dtype=np.float32,
            )
            profile = features[history_rows].multiply(weights[:, None]).sum(axis=0)
            profile = np.asarray(profile).ravel().astype(np.float32)
            norm = float(np.linalg.norm(profile))
            if norm > 0:
                profile /= norm
                scores = np.asarray(candidate_features @ profile).ravel() * vector_weight
                category_affinity: dict[int, float] = {}
                parent_affinity: dict[int, float] = {}
                total_weight = float(weights.sum()) or 1.0
                for row, weight in zip(history_rows, weights):
                    category = int(categories[row])
                    parent = int(parents[row])
                    if category >= 0:
                        category_affinity[category] = category_affinity.get(category, 0.0) + float(weight) / total_weight
                    if parent >= 0:
                        parent_affinity[parent] = parent_affinity.get(parent, 0.0) + float(weight) / total_weight
                scores += category_weight * np.fromiter(
                    (category_affinity.get(int(value), 0.0) for value in candidate_categories),
                    dtype=np.float32, count=len(candidate_rows),
                )
                scores += parent_weight * np.fromiter(
                    (parent_affinity.get(int(value), 0.0) for value in candidate_parents),
                    dtype=np.float32, count=len(candidate_rows),
                )
                for item in history:
                    row = item_to_row.get(item)
                    if row is not None:
                        position = np.searchsorted(candidate_rows, row)
                        if position < len(candidate_rows) and candidate_rows[position] == row:
                            scores[position] = -np.inf
                take = min(candidate_pool, len(scores))
                if take:
                    indexes = np.argpartition(scores, -take)[-take:]
                    indexes = indexes[np.argsort(scores[indexes])[::-1]]
                    ranked = [int(candidate_ids[index]) for index in indexes if np.isfinite(scores[index])]
        raw_rankings[visitor] = ranked
        completed_ranking = ranked[:k]
        if len(completed_ranking) < k:
            fallback_users += 1
            used = set(completed_ranking)
            for item in popularity:
                if item not in history and item not in used:
                    completed_ranking.append(item)
                    used.add(item)
                    if len(completed_ranking) == k:
                        break
        recommendations[visitor] = completed_ranking
        progress.update(completed)
    progress.close(len(targets))
    return recommendations, raw_rankings, {
        "users_requiring_popularity_fallback": fallback_users,
    }


def math_log_weight(value: float) -> float:
    import math
    return math.log1p(float(value))
