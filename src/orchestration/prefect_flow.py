"""Prefect DAGs for RecoMart curation, modeling, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from prefect import flow, task

from src.core import DEFAULT_DB, RAW, connect, table_counts
from src.feature_store import register_features
from src.metadata import finish_pipeline_run, record_dataset, start_pipeline_run
from src.model_tracking import log_existing_artifact, log_result, model_run
from src.modeling import (
    build_content_model, evaluate_models, prepare_model_data,
    train_models, tune_hybrid,
)
from src.pipelines.bronze import build_bronze
from src.pipelines.features import build_features
from src.pipelines.gold import build_gold
from src.pipelines.silver import build_silver
from src.validation import validate


@task(name="ingest-bronze", retries=1, retry_delay_seconds=5)
def bronze_task(
    db_path: Path, raw_dir: Path, run_id: str, speed: float,
    limit: int | None, api_page_size: int,
) -> dict[str, int]:
    result = build_bronze(db_path, raw_dir, speed, limit, api_page_size)
    record_dataset(
        db_path, run_id, "bronze_events", "bronze",
        "Timestamp-preserving clickstream replay into the raw event schema.",
        [], "events.csv", raw_dir / "events.csv",
    )
    record_dataset(
        db_path, run_id, "bronze_category_tree", "bronze",
        "Direct batch CSV ingestion with nullable parent category IDs.",
        [], "category_tree.csv", raw_dir / "category_tree.csv",
    )
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        record_dataset(
            db_path, run_id, "bronze_item_properties", "bronze",
            "CSV-backed mock REST API ingestion; values preserve embedded commas.",
            [], filename, raw_dir / filename,
        )
    return result


@task(name="build-silver")
def silver_task(db_path: Path, run_id: str) -> dict[str, int]:
    db = connect(db_path)
    try:
        build_silver(db)
        result = table_counts(db)
    finally:
        db.close()
    record_dataset(
        db_path, run_id, "silver_user_events", "silver",
        "Validate event type and identifiers; standardize names and timestamps.",
        ["bronze_events"],
    )
    record_dataset(
        db_path, run_id, "silver_products", "silver",
        "Select latest item/property values and consolidate each item into one product.",
        ["bronze_item_properties"],
    )
    record_dataset(
        db_path, run_id, "silver_category_hierarchy", "silver",
        "Rename and type the category-parent reference hierarchy.",
        ["bronze_category_tree"],
    )
    return result


@task(name="build-gold")
def gold_task(db_path: Path, run_id: str, vector_size: int) -> dict[str, int]:
    db = connect(db_path)
    try:
        build_gold(db, vector_size)
        result = table_counts(db)
    finally:
        db.close()
    record_dataset(
        db_path, run_id, "gold_user_item_features", "gold",
        "Aggregate views, carts, purchases, weighted interaction score, and recency.",
        ["silver_user_events"],
    )
    record_dataset(
        db_path, run_id, "gold_item_features", "gold",
        f"Hash anonymous properties and category hierarchy into {vector_size} dimensions.",
        ["silver_products", "silver_category_hierarchy"],
    )
    return result


@task(name="validate-curated-data")
def validation_task(db_path: Path) -> dict:
    result = validate(db_path)
    if not result["ok"]:
        raise RuntimeError(f"Data validation failed: {result}")
    return result


@task(name="build-features")
def features_task(
    db_path: Path, run_id: str, neighbors: int,
    min_cooccurrence: int, max_history: int,
) -> dict[str, int]:
    db = connect(db_path)
    try:
        build_features(db, neighbors, min_cooccurrence, max_history)
        result = table_counts(db)
    finally:
        db.close()
    record_dataset(
        db_path, run_id, "feature_user_activity", "feature_store",
        "Per-user activity frequency, totals, and average interaction score.",
        ["gold_user_item_features"],
    )
    record_dataset(
        db_path, run_id, "feature_item_popularity", "feature_store",
        "Per-item popularity, average interaction score, conversion, and rank.",
        ["gold_user_item_features", "gold_item_features"],
    )
    record_dataset(
        db_path, run_id, "feature_item_cooccurrence", "feature_store",
        "Weighted item cosine co-occurrence neighbours from Gold interactions.",
        ["gold_user_item_features"],
    )
    return result


@task(name="register-features")
def registry_task(db_path: Path, run_id: str, retention: int) -> dict:
    manifest = register_features(db_path, run_id=run_id, retention=retention)
    for view in manifest["feature_views"]:
        record_dataset(
            db_path, run_id, view["source_table"] + "_versions", "feature_store",
            "Append-only versioned snapshot registered in the feature store.",
            [view["source_table"]],
        )
    return manifest


@task(name="prepare-temporal-model-data")
def split_task(db_path: Path, run_id: str, target: str, test_days: int) -> dict:
    result = prepare_model_data(db_path, target, test_days)
    record_dataset(
        db_path, run_id, "model_train_user_items", "feature_store",
        "Point-in-time pre-cutoff user-item aggregation for model training.",
        ["silver_user_events"],
    )
    record_dataset(
        db_path, run_id, "model_test_targets", "evaluation",
        "Novel available target interactions from the final temporal test window.",
        ["silver_user_events", "silver_products"],
    )
    return result


@task(name="train-item-cf")
def train_task(
    db_path: Path, run_id: str, max_history: int,
    min_cooccurrence: int, neighbors: int,
) -> dict:
    parameters = {
        "max_history": max_history, "min_cooccurrence": min_cooccurrence,
        "neighbors": neighbors,
    }
    with model_run(
        "item-cf-training", parameters,
        {"pipeline_run_id": run_id, "model_family": "item-cf"},
    ) as mlflow:
        result = train_models(db_path, max_history, min_cooccurrence, neighbors)
        log_result(mlflow, result, "training/result.json")
    record_dataset(
        db_path, run_id, "model_item_similarity", "feature_store",
        "Weighted item cosine co-occurrence neighbors from pre-cutoff interactions.",
        ["model_train_user_items"],
    )
    return result


@task(name="build-content-model")
def content_task(
    db_path: Path, model_dir: Path, run_id: str, vector_size: int,
) -> dict:
    with model_run(
        "content-model-build", {"vector_size": vector_size},
        {"pipeline_run_id": run_id, "model_family": "content"},
    ) as mlflow:
        result = build_content_model(db_path, model_dir, vector_size)
        log_result(mlflow, result, "training/result.json")
        log_existing_artifact(mlflow, model_dir / "metadata.json", "model_metadata")
    return result


@task(name="tune-hybrid")
def tune_task(
    db_path: Path, model_dir: Path, run_id: str,
    validation_days: int, k: int, max_history: int,
    min_cooccurrence: int, neighbors: int,
) -> dict:
    parameters = {
        "validation_days": validation_days, "k": k,
        "max_history": max_history, "min_cooccurrence": min_cooccurrence,
        "neighbors": neighbors,
    }
    with model_run(
        "hybrid-weight-tuning", parameters,
        {"pipeline_run_id": run_id, "model_family": "hybrid"},
    ) as mlflow:
        result = tune_hybrid(
            db_path, model_dir, validation_days, None, k,
            max_history, min_cooccurrence, neighbors,
        )
        mlflow.log_params({
            f"selected_{key}": value for key, value in result["best"].items()
            if "@" not in key
        })
        log_result(mlflow, result, "tuning/result.json")
        log_existing_artifact(mlflow, model_dir / "tuning.json", "model_metadata")
    record_dataset(
        db_path, run_id, "model_hybrid_config", "feature_store",
        "Grid-search content and fusion weights on a pre-test validation window.",
        ["model_train_user_items", "model_item_similarity", "gold_item_features"],
    )
    return result


@task(name="evaluate-models")
def evaluation_task(
    db_path: Path, model_dir: Path, report_path: Path, run_id: str, k: int,
) -> dict:
    with model_run(
        "final-model-evaluation", {"k": k},
        {"pipeline_run_id": run_id, "evaluation": "final-test"},
    ) as mlflow:
        result = evaluate_models(db_path, k, model_dir)
        log_result(mlflow, result, "evaluation/result.json")
    hybrid = result["models"]["item_cf_content_hybrid"]
    item_cf = result["models"]["item_collaborative_filtering"]
    metrics = {
        "k": result["k"],
        "eligible_users": result["eligible_users"],
        "warm_users": result["warm_users"],
        "cold_start_users": result["cold_start_users"],
        "hybrid": {
            "precision": hybrid[f"precision@{k}"],
            "recall": hybrid[f"recall@{k}"],
            "ndcg": hybrid[f"ndcg@{k}"],
            "hit_rate": hybrid[f"hit_rate@{k}"],
        },
        "hybrid_warm": {
            "precision": hybrid["segments"]["warm_users"][f"precision@{k}"],
            "recall": hybrid["segments"]["warm_users"][f"recall@{k}"],
            "ndcg": hybrid["segments"]["warm_users"][f"ndcg@{k}"],
            "hit_rate": hybrid["segments"]["warm_users"][f"hit_rate@{k}"],
        },
        "item_cf_warm": {
            "precision": item_cf["segments"]["warm_users"][f"precision@{k}"],
            "recall": item_cf["segments"]["warm_users"][f"recall@{k}"],
            "ndcg": item_cf["segments"]["warm_users"][f"ndcg@{k}"],
            "hit_rate": item_cf["segments"]["warm_users"][f"hit_rate@{k}"],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result


@flow(name="recomart-curation", log_prints=True)
def curation_flow(
    db_path: Path = DEFAULT_DB, raw_dir: Path = RAW,
    api_page_size: int = 100_000, vector_size: int = 256,
    neighbors: int = 50, min_cooccurrence: int = 2, max_history: int = 30,
    feature_retention: int = 5, speed: float = 0, limit: int | None = None,
) -> dict:
    run_id = str(uuid4())
    parameters = {
        "db_path": str(db_path), "raw_dir": str(raw_dir),
        "api_page_size": api_page_size, "vector_size": vector_size,
        "neighbors": neighbors, "min_cooccurrence": min_cooccurrence,
        "max_history": max_history, "feature_retention": feature_retention,
        "speed": speed, "limit": limit,
    }
    start_pipeline_run(db_path, run_id, "recomart-curation", parameters)
    try:
        bronze = bronze_task(db_path, raw_dir, run_id, speed, limit, api_page_size)
        silver = silver_task(db_path, run_id)
        gold = gold_task(db_path, run_id, vector_size)
        features = features_task(
            db_path, run_id, neighbors, min_cooccurrence, max_history
        )
        registry = registry_task(db_path, run_id, feature_retention)
        checks = validation_task(db_path)
        finish_pipeline_run(db_path, run_id, "COMPLETED")
        return {"run_id": run_id, "bronze": bronze, "silver": silver,
                "gold": gold, "features": features, "registry": registry,
                "validation": checks}
    except Exception as error:
        finish_pipeline_run(db_path, run_id, "FAILED", str(error))
        raise


@flow(name="recomart-modeling", log_prints=True)
def modeling_flow(
    db_path: Path = DEFAULT_DB, model_dir: Path = Path("models/content"),
    report_path: Path = Path("reports/model_metrics.json"),
    target: str = "transaction", test_days: int = 14,
    validation_days: int = 14, vector_size: int = 256, k: int = 10,
    max_history: int = 30, min_cooccurrence: int = 2, neighbors: int = 50,
) -> dict:
    run_id = str(uuid4())
    parameters = {key: str(value) for key, value in locals().items()}
    start_pipeline_run(db_path, run_id, "recomart-modeling", parameters)
    try:
        split = split_task(db_path, run_id, target, test_days)
        trained = train_task(db_path, run_id, max_history, min_cooccurrence, neighbors)
        content = content_task(db_path, model_dir, run_id, vector_size)
        tuned = tune_task(
            db_path, model_dir, run_id, validation_days, k,
            max_history, min_cooccurrence, neighbors,
        )
        evaluation = evaluation_task(db_path, model_dir, report_path, run_id, k)
        finish_pipeline_run(db_path, run_id, "COMPLETED")
        return {"run_id": run_id, "split": split, "training": trained,
                "content": content, "tuning": tuned, "evaluation": evaluation}
    except Exception as error:
        finish_pipeline_run(db_path, run_id, "FAILED", str(error))
        raise


@flow(name="recomart-full-pipeline", log_prints=True)
def recomart_flow(
    db_path: Path = DEFAULT_DB, raw_dir: Path = RAW,
    model_dir: Path = Path("models/content"),
    report_path: Path = Path("reports/model_metrics.json"),
    api_page_size: int = 100_000, vector_size: int = 256,
    target: str = "transaction", test_days: int = 14,
    validation_days: int = 14, k: int = 10,
    max_history: int = 30, min_cooccurrence: int = 2, neighbors: int = 50,
    feature_retention: int = 5, speed: float = 0,
    limit: int | None = None,
) -> dict:
    """Run the complete curation and modeling DAG with deployable parameters."""
    curated = curation_flow(
        db_path=db_path, raw_dir=raw_dir, api_page_size=api_page_size,
        vector_size=vector_size, neighbors=neighbors,
        min_cooccurrence=min_cooccurrence, max_history=max_history,
        feature_retention=feature_retention, speed=speed, limit=limit,
    )
    modeled = modeling_flow(
        db_path=db_path, model_dir=model_dir, report_path=report_path,
        target=target, test_days=test_days, validation_days=validation_days,
        vector_size=vector_size, k=k, max_history=max_history,
        min_cooccurrence=min_cooccurrence, neighbors=neighbors,
    )
    return {"curation": curated, "modeling": modeled}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--mode", choices=("full", "curation", "modeling"), default="full")
    command.add_argument("--db", type=Path, default=DEFAULT_DB)
    command.add_argument("--raw-dir", type=Path, default=RAW)
    command.add_argument("--model-dir", type=Path, default=Path("models/content"))
    command.add_argument("--report", type=Path, default=Path("reports/model_metrics.json"))
    command.add_argument("--api-page-size", type=int, default=100_000)
    command.add_argument("--vector-size", type=int, default=256)
    command.add_argument("--target", choices=("transaction", "high-intent"), default="transaction")
    command.add_argument("--test-days", type=int, default=14)
    command.add_argument("--validation-days", type=int, default=14)
    command.add_argument("--k", type=int, default=10)
    command.add_argument("--max-history", type=int, default=30)
    command.add_argument("--min-cooccurrence", type=int, default=2)
    command.add_argument("--neighbors", type=int, default=50)
    command.add_argument("--feature-retention", type=int, default=5)
    command.add_argument("--speed", type=float, default=0)
    command.add_argument("--limit", type=int)
    return command


def main() -> None:
    args = parser().parse_args()
    common = {"db_path": args.db, "vector_size": args.vector_size}
    if args.mode == "curation":
        result = curation_flow(
            **common, raw_dir=args.raw_dir, api_page_size=args.api_page_size,
            neighbors=args.neighbors, min_cooccurrence=args.min_cooccurrence,
            max_history=args.max_history, feature_retention=args.feature_retention,
            speed=args.speed, limit=args.limit,
        )
    elif args.mode == "modeling":
        result = modeling_flow(
            **common, model_dir=args.model_dir, report_path=args.report,
            target=args.target, test_days=args.test_days,
            validation_days=args.validation_days, k=args.k,
            max_history=args.max_history,
            min_cooccurrence=args.min_cooccurrence, neighbors=args.neighbors,
        )
    else:
        result = recomart_flow(
            **common, raw_dir=args.raw_dir, model_dir=args.model_dir,
            report_path=args.report, api_page_size=args.api_page_size,
            target=args.target, test_days=args.test_days,
            validation_days=args.validation_days, k=args.k,
            max_history=args.max_history,
            min_cooccurrence=args.min_cooccurrence,
            neighbors=args.neighbors, feature_retention=args.feature_retention,
            speed=args.speed, limit=args.limit,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
