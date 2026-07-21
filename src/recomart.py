"""RecoMart command-line interface and backwards-compatible public API."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src import core
from src.evaluation import evaluate_popularity
from src.ingestion import categories, events, products
from src.ingestion.landing import land_sources, snapshot_as_dict
from src.feature_store import (
    get_online_features, get_training_features, list_registry, register_features,
    view_names,
)
from src.metadata import latest_lineage
from src.modeling import (
    build_content_model, evaluate_models, generate_eda_plots,
    prepare_model_data, profile_gold, recommend, train_models, tune_hybrid,
)
from src.quality import generate_quality_report
from src.pipelines.bronze import build_bronze
from src.pipelines.runner import transform
from src.validation import validate

RAW = core.RAW
LANDING = core.LANDING
DEFAULT_DB = core.DEFAULT_DB
connect = core.connect
counts = core.table_counts


def replay_events(db_path: Path, speed: float = 0, limit: int | None = None) -> int:
    return events.replay_events(db_path, RAW, speed, limit)


def ingest_categories(db_path: Path) -> int:
    return categories.ingest_categories(db_path, RAW)


def make_server(host: str, port: int):
    return products.make_server(host, port, RAW)


def ingest_products(db_path: Path, api_url: str, page_size: int = 50_000, limit: int | None = None) -> int:
    return products.ingest_products(db_path, api_url, page_size, limit)


def run_all(args: argparse.Namespace) -> dict[str, int]:
    build_bronze(args.db, RAW, args.speed, args.limit, args.api_page_size)
    return transform(args.db, args.vector_size)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--db", type=Path, default=DEFAULT_DB)
    command.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    sub = command.add_subparsers(dest="command", required=True)
    landing = sub.add_parser("stage-landing")
    landing.add_argument("--landing-dir", type=Path, default=LANDING)
    landing.add_argument("--ingestion-date")
    replay = sub.add_parser("replay-events")
    replay.add_argument("--speed", type=float, default=0)
    replay.add_argument("--limit", type=int)
    sub.add_parser("ingest-categories")
    api = sub.add_parser("serve-api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    item_properties = sub.add_parser("ingest-products")
    item_properties.add_argument("--api-url", default="http://127.0.0.1:8000")
    item_properties.add_argument("--api-page-size", type=int, default=50_000)
    item_properties.add_argument("--limit", type=int)
    bronze = sub.add_parser("build-bronze")
    bronze.add_argument("--speed", type=float, default=0)
    bronze.add_argument("--limit", type=int)
    bronze.add_argument("--api-page-size", type=int, default=50_000)
    silver = sub.add_parser("build-silver")
    gold = sub.add_parser("build-gold")
    gold.add_argument("--vector-size", type=int, default=256)
    feature = sub.add_parser("build-features")
    feature.add_argument("--neighbors", type=int, default=50)
    feature.add_argument("--min-cooccurrence", type=int, default=2)
    feature.add_argument("--max-history", type=int, default=30)
    transformed = sub.add_parser("transform")
    transformed.add_argument("--vector-size", type=int, default=256)
    transformed.add_argument("--neighbors", type=int, default=50)
    transformed.add_argument("--min-cooccurrence", type=int, default=2)
    transformed.add_argument("--max-history", type=int, default=30)
    sub.add_parser("validate")
    quality = sub.add_parser("quality-report")
    quality.add_argument(
        "--json-report", type=Path,
        default=Path("reports/data_quality_report.json"),
    )
    quality.add_argument(
        "--pdf-report", type=Path,
        default=Path("output/pdf/recomart_data_quality_report.pdf"),
    )
    quality.add_argument("--sample-rows", type=int, default=100_000)
    sub.add_parser("show-lineage")
    register = sub.add_parser("register-features")
    register.add_argument("--version")
    register.add_argument("--retention", type=int, default=5)
    sub.add_parser("show-registry")
    get_features = sub.add_parser("get-features")
    get_features.add_argument("--view", required=True, choices=view_names())
    get_features.add_argument("--id", type=int, nargs="+", dest="ids")
    get_features.add_argument("--version", default="latest")
    get_features.add_argument(
        "--for", choices=("inference", "training"), default="inference",
        dest="retrieval",
    )
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--k", type=int, default=10)
    evaluation.add_argument(
        "--target", choices=("transaction", "high-intent"), default="transaction"
    )
    evaluation.add_argument("--test-days", type=int, default=14)
    evaluation.add_argument("--cutoff-ms", type=int)
    profile = sub.add_parser("profile-gold")
    profile.add_argument("--top", type=int, default=10)
    profile_plots = sub.add_parser("profile-plots")
    profile_plots.add_argument("--top", type=int, default=15)
    profile_plots.add_argument("--out-dir", type=Path, default=Path("reports/eda"))
    split = sub.add_parser("prepare-model-data")
    split.add_argument(
        "--target", choices=("transaction", "high-intent"), default="transaction"
    )
    split.add_argument("--test-days", type=int, default=14)
    split.add_argument("--cutoff-ms", type=int)
    training = sub.add_parser("train-models")
    training.add_argument("--max-history", type=int, default=30)
    training.add_argument("--min-cooccurrence", type=int, default=2)
    training.add_argument("--neighbors", type=int, default=50)
    model_evaluation = sub.add_parser("evaluate-models")
    model_evaluation.add_argument("--k", type=int, default=10)
    model_evaluation.add_argument("--content-model-dir", type=Path, default=Path("models/content"))
    content = sub.add_parser("build-content-model")
    content.add_argument("--model-dir", type=Path, default=Path("models/content"))
    content.add_argument("--vector-size", type=int, default=256)
    tuning = sub.add_parser("tune-hybrid")
    tuning.add_argument("--content-model-dir", type=Path, default=Path("models/content"))
    tuning.add_argument("--validation-days", type=int, default=14)
    tuning.add_argument("--validation-cutoff-ms", type=int)
    tuning.add_argument("--k", type=int, default=10)
    tuning.add_argument("--max-history", type=int, default=30)
    tuning.add_argument("--min-cooccurrence", type=int, default=2)
    tuning.add_argument("--neighbors", type=int, default=50)
    inference = sub.add_parser("recommend")
    inference.add_argument("--visitor-id", type=int, required=True)
    inference.add_argument("--k", type=int, default=10)
    inference.add_argument(
        "--content-model-dir", type=Path, default=Path("models/content")
    )
    inference.add_argument("--max-history", type=int, default=30)
    run = sub.add_parser("run")
    run.add_argument("--speed", type=float, default=0)
    run.add_argument("--limit", type=int)
    run.add_argument("--api-page-size", type=int, default=50_000)
    run.add_argument("--vector-size", type=int, default=256)
    return command


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(__name__).info(
        "Command '%s' using database %s", args.command, args.db
    )
    if args.command == "serve-api":
        server = make_server(args.host, args.port)
        print(f"Product API listening on http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    if args.command == "stage-landing":
        result = snapshot_as_dict(
            land_sources(RAW, args.landing_dir, args.ingestion_date)
        )
    elif args.command == "replay-events":
        result = {"bronze_events": replay_events(args.db, args.speed, args.limit)}
    elif args.command == "ingest-categories":
        result = {"bronze_category_tree": ingest_categories(args.db)}
    elif args.command == "ingest-products":
        result = {"bronze_item_properties": ingest_products(args.db, args.api_url, args.api_page_size, args.limit)}
    elif args.command == "build-bronze":
        result = build_bronze(args.db, RAW, args.speed, args.limit, args.api_page_size)
    elif args.command == "build-silver":
        db = connect(args.db)
        try:
            from src.pipelines.silver import build_silver
            build_silver(db)
            result = counts(db)
        finally:
            db.close()
    elif args.command == "build-gold":
        db = connect(args.db)
        try:
            from src.pipelines.gold import build_gold
            build_gold(db, args.vector_size)
            result = counts(db)
        finally:
            db.close()
    elif args.command == "build-features":
        db = connect(args.db)
        try:
            from src.pipelines.features import build_features
            build_features(db, args.neighbors, args.min_cooccurrence, args.max_history)
            result = counts(db)
        finally:
            db.close()
    elif args.command == "transform":
        result = transform(
            args.db, args.vector_size, args.neighbors,
            args.min_cooccurrence, args.max_history,
        )
    elif args.command == "validate":
        result = validate(args.db)
    elif args.command == "quality-report":
        report = generate_quality_report(
            args.db, args.json_report, args.pdf_report, args.sample_rows
        )
        result = {
            "success": report["success"],
            "critical_failures": report["critical_failures"],
            "warning_failures": sum(
                not check["success"] and check["severity"] == "warning"
                for check in report["checks"]
            ),
            "great_expectations_passed": sum(
                result["success"] for result in report["great_expectations"]
            ),
            "great_expectations_total": len(report["great_expectations"]),
            "json_report": report["json_report"],
            "pdf_report": report["pdf_report"],
        }
    elif args.command == "show-lineage":
        result = latest_lineage(args.db)
    elif args.command == "register-features":
        result = register_features(args.db, args.version, args.retention)
    elif args.command == "show-registry":
        result = list_registry(args.db)
    elif args.command == "get-features":
        if args.retrieval == "inference":
            result = get_online_features(args.db, args.view, args.ids)
        else:
            result = get_training_features(
                args.db, args.view, args.ids, args.version
            )
    elif args.command == "evaluate":
        result = evaluate_popularity(
            args.db, args.k, args.target, args.test_days, args.cutoff_ms
        )
    elif args.command == "profile-gold":
        result = profile_gold(args.db, args.top)
    elif args.command == "profile-plots":
        result = generate_eda_plots(args.db, args.out_dir, args.top)
    elif args.command == "prepare-model-data":
        result = prepare_model_data(
            args.db, args.target, args.test_days, args.cutoff_ms
        )
    elif args.command == "train-models":
        result = train_models(
            args.db, args.max_history, args.min_cooccurrence, args.neighbors
        )
    elif args.command == "evaluate-models":
        result = evaluate_models(
            args.db, args.k, args.content_model_dir
        )
    elif args.command == "build-content-model":
        result = build_content_model(args.db, args.model_dir, args.vector_size)
    elif args.command == "tune-hybrid":
        result = tune_hybrid(
            args.db, args.content_model_dir, args.validation_days,
            args.validation_cutoff_ms, args.k, args.max_history,
            args.min_cooccurrence, args.neighbors,
        )
    elif args.command == "recommend":
        result = recommend(
            args.db, args.visitor_id, args.k,
            args.content_model_dir, args.max_history,
        )
    else:
        result = run_all(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        (args.command == "validate" and not result["ok"])
        or (args.command == "quality-report" and not result["success"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
