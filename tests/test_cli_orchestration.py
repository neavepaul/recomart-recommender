import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import recomart
from src.orchestration import prefect_flow as orchestration


@pytest.mark.parametrize(
    ("argv", "target", "result"),
    [
        (["replay-events"], "replay_events", 3),
        (["ingest-categories"], "ingest_categories", 4),
        (["ingest-products"], "ingest_products", 5),
        (["build-bronze"], "build_bronze", {"bronze_events": 3}),
        (["transform"], "transform", {"gold_item_features": 2}),
        (["validate"], "validate", {"ok": True}),
        (
            ["quality-report", "--sample-rows", "10"],
            "generate_quality_report",
            {
                "success": True, "critical_failures": 0,
                "checks": [], "great_expectations": [],
                "json_report": "quality.json", "pdf_report": "quality.pdf",
            },
        ),
        (["show-lineage"], "latest_lineage", [{"run_id": "r1"}]),
        (["register-features"], "register_features", {"feature_version": "v1"}),
        (["show-registry"], "list_registry", [{"feature_view": "users"}]),
        (
            ["get-features", "--view", "user_activity", "--id", "1"],
            "get_online_features",
            {"mode": "inference", "rows": []},
        ),
        (
            ["get-features", "--view", "user_activity", "--id", "1", "--for", "training"],
            "get_training_features",
            {"mode": "training", "rows": []},
        ),
        (["evaluate"], "evaluate_popularity", {"precision@10": 0.1}),
        (["profile-gold"], "profile_gold", {"size": {"users": 1}}),
        (["profile-plots"], "generate_eda_plots", {"plots": {}}),
        (["prepare-model-data"], "prepare_model_data", {"target": "transaction"}),
        (["train-models"], "train_models", {"popularity_items": 2}),
        (["evaluate-models"], "evaluate_models", {"models": {}}),
        (["build-content-model"], "build_content_model", {"items": 2}),
        (["tune-hybrid"], "tune_hybrid", {"best": {}}),
        (
            ["recommend", "--visitor-id", "1"],
            "recommend",
            {"visitor_id": 1, "recommendations": []},
        ),
        (["run"], "run_all", {"gold_item_features": 2}),
    ],
)
def test_cli_routes_public_commands(monkeypatch, capsys, argv, target, result):
    operation = MagicMock(return_value=result)
    monkeypatch.setattr(recomart, target, operation)
    monkeypatch.setattr(sys, "argv", ["recomart", *argv])

    recomart.main()

    payload = json.loads(capsys.readouterr().out)
    operation.assert_called_once()
    if argv[0] == "replay-events":
        assert payload == {"bronze_events": 3}
    elif argv[0] == "ingest-categories":
        assert payload == {"bronze_category_tree": 4}
    elif argv[0] == "ingest-products":
        assert payload == {"bronze_item_properties": 5}
    elif argv[0] == "quality-report":
        assert payload["success"] is True
        assert payload["critical_failures"] == 0
    else:
        assert payload == result


@pytest.mark.parametrize(
    ("command", "module_name", "function_name", "extra_args"),
    [
        ("build-silver", "src.pipelines.silver", "build_silver", []),
        ("build-gold", "src.pipelines.gold", "build_gold", ["--vector-size", "32"]),
        (
            "build-features",
            "src.pipelines.features",
            "build_features",
            ["--neighbors", "7", "--min-cooccurrence", "1", "--max-history", "9"],
        ),
    ],
)
def test_cli_build_commands_close_database(
    monkeypatch, capsys, command, module_name, function_name, extra_args
):
    module = __import__(module_name, fromlist=[function_name])
    builder = MagicMock()
    database = MagicMock()
    monkeypatch.setattr(module, function_name, builder)
    monkeypatch.setattr(recomart, "connect", MagicMock(return_value=database))
    monkeypatch.setattr(recomart, "counts", MagicMock(return_value={"rows": 2}))
    monkeypatch.setattr(sys, "argv", ["recomart", command, *extra_args])

    recomart.main()

    assert json.loads(capsys.readouterr().out) == {"rows": 2}
    builder.assert_called_once()
    database.close.assert_called_once()


def test_cli_server_always_closes(monkeypatch, capsys):
    server = MagicMock()
    server.serve_forever.side_effect = KeyboardInterrupt
    monkeypatch.setattr(recomart, "make_server", MagicMock(return_value=server))
    monkeypatch.setattr(sys, "argv", ["recomart", "serve-api", "--port", "9000"])

    recomart.main()

    assert "http://127.0.0.1:9000" in capsys.readouterr().out
    server.server_close.assert_called_once()


def test_cli_failed_validation_has_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(recomart, "validate", MagicMock(return_value={"ok": False}))
    monkeypatch.setattr(sys, "argv", ["recomart", "validate"])
    with pytest.raises(SystemExit, match="1"):
        recomart.main()
    assert json.loads(capsys.readouterr().out) == {"ok": False}


def test_cli_failed_quality_report_has_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(
        recomart, "generate_quality_report", MagicMock(return_value={
            "success": False, "critical_failures": 1,
            "checks": [], "great_expectations": [],
            "json_report": "quality.json", "pdf_report": "quality.pdf",
        })
    )
    monkeypatch.setattr(sys, "argv", ["recomart", "quality-report"])
    with pytest.raises(SystemExit, match="1"):
        recomart.main()
    assert json.loads(capsys.readouterr().out)["success"] is False


def test_cli_stages_landing_snapshot(monkeypatch, capsys):
    snapshot = MagicMock()
    monkeypatch.setattr(recomart, "land_sources", landing := MagicMock(return_value=snapshot))
    monkeypatch.setattr(
        recomart, "snapshot_as_dict", MagicMock(return_value={"ingestion_date": "2026-07-21"})
    )
    monkeypatch.setattr(
        sys, "argv", ["recomart", "stage-landing", "--ingestion-date", "2026-07-21"]
    )
    recomart.main()
    assert json.loads(capsys.readouterr().out)["ingestion_date"] == "2026-07-21"
    landing.assert_called_once()


def test_cli_compatibility_helpers_delegate(monkeypatch, tmp_path):
    monkeypatch.setattr(recomart.events, "replay_events", MagicMock(return_value=1))
    monkeypatch.setattr(recomart.categories, "ingest_categories", MagicMock(return_value=2))
    monkeypatch.setattr(recomart.products, "make_server", MagicMock(return_value="server"))
    monkeypatch.setattr(recomart.products, "ingest_products", MagicMock(return_value=3))
    assert recomart.replay_events(tmp_path / "x.db", 2.0, 5) == 1
    assert recomart.ingest_categories(tmp_path / "x.db") == 2
    assert recomart.make_server("localhost", 8) == "server"
    assert recomart.ingest_products(tmp_path / "x.db", "http://api", 10, 5) == 3


def test_curation_tasks_build_and_record_lineage(monkeypatch, tmp_path):
    db_path = tmp_path / "curated.db"
    raw_dir = tmp_path / "raw"
    database = MagicMock()
    monkeypatch.setattr(orchestration, "connect", MagicMock(return_value=database))
    monkeypatch.setattr(orchestration, "table_counts", MagicMock(return_value={"rows": 3}))
    monkeypatch.setattr(orchestration, "record_dataset", records := MagicMock())
    monkeypatch.setattr(orchestration, "build_bronze", MagicMock(return_value={"bronze": 1}))
    monkeypatch.setattr(orchestration, "build_silver", MagicMock())
    monkeypatch.setattr(orchestration, "build_gold", MagicMock())
    monkeypatch.setattr(orchestration, "build_features", MagicMock())

    assert orchestration.bronze_task.fn(db_path, raw_dir, "r1", 0, 10, 100) == {"bronze": 1}
    assert records.call_count == 4
    records.reset_mock()
    assert orchestration.silver_task.fn(db_path, "r1") == {"rows": 3}
    assert records.call_count == 3
    records.reset_mock()
    assert orchestration.gold_task.fn(db_path, "r1", 32) == {"rows": 3}
    assert records.call_count == 2
    records.reset_mock()
    assert orchestration.features_task.fn(db_path, "r1", 7, 1, 9) == {"rows": 3}
    assert records.call_count == 3
    assert database.close.call_count == 3


def test_registry_split_and_validation_tasks(monkeypatch, tmp_path):
    db_path = tmp_path / "features.db"
    manifest = {"feature_views": [{"source_table": "feature_users"}]}
    monkeypatch.setattr(orchestration, "register_features", MagicMock(return_value=manifest))
    monkeypatch.setattr(orchestration, "record_dataset", records := MagicMock())
    assert orchestration.registry_task.fn(db_path, "r1", 2) == manifest
    records.assert_called_once()

    monkeypatch.setattr(orchestration, "prepare_model_data", MagicMock(return_value={"split": 1}))
    records.reset_mock()
    assert orchestration.split_task.fn(db_path, "r1", "transaction", 14) == {"split": 1}
    assert records.call_count == 2

    monkeypatch.setattr(orchestration, "validate", MagicMock(return_value={"ok": True}))
    assert orchestration.validation_task.fn(db_path) == {"ok": True}
    monkeypatch.setattr(orchestration, "validate", MagicMock(return_value={"ok": False}))
    with pytest.raises(RuntimeError, match="validation failed"):
        orchestration.validation_task.fn(db_path)


@contextmanager
def _tracking_context(client):
    yield client


def test_model_tasks_track_results_and_write_report(monkeypatch, tmp_path):
    db_path = tmp_path / "models.db"
    model_dir = tmp_path / "content"
    report = tmp_path / "reports" / "metrics.json"
    client = MagicMock()
    monkeypatch.setattr(orchestration, "model_run", lambda *args, **kwargs: _tracking_context(client))
    monkeypatch.setattr(orchestration, "log_result", logged := MagicMock())
    monkeypatch.setattr(orchestration, "log_existing_artifact", artifact := MagicMock())
    monkeypatch.setattr(orchestration, "record_dataset", records := MagicMock())
    monkeypatch.setattr(orchestration, "train_models", MagicMock(return_value={"trained": 1}))
    assert orchestration.train_task.fn(db_path, "r1", 30, 2, 50) == {"trained": 1}
    records.assert_called_once()

    monkeypatch.setattr(orchestration, "build_content_model", MagicMock(return_value={"items": 3}))
    assert orchestration.content_task.fn(db_path, model_dir, "r1", 256) == {"items": 3}
    artifact.assert_called()

    tuning = {"best": {"item_cf_weight": 0.9, "ndcg@10": 0.2}}
    monkeypatch.setattr(orchestration, "tune_hybrid", MagicMock(return_value=tuning))
    assert orchestration.tune_task.fn(db_path, model_dir, "r1", 14, 10, 30, 2, 50) == tuning
    client.log_params.assert_called_with({"selected_item_cf_weight": 0.9})

    segment = {"precision@10": 0.1, "recall@10": 0.2, "ndcg@10": 0.3, "hit_rate@10": 0.4}
    result = {
        "k": 10, "eligible_users": 5, "warm_users": 2, "cold_start_users": 3,
        "models": {
            "item_cf_content_hybrid": {**segment, "segments": {"warm_users": segment}},
            "item_collaborative_filtering": {"segments": {"warm_users": segment}},
        },
    }
    monkeypatch.setattr(orchestration, "evaluate_models", MagicMock(return_value=result))
    assert orchestration.evaluation_task.fn(db_path, model_dir, report, "r1", 10) == result
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["hybrid"]["ndcg"] == 0.3
    assert saved["item_cf_warm"]["recall"] == 0.2
    assert logged.call_count == 4


def test_curation_flow_records_success_and_failure(monkeypatch, tmp_path):
    db_path = tmp_path / "flow.db"
    monkeypatch.setattr(orchestration, "uuid4", lambda: "run-1")
    monkeypatch.setattr(orchestration, "start_pipeline_run", started := MagicMock())
    monkeypatch.setattr(orchestration, "finish_pipeline_run", finished := MagicMock())
    monkeypatch.setattr(orchestration, "bronze_task", MagicMock(return_value={"b": 1}))
    monkeypatch.setattr(orchestration, "silver_task", MagicMock(return_value={"s": 1}))
    monkeypatch.setattr(orchestration, "gold_task", MagicMock(return_value={"g": 1}))
    monkeypatch.setattr(orchestration, "features_task", MagicMock(return_value={"f": 1}))
    monkeypatch.setattr(orchestration, "registry_task", MagicMock(return_value={"r": 1}))
    monkeypatch.setattr(orchestration, "validation_task", MagicMock(return_value={"ok": True}))
    result = orchestration.curation_flow.fn(db_path=db_path)
    assert result["run_id"] == "run-1"
    started.assert_called_once()
    finished.assert_called_with(db_path, "run-1", "COMPLETED")

    orchestration.validation_task.side_effect = ValueError("bad data")
    with pytest.raises(ValueError, match="bad data"):
        orchestration.curation_flow.fn(db_path=db_path)
    finished.assert_called_with(db_path, "run-1", "FAILED", "bad data")


def test_modeling_and_full_flows_filter_parameters(monkeypatch, tmp_path):
    db_path = tmp_path / "flow.db"
    monkeypatch.setattr(orchestration, "uuid4", lambda: "run-2")
    monkeypatch.setattr(orchestration, "start_pipeline_run", MagicMock())
    monkeypatch.setattr(orchestration, "finish_pipeline_run", finished := MagicMock())
    for name in ("split_task", "train_task", "content_task", "tune_task", "evaluation_task"):
        monkeypatch.setattr(orchestration, name, MagicMock(return_value={name: 1}))
    result = orchestration.modeling_flow.fn(db_path=db_path)
    assert result["run_id"] == "run-2"
    finished.assert_called_with(db_path, "run-2", "COMPLETED")

    orchestration.split_task.side_effect = RuntimeError("split failed")
    with pytest.raises(RuntimeError, match="split failed"):
        orchestration.modeling_flow.fn(db_path=db_path)
    finished.assert_called_with(db_path, "run-2", "FAILED", "split failed")

    curated = MagicMock(return_value={"curated": True})
    modeled = MagicMock(return_value={"modeled": True})
    monkeypatch.setattr(orchestration, "curation_flow", curated)
    monkeypatch.setattr(orchestration, "modeling_flow", modeled)
    full = orchestration.recomart_flow.fn(
        db_path=db_path, raw_dir=tmp_path / "raw", api_page_size=12,
        feature_retention=3, target="transaction", k=5,
    )
    assert full == {"curation": {"curated": True}, "modeling": {"modeled": True}}
    assert "raw_dir" not in modeled.call_args.kwargs
    assert "target" not in curated.call_args.kwargs


@pytest.mark.parametrize(("mode", "target"), [("curation", "curation_flow"), ("modeling", "modeling_flow"), ("full", "recomart_flow")])
def test_orchestration_cli_modes(monkeypatch, capsys, mode, target):
    operation = MagicMock(return_value={"mode": mode, "path": Path("x")})
    monkeypatch.setattr(orchestration, target, operation)
    monkeypatch.setattr(sys, "argv", ["prefect-flow", "--mode", mode])
    orchestration.main()
    assert json.loads(capsys.readouterr().out)["mode"] == mode
    operation.assert_called_once()
