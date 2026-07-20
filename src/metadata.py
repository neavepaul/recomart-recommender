"""Persistent pipeline-run and dataset-lineage metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import connect


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_version(path: Path) -> str | None:
    """Return a content hash suitable for source-version metadata."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def initialize_metadata(db_path: Path) -> None:
    db = connect(db_path)
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata_pipeline_runs (
                run_id TEXT PRIMARY KEY,
                flow_name TEXT NOT NULL,
                orchestrator TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                parameters_json TEXT NOT NULL,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS metadata_dataset_lineage (
                lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                layer TEXT NOT NULL,
                storage_uri TEXT NOT NULL,
                source_name TEXT,
                source_uri TEXT,
                source_version TEXT,
                source_modified_at TEXT,
                ingestion_timestamp TEXT NOT NULL,
                row_count INTEGER,
                transformation TEXT NOT NULL,
                upstream_datasets_json TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES metadata_pipeline_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS ix_metadata_lineage_dataset
              ON metadata_dataset_lineage(dataset_name,ingestion_timestamp);
            CREATE INDEX IF NOT EXISTS ix_metadata_lineage_run
              ON metadata_dataset_lineage(run_id);
        """)
        db.commit()
    finally:
        db.close()


def start_pipeline_run(
    db_path: Path,
    run_id: str,
    flow_name: str,
    parameters: dict[str, Any],
    orchestrator: str = "prefect",
) -> None:
    initialize_metadata(db_path)
    db = connect(db_path)
    try:
        db.execute(
            """INSERT OR REPLACE INTO metadata_pipeline_runs
               (run_id,flow_name,orchestrator,status,started_at,completed_at,
                parameters_json,error_message)
               VALUES (?,?,?,?,?,NULL,?,NULL)""",
            (
                run_id, flow_name, orchestrator, "RUNNING", utc_now(),
                json.dumps(parameters, sort_keys=True, default=str),
            ),
        )
        db.commit()
    finally:
        db.close()


def finish_pipeline_run(
    db_path: Path, run_id: str, status: str, error_message: str | None = None
) -> None:
    db = connect(db_path)
    try:
        db.execute(
            """UPDATE metadata_pipeline_runs
               SET status=?,completed_at=?,error_message=? WHERE run_id=?""",
            (status, utc_now(), error_message, run_id),
        )
        db.commit()
    finally:
        db.close()


def record_dataset(
    db_path: Path,
    run_id: str,
    dataset_name: str,
    layer: str,
    transformation: str,
    upstream_datasets: list[str],
    source_name: str | None = None,
    source_path: Path | None = None,
    schema_version: str = "1.0",
) -> None:
    """Record one materialized table and its immediate lineage."""
    db = connect(db_path)
    try:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (dataset_name,),
        ).fetchone()
        row_count = (
            db.execute(f"SELECT COUNT(*) FROM {dataset_name}").fetchone()[0]
            if exists else None
        )
        resolved_source = source_path.resolve() if source_path else None
        modified_at = None
        if resolved_source and resolved_source.exists():
            modified_at = datetime.fromtimestamp(
                resolved_source.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        db.execute(
            """INSERT INTO metadata_dataset_lineage
               (run_id,dataset_name,layer,storage_uri,source_name,source_uri,
                source_version,source_modified_at,ingestion_timestamp,row_count,
                transformation,upstream_datasets_json,schema_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, dataset_name, layer,
                f"sqlite:///{db_path.resolve().as_posix()}#{dataset_name}",
                source_name,
                resolved_source.as_uri() if resolved_source else None,
                file_version(resolved_source) if resolved_source else None,
                modified_at, utc_now(), row_count, transformation,
                json.dumps(upstream_datasets), schema_version,
            ),
        )
        db.commit()
    finally:
        db.close()


def latest_lineage(db_path: Path) -> list[dict[str, Any]]:
    """Return the latest metadata record for every materialized dataset."""
    db = connect(db_path)
    db.row_factory = __import__("sqlite3").Row
    try:
        rows = db.execute("""
            SELECT * FROM metadata_dataset_lineage l
            WHERE lineage_id=(
              SELECT MAX(lineage_id) FROM metadata_dataset_lineage
              WHERE dataset_name=l.dataset_name
            ) ORDER BY layer,dataset_name
        """).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()
