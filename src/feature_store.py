"""Custom feature store for versioned feature registration and retrieval.

RecoMart keeps its engineered features (Task 6) in the same SQLite database as
the medallion layers. This module adds a lightweight feature store on top of
those tables so downstream training and inference share one governed source:

* A ``feature_registry`` table documents every feature view — its entity, key,
  source table, feature columns, transformation, and current version.
* Append-only snapshot tables (``<source>_versions``) preserve the feature
  values produced by each pipeline run, tagged with a ``feature_version``.
* Retrieval helpers return the latest snapshot for inference or a specific
  historical version for reproducible training.

The design deliberately avoids external dependencies (no Feast): it reuses the
existing SQLite store and lineage ``run_id`` convention, keeping the project
dependency-minimal while still providing versioned, documented feature access.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core import connect
from src.metadata import utc_now

logger = logging.getLogger(__name__)

DEFAULT_RETENTION = 5
DEFAULT_MANIFEST = Path("reports/feature_registry.json")


@dataclass(frozen=True)
class FeatureView:
    """Declarative definition of one retrievable feature group."""

    name: str
    entity: str
    entity_key: str
    source_table: str
    transformation: str
    multi_row: bool = field(default=False)

    @property
    def version_table(self) -> str:
        return f"{self.source_table}_versions"


FEATURE_VIEWS: tuple[FeatureView, ...] = (
    FeatureView(
        name="user_activity",
        entity="user",
        entity_key="visitor_id",
        source_table="feature_user_activity",
        transformation=(
            "Per-user activity frequency, event-type counts, average interaction "
            "score, purchase rate, and last-active timestamp."
        ),
    ),
    FeatureView(
        name="item_popularity",
        entity="item",
        entity_key="item_id",
        source_table="feature_item_popularity",
        transformation=(
            "Per-item popularity, distinct users, average interaction score, "
            "conversion rate, and global popularity rank (cold items retained)."
        ),
    ),
    FeatureView(
        name="item_cooccurrence",
        entity="item",
        entity_key="source_item_id",
        source_table="feature_item_cooccurrence",
        transformation=(
            "Top weighted item cosine-similarity neighbours per source item from "
            "the full Gold interaction history."
        ),
        multi_row=True,
    ),
)

_VIEWS_BY_NAME = {view.name: view for view in FEATURE_VIEWS}


def view_names() -> list[str]:
    return [view.name for view in FEATURE_VIEWS]


def get_view(name: str) -> FeatureView:
    try:
        return _VIEWS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature view '{name}'. Available: {', '.join(view_names())}"
        ) from None


def initialize_registry(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS feature_registry (
            feature_view TEXT NOT NULL,
            entity TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            feature_columns_json TEXT NOT NULL,
            source_table TEXT NOT NULL,
            transformation TEXT NOT NULL,
            feature_version TEXT NOT NULL,
            run_id TEXT,
            row_count INTEGER,
            created_at TEXT NOT NULL,
            PRIMARY KEY (feature_view, feature_version)
        );
        CREATE INDEX IF NOT EXISTS ix_feature_registry_view
          ON feature_registry(feature_view, created_at);
    """)
    db.commit()


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]


def _feature_columns(db: sqlite3.Connection, view: FeatureView) -> list[str]:
    return [c for c in _columns(db, view.source_table) if c != view.entity_key]


def _ensure_version_table(db: sqlite3.Connection, view: FeatureView) -> None:
    version_table = view.version_table
    if not _table_exists(db, version_table):
        db.execute(
            f"""CREATE TABLE {version_table} AS
                SELECT *, CAST(NULL AS TEXT) AS feature_version,
                       CAST(NULL AS TEXT) AS created_at
                FROM {view.source_table} WHERE 0"""
        )
        db.execute(
            f"CREATE INDEX ix_{version_table}_lookup "
            f"ON {version_table}(feature_version, {view.entity_key})"
        )


def _prune_versions(
    db: sqlite3.Connection, view: FeatureView, retention: int
) -> None:
    # ``created_at`` is intentionally human-readable metadata, not an ordering
    # key: several registrations can receive the same clock value. SQLite's
    # registry rowid gives every successful insert an unambiguous sequence.
    keep_versions = [
        row[0]
        for row in db.execute(
            """SELECT feature_version FROM feature_registry
               WHERE feature_view=? ORDER BY rowid DESC LIMIT ?""",
            (view.name, retention),
        )
    ]
    placeholders = ",".join("?" for _ in keep_versions)
    db.execute(
        f"DELETE FROM {view.version_table} "
        f"WHERE feature_version NOT IN ({placeholders})",
        keep_versions,
    )
    db.execute(
        f"DELETE FROM feature_registry WHERE feature_view=? "
        f"AND feature_version NOT IN ({placeholders})",
        [view.name, *keep_versions],
    )


def register_features(
    db_path: Path,
    run_id: str | None = None,
    retention: int = DEFAULT_RETENTION,
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Snapshot the current feature tables as a new version and update the registry.

    Each registered version is identified by ``feature_version`` (the pipeline
    ``run_id`` when available, otherwise a generated UUID). The current rows of
    every feature source table are copied into its append-only ``*_versions``
    snapshot, tagged with the version and a creation timestamp. Old versions
    beyond ``retention`` are pruned. A JSON manifest documenting the registered
    views is written to ``manifest_path``.
    """
    if retention < 1:
        raise ValueError("retention must be at least 1")
    version = run_id or str(uuid4())
    created_at = utc_now()
    db = connect(db_path)
    try:
        initialize_registry(db)
        duplicate = db.execute(
            "SELECT 1 FROM feature_registry WHERE feature_version=? LIMIT 1",
            (version,),
        ).fetchone()
        if duplicate is not None:
            raise RuntimeError(
                f"Feature version '{version}' is already registered; "
                "registered feature snapshots are immutable."
            )
        registered: list[dict[str, Any]] = []
        for view in FEATURE_VIEWS:
            if not _table_exists(db, view.source_table):
                raise RuntimeError(
                    f"Missing feature table '{view.source_table}'; "
                    "run build-features first."
                )
            feature_columns = _feature_columns(db, view)
            _ensure_version_table(db, view)
            db.execute(
                f"""INSERT INTO {view.version_table}
                    SELECT *, ?, ? FROM {view.source_table}""",
                (version, created_at),
            )
            row_count = db.execute(
                f"SELECT COUNT(*) FROM {view.source_table}"
            ).fetchone()[0]
            db.execute(
                """INSERT INTO feature_registry
                   (feature_view, entity, entity_key, feature_columns_json,
                    source_table, transformation, feature_version, run_id,
                    row_count, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    view.name, view.entity, view.entity_key,
                    json.dumps(feature_columns), view.source_table,
                    view.transformation, version, run_id, row_count, created_at,
                ),
            )
            _prune_versions(db, view, retention)
            registered.append({
                "feature_view": view.name,
                "entity": view.entity,
                "entity_key": view.entity_key,
                "source_table": view.source_table,
                "feature_columns": feature_columns,
                "transformation": view.transformation,
                "row_count": row_count,
            })
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    manifest = {
        "feature_version": version,
        "run_id": run_id,
        "created_at": created_at,
        "retention": retention,
        "feature_views": registered,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Feature registry manifest written to %s", manifest_path)
    return manifest


def list_registry(db_path: Path) -> list[dict[str, Any]]:
    """Return the latest registered version for every feature view."""
    db = connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        if not _table_exists(db, "feature_registry"):
            return []
        rows = db.execute("""
            SELECT r.* FROM feature_registry r
            WHERE r.rowid=(
                SELECT MAX(newer.rowid) FROM feature_registry newer
                WHERE newer.feature_view=r.feature_view
            ) ORDER BY r.feature_view
        """).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            record["feature_columns"] = json.loads(
                record.pop("feature_columns_json")
            )
            result.append(record)
        return result
    finally:
        db.close()


def resolve_version(db_path: Path, view_name: str, version: str = "latest") -> str:
    """Resolve ``"latest"`` to the newest registered version for a view."""
    view = get_view(view_name)
    db = connect(db_path)
    try:
        if not _table_exists(db, "feature_registry"):
            raise RuntimeError("Feature registry is empty; run register-features.")
        if version == "latest":
            row = db.execute(
                """SELECT feature_version FROM feature_registry
                   WHERE feature_view=? ORDER BY rowid DESC LIMIT 1""",
                (view.name,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"No registered version for feature view '{view.name}'."
                )
            return row[0]
        exists = db.execute(
            """SELECT 1 FROM feature_registry
               WHERE feature_view=? AND feature_version=?""",
            (view.name, version),
        ).fetchone()
        if exists is None:
            raise RuntimeError(
                f"Version '{version}' not registered for feature view "
                f"'{view.name}'."
            )
        return version
    finally:
        db.close()


def _fetch(
    db_path: Path,
    view: FeatureView,
    version: str,
    entity_ids: list[Any] | None,
) -> list[dict[str, Any]]:
    db = connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        params: list[Any] = [version]
        query = (
            f"SELECT * FROM {view.version_table} WHERE feature_version=?"
        )
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            query += f" AND {view.entity_key} IN ({placeholders})"
            params.extend(entity_ids)
        if view.multi_row:
            query += f" ORDER BY {view.entity_key}"
        rows = db.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()


def get_online_features(
    db_path: Path, view_name: str, entity_ids: list[Any] | None = None
) -> dict[str, Any]:
    """Retrieve the latest feature version for inference."""
    view = get_view(view_name)
    version = resolve_version(db_path, view_name, "latest")
    rows = _fetch(db_path, view, version, entity_ids)
    return {
        "feature_view": view.name,
        "entity_key": view.entity_key,
        "feature_version": version,
        "mode": "inference",
        "rows": rows,
    }


def get_training_features(
    db_path: Path,
    view_name: str,
    entity_ids: list[Any] | None = None,
    version: str = "latest",
) -> dict[str, Any]:
    """Retrieve a specific immutable historical snapshot for training."""
    view = get_view(view_name)
    resolved = resolve_version(db_path, view_name, version)
    rows = _fetch(db_path, view, resolved, entity_ids)
    return {
        "feature_view": view.name,
        "entity_key": view.entity_key,
        "feature_version": resolved,
        "mode": "training",
        "rows": rows,
    }
