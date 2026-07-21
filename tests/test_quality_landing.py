import csv
import importlib.util
import json
from pathlib import Path

import pytest

from src.ingestion.landing import land_sources
from src.pipelines.bronze import build_bronze
from src.pipelines.runner import transform
from src.quality import generate_quality_report


def _write(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _raw_sources(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir()
    _write(
        raw / "events.csv",
        ["timestamp", "visitorid", "event", "itemid", "transactionid"],
        [
            [1000, 1, "view", 10, ""],
            [2000, 1, "addtocart", 10, ""],
            [3000, 1, "transaction", 10, 99],
            [4000, 2, "invalid", 10, ""],
        ],
    )
    _write(
        raw / "item_properties_part1.csv",
        ["timestamp", "itemid", "property", "value"],
        [[1000, 10, "categoryid", "7"], [1000, 10, "available", "1"]],
    )
    _write(
        raw / "item_properties_part2.csv",
        ["timestamp", "itemid", "property", "value"],
        [[1000, 10, "400", "n1"]],
    )
    _write(
        raw / "category_tree.csv",
        ["categoryid", "parentid"],
        [[7, ""]],
    )
    return raw


def test_landing_snapshots_are_partitioned_immutable_and_idempotent(tmp_path):
    raw = _raw_sources(tmp_path)
    landing = tmp_path / "landing"
    first = land_sources(raw, landing, "2026-07-21")
    second = land_sources(raw, landing, "2026-07-21")

    assert first.events_dir == second.events_dir
    assert "source=retailrocket" in str(first.events_dir)
    assert "type=clickstream" in str(first.events_dir)
    assert "ingestion_date=2026-07-21" in str(first.events_dir)
    assert "source_version=" in str(first.events_dir)
    assert (first.events_dir / "events.csv").read_bytes() == (raw / "events.csv").read_bytes()
    assert first.manifest_path != second.manifest_path
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 4
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])


def test_landing_rejects_bad_date_and_missing_sources(tmp_path):
    raw = _raw_sources(tmp_path)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        land_sources(raw, tmp_path / "landing", "21-07-2026")
    (raw / "events.csv").unlink()
    with pytest.raises(FileNotFoundError, match="events.csv"):
        land_sources(raw, tmp_path / "landing", "2026-07-21")


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("pandas", "great_expectations", "reportlab")),
    reason="quality-report dependencies are not installed",
)
def test_quality_report_profiles_validates_and_writes_pdf(tmp_path):
    raw = _raw_sources(tmp_path)
    db_path = tmp_path / "quality.db"
    build_bronze(
        db_path, raw, limit=None, api_page_size=2,
        landing_dir=tmp_path / "landing", ingestion_date="2026-07-21",
    )
    transform(db_path, vector_size=16, neighbors=5, min_cooccurrence=1, max_history=10)
    json_path = tmp_path / "quality.json"
    pdf_path = tmp_path / "quality.pdf"

    report = generate_quality_report(
        db_path, json_path=json_path, pdf_path=pdf_path, sample_rows=10
    )

    assert report["success"] is True
    assert json_path.exists() and json_path.stat().st_size > 500
    assert pdf_path.exists() and pdf_path.read_bytes().startswith(b"%PDF")
    assert {profile["table"] for profile in report["profiles"]} == {
        "silver_user_events", "silver_products",
        "gold_user_item_features", "gold_item_features",
    }
    rejection = next(
        check for check in report["checks"]
        if check["name"] == "bronze_to_silver_rejections"
    )
    assert rejection["violations"] == 1
    assert all(result["success"] for result in report["great_expectations"])
