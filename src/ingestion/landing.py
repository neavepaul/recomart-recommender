"""Immutable, partitioned landing snapshots between source files and Bronze."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LandingSnapshot:
    events_dir: Path
    products_dir: Path
    categories_dir: Path
    manifest_path: Path
    ingestion_date: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _land_group(
    source_dir: Path,
    landing_root: Path,
    data_type: str,
    filenames: tuple[str, ...],
    ingestion_date: str,
) -> tuple[Path, list[dict[str, object]]]:
    sources = [source_dir / filename for filename in filenames]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))
    fingerprints = {path.name: _sha256(path) for path in sources}
    combined = hashlib.sha256(
        "".join(f"{name}:{fingerprints[name]}" for name in sorted(fingerprints)).encode()
    ).hexdigest()[:16]
    destination = (
        landing_root
        / "source=retailrocket"
        / f"type={data_type}"
        / f"ingestion_date={ingestion_date}"
        / f"source_version={combined}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in sources:
        target = destination / source.name
        if target.exists():
            if _sha256(target) != fingerprints[source.name]:
                raise RuntimeError(f"Landing snapshot collision at {target}")
        else:
            logger.info("Landing %s -> %s", source, target)
            shutil.copy2(source, target)
        records.append({
            "filename": source.name,
            "source_path": str(source.resolve()),
            "landing_path": str(target.resolve()),
            "sha256": fingerprints[source.name],
            "bytes": source.stat().st_size,
            "data_type": data_type,
        })
    return destination, records


def land_sources(
    source_dir: Path,
    landing_root: Path,
    ingestion_date: str | None = None,
) -> LandingSnapshot:
    """Copy source CSVs into immutable source/type/date/version partitions."""
    now = datetime.now(timezone.utc)
    partition_date = ingestion_date or now.date().isoformat()
    try:
        datetime.strptime(partition_date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("ingestion_date must use YYYY-MM-DD") from error
    events_dir, event_records = _land_group(
        source_dir, landing_root, "clickstream", ("events.csv",), partition_date
    )
    products_dir, product_records = _land_group(
        source_dir,
        landing_root,
        "product_metadata",
        ("item_properties_part1.csv", "item_properties_part2.csv"),
        partition_date,
    )
    categories_dir, category_records = _land_group(
        source_dir,
        landing_root,
        "category_reference",
        ("category_tree.csv",),
        partition_date,
    )
    manifest_dir = landing_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / (
        "ingestion_timestamp=" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    manifest = {
        "source": "retailrocket",
        "ingested_at": now.isoformat(),
        "ingestion_date": partition_date,
        "files": event_records + product_records + category_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Landing snapshot complete: %s", manifest_path)
    return LandingSnapshot(
        events_dir, products_dir, categories_dir, manifest_path, partition_date
    )


def snapshot_as_dict(snapshot: LandingSnapshot) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(snapshot).items()}
