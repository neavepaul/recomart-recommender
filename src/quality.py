"""Pandas profiling, Great Expectations validation, and PDF quality reporting."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from src.core import connect, table_counts

logger = logging.getLogger(__name__)


EXPECTED_SCHEMAS = {
    "bronze_events": ["timestamp", "visitorid", "event", "itemid", "transactionid"],
    "bronze_item_properties": ["timestamp", "itemid", "property", "value"],
    "bronze_category_tree": ["categoryid", "parentid"],
    "silver_user_events": [
        "event_timestamp", "visitor_id", "item_id", "event_type",
        "transaction_id", "event_timestamp_ms",
    ],
    "silver_products": ["item_id", "category_id", "available", "encoded_properties"],
    "silver_category_hierarchy": ["category_id", "parent_category_id"],
    "gold_user_item_features": [
        "visitor_id", "item_id", "view_count", "cart_count", "purchase_count",
        "interaction_score", "last_interaction_timestamp",
    ],
    "gold_item_features": [
        "item_id", "category_id", "parent_category_id", "available",
        "item_feature_vector",
    ],
}


def _dependencies():
    try:
        import great_expectations as gx
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Quality reporting requires pandas and great-expectations"
        ) from error
    return pd, gx


def _full_table_checks(db: sqlite3.Connection) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def violations(
        name: str, description: str, sql: str, severity: str = "critical"
    ) -> None:
        observed = int(db.execute(sql).fetchone()[0] or 0)
        checks.append({
            "name": name,
            "description": description,
            "severity": severity,
            "success": observed == 0,
            "violations": observed,
        })

    existing = {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for table, expected in EXPECTED_SCHEMAS.items():
        actual = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
        checks.append({
            "name": f"schema_{table}",
            "description": f"Expected ordered columns: {', '.join(expected)}",
            "severity": "critical",
            "success": table in existing and actual == expected,
            "violations": 0 if table in existing and actual == expected else 1,
            "observed": actual,
        })
    missing = sorted(set(EXPECTED_SCHEMAS) - existing)
    if missing:
        return checks

    violations(
        "missing_bronze_event_keys",
        "Bronze events require timestamp, visitor, item, and event values.",
        "SELECT COUNT(*) FROM bronze_events WHERE timestamp IS NULL OR visitorid IS NULL "
        "OR itemid IS NULL OR event IS NULL",
    )
    violations(
        "duplicate_bronze_events",
        "Exact duplicate Bronze events are reported as a warning.",
        "SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM bronze_events "
        "GROUP BY timestamp,visitorid,event,itemid,transactionid HAVING n>1)",
        "warning",
    )
    violations(
        "invalid_silver_event_types",
        "Silver event types must be view, addtocart, or transaction.",
        "SELECT COUNT(*) FROM silver_user_events WHERE event_type NOT IN "
        "('view','addtocart','transaction') OR event_type IS NULL",
    )
    violations(
        "missing_silver_event_keys",
        "Silver event identifiers and timestamps must be populated.",
        "SELECT COUNT(*) FROM silver_user_events WHERE visitor_id IS NULL "
        "OR item_id IS NULL OR event_timestamp_ms IS NULL OR event_timestamp IS NULL",
    )
    violations(
        "transaction_id_consistency",
        "Transactions require an ID; non-transactions must not have one.",
        "SELECT COUNT(*) FROM silver_user_events WHERE "
        "(event_type='transaction' AND transaction_id IS NULL) OR "
        "(event_type!='transaction' AND transaction_id IS NOT NULL)",
    )
    violations(
        "availability_domain",
        "Availability values must be 0, 1, or null.",
        "SELECT COUNT(*) FROM silver_products WHERE available NOT IN (0,1) "
        "AND available IS NOT NULL",
    )
    violations(
        "duplicate_silver_products",
        "Silver has one row per product.",
        "SELECT COUNT(*)-COUNT(DISTINCT item_id) FROM silver_products",
    )
    violations(
        "orphan_event_items",
        "Silver event items should exist in the product catalog.",
        "SELECT COUNT(*) FROM silver_user_events e LEFT JOIN silver_products p "
        "ON p.item_id=e.item_id WHERE p.item_id IS NULL",
        "warning",
    )
    violations(
        "orphan_parent_categories",
        "Every non-null parent category should exist in the hierarchy.",
        "SELECT COUNT(*) FROM silver_category_hierarchy c "
        "LEFT JOIN silver_category_hierarchy p "
        "ON p.category_id=c.parent_category_id "
        "WHERE c.parent_category_id IS NOT NULL AND p.category_id IS NULL",
        "warning",
    )
    violations(
        "missing_property_values",
        "Bronze product property keys and values must be populated.",
        "SELECT COUNT(*) FROM bronze_item_properties WHERE timestamp IS NULL "
        "OR itemid IS NULL OR property IS NULL OR value IS NULL OR value=''",
    )
    gold_duplicates = int(db.execute(
        "SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n "
        "FROM gold_user_item_features GROUP BY visitor_id,item_id HAVING n>1)"
    ).fetchone()[0] or 0)
    checks.append({
        "name": "duplicate_gold_user_items",
        "description": "Gold has one row per visitor-item pair.",
        "severity": "critical",
        "success": gold_duplicates == 0,
        "violations": gold_duplicates,
    })
    violations(
        "interaction_score_formula",
        "Gold score must equal views + 3*carts + 5*purchases.",
        "SELECT COUNT(*) FROM gold_user_item_features WHERE interaction_score "
        "!= view_count + 3*cart_count + 5*purchase_count",
    )
    violations(
        "missing_gold_vectors",
        "Every Gold item requires a non-empty content vector.",
        "SELECT COUNT(*) FROM gold_item_features WHERE item_feature_vector IS NULL "
        "OR item_feature_vector=''",
    )
    bronze_events = db.execute("SELECT COUNT(*) FROM bronze_events").fetchone()[0]
    silver_events = db.execute("SELECT COUNT(*) FROM silver_user_events").fetchone()[0]
    checks.append({
        "name": "bronze_to_silver_rejections",
        "description": "Rows removed by Silver validation (informational).",
        "severity": "information",
        "success": True,
        "violations": max(0, int(bronze_events - silver_events)),
        "observed": {"bronze_rows": bronze_events, "silver_rows": silver_events},
    })
    return checks


def _profile_dataframe(pd, table: str, dataframe, full_rows: int) -> dict[str, Any]:
    columns = []
    for name in dataframe.columns:
        series = dataframe[name]
        record: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty:
            record.update({
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "mean": float(series.mean()),
            })
        columns.append(record)
    return {
        "table": table,
        "full_row_count": int(full_rows),
        "sample_row_count": int(len(dataframe)),
        "sample_duplicate_rows": int(dataframe.duplicated().sum()),
        "sample_missing_values": int(dataframe.isna().sum().sum()),
        "columns": columns,
    }


def _gx_validate_dataframe(
    context, name: str, dataframe, expectations
) -> list[dict[str, Any]]:
    source = context.data_sources.add_pandas(f"{name}_source")
    asset = source.add_dataframe_asset(name=f"{name}_asset")
    definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = definition.get_batch(batch_parameters={"dataframe": dataframe})
    results = []
    for expectation in expectations:
        result = batch.validate(expectation)
        details = dict(getattr(result, "result", {}) or {})
        results.append({
            "dataset": name,
            "expectation": type(expectation).__name__,
            "success": bool(result.success),
            "column": getattr(expectation, "column", None),
            "unexpected_count": int(details.get("unexpected_count", 0) or 0),
            "unexpected_percent": float(details.get("unexpected_percent", 0) or 0),
        })
    return results


def _pandas_and_gx_profiles(
    db: sqlite3.Connection, sample_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        pd, gx = _dependencies()
        logging.getLogger("great_expectations").setLevel(logging.WARNING)
        context = gx.get_context(mode="ephemeral")
    finally:
        logging.disable(prior_disable_level)
    from great_expectations.data_context.types.base import ProgressBarsConfig
    context.variables.progress_bars = ProgressBarsConfig(
        globally=False, metric_calculations=False
    )
    specifications = [
        (
            "silver_user_events",
            [
                gx.expectations.ExpectColumnValuesToNotBeNull(column="visitor_id"),
                gx.expectations.ExpectColumnValuesToNotBeNull(column="item_id"),
                gx.expectations.ExpectColumnValuesToNotBeNull(column="event_timestamp_ms"),
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column="event_type", value_set=["view", "addtocart", "transaction"]
                ),
            ],
        ),
        (
            "silver_products",
            [
                gx.expectations.ExpectColumnValuesToNotBeNull(column="item_id"),
                gx.expectations.ExpectColumnValuesToBeUnique(column="item_id"),
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column="available", value_set=[0, 1], mostly=1.0
                ),
            ],
        ),
        (
            "gold_user_item_features",
            [
                gx.expectations.ExpectColumnValuesToNotBeNull(column="visitor_id"),
                gx.expectations.ExpectColumnValuesToNotBeNull(column="item_id"),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="interaction_score", min_value=1
                ),
            ],
        ),
        (
            "gold_item_features",
            [
                gx.expectations.ExpectColumnValuesToBeUnique(column="item_id"),
                gx.expectations.ExpectColumnValuesToNotBeNull(
                    column="item_feature_vector"
                ),
            ],
        ),
    ]
    profiles: list[dict[str, Any]] = []
    validation_results: list[dict[str, Any]] = []
    for table, expectations in specifications:
        logger.info("Quality profile: loading up to %s rows from %s", f"{sample_rows:,}", table)
        full_rows = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        dataframe = pd.read_sql_query(
            f"SELECT * FROM {table} LIMIT ?", db, params=(sample_rows,)
        )
        profiles.append(_profile_dataframe(pd, table, dataframe, full_rows))
        validation_results.extend(
            _gx_validate_dataframe(context, table, dataframe, expectations)
        )
    return profiles, validation_results


def _pdf_dependencies():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError as error:
        raise RuntimeError("PDF reporting requires reportlab") from error
    return {
        "colors": colors, "TA_CENTER": TA_CENTER, "A4": A4,
        "ParagraphStyle": ParagraphStyle, "getSampleStyleSheet": getSampleStyleSheet,
        "mm": mm, "PageBreak": PageBreak, "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate, "Spacer": Spacer,
        "Table": Table, "TableStyle": TableStyle,
    }


def write_quality_pdf(report: dict[str, Any], output_path: Path) -> None:
    r = _pdf_dependencies()
    colors = r["colors"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = r["getSampleStyleSheet"]()
    styles.add(r["ParagraphStyle"](
        name="ReportTitle", parent=styles["Title"], alignment=r["TA_CENTER"],
        textColor=colors.HexColor("#17324D"), fontSize=22, leading=27,
    ))
    styles.add(r["ParagraphStyle"](
        name="Small", parent=styles["BodyText"], fontSize=7.5, leading=9.5,
    ))
    document = r["SimpleDocTemplate"](
        str(output_path), pagesize=r["A4"], rightMargin=16*r["mm"],
        leftMargin=16*r["mm"], topMargin=18*r["mm"], bottomMargin=18*r["mm"],
        title="RecoMart Data Quality Report", author="RecoMart Data Platform Team",
    )

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667788"))
        canvas.drawString(16*r["mm"], 10*r["mm"], "RecoMart - Data Quality Report")
        canvas.drawRightString(194*r["mm"], 10*r["mm"], f"Page {doc.page}")
        canvas.restoreState()

    def paragraph(value, style="Small"):
        return r["Paragraph"](escape(str(value)), styles[style])

    def table(rows, widths):
        item = r["Table"](rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(r["TableStyle"]([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C4CE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return item

    story = [
        paragraph("RecoMart Data Quality Report", "ReportTitle"),
        r["Spacer"](1, 5*r["mm"]),
        paragraph(f"Generated: {report['generated_at']}", "BodyText"),
        paragraph(f"Database: {report['database']}", "BodyText"),
        paragraph(
            "Overall status: PASS" if report["success"] else "Overall status: FAIL",
            "Heading2",
        ),
        paragraph(
            "Full-table SQL checks validate integrity across every stored row. "
            "Pandas supplies descriptive profiles and Great Expectations validates "
            f"deterministic samples of up to {report['sample_rows']:,} rows per dataset.",
            "BodyText",
        ),
        r["Spacer"](1, 4*r["mm"]),
        paragraph("Dataset row counts", "Heading2"),
    ]
    count_rows = [[paragraph("Dataset"), paragraph("Rows")]] + [
        [paragraph(name), paragraph(f"{count:,}")]
        for name, count in report["table_counts"].items()
    ]
    story += [
        table(count_rows, [120*r["mm"], 40*r["mm"]]),
        r["PageBreak"](),
    ]
    story.append(paragraph("Full-table validation checks", "Heading2"))
    check_rows = [[paragraph(x) for x in ("Check", "Severity", "Status", "Violations")]]
    for check in report["checks"]:
        check_rows.append([
            paragraph(check["name"]), paragraph(check["severity"]),
            paragraph("PASS" if check["success"] else "FAIL"),
            paragraph(f"{check['violations']:,}"),
        ])
    story += [
        table(check_rows, [88*r["mm"], 28*r["mm"], 22*r["mm"], 28*r["mm"]]),
        r["Spacer"](1, 6*r["mm"]),
        paragraph("Pandas profiling summary", "Heading2"),
    ]
    profile_rows = [[paragraph(x) for x in ("Dataset", "Full rows", "Sample", "Missing", "Duplicates")]]
    for profile in report["profiles"]:
        profile_rows.append([
            paragraph(profile["table"]), paragraph(f"{profile['full_row_count']:,}"),
            paragraph(f"{profile['sample_row_count']:,}"),
            paragraph(f"{profile['sample_missing_values']:,}"),
            paragraph(f"{profile['sample_duplicate_rows']:,}"),
        ])
    story += [table(profile_rows, [65*r["mm"], 27*r["mm"], 25*r["mm"], 25*r["mm"], 25*r["mm"]])]
    story += [r["Spacer"](1, 6*r["mm"]), paragraph("Great Expectations results", "Heading2")]
    gx_rows = [[paragraph(x) for x in ("Dataset", "Expectation", "Column", "Status", "Unexpected")]]
    for result in report["great_expectations"]:
        gx_rows.append([
            paragraph(result["dataset"]), paragraph(result["expectation"]),
            paragraph(result.get("column") or "table"),
            paragraph("PASS" if result["success"] else "FAIL"),
            paragraph(f"{result['unexpected_count']:,}"),
        ])
    story += [table(gx_rows, [42*r["mm"], 61*r["mm"], 27*r["mm"], 18*r["mm"], 22*r["mm"]])]
    failures = [check for check in report["checks"] if not check["success"]]
    story += [r["Spacer"](1, 6*r["mm"]), paragraph("Issues and interpretation", "Heading2")]
    if failures:
        for check in failures:
            story.append(paragraph(
                f"{check['severity'].upper()}: {check['name']} - "
                f"{check['violations']:,} violations. {check['description']}",
                "BodyText",
            ))
    else:
        story.append(paragraph("No validation issues were detected.", "BodyText"))
    document.build(story, onFirstPage=page, onLaterPages=page)


def generate_quality_report(
    db_path: Path,
    json_path: Path = Path("reports/data_quality_report.json"),
    pdf_path: Path = Path("output/pdf/recomart_data_quality_report.pdf"),
    sample_rows: int = 100_000,
) -> dict[str, Any]:
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    db = connect(db_path)
    try:
        logger.info("Quality report: running full-table integrity checks")
        checks = _full_table_checks(db)
        logger.info("Quality report: running Pandas profiles and Great Expectations")
        profiles, gx_results = _pandas_and_gx_profiles(db, sample_rows)
        counts = table_counts(db)
    finally:
        db.close()
    critical_failures = [
        check for check in checks
        if check["severity"] == "critical" and not check["success"]
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(db_path.resolve()),
        "success": not critical_failures and all(
            result["success"] for result in gx_results
        ),
        "critical_failures": len(critical_failures),
        "sample_rows": sample_rows,
        "table_counts": counts,
        "checks": checks,
        "profiles": profiles,
        "great_expectations": gx_results,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_quality_pdf(report, pdf_path)
    logger.info("Quality report written to %s and %s", json_path, pdf_path)
    report["json_report"] = str(json_path)
    report["pdf_report"] = str(pdf_path)
    return report
