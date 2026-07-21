"""EDA plot generation for Gold recommendation data.

Renders the summary plots required by the data preparation stage (interaction
distribution histograms, item/category popularity bars, and an item
co-occurrence heatmap) as PNG files. Statistics are sourced from
``profile_gold`` and a few direct queries so the plots stay consistent with the
JSON profile report.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, file-only rendering
import matplotlib.pyplot as plt  # noqa: E402

from src.core import connect  # noqa: E402
from src.modeling.profile import profile_gold  # noqa: E402

logger = logging.getLogger(__name__)


def _histogram_rows(db: sqlite3.Connection, entity: str) -> list[tuple[int, int]]:
    if entity not in {"visitor_id", "item_id"}:
        raise ValueError("unsupported histogram entity")
    return db.execute(
        f"""SELECT interactions, COUNT(*) FROM (
                SELECT {entity}, COUNT(*) AS interactions
                FROM gold_user_item_features GROUP BY {entity}
            ) GROUP BY interactions ORDER BY interactions"""
    ).fetchall()


def _cooccurrence_matrix(
    db: sqlite3.Connection, item_ids: list[int]
) -> list[list[int]]:
    """Symmetric matrix of shared users between the given items."""
    index = {item: position for position, item in enumerate(item_ids)}
    size = len(item_ids)
    matrix = [[0] * size for _ in range(size)]
    if not item_ids:
        return matrix
    placeholders = ",".join("?" for _ in item_ids)
    diagonal = db.execute(
        f"""SELECT item_id, COUNT(DISTINCT visitor_id)
            FROM gold_user_item_features WHERE item_id IN ({placeholders})
            GROUP BY item_id""",
        item_ids,
    )
    for item, users in diagonal:
        matrix[index[item]][index[item]] = users
    pairs = db.execute(
        f"""SELECT a.item_id, b.item_id, COUNT(*) AS shared
            FROM gold_user_item_features a
            JOIN gold_user_item_features b
              ON a.visitor_id = b.visitor_id AND a.item_id < b.item_id
            WHERE a.item_id IN ({placeholders}) AND b.item_id IN ({placeholders})
            GROUP BY a.item_id, b.item_id""",
        item_ids + item_ids,
    )
    for left, right, shared in pairs:
        i, j = index[left], index[right]
        matrix[i][j] = shared
        matrix[j][i] = shared
    return matrix


def _save(fig: "plt.Figure", out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote plot %s", path)
    return path


def _plot_interaction_totals(profile: dict[str, Any], out_dir: Path) -> Path:
    totals = profile["interaction_totals"]
    labels = ["views", "add_to_carts", "purchases"]
    values = [totals["views"], totals["add_to_carts"], totals["purchases"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("Interaction totals by event type")
    ax.set_ylabel("count")
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.margins(y=0.15)
    return _save(fig, out_dir, "interaction_totals.png")


def _plot_distribution(
    rows: list[tuple[int, int]], title: str, xlabel: str, out_dir: Path, name: str
) -> Path:
    values = [value for value, _ in rows]
    frequencies = [frequency for _, frequency in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values, frequencies, width=0.9, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of entities (log scale)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    return _save(fig, out_dir, name)


def _plot_top_items(profile: dict[str, Any], out_dir: Path) -> Path:
    items = profile["most_common_items"]
    labels = [str(entry["item_id"]) for entry in items][::-1]
    scores = [entry["interaction_score"] for entry in items][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, scores, color="#55a868")
    ax.set_title(f"Top {len(items)} items by interaction score")
    ax.set_xlabel("weighted interaction score")
    ax.set_ylabel("item_id")
    return _save(fig, out_dir, "top_items.png")


def _plot_top_categories(profile: dict[str, Any], out_dir: Path) -> Path:
    categories = [
        entry for entry in profile["most_common_categories"]
        if entry["category_id"] is not None
    ]
    labels = [str(entry["category_id"]) for entry in categories][::-1]
    scores = [entry["interaction_score"] for entry in categories][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, scores, color="#c44e52")
    ax.set_title(f"Top {len(categories)} categories by interaction score")
    ax.set_xlabel("weighted interaction score")
    ax.set_ylabel("category_id")
    return _save(fig, out_dir, "top_categories.png")


def _plot_cooccurrence_heatmap(
    db: sqlite3.Connection, profile: dict[str, Any], out_dir: Path
) -> Path:
    item_ids = [entry["item_id"] for entry in profile["most_common_items"]]
    matrix = _cooccurrence_matrix(db, item_ids)
    labels = [str(item) for item in item_ids]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Shared-user co-occurrence among top items")
    fig.colorbar(image, ax=ax, label="users interacting with both")
    return _save(fig, out_dir, "item_cooccurrence_heatmap.png")


def generate_eda_plots(
    db_path: Path, out_dir: Path = Path("reports/eda"), top_n: int = 15
) -> dict[str, Any]:
    """Render the EDA summary plots and return their paths plus the profile."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_gold(db_path, top_n)
    db = connect(db_path)
    try:
        logger.info("Generating EDA plots in %s", out_dir)
        plots = {
            "interaction_totals": _plot_interaction_totals(profile, out_dir),
            "items_per_user": _plot_distribution(
                _histogram_rows(db, "visitor_id"),
                "Items per user distribution", "items per user",
                out_dir, "items_per_user_hist.png",
            ),
            "users_per_item": _plot_distribution(
                _histogram_rows(db, "item_id"),
                "Users per item distribution", "users per item",
                out_dir, "users_per_item_hist.png",
            ),
            "top_items": _plot_top_items(profile, out_dir),
            "top_categories": _plot_top_categories(profile, out_dir),
            "item_cooccurrence_heatmap": _plot_cooccurrence_heatmap(
                db, profile, out_dir
            ),
        }
    finally:
        db.close()
    logger.info("EDA plots complete: %s files", len(plots))
    return {"output_dir": str(out_dir), "plots": {k: str(v) for k, v in plots.items()}}
