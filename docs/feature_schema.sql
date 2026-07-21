-- RecoMart — Task 6 Feature Engineering SQL Schema
-- ---------------------------------------------------------------------------
-- Feature tables materialised by src/pipelines/features.py (build_features).
-- Source: the Gold layer (gold_user_item_features, gold_item_features).
-- Storage engine at runtime is SQLite; this DDL documents the logical schema
-- and is portable to a warehouse with minimal type changes.
-- ---------------------------------------------------------------------------

-- Per-user activity features -------------------------------------------------
-- Grain: one row per visitor. Captures activity frequency and average rating
-- (weighted interaction score) per user.
CREATE TABLE feature_user_activity (
    visitor_id               INTEGER PRIMARY KEY,
    distinct_items           INTEGER NOT NULL,  -- number of distinct items touched
    total_events             INTEGER NOT NULL,  -- views + carts + purchases (activity frequency)
    view_count               INTEGER NOT NULL,
    cart_count               INTEGER NOT NULL,
    purchase_count           INTEGER NOT NULL,
    total_interaction_score  REAL    NOT NULL,  -- SUM(interaction_score)
    avg_interaction_score    REAL    NOT NULL,  -- total_interaction_score / distinct_items
    purchase_rate            REAL    NOT NULL,  -- purchase_count / total_events
    last_active_timestamp    TEXT               -- MAX(last_interaction_timestamp)
);

-- Per-item popularity features -----------------------------------------------
-- Grain: one row per catalog item (includes cold items with zero interactions).
CREATE TABLE feature_item_popularity (
    item_id                  INTEGER PRIMARY KEY,
    category_id              INTEGER,
    parent_category_id       INTEGER,
    available                INTEGER,
    distinct_users           INTEGER NOT NULL,  -- COUNT(DISTINCT visitor_id)
    view_count               INTEGER NOT NULL,
    cart_count               INTEGER NOT NULL,
    purchase_count           INTEGER NOT NULL,
    total_interaction_score  REAL    NOT NULL,
    avg_interaction_score    REAL    NOT NULL,  -- total_interaction_score / distinct_users
    conversion_rate          REAL    NOT NULL,  -- purchase_count / view_count
    popularity_rank          INTEGER NOT NULL   -- dense rank by total_interaction_score DESC
);

-- Item co-occurrence / similarity features -----------------------------------
-- Grain: one row per (source item, similar item) neighbour pair.
-- cosine similarity of weighted co-occurrence over the full Gold interactions.
CREATE TABLE feature_item_cooccurrence (
    source_item_id           INTEGER NOT NULL,
    similar_item_id          INTEGER NOT NULL,
    similarity               REAL    NOT NULL,  -- cosine similarity in [0, 1]
    cooccurring_users        INTEGER NOT NULL,  -- users interacting with both items
    neighbor_rank            INTEGER NOT NULL,  -- 1 = most similar neighbour
    PRIMARY KEY (source_item_id, similar_item_id)
);

-- Supporting index used for top-k neighbour retrieval.
CREATE INDEX ix_feature_item_cooccurrence_source
    ON feature_item_cooccurrence (source_item_id, neighbor_rank);

-- Intermediate table (weighted item-pair dot products and co-occurrence counts)
-- produced during similarity computation; retained for lineage / debugging.
CREATE TABLE feature_item_pair_stats (
    item_i                   INTEGER NOT NULL,
    item_j                   INTEGER NOT NULL,
    dot_product              REAL    NOT NULL,
    cooccurring_users        INTEGER NOT NULL,
    PRIMARY KEY (item_i, item_j)
);
