# RecoMart Recommender Data Pipeline

RecoMart is a local, dependency-free data-curation pipeline for the RetailRocket
e-commerce dataset. It uses the Bronze, Silver, and Gold medallion architecture
to turn raw interaction and product data into recommendation-ready features.

All layers are stored in one SQLite database (`data/recomart.db` by default).
They remain separate through the `bronze_`, `silver_`, and `gold_` table prefixes.
Each pipeline reads its input from the prior layer's database tables; only Bronze
reads from external sources.

```text
events.csv -----------+
                       +--> Bronze tables --> Silver tables --> Gold tables
Product REST API ------+
                       |
category_tree.csv -----+
```

## Data sources and ingestion patterns

| Source | Simulation / ingestion pattern | Bronze destination | Purpose |
|---|---|---|---|
| `events.csv` | Timestamp-based clickstream replay | `bronze_events` | User views, cart additions, and transactions |
| `item_properties_part1.csv` and `item_properties_part2.csv` | Mock paginated REST API, consumed over HTTP | `bronze_item_properties` | Product category, availability, and anonymous properties |
| `category_tree.csv` | Batch CSV ingestion | `bronze_category_tree` | Category-to-parent relationships |

The item-properties reader handles normal quoted CSV commas and also restores
unquoted commas in the final `value` column. Therefore an item property such as
`blue,small,sale` remains one logical value rather than becoming extra columns.

## Project structure

```text
src/
|-- core.py                    # configuration, database connection, shared helpers
|-- ingestion/
|   |-- events.py              # clickstream replay source adapter
|   |-- categories.py          # category CSV source adapter
|   |-- item_property_csv.py   # resilient item-properties CSV parser
|   |-- product_api.py         # mock paginated HTTP source
|   `-- products.py            # REST client source adapter
|-- pipelines/
|   |-- bronze.py              # coordinates source adapters into Bronze
|   |-- silver.py              # Bronze-to-Silver curation
|   |-- gold.py                # Silver-to-Gold feature generation
|   `-- runner.py              # complete Silver + Gold orchestration
|-- validation.py              # data-quality checks
`-- recomart.py                # command-line interface
```

## Pipeline behaviour

### Bronze: raw ingestion

Run with:

```powershell
python -m src.recomart build-bronze --limit 10000 --api-page-size 1000
```

`build-bronze` is the ingestion pipeline. It reads each external source and
rebuilds the following raw tables:

| Table | Columns | What the pipeline does |
|---|---|---|
| `bronze_events` | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` | Reads `events.csv` in source order and optionally delays between timestamp batches to simulate real-time arrival. It makes no business-level changes to event data. |
| `bronze_item_properties` | `timestamp`, `itemid`, `property`, `value` | Starts an in-process mock REST API backed by the two raw CSV parts, requests it page by page, and stores the API records as received. |
| `bronze_category_tree` | `categoryid`, `parentid` | Batch loads the category hierarchy CSV unchanged. |

The mock product API is also available separately for a live demonstration:

```powershell
# Terminal 1
python -m src.recomart serve-api --port 8000

# Terminal 2
python -m src.recomart ingest-products --api-url http://127.0.0.1:8000
```

Clickstream replay options:

- `--speed 0` ingests as fast as possible.
- A positive `--speed` scales timestamp gaps. For example, `--speed 86400`
  maps one source day to approximately one second.
- `--limit` limits input rows for a fast local test.

### Silver: cleaning and structuring

Run after Bronze with:

```powershell
python -m src.recomart build-silver
```

`build-silver` reads only the three Bronze tables and rebuilds these curated
tables:

| Table | Columns | Transformations |
|---|---|---|
| `silver_user_events` | `event_timestamp`, `visitor_id`, `item_id`, `event_type`, `transaction_id` | Converts epoch milliseconds to timestamps, renames fields to a consistent style, retains only `view`, `addtocart`, and `transaction`, and drops invalid negative IDs or non-positive timestamps. |
| `silver_products` | `item_id`, `category_id`, `available`, `encoded_properties` | Finds the latest value for each `(itemid, property)` based on timestamp, takes latest `categoryid` and `available`, and collects all remaining anonymous properties as a JSON array. |
| `silver_category_hierarchy` | `category_id`, `parent_category_id` | Renames the raw category columns into a consistent model schema. |

The anonymous properties are retained because they can still indicate content
similarity, even though the property names and values are anonymized.

### Gold: recommendation-ready features

Run after Silver with:

```powershell
python -m src.recomart build-gold --vector-size 256
```

`build-gold` reads only Silver tables and rebuilds model-ready tables:

| Table | Columns | Transformations |
|---|---|---|
| `gold_user_item_features` | `visitor_id`, `item_id`, `view_count`, `cart_count`, `purchase_count`, `interaction_score`, `last_interaction_timestamp` | Aggregates each visitor-item history. Interaction score weights are view = 1, add-to-cart = 3, transaction = 5. |
| `gold_item_features` | `item_id`, `category_id`, `parent_category_id`, `available`, `item_feature_vector` | Joins products to category hierarchy and creates a sparse content vector from anonymous properties, category, and parent category. |

`item_feature_vector` is stored as sparse JSON. It uses deterministic feature
hashing: every product token is assigned to a numbered vector bucket and only
non-zero buckets are stored.

`--vector-size` controls the number of possible buckets:

- `64`: quick demo or test run.
- `256`: default and a sensible starting point.
- `512` or `1024`: fewer hash collisions for the full dataset, with more
  storage and model input dimensions.

Products with `available = 0` are intentionally retained in Gold. A serving or
ranking step can use this field to avoid recommending unavailable products.

## Running the complete curation flow

Set the bundled Python runtime once per PowerShell session:

```powershell
$python = "C:\Users\neave\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
```

### Small smoke test

```powershell
python -m src.recomart --db data\test.db build-bronze --limit 10000 --api-page-size 1000
python -m src.recomart --db data\test.db build-silver
python -m src.recomart --db data\test.db build-gold --vector-size 256
python -m src.recomart --db data\test.db validate
```

### One-command run

This runs Bronze ingestion, Silver curation, and Gold feature generation:

```powershell
python -m src.recomart --db data\test.db run --limit 10000 --api-page-size 1000
python -m src.recomart --db data\test.db validate
```

### Full dataset

Omit `--limit` after the small test succeeds:

```powershell
python -m src.recomart --db data\recomart.db run --api-page-size 5000
python -m src.recomart --db data\recomart.db validate
```

## Validation

Run:

```powershell
python -m src.recomart --db data\recomart.db validate
```

Validation returns table row counts and confirms:

- Silver contains only supported event types.
- Silver product IDs are unique.
- Gold visitor-item pairs are unique.
- Gold interaction scores equal `views + (3 × carts) + (5 × purchases)`.
- Every Gold item has an item-feature vector.

A successful validation result includes `"ok": true`.

## Idempotency and storage

Every pipeline rebuilds its destination tables transactionally. Re-running
`build-bronze`, `build-silver`, or `build-gold` replaces that layer's prior
output; it does not append duplicate data. The command prints the resulting row
counts so each run can be checked immediately.

The generated SQLite files are ignored by Git (`data/*.db`).
