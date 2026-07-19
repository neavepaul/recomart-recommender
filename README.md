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

## Progress and logging

Commands log their current stage to stderr by default. Long row-oriented work
shows a tqdm-style counter with processed rows, throughput, and elapsed time.
Long SQLite statements and index builds show a liveness counter, so a quiet
terminal does not look like a stalled process.

Example output:

```text
21:21:27 | INFO | Bronze pipeline started
Bronze events: 2,000,000 rows | 84,210 rows/s | 00:23
Bronze product properties: 8,500,000 rows | 31,400 rows/s | 04:30
Silver latest product properties: 91,000,000 SQLite steps | 00:42
Gold item vectors: 300,000 items | 18,500 items/s | 00:16
```

The final JSON result is written to stdout, while progress is written to stderr.
Log verbosity can be changed before the command name:

```powershell
python -m src.recomart --db data\recomart.db --log-level WARNING run
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

## Offline recommendation evaluation

### Complete model-development workflow

After building Gold, run these commands in order:

```powershell
# 1. Inspect Gold interactions, activity, sparsity, availability, items and categories
& $python -m src.recomart --db data\recomart.db profile-gold --top 10

# 2. Persist a leakage-safe temporal split
& $python -m src.recomart --db data\recomart.db prepare-model-data `
    --target transaction --test-days 14

# 3. Train the popularity baseline and item-based collaborative filtering
& $python -m src.recomart --db data\recomart.db train-models `
    --max-history 30 --min-cooccurrence 2 --neighbors 50

# 4. Evaluate both models on exactly the same future targets
& $python -m src.recomart --db data\recomart.db evaluate-models --k 10
```

`prepare-model-data` reads timestamped Silver events because the normal Gold
interaction table summarizes the entire timeline. It writes a training table
with the same aggregated user-item features as Gold, but only from events before
the cutoff. It also writes novel, available future purchase targets. This avoids
training on behavior that belongs in the test period.

`train-models` persists two models:

- `model_popularity` ranks available items by training-period interaction score.
- `model_item_similarity` is a weighted item-to-item cosine collaborative model.
  Items become similar when the same users interacted with both. User histories
  are capped by `--max-history` to control pair growth, weak pairs are removed by
  `--min-cooccurrence`, and the best `--neighbors` similarities per item are kept.

The collaborative recommender scores candidates from each user's weighted
training history. It excludes seen and unavailable items and uses popularity to
fill empty recommendation slots for sparse or cold-start users. The evaluation
report compares both models and reports collaborative-filtering lift over the
popularity benchmark.

The persisted modeling tables are:

| Table | Purpose |
|---|---|
| `model_split_metadata` | Cutoff, target definition, and source date range |
| `model_train_user_items` | Pre-cutoff user-item training features |
| `model_test_targets` | Novel available post-cutoff target items |
| `model_popularity` | Trained global popularity ranks |
| `model_item_pair_stats` | Weighted co-occurrence sufficient statistics |
| `model_item_similarity` | Top cosine-similar neighbors for each item |

### Standalone popularity evaluation

The project includes a time-based weighted-popularity baseline. Earlier Silver
events train the ranking; later transactions or high-intent events become the
implicit test targets. Recommendations are limited to available products and
items already seen during training are excluded from both recommendations and
novel-item targets.

Evaluate future purchases over the final 14 days:

```powershell
& $python -m src.recomart --db data\recomart.db evaluate --k 10 --target transaction --test-days 14
```

Evaluate future add-to-cart or purchase events:

```powershell
& $python -m src.recomart --db data\recomart.db evaluate --k 10 --target high-intent --test-days 14
```

For a fixed reproducible boundary, pass a RetailRocket epoch-millisecond value
with `--cutoff-ms`. The report includes:

- `precision@K`: relevant future items divided by K recommendations.
- `recall@K`: relevant future items retrieved divided by all relevant targets.
- `ndcg@K`: ranking quality, rewarding relevant items near the top.
- `hit_rate@K`: fraction of evaluated users receiving at least one correct item.
- `catalog_coverage@K`: fraction of available products recommended to anyone.
- Eligible user, target-item, and catalog counts.

This popularity result is the benchmark that future collaborative-filtering and
hybrid models should beat using exactly the same temporal split and targets.

## Idempotency and storage

Every pipeline rebuilds its destination tables transactionally. Re-running
`build-bronze`, `build-silver`, or `build-gold` replaces that layer's prior
output; it does not append duplicate data. The command prints the resulting row
counts so each run can be checked immediately.

The generated SQLite files are ignored by Git (`data/*.db`).
