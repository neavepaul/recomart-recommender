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

### Feature engineering and transformation

Run after Gold with:

```powershell
python -m src.recomart build-features --neighbors 50 --min-cooccurrence 2
```

`build-features` reads the Gold tables and materialises reusable feature tables
for recommendation algorithms. The full SQL schema is in
[docs/feature_schema.sql](docs/feature_schema.sql).

| Table | Grain | Feature logic |
|---|---|---|
| `feature_user_activity` | one row per user | Activity frequency (`total_events`, `distinct_items`), event-type counts, and average rating per user (`avg_interaction_score` = total interaction score / distinct items), plus `purchase_rate` and last-active timestamp. |
| `feature_item_popularity` | one row per catalog item | Popularity (`distinct_users`, `total_interaction_score`, `popularity_rank`), average rating per item (`avg_interaction_score` = score / distinct users), and `conversion_rate` (purchases / views). Cold items are kept with zero counts. |
| `feature_item_cooccurrence` | one row per item-item neighbour | Co-occurrence / similarity feature: weighted item cosine similarity over the full Gold interactions, keeping the top `--neighbors` neighbours per item filtered by `--min-cooccurrence`. |

"Rating" is the weighted interaction score (view = 1, add-to-cart = 3,
transaction = 5); the dataset has no explicit star ratings. The co-occurrence
computation is shared with model training (`src/cooccurrence.py`) so the feature
and modelling similarities stay consistent. These tables are also produced
automatically by `transform` and the curation flow, and are recorded in the
dataset lineage under the `feature_store` layer.

### Feature store and versioned retrieval

The feature tables above are the live "current" values. A lightweight custom
feature store (`src/feature_store.py`) sits on top of them to document each
feature group and to preserve historical versions for reproducible retrieval.
It is intentionally dependency-free — no Feast — reusing the same SQLite store
and the pipeline `run_id` convention.

Register the current features as a new version:

```powershell
python -m src.recomart --db data\recomart.db register-features --retention 5
```

Registration does three things:

- Documents each feature view in the `feature_registry` table (entity, entity
  key, source table, feature columns, transformation, version, row count).
- Snapshots the current rows of every feature table into an append-only
  `<table>_versions` table, tagged with a `feature_version` and timestamp.
- Writes a JSON manifest to `reports/feature_registry.json` describing the
  registered views. `--retention` keeps the most recent N versions per view.

The three registered feature views are:

| Feature view | Entity | Entity key | Source table |
|---|---|---|---|
| `user_activity` | user | `visitor_id` | `feature_user_activity` |
| `item_popularity` | item | `item_id` | `feature_item_popularity` |
| `item_cooccurrence` | item | `source_item_id` | `feature_item_cooccurrence` |

Inspect the latest registered version of every feature view:

```powershell
python -m src.recomart --db data\recomart.db show-registry
```

Retrieve features for one or more entities. Inference uses the latest version;
training pins an explicit version for a reproducible, point-in-time read:

```powershell
# Inference: latest feature values for serving
python -m src.recomart --db data\recomart.db get-features `
    --view item_popularity --id 315543 --for inference

# Training: a specific historical version by run/version id
python -m src.recomart --db data\recomart.db get-features `
    --view user_activity --id 12345 --for training --version <feature_version>
```

The curation flow registers a feature version automatically after building the
feature tables, using the flow `run_id` as the `feature_version`. This links
every retrievable feature snapshot back to its pipeline run in the dataset
lineage, and `reports/feature_registry.json` is versioned by DVC.

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
python -m src.recomart --db data\recomart.db run --api-page-size 100000
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

# 4. Build normalized product-content features from Gold metadata
& $python -m src.recomart --db data\recomart.db build-content-model `
    --vector-size 256

# 5. Tune content and fusion weights on a pre-test validation window
& $python -m src.recomart --db data\recomart.db tune-hybrid `
    --validation-days 14 --k 10

# 6. Evaluate popularity, item-CF, content, and the tuned hybrid
& $python -m src.recomart --db data\recomart.db evaluate-models --k 10
```

### Content-based and hybrid recommendation

`build-content-model` reads `gold_item_features`, not the raw source files. It
turns each Gold sparse vector into a normalized item vector and persists the
matrix under `models/content/`. The vectors already encode anonymized product
properties, category, and parent category. The recommender also adds explicit
same-category and same-parent affinity so that the category tree influences
ranking directly.

For each warm user, content recommendation creates a weighted profile from the
items in that user's pre-cutoff history. It ranks unseen, available products by
80% vector cosine similarity, 15% category affinity, and 5% parent-category
affinity. Interaction weights use `log1p(interaction_score)` so stronger actions
matter without allowing one large score to dominate the profile.

`tune-hybrid` uses the final portion of the training period as validation data;
it never reads the final test targets while selecting weights. It rebuilds a
validation-only item-CF snapshot, tests five content-weight combinations and
eleven item-CF/content fusion weights, and maximizes warm-user NDCG@K across 55
configurations. The winning configuration is stored in `model_hybrid_config`
and `models/content/tuning.json`. Re-running `evaluate-models` automatically
uses it. Use `--validation-cutoff-ms` when an exact reproducible validation
boundary is required.

The hybrid combines the top item-CF and content candidate lists with reciprocal
rank fusion. Before tuning, the defaults are 60% item-CF and 40% content. Tuning
preserves the behavioral signal while allowing metadata-similar candidates into the result. The report
contains `content_similarity` and `item_cf_content_hybrid`, including warm/cold
segments and lift over popularity. Users with no training history cannot form a
collaborative or content profile, so those cold-start users still receive the
popularity fallback.

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
| `model_hybrid_config` | Validation-selected content and fusion weights |

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

## Operational tooling

### Prefect orchestration

The existing modules are exposed as Prefect tasks and flows in
`src/orchestration/prefect_flow.py`. The full DAG is:

```text
Bronze ingestion -> Silver cleaning -> Gold features -> validation
    -> temporal split -> item-CF training -> content model
    -> hybrid tuning -> final evaluation
```

Run only curation, only modeling, or the complete workflow:

```powershell
python -m src.orchestration.prefect_flow --mode curation `
    --db data\recomart.db --api-page-size 100000

python -m src.orchestration.prefect_flow --mode modeling `
    --db data\recomart.db --validation-days 14 --k 10

python -m src.orchestration.prefect_flow --mode full `
    --db data\recomart.db --api-page-size 100000
```

Direct execution starts a temporary local Prefect server. For persistent run
history and the Prefect UI, start a server in one terminal and point the flow at
it from another:

```powershell
prefect server start
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
python -m src.orchestration.prefect_flow --mode full
```

### Dataset lineage metadata

Prefect flows persist operational metadata inside the RecoMart database:

| Table | Contents |
|---|---|
| `metadata_pipeline_runs` | Flow name, parameters, orchestrator, status, start/end timestamps and errors |
| `metadata_dataset_lineage` | Source URI and SHA-256 version, source modification time, ingestion timestamp, row count, storage URI, transformation, upstream datasets and schema version |

Inspect the latest record for every dataset:

```powershell
python -m src.recomart --db data\recomart.db show-lineage
```

### DVC data and pipeline versioning

`data/raw.dvc` versions the four RetailRocket source files in the DVC
content-addressed cache. `dvc.yaml` versions the transformed SQLite database,
content-model artifacts and final metric report as outputs of the complete
Prefect pipeline. Configuration is held in `params.yaml`, and `dvc.lock`
captures the exact dependency and output versions.

```powershell
dvc status
dvc repro
dvc metrics show
```

The local DVC cache is sufficient for local version checkout. Before sharing
data between machines, configure a remote and push the cache:

```powershell
dvc remote add -d storage <remote-url>
dvc push
```

Git tracks `data/raw.dvc`, `dvc.yaml`, `dvc.lock`, `params.yaml`, `.dvc/config`
and `.dvcignore`; it does not track the large data files themselves.

### MLflow model tracking

The Prefect modeling flow creates separate MLflow runs for item-CF training,
content-model construction, hybrid tuning and final evaluation. Each run stores
parameters, nested metrics, identifying tags and JSON metadata artifacts.
Tracking metadata is stored in `mlflow.db`; run artifacts are stored locally
under MLflow's artifact directory.

Open the local MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then visit `http://127.0.0.1:5000` and open the `recomart-recommender`
experiment. Generated MLflow databases and artifacts are ignored by Git.
