# Prefect execution evidence

Generated on 21 July 2026 against the complete RetailRocket source data.

## Daily deployment

- Flow/deployment: `recomart-full-pipeline/daily-full-pipeline`
- Deployment ID: `3987c56b-dd4e-4be7-a006-0c68882c2ec9`
- Schedule: `0 2 * * *`
- Timezone: `Asia/Kolkata`
- Schedule slug: `daily-2am-ist`
- Active: yes
- Concurrency limit: 1
- Work pool: `recomart-process` (`READY` when verified)
- Worker: `recomart-daily-worker`
- Version-controlled definition: `prefect.yaml`

The local Prefect server and process worker must remain running for the 02:00
job to start. Registration and startup commands are documented in the main
README.

## Successful full-data run

- Run name: `submission-evidence-full-data-20260721`
- Run ID: `9793e09b-5670-45c3-87c9-735c22d97478`
- State: `Completed`
- Start: `2026-07-21 23:15:43 IST`
- End: `2026-07-21 23:21:30 IST`
- Runtime: 347.265 seconds
- Full-data parameters: `limit=null`, `api_page_size=100000`
- Parent-flow API logs: `prefect-full-run-9793e09b.json`
- Detailed successful curation/task logs: `prefect-curation-success-efbd1555.json`

Key full-data row counts:

| Dataset | Rows |
|---|---:|
| Bronze events | 2,756,101 |
| Bronze item properties | 20,275,902 |
| Bronze category tree | 1,669 |
| Silver user events | 2,756,101 |
| Silver products | 417,053 |
| Silver category hierarchy | 1,669 |
| Gold user-item features | 2,145,179 |
| Gold item features | 417,053 |

## Controlled retry/failure evidence

- Run name: `submission-evidence-controlled-failure-20260721`
- Run ID: `7e8e92eb-9afb-44dd-a2e6-f10456b7de58`
- State: `Failed` (expected)
- Isolated database: `data/evidence-failure.db`
- Deliberately missing source: `data/missing-evidence-source/events.csv`
- API logs: `prefect-controlled-failure-7e8e92eb.json`

The Bronze task logged `Retry 1/1 will start 5 second(s) from now`, retried,
then logged `Retries are exhausted`. This is deliberate evidence of ingestion
error handling and does not indicate a defect in the full-data run.

## Validation, reproducibility, and tests

- Data quality: success, 0 critical failures, 12/12 Great Expectations checks
- Quality JSON: `../data_quality_report.json`
- Quality PDF: `../../output/pdf/recomart_data_quality_report.pdf`
- DVC: `Data and pipelines are up to date.` after `dvc repro`
- Tests: 56 passed
- Line coverage: 91.68% (90% threshold satisfied)

## Final model evidence

Evaluation used 895 eligible transaction-target users (139 warm and 756 cold).

| Segment/model | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 |
|---|---:|---:|---:|---:|
| Hybrid, all users | 0.004581 | 0.034016 | 0.024734 | 0.045810 |
| Hybrid, warm users | 0.010072 | 0.089928 | 0.052153 | 0.100719 |
| Item-CF, warm users | 0.007194 | 0.064748 | 0.039164 | 0.071942 |

The machine-readable metrics are generated at `../model_metrics.json` and
tracked as a DVC metric/output even though the generated file is Git-ignored.

## Captured Prefect UI screenshots

The following submission-ready images are in `../../output/evidence/`:

1. `prefect-daily-deployment.png` — next 02:00 run, deployment status, completed
   full-data run, and controlled failure.
2. `prefect-full-run-completed.png` — completed full pipeline and its curation
   and modeling subflow timeline.
3. `prefect-curation-tasks-completed.png` — Bronze, Silver, Gold, feature,
   registry, and validation task timeline, all completed.
4. `prefect-controlled-failure-retry.png` — missing-source error, Retry 1/1,
   five-second delay, and exhausted-retries message.
5. `prefect-work-pool-worker-online.png` — ready process pool and online
   `recomart-daily-worker` heartbeat.

The exact cron and timezone are also preserved in `prefect.yaml` and
`run-summary.json`, avoiding reliance on screenshots alone.
