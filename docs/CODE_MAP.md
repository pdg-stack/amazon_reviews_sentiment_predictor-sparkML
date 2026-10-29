# Code map: how the files actually depend on each other

A quick-reference for "if I change file X, what else might be affected?" and "where do I even start reading?" — not a restatement of what each file does (see `ARCHITECTURE_NOTES.md`/`OBSERVABILITY.md`/`model_plan.md` for that), just the *dependency graph* between files and the entry points that actually run them.

## The one file everything else depends on

```
src/config.py
```

No internal imports of its own (stdlib + PySpark only). Every other file below imports from it — it's the single place that defines "where do paths live" and "how do we start a Spark session." Change something here and check every section below, since nothing in `src/` is independent of it.

## Dependency graph

An arrow means "imports from" — read `A -> B` as "A imports B," i.e. changing B can affect A.

```
src/config.py
  ^
  |  (imported by everything below -- arrows omitted per-node for readability,
  |   shown explicitly only where a file imports something OTHER than config)
  |
src/pipeline/
  data_download.py    -> config
  preprocess.py        -> config
  train.py              -> config
  evaluate.py           -> config
  run_pipeline.py        -> pipeline.{data_download, preprocess, train, evaluate}  (orchestrator; no logic of its own)

src/serving/
  predict.py            -> config
  set_active_model.py   -> config

src/reports/
  compare_versions.py   -> config

src/api/
  security.py            (no internal imports -- stdlib + FastAPI only)
  logging_config.py     -> config
  db.py                  -> config
  drift.py               -> api.db, config
  app.py                 -> api.{db, drift, logging_config, security}, config   (the one file that pulls in all the others)

src/scripts/   (standalone utilities -- none of these import each other)
  generate_api_key.py    (no internal imports)
  make_training_sample.py -> config
  modal_load_test.py      (no internal imports -- deliberately self-contained, see its own docstring on why)
  tail_to_clf.py          (no internal imports)

docker/   (EMR Serverless entrypoints -- not under src/, but still import from it)
  emr_train_job.py       -> config, pipeline.{train, evaluate}
  emr_evaluate_job.py    -> config, pipeline.evaluate

tests/
  test_security.py      -> api.security
  test_drift.py          -> api.{db, drift}
```

**Reading this graph for a specific question:**
- *"I'm changing `config.py` — what's affected?"* Everything. Grep for `from src.config import` to see exactly which names each file actually uses.
- *"I'm changing `src/api/db.py`'s function signatures — what's affected?"* `src/api/drift.py` and `src/api/app.py` (directly), plus `tests/test_drift.py` (which mocks it).
- *"I'm changing `train.py`'s CLI arguments — what's affected?"* `src/pipeline/run_pipeline.py` (calls its `main()`) and `docker/emr_train_job.py` (same).
- *"Can I change `evaluate.py` without touching `train.py`?"* Yes — nothing in the graph above points from `train.py` to `evaluate.py` or back; they're only connected through `run_pipeline.py`'s orchestration and through files on disk (the registered model, `baseline_stats.json`).

## Entry points: where execution actually starts

Nothing above runs on its own — each file is a module, not a running process. These are the actual commands/triggers that start something, and which file each one enters at:

| How it's invoked | Enters at | What it does |
|---|---|---|
| `python src/pipeline/run_pipeline.py` | `run_pipeline.py` | full pipeline, in order: download -> preprocess -> train -> evaluate |
| `python src/pipeline/data_download.py` | `data_download.py` | just the download step |
| `python src/pipeline/preprocess.py` | `preprocess.py` | just the preprocessing step |
| `python src/pipeline/train.py [--sample-fraction/--n-trials/--cv-folds/--train-path]` | `train.py` | Optuna-tuned training run, logs everything to MLflow |
| `python src/pipeline/evaluate.py --version N` | `evaluate.py` | scores a registered version against the held-out test set, writes charts + `baseline_stats.json` |
| `python src/serving/predict.py --input ... --output ...` | `predict.py` | batch scoring from the command line |
| `python src/serving/set_active_model.py --version N [--reload-api]` | `set_active_model.py` | the only thing that ever updates `models/active/` |
| `python src/reports/compare_versions.py` | `compare_versions.py` | side-by-side chart comparing registered versions |
| `python src/scripts/generate_api_key.py [--count N]` | `generate_api_key.py` | writes `API_KEY`/`API_KEYS` to `.env` |
| `python src/scripts/make_training_sample.py` | `make_training_sample.py` | draws a train/eval sample from `train.parquet` for cloud training (never touches `test.parquet`) |
| `modal run src/scripts/modal_load_test.py` | `modal_load_test.py` | the real bell-curve load generator, against a live deployment |
| `python src/scripts/tail_to_clf.py <logfile>` | `tail_to_clf.py` | piped into GoAccess by the `log-viewer` container's own command — never run standalone in practice |
| `uvicorn src.api.app:app` | `app.py` | the FastAPI server itself |
| `spark-submit local:///workspace/docker/emr_train_job.py` | `docker/emr_train_job.py` | EMR Serverless's training entrypoint -- runs `train_main()` then `evaluate_main()` in one job |
| `spark-submit local:///workspace/docker/emr_evaluate_job.py` | `docker/emr_evaluate_job.py` | EMR Serverless recovery entrypoint -- just `evaluate_main()`, for when training succeeded but the report-write step failed |
| `pytest` | `tests/*.py` | unit tests (mock `src.api.db`/`src.api.drift` rather than hitting real infra) |

## A typical request's call path, end to end

Useful for "where do I even put a breakpoint" when debugging a live `/predict` call:

```
uvicorn (ASGI server)
  -> src/api/app.py: log_requests middleware (reads request body up front, times the call)
    -> src/api/security.py: require_api_key (checks X-API-Key against API_KEYS/API_KEY)
    -> src/api/app.py: predict()
      -> _combine()           (title + text -> review_text, matching preprocess.py's training-time logic)
      -> _score()              (runs the loaded Spark model, builds {sentiment, probability} per text)
      -> src/api/db.py: insert_prediction()   (one row per text scored; no-op if APP_DB_HOST unset)
    <- response returned
  <- log_requests middleware finishes the one log entry for this request (success or failure, same code path)
```

And the drift-check path, triggered separately (manually, or by the cloud deployment's EventBridge rule — never by a `/predict` call itself):

```
GET /admin/drift-report
  -> src/api/app.py: drift_report()
    -> src/api/drift.py: run_drift_check()
      -> load_baseline()                 (reads models/active/BASELINE_STATS.json)
      -> src/api/db.py: query_recent_predictions()   (reads app.predictions, last window_seconds)
      -> compute_psi()  x2                (prediction confidence, review length)
    -> src/api/db.py: insert_drift_check()   (writes the result to app.drift_checks; no-op if APP_DB_HOST unset)
  <- the same result is both returned as the HTTP response and persisted for Grafana to read later
```
