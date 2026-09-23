# Architecture

See [`diagrams/architecture-diagram.html`](diagrams/architecture-diagram.html) for an interactive diagram covering everything below.

## Overview

A containerized PySpark project: explore Amazon review data, tune and train a sentiment-classification model, evaluate it, and serve it both as a batch job and a secured real-time API. Everything runs inside one Docker image — no Java, Spark, or project-specific Python setup touches the host machine.

## Container layout

`docker/Dockerfile` builds an image with OpenJDK 17 (Spark needs a Java runtime to work) and the full Python toolchain: PySpark, pandas/matplotlib/seaborn for exploration and charts, MLflow for experiment tracking, FastAPI for serving, the Kaggle client for downloading data, and Jupyter for notebooks. VS Code's Dev Containers extension builds and connects to this image, so editing and debugging feel normal while everything actually runs inside the container.

Ports forwarded: **4040** (Spark's UI, while a job is running), **8000** (the API — also serves `/metrics`), **5000** (the MLflow UI).

### An alternative: split services via Docker Compose

`docker-compose.yml` (repo root) is an additional way to run this project alongside the single-container setup above — it splits the API, the MLflow UI, a persistent Spark cluster, and on-demand training jobs into services that each run independently, all built from the same Dockerfile so there's still just one dependency set. The regular dev-container workflow above is completely unaffected. See [`docs/DOCKER_COMPOSE.md`](DOCKER_COMPOSE.md) for the full setup, the difference between the two things both called "Spark UI," and how to run it.

## Data flow

```
Kaggle (kritanjalijain/amazon-reviews)
        |  src/data_download.py  (needs a Kaggle API token;
        |                         skips automatically if data is already downloaded)
        v
   data/raw/*.csv                     (not committed to the repo)
        |  src/preprocess.py  (Spark: cleans text, fixes types, combines
        |                      title + body into review_text; skips
        |                      automatically if already processed)
        v
   data/processed/*.parquet           (not committed; has review_text = title + body)
        |
        +--> src/notebooks/01_eda.ipynb              (exploration + charts, run by hand)
        +--> src/notebooks/02_pipeline_walkthrough.ipynb  (a stage-by-stage walkthrough, run by hand)
        |
        |  src/train.py
        |    - 80/20 train/validation split
        |    - Optuna hyperparameter search over the pipeline
        |    - logs every run to MLflow, registers a new model version
        v
   MLflow registry ("amazon_review_sentiment_predictor", versioned)
        |  src/set_active_model.py --version N   (explicit, on demand only)
        v
   models/active/  (+ ACTIVE_MODEL.json)          (not committed)
        |
        +--> src/predict.py        (batch scoring from the command line)
        +--> src/api/app.py        (the FastAPI /predict, /health endpoints)
        |
        |  src/evaluate.py [--version N]  (scores data/processed/test.parquet
        |                                  against models/active/, or any
        |                                  specific registered version directly --
        |                                  useful for comparing candidates
        |                                  before activating one)
        v
   docs/reports/*.png  or  docs/reports/version_<N>/*.png  (confusion matrix, ROC curve, tuning chart)   (not committed)
```

## Why promoting a model is a separate, explicit step

`train.py` only ever adds a new *version* to the registry — it never changes what's actually being served. `models/active/`, the only place `predict.py` and the API read from, only changes when someone explicitly runs `set_active_model.py --version N`. That means switching which model is live is always a deliberate, recorded action (written to `ACTIVE_MODEL.json`), never something that happens automatically as a side effect of training. `evaluate.py --version N` lets you score any registered version against the held-out test set without touching `models/active/` at all, so different candidates can be compared before deciding which one to activate.

Activating a version doesn't, by itself, affect an API that's already running — that's also a separate, explicit step (`set_active_model.py --version N --reload-api`, covered below).

## Three ways to use the model

1. **Batch** (`src/predict.py`) — a command-line tool that scores a whole file of reviews in one pass. The natural choice for "predict sentiment across a dataset."
2. **Notebooks** — interactive exploration, run by hand, not part of any automated pipeline.
3. **Real-time API** (`src/api/app.py`) — a FastAPI service that keeps one Spark session and one loaded model ready, answering requests one (or a few) at a time over HTTP.

## How the API is put together

- `POST /predict` requires an `X-API-Key` header, checked against a key stored in a local `.env` file (generate one with `src/scripts/generate_api_key.py`). It accepts `text`/`texts` and an optional `title`/`titles`, combined into `review_text` the same way `preprocess.py` does, so a request sees exactly the kind of input the model was trained on. `GET /health` needs no key — it's a basic liveness check.
- `POST /admin/reload-model` (also key-protected) re-reads `models/active/` into the running process without restarting it. `set_active_model.py --reload-api` calls this automatically right after activating a version, so "pick a version" and "make the running API actually serve it" can be one command instead of two — without it, the API keeps serving whatever it loaded at startup until it's restarted.
- `GET /admin/drift-report` (also key-protected) returns the latest result from the background drift check — see Observability below.
- Every request the API handles is logged; see Observability below and `docs/OBSERVABILITY.md` for the full schema.
- Keeping one long-running Spark session in the process is fine for local or demo use, but it isn't how you'd serve high traffic in production. A production deployment would more likely export the trained model's weights and score outside the Java runtime entirely (for example with MLeap or ONNX) rather than keep a full Spark session alive per running copy — that's a future direction, not something built here.

## Observability

The API has three observability pieces on top of everything above — the full design, schema, and caveats are in [`OBSERVABILITY.md`](OBSERVABILITY.md):

- **Logging**: every request is logged as one JSON line via Python's standard `logging` module (`src/api/logging_config.py`), including failed requests (400/401/500), which used to go unrecorded. Health checks and metrics scrapes are logged separately from real traffic so they don't crowd it out.
- **Metrics**: `GET /metrics` (Prometheus's plain-text format, via `prometheus-fastapi-instrumentator`) exposes request counts and response-time histograms, plus two custom numbers: `active_model_version` and `prediction_drift_psi`.
- **Model drift**: `src/evaluate.py` captures a baseline — prediction confidence and review length — from the held-out test set; `set_active_model.py` copies it into `models/active/BASELINE_STATS.json` when a version is activated; a background check in the API (`src/api/drift.py`) compares recent live traffic against that baseline every few minutes, using a statistic called the Population Stability Index, and surfaces the result through the metrics above and `GET /admin/drift-report`. This catches *shifts* in the kind of input and output the model is seeing, not whether individual predictions are *correct* — there's no way to check that without labeled real-world data.

## Experiment tracking and the model registry (MLflow)

`models/mlruns/mlflow.db` — a local file, not committed to the repo, kept inside `models/mlruns/` rather than at the project root — holds the metadata for every `train.py` run: the hyperparameter grid that was tried, per-fold results, the winning configuration, and the registered model version. The rest of `models/mlruns/` holds the actual model files. A simpler file-based setup can't support the model registry that `set_active_model.py` depends on, which is why this local database is used instead. `mlflow ui --backend-store-uri sqlite:///models/mlruns/mlflow.db` (port 5000) browses this history, and the same registry is what `set_active_model.py` relies on for versioning.

## API documentation and testing

A Postman collection ("Amazon Reviews Sentiment Predictor API") documents and exercises `/health` and `/predict`, with separate "Local" and "Production" environments — the latter's URL gets filled in once a cloud deployment exists, so pointing the collection at a deployed instance is a one-click environment swap, not a collection edit.

## Path to a cloud deployment (a plan, not built yet)

- **Batch**: the same Spark job — `preprocess.py` → `train.py` → `evaluate.py` — can run unchanged on a managed Spark platform (Databricks, EMR, Dataproc), since the trained pipeline's saved format is portable.
- **Real-time**: the current FastAPI container could run as-is on any container host, but for handling meaningfully more traffic, the recommended next step is exporting the model's coefficients and vocabulary and serving them with a lighter, non-Java scorer, while keeping the same `/predict` contract and API-key check described above.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full, step-by-step plan.
