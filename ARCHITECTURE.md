# Architecture

## Overview
A containerized PySpark project: explore Amazon review data, tune and train a sentiment-classification model, evaluate it, and serve it both as a batch job and a secured real-time API. Everything runs inside one Docker image — no Java, Spark, or project-specific Python environment touches the host machine.

## Container layout
`.devcontainer/Dockerfile` builds a `python:3.11-slim` image with OpenJDK 17 (Spark needs a JVM) and the full Python toolchain: PySpark, pandas/matplotlib/seaborn (EDA + charts), MLflow (tracking/registry), FastAPI + uvicorn (serving), kaggle (data download), Jupyter (notebooks). VS Code's Dev Containers extension builds and attaches to this image, giving a normal editing/debugging experience with everything executing inside the container.

Ports forwarded: **4040** (Spark UI, while a job is running), **8000** (FastAPI), **5000** (MLflow UI).

## Data flow
```
Kaggle (kritanjalijain/amazon-reviews)
        |  src/data_download.py  (Kaggle API, needs .kaggle/access_token;
        |                         auto-skips if data/raw or data/processed already populated)
        v
   data/raw/*.csv                     (gitignored)
        |  src/preprocess.py  (Spark: clean text, cast types, combine
        |                      title+body into review_text; auto-skips if
        |                      data/processed already populated)
        v
   data/processed/*.parquet           (gitignored; has review_text = title + body)
        |
        +--> notebooks/01_eda.ipynb              (human-run EDA + charts)
        +--> notebooks/02_pipeline_walkthrough.ipynb  (human-run, stage-by-stage transformer demo)
        |
        |  src/train.py
        |    - 80/20 train/validation split
        |    - CrossValidator + ParamGridBuilder over the Spark ML Pipeline
        |    - logs every run to MLflow, registers a new model version
        v
   MLflow registry ("amazon_review_sentiment_predictor", versioned)
        |  src/set_active_model.py --version N   (EXPLICIT, ON-DEMAND ONLY)
        v
   models/active/  (+ ACTIVE_MODEL.json)          (gitignored)
        |
        +--> src/predict.py        (batch CLI inference)
        +--> src/api/app.py        (FastAPI /predict, /health)
        |
        |  src/evaluate.py [--version N]  (scores data/processed/test.parquet
        |                                  against models/active/, or any
        |                                  specific registered version directly --
        |                                  the latter for comparing candidates
        |                                  before activating one)
        v
   reports/*.png  or  reports/version_<N>/*.png  (confusion matrix, ROC curve, tuning-improvement chart)   (gitignored)
```

## Why the model-serving path is gated behind an explicit step
`train.py` only ever adds a new *version* to the MLflow registry — it never changes what's live. `models/active/` (the only place `predict.py` and the API load from) is only ever updated by someone explicitly running `set_active_model.py --version N`. This means promoting a newly trained model to "the one being served" is always a deliberate, auditable action (recorded in `ACTIVE_MODEL.json`), never an automatic side effect of training. `evaluate.py --version N` lets you score any registered version against the held-out test set *without* touching `models/active/` at all, so you can compare candidates before deciding which one to activate.

By the same logic, activating a version does not, by itself, touch a currently-running API process — that's also an explicit step, via `set_active_model.py --version N --reload-api` (see API design below).

## Consuming the model — three ways
1. **Batch** (`src/predict.py`) — CLI entrypoint, scores a CSV/parquet file of reviews in one pass. This is the natural fit for "predict sentiment on a test dataset."
2. **Notebooks** — interactive, human-run exploration; not part of any automated pipeline.
3. **Real-time API** (`src/api/app.py`) — a FastAPI service that keeps one Spark session and one loaded model resident, and answers single-review (or small-batch) requests synchronously over HTTP.

## API design
- `POST /predict` requires an `X-API-Key` header (checked against the `API_KEY` env var in `.env`, generated on demand by `scripts/generate_api_key.py`). Accepts `text`/`texts` and an optional `title`/`titles`, combined into `review_text` the same way `preprocess.py` does, so a request sees exactly what the model was trained on. `GET /health` is unauthenticated, for basic liveness checks.
- `POST /admin/reload-model` (also `X-API-Key`-protected) re-reads `models/active/` into the running process without a restart. `set_active_model.py --reload-api` calls this automatically right after activating a version, so "pick a version" and "make the running API serve it" can be one command — without it, the API keeps serving whatever it loaded at startup until it's restarted.
- Every `/predict` hit is logged (timestamp, input, prediction, status, latency) to `logs/api_requests.log` as JSON lines — an audit trail, and a foundation for any future "compare predictions to real outcomes" analysis.
- A long-lived in-process Spark session is fine for local/demo latency; it is **not** a high-throughput production serving pattern. A production deployment would more likely export the fitted model's weights/vocabulary and score outside the JVM (e.g. via MLeap or ONNX) rather than keep a full Spark session alive per replica — noted here as a future direction, not built in this pass.

## Experiment tracking & model registry (MLflow)
`mlflow.db` (a local SQLite file, gitignored) holds run/metric/registry metadata for every `train.py` run -- hyperparameter grid, per-fold metrics, the selected best configuration, and the registered model version -- while `mlruns/` (also gitignored) holds the actual model artifacts. A plain file-store tracking URI can't back the Model Registry that `set_active_model.py` depends on, which is why SQLite is used instead of the simpler `file:./mlruns` approach. `mlflow ui --backend-store-uri sqlite:///mlflow.db` (port 5000) browses this history. The registry is also the versioning mechanism behind `set_active_model.py`.

## API documentation & testing (Postman)
A Postman collection ("Amazon Reviews Sentiment Predictor API") in workspace "PDG's Workspace" documents and exercises `/health` and `/predict`, with "Local" and "Production" environments (the latter's `baseUrl` filled in once a cloud deployment exists) so hitting a future deployed instance is a one-click environment swap, not a collection edit.

## Path to a cloud deployment (future work, not built now)
- **Batch**: the same Spark job (`preprocess.py` -> `train.py` -> `evaluate.py`) can run unchanged on a managed Spark cluster (Databricks, EMR, Dataproc) — `PipelineModel`'s save format is portable.
- **Real-time**: the current FastAPI container could run as-is on any container host (Cloud Run, ECS, etc.), but for meaningful request-per-second scaling, the recommended next step is exporting the trained model's coefficients/vocabulary and serving with a lighter, non-JVM scorer, keeping the same `/predict` contract and API-key security model documented above.
