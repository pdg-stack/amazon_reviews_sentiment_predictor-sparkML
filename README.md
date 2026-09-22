# Amazon Reviews Sentiment Predictor (Spark ML)

A containerized PySpark project that explores the [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) dataset, tunes and trains a sentiment-classification model (using Spark's machine-learning library, with hyperparameter search via Optuna tracked in MLflow), evaluates it with charts, and serves it both as a batch job and a secured real-time API.

See [`docs/ARCHITECTURE_NOTES.md`](docs/ARCHITECTURE_NOTES.md) for how the pieces fit together (or [`docs/diagrams/architecture-diagram.html`](docs/diagrams/architecture-diagram.html) for an interactive diagram of the same thing), [`docs/model_plan.md`](docs/model_plan.md) for why the modeling pipeline is built the way it is, [`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md) for how the model has changed across versions, [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) for the API's logging, metrics, and drift-monitoring design, and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the (not yet built) cloud deployment plan.

The steps below use VS Code's Dev Container feature — one container, everything run by hand. For an alternative that splits the API, the MLflow UI, a persistent Spark cluster, and batch jobs into separately-running services with `docker compose`, see [`docs/DOCKER_COMPOSE.md`](docs/DOCKER_COMPOSE.md) (or [`docs/diagrams/docker-compose-diagram.html`](docs/diagrams/docker-compose-diagram.html) for a diagram of that setup) instead — it's an addition, not a replacement for the steps below.

**No data is ever committed to this repo** — you download your own copy locally (see below).

## Prerequisites

- Docker Desktop
- VS Code with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)
- A Kaggle account (for the API token)

## 1. Get a Kaggle API token

On Kaggle.com: your profile → **Settings** → **API** → **Create New Token**. The version of the `kaggle` package used here (2.2.x and up) authenticates using a single token string, not the older `kaggle.json` format. Save the token as plain text in `.kaggle/access_token` in this project (this file isn't committed):

```
printf '%s' "<your token>" > .kaggle/access_token
```

> Note: if you've used `kaggle login` before, that produces a different kind of credentials file (`credentials.json`, a JSON object with `refresh_token`/`access_token` fields) — this project expects the plain-text single-token file described above instead.

## 2. Open in a dev container

In VS Code: Command Palette → **Dev Containers: Reopen in Container**. The first run builds the image (Java, PySpark, MLflow, FastAPI, and everything else); later runs reuse it.

## 3. Get the data and run the pipeline

```
python src/run_pipeline.py
```

This runs, in order: `data_download.py` → `preprocess.py` → `train.py` → `evaluate.py`. You can also run each stage on its own (see each script's `--help`); a few useful flags:

- `train.py --sample-fraction 0.05` — train on a 5% sample instead of the full dataset (about 3.6 million rows), useful for iterating quickly.
- `data_download.py` and `preprocess.py` each skip their work automatically if the relevant `data/` folder is already populated — pass `--force` to redo them anyway.
- `run_pipeline.py --skip-download` — skip the download step entirely, on top of its own automatic skip above.

## 4. Explore interactively

- `src/notebooks/01_eda.ipynb` — class balance, review-length distribution, common words (run this after step 3 has downloaded and processed the data).
- `src/notebooks/02_pipeline_walkthrough.ipynb` — a cell-by-cell walkthrough of each pipeline stage (tokenize → remove stopwords → hash → weight), the same steps `train.py` runs, shown one at a time.

Neither notebook runs automatically — open and run them yourself whenever you'd like.

## 5. Compare versions, then pick one to serve

`train.py` never changes what's actually "live" — it only registers a new version in MLflow. Before deciding which one to use, score any specific version against the held-out test set directly (this doesn't touch `models/active/`):

```
mlflow ui --port 5000 --backend-store-uri sqlite:///models/mlruns/mlflow.db   # browse runs, metrics, and versions in a browser
python src/evaluate.py --version 1   # metrics + charts -> docs/reports/version_1/
python src/evaluate.py --version 2   # metrics + charts -> docs/reports/version_2/
```

Once you've decided, activate the version you want:

```
python src/set_active_model.py --version N
```

This updates `models/active/` (and `models/active/ACTIVE_MODEL.json`, which records which version and run is live). Nothing else does this automatically — running this command is the only way the served model changes. **On its own, this doesn't restart or update an API that's already running** — see step 7's `--reload-api` flag for a way to do both in one command.

## 6. Batch predictions

```
python src/predict.py --input <path-to-csv-or-parquet> --output predictions.csv
```

The model expects a `review_text` field (title and body combined). If your input file already has one — like `data/processed/*.parquet` does — it's used as-is; otherwise, pass `--text-col`/`--title-col` to point at your own columns, and they'll be combined the same way training combined them:

```
python src/predict.py --input reviews.csv --text-col body --title-col headline --output out.csv
```

## 7. Real-time API

Generate your local API key once (this prints it and writes it to a local, uncommitted `.env` file):

```
python src/scripts/generate_api_key.py
```

Start the API:

```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Call it (a `title` is optional, and gets combined with `text` the same way training combined them):

```
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: <the key printed above>" \
  -H "Content-Type: application/json" \
  -d '{"title": "Best purchase ever", "text": "This product is amazing, I love it!"}'
# -> {"sentiment": "positive", "probability": 0.93}
```

`GET /health` needs no key. Every request the API handles is logged — real traffic to `logs/api_requests.log`, health checks and metrics scrapes to their own file, `logs/api_probes.log`. `GET /metrics` (Prometheus format, no key needed) and `GET /admin/drift-report` (needs the key) are also available — see step 8.

**Switching the model version on a running API**: as noted in step 5, `set_active_model.py` on its own doesn't restart or update an API that's already running. To do both in one command:

```
python src/set_active_model.py --version N --reload-api
```

This activates version `N` and then calls the API's `POST /admin/reload-model` endpoint (using the same key) so it starts serving that version right away — no restart needed. If the API isn't running, activation still succeeds, and the command just prints a note to start or restart it yourself.

A Postman collection ("Amazon Reviews Sentiment Predictor API") documents and exercises both endpoints, with separate "Local" and "Production" environments.

## 8. Observability

`GET /metrics` (no key needed) exposes Prometheus-format request counters and response-time histograms, plus `active_model_version` and `prediction_drift_psi` gauges. `GET /admin/drift-report` (needs the key) returns the latest result from a background check that compares recent `/predict` traffic against a baseline captured during evaluation:

```
python src/evaluate.py --version N        # also writes docs/reports/version_N/baseline_stats.json
python src/set_active_model.py --version N  # copies it to models/active/BASELINE_STATS.json
```

For a live view of the logs, or a dashboard with charts, see the "Browsing the logs" section in [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) (full design, log format, and drift thresholds are there too).

## Notes

- If running on the full 3.6-million-row dataset feels slow, try increasing Docker Desktop's memory allocation (Settings → Resources).
- Switching the API's model version always requires explicitly running `set_active_model.py` — this is intentional (see `docs/ARCHITECTURE_NOTES.md`). Add `--reload-api` to also update a running API immediately, instead of restarting it separately.
