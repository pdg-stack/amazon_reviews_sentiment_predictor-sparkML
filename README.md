# Amazon Reviews Sentiment Predictor (Spark ML)

A containerized PySpark project that explores the [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) dataset, tunes and trains a sentiment-classification model (Spark MLlib, with k-fold cross-validation and hyperparameter search tracked in MLflow), evaluates it with charts, and serves it both as a batch job and a secured real-time API.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pieces fit together, [`docs/model_plan.md`](docs/model_plan.md) for why the modeling pipeline is built the way it is, [`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md) for how the model has evolved across versions, and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the (not-yet-built) cloud deployment plan.

The steps below use the VS Code Dev Container (one container, everything run by hand). For an alternative that splits FastAPI, the MLflow UI, a persistent Spark cluster, and batch jobs into independently-running services via `docker compose`, see [`docs/DOCKER_COMPOSE.md`](docs/DOCKER_COMPOSE.md) instead -- it's additive, not a replacement for the steps below.

**Data is never committed to this repo** — you download your own copy locally (see below).

## Prerequisites
- Docker Desktop
- VS Code with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)
- A Kaggle account (for the API token)

## 1. Get a Kaggle API token
Kaggle.com -> your profile -> **Settings** -> **API** -> **Create New Token**. The `kaggle` package (2.2.x+) used here authenticates from a single token string, not the older `kaggle.json` `{"username", "key"}` format. Save the token as plain text in `.kaggle/access_token` in this project (gitignored — never commit it):
```
printf '%s' "<your token>" > .kaggle/access_token
```

> Note: `kaggle login`-style OAuth credentials (a `credentials.json` with `refresh_token`/`access_token` *fields*, i.e. a JSON object) are a **different** mechanism and won't be picked up the same way — this project expects the plain-text single-token file above.

## 2. Open in a dev container
In VS Code: Command Palette -> **Dev Containers: Reopen in Container**. First run builds the image (Java 17 + PySpark + MLflow + FastAPI + friends); later runs reuse it.

## 3. Get the data and run the pipeline
```
python src/run_pipeline.py
```
This chains `data_download.py` -> `preprocess.py` -> `train.py` -> `evaluate.py`. You can also run each stage individually (see each script's `--help`); useful flags:
- `train.py --sample-fraction 0.05` — iterate quickly on a 5% sample instead of the full ~3.6M-row training set.
- `data_download.py`/`preprocess.py` automatically skip their work (no re-download, no reprocessing) if `data/raw/`/`data/processed/` are already populated — pass `--force` to redo them anyway.
- `run_pipeline.py --skip-download` — skip calling `data_download.py` entirely (on top of its own auto-skip above).

## 4. Explore interactively
- `notebooks/01_eda.ipynb` — class balance, review-length distribution, common terms (run after step 3's download+preprocess).
- `notebooks/02_pipeline_walkthrough.ipynb` — a cell-by-cell walkthrough of each pipeline stage (tokenize -> stopwords -> hash -> IDF), same content as `train.py`'s verbose printout, in notebook form.

Neither notebook is called automatically by any script — open and run them yourself whenever you want.

## 5. Compare versions, then pick one to serve
`train.py` never changes what's "live" — it only registers a new version in MLflow. Before deciding, score any specific version against the held-out test set directly (doesn't touch `models/active/`):
```
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db   # browse runs, metrics, and registered versions in a browser
python src/evaluate.py --version 1   # metrics + charts -> reports/version_1/
python src/evaluate.py --version 2   # metrics + charts -> reports/version_2/
```
Once you've decided, activate the version you want:
```
python src/set_active_model.py --version N
```
This populates `models/active/` (and `models/active/ACTIVE_MODEL.json`, recording which version + run is live). Nothing else updates this automatically — running this command is the only way the served model changes. **This alone does not restart or update an already-running API** — see step 7's `--reload-api` for a one-command way to do both together.

## 6. Batch predictions
```
python src/predict.py --input <path-to-csv-or-parquet> --output predictions.csv
```
The model expects a `review_text` field (title + body combined). If your input already has one (e.g. `data/processed/*.parquet`) it's used as-is; otherwise pass `--text-col`/`--title-col` to point at your columns and it's built the same way training built it:
```
python src/predict.py --input reviews.csv --text-col body --title-col headline --output out.csv
```

## 7. Real-time API
Generate your local API key once (prints it and writes it to a gitignored `.env`):
```
python scripts/generate_api_key.py
```
Start the API:
```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Call it (optionally include a `title`, combined with `text` the same way training combined them):
```
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: <the key printed above>" \
  -H "Content-Type: application/json" \
  -d '{"title": "Best purchase ever", "text": "This product is amazing, I love it!"}'
# -> {"sentiment": "positive", "probability": 0.93}
```
`GET /health` needs no key. Every `/predict` call is logged to `logs/api_requests.log`.

**Switching the model version on a running API**: `set_active_model.py` alone does not restart or update an already-running API (see step 5). To do both in one command:
```
python src/set_active_model.py --version N --reload-api
```
This activates version `N` and then calls the API's `POST /admin/reload-model` (same `X-API-Key` auth) so it starts serving it immediately — no restart needed. If the API isn't running, activation still succeeds and it just prints a note to start/restart it manually.

A Postman collection ("Amazon Reviews Sentiment Predictor API", workspace "PDG's Workspace") documents and exercises both endpoints, with separate "Local"/"Production" environments.

## Notes
- If full-dataset runs (3.6M rows) feel slow, increase Docker Desktop's memory allocation (Settings -> Resources).
- Switching the API's model version always requires an explicit `set_active_model.py` run — this is intentional (see `ARCHITECTURE.md`). Add `--reload-api` to also update a running API immediately instead of needing a separate restart.
