# Cloud Deployment Guide

This is a **plan**, not yet implemented -- everything below describes how to take this project from "runs in a local devcontainer" to "runs in the cloud." It's deliberately provider-agnostic: every step is described in terms of what has to happen (build an image, push it somewhere, run it somewhere with these inputs), so it applies to AWS, GCP, Azure, or a self-managed Kubernetes cluster equally. Where a provider's specific service name is useful as an example, it's given as one option among several, not a requirement.

Two independent deployment tracks, matching how the project is already split locally:

1. **Batch** -- `preprocess.py` -> `train.py` -> `evaluate.py`, run on a schedule or on demand, on a managed Spark cluster.
2. **Real-time** -- the FastAPI `/predict` service, run as a long-lived container behind a public endpoint.

They don't have to ship together -- you could deploy just the API against a model trained locally, or run batch training in the cloud while still serving locally. This doc covers both since most real setups eventually want both.

## Prerequisite: make paths and config environment-overridable
Everything in `src/config.py` currently resolves paths relative to the project's own folder (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`) -- correct for local/devcontainer use, but a cloud batch job's storage isn't a local folder, and a deployed API container won't have this repo's exact directory layout unless you bake it in. Before deploying, `config.py` needs each path to be **overridable via an environment variable, falling back to today's local default**, e.g.:

```python
DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
```
This keeps local development unchanged (no env vars set = today's behavior) while letting cloud deployments point `DATA_RAW_DIR`/`DATA_PROCESSED_DIR` at object storage (see below) and `MLFLOW_TRACKING_URI` at a real tracking server instead of a local SQLite file. This is the one piece of actual code work everything below assumes -- not done yet, flagged here so it isn't a surprise mid-deployment.

## Batch track: training/evaluation on a managed Spark cluster

### What has to change
- **Data location**: `data/raw`/`data/processed` need to live in cloud object storage (S3 / GCS / ADLS) instead of the local filesystem, since a cluster's workers don't share this container's disk. Spark reads/writes `s3a://`, `gs://`, `abfss://` paths natively via `spark.read.csv(...)`/`.parquet(...)` -- once `DATA_RAW_DIR`/`DATA_PROCESSED_DIR` are overridable (above), pointing them at a bucket path is the only change `preprocess.py`/`train.py` need.
- **MLflow tracking**: the local SQLite file (`mlruns/mlflow.db`) doesn't work from a distributed cluster (SQLite has no concurrent-writer story across machines). Point `MLFLOW_TRACKING_URI` at a real tracking server instead -- either a small managed Postgres/MySQL + object-storage artifact root you stand up yourself (`mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...`), or a managed offering if your platform has one (e.g. Databricks' built-in MLflow). The Model Registry concept (`set_active_model.py` relies on `client.get_model_version`/`models:/name/N` URIs) works the same way against any of these -- only the connection string changes.
- **The Kaggle download step** (`data_download.py`) is a one-time/occasional step, not really part of the recurring cloud job -- simplest is to run it once (locally or in a one-off job) and land the raw CSVs in cloud storage, rather than re-running it as part of every scheduled job.

### Running it
The container image already built by `.devcontainer/Dockerfile` (Java 17 + PySpark + the project's Python deps) is the same image a managed Spark platform needs, possibly with minor adjustments for that platform's expected entrypoint/base image conventions. Concretely, any of these fit the same shape:
- **Managed Spark-as-a-platform** (Databricks on any cloud, EMR on AWS, Dataproc on GCP, HDInsight/Synapse on Azure): package `preprocess.py`/`train.py`/`evaluate.py` as a job the platform runs (a notebook, a `spark-submit`-style job, or a scheduled workflow step), pointed at the object-storage paths and tracking URI above.
- **Spark-on-Kubernetes** (if you're already running k8s): the same Docker image runs via the Spark Operator or `spark-submit --master k8s://...`, with the same env vars for storage/tracking.
- **Scheduling**: a cron-like trigger (the platform's own scheduler, or Airflow/Dagster/GitHub Actions calling into it) replaces manually running `python src/run_pipeline.py`.

### What doesn't change
The actual pipeline logic (`Tokenizer` -> `StopWordsRemover` -> `HashingTF` -> `IDF` -> `LogisticRegression`, `CrossValidator`, MLflow logging/registration) is unchanged -- `PipelineModel`'s save format and MLflow's APIs work identically on a managed cluster. `set_active_model.py`'s "explicit, on-demand activation" design (see `ARCHITECTURE.md`) also carries over unchanged -- it's still a deliberate step, just now pointed at a shared registry instead of a local one.

## Real-time track: deploying the FastAPI service

### What has to change
- **The model artifact**: `models/active/` needs to be available to the running container. Two reasonable approaches:
  - **Bake it into the image** at build time (simplest, but means a new image build+deploy every time the served version changes).
  - **Pull it at container startup** from cloud storage or directly from the MLflow registry (`mlflow.spark.load_model("models:/amazon_review_sentiment_predictor/N")`, same call `set_active_model.py` already makes) -- decouples "deploy a new container" from "change the served model version," closer to the local `--reload-api` workflow.
- **Secrets**: `API_KEY` currently lives in a local, gitignored `.env`. In the cloud, use the platform's secret manager (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, or a Kubernetes Secret) and inject it as an environment variable at container start -- never bake it into the image or commit it.
- **Networking**: expose port 8000 behind whatever the platform's standard ingress/load balancer is, with HTTPS termination (almost every managed container host does this for you -- ECS/Fargate + ALB, Cloud Run's built-in HTTPS, Container Apps' built-in ingress, or a k8s Ingress controller).

### Where to run it
Any container host works unchanged, since the app is already just a Docker image exposing one HTTP port -- ECS/Fargate or App Runner on AWS, Cloud Run on GCP, Container Apps on Azure, or plain Kubernetes. Pick based on what you already operate elsewhere, not because this app needs anything platform-specific. `GET /health` (unauthenticated, already built) maps directly onto whatever health-check convention that host expects.

### Performance caveat (carried over from `ARCHITECTURE.md`)
The current app keeps a resident `SparkSession` per process -- fine for the request volumes this project has seen so far, but each replica pays full JVM startup cost and Spark's per-request overhead doesn't shrink just because the model is small. If request volume grows enough for this to matter, the recommended next step (not built) is exporting the trained model's coefficients/vocabulary (e.g. via MLeap, or a hand-rolled scorer using the `LogisticRegressionModel`'s coefficients + the same hashing scheme) and serving with a lightweight, non-JVM process -- same `/predict` contract, same API-key model, much cheaper per replica. Until then, deploying the current container as-is is a reasonable starting point, just not one to scale to high RPS.

### After it's deployed
- Update the Postman "Production" environment's `baseUrl` (currently blank, see `ARCHITECTURE.md`) to the deployed URL, and its `apiKey` to whatever secret was provisioned for that environment -- never reuse the local dev key (see the earlier discussion on why per-environment secrets matter).
- `set_active_model.py --reload-api --api-url https://<deployed-url>` works against a deployed instance exactly as it does locally, once `API_KEY` is available to whoever runs that command.

## Summary checklist
- [ ] Make `config.py` paths/`MLFLOW_TRACKING_URI` environment-overridable (see Prerequisite above) -- the one real code change this plan assumes.
- [ ] Stand up (or pick a managed) MLflow tracking backend reachable from both the batch job and anyone running `set_active_model.py`/`evaluate.py` against it.
- [ ] Land raw/processed data in cloud object storage; point `data_download.py`'s output there (or run it once and upload manually).
- [ ] Package and schedule the batch job on a managed Spark platform of your choice.
- [ ] Decide bake-in vs. pull-at-startup for the API's model artifact.
- [ ] Move `API_KEY` (and any future secrets) into the platform's secret manager.
- [ ] Deploy the API container to any container host, behind HTTPS, with `GET /health` wired to its health check.
- [ ] Update the Postman "Production" environment once the URL and its secret exist.
