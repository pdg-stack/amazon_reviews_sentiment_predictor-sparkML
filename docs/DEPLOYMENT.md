# Cloud Deployment Guide

This is a **plan**, not something already built — it describes how to take this project from "runs in a local dev container" to "runs in the cloud." It's written to apply to any provider (AWS, GCP, Azure, or a self-managed Kubernetes cluster) rather than committing to one. Where a specific service name is used, it's given as one example among several, not a requirement.

There are two separate deployment tracks, matching how the project is already split locally:

1. **Batch** — `preprocess.py` → `train.py` → `evaluate.py`, run on a schedule or on demand, on a managed Spark cluster.
2. **Real-time** — the FastAPI `/predict` service, run as a long-lived container behind a public address.

They don't have to ship together. You could deploy just the API against a model trained locally, or run training in the cloud while still serving locally. This guide covers both since most real setups eventually want both.

## First: make file paths and settings configurable

Right now, every path in `src/config.py` is worked out relative to this project's own folder on disk — correct for local development, but a cloud batch job doesn't have a local folder, and a deployed API container won't have this exact folder layout unless it's built into the image. Before deploying, each path in `config.py` needs to be **overridable through an environment variable, falling back to today's local default** if that variable isn't set:

```python
DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
```

This keeps local development exactly as it is today (nothing changes if no environment variables are set), while letting a cloud deployment point `DATA_RAW_DIR`/`DATA_PROCESSED_DIR` at cloud storage and `MLFLOW_TRACKING_URI` at a real tracking server instead of a local file. This is the one actual code change everything below assumes — it isn't done yet.

## Batch track: training and evaluation on a managed Spark cluster

### What has to change

- **Where the data lives**: `data/raw` and `data/processed` need to move to cloud storage (S3, GCS, or Azure's equivalent), since a cluster's worker machines don't share this container's disk. Spark can read and write those cloud paths directly — once the paths above are configurable, pointing them at a storage bucket is the only change `preprocess.py`/`train.py` need.
- **Where MLflow tracks runs**: the local file (`models/mlruns/mlflow.db`) doesn't work from a distributed cluster, since a plain file can't handle multiple machines writing to it at once. Point `MLFLOW_TRACKING_URI` at a real tracking server instead — either a small database you run yourself, or a managed option if your cloud platform offers one (Databricks, for example, has MLflow built in). The registry that `set_active_model.py` relies on works the same way against any of these; only the connection string changes.
- **Downloading the raw data**: `data_download.py` is a one-time or occasional step, not something that needs to re-run as part of every scheduled job. Simplest is to run it once, locally or as a one-off job, and leave the raw files in cloud storage from then on.

### Where to run it

The image already built by `docker/Dockerfile` is the same image a managed Spark platform needs, possibly with small adjustments for that platform's conventions. A few options that all fit the same shape:

- **A managed Spark platform** — Databricks, EMR on AWS, Dataproc on GCP, or Azure's equivalents — running `preprocess.py`/`train.py`/`evaluate.py` as a scheduled job, pointed at the cloud storage paths and tracking server above.
- **Spark on Kubernetes**, if that's already how you run things — the same image, via the Spark Operator or `spark-submit`, with the same environment variables.
- **Scheduling**: whatever replaces manually running `python src/run_pipeline.py` — the platform's own scheduler, or a tool like Airflow or GitHub Actions calling into it.

### What doesn't change

The pipeline itself — tokenize, remove stopwords, hash, weight, classify, tuned with cross-validation and logged to MLflow — works the same on a managed cluster as it does locally; nothing about its logic depends on where it runs. `set_active_model.py`'s deliberate, explicit activation step (see `ARCHITECTURE_NOTES.md`) also carries over unchanged — it's still a manual step, just now pointed at a shared registry instead of a local one.

### A stepping stone that already exists: the docker-compose Spark cluster

`docker-compose.yml` (see `docs/DOCKER_COMPOSE.md`) already runs a real, standalone Spark cluster locally, with the API and batch jobs submitting to it as clients rather than running their own private Spark sessions — the same basic shape a managed cloud platform uses. It isn't a cloud deployment, but it's a reasonable way to check that "my code submits to a cluster instead of running its own" actually works before committing to a specific provider.

## Real-time track: deploying the FastAPI service

### What has to change

- **The trained model**: `models/active/` needs to be available inside the running container. Two reasonable approaches:
  - **Build it into the image** — simplest, but means building and deploying a new image every time the served model version changes.
  - **Download it at startup** from cloud storage or directly from the MLflow registry (the same call `set_active_model.py` already makes) — this separates "deploy a new container" from "change which model version is being served," closer to how the `--reload-api` option already works locally.
- **Secrets**: `API_KEY` currently lives in a local file that isn't committed to the repo. In the cloud, use the platform's secret manager instead (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, or a Kubernetes Secret) and pass it in as an environment variable when the container starts — never build it into the image or commit it.
- **Networking**: expose port 8000 behind whatever load balancer or ingress the platform provides, with HTTPS enabled — most managed container hosts handle this automatically.

### Where to run it

Any container host works without changes, since the app is already just a Docker image exposing one HTTP port: ECS/Fargate or App Runner on AWS, Cloud Run on GCP, Container Apps on Azure, or plain Kubernetes. Pick whichever one you already use elsewhere — this app doesn't need anything platform-specific. `GET /health`, which already exists and needs no authentication, works directly as whatever health-check endpoint that host expects.

### A performance note carried over from `ARCHITECTURE_NOTES.md`

The app currently keeps one Spark session running per process, which is fine for the traffic this project has seen so far, but each running copy pays the cost of starting a JVM, and that overhead doesn't shrink just because the model itself is small. If traffic ever grows enough for this to matter, the next step (not built here) is exporting the trained model's underlying numbers — its learned weights and vocabulary — and scoring with a lighter, non-JVM process instead, keeping the same `/predict` contract and the same API-key check. Until then, deploying the current container as-is is a reasonable starting point, just not one built to handle high request volume.

### The logging and metrics setup carries over without code changes

`GET /metrics`, the `active_model_version` and `prediction_drift_psi` gauges, and the structured logs in `logs/api_requests.log` and `logs/api_probes.log` (see `docs/OBSERVABILITY.md`) all work unchanged in a cloud container — point whatever metrics scraper the platform uses at `/metrics`. The one thing that does need to move is where the logs end up: a local rotating file doesn't survive a container restart in most cloud setups, so the log destination should be redirected to the platform's own log collection (CloudWatch Logs, Cloud Logging, or similar) instead of a local file — not done here, for the same reason the `config.py` changes above aren't done yet.

### After it's deployed

- If you're using the Postman collection mentioned in `ARCHITECTURE_NOTES.md` to exercise the API, fill in its "Production" environment with the deployed URL and a separate API key provisioned for that environment — never reuse the local development key.
- `set_active_model.py --reload-api --api-url https://<deployed-url>` works against a deployed instance exactly as it does locally, once that instance's `API_KEY` is available to whoever runs the command.

## Summary checklist

- [ ] Make `config.py`'s paths and `MLFLOW_TRACKING_URI` overridable through environment variables (see the first section above) — the one real code change this whole plan assumes.
- [ ] Set up (or choose a managed) MLflow tracking server reachable from both the batch job and anyone running `set_active_model.py`/`evaluate.py`.
- [ ] Move the raw and processed data to cloud storage; point `data_download.py`'s output there, or run it once locally and upload the result.
- [ ] Package and schedule the batch job on a managed Spark platform.
- [ ] Decide whether the API's model gets built into the image or downloaded at startup.
- [ ] Move `API_KEY` (and any future secrets) into the platform's secret manager.
- [ ] Deploy the API container to a container host, behind HTTPS, with `GET /health` wired up as its health check.
- [ ] Update the Postman "Production" environment once the URL and its secret exist.
