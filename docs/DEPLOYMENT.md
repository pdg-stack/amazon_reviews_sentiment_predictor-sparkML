# Cloud Deployment Guide

This is a **plan**, not something already built — it describes how to take this project from "runs in a local dev container" to "runs in the cloud." Most of the steps below are written to apply to any provider (AWS, GCP, Azure, or a self-managed Kubernetes cluster) rather than committing to one. The section right below, though, picks a concrete direction — a comparison between AWS and Modal.com, since training on a local machine has become slow enough (and disruptive enough to everything else running on it) that moving it off-laptop is now worth doing.

There are two separate deployment tracks, matching how the project is already split locally:

1. **Batch** — `preprocess.py` → `train.py` → `evaluate.py`, run on a schedule or on demand, on a managed Spark cluster.
2. **Real-time** — the FastAPI `/predict` service, run as a long-lived container behind a public address.

They don't have to ship together. You could deploy just the API against a model trained locally, or run training in the cloud while still serving locally. This guide covers both since most real setups eventually want both.

## Choosing a provider: AWS vs Modal.com

Three separate things need a home, and they don't all want the same kind of platform:

1. **Training** — the heaviest, most resource-hungry piece, and the direct reason this comparison exists.
2. **The always-on stack** — the FastAPI server, MLflow UI, and the GoAccess log dashboard (everything `docker-compose.yml` runs today).
3. **Large batch scoring** — the ~3.5M-row batch inference run being planned, which comes in two different shapes that matter for this choice (explained below).

### Training: AWS fits this project's actual architecture; Modal would mean giving up the real Spark cluster

The one fact that decides most of this comparison: **Modal's multi-node clustering requires GPUs and explicitly does not support CPU-only clustered functions.** That's stated directly in Modal's own docs, not an assumption — clustered functions on Modal need "full GPU utilization per node," and the feature is built around PyTorch-style distributed training (`torchrun`), not Spark. There is no way to stand up something equivalent to this project's `spark-master`/`spark-worker` pair on Modal.

That leaves two real options:

- **AWS EMR Serverless** — a managed, pay-per-second Spark runtime. `train.py`/`preprocess.py`/`evaluate.py` run close to unchanged (the pipeline logic doesn't care where Spark executes), and it genuinely preserves the "real distributed cluster" property this project's `docker-compose.yml` was specifically built to demonstrate. Priced per vCPU-second and GB-second actually used (roughly $0.053/vCPU-hour + $0.0058/GB-hour in US East as of this writing), with no charge while nothing is running — no cluster to leave on overnight, unlike a standing EMR-on-EC2 cluster.
- **Modal, running Spark in embedded `local[*]` mode inside one large container** — this drops the "distributed cluster" architecture entirely and falls back to exactly the same single-process Spark mode the plain `.devcontainer` workflow already uses locally. That's not necessarily a bad trade — at this project's data volumes (a 5%–10% sample, or even the full ~4M rows), a single sufficiently large machine handles it fine, and it's operationally simpler (one function, no master/worker to keep in sync). Priced per physical-core-second (roughly $0.047/physical-core-hour before regional multipliers, so meaningfully cheaper per core than EMR Serverless), with $30/month in free credits on the Starter plan.

At this project's actual scale, **cost is a non-factor either way** — a training run shaped like the one just done (a couple of vCPUs for well under an hour) costs pennies on both platforms. The real decision is architectural: EMR Serverless keeps training as a genuine Spark cluster job; Modal keeps the code path most similar to what already runs in the dev container, but as a single powerful machine rather than a cluster.

**Recommendation**: EMR Serverless, specifically because it's the smallest possible change from what's already built and tested (the same `docker/Dockerfile`, the same `train.py`, still a real Spark job) — the whole point of moving training off the laptop is to stop babysitting it, and reusing the existing Spark code path with no rewrite is the fastest way to get there.

### The always-on stack (API, MLflow UI, log dashboard): AWS, not close

Modal's unit of deployment is an individual serverless function or web endpoint — there's no equivalent to `docker-compose.yml`'s internally-networked, multi-service stack (`spark://spark-master:7077`-style service discovery, shared bind mounts, several long-lived named containers talking to each other). Hosting `mlflow-ui`, `log-viewer`, and `api` as separate Modal apps would mean re-architecting how they find and talk to each other, not a lift-and-shift.

AWS has two options that both fit without a rewrite:

- **Simplest**: one EC2 instance running `docker-compose.yml` almost exactly as it runs locally today — the least amount of new work, at the cost of managing a VM yourself (patching, restarts).
- **More production-grade**: ECS/Fargate, one service per current `docker-compose` service — managed restarts and scaling, no OS to patch, and cost that tracks actual usage per service instead of one always-on VM.

Either way, this piece belongs on AWS.

### Batch inference at ~3.5M rows: depends which kind of "batch" this is

Two different things could be meant here, and they favor different tools:

- **Bulk scoring via Spark directly** — the same shape `predict.py` already uses (`model.transform()` over a whole DataFrame in one job). Spark already parallelizes this across the dataset natively in a single job; EMR Serverless (the same platform training would already use) handles this well with no new architecture needed. Modal adds nothing here — there's no per-row work to fan out, it's one Spark job either way.
- **Replaying rows through the live `/predict` API** — many independent HTTP calls, one per row (or small batch), potentially paced to look like realistic traffic (this was the design discussed earlier this session for feeding the drift checker a larger sample). This *is* Modal's strongest case: fanning out hundreds of small, short-lived functions that each just call the hosted API is exactly the "scale up for a burst, back to zero after" pattern Modal is built for, and it would genuinely be faster than a single sequential client working through 3.5M calls one at a time.

If it's the second shape — the API server itself should still live on AWS (per the section above); Modal's role would just be the bursty *client* driving traffic against it, not a replacement for where the server lives.

### Summary

| Piece | Recommendation | Why |
|---|---|---|
| Training | AWS (EMR Serverless) | Modal can't run a CPU-only distributed Spark cluster at all; EMR Serverless is the smallest change from what already exists |
| API + MLflow UI + log dashboard | AWS (EC2 or ECS/Fargate) | Modal has no equivalent to a persistent, internally-networked multi-service stack |
| Bulk Spark-native scoring | AWS (EMR Serverless) | Same reasoning as training — Spark already parallelizes this in one job |
| Batch scoring via repeated API calls | Modal (calling an AWS-hosted API) | Modal's serverless fan-out is built exactly for many independent, bursty calls |

Net: AWS is the primary platform for everything that needs to keep running or that's still fundamentally a Spark job; Modal's only clear role is as an optional, bursty *client* for hitting the API at scale — not a place to host the stack or the cluster itself.

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

The image already built by `docker/Dockerfile` is the same image a managed Spark platform needs, possibly with small adjustments for that platform's conventions. A few options that all fit the same shape — see "Choosing a provider" above for why **EMR Serverless** is the recommended one:

- **A managed Spark platform** — EMR Serverless on AWS (recommended, see above), Databricks, Dataproc on GCP, or Azure's equivalents — running `preprocess.py`/`train.py`/`evaluate.py` as a scheduled job, pointed at the cloud storage paths and tracking server above.
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

Any container host works without changes, since the app is already just a Docker image exposing one HTTP port: ECS/Fargate or App Runner on AWS (recommended — see "Choosing a provider" above, same reasoning covers `mlflow-ui` and `log-viewer` too), Cloud Run on GCP, Container Apps on Azure, or plain Kubernetes. `GET /health`, which already exists and needs no authentication, works directly as whatever health-check endpoint that host expects.

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
