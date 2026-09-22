# Docker Compose: split-service deployment

See [`diagrams/docker-compose-diagram.html`](diagrams/docker-compose-diagram.html) for an interactive diagram of this setup.

`docker-compose.yml` (repo root) is an **alternative** way to run this project, alongside the `.devcontainer` workflow described in the main [`README.md`](../README.md) and [`ARCHITECTURE_NOTES.md`](ARCHITECTURE_NOTES.md) — not a replacement for it. The devcontainer keeps working exactly as before: one container, Spark running in its simple built-in ("embedded") mode, and you run whichever script you want by hand. Nothing here requires the devcontainer to be touched or even set up.

What compose adds: genuinely separate services that each run independently, instead of one container doing everything — backed by a real, standalone Spark cluster instead of each process starting its own throwaway Spark session.

## Why a real Spark cluster

For the Spark UI to be its own independent, always-there service — rather than "whichever container happens to be running a job right now" — there needs to be an actual Spark cluster process (`spark-master`) that exists whether or not anything is currently running on it. `docker-compose.yml` starts `spark-master` and `spark-worker`, and the API and pipeline containers become Spark *clients* that submit work to `spark://spark-master:7077`, instead of each one running its own private Spark session.

**The Spark version on the client and on the cluster has to match exactly**, or jobs fail with serialization errors between the two. Rather than pull in a second Spark distribution that could drift out of sync, `spark-master` and `spark-worker` are built from the **exact same Dockerfile** as every other service — the same pinned Spark version everywhere, one dependency set, no risk of mismatch.

### A gotcha: the pip-installed Spark package doesn't include the usual startup scripts

`pip install pyspark` bundles the full Spark distribution, but the version installed this way is missing the `start-master.sh` / `start-worker.sh` scripts that normally start a Spark cluster (confirmed by checking inside this project's own image — they're just not there). `docker-compose.yml` starts the master and worker directly through a lower-level command instead (`spark-class`), which is what those missing scripts would have called anyway:

```
spark-class org.apache.spark.deploy.master.Master --host 0.0.0.0 --port 7077 --webui-port 8080
spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077 --webui-port 8081
```

This also runs in the foreground, which is what a container needs as its main process (the missing scripts would have run in the background and exited immediately — the wrong shape for a container).

## Two different things are both called "Spark UI" — worth telling apart

- **Master UI** (`spark-master`, port **8080**): always there at `http://localhost:8080` whenever `spark-master` is running, whether or not a job is active. This is the one that's genuinely a persistent, independent service — it shows registered workers, running and completed jobs, and how much of the cluster's resources are in use.
- **Driver UI** (port **4040**): only exists while a specific job's driver process is alive — either the API's long-running Spark session, or a one-off `pipeline` job. It's tied to that process and disappears when it exits, which is just how Spark works. It's exposed on the `api` service (and reachable on `pipeline` with `--service-ports`, see below) so you can look at it while something is running, but it isn't a standing service the way the master UI is.

With the cluster in place, the actual work happens on the `spark-worker` container, not wherever the driver happens to be running — the master UI's list of workers and applications is how you can confirm that's really happening, rather than just trusting the configuration.

## Services

| Service | Purpose | Ports | Always on? |
|---|---|---|---|
| `spark-master` | The persistent Spark cluster master, plus its Master UI | 8080 (UI), 7077 (cluster) | yes |
| `spark-worker` | Actually executes the tasks submitted to the cluster | 8081 (worker UI, random host port) | yes; scale with `--scale spark-worker=N` |
| `mlflow-ui` | Browses `models/mlruns/` — run history, metrics, registered model versions | 5000 | yes |
| `api` | FastAPI `/predict`, `/health`, `/admin/reload-model` — a Spark client | 8000 (API), 4040 (driver UI, while a request runs) | yes |
| `log-viewer` | Web dashboard over `logs/api_requests.log` (GoAccess) | 8081 | yes |
| `pipeline` | Runs `train.py`, `evaluate.py`, `preprocess.py`, `data_download.py`, `set_active_model.py` — a Spark client | none by default | **no** — only starts when you explicitly ask for it |

All services except `log-viewer` build from the same `docker/Dockerfile`, so there's exactly one pinned dependency set behind all of them. `log-viewer` uses its own small image (`docker/log-viewer.Dockerfile`) since GoAccess has nothing to do with the rest of the project's dependencies.

### Shared files (bind mounts, not named volumes)

Consistent with how this project already works — see `ARCHITECTURE_NOTES.md` — everything is mounted in from the host rather than baked into the image. `mlflow-ui`, `api`, and `pipeline` all mount the **entire repo root** into the container at `/workspace`, the same thing the `.devcontainer` workflow does automatically. That one mount covers `data/`, `models/` (including `models/mlruns/`), `logs/`, `docs/reports/`, and the `src/` code itself (the image never copies source code in), so everything stays in sync automatically between the host and every container. `spark-master`/`spark-worker` don't need this mount — they only need the Spark binaries already baked into the image.

### Secrets

The `api` and `pipeline` services read `API_KEY` from a local file (`.env`, excluded from git) the same way the devcontainer workflow already does — never written into the compose file or the image itself. Run `src/scripts/generate_api_key.py` first if you don't already have that file.

## Running it

Start the always-on services:

```
docker compose up -d spark-master spark-worker mlflow-ui api log-viewer
```

- Master UI: http://localhost:8080 — check that at least one worker shows up here before trusting anything else.
- FastAPI: http://localhost:8000 (`GET /health` needs no key; `/predict` needs an `X-API-Key` header, same as the devcontainer workflow).
- MLflow UI: http://localhost:5000.
- Log dashboard: http://localhost:8081.

A plain `docker compose up`, with no services named, starts the same set — `pipeline` is left out because it's marked as an on-demand-only service, which Compose skips unless you name it explicitly.

Scale workers:

```
docker compose up -d --scale spark-worker=3 spark-worker
```

### Batch jobs

Batch jobs never start automatically and never share a container with the always-on services — always run as an explicit one-off command:

```
docker compose run --rm pipeline python src/data_download.py
docker compose run --rm pipeline python src/preprocess.py
docker compose run --rm pipeline python src/train.py --sample-fraction 0.05
docker compose run --rm pipeline python src/evaluate.py --version 3
docker compose run --rm pipeline python src/set_active_model.py --version 3
```

`docker compose run` doesn't expose the service's usual ports by default — add `--service-ports` to a specific command if you want to reach that job's driver UI (port 4040) while it's running.

## Confirming a `/predict` call actually runs on the cluster

A request to `/predict` should show up as a running (or, right after, completed) application in the Master UI at http://localhost:8080, and `docker compose logs spark-worker` should show it doing work — confirming the request didn't silently fall back to running locally instead.

## Staying compatible with `.devcontainer`

`src/config.py` reads an optional `SPARK_MASTER_URL` setting, defaulting to Spark's simple built-in mode if it isn't set:

```python
master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")
```

Only the compose services (`api`, `pipeline`) set this. The `.devcontainer` workflow never does, so it keeps behaving exactly as it always has — nothing here changes anything for anyone using the existing setup.
