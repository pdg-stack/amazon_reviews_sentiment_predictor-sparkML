# Docker Compose: split-service deployment

`docker-compose.yml` (repo root) is an **alternative** to the `.devcontainer` workflow described in the main [`README.md`](../README.md) and [`ARCHITECTURE.md`](../ARCHITECTURE.md) -- not a replacement. The devcontainer keeps working completely unchanged: it's still one container, PySpark in embedded `local[*]` mode, you run whichever script you want by hand. Nothing here requires the devcontainer to be set up or touched.

What compose adds: genuinely separate, independently-running services instead of "one container you run everything inside," backed by a real standalone Spark cluster instead of each process's own embedded `local[*]` session.

## Why a real Spark cluster

For "Spark UI" to be its own persistent, independent service -- not just "whichever container happens to have an active SparkSession right now" -- there has to be an actual Spark cluster process (`spark-master`) that exists regardless of whether anything is currently running on it. `docker-compose.yml` stands up `spark-master` + `spark-worker`, and the API and pipeline containers become Spark *clients* that submit to `spark://spark-master:7077` instead of running embedded local Spark.

**Client and cluster Spark versions must match exactly**, or you get serialization/protocol errors between the driver and the executors. Rather than introduce a second Spark distribution (e.g. a Bitnami image) that could drift out of version sync, `spark-master`/`spark-worker` are built **from the exact same `.devcontainer/Dockerfile`** as every other service -- same pinned `pyspark==3.5.3` everywhere, just a different `command:` per service. One Dockerfile, one dependency set, no version-drift risk.

### A gotcha worth documenting: the pip pyspark package doesn't ship start-master.sh/start-worker.sh
`pip install pyspark` bundles the full Spark distribution (jars, `bin/`, `sbin/`), but the PyPI wheel's `sbin/` is trimmed down -- it ships `spark-daemon.sh`, `spark-config.sh`, and the history-server scripts, but **not** `start-master.sh`/`start-worker.sh`/`start-all.sh` (verified by inspecting the installed package inside this project's image; there is no separate download step to make them appear). `docker-compose.yml` therefore starts the master/worker directly via `spark-class`, which is what those missing scripts would have wrapped anyway:

```
spark-class org.apache.spark.deploy.master.Master --host 0.0.0.0 --port 7077 --webui-port 8080
spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077 --webui-port 8081
```

This also runs in the foreground, which is exactly what a container's `command:` needs (the `sbin` scripts, had they existed, daemonize into the background and return -- wrong shape for a container's main process). `SPARK_HOME` is set in `.devcontainer/Dockerfile` (`/usr/local/lib/python3.11/site-packages/pyspark`) so `spark-class` and its supporting scripts can find their dependent jars; it's a no-op for the `.devcontainer` workflow, which never invokes these scripts.

## Two different "Spark UI"s -- don't conflate them

- **Master UI** (`spark-master`, port **8080**): persistent, reachable at `http://localhost:8080` any time `spark-master` is running, independent of whether a job is active. This is what genuinely satisfies "Spark UI as its own independent service" -- it shows registered workers, active/completed applications, and cluster resource usage at all times.
- **Driver/application UI** (port **4040**): only exists while a specific application's driver process is alive -- the API's long-lived `SparkSession`, or a one-off `pipeline` job. It's hosted by the driver itself, which is inherent to Spark's architecture and can't be fully separated out even with a cluster. Exposed on the `api` service (and reachable on `pipeline` with `--service-ports`, see below) so it's available when something is actually running, but it is **not** a persistent service the way the master UI is -- it disappears when the driver process exits.

With the cluster in place, actual task *execution* happens on `spark-worker` container(s), not inside whichever container holds the driver -- the master UI's worker/application listing is how you confirm that's really happening (see verification steps below), not just "trust the config."

## Services

| Service | Purpose | Ports | Always on? |
|---|---|---|---|
| `spark-master` | Persistent Spark cluster master + Master UI | 8080 (UI), 7077 (cluster) | yes |
| `spark-worker` | Executes tasks submitted to the cluster | 8081 (worker UI, random host port) | yes; scale with `--scale spark-worker=N` |
| `mlflow-ui` | Browses `mlruns/` run/metric/registry history | 5000 | yes |
| `api` | FastAPI `/predict`, `/health`, `/admin/reload-model` -- a Spark client | 8000 (API), 4040 (driver UI, while a request runs) | yes |
| `pipeline` | `train.py`/`evaluate.py`/`preprocess.py`/`data_download.py`/`set_active_model.py` -- a Spark client | none by default | **no** -- `profiles: ["tools"]`, on-demand only |

All services build from the same `.devcontainer/Dockerfile` (`build: {context: ., dockerfile: .devcontainer/Dockerfile}`), so there is exactly one pinned dependency set for everything.

### Shared state (bind mounts, not named volumes)
Consistent with this project's existing philosophy (see `ARCHITECTURE.md`: everything is bind-mounted from the host, nothing project-specific is baked into the image), `mlflow-ui`, `api`, and `pipeline` all bind-mount the **entire repo root to `/workspace`** -- the same thing `.devcontainer` does. That single mount covers `data/`, `models/`, `mlruns/`, `logs/`, `reports/`, and the `src/` code itself (the image never `COPY`s source code in), so all of it stays in sync with the host and with each other automatically. `spark-master`/`spark-worker` mount nothing -- they only need the Spark binaries already in the image.

### Secrets
The `api` and `pipeline` services load `API_KEY` from a local, gitignored `.env` via `env_file:` (the same file `python-dotenv` also loads at API startup, and the same one `scripts/generate_api_key.py` writes) -- never baked into the compose file or the image.

## Running it

Start the always-on services:
```
docker compose up -d spark-master spark-worker mlflow-ui api
```
- Master UI: http://localhost:8080 -- confirm the worker(s) actually registered (non-zero worker count) before trusting anything else.
- FastAPI: http://localhost:8000 (`GET /health` is unauthenticated; `/predict` needs `X-API-Key`, same as the devcontainer workflow -- run `scripts/generate_api_key.py` first if you don't have a `.env` yet).
- MLflow UI: http://localhost:5000.

A plain `docker compose up` (no explicit service list) starts the same four -- `pipeline` is excluded because it's `profiles: ["tools"]`, which Compose omits from the default profile set unless you name it explicitly.

Scale workers:
```
docker compose up -d --scale spark-worker=3 spark-worker
```

### Batch jobs
Never auto-started, never sharing a container with the UIs -- always an explicit, one-off `run`:
```
docker compose run --rm pipeline python src/data_download.py
docker compose run --rm pipeline python src/preprocess.py
docker compose run --rm pipeline python src/train.py --sample-fraction 0.05
docker compose run --rm pipeline python src/evaluate.py --version 3
docker compose run --rm pipeline python src/set_active_model.py --version 3
```
`docker compose run` doesn't publish the service's declared ports by default; add `--service-ports` to a specific invocation if you want to reach that job's driver UI (port 4040) while it runs.

## Verifying a `/predict` call actually runs on the cluster
A request hitting `/predict` should show up as a registered application in the Master UI (http://localhost:8080) while it's in flight (or in "Completed Applications" right after), and `docker compose logs spark-worker` should show task execution -- not silently falling back to a local mode, since `SPARK_MASTER_URL=spark://spark-master:7077` is set explicitly via `environment:` on `api`, and `src/config.py`'s `get_spark_session()` only uses that value when it's set (see below).

## Backward compatibility with `.devcontainer`
`src/config.py`'s `get_spark_session()` reads an optional `SPARK_MASTER_URL` env var, defaulting to `"local[*]"` if it's unset:
```python
master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")
```
Only the compose services (`api`, `pipeline`) set this explicitly. The `.devcontainer` VS Code workflow never sets it, so it keeps getting `local[*]` exactly as before -- nothing about this change affects anyone using the existing setup.
