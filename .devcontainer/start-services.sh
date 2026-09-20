#!/bin/bash
# Default container entrypoint: starts the MLflow UI in the background, then
# runs the FastAPI app in the foreground (as PID 1, so `docker stop`/signals
# work as expected). The Spark UI (port 4040) needs no separate step here --
# it comes up automatically the moment the API creates its SparkSession at
# startup (see src/api/app.py's load_model()).
#
# This only runs if the image's default CMD is used, i.e. `docker run` (or
# VS Code's Dev Containers attach) without an explicit command override --
# it does not interfere with opening an interactive shell in the container.
set -e

mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db &

exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000
