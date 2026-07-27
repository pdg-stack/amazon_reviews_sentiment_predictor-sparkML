"""
Explicitly activates ONE specific trained model version for local serving.

Both the FastAPI app (src/api/app.py) and the batch CLI (src/predict.py)
load their model only from models/active/ -- this script is the only thing
that ever writes to that directory. train.py only ever *registers* new
versions in MLflow; it never touches models/active/. Switching versions is
therefore always a deliberate, on-demand, auditable action:

    python src/set_active_model.py --version 3

By default the running API does NOT pick this up automatically -- there is
no background polling/hot-reload by design (see ARCHITECTURE.md). To set the
active version AND have a currently-running API start serving it, in one
command, add --reload-api:

    python src/set_active_model.py --version 3 --reload-api

This calls the API's POST /admin/reload-model endpoint (added for exactly
this) after activation. If the API isn't reachable (not running, wrong URL,
etc.), that call just prints a warning -- activation itself still succeeds,
you'd just need to start/restart the API manually to pick it up.
"""

import argparse
import datetime
import json
import os
import shutil
import urllib.error
import urllib.request

import mlflow
import mlflow.spark
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.config import (
    ACTIVE_MODEL_DIR,
    ACTIVE_MODEL_MANIFEST,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    get_spark_session,
)

load_dotenv(PROJECT_ROOT / ".env")


def _reload_running_api(api_url: str) -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print(
            f"--reload-api: no API_KEY found in {PROJECT_ROOT / '.env'} -- "
            "run scripts/generate_api_key.py first. Skipping API reload."
        )
        return

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/admin/reload-model",
        method="POST",
        headers={"X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"--reload-api: API reloaded -> {response.read().decode()}")
    except (urllib.error.URLError, TimeoutError) as e:
        print(
            f"--reload-api: could not reach the API at {api_url} ({e}). "
            "Activation still succeeded -- start/restart the API manually to pick it up."
        )


def main(version: int, reload_api: bool = False, api_url: str = "http://localhost:8000") -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    get_spark_session("set-active-model")  # mlflow's spark flavor needs a live SparkSession

    client = MlflowClient()
    model_version = client.get_model_version(MLFLOW_MODEL_NAME, str(version))
    print(f"Activating '{MLFLOW_MODEL_NAME}' version {version} (run_id={model_version.run_id}) ...")

    spark_model = mlflow.spark.load_model(f"models:/{MLFLOW_MODEL_NAME}/{version}")

    if ACTIVE_MODEL_DIR.exists():
        shutil.rmtree(ACTIVE_MODEL_DIR)
    mlflow.spark.save_model(spark_model, path=str(ACTIVE_MODEL_DIR))

    manifest = {
        "version": version,
        "run_id": model_version.run_id,
        "activated_at": datetime.datetime.now().isoformat(),
    }
    ACTIVE_MODEL_MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"models/active/ now serves version {version}.")

    if reload_api:
        _reload_running_api(api_url)
    else:
        print("Restart the API (or re-run predict.py) to pick this up, or rerun with --reload-api.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", type=int, required=True, help="Registered model version to activate."
    )
    parser.add_argument(
        "--reload-api",
        action="store_true",
        help="Also call a running API's /admin/reload-model so it picks up this version immediately.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running API, used only with --reload-api (default: http://localhost:8000).",
    )
    args = parser.parse_args()
    main(version=args.version, reload_api=args.reload_api, api_url=args.api_url)
