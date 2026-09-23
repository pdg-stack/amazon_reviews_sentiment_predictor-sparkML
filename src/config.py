"""
Shared paths and Spark session helper, reused by every script and notebook in
this project so there is exactly one place that defines "where does data/
models/reports live" and "how do we start Spark."
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession

# All paths are relative to the project root (this file's grandparent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Every trained model version lives in MLflow's registry (see train.py). This
# is the ONE local path the API and predict.py load from -- it is only ever
# updated by explicitly running set_active_model.py, never automatically.
MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_MODEL_DIR = MODELS_DIR / "active"
ACTIVE_MODEL_MANIFEST = ACTIVE_MODEL_DIR / "ACTIVE_MODEL.json"
# Drift baseline (prediction-confidence + review-length histograms) for
# whichever version is currently active -- copied here from
# docs/reports/version_N/baseline_stats.json by set_active_model.py, so it always
# matches the model actually being served. See src/api/drift.py.
ACTIVE_BASELINE_PATH = ACTIVE_MODEL_DIR / "BASELINE_STATS.json"

REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
MLRUNS_DIR = PROJECT_ROOT / "models" / "mlruns"  # artifact storage (model files, etc.)
MLFLOW_DB_PATH = MLRUNS_DIR / "mlflow.db"  # run/metric/registry metadata -- kept alongside the artifacts it describes, not loose at the project root

# A plain file-store tracking URI (file:./mlruns) can't back the Model
# Registry (registered_model_name=, get_model_version(), models:/name/N URIs
# -- everything set_active_model.py relies on) and MLflow 3.x additionally
# refuses to open it at all without an opt-in env var. SQLite is the
# lightest backend that supports the registry, with no separate server to run.
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_MODEL_NAME = "amazon_review_sentiment_predictor"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
TRAIN_PARQUET = DATA_PROCESSED_DIR / "train.parquet"
TEST_PARQUET = DATA_PROCESSED_DIR / "test.parquet"

KAGGLE_DATASET = "kritanjalijain/amazon-reviews"


def get_spark_session(app_name: str = "amazon-reviews-sentiment") -> SparkSession:
    """
    One shared way to start Spark, used by every script/notebook.

    Master defaults to local[*] -- all CPU cores available to the container --
    which is correct for the .devcontainer VS Code workflow: one container,
    no cluster, so this is the right mode there and stays the unchanged
    default for anyone using that setup. In local mode the driver and
    executor share one JVM, so spark.driver.memory is the only heap that
    matters. The 1g default was fine for train.py's --sample-fraction runs
    but OOM'd evaluate.py/predict.py against the full, un-sampled 400K-row
    test set -- 2g gives enough headroom for that on this container's
    ~3.8GB total allocation.

    The docker-compose setup (docker-compose.yml) is the one exception: it
    runs a real standalone Spark cluster (spark-master/spark-worker) and
    sets SPARK_MASTER_URL=spark://spark-master:7077 via `environment:` on
    the api/pipeline services so they submit to it as clients instead of
    running embedded local Spark. Nothing else needs to set this env var --
    it's unset (and this falls back to local[*]) everywhere else, including
    the .devcontainer image.
    """
    master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")  # small container, no need for Spark's 200-partition default
    )
    # Standalone Spark's default is "an application gets ALL available cores
    # in the cluster, for as long as it's alive" -- fine for a one-off batch
    # job, but fatal for a small cluster once a long-lived app is in the mix:
    # confirmed by testing, the api service's persistent SparkSession
    # silently starved a `pipeline` job of every executor slot (it sat
    # printing "Initial job has not accepted any resources" indefinitely)
    # because api had already claimed both of the single worker's cores and
    # never gives them back. SPARK_CORES_MAX (unset by default, so this is a
    # no-op for local[*] and for any cluster session that doesn't need it)
    # lets a service cap how many cores it claims, leaving the rest free for
    # other concurrent applications -- see docker-compose.yml, where `api`
    # sets it to 1.
    cores_max = os.environ.get("SPARK_CORES_MAX")
    if cores_max:
        builder = builder.config("spark.cores.max", cores_max)
    return builder.getOrCreate()
