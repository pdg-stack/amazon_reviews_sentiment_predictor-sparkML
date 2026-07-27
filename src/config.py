"""
Shared paths and Spark session helper, reused by every script and notebook in
this project so there is exactly one place that defines "where does data/
models/reports live" and "how do we start Spark."
"""

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

REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"  # artifact storage (model files, etc.)
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"  # run/metric/registry metadata

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

    local[*] uses all CPU cores available to the container -- there's no
    cluster here, just a single container, so this is the right mode for
    this project.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")  # small container, no need for Spark's 200-partition default
        .getOrCreate()
    )
