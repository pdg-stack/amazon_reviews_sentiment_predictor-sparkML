"""
Exports the currently active trained PipelineModel's learned parameters
(models/active/, set by set_active_model.py) into a small, Spark-free JSON
file that src/serving/lightweight_scorer.py scores requests from without
ever starting a SparkSession.

Why this needs its own step rather than deriving scoring logic live: the
five pipeline stages (Tokenizer, StopWordsRemover, HashingTF, IDF,
LogisticRegression) are plain-Python-reimplementable, but their LEARNED/
configured values -- the IDF weights, the LogisticRegression coefficients,
which numFeatures Optuna picked, which stop words were used -- only exist
inside the fitted Spark PipelineModel, so they have to be pulled out once
and saved, not recomputed on every request.

Run with:
    python src/serving/export_lightweight_model.py

set_active_model.py calls export_from_model() automatically after
activating a version (reusing the model it already loaded, rather than
loading it a second time), so models/active/lightweight_model.json always
matches whichever version models/active/ currently holds. This script's
own __main__ path is only for regenerating it by hand, without reactivating.
"""

import json
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import HashingTF, IDFModel, StopWordsRemover

from src.config import ACTIVE_MODEL_DIR, MLFLOW_TRACKING_URI, get_spark_session

LIGHTWEIGHT_MODEL_PATH = ACTIVE_MODEL_DIR / "lightweight_model.json"


def export_from_model(pipeline_model: PipelineModel, output_path: Path = LIGHTWEIGHT_MODEL_PATH) -> None:
    remover = next(s for s in pipeline_model.stages if isinstance(s, StopWordsRemover))
    hashing_tf = next(s for s in pipeline_model.stages if isinstance(s, HashingTF))
    idf_model = next(s for s in pipeline_model.stages if isinstance(s, IDFModel))
    lr_model = next(s for s in pipeline_model.stages if isinstance(s, LogisticRegressionModel))

    data = {
        "num_features": hashing_tf.getNumFeatures(),
        "stop_words": remover.getStopWords(),
        "case_sensitive": remover.getCaseSensitive(),
        "idf": idf_model.idf.toArray().tolist(),
        "coefficients": lr_model.coefficients.toArray().tolist(),
        "intercept": float(lr_model.intercept),
    }
    output_path.write_text(json.dumps(data))
    print(f"Wrote {output_path} ({len(data['coefficients'])} features).")


def export(active_model_dir: Path = ACTIVE_MODEL_DIR, output_path: Path = LIGHTWEIGHT_MODEL_PATH) -> None:
    """Standalone entrypoint: loads models/active/ fresh, then exports from it."""
    import mlflow
    import mlflow.spark

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    get_spark_session("export-lightweight-model")  # mlflow.spark.load_model needs a live SparkSession
    pipeline_model = mlflow.spark.load_model(str(active_model_dir))
    export_from_model(pipeline_model, output_path)


if __name__ == "__main__":
    export()
