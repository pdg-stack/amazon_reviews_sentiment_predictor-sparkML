"""
Evaluates a model against the fully held-out test.parquet -- data that has
never influenced training, tuning, or the validation check in train.py. By
default this scores the currently active model (models/active/, set by
set_active_model.py); pass --version N to instead score any specific
registered MLflow version directly, without touching models/active/ -- e.g.
to compare a newly trained candidate against the current version before
deciding whether to activate it:

    python src/evaluate.py --version 1
    python src/evaluate.py --version 2

Prints accuracy/F1/AUC and writes three charts to reports/ (or
reports/version_<N>/ when --version is given, so comparisons don't clobber
each other):

  - confusion_matrix.png : where the model's predictions land vs. reality
  - roc_curve.png        : true-positive vs. false-positive tradeoff
  - tuning_improvement.png : baseline vs. every CV candidate vs. the final
                             tuned model, pulled from the MLflow run history
                             logged by train.py -- this is the "did
                             hyperparameter tuning actually help" chart.
                             Always reflects the *most recent* train.py run,
                             regardless of which --version is being scored.
"""

import argparse
import datetime
import json

import matplotlib

matplotlib.use("Agg")  # no display inside the container
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from src.config import (
    ACTIVE_MODEL_DIR,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    TEST_PARQUET,
    get_spark_session,
)


def compute_metrics(predictions):
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
    f1 = multi_eval.setMetricName("f1").evaluate(predictions)
    auc = binary_eval.evaluate(predictions)
    return accuracy, f1, auc


def plot_confusion_matrix(predictions, out_path):
    counts = (
        predictions.groupBy("label", "prediction")
        .count()
        .toPandas()  # at most 4 rows for a binary problem -- safe to collect
        .pivot(index="label", columns="prediction", values="count")
        .fillna(0)
        .reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(counts, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticklabels(["negative", "positive"], rotation=0)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(predictions, auc, out_path):
    # PySpark's own BinaryClassificationMetrics only exposes areaUnderROC/
    # areaUnderPR to Python (the .roc()/.pr() curve methods are Scala-only),
    # so the curve itself is built here directly: bucket the predicted
    # probabilities into 100 bins with ONE Spark aggregation, then sweep the
    # classification threshold from 1.0 down to 0.0 in pandas (100 rows,
    # cheap) to get cumulative true/false-positive rates at each threshold.
    scored = predictions.select(vector_to_array(F.col("probability"))[1].alias("score"), F.col("label"))
    bucketed = scored.withColumn("bucket", F.least(F.floor(F.col("score") * 100), F.lit(100)))
    counts = (
        bucketed.groupBy("bucket")
        .agg(
            F.sum((F.col("label") == 1.0).cast("int")).alias("pos"),
            F.sum((F.col("label") == 0.0).cast("int")).alias("neg"),
        )
        .toPandas()
        .set_index("bucket")
        .reindex(range(0, 101), fill_value=0)
        .sort_index(ascending=False)  # threshold 1.0 -> 0.0
    )

    total_pos, total_neg = counts["pos"].sum(), counts["neg"].sum()
    tpr = counts["pos"].cumsum() / total_pos if total_pos else counts["pos"].cumsum() * 0.0
    fpr = counts["neg"].cumsum() / total_neg if total_neg else counts["neg"].cumsum() * 0.0
    fpr_points = [0.0] + fpr.tolist()
    tpr_points = [0.0] + tpr.tolist()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_points, tpr_points, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tuning_improvement(out_path):
    """
    Reads train.py's MLflow run history to show whether hyperparameter
    tuning actually helped: the untuned baseline, every CV candidate's
    average cross-validated AUC, and the final tuned model's validation AUC.
    """
    runs = mlflow.search_runs(experiment_names=["amazon_review_sentiment"])
    if runs.empty:
        print("No MLflow runs found yet -- run src/train.py first. Skipping tuning-improvement chart.")
        return

    run_name_col = "tags.mlflow.runName"
    baseline_runs = runs[runs[run_name_col] == "baseline"]
    candidate_runs = runs[runs[run_name_col] == "cv_candidate"]
    parent_runs = runs[runs[run_name_col] == "train_run"]

    if baseline_runs.empty or parent_runs.empty:
        print("Incomplete MLflow history -- run src/train.py first. Skipping tuning-improvement chart.")
        return

    baseline_auc = baseline_runs.iloc[-1]["metrics.validation_auc"]
    tuned_auc = parent_runs.iloc[-1]["metrics.validation_auc"]
    candidate_aucs = candidate_runs["metrics.cv_avg_auc"].dropna().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(["Baseline\n(default params)", "Tuned\n(best CV params)"], [baseline_auc, tuned_auc], color=["#999999", "#4C72B0"])
    axes[0].set_ylabel("Validation AUC")
    axes[0].set_title("Did hyperparameter tuning help?")
    for i, v in enumerate([baseline_auc, tuned_auc]):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center")

    axes[1].scatter(range(1, len(candidate_aucs) + 1), sorted(candidate_aucs), color="#4C72B0")
    axes[1].axhline(baseline_auc, linestyle="--", color="#999999", label="Baseline")
    axes[1].set_xlabel("Hyperparameter combination (sorted)")
    axes[1].set_ylabel("Avg. cross-validated AUC")
    axes[1].set_title("Every grid candidate's CV score")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(version: int | None = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if version is not None:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{version}"
        out_dir = REPORTS_DIR / f"version_{version}"
        label = f"version {version}"
    else:
        model_uri = str(ACTIVE_MODEL_DIR)
        out_dir = REPORTS_DIR
        label = "active model"
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = get_spark_session("evaluate")

    print(f"Loading {label} from {model_uri} ...")
    model = mlflow.spark.load_model(model_uri)

    print(f"Loading {TEST_PARQUET} ...")
    test_df = spark.read.parquet(str(TEST_PARQUET))
    predictions = model.transform(test_df).cache()

    accuracy, f1, auc = compute_metrics(predictions)
    print(f"Test set ({label}) -> accuracy={accuracy:.4f} f1={f1:.4f} auc={auc:.4f}")

    plot_confusion_matrix(predictions, out_dir / "confusion_matrix.png")
    plot_roc_curve(predictions, auc, out_dir / "roc_curve.png")
    plot_tuning_improvement(out_dir / "tuning_improvement.png")

    # Persisted (not just printed) so a separate script can later compare
    # several versions' test-set metrics side by side without re-scoring.
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "version": version if version is not None else "active",
                "model_uri": model_uri,
                "accuracy": accuracy,
                "f1": f1,
                "auc": auc,
                "evaluated_at": datetime.datetime.now().isoformat(),
            },
            indent=2,
        )
    )
    print(f"Charts + metrics.json written to {out_dir}")

    spark.stop()
    return accuracy, f1, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Evaluate this specific registered MLflow version instead of the active model.",
    )
    args = parser.parse_args()
    main(version=args.version)
