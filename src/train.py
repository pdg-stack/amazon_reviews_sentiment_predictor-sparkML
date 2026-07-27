"""
Trains the Amazon Reviews sentiment classifier.

This script is intentionally verbose and heavily commented -- the goal is
that someone new to Spark ML pipelines can read the printed output and the
comments here and understand not just *what* each step does, but *why* it's
there. See model_plan.md for the full narrative version of this reasoning.

What happens when you run this script:
  1. Load data/processed/train.parquet (polarity {1,2} was already mapped
     to label {0,1}, and title+body already combined into `review_text`,
     in preprocess.py).
  2. Split into an 80% training slice and a 20% validation slice.
  3. Print a small, illustrated walkthrough of each pipeline stage on a
     5-row sample, purely for learning -- this doesn't affect the real fit.
  4. Build the real Pipeline, wrap it in a CrossValidator (k-fold
     cross-validation + a hyperparameter grid), and fit it on the 80% split.
  5. Log the whole search (baseline + every grid candidate + the winner) to
     MLflow, and register the winning model as a new version -- this does
     NOT change what the API/predict.py currently serve; see
     set_active_model.py for that.
"""

import argparse

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, TRAIN_PARQUET, get_spark_session


def print_stage_by_stage_walkthrough(sample_df):
    """
    Runs each transformer on a tiny 5-row sample and prints before/after, so
    the *purpose* of each stage is visible, not just the final vectors a
    model actually trains on. This is a separate, throwaway demonstration --
    it does not feed into the real Pipeline fit below.
    """
    print("\n" + "=" * 78)
    print("STAGE-BY-STAGE WALKTHROUGH (5-row sample, for illustration only)")
    print("=" * 78)

    print("\n--- Input: title + body combined into one review_text field ---")
    print("(preprocess.py already did this; shown here so the model's actual input is visible)")
    sample_df.select("title", "text", "review_text").show(truncate=50)

    print("--- Tokenizer: splits review_text into individual words ---")
    tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
    tokenized = tokenizer.transform(sample_df)
    tokenized.select("review_text", "words").show(truncate=60)

    print("--- StopWordsRemover: drops common low-signal words (\"the\", \"is\", \"a\", ...) ---")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    filtered = remover.transform(tokenized)
    filtered.select("words", "filtered_words").show(truncate=60)

    print("--- HashingTF: hashes remaining words into a fixed-size numeric vector ---")
    print("(a small numFeatures is used here only so the printed vectors stay readable)")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2**8)
    hashed = hashing_tf.transform(filtered)
    hashed.select("filtered_words", "raw_features").show(truncate=60)

    print("--- IDF: re-weights the hashed vector so rare-but-informative words count more ---")
    idf_model = IDF(inputCol="raw_features", outputCol="features").fit(hashed)
    reweighted = idf_model.transform(hashed)
    reweighted.select("raw_features", "features").show(truncate=60)

    print("=" * 78)
    print("End of walkthrough. The real Pipeline below repeats these same steps")
    print("(with a production-sized numFeatures) followed by LogisticRegression,")
    print("run via CrossValidator across the full training split.")
    print("=" * 78 + "\n")


def build_pipeline():
    """
    The real pipeline, shared by both the untuned baseline and the tuned
    search below. The stage objects (hashing_tf, lr) are returned too, so
    ParamGridBuilder can target their params directly.
    """
    tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, lr])
    return pipeline, hashing_tf, lr


def evaluate_split(model, df):
    """Returns (accuracy, f1, auc) for a fitted model on a held-out split."""
    predictions = model.transform(df)
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
    f1 = multi_eval.setMetricName("f1").evaluate(predictions)
    auc = binary_eval.evaluate(predictions)
    return accuracy, f1, auc


def main(sample_fraction: float | None = None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("amazon_review_sentiment")

    spark = get_spark_session("train")

    print(f"Loading {TRAIN_PARQUET} ...")
    train_full = spark.read.parquet(str(TRAIN_PARQUET))
    if sample_fraction is not None:
        print(f"--sample-fraction {sample_fraction}: sampling training data for fast iteration.")
        train_full = train_full.sample(fraction=sample_fraction, seed=42)

    # Step 2: an 80/20 split. CrossValidator folds the 80% split internally
    # for tuning; the 20% here is a fresh check on the winning model, never
    # seen by any CV fold. The fully held-out test.parquet (used only by
    # evaluate.py) is never touched by anything in this file.
    train_split, validation_split = train_full.randomSplit([0.8, 0.2], seed=42)
    train_split.cache()
    validation_split.cache()
    print(
        f"Train split: {train_split.count()} rows | "
        f"Validation split: {validation_split.count()} rows"
    )

    # Step 3: purely educational, doesn't affect the real fit below.
    print_stage_by_stage_walkthrough(train_full.limit(5))

    pipeline, hashing_tf, lr = build_pipeline()

    with mlflow.start_run(run_name="train_run"):
        mlflow.log_param("sample_fraction", sample_fraction if sample_fraction is not None else 1.0)
        mlflow.log_param("input_field", "review_text (title + body)")

        # --- Baseline: default hyperparameters, no tuning. This is the
        # "before" side of the before/after tuning-improvement chart in
        # evaluate.py. ---
        print("\nFitting untuned baseline (default hyperparameters) ...")
        with mlflow.start_run(run_name="baseline", nested=True):
            baseline_model = pipeline.fit(train_split)
            b_acc, b_f1, b_auc = evaluate_split(baseline_model, validation_split)
            mlflow.log_params(
                {
                    "numFeatures": hashing_tf.getNumFeatures(),
                    "regParam": lr.getRegParam(),
                    "elasticNetParam": lr.getElasticNetParam(),
                }
            )
            mlflow.log_metrics(
                {"validation_accuracy": b_acc, "validation_f1": b_f1, "validation_auc": b_auc}
            )
            print(f"Baseline -> accuracy={b_acc:.4f} f1={b_f1:.4f} auc={b_auc:.4f}")

        # --- Hyperparameter tuning: k-fold cross-validation over a small
        # grid. Each combination is evaluated on 5 folds of train_split;
        # CrossValidator returns the refit winner plus every candidate's
        # average metric (cv_model.avgMetrics, aligned with param_grid). ---
        param_grid = (
            ParamGridBuilder()
            .addGrid(hashing_tf.numFeatures, [2**16, 2**18])
            .addGrid(lr.regParam, [0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5])
            .build()
        )
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
            numFolds=5,
            parallelism=2,
            seed=42,
        )

        print(
            f"\nRunning 5-fold cross-validation over {len(param_grid)} hyperparameter "
            f"combinations ({5 * len(param_grid)} model fits total -- use --sample-fraction "
            "for a quick test run) ..."
        )
        cv_model = cv.fit(train_split)

        # Log every grid candidate as its own nested run, so evaluate.py can
        # chart the whole search landscape, not just the winner.
        for params, avg_auc in zip(param_grid, cv_model.avgMetrics):
            readable_params = {p.name: v for p, v in params.items()}
            with mlflow.start_run(run_name="cv_candidate", nested=True):
                mlflow.log_params(readable_params)
                mlflow.log_metric("cv_avg_auc", avg_auc)

        best_model = cv_model.bestModel
        best_hashing_tf = best_model.stages[2]
        best_lr = best_model.stages[4]
        best_params = {
            "numFeatures": best_hashing_tf.getNumFeatures(),
            "regParam": best_lr.getRegParam(),
            "elasticNetParam": best_lr.getElasticNetParam(),
        }
        print(f"\nBest hyperparameters found: {best_params}")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        t_acc, t_f1, t_auc = evaluate_split(best_model, validation_split)
        mlflow.log_metrics(
            {"validation_accuracy": t_acc, "validation_f1": t_f1, "validation_auc": t_auc}
        )
        print(f"Tuned model on validation split -> accuracy={t_acc:.4f} f1={t_f1:.4f} auc={t_auc:.4f}")
        print(f"(baseline was                     accuracy={b_acc:.4f} f1={b_f1:.4f} auc={b_auc:.4f})")

        # Registers a NEW version under MLFLOW_MODEL_NAME. This does not
        # change what the API/predict.py serve -- see set_active_model.py.
        mlflow.spark.log_model(best_model, "model", registered_model_name=MLFLOW_MODEL_NAME)
        print(f"\nRegistered new model version under '{MLFLOW_MODEL_NAME}'.")
        print(f"Browse it with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        print(
            "Run `python src/set_active_model.py --version N` to make a specific "
            "version the one predict.py/the API actually use."
        )

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help=(
            "Train on a random sample of this fraction (e.g. 0.05) instead of the "
            "full dataset, for fast iteration."
        ),
    )
    args = parser.parse_args()
    main(sample_fraction=args.sample_fraction)
