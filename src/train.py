"""
Trains the Amazon Reviews sentiment classifier.

This script is intentionally verbose and heavily commented -- the goal is
that someone new to Spark ML pipelines can read the printed output and the
comments here and understand not just *what* each step does, but *why* it's
there. See docs/model_plan.md for the full narrative version of this reasoning.

What happens when you run this script:
  1. Load data/processed/train.parquet (polarity {1,2} was already mapped
     to label {0,1}, and title+body already combined into `review_text`,
     in preprocess.py).
  2. Split into an 80% training slice and a 20% validation slice.
  3. Print a small, illustrated walkthrough of each pipeline stage on a
     5-row sample, purely for learning -- this doesn't affect the real fit.
  4. Build the real Pipeline, then search for good hyperparameters with
     Optuna: each trial fits once on the 80% split and is scored on the 20%
     split, and Optuna's sampler picks the next trial's values based on
     every previous trial's result (unlike a fixed grid, which tries the
     same points regardless of what earlier points showed).
  5. Log the whole search (baseline + every trial + the winner) to MLflow,
     and register the winning model as a new version -- this does NOT
     change what the API/predict.py currently serve; see
     set_active_model.py for that.
"""

import argparse

import mlflow
import mlflow.spark
import optuna
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
    print("with its hyperparameters searched by Optuna across the training split.")
    print("=" * 78 + "\n")


def build_pipeline():
    """
    The real pipeline, shared by both the untuned baseline and the tuned
    search below. The stage objects (hashing_tf, lr) are returned too, so
    each Optuna trial (and the final refit) can set their hyperparameters
    directly via setNumFeatures()/setRegParam()/setElasticNetParam().
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


def main(sample_fraction: float | None = None, n_trials: int = 30, cv_folds: int = 1) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("amazon_review_sentiment")

    spark = get_spark_session("train")

    print(f"Loading {TRAIN_PARQUET} ...")
    train_full = spark.read.parquet(str(TRAIN_PARQUET))
    if sample_fraction is not None:
        print(f"--sample-fraction {sample_fraction}: sampling training data for fast iteration.")
        train_full = train_full.sample(fraction=sample_fraction, seed=42)

    # Step 2: an 80/20 split. train_split is what every fit (baseline and
    # every Optuna trial) actually trains on; validation_split is what every
    # trial is *scored* against, so it's Optuna's search objective, not a
    # one-time-only check -- worth being explicit about, since it means
    # validation_split's score is no longer purely untouched-until-the-end
    # the way a single grid-search sanity check was. The fully held-out
    # test.parquet (used only by evaluate.py) is never touched by anything in
    # this file, so it remains the one truly unbiased number.
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
        mlflow.log_param("tuning_method", "optuna_tpe")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv_folds)

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

        # --- Hyperparameter tuning: Optuna searches numFeatures/regParam/
        # elasticNetParam, using its default TPE (Tree-structured Parzen
        # Estimator) sampler -- unlike a fixed grid, each new trial's values
        # are chosen based on what every earlier trial scored, so it can
        # explore a continuous range effectively instead of only a handful
        # of hand-picked points. binary_evaluator is reused across every
        # trial (and the final refit below) rather than constructed per call.
        #
        # cv_folds controls how each trial is SCORED, independent of how
        # Optuna picks values:
        #   - cv_folds <= 1 (default): one fit on train_split, one check on
        #     validation_split. Cheapest per trial, so n_trials can be large.
        #   - cv_folds > 1: each trial is k-fold cross-validated instead,
        #     via CrossValidator with a single-combination param grid (a
        #     ParamGridBuilder with no .addGrid() calls just evaluates
        #     whatever's already set on trial_hashing_tf/trial_lr below,
        #     across cv_folds folds of train_split) -- reusing Spark's own
        #     fold-splitting rather than hand-rolling it. Worth it mainly
        #     when training on a small sample, where a single validation
        #     split would be too small to give a trustworthy score. ---
        binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        def objective(trial: optuna.Trial) -> float:
            num_features = trial.suggest_categorical("numFeatures", [2**14, 2**16, 2**18, 2**20])
            reg_param = trial.suggest_float("regParam", 1e-4, 1.0, log=True)
            elastic_net_param = trial.suggest_float("elasticNetParam", 0.0, 1.0)

            trial_pipeline, trial_hashing_tf, trial_lr = build_pipeline()
            trial_hashing_tf.setNumFeatures(num_features)
            trial_lr.setRegParam(reg_param)
            trial_lr.setElasticNetParam(elastic_net_param)

            if cv_folds <= 1:
                trial_model = trial_pipeline.fit(train_split)
                auc = binary_evaluator.evaluate(trial_model.transform(validation_split))
            else:
                single_combo_grid = ParamGridBuilder().build()
                cv = CrossValidator(
                    estimator=trial_pipeline,
                    estimatorParamMaps=single_combo_grid,
                    evaluator=binary_evaluator,
                    numFolds=cv_folds,
                    seed=42,
                )
                auc = cv.fit(train_split).avgMetrics[0]

            # Logged under the SAME run name/metric key the old grid search
            # used ("cv_candidate"/"cv_avg_auc"), purely so evaluate.py's
            # existing tuning_improvement chart keeps working unchanged --
            # despite the name, this is one score per trial (single-split or
            # k-fold-averaged, depending on cv_folds above), not a fixed
            # grid's candidate.
            with mlflow.start_run(run_name="cv_candidate", nested=True):
                mlflow.log_params(
                    {"numFeatures": num_features, "regParam": reg_param, "elasticNetParam": elastic_net_param}
                )
                mlflow.log_metric("cv_avg_auc", auc)

            return auc

        cv_desc = "single-split" if cv_folds <= 1 else f"{cv_folds}-fold CV"
        print(f"\nRunning Optuna search: {n_trials} trials (TPE sampler, {cv_desc} scoring per trial) ...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"\nBest hyperparameters found: {best_params}")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Optuna trials don't keep their fitted PipelineModel around (that
        # would mean holding n_trials models in memory at once) -- refit once
        # more with the winning hyperparameters, on the same `pipeline` (and
        # its hashing_tf/lr stage objects) the baseline fit above already
        # used, to get the actual model object to evaluate and register below.
        hashing_tf.setNumFeatures(best_params["numFeatures"])
        lr.setRegParam(best_params["regParam"])
        lr.setElasticNetParam(best_params["elasticNetParam"])
        best_model = pipeline.fit(train_split)

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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run (default: 30). Each trial is one pipeline fit.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help=(
            "Cross-validation folds PER TRIAL (default: 1, meaning a single train/validation "
            "split -- not real cross-validation). Set e.g. 3 to k-fold each trial instead, "
            "which costs more per trial but gives a more trustworthy score on a small sample."
        ),
    )
    args = parser.parse_args()
    main(sample_fraction=args.sample_fraction, n_trials=args.n_trials, cv_folds=args.cv_folds)
