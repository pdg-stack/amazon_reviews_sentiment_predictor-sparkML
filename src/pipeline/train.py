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

from __future__ import annotations  # `X | None` annotations need this on Python <3.10 (EMR Serverless's base image)

import argparse
import functools

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

from src.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    TRAIN_PARQUET,
    get_spark_session,
)


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


def main(
    sample_fraction: float | None = None,
    n_trials: int = 30,
    cv_folds: int = 1,
    train_path: str | None = None,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # MLFLOW_ARTIFACT_ROOT only matters the FIRST time this experiment is
    # created against a given tracking store (e.g. a brand-new RDS backend) --
    # an experiment that already exists (the normal local case, or a second
    # cloud run) keeps whatever artifact_location it was created with.
    if mlflow.get_experiment_by_name("amazon_review_sentiment") is None and MLFLOW_ARTIFACT_ROOT:
        mlflow.create_experiment("amazon_review_sentiment", artifact_location=MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment("amazon_review_sentiment")

    spark = get_spark_session("train")

    input_path = train_path or str(TRAIN_PARQUET)
    print(f"Loading {input_path} ...")
    train_full = spark.read.parquet(input_path)
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
    train_rows, validation_rows = train_split.count(), validation_split.count()
    print(f"Train split: {train_rows} rows | Validation split: {validation_rows} rows")

    # Pre-split train_split into cv_folds disjoint folds ONCE, reused by every
    # trial's objective() call below -- cached since each fold is read
    # cv_folds-1 times per trial (as part of some other fold's training set)
    # across all n_trials trials. Only built when actually needed.
    folds = None
    if cv_folds > 1:
        folds = [f.cache() for f in train_split.randomSplit([1.0 / cv_folds] * cv_folds, seed=42)]

    # Step 3: purely educational, doesn't affect the real fit below.
    print_stage_by_stage_walkthrough(train_full.limit(5))

    pipeline, hashing_tf, lr = build_pipeline()

    with mlflow.start_run(run_name="train_run"):
        # `train_source` records exactly what data this run trained on --
        # the actual file/path used, not just a fraction, so a --train-path
        # override (a pre-sampled cloud file, e.g.) is never mislabeled as
        # "the full dataset" the way a bare sample_fraction=1.0 would read.
        mlflow.log_param("train_source", input_path)
        if sample_fraction is not None:
            mlflow.log_param("sample_fraction", sample_fraction)
        mlflow.log_param("train_split_rows", train_rows)
        mlflow.log_param("validation_split_rows", validation_rows)
        mlflow.log_param("input_field", "review_text (title + body)")
        mlflow.log_param("tuning_method", "optuna_tpe")
        mlflow.log_param("optimization_metric", "auc (areaUnderROC)")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv_folds)

        # --- Baseline: default hyperparameters, no tuning. This is the
        # "before" side of the before/after tuning-improvement chart in
        # evaluate.py. Logged both to its own nested run (below) AND, with a
        # baseline_ prefix, directly on this parent run -- so it's visible
        # at a glance without drilling into a child run. ---
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
            # Not registered (no registered_model_name) -- only the final
            # winning model below becomes a registry version. This is just
            # this run's own artifact, browsable/downloadable from its page.
            mlflow.spark.log_model(baseline_model, "model")
            print(f"Baseline -> accuracy={b_acc:.4f} f1={b_f1:.4f} auc={b_auc:.4f}")
        mlflow.log_metrics(
            {"baseline_accuracy": b_acc, "baseline_f1": b_f1, "baseline_auc": b_auc}
        )

        # --- Hyperparameter tuning: Optuna searches numFeatures/regParam/
        # elasticNetParam, using its default TPE (Tree-structured Parzen
        # Estimator) sampler -- unlike a fixed grid, each new trial's values
        # are chosen based on what every earlier trial scored, so it can
        # explore a continuous range effectively instead of only a handful
        # of hand-picked points. The metric being optimized throughout is
        # AUC (areaUnderROC) -- see optimization_metric above.
        #
        # Every trial is its own nested MLflow run, "run<trial number>"
        # (1-indexed), holding that trial's hyperparameters and its own
        # fitted model artifact (unregistered -- only the final winner below
        # becomes a registry version; every other logged model here is
        # browsable/downloadable from its own run page only):
        #   - cv_folds <= 1 (default): "run<N>" alone holds the single
        #     train/validation-split auc + that fit's model. Cheapest per
        #     trial, so n_trials can be large.
        #   - cv_folds > 1: "run<N>" is itself a parent for cv_folds child
        #     runs, "run<N>_cv<K>" (1-indexed), each one fold's own fit +
        #     auc + model on that fold -- hand-rolled (not Spark's
        #     CrossValidator, which only exposes an already-averaged score
        #     and discards each fold's model) specifically so each fold is
        #     individually visible/comparable in MLflow, not just the
        #     average. "run<N>" itself then holds the params plus auc_mean
        #     and one auc_fold<K> per fold (no model of its own -- each
        #     fold's model lives on its own "run<N>_cv<K>" run). Worth the
        #     extra cost mainly when training on a small sample, where a single
        #     validation split would be too small to give a trustworthy
        #     score. ---
        binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        def fit_and_score(train_df, eval_df, num_features, reg_param, elastic_net_param):
            fold_pipeline, fold_hashing_tf, fold_lr = build_pipeline()
            fold_hashing_tf.setNumFeatures(num_features)
            fold_lr.setRegParam(reg_param)
            fold_lr.setElasticNetParam(elastic_net_param)
            fold_model = fold_pipeline.fit(train_df)
            auc = binary_evaluator.evaluate(fold_model.transform(eval_df))
            return fold_model, auc

        def objective(trial: optuna.Trial) -> float:
            num_features = trial.suggest_categorical("numFeatures", [2**14, 2**16, 2**18, 2**20])
            reg_param = trial.suggest_float("regParam", 1e-4, 1.0, log=True)
            elastic_net_param = trial.suggest_float("elasticNetParam", 0.0, 1.0)
            trial_params = {
                "numFeatures": num_features,
                "regParam": reg_param,
                "elasticNetParam": elastic_net_param,
            }
            run_label = f"run{trial.number + 1}"

            with mlflow.start_run(run_name=run_label, nested=True):
                mlflow.log_params(trial_params)

                if cv_folds <= 1:
                    model, auc = fit_and_score(
                        train_split, validation_split, num_features, reg_param, elastic_net_param
                    )
                    mlflow.log_metric("auc", auc)
                    mlflow.spark.log_model(model, "model")
                    return auc

                fold_aucs = []
                for fold_idx in range(cv_folds):
                    val_fold = folds[fold_idx]
                    train_fold = functools.reduce(
                        lambda a, b: a.union(b),
                        [f for i, f in enumerate(folds) if i != fold_idx],
                    )
                    fold_model, fold_auc = fit_and_score(
                        train_fold, val_fold, num_features, reg_param, elastic_net_param
                    )
                    fold_aucs.append(fold_auc)
                    with mlflow.start_run(run_name=f"{run_label}_cv{fold_idx + 1}", nested=True):
                        mlflow.log_params(trial_params)
                        mlflow.log_param("fold", fold_idx + 1)
                        mlflow.log_metric("auc", fold_auc)
                        mlflow.spark.log_model(fold_model, "model")

                avg_auc = sum(fold_aucs) / len(fold_aucs)
                mlflow.log_metric("auc_mean", avg_auc)
                for i, fold_auc in enumerate(fold_aucs):
                    mlflow.log_metric(f"auc_fold{i + 1}", fold_auc)
                return avg_auc

        cv_desc = "single-split" if cv_folds <= 1 else f"{cv_folds}-fold CV"
        print(f"\nRunning Optuna search: {n_trials} trials (TPE sampler, {cv_desc} scoring per trial) ...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"\nBest hyperparameters found: {best_params}")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_trial_auc", study.best_value)

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
            "Run `python src/serving/set_active_model.py --version N` to make a specific "
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
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help=(
            "Read training data from this path instead of data/processed/train.parquet -- "
            "a local path or an s3://... URI (e.g. for a pre-sampled/pre-split cloud run). "
            "Spark reads it directly, so any Spark-supported URI scheme works."
        ),
    )
    args = parser.parse_args()
    main(
        sample_fraction=args.sample_fraction,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        train_path=args.train_path,
    )
