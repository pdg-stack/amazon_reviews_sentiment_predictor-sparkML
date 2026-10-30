"""
Proves lightweight_scorer.py agrees with the real Spark PipelineModel,
rather than trusting the hand-written replication (Tokenizer, HashingTF's
murmur3 bucket hashing, IDF, the logistic sigmoid) by inspection alone.

Scores the same sample of real review text through BOTH:
  - the actual Spark model in models/active/ (ground truth), and
  - models/active/lightweight_model.json via LightweightModel,
and asserts every row's predicted class matches and its probability is
within a small tolerance (floating-point order-of-operations differences
between Spark's JVM arithmetic and this module's Python arithmetic are
expected; a real algorithmic mismatch would show up as either a flipped
class or a gap far larger than that tolerance, not a near-miss).

Run with:
    python src/serving/validate_lightweight_model.py [--sample-size 200]
"""

import argparse

import mlflow
import mlflow.spark

from src.config import ACTIVE_MODEL_DIR, MLFLOW_TRACKING_URI, TEST_PARQUET, get_spark_session
from src.serving.export_lightweight_model import LIGHTWEIGHT_MODEL_PATH
from src.serving.lightweight_scorer import LightweightModel

PROBABILITY_TOLERANCE = 1e-3


def main(sample_size: int) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    spark = get_spark_session("validate-lightweight-model")

    print(f"Loading the real Spark model from {ACTIVE_MODEL_DIR} ...")
    spark_model = mlflow.spark.load_model(str(ACTIVE_MODEL_DIR))

    print(f"Loading the lightweight export from {LIGHTWEIGHT_MODEL_PATH} ...")
    lightweight_model = LightweightModel(LIGHTWEIGHT_MODEL_PATH)

    print(f"Sampling {sample_size} rows from {TEST_PARQUET} ...")
    sample_df = spark.read.parquet(str(TEST_PARQUET)).limit(sample_size)
    review_texts = [row["review_text"] for row in sample_df.select("review_text").collect()]

    spark_predictions = (
        spark_model.transform(sample_df.select("review_text"))
        .select("review_text", "prediction", "probability")
        .collect()
    )
    lightweight_results = lightweight_model.score_batch(review_texts)

    mismatched_class = []
    mismatched_probability = []
    for i, (spark_row, lightweight_row) in enumerate(zip(spark_predictions, lightweight_results)):
        spark_class = int(spark_row["prediction"])
        spark_probability = float(spark_row["probability"][spark_class])
        lightweight_class = 1 if lightweight_row["sentiment"] == "positive" else 0
        lightweight_probability = lightweight_row["probability"]

        if spark_class != lightweight_class:
            mismatched_class.append((i, spark_class, lightweight_class))
        elif abs(spark_probability - lightweight_probability) > PROBABILITY_TOLERANCE:
            mismatched_probability.append((i, spark_probability, lightweight_probability))

    total = len(spark_predictions)
    print(f"\nChecked {total} rows.")
    print(f"Class mismatches: {len(mismatched_class)}")
    for i, spark_class, lightweight_class in mismatched_class[:10]:
        print(f"  row {i}: spark={spark_class} lightweight={lightweight_class}")
    print(f"Probability mismatches (> {PROBABILITY_TOLERANCE}): {len(mismatched_probability)}")
    for i, spark_p, lightweight_p in mismatched_probability[:10]:
        print(f"  row {i}: spark={spark_p:.6f} lightweight={lightweight_p:.6f}")

    if mismatched_class or mismatched_probability:
        raise SystemExit(
            f"\nFAILED: {len(mismatched_class)} class + {len(mismatched_probability)} "
            "probability mismatches -- the lightweight scorer does not yet agree with "
            "the real Spark model. Do not switch the API over until this passes."
        )
    print("\nPASSED: lightweight_scorer.py agrees with the real Spark model on every row.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    main(args.sample_size)
