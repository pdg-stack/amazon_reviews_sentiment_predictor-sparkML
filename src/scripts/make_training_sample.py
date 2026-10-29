"""
Draws a reproducible sample from data/processed/train.parquet and splits it
into a train/eval pair, WITHOUT ever touching data/processed/test.parquet --
that file stays reserved for Phase 2's inference/load-test work (see
docs/DEPLOYMENT.md), not training-time evaluation.

    python -m src.scripts.make_training_sample --fraction 0.05 --train-frac 0.8 \
        --output-dir s3://arsp-183088117536-us-east-1/data/processed

Both the sample and the 80/20 split use the same seed (42) as every other
sampling/splitting operation in this project (preprocess.py, train.py), so
the same --fraction/--train-frac always reproduces the same two files.

Writes two parquet datasets under --output-dir:
  - train_<pct>pct.parquet : the 80% (or --train-frac) slice -- fed to
    train.py via --train-path. train.py does its own internal 80/20 split on
    top of this for Optuna's tuning/scoring; that's a separate, nested split.
  - eval_<pct>pct.parquet  : the remaining slice -- fed to evaluate.py via
    --test-path. Never seen by train.py, so it's this run's genuinely
    held-out check, at the scale this run actually trained on. Its metrics
    are NOT comparable to earlier versions' (those score against the full,
    fixed 400K-row official test set) -- this is a separate, self-consistent
    test set for this run only.
"""

import argparse

from src.config import TRAIN_PARQUET, get_spark_session


def main(fraction: float, train_frac: float, output_dir: str) -> None:
    spark = get_spark_session("make-training-sample")

    print(f"Loading {TRAIN_PARQUET} ...")
    full = spark.read.parquet(str(TRAIN_PARQUET))

    print(f"Sampling {fraction:.0%} (seed=42) ...")
    sample = full.sample(fraction=fraction, seed=42)
    sample.cache()
    total = sample.count()

    print(f"Splitting {train_frac:.0%}/{1 - train_frac:.0%} (seed=42) ...")
    train_slice, eval_slice = sample.randomSplit([train_frac, 1 - train_frac], seed=42)

    pct = round(fraction * 100)
    train_path = f"{output_dir.rstrip('/')}/train_{pct}pct.parquet"
    eval_path = f"{output_dir.rstrip('/')}/eval_{pct}pct.parquet"

    train_slice.write.mode("overwrite").parquet(train_path)
    eval_slice.write.mode("overwrite").parquet(eval_path)

    train_count = train_slice.count()
    eval_count = eval_slice.count()
    print(f"\nSampled {total} rows total from train.parquet:")
    print(f"  {train_count} rows -> {train_path}")
    print(f"  {eval_count} rows -> {eval_path}")
    print("\ndata/processed/test.parquet was not read or touched.")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of train.parquet to sample (e.g. 0.05).")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of the sample kept as the train slice (default: 0.8).")
    parser.add_argument("--output-dir", type=str, required=True, help="Local path or s3://... prefix to write both output datasets under.")
    args = parser.parse_args()
    main(fraction=args.fraction, train_frac=args.train_frac, output_dir=args.output_dir)
