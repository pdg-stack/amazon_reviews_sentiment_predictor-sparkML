"""
EMR Serverless job entrypoint for a Phase 1 training run: trains via
src.pipeline.train against a pre-sampled/pre-split S3 dataset (see
src/scripts/make_training_sample.py), evaluates the newly-registered version
against its held-out slice via src.pipeline.evaluate, then uploads that
version's report directory to S3 -- its local disk disappears with the
ephemeral job.

    spark-submit --deploy-mode cluster \
        local:///workspace/docker/emr_train_job.py -- \
        --train-path s3://bucket/data/processed/train_5pct.parquet \
        --eval-path s3://bucket/data/processed/eval_5pct.parquet \
        --cv-folds 5 --n-trials 30 \
        --reports-s3-prefix s3://bucket/reports
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import boto3
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_MODEL_NAME, REPORTS_DIR
from src.pipeline.evaluate import main as evaluate_main
from src.pipeline.train import main as train_main


def latest_registered_version() -> int:
    versions = MlflowClient().search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    return max(int(v.version) for v in versions)


def upload_dir_to_s3(local_dir: Path, s3_uri: str) -> None:
    parsed = urlparse(s3_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(path), bucket, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--reports-s3-prefix", required=True)
    args = parser.parse_args()

    train_main(train_path=args.train_path, cv_folds=args.cv_folds, n_trials=args.n_trials)

    version = latest_registered_version()
    print(f"Registered version {version}. Evaluating against the held-out slice ...")
    evaluate_main(version=version, test_path=args.eval_path)

    local_dir = REPORTS_DIR / f"version_{version}"
    s3_dest = f"{args.reports_s3_prefix.rstrip('/')}/version_{version}/"
    print(f"Uploading {local_dir} -> {s3_dest} ...")
    upload_dir_to_s3(local_dir, s3_dest)

    print(f"\nDone. Registered + evaluated version {version}. Reports at {s3_dest}")


if __name__ == "__main__":
    main()
