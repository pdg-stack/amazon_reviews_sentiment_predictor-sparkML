"""
EMR Serverless recovery entrypoint: evaluates an ALREADY-REGISTERED model
version and uploads its report directory to S3, without rerunning
train_main() -- see emr_train_job.py, which this mirrors the second half of.
Exists because training can succeed and register a version while the
following report-write step still fails (e.g. a container permission bug);
re-submitting emr_train_job.py in that case would re-pay for the whole
Optuna search just to retry a step that never touched training at all.

    spark-submit --deploy-mode cluster \
        local:///workspace/docker/emr_evaluate_job.py -- \
        --version 1 \
        --eval-path s3://bucket/data/processed/eval_5pct.parquet \
        --reports-s3-prefix s3://bucket/reports
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import boto3

from src.config import REPORTS_DIR
from src.pipeline.evaluate import main as evaluate_main


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
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--reports-s3-prefix", required=True)
    args = parser.parse_args()

    evaluate_main(version=args.version, test_path=args.eval_path)

    local_dir = REPORTS_DIR / f"version_{args.version}"
    s3_dest = f"{args.reports_s3_prefix.rstrip('/')}/version_{args.version}/"
    print(f"Uploading {local_dir} -> {s3_dest} ...")
    upload_dir_to_s3(local_dir, s3_dest)

    print(f"\nDone. Evaluated version {args.version}. Reports at {s3_dest}")


if __name__ == "__main__":
    main()
