"""
Downloads the Amazon Reviews dataset from Kaggle into data/raw/.

Requires a Kaggle API token at .kaggle/access_token (plain text, see README)
-- inside the devcontainer this is bind-mounted to /root/.kaggle, which is
where the `kaggle` package looks by default.
"""

import argparse

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    KAGGLE_DATASET,
    TEST_CSV,
    TEST_PARQUET,
    TRAIN_CSV,
    TRAIN_PARQUET,
)


def main(force: bool = False) -> None:
    if not force and TRAIN_PARQUET.exists() and TEST_PARQUET.exists():
        print(f"Processed data already present at {DATA_PROCESSED_DIR} -- skipping download (use --force to re-download).")
        return
    if not force and TRAIN_CSV.exists() and TEST_CSV.exists():
        print(f"Raw data already present at {DATA_RAW_DIR} (use --force to re-download).")
        return

    # Imported lazily: importing `kaggle` eagerly validates credentials at
    # import time, which would break --help and anything that doesn't
    # actually need to download data.
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} into {DATA_RAW_DIR} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW_DIR), unzip=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if data/raw already has the files."
    )
    args = parser.parse_args()
    main(force=args.force)
