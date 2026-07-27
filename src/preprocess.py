"""
Loads the raw Amazon Reviews CSVs, cleans/casts them, and writes Parquet to
data/processed/. Parquet is typed and columnar, so every later stage
(EDA, training, evaluation) reads it far faster than re-parsing the raw CSV
each time.
"""

import argparse

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.config import (
    DATA_PROCESSED_DIR,
    TEST_CSV,
    TEST_PARQUET,
    TRAIN_CSV,
    TRAIN_PARQUET,
    get_spark_session,
)

# The raw CSVs ship with no header row. This is the documented schema for
# kritanjalijain/amazon-reviews: polarity is 1=negative, 2=positive.
RAW_SCHEMA = StructType(
    [
        StructField("polarity", IntegerType(), nullable=False),
        StructField("title", StringType(), nullable=True),
        StructField("text", StringType(), nullable=False),
    ]
)


def _load_and_clean(spark, csv_path):
    df = spark.read.csv(
        str(csv_path), header=False, schema=RAW_SCHEMA, multiLine=True, escape='"'
    )
    return (
        df
        # MLlib's binary classifiers expect a {0.0, 1.0} label; the raw
        # data uses {1, 2}, so shift it down by one.
        .withColumn("label", (F.col("polarity") - 1).cast("double"))
        .withColumn("text", F.trim(F.col("text")))
        .filter(F.col("text").isNotNull() & (F.length("text") > 0))
        # The model trains on title + body together -- a review's title is
        # often a concentrated sentiment signal on its own (e.g. a title of
        # just "Terrible!!") that's wasteful to throw away. concat_ws skips
        # null title values automatically, so this degrades gracefully to
        # just the body text when a review has no title.
        .withColumn("review_text", F.trim(F.concat_ws(" ", F.col("title"), F.col("text"))))
    )


def main(force: bool = False) -> None:
    if not force and TRAIN_PARQUET.exists() and TEST_PARQUET.exists():
        print(f"Processed data already present at {DATA_PROCESSED_DIR} (use --force to reprocess).")
        return

    spark = get_spark_session("preprocess")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {TRAIN_CSV} ...")
    train_df = _load_and_clean(spark, TRAIN_CSV)
    train_df.write.mode("overwrite").parquet(str(TRAIN_PARQUET))
    print(f"Wrote {train_df.count()} rows to {TRAIN_PARQUET}")

    print(f"Reading {TEST_CSV} ...")
    test_df = _load_and_clean(spark, TEST_CSV)
    test_df.write.mode("overwrite").parquet(str(TEST_PARQUET))
    print(f"Wrote {test_df.count()} rows to {TEST_PARQUET}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if data/processed already has the files."
    )
    args = parser.parse_args()
    main(force=args.force)
