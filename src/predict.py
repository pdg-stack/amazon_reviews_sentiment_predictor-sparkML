"""
Batch CLI inference: scores every row in an input CSV/Parquet file using the
currently active model (models/active/, set by set_active_model.py) and
writes predictions to a CSV.

The model expects a `review_text` column (title + body combined, see
preprocess.py). If the input already has one (e.g. data/processed/*.parquet),
it's used as-is; otherwise it's built here from --text-col (required) and
--title-col (optional) -- matching the same title+body combination used at
training time.

Examples:
    python src/predict.py --input data/processed/test.parquet --output predictions.csv
    python src/predict.py --input reviews.csv --text-col body --title-col headline --output out.csv
"""

import argparse

import mlflow
import mlflow.spark
from pyspark.sql import functions as F

from src.config import ACTIVE_MODEL_DIR, MLFLOW_TRACKING_URI, get_spark_session


def _read_input(spark, input_path: str):
    if input_path.endswith(".parquet"):
        return spark.read.parquet(input_path)
    return spark.read.option("header", True).option("inferSchema", True).csv(input_path)


def _ensure_review_text(df, text_col: str, title_col: str | None):
    if "review_text" in df.columns:
        return df
    if text_col != "text":
        df = df.withColumnRenamed(text_col, "text")
    if title_col and title_col in df.columns:
        if title_col != "title":
            df = df.withColumnRenamed(title_col, "title")
        return df.withColumn("review_text", F.trim(F.concat_ws(" ", F.col("title"), F.col("text"))))
    return df.withColumn("review_text", F.trim(F.col("text")))


def main(input_path: str, output_path: str, text_col: str = "text", title_col: str | None = None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    spark = get_spark_session("predict")

    print(f"Loading active model from {ACTIVE_MODEL_DIR} ...")
    model = mlflow.spark.load_model(str(ACTIVE_MODEL_DIR))

    print(f"Reading {input_path} ...")
    df = _read_input(spark, input_path)
    df = _ensure_review_text(df, text_col, title_col)

    predictions = model.transform(df)
    result = predictions.select(
        "review_text",
        F.when(F.col("prediction") == 1.0, "positive").otherwise("negative").alias("sentiment"),
        F.col("probability"),
    )

    print(f"Writing predictions to {output_path} ...")
    result.toPandas().to_csv(output_path, index=False)
    print("Done.")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to a CSV or Parquet file."
    )
    parser.add_argument("--output", required=True, help="Path to write predictions CSV to.")
    parser.add_argument(
        "--text-col", default="text", help="Name of the input's body-text column, if not already 'text'."
    )
    parser.add_argument(
        "--title-col", default=None, help="Name of the input's title column, if it has one."
    )
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output, text_col=args.text_col, title_col=args.title_col)
