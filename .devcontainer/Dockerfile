FROM python:3.11-slim-bookworm
# Pinned to bookworm (Debian 12): Spark 3.5 officially supports Java 8/11/17,
# and Debian trixie (13) dropped openjdk-17 from its main repos in favor of 21.

# Spark needs a JVM; PySpark does not bundle one.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    pandas \
    matplotlib \
    seaborn \
    pyarrow \
    kaggle \
    jupyter \
    ipykernel \
    mlflow \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv

WORKDIR /workspace
