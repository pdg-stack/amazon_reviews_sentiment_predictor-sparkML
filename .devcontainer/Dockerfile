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
    python-dotenv \
    prometheus-fastapi-instrumentator \
    pytest

WORKDIR /workspace

# Only takes effect if the container is run without an explicit command
# override (VS Code's Dev Containers attach ignores this and opens a plain
# shell instead) -- starts the MLflow UI + FastAPI app together so both are
# reachable without a manual step. See start-services.sh for details.
COPY .devcontainer/start-services.sh /usr/local/bin/start-services.sh
RUN chmod +x /usr/local/bin/start-services.sh
CMD ["/usr/local/bin/start-services.sh"]
