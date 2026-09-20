FROM python:3.11-slim-bookworm
# Pinned to bookworm (Debian 12): Spark 3.5 officially supports Java 8/11/17,
# and Debian trixie (13) dropped openjdk-17 from its main repos in favor of 21.

# Spark needs a JVM; PySpark does not bundle one. procps gives Spark's
# scripts (load-spark-env.sh, spark-daemon.sh) a working `ps` -- without it
# they still run, just with a harmless "ps: command not found" warning.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    procps \
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

# `pip install pyspark` bundles the full Spark distribution (jars, bin/,
# sbin/) under site-packages/pyspark -- SPARK_HOME points Spark's own
# scripts (spark-class, load-spark-env.sh, etc.) at it so they can find
# their dependent jars. Needed for the docker-compose spark-master/
# spark-worker services (see docker-compose.yml); harmless everywhere else
# (the .devcontainer / local[*] workflow doesn't read this var).
#
# NOTE: the PyPI pyspark wheel's sbin/ is trimmed down and does NOT ship
# start-master.sh/start-worker.sh (only spark-daemon.sh, spark-config.sh,
# and the history-server scripts) -- verified by inspecting the installed
# package. docker-compose.yml therefore starts the master/worker directly
# via `spark-class org.apache.spark.deploy.master.Master` / `...Worker`,
# which is what those missing scripts would have wrapped anyway, and which
# runs in the foreground -- exactly what a container command needs.
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

WORKDIR /workspace
