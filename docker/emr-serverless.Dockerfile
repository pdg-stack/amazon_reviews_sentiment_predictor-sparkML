# EMR Serverless custom image for the training job (Phase 1). Unlike
# docker/Dockerfile (the local/docker-compose image), this MUST be built FROM
# EMR Serverless's own base image -- it already has the matching Spark/Java/
# Python/Hadoop runtime EMR Serverless expects; only extra libraries get
# layered on top. release label must match infra/variables.tf's
# emr_release_label.
FROM public.ecr.aws/emr-serverless/spark/emr-7.2.0:latest

USER root

# Same extra libraries docker/Dockerfile installs, minus what the base image
# already ships (pyspark, boto3) and minus dev-only tools (jupyter, kaggle,
# pytest) this job never uses. psycopg2-binary is new here -- MLflow needs it
# to talk to the RDS Postgres backend store.
RUN pip3 install --no-cache-dir \
    optuna \
    mlflow \
    psycopg2-binary \
    pandas \
    matplotlib \
    seaborn \
    pyarrow \
    python-dotenv \
    boto3

COPY src /workspace/src
COPY docker/emr_train_job.py /workspace/docker/emr_train_job.py
COPY docker/emr_evaluate_job.py /workspace/docker/emr_evaluate_job.py
ENV PYTHONPATH=/workspace:$PYTHONPATH

# COPY above runs as root, so everything it created under /workspace is
# root-owned -- confirmed the hard way: evaluate.py's REPORTS_DIR.mkdir()
# (docs/reports/version_N/, created fresh at runtime, not something COPY
# ever populates) hit PermissionError the first time this ran as hadoop.
# chown the whole tree rather than just the paths used today, so any
# future runtime-created path under /workspace is covered too.
RUN chown -R hadoop:hadoop /workspace

# Must end back on the base image's own runtime user -- EMR Serverless's job
# harness expects it, not root.
USER hadoop:hadoop
