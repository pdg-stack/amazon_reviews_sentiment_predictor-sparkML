# Small, standalone image for the log-viewer service (see docker-compose.yml).
# Not built from .devcontainer/Dockerfile like every other service -- GoAccess
# has nothing to do with the Spark/MLflow/FastAPI stack the rest of this
# project shares, so it gets its own minimal image instead of carrying that
# whole dependency set for one log dashboard.
FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends goaccess \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
