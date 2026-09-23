"""
Logging setup for the FastAPI app: three loggers with different jobs.

  - "api.requests" -- one JSON line per real HTTP request (see app.py's
    log_requests middleware), written to logs/api_requests.log via a
    RotatingFileHandler so the file can't grow unbounded. The formatter
    emits the bare message only, since the middleware already builds a
    complete JSON string -- this preserves the existing "JSON lines" file
    contract that src/api/drift.py reads back.
  - "api.probes" -- one JSON line per GET /health or GET /metrics hit,
    written to logs/api_probes.log (same rotation policy, separate file so
    liveness checks and Prometheus scrapes don't drown out real traffic in
    api_requests.log).
  - "api.app" -- general operational messages (startup, drift-check
    warnings/errors), printed to the console like everything else this
    project already logs via print().
"""

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _file_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't also print these JSON lines to the console
    if not logger.handlers:  # uvicorn --reload re-imports modules; avoid duplicate handlers
        handler = RotatingFileHandler(LOGS_DIR / filename, maxBytes=10_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


request_logger = _file_logger("api.requests", "api_requests.log")
probe_logger = _file_logger("api.probes", "api_probes.log")

app_logger = logging.getLogger("api.app")
app_logger.setLevel(logging.INFO)
