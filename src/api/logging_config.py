"""
Logging setup for the FastAPI app: two loggers with different jobs.

  - "api.requests" -- one JSON line per HTTP request (see app.py's
    log_requests middleware), written to logs/api_requests.log via a
    RotatingFileHandler so the file can't grow unbounded. The formatter
    emits the bare message only, since the middleware already builds a
    complete JSON string -- this preserves the existing "JSON lines" file
    contract that src/api/drift.py reads back.
  - "api.app" -- general operational messages (startup, drift-check
    warnings/errors), printed to the console like everything else this
    project already logs via print().
"""

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

request_logger = logging.getLogger("api.requests")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False  # don't also print these JSON lines to the console
if not request_logger.handlers:  # uvicorn --reload re-imports modules; avoid duplicate handlers
    _handler = RotatingFileHandler(LOGS_DIR / "api_requests.log", maxBytes=10_000_000, backupCount=5)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(_handler)

app_logger = logging.getLogger("api.app")
app_logger.setLevel(logging.INFO)
