"""
FastAPI real-time serving layer.

    POST /predict             (requires X-API-Key) -> {"sentiment": ..., "probability": ...}
    POST /admin/reload-model  (requires X-API-Key) -> reloads models/active/ into memory, no restart needed
    GET  /admin/drift-report  (requires X-API-Key) -> computes a fresh drift check now (see src/api/drift.py)
    GET  /health               (no auth)            -> {"status": "ok"}
    GET  /metrics               (no auth)            -> Prometheus text exposition

Loads models/active/ (set by set_active_model.py) at startup and keeps the
SparkSession resident for the life of the process -- fine for demo/dev
latency, not a high-throughput production pattern (see docs/ARCHITECTURE_NOTES.md).

Every request is logged as one JSON line (including client_ip/user_agent),
via the log_requests middleware below. GET /health and GET /metrics
(liveness checks / Prometheus scrapes) go to logs/api_probes.log; everything
else goes to logs/api_requests.log. /predict calls carry extra detail
(input/results/lengths) and are tagged "event": "predict"; a failed
/predict call (400/401) still logs the input that was sent, just without
results. Every successful /predict call also writes one row per scored text
to the `app.predictions` table (src/api/db.py) when APP_DB_HOST is
configured (a cloud deployment) -- a no-op locally. GET /admin/drift-report
computes a fresh check against that table on demand (no internal loop --
see docs/OBSERVABILITY.md); an external scheduler (EventBridge Scheduler in
the cloud) calls it on a cadence, not this process.

The model expects a `review_text` field (title + body combined, see
preprocess.py/docs/model_plan.md) -- /predict accepts an optional `title`/`titles`
alongside `text`/`texts` and combines them the same way, so a request run
through this endpoint sees exactly what the model was trained on.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import datetime
import hashlib
import json
import os
import time
from typing import List, Optional, Union

import mlflow
import mlflow.spark
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.api import db, drift
from src.api.logging_config import app_logger, probe_logger, request_logger
from src.api.security import require_api_key
from src.config import ACTIVE_MODEL_DIR, ACTIVE_MODEL_MANIFEST, MLFLOW_TRACKING_URI, PROJECT_ROOT, get_spark_session

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Amazon Reviews Sentiment Predictor API")

# Registered before the logging middleware below on purpose: Starlette builds
# its middleware stack in reverse registration order, so whichever of these
# two is added LAST ends up OUTERMOST. Instrumentator first -> the logging
# middleware ends up outermost -> its status/latency reflect the true
# client-facing response, not something an inner layer produced.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version", "Currently active model version, from models/active/ACTIVE_MODEL.json"
)
DRIFT_PSI = Gauge(
    "prediction_drift_psi", "Latest Population Stability Index score per drift signal", ["feature"]
)

# How far back GET /admin/drift-report looks each time it's called -- 120s
# (2 min) matches the cadence an external scheduler (EventBridge Scheduler
# in the cloud) calls it at, so each check's window lines up with the gap
# since the last one. At this cadence MIN_SAMPLE_SIZE (drift.py, 30) needs
# roughly 15+ predictions/minute of sustained traffic to clear at all --
# below that, most checks report "insufficient_data" rather than a verdict.
# Env-configurable for a busier/quieter deployment.
DRIFT_WINDOW_SECONDS = int(os.environ.get("DRIFT_WINDOW_SECONDS", "120"))

_spark = None
_model = None


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    title: Optional[str] = None
    titles: Optional[List[str]] = None


def _load_active_model():
    global _model
    _model = mlflow.spark.load_model(str(ACTIVE_MODEL_DIR))


def _set_active_model_version_gauge():
    if not ACTIVE_MODEL_MANIFEST.exists():
        return
    manifest = json.loads(ACTIVE_MODEL_MANIFEST.read_text())
    version = manifest.get("version")
    if isinstance(version, int):
        ACTIVE_MODEL_VERSION.set(version)


@app.on_event("startup")
def load_model():
    global _spark
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _spark = get_spark_session("api")
    _load_active_model()
    _set_active_model_version_gauge()
    app_logger.info("Model loaded, Spark session started.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    # Captured up front so it's available even if the request fails before
    # predict() ever runs (e.g. a bad API key, rejected by the require_api_key
    # dependency) -- Starlette caches the body bytes, so the handler can still
    # read the same body normally afterward.
    raw_input = None
    if request.method == "POST" and request.url.path == "/predict":
        try:
            raw_input = json.loads(await request.body())
        except Exception:
            raw_input = None

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.time() - start) * 1000, 2)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": "access",
            "method": request.method,
            "path": request.url.path,
            "status": 500,
            "latency_ms": latency_ms,
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
        if raw_input is not None:
            entry["input"] = raw_input
        request_logger.info(json.dumps(entry))
        app_logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
        raise  # re-raise, never swallow -- ServerErrorMiddleware (above us) still needs this to produce the client's 500

    latency_ms = round((time.time() - start) * 1000, 2)

    if request.url.path in ("/health", "/metrics"):
        probe_logger.info(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                    "client_ip": client_ip,
                }
            )
        )
        return response

    predict_detail = getattr(request.state, "predict_detail", None)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": "predict" if predict_detail else "access",
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "user_agent": user_agent,
    }
    if predict_detail:
        entry.update(predict_detail)
    elif response.status_code >= 400 and raw_input is not None:
        # A failed /predict call (bad payload, bad key) never reaches the
        # point where predict() sets predict_detail -- still record what was
        # sent so a 400/401 entry shows why, not just that it happened.
        entry["input"] = raw_input
    request_logger.info(json.dumps(entry))
    return response


def _combine(texts: List[str], titles: Optional[List[str]]) -> List[str]:
    if not titles:
        return [t.strip() for t in texts]
    return [f"{title} {text}".strip() if title else text.strip() for title, text in zip(titles, texts)]


def _score(review_texts: List[str]) -> List[dict]:
    df = _spark.createDataFrame([(t,) for t in review_texts], ["review_text"])
    predictions = _model.transform(df).select("review_text", "prediction", "probability").collect()
    results = []
    for row in predictions:
        sentiment = "positive" if row["prediction"] == 1.0 else "negative"
        probability = float(row["probability"][int(row["prediction"])])
        results.append({"sentiment": sentiment, "probability": round(probability, 4)})
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


def _active_model_identity() -> tuple[Optional[int], Optional[str]]:
    if not ACTIVE_MODEL_MANIFEST.exists():
        return None, None
    manifest = json.loads(ACTIVE_MODEL_MANIFEST.read_text())
    return manifest.get("version"), manifest.get("run_id")


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: PredictRequest, request: Request) -> Union[dict, List[dict]]:
    received_at = datetime.datetime.utcnow()
    if payload.text is None and not payload.texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'.")

    texts = payload.texts if payload.texts is not None else [payload.text]
    titles = payload.titles if payload.titles is not None else ([payload.title] if payload.title is not None else None)
    if titles is not None and len(titles) != len(texts):
        raise HTTPException(status_code=400, detail="'titles' must be the same length as 'texts'.")

    review_texts = _combine(texts, titles)
    results = _score(review_texts)
    responded_at = datetime.datetime.utcnow()

    # Stashed on request.state rather than written to a file here -- the
    # log_requests middleware above picks this up after call_next() returns
    # and merges it into the one log entry it writes for this request, so
    # there's a single logging call site for both success and error paths.
    request.state.predict_detail = {
        "input": {"text": texts, "title": titles},
        "results": results,
        # Word count of the COMBINED review_text (title + body), matching
        # exactly what evaluate.py's baseline_stats.json measures -- not the
        # raw pre-combine text/title above, which drift.py never reads.
        "review_text_lengths": [len(rt.split()) for rt in review_texts],
    }

    # One app.predictions row per text scored (a batch call scores several
    # texts in one request, so one API call can write several rows) -- a
    # no-op locally where APP_DB_HOST isn't set. api_key_id is a short,
    # non-reversible fingerprint of the caller's key, not the key itself.
    api_key = request.headers.get("x-api-key")
    api_key_id = hashlib.sha256(api_key.encode()).hexdigest()[:8] if api_key else None
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    model_version, run_id = _active_model_identity()
    titles_for_rows = titles if titles is not None else [None] * len(texts)
    # review_text stored here is the COMBINED text (title + body) -- what was
    # actually scored, matching drift.py's word-count expectations -- not the
    # raw pre-combine `texts` input above.
    for title, combined_text, result in zip(titles_for_rows, review_texts, results):
        db.insert_prediction(
            review_title=title,
            review_text=combined_text,
            sentiment=result["sentiment"],
            probability=result["probability"],
            client_ip=client_ip,
            user_agent=user_agent,
            api_key_id=api_key_id,
            received_at=received_at,
            responded_at=responded_at,
            model_version=model_version,
            run_id=run_id,
        )

    return results[0] if payload.texts is None else results


@app.post("/admin/reload-model", dependencies=[Depends(require_api_key)])
def reload_model():
    """
    Re-reads models/active/ into memory without restarting the process --
    called automatically by `set_active_model.py --reload-api` after it
    activates a new version, so "set the active version" and "make the API
    actually use it" can be one command instead of two.
    """
    _load_active_model()
    _set_active_model_version_gauge()
    manifest = json.loads(ACTIVE_MODEL_MANIFEST.read_text()) if ACTIVE_MODEL_MANIFEST.exists() else None
    app_logger.info(f"Model reloaded: {manifest}")
    return {"status": "reloaded", "active_model": manifest}


@app.get("/admin/drift-report", dependencies=[Depends(require_api_key)])
def drift_report():
    """
    Computes a fresh drift check right now, against the last
    DRIFT_WINDOW_SECONDS of app.predictions (see src/api/drift.py), saves
    the result to app.drift_checks, and returns it. No internal loop -- an
    external scheduler (EventBridge Scheduler, every 2 min in the cloud)
    is what calls this on a cadence; Grafana then reads app.drift_checks on
    demand whenever the dashboard is viewed, not this endpoint directly.
    """
    report = drift.run_drift_check(DRIFT_WINDOW_SECONDS)
    db.insert_drift_check(report)
    for feature, score in report.get("metrics", {}).items():
        DRIFT_PSI.labels(feature=feature).set(score)
    if report.get("status") in ("warning", "alert"):
        app_logger.warning(f"Model drift detected: {report}")
    return report
