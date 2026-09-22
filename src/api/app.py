"""
FastAPI real-time serving layer.

    POST /predict             (requires X-API-Key) -> {"sentiment": ..., "probability": ...}
    POST /admin/reload-model  (requires X-API-Key) -> reloads models/active/ into memory, no restart needed
    GET  /admin/drift-report  (requires X-API-Key) -> latest automated drift check (see src/api/drift.py)
    GET  /health               (no auth)            -> {"status": "ok"}
    GET  /metrics               (no auth)            -> Prometheus text exposition

Loads models/active/ (set by set_active_model.py) at startup and keeps the
SparkSession resident for the life of the process -- fine for demo/dev
latency, not a high-throughput production pattern (see docs/ARCHITECTURE_NOTES.md).

Every request is logged as one JSON line, via the log_requests middleware
below. GET /health and GET /metrics (liveness checks / Prometheus scrapes)
go to logs/api_probes.log; everything else goes to logs/api_requests.log.
/predict calls carry extra detail (input/results/lengths) and are tagged
"event": "predict" so src/api/drift.py can read them back; a failed
/predict call (400/401) still logs the input that was sent, just without
results. A background task (started at startup) periodically compares
recent /predict traffic against a baseline captured at evaluation time --
see docs/OBSERVABILITY.md for the full design.

The model expects a `review_text` field (title + body combined, see
preprocess.py/docs/model_plan.md) -- /predict accepts an optional `title`/`titles`
alongside `text`/`texts` and combines them the same way, so a request run
through this endpoint sees exactly what the model was trained on.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
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

from src.api import drift
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

# Env-configurable so a low-traffic dev instance and a busier one can each
# pick a sensible cadence/window without code changes -- see docs/OBSERVABILITY.md.
DRIFT_CHECK_INTERVAL_SECONDS = int(os.environ.get("DRIFT_CHECK_INTERVAL_SECONDS", "900"))
DRIFT_WINDOW_SECONDS = int(os.environ.get("DRIFT_WINDOW_SECONDS", "3600"))

_spark = None
_model = None
_latest_drift_report: dict = {"status": "not_yet_computed"}
_drift_task: Optional["asyncio.Task"] = None


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


async def _run_drift_check_and_record():
    global _latest_drift_report
    try:
        # run_drift_check() does synchronous file I/O (tailing the request
        # log); offloaded to a thread so it never blocks the event loop that
        # concurrent /predict requests are running on.
        report = await asyncio.to_thread(drift.run_drift_check, DRIFT_WINDOW_SECONDS)
    except Exception:
        app_logger.exception("Drift check failed")
        return
    _latest_drift_report = report
    for feature, score in report.get("metrics", {}).items():
        DRIFT_PSI.labels(feature=feature).set(score)
    if report.get("status") in ("warning", "alert"):
        app_logger.warning(f"Model drift detected: {report}")


async def _drift_loop():
    await _run_drift_check_and_record()  # first check right away, not after the first interval
    while True:
        await asyncio.sleep(DRIFT_CHECK_INTERVAL_SECONDS)
        await _run_drift_check_and_record()


@app.on_event("startup")
def load_model():
    global _spark
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _spark = get_spark_session("api")
    _load_active_model()
    _set_active_model_version_gauge()
    app_logger.info("Model loaded, Spark session started.")


@app.on_event("startup")
async def start_drift_loop():
    global _drift_task
    _drift_task = asyncio.create_task(_drift_loop())


@app.on_event("shutdown")
async def stop_drift_loop():
    if _drift_task is not None:
        _drift_task.cancel()


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


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: PredictRequest, request: Request) -> Union[dict, List[dict]]:
    if payload.text is None and not payload.texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'.")

    texts = payload.texts if payload.texts is not None else [payload.text]
    titles = payload.titles if payload.titles is not None else ([payload.title] if payload.title is not None else None)
    if titles is not None and len(titles) != len(texts):
        raise HTTPException(status_code=400, detail="'titles' must be the same length as 'texts'.")

    review_texts = _combine(texts, titles)
    results = _score(review_texts)

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
    Latest result from the background drift-check loop (see
    src/api/drift.py and docs/OBSERVABILITY.md) -- not computed on demand,
    just returns whatever the last scheduled check found.
    """
    return _latest_drift_report
