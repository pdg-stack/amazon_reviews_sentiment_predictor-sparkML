"""
FastAPI real-time serving layer.

    POST /predict            (requires X-API-Key) -> {"sentiment": ..., "probability": ...}
    POST /admin/reload-model (requires X-API-Key) -> reloads models/active/ into memory, no restart needed
    GET  /health              (no auth)            -> {"status": "ok"}

Loads models/active/ (set by set_active_model.py) at startup and keeps the
SparkSession resident for the life of the process -- fine for demo/dev
latency, not a high-throughput production pattern (see ARCHITECTURE.md).
Every /predict call is logged as a JSON line to logs/api_requests.log.

The model expects a `review_text` field (title + body combined, see
preprocess.py/model_plan.md) -- /predict accepts an optional `title`/`titles`
alongside `text`/`texts` and combines them the same way, so a request run
through this endpoint sees exactly what the model was trained on.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import time
from typing import List, Optional, Union

import mlflow
import mlflow.spark
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.api.security import require_api_key
from src.config import ACTIVE_MODEL_DIR, LOGS_DIR, MLFLOW_TRACKING_URI, PROJECT_ROOT, get_spark_session

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Amazon Reviews Sentiment Predictor API")

_spark = None
_model = None
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_log_path = LOGS_DIR / "api_requests.log"


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    title: Optional[str] = None
    titles: Optional[List[str]] = None


def _load_active_model():
    global _model
    _model = mlflow.spark.load_model(str(ACTIVE_MODEL_DIR))


@app.on_event("startup")
def load_model():
    global _spark
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _spark = get_spark_session("api")
    _load_active_model()


def _combine(texts: List[str], titles: Optional[List[str]]) -> List[str]:
    if not titles:
        return [t.strip() for t in texts]
    return [f"{title} {text}".strip() if title else text.strip() for title, text in zip(titles, texts)]


def _score(texts: List[str], titles: Optional[List[str]] = None) -> List[dict]:
    review_texts = _combine(texts, titles)
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
def predict(payload: PredictRequest) -> Union[dict, List[dict]]:
    if payload.text is None and not payload.texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'.")

    texts = payload.texts if payload.texts is not None else [payload.text]
    titles = payload.titles if payload.titles is not None else ([payload.title] if payload.title is not None else None)
    if titles is not None and len(titles) != len(texts):
        raise HTTPException(status_code=400, detail="'titles' must be the same length as 'texts'.")

    start = time.time()
    results = _score(texts, titles)
    latency_ms = round((time.time() - start) * 1000, 2)

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {"text": texts, "title": titles},
        "results": results,
        "latency_ms": latency_ms,
    }
    with open(_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

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
    manifest_path = ACTIVE_MODEL_DIR / "ACTIVE_MODEL.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {"status": "reloaded", "active_model": manifest}
