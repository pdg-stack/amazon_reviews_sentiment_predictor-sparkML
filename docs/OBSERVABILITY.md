# Observability

How the API's logging, metrics, and automated drift check work -- an addition on top of the training/serving pipeline described in [`ARCHITECTURE_NOTES.md`](ARCHITECTURE_NOTES.md).

## Logging
Every HTTP request the API handles (except `GET /health` and `GET /metrics`, which are excluded so liveness probes and Prometheus scrapes don't flood the file) is logged as one JSON line to `logs/api_requests.log`, written by a `RotatingFileHandler` (`src/api/logging_config.py`, 10MB cap, 5 backups) so the file can't grow unbounded. The write happens in one place -- the `log_requests` middleware in `src/api/app.py` -- which fires for both successful and failed requests, unlike the previous per-handler write.

Every entry has:
```json
{"timestamp": "...", "event": "access", "method": "POST", "path": "/predict", "status": 401, "latency_ms": 3.1}
```
`event` is `"access"` for everything, or `"predict"` for a successful `/predict` call, which carries extra fields:
```json
{
  "timestamp": "...", "event": "predict", "method": "POST", "path": "/predict", "status": 200, "latency_ms": 812.4,
  "input": {"text": [...], "title": [...]},
  "results": [{"sentiment": "positive", "probability": 0.87}],
  "review_text_lengths": [14]
}
```
`review_text_lengths` is the word count of the *combined* `title + text` string the model actually scored (matching what `preprocess.py` builds at training time), not the raw input -- this is what the drift check compares against the training-set baseline.

Two loggers exist (`src/api/logging_config.py`): `api.requests` (the file above) and `api.app` (console -- startup messages, drift-loop warnings/errors).

## Metrics
`GET /metrics` (Prometheus text exposition format, via `prometheus-fastapi-instrumentator`, no auth -- same convention as `/health`) exposes:
- Request counters and latency histograms per endpoint/method/status code, added automatically by the instrumentator.
- `active_model_version` -- a gauge set at startup and on every `/admin/reload-model` call, from `models/active/ACTIVE_MODEL.json`.
- `prediction_drift_psi{feature=...}` -- the latest PSI score per drift signal (see below), updated each time the background check runs.

**Known limitation**: these gauges hold state in-process. If this API is ever run with multiple worker processes (`uvicorn --workers N`, gunicorn, etc.) instead of the current single process, they'd need `prometheus_client`'s multiprocess mode to stay accurate -- not needed today, just noted here so it isn't a surprise later.

## Model drift
There's no ground truth for live traffic -- nobody labels incoming reviews in real time -- so this can't measure prediction *accuracy*. What it measures instead: whether the *shape* of live traffic (how confident the model is, how long the reviews are) still looks like the shape of the data the model was evaluated on. A shift is a signal worth a human look, not proof the model is wrong.

**The pipeline:**
1. `evaluate.py` computes a baseline from the held-out test set: histograms of (a) predicted-class confidence -- `max(P(negative), P(positive))`, the same quantity already logged as `results[].probability` above, bucketed into 20 fixed-width bins over `[0.5, 1.0]` (confidence for a binary classifier can't fall below 0.5, so bucketing over `[0, 1]` would leave half the bins permanently empty) -- and (b) `review_text` word count, bucketed into 20 quantile-based bins (deduplicated, since discrete word counts often produce repeated quantile boundaries). Written to `reports/version_N/baseline_stats.json`.
2. `set_active_model.py --version N` copies that file to `models/active/BASELINE_STATS.json` when it activates version N, so the baseline on disk always matches whichever model is actually serving. If a version was never evaluated with this baseline-capture code, activation prints a warning and drift checks report `"unavailable"` until you re-run `evaluate.py --version N`.
3. A background `asyncio` loop inside the API (`src/api/drift.py`, started at startup) runs once immediately, then every `DRIFT_CHECK_INTERVAL_SECONDS` (env-configurable, default 900s / 15 min): it tails the last `DRIFT_WINDOW_SECONDS` (default 3600s / 1hr) of `"event": "predict"` entries from `logs/api_requests.log`, re-buckets their confidence/length values into the SAME baseline bin edges, and computes the **Population Stability Index (PSI)** between the two distributions for each signal.

**Reading a report** (`GET /admin/drift-report`, `X-API-Key` required):
```json
{
  "status": "ok",
  "sample_size": 142,
  "window_seconds": 3600,
  "checked_at": "...",
  "metrics": {"prediction_confidence_psi": 0.04, "review_length_psi": 0.11}
}
```
`status` is the worse of the two PSI classifications, using the standard convention: PSI < 0.1 -> `"ok"`, 0.1-0.25 -> `"warning"`, >= 0.25 -> `"alert"`. Two other statuses short-circuit before any PSI is computed: `"unavailable"` (no baseline for the active model yet -- step 2 above never ran) and `"insufficient_data"` (fewer than 30 predict calls in the window -- PSI on a handful of samples is just noise).

**Honest limitation**: this is a low-traffic, single-developer dev API. A 15-minute interval over a 1-hour window needing 30+ samples means real usage will often sit at `"insufficient_data"` rather than a real verdict -- that's expected, not a bug. It demonstrates the mechanism; treat any `"warning"`/`"alert"` you do see as illustrative rather than statistically rigorous, and don't read too much into `"ok"` on a tiny sample either.

"Alerting" here means a structured `WARNING`-level log line (via the `api.app` logger) plus the non-`"ok"` status above -- there's no external paging (Slack/email/PagerDuty) built in. The Prometheus gauges could be wired into Alertmanager/Grafana alerting later if this ever needs to page someone, but that's not built now.

## What this doesn't do
- **No accuracy tracking.** Nothing here tells you whether predictions are *correct* -- that needs labeled ground truth on live traffic, which is a separate, heavier, human-in-the-loop process this project doesn't build.
- **No batch-CLI logging.** `predict.py` (batch scoring) doesn't write to `logs/api_requests.log` or feed the drift check -- this is entirely about the live `/predict` endpoint's traffic.
- **No dashboard beyond Prometheus's own text output and the MLflow UI.** No Grafana/Prometheus server is stood up by this project (see the scope note in `ARCHITECTURE_NOTES.md`) -- `/metrics` is scrape-ready for one if you add it yourself.
