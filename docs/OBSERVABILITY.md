# Observability

How the API's logging, metrics, and automated drift check work — an addition on top of the training and serving pipeline described in [`ARCHITECTURE_NOTES.md`](ARCHITECTURE_NOTES.md).

## Logging

The API writes a log entry for every request it handles. Routine liveness checks and metrics scrapes go to their own file, `logs/api_probes.log`, so they don't drown out real traffic; everything else — including failed requests — goes to `logs/api_requests.log`. Both files are written by a `RotatingFileHandler` (`src/api/logging_config.py`, 10MB cap, 5 backups kept) so neither can grow without limit. All of this happens in one place, the `log_requests` middleware in `src/api/app.py`, so both successful and failed requests are covered the same way.

**`logs/api_requests.log`** — one JSON line per entry:

```json
{"timestamp": "...", "event": "access", "method": "POST", "path": "/predict", "status": 401, "latency_ms": 3.1, "input": {"text": "..."}}
```

`event` is `"access"` for most entries, or `"predict"` for a *successful* `/predict` call, which carries more detail:

```json
{
  "timestamp": "...", "event": "predict", "method": "POST", "path": "/predict", "status": 200, "latency_ms": 812.4,
  "input": {"text": [...], "title": [...]},
  "results": [{"sentiment": "positive", "probability": 0.87}],
  "review_text_lengths": [14]
}
```

`review_text_lengths` is the word count of the combined `title + text` string the model actually scored (matching how `preprocess.py` builds that field at training time) — this is what the drift check compares against the training-set baseline, not the raw input.

A **failed** `/predict` call (bad payload → 400, wrong key → 401, or a server-side error → 500) still records what was sent, under `input`, even though there's no `results` to report — useful for seeing what a client sent that caused the failure. This works even for a bad-key request, which never reaches the code that normally builds that detail, because the request body is read once, up front, in the middleware itself, before anything else runs.

**`logs/api_probes.log`** — a lighter entry for every `GET /health` and `GET /metrics` hit:

```json
{"timestamp": "...", "method": "GET", "path": "/metrics", "status": 200, "latency_ms": 0.4}
```

Three loggers exist in `src/api/logging_config.py`: `api.requests` (the first file above), `api.probes` (the second), and `api.app` (printed to the console — startup messages, drift-check warnings and errors).

### Browsing the logs

For a quick live view, [Logdy](https://logdy.dev) (open source, self-hosted) can tail either file straight into a browser with no setup. For actual analysis — charts of request volume, status codes, and response times over time, not just a live feed — the `log-viewer` service in `docker-compose.yml` runs [GoAccess](https://goaccess.io) (also open source) against `logs/api_requests.log` and serves a dashboard at `http://localhost:8081`. See `docs/DOCKER_COMPOSE.md` for how to start it. GoAccess expects a well-known access-log format, so a small script (`src/scripts/tail_to_clf.py`) converts this project's JSON lines into that format on the fly — the log file itself doesn't need to change.

## Metrics

`GET /metrics` (Prometheus's plain-text format, via `prometheus-fastapi-instrumentator`, no authentication needed — same as `/health`) exposes:

- Request counts and response-time histograms per endpoint, method, and status code, added automatically by the instrumentator.
- `active_model_version` — which model version is currently live, read from `models/active/ACTIVE_MODEL.json` at startup and again on every `/admin/reload-model` call.
- `prediction_drift_psi{feature=...}` — the latest drift score per signal (explained below), updated each time the background check runs.

**One limitation worth knowing**: these numbers are held in the process's own memory. If this API were ever run as multiple worker processes instead of the single one it uses today, each worker would report its own numbers rather than a combined total — not a concern for the current setup, just worth knowing if that ever changes.

## Model drift

There's no way to check whether live predictions are *correct* — nobody labels incoming reviews in real time. What this measures instead is whether the *shape* of live traffic (how confident the model tends to be, how long the reviews tend to be) still resembles the shape of the data the model was evaluated on. A shift is a signal worth a human look, not proof that the model is wrong.

**How it works:**

1. `evaluate.py` builds a baseline from the held-out test set: a distribution of (a) how confident the model was in its predicted class — the same number already logged as `results[].probability` above — and (b) how long each review's combined text was, in words. Both are written to `docs/reports/version_N/baseline_stats.json`.
2. `set_active_model.py --version N` copies that file to `models/active/BASELINE_STATS.json` whenever it activates version N, so the baseline on disk always matches whichever model is actually serving requests. If a version was registered before this baseline-capture code existed, activation prints a warning and drift checks report `"unavailable"` until that version is re-evaluated.
3. A background task inside the API (`src/api/drift.py`, started when the API starts) checks in regularly — every 15 minutes by default — looking at the last hour of successful `/predict` calls and comparing their confidence and length distributions against the baseline, using a statistic called the **Population Stability Index (PSI)**, which measures how much a distribution has shifted from a reference one. A higher PSI means a bigger shift.

**Reading a report** (`GET /admin/drift-report`, needs the `X-API-Key` header):

```json
{
  "status": "ok",
  "sample_size": 142,
  "window_seconds": 3600,
  "checked_at": "...",
  "metrics": {"prediction_confidence_psi": 0.04, "review_length_psi": 0.11}
}
```

`status` follows the standard convention for reading a PSI score: under 0.1 is `"ok"`, 0.1 to 0.25 is `"warning"`, and 0.25 or above is `"alert"` — whichever of the two signals scores worse decides the overall status. Two other statuses can appear before any PSI is even computed: `"unavailable"` (no baseline exists yet for the active model — step 2 above hasn't run) and `"insufficient_data"` (fewer than 30 predictions in the time window — a PSI score computed on a handful of samples is mostly noise).

**Worth knowing**: this is built for a low-traffic, single-developer setup. Checking every 15 minutes over a 1-hour window, and needing at least 30 samples to say anything, means real usage will often land on `"insufficient_data"` rather than a real verdict — that's expected, not a bug. Treat any `"warning"` or `"alert"` you do see as illustrative rather than statistically rigorous, and don't read too much into an `"ok"` result from a tiny sample either.

"Alerting" here just means a log line at `WARNING` level (via the `api.app` logger) plus the non-`"ok"` status above — there's no email or chat notification built in. The Prometheus gauge could be connected to an alerting tool later if this ever needs to notify someone directly, but that isn't built now.

## What this doesn't do

- **It doesn't track accuracy.** Nothing here can tell you whether predictions are *correct* — that needs labeled ground truth on live traffic, a separate and much heavier process this project doesn't build.
- **It doesn't cover batch scoring.** `predict.py`, the command-line batch tool, doesn't write to either log file and isn't part of the drift check — this is entirely about the live `/predict` endpoint.
- **There's no dashboard beyond what's described above.** No permanent metrics-storage server is run by this project (see the scope note in `ARCHITECTURE_NOTES.md`); `/metrics` is ready to be scraped by one if you add it yourself.
