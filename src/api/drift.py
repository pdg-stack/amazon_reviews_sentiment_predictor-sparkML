"""
Automated drift monitoring for live /predict traffic (see docs/OBSERVABILITY.md
for the full design). Pure Python/numpy, no Spark -- this module only ever
reads JSON, so it's cheap to run on a timer and easy to unit test in
isolation (tests/test_drift.py).

The mechanism, in short:
  1. evaluate.py computes a "baseline" histogram (prediction confidence +
     review length) from the held-out test set and writes it to
     reports/version_N/baseline_stats.json.
  2. set_active_model.py copies that file to models/active/BASELINE_STATS.json
     when it activates version N, so the baseline on disk always matches
     whichever model is actually serving traffic.
  3. run_drift_check() (called on a timer by app.py) builds the SAME kind of
     histogram from recent entries in logs/api_requests.log and compares the
     two with the Population Stability Index (PSI) -- a standard, dependency
     -light way to ask "does live traffic look like training data?".

This is a proxy for "did the input/output distribution shift", not a
measurement of prediction *accuracy* -- there's no ground truth for live
traffic to compare against. See docs/OBSERVABILITY.md for that caveat and
for why low-traffic windows will often report "insufficient_data" rather
than a real verdict.
"""

import json
from datetime import datetime, timedelta

import numpy as np

from src.config import ACTIVE_BASELINE_PATH, LOGS_DIR

REQUEST_LOG_PATH = LOGS_DIR / "api_requests.log"

# PSI needs a reasonably sized sample to not just be noise -- below this,
# report "insufficient_data" rather than a misleadingly precise-looking score.
MIN_SAMPLE_SIZE = 30

# Standard PSI convention: <0.1 stable, 0.1-0.25 moderate shift, >=0.25 major shift.
PSI_WARNING_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25

_STATUS_ORDER = ["ok", "warning", "alert"]


def compute_psi(baseline_edges, baseline_counts, sample_values, eps: float = 1e-4) -> float:
    """
    Population Stability Index between a precomputed baseline histogram
    (edges + counts, as written by evaluate.py) and a fresh sample of raw
    values, bucketed into the same edges.

    PSI = sum_over_bins( (actual% - expected%) * ln(actual% / expected%) ).
    An epsilon floor avoids log(0)/divide-by-zero for bins that are empty in
    one distribution but not the other -- without it, a single empty bin
    would make PSI blow up to infinity instead of reporting a large-but-finite
    shift.
    """
    baseline_counts = np.asarray(baseline_counts, dtype=float)
    total_baseline = baseline_counts.sum()
    if total_baseline == 0 or len(baseline_edges) < 2 or len(sample_values) == 0:
        return 0.0

    expected_pct = baseline_counts / total_baseline

    sample_counts, _ = np.histogram(sample_values, bins=baseline_edges)
    total_sample = sample_counts.sum()
    if total_sample == 0:
        return 0.0
    actual_pct = sample_counts / total_sample

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def load_baseline() -> dict | None:
    if not ACTIVE_BASELINE_PATH.exists():
        return None
    return json.loads(ACTIVE_BASELINE_PATH.read_text())


def _tail_recent_predictions(window_seconds: int) -> list[dict]:
    """
    Reads logs/api_requests.log (the CURRENT file only -- rotated backups
    are not consulted, which is fine for any window smaller than "however
    long it takes to fill 10MB") and returns predict-shaped entries whose
    timestamp falls within the last window_seconds.
    """
    if not REQUEST_LOG_PATH.exists():
        return []

    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    rows = []
    with open(REQUEST_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") != "predict":
                continue
            ts_raw = entry.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
            if ts < cutoff:
                continue
            rows.append(entry)
    return rows


def _classify(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "alert"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "ok"


def run_drift_check(window_seconds: int = 3600) -> dict:
    """
    Returns a report dict, always including "status" and "checked_at".

    status:
      - "unavailable"      -- no baseline for the active model yet
      - "insufficient_data" -- fewer than MIN_SAMPLE_SIZE predict calls in the window
      - "ok" / "warning" / "alert" -- the worse of the two PSI scores

    sample_size is always surfaced prominently so "insufficient_data" isn't
    mistaken for a broken check on a low-traffic API.
    """
    checked_at = datetime.now().isoformat()
    baseline = load_baseline()
    if baseline is None:
        return {
            "status": "unavailable",
            "reason": (
                "No baseline_stats.json for the active model -- run "
                "`evaluate.py --version N` for this version, then re-activate it "
                "with set_active_model.py."
            ),
            "checked_at": checked_at,
        }

    recent = _tail_recent_predictions(window_seconds)
    sample_size = len(recent)
    if sample_size < MIN_SAMPLE_SIZE:
        return {
            "status": "insufficient_data",
            "sample_size": sample_size,
            "required": MIN_SAMPLE_SIZE,
            "window_seconds": window_seconds,
            "checked_at": checked_at,
        }

    confidences, lengths = [], []
    for entry in recent:
        for result in entry.get("results", []):
            probability = result.get("probability")
            if probability is not None:
                confidences.append(probability)
        lengths.extend(entry.get("review_text_lengths", []))

    confidence_bins = baseline.get("confidence_bins", {})
    length_bins = baseline.get("length_bins", {})
    psi_confidence = compute_psi(confidence_bins.get("edges", []), confidence_bins.get("counts", []), confidences)
    psi_length = compute_psi(length_bins.get("edges", []), length_bins.get("counts", []), lengths)

    status = max(_classify(psi_confidence), _classify(psi_length), key=_STATUS_ORDER.index)

    return {
        "status": status,
        "sample_size": sample_size,
        "window_seconds": window_seconds,
        "checked_at": checked_at,
        "metrics": {
            "prediction_confidence_psi": round(psi_confidence, 4),
            "review_length_psi": round(psi_length, 4),
        },
    }
