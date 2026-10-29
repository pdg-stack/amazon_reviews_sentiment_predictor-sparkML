"""
On-demand drift check for live /predict traffic (see docs/OBSERVABILITY.md
for the full design). Pure Python/numpy plus one DB query -- no Spark -- so
it's cheap to run and easy to unit test in isolation (tests/test_drift.py,
with src.api.db mocked/stubbed).

The mechanism, in short:
  1. evaluate.py computes a "baseline" histogram (prediction confidence +
     review length) from the held-out test set and writes it to
     docs/reports/version_N/baseline_stats.json.
  2. set_active_model.py copies that file to models/active/BASELINE_STATS.json
     when it activates version N, so the baseline on disk always matches
     whichever model is actually serving traffic.
  3. run_drift_check() -- called on demand by GET /admin/drift-report, in
     turn triggered externally on a schedule (EventBridge Scheduler, every
     2 min) rather than by an internal loop -- builds the SAME kind of
     histogram from recent rows in the `app.predictions` table (the last
     window_seconds) and compares the two with the Population Stability
     Index (PSI), a standard, dependency-light way to ask "does live traffic
     look like training data?". The caller (app.py) persists the result to
     `app.drift_checks`; this module only computes it.

This is a proxy for "did the input/output distribution shift", not a
measurement of prediction *accuracy* -- there's no ground truth for live
traffic to compare against. See docs/OBSERVABILITY.md for that caveat and
for why low-traffic windows will often report "insufficient_data" rather
than a real verdict.
"""

from __future__ import annotations  # `X | None` annotations need this on Python <3.10 (EMR Serverless's base image)

import json
from datetime import datetime

import numpy as np

from src.api import db
from src.config import ACTIVE_BASELINE_PATH

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
    Rows from `app.predictions` (via src.api.db) in the last window_seconds
    -- was a log-file tail before Phase 2/3; now that structured records
    live in Postgres, querying is both simpler and doesn't require log
    rotation to stay out of the way.
    """
    return db.query_recent_predictions(window_seconds)


def _classify(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "alert"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "ok"


def run_drift_check(window_seconds: int = 120) -> dict:
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

    # One row per prediction (not per API call -- a batch /predict request
    # writes one predictions row per text scored, see app.py), already flat.
    confidences = [row["probability"] for row in recent if row.get("probability") is not None]
    lengths = [row["review_length"] for row in recent if row.get("review_length") is not None]

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
