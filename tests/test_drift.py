"""
Unit tests for src/api/drift.py. Deliberately Spark-free (no SparkSession,
no PySpark import anywhere in this file) -- drift.py is pure Python/numpy by
design specifically so it can be tested this cheaply. See docs/OBSERVABILITY.md.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.api import drift


# --- compute_psi ---------------------------------------------------------


def test_compute_psi_identical_distribution_is_near_zero():
    edges = np.linspace(0.5, 1.0, 11).tolist()
    counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    rng = np.random.default_rng(42)
    # sample drawn from (approximately) the same uniform shape as the baseline
    sample = rng.uniform(0.5, 1.0, size=1000)
    psi = drift.compute_psi(edges, counts, sample)
    assert psi < 0.02


def test_compute_psi_shifted_distribution_is_high():
    edges = np.linspace(0.5, 1.0, 11).tolist()
    counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # uniform baseline
    # every sample value crammed into the single top bin -- as different as it gets
    sample = [0.99] * 500
    psi = drift.compute_psi(edges, counts, sample)
    assert psi > 1.0


def test_compute_psi_empty_sample_returns_zero():
    edges = np.linspace(0.5, 1.0, 11).tolist()
    counts = [10] * 10
    assert drift.compute_psi(edges, counts, []) == 0.0


def test_compute_psi_empty_baseline_returns_zero():
    edges = np.linspace(0.5, 1.0, 11).tolist()
    assert drift.compute_psi(edges, [0] * 10, [0.6, 0.7, 0.8]) == 0.0


def test_compute_psi_handles_deduped_edges():
    # simulates evaluate.py's np.unique(np.quantile(...)) output when many
    # duplicate quantile boundaries collapse to fewer, unevenly spaced bins
    edges = [1.0, 3.0, 3.0, 3.0, 7.0, 12.0]
    deduped_edges = np.unique(edges).tolist()  # -> [1.0, 3.0, 7.0, 12.0]
    counts = [5, 5, 5]  # 3 bins for 4 edges
    psi = drift.compute_psi(deduped_edges, counts, [2, 4, 4, 9, 9, 9])
    assert psi >= 0.0  # just needs to not blow up (inf/NaN) on ragged bins


def test_compute_psi_no_log_of_zero_blowup():
    # a bin that's empty in the baseline but populated live (or vice versa)
    # must not produce inf/NaN -- that's exactly what the epsilon floor is for
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts = [0, 0, 0, 0, 100]  # all baseline mass in the last bin
    sample = [0.55] * 50  # all live mass in the first bin -- the opposite extreme
    psi = drift.compute_psi(edges, counts, sample)
    assert np.isfinite(psi)
    assert psi > 0


# --- load_baseline ---------------------------------------------------------


def test_load_baseline_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "ACTIVE_BASELINE_PATH", tmp_path / "does_not_exist.json")
    assert drift.load_baseline() is None


def test_load_baseline_reads_json(tmp_path, monkeypatch):
    baseline_path = tmp_path / "BASELINE_STATS.json"
    payload = {"confidence_bins": {"edges": [0.5, 1.0], "counts": [1]}}
    baseline_path.write_text(json.dumps(payload))
    monkeypatch.setattr(drift, "ACTIVE_BASELINE_PATH", baseline_path)
    assert drift.load_baseline() == payload


# --- _tail_recent_predictions ---------------------------------------------------------


def _log_line(event: str, timestamp: datetime, **extra) -> str:
    entry = {"timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **extra}
    return json.dumps(entry)


def test_tail_recent_predictions_filters_event_window_and_bad_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "api_requests.log"
    now = datetime.now()
    lines = [
        _log_line("predict", now - timedelta(minutes=5), results=[{"probability": 0.9}]),  # in window, predict -> kept
        _log_line("access", now - timedelta(minutes=5), status=200),  # in window, but not "predict" -> dropped
        _log_line("predict", now - timedelta(hours=2), results=[{"probability": 0.5}]),  # predict, but outside window -> dropped
        "not valid json at all",  # malformed -> skipped, must not raise
        "",  # blank line -> skipped
    ]
    log_path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(drift, "REQUEST_LOG_PATH", log_path)

    kept = drift._tail_recent_predictions(window_seconds=3600)
    assert len(kept) == 1
    assert kept[0]["results"][0]["probability"] == 0.9


def test_tail_recent_predictions_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "REQUEST_LOG_PATH", tmp_path / "nope.log")
    assert drift._tail_recent_predictions(window_seconds=3600) == []


# --- run_drift_check ---------------------------------------------------------


def test_run_drift_check_unavailable_without_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "ACTIVE_BASELINE_PATH", tmp_path / "missing.json")
    report = drift.run_drift_check()
    assert report["status"] == "unavailable"


def test_run_drift_check_insufficient_data_below_min_sample(tmp_path, monkeypatch):
    baseline_path = tmp_path / "BASELINE_STATS.json"
    baseline_path.write_text(
        json.dumps(
            {
                "confidence_bins": {"edges": np.linspace(0.5, 1.0, 11).tolist(), "counts": [1] * 10},
                "length_bins": {"edges": list(range(0, 110, 10)), "counts": [1] * 10},
            }
        )
    )
    monkeypatch.setattr(drift, "ACTIVE_BASELINE_PATH", baseline_path)
    monkeypatch.setattr(drift, "REQUEST_LOG_PATH", tmp_path / "empty.log")  # no entries at all

    report = drift.run_drift_check()
    assert report["status"] == "insufficient_data"
    assert report["sample_size"] == 0
    assert report["required"] == drift.MIN_SAMPLE_SIZE


def test_run_drift_check_ok_when_enough_matching_samples(tmp_path, monkeypatch):
    edges = np.linspace(0.5, 1.0, 11).tolist()
    baseline_path = tmp_path / "BASELINE_STATS.json"
    baseline_path.write_text(
        json.dumps(
            {
                "confidence_bins": {"edges": edges, "counts": [4] * 10},
                "length_bins": {"edges": list(range(0, 110, 10)), "counts": [4] * 10},
            }
        )
    )
    monkeypatch.setattr(drift, "ACTIVE_BASELINE_PATH", baseline_path)

    log_path = tmp_path / "api_requests.log"
    now = datetime.now()
    rng = np.random.default_rng(0)
    lines = []
    for p, length in zip(rng.uniform(0.5, 1.0, size=drift.MIN_SAMPLE_SIZE + 5), rng.integers(1, 100, size=drift.MIN_SAMPLE_SIZE + 5)):
        lines.append(
            _log_line(
                "predict",
                now,
                results=[{"probability": float(p)}],
                review_text_lengths=[int(length)],
            )
        )
    log_path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(drift, "REQUEST_LOG_PATH", log_path)

    report = drift.run_drift_check()
    assert report["status"] in ("ok", "warning", "alert")  # a real verdict, not a short-circuit
    assert report["sample_size"] == drift.MIN_SAMPLE_SIZE + 5
    assert "prediction_confidence_psi" in report["metrics"]
    assert "review_length_psi" in report["metrics"]
