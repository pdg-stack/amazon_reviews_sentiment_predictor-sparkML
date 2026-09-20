"""
Unit tests for src/api/security.py's require_api_key. Called directly as a
plain function (bypassing FastAPI's dependency-injection machinery, which
only matters when a real request comes through Depends()/Security()) -- pure
and Spark-free, like tests/test_drift.py.
"""

import pytest
from fastapi import HTTPException

from src.api.security import require_api_key


def test_require_api_key_raises_500_when_server_has_no_key_configured(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(api_key="anything")
    assert exc_info.value.status_code == 500


def test_require_api_key_raises_401_on_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_require_api_key_raises_401_on_missing_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(api_key=None)
    assert exc_info.value.status_code == 401


def test_require_api_key_passes_on_matching_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-key")
    assert require_api_key(api_key="correct-key") is None
