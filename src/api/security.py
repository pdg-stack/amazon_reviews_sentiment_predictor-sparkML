"""
API-key auth dependency for the FastAPI app. Checks the X-API-Key header
against either API_KEYS (a comma-separated pool, for simulating many
distinct callers -- see src/scripts/generate_api_key.py --count) or the
single-key API_KEY, both set by generate_api_key.py via a local .env file,
loaded through python-dotenv at app startup.
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    pool = os.environ.get("API_KEYS")
    expected = {k for k in pool.split(",") if k} if pool else None
    if not expected:
        single = os.environ.get("API_KEY")
        expected = {single} if single else None
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server has no API_KEY/API_KEYS configured -- run src/scripts/generate_api_key.py.",
        )
    if api_key not in expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
