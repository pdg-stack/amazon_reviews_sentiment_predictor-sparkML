"""
API-key auth dependency for the FastAPI app. Checks the X-API-Key header
against the API_KEY environment variable (set by src/scripts/generate_api_key.py
via a local .env file, loaded through python-dotenv at app startup).
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server has no API_KEY configured -- run src/scripts/generate_api_key.py.",
        )
    if api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
