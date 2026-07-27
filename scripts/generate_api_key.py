"""
Generates the API's secret key, on demand. Never called automatically by
any other script -- run this once before starting the API for the first
time (or whenever you want to rotate the key):

    python scripts/generate_api_key.py

Writes API_KEY=<value> into a local .env (gitignored) and prints the key
once so you can copy it into whatever client calls the API (curl, Postman).
"""

import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    if ENV_PATH.exists() and "API_KEY=" in ENV_PATH.read_text():
        response = input(f"{ENV_PATH} already has an API_KEY. Overwrite with a new one? [y/N] ")
        if response.strip().lower() != "y":
            print("Left the existing key untouched.")
            return

    key = secrets.token_urlsafe(32)
    ENV_PATH.write_text(f"API_KEY={key}\n")
    print(f"Wrote a new API key to {ENV_PATH}\n")
    print(f"API_KEY={key}")
    print(
        "\nCopy this into your client (curl -H, Postman's {{apiKey}} variable, etc.) "
        "-- it will not be shown again."
    )


if __name__ == "__main__":
    main()
