"""
Generates the API's secret key(s), on demand. Never called automatically by
any other script -- run this once before starting the API for the first
time (or whenever you want to rotate the key):

    python src/scripts/generate_api_key.py

Writes API_KEY=<value> into a local .env (gitignored) and prints the key
once so you can copy it into whatever client calls the API (curl, Postman).

--count N generates a pool of N keys instead of one -- for a caller that
needs to simulate many distinct API consumers (e.g. the Modal load generator
assigning one key per simulated session), not normal single-client use:

    python src/scripts/generate_api_key.py --count 200

Writes API_KEYS=<comma-separated> instead of API_KEY -- src/api/security.py
checks a request's key against this pool when it's set, falling back to the
single API_KEY check otherwise.
"""

import argparse
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def main(count: int) -> None:
    var_name = "API_KEYS" if count > 1 else "API_KEY"
    existing_lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []

    if any(line.startswith(f"{var_name}=") for line in existing_lines):
        response = input(f"{ENV_PATH} already has an {var_name}. Overwrite with new ones? [y/N] ")
        if response.strip().lower() != "y":
            print("Left the existing key(s) untouched.")
            return

    keys = [secrets.token_urlsafe(32) for _ in range(count)]
    value = ",".join(keys)

    # Replace the existing var_name line in place, or append it -- .env also
    # holds unrelated vars (Modal credentials) that a full-file overwrite
    # would otherwise destroy.
    new_line = f"{var_name}={value}"
    if any(line.startswith(f"{var_name}=") for line in existing_lines):
        existing_lines = [new_line if line.startswith(f"{var_name}=") else line for line in existing_lines]
    else:
        existing_lines.append(new_line)
    ENV_PATH.write_text("\n".join(existing_lines) + "\n")
    print(f"Wrote {count} key(s) to {ENV_PATH} as {var_name}\n")
    print(f"{var_name}={value}")
    print(
        "\nCopy this into your client (curl -H, Postman's {{apiKey}} variable, etc.) "
        "-- it will not be shown again."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="Number of keys to generate (default: 1).")
    args = parser.parse_args()
    main(count=args.count)
