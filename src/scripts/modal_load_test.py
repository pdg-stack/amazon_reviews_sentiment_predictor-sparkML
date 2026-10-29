"""
Modal load generator for the live Phase 2 API (docs/CLOUD_DEPLOYMENT_PLAN.md's
Phase 2 spec): a 20-minute bell-curve of simulated sessions peaking at 1000
concurrent, each drawing review text from the held-out test.parquet pool
(data/scripts/make_training_sample.py never touches this file -- it's
reserved for inference/load-testing, never training) so the live traffic the
API sees is realistic, not synthetic filler.

Session shape, matching the plan's own math: an initial request of 10-20
rows, then a 5-row batch every 1 minute for a 4-minute session (~30 rows,
4 requests/session). ~1000 concurrent lightweight async HTTP sessions is
I/O-bound, cheap work -- this all runs as ONE asyncio event loop inside a
single Modal container, not a distributed fan-out across many containers.

Each session is assigned one API key from a pool (generate_api_key.py
--count N) rather than everyone sharing one key, so request logs/predictions
reflect many distinct simulated callers, not one. Keys are passed in as a
Modal Secret built from the local .env at launch time -- never baked into
the image, never printed.

Run with:
    modal run src/scripts/modal_load_test.py
"""

import json
import math
import os
import random
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_POOL_PATH = PROJECT_ROOT / "data" / "processed" / "loadtest_pool.json"

API_URL = "https://dl3uvj6jcn78t.cloudfront.net/predict"
TOTAL_MINUTES = 20
PEAK_CONCURRENCY = 1000
SESSION_MINUTES = 4
BELL_CENTER_MIN = TOTAL_MINUTES / 2
BELL_SIGMA_MIN = 3.5

app = modal.App("arsp-load-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx")
    .add_local_file(str(DATA_POOL_PATH), "/data/loadtest_pool.json")
)


def _load_api_keys_secret() -> modal.Secret:
    # Read directly from the local .env, never printed -- API_KEYS (the
    # pool) if present, else the single API_KEY repeated (works, just
    # models every session as the same caller).
    #
    # This whole module gets re-imported inside the remote container too
    # (Modal reconstructs the function reference from source there), where
    # .env obviously doesn't exist -- confirmed by testing: it crashed with
    # FileNotFoundError before the function ever ran. The secret's real
    # values were already registered from THIS local evaluation by the time
    # that remote re-import happens, so returning a harmless placeholder
    # there (rather than the real file contents) doesn't lose anything.
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return modal.Secret.from_dict({"API_KEYS": "placeholder-remote-reimport-only"})

    values = {}
    for line in env_path.read_text().splitlines():
        if line.startswith("API_KEYS=") or line.startswith("API_KEY="):
            name, _, value = line.partition("=")
            values[name] = value
    if "API_KEYS" not in values and "API_KEY" not in values:
        raise RuntimeError("No API_KEY/API_KEYS found in .env -- run generate_api_key.py first.")
    return modal.Secret.from_dict(values)


def _bell_curve_new_sessions_per_minute() -> list[int]:
    peak_rate = PEAK_CONCURRENCY / SESSION_MINUTES  # Little's Law: concurrency = arrival_rate * session_length
    counts = []
    for minute in range(TOTAL_MINUTES):
        rate = peak_rate * math.exp(-((minute - BELL_CENTER_MIN) ** 2) / (2 * BELL_SIGMA_MIN**2))
        counts.append(round(rate))
    return counts


@app.function(image=image, timeout=TOTAL_MINUTES * 60 + SESSION_MINUTES * 60 + 120, secrets=[_load_api_keys_secret()])
async def generate_load():
    import asyncio

    import httpx

    with open("/data/loadtest_pool.json") as f:
        pool = json.load(f)  # list of [title, text] pairs

    keys_raw = os.environ.get("API_KEYS") or os.environ.get("API_KEY")
    api_keys = [k for k in keys_raw.split(",") if k]

    stats = {"requests_sent": 0, "requests_ok": 0, "requests_failed": 0, "sessions_started": 0}

    limits = httpx.Limits(max_connections=1500, max_keepalive_connections=400)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:

        async def send_batch(api_key: str, n: int) -> None:
            picks = random.sample(pool, n)
            titles = [p[0] for p in picks]
            texts = [p[1] for p in picks]
            stats["requests_sent"] += 1
            try:
                resp = await client.post(
                    API_URL,
                    headers={"X-API-Key": api_key},
                    json={"titles": titles, "texts": texts},
                )
                if resp.status_code == 200:
                    stats["requests_ok"] += 1
                else:
                    stats["requests_failed"] += 1
            except httpx.HTTPError:
                stats["requests_failed"] += 1

        async def run_session() -> None:
            stats["sessions_started"] += 1
            api_key = random.choice(api_keys)
            await send_batch(api_key, random.randint(10, 20))
            for _ in range(SESSION_MINUTES - 1):
                await asyncio.sleep(60)
                await send_batch(api_key, 5)

        new_sessions_per_minute = _bell_curve_new_sessions_per_minute()
        print(f"Bell curve (new sessions/min): {new_sessions_per_minute}")

        tasks = []
        start = time.monotonic()
        for minute, n_new in enumerate(new_sessions_per_minute):
            # Spread this minute's new sessions evenly across the minute
            # instead of bursting them all at once -- a smoother ramp.
            for i in range(n_new):
                delay = (i / max(n_new, 1)) * 60
                tasks.append(asyncio.create_task(_delayed_session(delay, run_session)))
            elapsed = time.monotonic() - start
            target = (minute + 1) * 60
            if elapsed < target:
                await asyncio.sleep(target - elapsed)
            print(f"minute {minute + 1}/{TOTAL_MINUTES}: {stats}")

        # Let in-flight sessions (up to SESSION_MINUTES more) finish.
        await asyncio.gather(*tasks)

    print(f"DONE: {stats}")
    return stats


async def _delayed_session(delay: float, session_fn) -> None:
    import asyncio

    await asyncio.sleep(delay)
    await session_fn()


@app.local_entrypoint()
def main():
    result = generate_load.remote()
    print(f"Final stats: {result}")
