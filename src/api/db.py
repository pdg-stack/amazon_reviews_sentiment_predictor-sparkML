"""
Connection pool + writes/queries against the `app` database (RDS Postgres,
Phase 2/3 -- predictions, drift_checks). See infra/app_db.tf for how it's
provisioned and docs/ for the DDL that creates its tables.

Every function here is a no-op (write) or returns an empty result (query)
when APP_DB_HOST isn't set -- i.e. local dev, with no RDS configured, works
exactly as before this module existed. Only a cloud deployment (.env.aws)
actually talks to Postgres.
"""

import datetime
from contextlib import contextmanager
from typing import Optional

from src.config import APP_DB_HOST, APP_DB_NAME, APP_DB_PASSWORD, APP_DB_PORT, APP_DB_USER

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        import psycopg2.pool

        _pool = psycopg2.pool.SimpleConnectionPool(
            1,
            10,
            host=APP_DB_HOST,
            port=APP_DB_PORT,
            dbname=APP_DB_NAME,
            user=APP_DB_USER,
            password=APP_DB_PASSWORD,
        )
    return _pool


def is_configured() -> bool:
    return APP_DB_HOST is not None


@contextmanager
def _cursor():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        pool.putconn(conn)


def insert_prediction(
    review_title: Optional[str],
    review_text: str,
    sentiment: str,
    probability: float,
    client_ip: Optional[str],
    user_agent: Optional[str],
    api_key_id: Optional[str],
    received_at: datetime.datetime,
    responded_at: datetime.datetime,
    model_version: Optional[int],
    run_id: Optional[str],
) -> None:
    if not is_configured():
        return
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (review_title, review_text, sentiment, probability, client_ip,
                 user_agent, api_key_id, received_at, responded_at, model_version, run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                review_title,
                review_text,
                sentiment,
                probability,
                client_ip,
                user_agent,
                api_key_id,
                received_at,
                responded_at,
                model_version,
                run_id,
            ),
        )


def query_recent_predictions(window_seconds: int) -> list[dict]:
    """
    Rows from the last window_seconds, shaped for src/api/drift.py:
    {"probability": float, "review_length": int} per row -- review length is
    computed here (word count of review_text) rather than stored as its own
    column, since it's fully derived from review_text and would otherwise
    just be redundant storage (see the predictions table's own DDL comment).
    """
    if not is_configured():
        return []
    with _cursor() as cur:
        cur.execute(
            """
            SELECT probability, array_length(regexp_split_to_array(review_text, '\\s+'), 1) AS review_length
            FROM predictions
            WHERE received_at >= now() - (%s || ' seconds')::interval
            """,
            (window_seconds,),
        )
        return [{"probability": row[0], "review_length": row[1]} for row in cur.fetchall()]


def insert_drift_check(report: dict) -> None:
    if not is_configured():
        return
    metrics = report.get("metrics", {})
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_checks
                (status, sample_size, window_seconds, prediction_confidence_psi, review_length_psi)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                report.get("status"),
                report.get("sample_size"),
                report.get("window_seconds"),
                metrics.get("prediction_confidence_psi"),
                metrics.get("review_length_psi"),
            ),
        )
