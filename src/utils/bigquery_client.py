"""
Optional BigQuery persistence layer.

BigQuery is NEVER a blocker. Every public function here catches all
exceptions, logs them via structured logging, and returns a simple
success flag. The rest of the app must be able to run identically
whether or not this succeeds.

Uses Application Default Credentials (ADC) only -- no service-account
JSON files are read or required.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from src.config import config

logger = logging.getLogger("careval.bigquery")

_SCHEMA_DESCRIPTION = [
    {"name": "log_id", "type": "STRING"},
    {"name": "logged_at", "type": "TIMESTAMP"},
    {"name": "user_gender", "type": "STRING"},
    {"name": "total_minutes", "type": "INTEGER"},
    {"name": "daily_value_inr", "type": "FLOAT"},
    {"name": "tasks_json", "type": "STRING"},
]


class BigQueryUnavailable(Exception):
    pass


def _get_client():
    """Lazily imports and constructs a BigQuery client using ADC.
    Returns None (never raises) if unavailable for any reason."""
    if not config.is_gcp:
        return None
    try:
        from google.cloud import bigquery

        return bigquery.Client(project=config.gcp_project_id)
    except Exception:
        logger.exception("BigQuery client construction failed; continuing without persistence")
        return None


def ensure_table_exists() -> bool:
    """Best-effort table creation. Returns True on success, False on
    any failure -- callers must not treat False as fatal."""
    client = _get_client()
    if client is None:
        return False
    try:
        from google.cloud import bigquery

        dataset_ref = f"{config.gcp_project_id}.{config.bigquery_dataset}"
        table_ref = f"{dataset_ref}.{config.bigquery_table}"

        try:
            client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            client.create_dataset(dataset, exists_ok=True)

        schema = [bigquery.SchemaField(f["name"], f["type"]) for f in _SCHEMA_DESCRIPTION]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table, exists_ok=True)
        return True
    except Exception:
        logger.exception("Failed to ensure BigQuery table exists; continuing gracefully")
        return False


def persist_day_log(user_gender: str | None, total_minutes: int, daily_value_inr: float, tasks: list[dict]) -> bool:
    """Writes one row summarizing a day's logged tasks. Fires ONLY when
    the caller has confirmed user consent -- this function assumes that
    check has already happened upstream. On any failure (missing creds,
    auth, network, schema mismatch) this logs and returns False without
    raising, so it can never crash the app."""
    client = _get_client()
    if client is None:
        logger.info("BigQuery not configured/available; skipping persistence for this session")
        return False

    try:
        if not ensure_table_exists():
            return False

        table_ref = f"{config.gcp_project_id}.{config.bigquery_dataset}.{config.bigquery_table}"
        row = {
            "log_id": _new_id(),
            "logged_at": dt.datetime.utcnow().isoformat(),
            "user_gender": user_gender or "unspecified",
            "total_minutes": total_minutes,
            "daily_value_inr": daily_value_inr,
            "tasks_json": json.dumps(tasks, default=str),
        }
        errors = client.insert_rows_json(table_ref, [row])
        if errors:
            logger.error("BigQuery insert reported errors: %s", errors)
            return False
        return True
    except Exception:
        logger.exception("BigQuery persistence failed; continuing without crashing")
        return False


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
