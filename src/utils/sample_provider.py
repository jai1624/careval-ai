"""
Sample-day provider.

Loads the three precomputed demo days from sample_days.json and turns
them directly into LoggedTask objects -- no Gemini calls, no
classification calls (activity codes are already pre-assigned in the
JSON), and no GCP calls. This lets the whole results/valuation/
comparison flow be demoed with zero credentials.
"""
from __future__ import annotations

import uuid

from src.config import config
from src.graph.state import LoggedTask
from src.utils.data_loader import load_sample_days, valid_activity_codes


def list_sample_days() -> list[dict]:
    return load_sample_days(str(config.sample_days_path))


def build_sample_day_tasks(sample_id: str) -> tuple[list[LoggedTask], str | None]:
    """Returns (tasks, user_gender) for the given sample day id."""
    samples = list_sample_days()
    valid_codes = valid_activity_codes(str(config.activity_codes_path))

    match = next((s for s in samples if s.get("id") == sample_id), None)
    if match is None:
        return [], None

    tasks: list[LoggedTask] = []
    for t in match.get("tasks", []):
        code = t.get("activity_code", "unclassified")
        if code not in valid_codes:
            code = "unclassified"
        tasks.append(
            LoggedTask(
                task_id=str(uuid.uuid4()),
                raw_text=t.get("raw_text", ""),
                normalized_task=t.get("normalized_task", ""),
                estimated_minutes=int(t.get("estimated_minutes", 15)),
                category=code,
                times_mentioned=1,
            )
        )
    return tasks, match.get("user_gender")
