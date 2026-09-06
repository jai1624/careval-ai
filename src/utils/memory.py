"""
Deterministic task memory: merging, deduplication, and running totals.

This module owns ALL state logic. Gemini never decides what counts as
a duplicate, never merges tasks, and never updates the running total --
that is 100% Python here.

Dedup rule: two tasks are considered "the same task" only if they match
on (normalized_task, category) AND have meaningfully similar wording in
raw_text. We deliberately do NOT dedup on normalized_task alone, since
"made breakfast" and "made lunch" would otherwise collide if a naive
implementation ignored category/context.

Correction rule: if a task judged to be the same is mentioned again
with a different duration, we treat it as a correction and update the
duration in place rather than adding a duplicate entry.
"""
from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher

from src.graph.state import ExtractedTask, LoggedTask

SIMILARITY_THRESHOLD = 0.72


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_for_compare(a), _normalize_for_compare(b)).ratio()


def _is_same_task(existing: LoggedTask, incoming: ExtractedTask) -> bool:
    if existing.category != incoming.category:
        return False
    # Compare on normalized_task + a slice of raw_text context, never
    # normalized_task alone, so "made breakfast" vs "made lunch" (same
    # category, different context) are correctly treated as distinct.
    norm_sim = _similarity(existing.normalized_task, incoming.normalized_task)
    raw_sim = _similarity(existing.raw_text, incoming.raw_text)
    combined = max(norm_sim, raw_sim)
    return combined >= SIMILARITY_THRESHOLD


def merge_task_into_day(day_tasks: list[LoggedTask], incoming: ExtractedTask) -> tuple[list[LoggedTask], str]:
    """Merge one incoming task into the day's task list.

    Returns (updated_task_list, action) where action is one of:
    "added", "duration_updated", "duplicate_ignored".
    """
    for i, existing in enumerate(day_tasks):
        if _is_same_task(existing, incoming):
            if existing.estimated_minutes != incoming.estimated_minutes:
                updated = existing.model_copy(
                    update={
                        "estimated_minutes": incoming.estimated_minutes,
                        "times_mentioned": existing.times_mentioned + 1,
                    }
                )
                day_tasks[i] = updated
                return day_tasks, "duration_updated"
            else:
                updated = existing.model_copy(update={"times_mentioned": existing.times_mentioned + 1})
                day_tasks[i] = updated
                return day_tasks, "duplicate_ignored"

    new_task = LoggedTask(
        task_id=str(uuid.uuid4()),
        raw_text=incoming.raw_text,
        normalized_task=incoming.normalized_task,
        estimated_minutes=incoming.estimated_minutes,
        category=incoming.category,
        times_mentioned=1,
    )
    day_tasks.append(new_task)
    return day_tasks, "added"


def merge_many(day_tasks: list[LoggedTask], incoming_tasks: list[ExtractedTask]) -> tuple[list[LoggedTask], list[str]]:
    actions = []
    for task in incoming_tasks:
        day_tasks, action = merge_task_into_day(day_tasks, task)
        actions.append(action)
    return day_tasks, actions


def total_minutes(day_tasks: list[LoggedTask]) -> int:
    """Real clock-time total. This deliberately does NOT attempt to
    account for concurrency/overlap between tasks -- it reports the
    sum of task-equivalent effort, which the UI must label clearly as
    distinct from elapsed clock time."""
    return sum(t.estimated_minutes for t in day_tasks)


def classified_minutes(day_tasks: list[LoggedTask]) -> int:
    """Minutes belonging to classified (valuable) tasks only, excluding
    'unclassified' tasks from any monetary valuation."""
    return sum(t.estimated_minutes for t in day_tasks if t.category != "unclassified")
