import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.state import ExtractedTask, LoggedTask


def test_negative_duration_is_clamped():
    task = ExtractedTask(raw_text="did stuff", normalized_task="did stuff", estimated_minutes=-15)
    assert task.estimated_minutes > 0


def test_zero_duration_is_clamped():
    task = ExtractedTask(raw_text="did stuff", normalized_task="did stuff", estimated_minutes=0)
    assert task.estimated_minutes > 0


def test_unrealistic_duration_is_clamped():
    task = ExtractedTask(raw_text="did stuff", normalized_task="did stuff", estimated_minutes=99999)
    assert task.estimated_minutes <= 1440


def test_empty_text_fields_get_placeholder():
    task = ExtractedTask(raw_text="   ", normalized_task="", estimated_minutes=20)
    assert task.raw_text != ""
    assert task.normalized_task != ""


def test_logged_task_requires_task_id():
    task = LoggedTask(
        task_id="abc-123",
        raw_text="cooked dinner",
        normalized_task="cooked dinner",
        estimated_minutes=30,
        category="COOK_MEAL",
    )
    assert task.task_id == "abc-123"
    assert task.times_mentioned == 1


def test_malformed_none_duration_defaults_safely():
    task = ExtractedTask(raw_text="x", normalized_task="x", estimated_minutes=None)  # type: ignore
    assert task.estimated_minutes > 0
