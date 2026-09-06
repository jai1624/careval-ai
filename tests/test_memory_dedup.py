import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.state import ExtractedTask, LoggedTask
from src.utils.memory import classified_minutes, merge_task_into_day, total_minutes


def test_new_distinct_task_is_added():
    day_tasks: list[LoggedTask] = []
    incoming = ExtractedTask(raw_text="cooked dinner", normalized_task="cooked dinner", estimated_minutes=30, category="COOK_MEAL")
    day_tasks, action = merge_task_into_day(day_tasks, incoming)
    assert action == "added"
    assert len(day_tasks) == 1


def test_breakfast_and_lunch_not_deduped_despite_same_category():
    day_tasks: list[LoggedTask] = []
    t1 = ExtractedTask(raw_text="made breakfast", normalized_task="made breakfast", estimated_minutes=20, category="COOK_MEAL")
    t2 = ExtractedTask(raw_text="made lunch", normalized_task="made lunch", estimated_minutes=30, category="COOK_MEAL")
    day_tasks, _ = merge_task_into_day(day_tasks, t1)
    day_tasks, action = merge_task_into_day(day_tasks, t2)
    assert action == "added"
    assert len(day_tasks) == 2


def test_repeated_mention_with_same_duration_is_duplicate_ignored():
    day_tasks: list[LoggedTask] = []
    t1 = ExtractedTask(raw_text="cooked dinner", normalized_task="cooked dinner", estimated_minutes=30, category="COOK_MEAL")
    day_tasks, _ = merge_task_into_day(day_tasks, t1)
    day_tasks, action = merge_task_into_day(day_tasks, t1)
    assert action == "duplicate_ignored"
    assert len(day_tasks) == 1
    assert day_tasks[0].times_mentioned == 2


def test_repeated_mention_with_different_duration_updates_in_place():
    day_tasks: list[LoggedTask] = []
    t1 = ExtractedTask(raw_text="cooked dinner", normalized_task="cooked dinner", estimated_minutes=30, category="COOK_MEAL")
    day_tasks, _ = merge_task_into_day(day_tasks, t1)
    t2 = ExtractedTask(raw_text="cooked dinner", normalized_task="cooked dinner", estimated_minutes=45, category="COOK_MEAL")
    day_tasks, action = merge_task_into_day(day_tasks, t2)
    assert action == "duration_updated"
    assert len(day_tasks) == 1
    assert day_tasks[0].estimated_minutes == 45


def test_different_category_same_normalized_text_not_merged():
    day_tasks: list[LoggedTask] = []
    t1 = ExtractedTask(raw_text="helped with homework", normalized_task="helped with homework", estimated_minutes=30, category="HOMEWORK")
    t2 = ExtractedTask(raw_text="helped with homework", normalized_task="helped with homework", estimated_minutes=30, category="CHILD_CARE")
    day_tasks, _ = merge_task_into_day(day_tasks, t1)
    day_tasks, action = merge_task_into_day(day_tasks, t2)
    assert action == "added"
    assert len(day_tasks) == 2


def test_total_and_classified_minutes():
    day_tasks: list[LoggedTask] = []
    for text, cat, mins in [("cooked", "COOK_MEAL", 30), ("mystery", "unclassified", 20)]:
        t = ExtractedTask(raw_text=text, normalized_task=text, estimated_minutes=mins, category=cat)
        day_tasks, _ = merge_task_into_day(day_tasks, t)
    assert total_minutes(day_tasks) == 50
    assert classified_minutes(day_tasks) == 30
