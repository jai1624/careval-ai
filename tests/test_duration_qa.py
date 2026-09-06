import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.orchestrator import (
    has_explicit_duration,
    is_duration_only,
    is_unsure_duration,
    parse_stated_minutes,
    rule_based_extract,
)


def test_parse_stated_minutes_reads_explicit_times():
    assert parse_stated_minutes("45 min") == 45
    assert parse_stated_minutes("I cooked for 45 minutes") == 45
    assert parse_stated_minutes("an hour") == 60
    assert parse_stated_minutes("half an hour") == 30
    assert parse_stated_minutes("20") == 20
    assert parse_stated_minutes("3") == 3
    assert parse_stated_minutes("240") == 240
    assert parse_stated_minutes("about 37") == 37
    assert parse_stated_minutes("cooking 40") == 40
    assert parse_stated_minutes("40 cooking") == 40
    assert parse_stated_minutes("20-30") == 25
    assert parse_stated_minutes("1.5 hours") == 90
    assert parse_stated_minutes("2h") == 120


def test_parse_stated_minutes_loose_accepts_any_number_in_reply():
    assert parse_stated_minutes("maybe 12?", loose=True) == 12
    assert parse_stated_minutes("it was 7", loose=True) == 7
    assert parse_stated_minutes("1", loose=True) == 1


def test_parse_stated_minutes_does_not_invent():
    assert parse_stated_minutes("I cooked dinner today") is None
    assert parse_stated_minutes("Cooking") is None
    assert parse_stated_minutes("not sure") is None
    assert parse_stated_minutes("I have 2 kids") is None


def test_has_explicit_duration():
    assert has_explicit_duration("cooked for 20 minutes")
    assert not has_explicit_duration("cooked dinner")


def test_is_unsure_duration():
    assert is_unsure_duration("not sure")
    assert is_unsure_duration("idk")
    assert not is_unsure_duration("45 minutes")


def test_is_duration_only():
    assert is_duration_only("45")
    assert is_duration_only("about 20 min")
    assert is_duration_only("1.5 hours")
    assert not is_duration_only("tutoring for 30 min")
    assert not is_duration_only("I have 2 kids")


def test_rule_based_extract_skips_tasks_without_duration():
    assert rule_based_extract("I cooked dinner") == []
    assert rule_based_extract("Cooking") == []


def test_rule_based_extract_keeps_tasks_with_duration():
    tasks = rule_based_extract("I cooked dinner for 45 minutes")
    assert len(tasks) == 1
    assert tasks[0].estimated_minutes == 45


def test_rule_based_extract_accepts_one_liner_number():
    tasks = rule_based_extract("cooking 40")
    assert len(tasks) == 1
    assert tasks[0].estimated_minutes == 40
