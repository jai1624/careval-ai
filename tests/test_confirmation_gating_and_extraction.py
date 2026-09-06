import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.conversational_agent import extract_tasks_from_message, parse_extraction_response


def test_parse_extraction_valid_json():
    raw = '[{"raw_text": "cooked dinner", "normalized_task": "cooked dinner", "estimated_minutes": 30}]'
    tasks = parse_extraction_response(raw)
    assert len(tasks) == 1
    assert tasks[0].normalized_task == "cooked dinner"
    assert tasks[0].category == "unclassified"


def test_parse_extraction_handles_markdown_fences():
    raw = '```json\n[{"raw_text": "washed dishes", "normalized_task": "washed dishes", "estimated_minutes": 15}]\n```'
    tasks = parse_extraction_response(raw)
    assert len(tasks) == 1


def test_parse_extraction_handles_empty_array():
    tasks = parse_extraction_response("[]")
    assert tasks == []


def test_parse_extraction_handles_garbage_gracefully():
    tasks = parse_extraction_response("not json at all { broken")
    assert tasks == []


def test_parse_extraction_handles_non_list_json():
    tasks = parse_extraction_response('{"raw_text": "oops"}')
    assert tasks == []


def test_parse_extraction_skips_malformed_items_but_keeps_valid_ones():
    raw = '[{"raw_text": "ok task", "normalized_task": "ok task", "estimated_minutes": 20}, "garbage_item"]'
    tasks = parse_extraction_response(raw)
    assert len(tasks) == 1
    assert tasks[0].normalized_task == "ok task"


def test_extract_tasks_returns_empty_without_gemini_client():
    result = extract_tasks_from_message("I cooked dinner", gemini_client=None)
    assert result == []


def test_extract_tasks_handles_empty_message():
    result = extract_tasks_from_message("", gemini_client=None)
    assert result == []
