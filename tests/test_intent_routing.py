import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.orchestrator import (
    is_idea_request,
    is_question_about_results,
    is_done_request,
    is_reset_request,
)


def test_is_idea_request_catches_suggest_phrases():
    assert is_idea_request("suggest me some ideas")
    assert is_idea_request("what can I do with this skill?")
    assert is_idea_request("any earning ideas?")
    assert is_idea_request("micro business suggestions")
    assert is_idea_request("I want to start a side hustle")


def test_is_idea_request_ignores_task_and_unsure():
    assert not is_idea_request("I cooked dinner")
    assert not is_idea_request("no idea")
    assert not is_idea_request("not sure")
    assert not is_idea_request("")


def test_is_question_about_results():
    assert is_question_about_results("how did you calculate this?")
    assert is_question_about_results("why is the number so high")
    assert is_question_about_results("can you explain the total")
    assert not is_question_about_results("another idea please")


def test_is_reset_request():
    assert is_reset_request("new day")
    assert is_reset_request("can we start over")
    assert not is_reset_request("I cooked again")


def test_is_done_request():
    assert is_done_request("that's all for today")
    assert is_done_request("I'm done")
    assert is_done_request("show me the total")
    assert not is_done_request("I cooked dinner")
