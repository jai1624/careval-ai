import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.classifier import UNCLASSIFIED, classify_task, keyword_classify

# MOSPI TUS codes: 311=cooking, 312=cleaning, 313=laundry,
#                  381=household mgmt, 411=homework, 412=childcare, 420=elder care


def test_keyword_classify_cooking():
    assert keyword_classify("I cooked breakfast for everyone") == "311"


def test_keyword_classify_cooked_for_mins():
    assert keyword_classify("cooked for 45 mins") == "311"


def test_keyword_classify_tutoring():
    assert keyword_classify("tutoring for 1 hour") == "411"


def test_keyword_classify_childcare():
    assert keyword_classify("childcare for 2 hours") == "412"


def test_keyword_classify_laundry():
    assert keyword_classify("folded clothes and did the laundry") == "313"


def test_keyword_classify_child_care():
    assert keyword_classify("gave the baby a bath and put her to sleep") == "412"


def test_keyword_classify_no_match_returns_none():
    assert keyword_classify("xyz totally unrelated gibberish qqq") is None


def test_classify_task_falls_back_to_unclassified_without_gemini():
    result = classify_task("xyz totally unrelated gibberish qqq", gemini_client=None)
    assert result == UNCLASSIFIED


def test_classify_task_prefers_keyword_match_over_gemini():
    # Even if a gemini_client were provided, keyword match should short-circuit
    class FakeClientThatShouldNotBeCalled:
        def classify_text(self, prompt):
            raise AssertionError("Gemini should not be called when keyword match succeeds")

    result = classify_task("cooked dinner tonight", gemini_client=FakeClientThatShouldNotBeCalled())
    assert result == "311"


def test_longest_keyword_match_wins():
    # "homework" is more specific than generic child-related words → 411
    assert keyword_classify("helped with homework after school") == "411"
