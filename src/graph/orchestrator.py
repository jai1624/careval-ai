"""
Lightweight LangGraph orchestration for CareVal AI.

Linear graph only: extract -> classify -> merge.
One optional Gemini call inside extract. No loops, no multi-agent handoff.
When Gemini is missing, a deterministic keyword parser keeps demos working.
"""
from __future__ import annotations

import logging
import re

from langgraph.graph import END, StateGraph

from src.agents.conversational_agent import (
    ensure_gemini,
    extract_tasks_from_message,
    gemini_ready,
)
from src.graph.state import ConversationState, ExtractedTask, LoggedTask
from src.utils.classifier import classify_task, keyword_classify
from src.utils.memory import merge_many

logger = logging.getLogger("careval.orchestrator")

# Times stay regex — "45m", "1.5 hours", "20-30" are messy to list.
_DURATION_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|h|minutes?|mins?|m)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<b>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOOSE_NUMBER_RE = re.compile(r"(?<![\d.])(?P<num>\d+(?:\.\d+)?)(?![\d.])")
_SPLIT_RE = re.compile(r"(?:,|;|\n|\band then\b|\bthen\b|\band\b|\.|!|\?)+", re.IGNORECASE)

# Short answers: padded phrase match, not regex. Apostrophes stripped so
# "that's all" and "thats all" both hit.
_UNSURE = ("not sure", "dont know", "no idea", "idk")
_IDEAS = (
    "ideas",
    "idea",
    "what can i do",
    "earn",
    "earning",
    "micro business",
    "microbusiness",
    "business",
    "side hustle",
    "sidehustle",
    "make money",
)
_RESULTS_Q = ("how", "why", "explain", "calculate", "breakdown", "details", "detail")
_RESET = ("new day", "start over", "reset", "start again", "another day")
_DONE = (
    "thats all",
    "im done",
    "done for today",
    "finish for today",
    "finished for today",
    "wrap up",
    "wrapping up",
    "show the total",
    "show me the total",
    "show results",
    "show the results",
    "show me the results",
    "show me results",
    "value my day",
    "value the day",
    "calculate",
)
_NAMED_MINUTES = (
    ("half an hour", 30),
    ("half hour", 30),
    ("an hour", 60),
    ("one hour", 60),
    ("all morning", 180),
    ("all afternoon", 180),
    ("all evening", 180),
    ("all day", 480),
)
_FILLERS = frozenset(
    {
        "about",
        "around",
        "approx",
        "approximately",
        "roughly",
        "maybe",
        "nearly",
        "for",
        "only",
        "just",
        "an",
        "a",
        "one",
        "half",
        "all",
        "to",
    }
)
_TIME_UNITS = frozenset(
    {"hours", "hour", "hrs", "hr", "h", "minutes", "minute", "mins", "min", "m"}
)
_DURATION_EXTRA = frozenset({"morning", "afternoon", "evening", "day"})


def _norm(text: str) -> str:
    cleaned = []
    for ch in (text or "").lower().replace("'", "").replace("'", ""):
        cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(cleaned).split())


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    hay = f" {_norm(text)} "
    return any(f" {p} " in hay for p in phrases)


def _is_plain_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _parse_duration_minutes(text: str, fallback: int = 30) -> int:
    matches = list(_DURATION_RE.finditer(text or ""))
    if not matches:
        return fallback
    total = 0.0
    for match in matches:
        num = float(match.group("num"))
        unit = match.group("unit").lower()
        if unit.startswith("h"):
            total += num * 60
        else:
            total += num
    return max(1, min(1440, int(round(total))))


def _clamp_minutes(value: float) -> int:
    return max(1, min(1440, int(round(value))))


def _number_to_minutes(num: float, unit: str | None = None) -> int:
    if unit and unit.lower().startswith("h"):
        return _clamp_minutes(num * 60)
    if unit is None and 0 < num < 12 and not float(num).is_integer():
        return _clamp_minutes(num * 60)
    return _clamp_minutes(num)


def parse_stated_minutes(text: str, *, loose: bool = False) -> int | None:
    """Minutes the user actually said. Returns None instead of inventing a default.

    Accepts one-liners: 40, 40m, 1.5 hours, 20-30, cooking 45, about 25.
    When loose=True (answering 'how long?'), any number in the reply counts.
    """
    if not (text or "").strip():
        return None
    lowered = text.strip().lower()
    if is_unsure_duration(lowered) and not _LOOSE_NUMBER_RE.search(lowered):
        return None
    for phrase, minutes in _NAMED_MINUTES:
        if _has_any(lowered, (phrase,)):
            return minutes
    range_match = _RANGE_RE.search(lowered)
    if range_match and not _DURATION_RE.search(lowered):
        low = float(range_match.group("a"))
        high = float(range_match.group("b"))
        return _number_to_minutes((low + high) / 2)
    if _DURATION_RE.search(lowered):
        return _parse_duration_minutes(lowered, fallback=30)

    numbers = list(_LOOSE_NUMBER_RE.finditer(lowered))
    if not numbers:
        return None
    first = float(numbers[0].group("num"))
    if loose:
        return _number_to_minutes(first)

    filler = " ".join(
        w.strip(" .,!?")
        for w in lowered.replace("~", " ").split()
        if w.strip(" .,!?") and w.strip(" .,!?").lower() not in _FILLERS
    )
    if _is_plain_number(filler):
        return _number_to_minutes(first)
    if len(numbers) == 1 and keyword_classify(text):
        raw_num = numbers[0].group("num")
        tokens = filler.split()
        if tokens and (
            tokens[0].startswith(raw_num) or tokens[-1].startswith(raw_num)
        ):
            return _number_to_minutes(first)
    return None


def has_explicit_duration(text: str) -> bool:
    return parse_stated_minutes(text) is not None


def is_unsure_duration(text: str) -> bool:
    return _has_any(text, _UNSURE)


def is_idea_request(text: str) -> bool:
    """True when the user is asking for earning / micro-business ideas, not logging a task."""
    if not (text or "").strip() or is_unsure_duration(text):
        return False
    return _has_any(text, _IDEAS)


def is_question_about_results(text: str) -> bool:
    """True when the user wants the rupee figure or comparison explained."""
    return _has_any(text, _RESULTS_Q)


def is_reset_request(text: str) -> bool:
    return _has_any(text, _RESET)


def is_done_request(text: str) -> bool:
    """True when the user wants to finish logging and see today's valuation."""
    return _has_any(text, _DONE)


def is_duration_only(text: str) -> bool:
    """True when the message is just a time, not a new task."""
    if keyword_classify(text):
        return False
    if parse_stated_minutes(text, loose=True) is None:
        return False
    leftover = [
        token
        for token in _norm(text).split()
        if token not in _FILLERS
        and token not in _TIME_UNITS
        and token not in _DURATION_EXTRA
        and not _is_plain_number(token)
    ]
    return leftover == []


def rule_based_extract(message: str) -> list[ExtractedTask]:
    """Deterministic fallback extractor for zero-credential demos.

    Splits the message into clauses, keyword-classifies each clause, and
    estimates minutes from explicit durations or category defaults.
    Never calls Gemini.
    """
    if not (message or "").strip():
        return []

    clauses = [c.strip() for c in _SPLIT_RE.split(message) if c and c.strip()]
    if not clauses:
        clauses = [message.strip()]

    tasks: list[ExtractedTask] = []
    seen: set[tuple[str, str]] = set()
    for clause in clauses:
        code = keyword_classify(clause)
        if not code:
            # Whole-message fallback once if clause-level matching fails.
            continue
        minutes = parse_stated_minutes(clause)
        if minutes is None:
            continue
        normalized = re.sub(r"\s+", " ", clause).strip().lower()
        key = (code, normalized[:48])
        if key in seen:
            continue
        seen.add(key)
        tasks.append(
            ExtractedTask(
                raw_text=clause,
                normalized_task=normalized[:80],
                estimated_minutes=minutes,
                category="unclassified",  # classify_node owns the final code
            )
        )

    if not tasks:
        # Last resort: classify the full message as one task.
        code = keyword_classify(message)
        minutes = parse_stated_minutes(message)
        if code and minutes is not None:
            tasks.append(
                ExtractedTask(
                    raw_text=message.strip(),
                    normalized_task=message.strip().lower()[:80],
                    estimated_minutes=minutes,
                    category="unclassified",
                )
            )
    return tasks


def extract_node(state: ConversationState) -> ConversationState:
    message = state.get("latest_message", "")
    ensure_gemini()
    try:
        tasks = extract_tasks_from_message(message) or rule_based_extract(message)
        state["extracted_tasks"] = [t.model_dump() for t in tasks]
        if not tasks and not gemini_ready():
            state["last_error"] = (
                "I couldn't read a clear task from that yet. "
                "Try something like 'cooked dinner for 45 minutes'."
            )
    except Exception:
        logger.exception("extract_node failed")
        fallback = rule_based_extract(message)
        state["extracted_tasks"] = [t.model_dump() for t in fallback]
        state["last_error"] = None if fallback else (
            "Something went wrong reading that message, but nothing was lost."
        )
    return state


def classify_node(state: ConversationState) -> ConversationState:
    ensure_gemini()
    extracted = [ExtractedTask(**t) for t in state.get("extracted_tasks", [])]
    classified: list[ExtractedTask] = []
    for task in extracted:
        try:
            code = classify_task(task.normalized_task or task.raw_text, gemini_client=gemini_ready())
        except Exception:
            logger.exception("classify_node failed for task: %s", task)
            code = "unclassified"
        classified.append(task.model_copy(update={"category": code}))
    state["classified_tasks"] = [t.model_dump() for t in classified]
    return state


def merge_node(state: ConversationState) -> ConversationState:
    day_tasks = [LoggedTask(**t) for t in state.get("day_tasks", [])]
    incoming = [ExtractedTask(**t) for t in state.get("classified_tasks", [])]
    try:
        updated_tasks, _actions = merge_many(day_tasks, incoming)
        state["day_tasks"] = [t.model_dump() for t in updated_tasks]
        state["total_minutes"] = sum(t.estimated_minutes for t in updated_tasks)
    except Exception:
        logger.exception("merge_node failed")
        state["last_error"] = "I had trouble saving that task, but your earlier progress is safe."
    return state


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("extract", extract_node)
    graph.add_node("classify", classify_node)
    graph.add_node("merge", merge_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "classify")
    graph.add_edge("classify", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


_compiled_graph = None


def run_turn(state: ConversationState) -> ConversationState:
    """Runs one extract -> classify -> merge turn. Never raises."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    try:
        return _compiled_graph.invoke(state)
    except Exception:
        logger.exception("Graph execution failed entirely")
        state["last_error"] = "Something went wrong processing that message. Please try again."
        return state
