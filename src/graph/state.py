"""
Pydantic data models and the lightweight LangGraph state schema.

Design note: LangGraph is used here purely as a thin, deterministic
orchestration wrapper around otherwise-plain Python functions (extract ->
classify -> merge). There is no autonomous looping, no multi-agent
handoff, and no tool-calling harness -- a single linear graph with one
LLM node.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

MAX_REASONABLE_MINUTES = 1440  # one full day


class ExtractedTask(BaseModel):
    """A single task extracted from the user's free-text message."""

    raw_text: str = Field(..., description="The user's own words describing the task")
    normalized_task: str = Field(..., description="Short, normalized description, e.g. 'made breakfast'")
    estimated_minutes: int = Field(..., description="Estimated duration in minutes")
    category: str = Field(default="unclassified", description="Internal activity code, set by the classifier")

    @field_validator("estimated_minutes", mode="before")
    @classmethod
    def clamp_minutes(cls, v) -> int:
        # Reject nonsensical durations by clamping into a sane band rather
        # than crashing on bad or missing LLM output. Runs BEFORE type
        # coercion so None / non-numeric input never raises a ValidationError.
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 15
        if v <= 0:
            return 5
        if v > MAX_REASONABLE_MINUTES:
            return MAX_REASONABLE_MINUTES
        return v

    @field_validator("raw_text", "normalized_task")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        return v if v else "unspecified activity"


class LoggedTask(ExtractedTask):
    """A task once it has been merged into the day's persistent memory."""

    task_id: str
    times_mentioned: int = 1


class ConversationState(TypedDict, total=False):
    """The single shared state object passed through the LangGraph graph."""

    latest_message: str
    chat_history: list[dict]
    extracted_tasks: list[dict]
    classified_tasks: list[dict]
    day_tasks: list[dict]
    total_minutes: int
    last_error: Optional[str]
