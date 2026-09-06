"""Gemini helpers: extract tasks from a sentence, or classify one phrase."""
from __future__ import annotations

import json
import logging
import re

from src.config import config
from src.graph.state import ExtractedTask

logger = logging.getLogger("careval.agent")

EXTRACT_PROMPT = """Extract household and caregiving tasks from this message.

Return ONLY a JSON array:
[{"raw_text": "<snippet>", "normalized_task": "<2-5 words>", "estimated_minutes": <int>}]

- One object per distinct task.
- No duration mentioned → return [].
- No task → return [].
- Do not invent tasks or minutes. No prose outside the JSON.
"""

CLASSIFY_PROMPT = "You are a strict, literal classifier. Follow instructions exactly."

_client = None
_init_error: str | None = None


def gemini_ready() -> bool:
    return _client is not None


def _use_vertex() -> bool:
    """GCP + project uses Vertex/ADC unless we only have a developer API key."""
    if not config.is_gcp or not config.gcp_project_id:
        return False
    return config.use_vertex_ai or not config.gemini_api_key


def ensure_gemini() -> bool:
    """Connect once. Local = API key. GCP = Vertex AI via ADC (or API key fallback)."""
    global _client, _init_error
    if _client is not None:
        return True
    try:
        from google import genai

        if _use_vertex():
            _client = genai.Client(
                vertexai=True,
                project=config.gcp_project_id,
                location=config.gcp_location,
            )
        elif config.gemini_api_key:
            _client = genai.Client(api_key=config.gemini_api_key)
        else:
            _init_error = "No Gemini credentials (set GEMINI_API_KEY, or ENVIRONMENT=gcp with GCP_PROJECT_ID)"
            return False
    except Exception as exc:
        logger.exception("Failed to initialize Gemini")
        _init_error = str(exc)
        if _use_vertex() and config.gemini_api_key:
            try:
                from google import genai

                _client = genai.Client(api_key=config.gemini_api_key)
            except Exception as fallback_exc:
                logger.exception("Gemini API-key fallback also failed")
                _init_error = str(fallback_exc)
    return _client is not None


def generate(system_prompt: str, user_prompt: str) -> str:
    if _client is None:
        raise RuntimeError(_init_error or "Gemini is not connected")
    response = _client.models.generate_content(
        model=config.gemini_model,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": 1024,
        },
    )
    return (response.text or "").strip()


def extract_tasks(message: str) -> str:
    return generate(EXTRACT_PROMPT, message)


def classify_text(prompt: str) -> str:
    return generate(CLASSIFY_PROMPT, prompt)


def parse_extraction_response(raw_response: str) -> list[ExtractedTask]:
    """Turn Gemini JSON into tasks. Bad output → empty list, never a crash."""
    if not raw_response:
        return []
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Gemini extraction returned non-JSON: %r", raw_response[:300])
        return []
    if not isinstance(data, list):
        logger.warning("Gemini extraction returned non-list JSON: %r", data)
        return []

    tasks: list[ExtractedTask] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            tasks.append(
                ExtractedTask(
                    raw_text=str(item.get("raw_text", "")),
                    normalized_task=str(item.get("normalized_task", "")),
                    estimated_minutes=int(item.get("estimated_minutes", 15) or 15),
                    category="unclassified",
                )
            )
        except Exception:
            logger.exception("Skipping malformed extracted task: %s", item)
    return tasks


_USE_SHARED = object()


def extract_tasks_from_message(message: str, gemini_client=_USE_SHARED) -> list[ExtractedTask]:
    """Pass gemini_client=None to skip Gemini (tests). Omit it to use the shared client."""
    if not (message or "").strip() or gemini_client is None:
        return []
    if not gemini_ready():
        return []
    try:
        return parse_extraction_response(extract_tasks(message))
    except Exception:
        logger.exception("Gemini extraction call failed")
        return []
