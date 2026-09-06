"""
Classification pipeline:
  1. Deterministic keyword substring matching (fast, free, no LLM).
  2. Gemini fallback -- strict constrained pick from the dynamically
     loaded set of valid activity codes only, max 2 retries.
  3. "unclassified" if both fail -- excluded from valuation and
     explained plainly to the user.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.agents.conversational_agent import classify_text
from src.config import config
from src.utils.data_loader import load_activity_codes, valid_activity_codes

logger = logging.getLogger("careval.classifier")

UNCLASSIFIED = "unclassified"


def keyword_classify(text: str) -> Optional[str]:
    """Deterministic substring match against dynamically-loaded keywords.

    Picks the activity code whose keyword list contains the longest
    matching substring (longest match wins to avoid short, ambiguous
    words like "kids" beating a more specific phrase like "homework").
    """
    if not text:
        return None
    text_lower = text.lower()
    best_code = None
    best_len = 0
    codes = load_activity_codes(str(config.activity_codes_path))
    for code, info in codes.items():
        for kw in info.keywords:
            if kw and kw in text_lower and len(kw) > best_len:
                best_code = code
                best_len = len(kw)
    return best_code


def gemini_classify(text: str, gemini_client) -> Optional[str]:
    """Strict constrained fallback classification using Gemini.

    The model is only ever allowed to pick from the dynamically loaded
    set of valid activity codes (plus 'unclassified'). Any response
    outside that set, or any failure, results in None so the caller
    can fall back to 'unclassified'.
    """
    if gemini_client is None:
        return None

    valid_codes = sorted(valid_activity_codes(str(config.activity_codes_path)))
    if not valid_codes:
        return None

    codes_desc = "\n".join(
        f"- {c}: {info.label}" + (f" — {info.description}" if info.description else "")
        for c, info in load_activity_codes(str(config.activity_codes_path)).items()
    )

    prompt = (
        "You are a strict classifier. Given a short description of a household "
        "or caregiving task, pick EXACTLY ONE activity code from this fixed list "
        "that best matches it. Respond with ONLY the activity code, nothing else. "
        "If nothing fits well, respond with exactly: unclassified\n\n"
        f"Valid activity codes:\n{codes_desc}\n\n"
        f"Task description: \"{text}\"\n\n"
        "Answer with only the activity code:"
    )

    for attempt in range(2):
        try:
            response = classify_text(prompt)
            candidate = (response or "").strip()
            # Strip common formatting artifacts defensively
            candidate = candidate.strip("`").strip().split("\n")[0].strip()
            if candidate in valid_codes:
                return candidate
            if candidate == UNCLASSIFIED:
                return UNCLASSIFIED
            logger.warning("Gemini classify attempt %d returned invalid code: %r", attempt, candidate)
        except Exception:
            logger.exception("Gemini classification attempt %d failed", attempt)
    return None


def classify_task(text: str, gemini_client=None) -> str:
    """Full pipeline: keyword match -> Gemini fallback -> unclassified."""
    code = keyword_classify(text)
    if code:
        return code

    code = gemini_classify(text, gemini_client)
    if code and code != UNCLASSIFIED:
        return code

    return UNCLASSIFIED
