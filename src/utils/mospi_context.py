"""
MoSPI time-use benchmark context.

Minutes come from a structured JSON document (optional live URL, else the
cached file). This module never returns wages and never calls Gemini.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.config import config
from src.utils.data_loader import load_mospi_benchmarks

logger = logging.getLogger("careval.mospi_context")

_FETCH_TIMEOUT_SEC = 4


@dataclass(frozen=True)
class MospiBenchmarkPack:
    domestic: dict
    caregiving: dict
    source_label: str
    catalog_url: str
    used_cached_fallback: bool


def _extract_minutes(block: object) -> dict | None:
    if not isinstance(block, dict):
        return None
    try:
        female = int(block["female_minutes_per_day"])
        male = int(block["male_minutes_per_day"])
    except (KeyError, TypeError, ValueError):
        return None
    if female < 0 or male < 0:
        return None
    return {"female_minutes_per_day": female, "male_minutes_per_day": male}


def _parse_benchmark_document(data: dict) -> tuple[dict, dict] | None:
    domestic = _extract_minutes(data.get("domestic"))
    caregiving = _extract_minutes(data.get("caregiving"))
    if not domestic or not caregiving:
        return None
    return domestic, caregiving


def _fetch_live_document(url: str) -> dict | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "CareValAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SEC) as resp:
            raw = resp.read()
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        logger.warning("MoSPI live benchmark fetch failed; using cached file")
        return None


def load_mospi_time_use_benchmarks() -> MospiBenchmarkPack:
    """Official-minutes pack for comparisons only. Wages are never loaded here."""
    catalog_url = config.mospi_catalog_url
    live_url = (config.mospi_benchmarks_url or "").strip() or None

    if live_url:
        live = _fetch_live_document(live_url)
        if live:
            parsed = _parse_benchmark_document(live)
            if parsed:
                domestic, caregiving = parsed
                return MospiBenchmarkPack(
                    domestic=domestic,
                    caregiving=caregiving,
                    source_label=str(live.get("source") or "MoSPI Time Use Survey (live JSON)"),
                    catalog_url=str(live.get("catalog_url") or catalog_url),
                    used_cached_fallback=False,
                )
            logger.warning("Live MoSPI JSON missing required minute fields; using cache")

    cached = load_mospi_benchmarks(str(config.mospi_benchmarks_path))
    parsed = _parse_benchmark_document(cached)
    if parsed:
        domestic, caregiving = parsed
    else:
        domestic = {"female_minutes_per_day": 289, "male_minutes_per_day": 88}
        caregiving = {"female_minutes_per_day": 137, "male_minutes_per_day": 75}
    return MospiBenchmarkPack(
        domestic=domestic,
        caregiving=caregiving,
        source_label=str(cached.get("source") or "MoSPI Time Use Survey (cached)"),
        catalog_url=str(cached.get("catalog_url") or catalog_url),
        used_cached_fallback=True,
    )
