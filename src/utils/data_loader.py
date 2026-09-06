"""
Dynamic data loading for activity codes, wages, taxonomy, and MoSPI benchmarks.

NOTHING in this module hardcodes an activity code. All valid codes are
derived at runtime from tus_activity_codes.csv, and every downstream
consumer (classifier, valuation engine, taxonomy lookup) is driven off
that dynamically-loaded set.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("careval.data_loader")


@dataclass(frozen=True)
class ActivityCode:
    code: str
    label: str        # sub_category
    category: str     # "domestic" | "caregiving" derived from major_division
    keywords: tuple[str, ...]
    description: str = ""  # human-readable description


@dataclass(frozen=True)
class WageInfo:
    activity_code: str
    base_hourly_rate_inr: float
    benchmark_role: str
    source: str
    is_synthetic_fallback: bool = True


def _safe_read_csv(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("Data file missing: %s", path)
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        logger.exception("Failed to parse CSV: %s", path)
        return []


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        logger.error("Data file missing: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to parse JSON: %s", path)
        return {}


@lru_cache(maxsize=1)
def load_activity_codes(path: str) -> dict[str, ActivityCode]:
    """Returns {activity_code: ActivityCode} from the PAM2026 CSV schema.

    Required columns: activity_code, major_division, sub_category, keywords, description.
    """
    rows = _safe_read_csv(Path(path))
    result: dict[str, ActivityCode] = {}
    for row in rows:
        try:
            code = (row.get("activity_code") or "").strip()
            label = (row.get("sub_category") or "").strip()
            major = (row.get("major_division") or "").strip().lower()
            if not code or not label or not major:
                logger.warning("Skipping activity row missing PAM2026 columns: %s", row)
                continue
            if "division 4" in major or "caregiving" in major:
                category = "caregiving"
            elif "division 3" in major or "domestic" in major:
                category = "domestic"
            else:
                logger.warning("Skipping activity row with unknown major_division: %s", row)
                continue
            keywords = tuple(
                kw.strip().lower()
                for kw in (row.get("keywords") or "").split(",")
                if kw.strip()
            )
            result[code] = ActivityCode(
                code=code,
                label=label,
                category=category,
                keywords=keywords,
                description=(row.get("description") or "").strip(),
            )
        except Exception:
            logger.exception("Skipping malformed activity code row: %s", row)
    return result


@lru_cache(maxsize=1)
def load_wage_equivalents(path: str) -> dict[str, WageInfo]:
    rows = _safe_read_csv(Path(path))
    result: dict[str, WageInfo] = {}
    for row in rows:
        try:
            code = row["activity_code"].strip()
            if not code:
                continue
            rate = float(row.get("base_hourly_rate_inr", 0) or 0)
            if rate <= 0:
                logger.warning("Non-positive wage rate for %s, skipping", code)
                continue
            fallback_raw = (row.get("is_synthetic_fallback") or "true").strip().lower()
            result[code] = WageInfo(
                activity_code=code,
                base_hourly_rate_inr=rate,
                benchmark_role=row.get("benchmark_role", "").strip(),
                source=row.get("source", "").strip(),
                is_synthetic_fallback=fallback_raw in {"1", "true", "yes", "on", ""},
            )
        except Exception:
            logger.exception("Skipping malformed wage row: %s", row)
    return result


@lru_cache(maxsize=1)
def load_taxonomy(path: str) -> dict:
    return _safe_read_json(Path(path))


@lru_cache(maxsize=1)
def load_mospi_benchmarks(path: str) -> dict:
    data = _safe_read_json(Path(path))
    if not data:
        # Hard fallback to the locked spec values so the app never crashes,
        # even if the JSON file is somehow corrupted or missing.
        logger.error("MoSPI benchmark file unreadable; using locked fallback constants")
        return {
            "domestic": {"female_minutes_per_day": 289, "male_minutes_per_day": 88},
            "caregiving": {"female_minutes_per_day": 137, "male_minutes_per_day": 75},
        }
    return data


@lru_cache(maxsize=1)
def load_sample_days(path: str) -> list[dict]:
    data = _safe_read_json(Path(path))
    return data.get("sample_days", [])


def valid_activity_codes(path: str) -> set[str]:
    """The single dynamic source of truth for what counts as a valid code."""
    return set(load_activity_codes(path).keys())


def clear_caches() -> None:
    """Used by tests to force re-reading data files."""
    load_activity_codes.cache_clear()
    load_wage_equivalents.cache_clear()
    load_taxonomy.cache_clear()
    load_mospi_benchmarks.cache_clear()
    load_sample_days.cache_clear()
