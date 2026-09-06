"""
Illustrative replacement-cost valuation + MoSPI time-use comparison.


Rupees = (minutes / 60) * mapped paid-role hourly rate in Python.
MoSPI TUS (live JSON or cached file) supplies minutes/day only — never wages.
Gemini never sees or influences a rupee figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import config
from src.graph.state import LoggedTask
from src.utils.data_loader import load_activity_codes, load_taxonomy, load_wage_equivalents
from src.utils.mospi_context import MospiBenchmarkPack, load_mospi_time_use_benchmarks

MANDATORY_DISCLAIMER = (
    "This is an illustrative replacement-cost estimate, not an audited or "
    "government-certified valuation."
)

ANNUAL_DAYS = 365

_SKILL_ORDER = (
    "Cooking",
    "Planning",
    "Caregiving",
    "Household management",
)


@dataclass
class TaskValue:
    task_id: str
    display_label: str
    icon: str
    category: str
    benchmark_role: str
    minutes: int
    hourly_rate: float
    value_inr: float
    used_synthetic_fallback: bool


@dataclass
class ValuationResult:
    task_values: list[TaskValue] = field(default_factory=list)
    daily_value_inr: float = 0.0
    annual_value_inr: float = 0.0
    total_minutes_logged: int = 0
    classified_minutes: int = 0
    unclassified_minutes: int = 0
    skills: list[str] = field(default_factory=list)
    used_synthetic_fallback: bool = True
    rate_source_note: str = ""
    disclaimer: str = MANDATORY_DISCLAIMER


def format_duration(total_minutes: int) -> str:
    hours, minutes = divmod(max(0, total_minutes), 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _collect_skills(day_tasks: list[LoggedTask], taxonomy: dict) -> list[str]:
    seen: set[str] = set()
    for task in day_tasks:
        if task.category == "unclassified":
            continue
        for skill in taxonomy.get(task.category, {}).get("skills") or []:
            if isinstance(skill, str) and skill.strip():
                seen.add(skill.strip())
    ordered = [s for s in _SKILL_ORDER if s in seen]
    extras = sorted(seen - set(_SKILL_ORDER))
    return ordered + extras


def compute_valuation(day_tasks: list[LoggedTask]) -> ValuationResult:
    wages = load_wage_equivalents(str(config.wage_equivalents_path))
    taxonomy = load_taxonomy(str(config.taxonomy_path))

    task_values: list[TaskValue] = []
    daily_value = 0.0
    classified_min = 0
    unclassified_min = 0
    any_synthetic = False

    for task in day_tasks:
        if task.category == "unclassified" or task.category not in wages:
            unclassified_min += task.estimated_minutes
            continue

        wage_info = wages[task.category]
        tax_entry = taxonomy.get(task.category, {})
        value = round((task.estimated_minutes / 60.0) * wage_info.base_hourly_rate_inr, 2)
        daily_value += value
        classified_min += task.estimated_minutes
        if wage_info.is_synthetic_fallback:
            any_synthetic = True

        task_values.append(
            TaskValue(
                task_id=task.task_id,
                display_label=tax_entry.get("display_label", task.normalized_task.title()),
                icon=tax_entry.get("icon", "✨"),
                category=task.category,
                benchmark_role=wage_info.benchmark_role,
                minutes=task.estimated_minutes,
                hourly_rate=wage_info.base_hourly_rate_inr,
                value_inr=value,
                used_synthetic_fallback=wage_info.is_synthetic_fallback,
            )
        )

    daily_value = round(daily_value, 2)
    used_fallback = any_synthetic or not task_values
    rate_note = (
        "Hourly rates are 2025–26 urban India market references "
        "(platforms: Urban Company, UrbanPro, Justdial), set conservatively at the lower-middle of confirmed ranges."
        if used_fallback
        else "Hourly rates are mapped 2025–26 market references for comparable paid roles."
    )

    return ValuationResult(
        task_values=task_values,
        daily_value_inr=daily_value,
        annual_value_inr=round(daily_value * ANNUAL_DAYS, 2),
        total_minutes_logged=sum(t.estimated_minutes for t in day_tasks),
        classified_minutes=classified_min,
        unclassified_minutes=unclassified_min,
        skills=_collect_skills(day_tasks, taxonomy),
        used_synthetic_fallback=used_fallback,
        rate_source_note=rate_note,
    )


@dataclass
class MospiComparison:
    domain: str  # "domestic" | "caregiving"
    user_minutes: int
    female_avg_minutes: int
    male_avg_minutes: int
    comparison_text: str
    source_label: str = ""
    catalog_url: str = ""
    used_cached_fallback: bool = True


def _build_comparison_text(domain_label: str, user_minutes: int, female_avg: int, male_avg: int) -> str:
    if user_minutes <= 0:
        return f"You haven't logged any {domain_label} time yet."

    parts = []
    if female_avg > 0:
        ratio_f = user_minutes / female_avg
        if ratio_f >= 1.05:
            parts.append(f"about {ratio_f:.1f}x the average time women in India spend on {domain_label} work daily")
        elif ratio_f <= 0.95:
            parts.append(f"about {ratio_f:.1f}x (a bit below) the average time women spend on {domain_label} work daily")
        else:
            parts.append(f"in line with the average time women spend on {domain_label} work daily")

    if male_avg > 0:
        ratio_m = user_minutes / male_avg
        parts.append(f"roughly {ratio_m:.1f}x what the average man in India spends on {domain_label} work daily")

    return "Today you logged " + " -- ".join(parts) + "."


def compute_mospi_comparisons(
    day_tasks: list[LoggedTask],
    pack: MospiBenchmarkPack | None = None,
) -> list[MospiComparison]:
    """Domestic and caregiving are ALWAYS compared separately, never combined."""
    pack = pack or load_mospi_time_use_benchmarks()

    domestic_minutes = 0
    caregiving_minutes = 0
    codes = load_activity_codes(str(config.activity_codes_path))
    for task in day_tasks:
        info = codes.get(task.category)
        if info is None:
            continue
        if info.category == "domestic":
            domestic_minutes += task.estimated_minutes
        elif info.category == "caregiving":
            caregiving_minutes += task.estimated_minutes

    results = []
    for domain, user_minutes, block in (
        ("domestic", domestic_minutes, pack.domestic),
        ("caregiving", caregiving_minutes, pack.caregiving),
    ):
        female = block.get("female_minutes_per_day", 0)
        male = block.get("male_minutes_per_day", 0)
        results.append(
            MospiComparison(
                domain=domain,
                user_minutes=user_minutes,
                female_avg_minutes=female,
                male_avg_minutes=male,
                comparison_text=_build_comparison_text(domain, user_minutes, female, male),
                source_label=pack.source_label,
                catalog_url=pack.catalog_url,
                used_cached_fallback=pack.used_cached_fallback,
            )
        )
    return results



def get_ranked_categories(day_tasks: list[LoggedTask]) -> list[str]:
    """Non-unclassified categories ordered by total duration descending.

    Used to prioritize micro-business ideas by what the user actually
    spent the most time on today. Does not change rupee valuation.
    """
    totals: dict[str, int] = {}
    for task in day_tasks:
        if task.category == "unclassified":
            continue
        totals[task.category] = totals.get(task.category, 0) + task.estimated_minutes
    return sorted(totals, key=lambda code: (-totals[code], code))


def get_dominant_category(day_tasks: list[LoggedTask]) -> str | None:
    """Return the activity code the user spent the most minutes on today."""
    ranked = get_ranked_categories(day_tasks)
    return ranked[0] if ranked else None


def get_micro_business_idea(category: str | None) -> dict | None:
    if not category:
        return None
    taxonomy = load_taxonomy(str(config.taxonomy_path))
    return taxonomy.get(category, {}).get("micro_business")


def get_market_enquiries(day_tasks: list[LoggedTask]) -> list[dict]:
    """Local enquiry links for services that match today's logged work.

    These are suggestion links only — never used to set rupee valuation.
    """
    taxonomy = load_taxonomy(str(config.taxonomy_path))
    seen: set[str] = set()
    results: list[dict] = []
    for task in day_tasks:
        if task.category in seen or task.category == "unclassified":
            continue
        entry = taxonomy.get(task.category) or {}
        enquiry = entry.get("market_enquiry")
        if not isinstance(enquiry, dict):
            continue
        links = [
            link
            for link in (enquiry.get("links") or [])
            if isinstance(link, dict) and link.get("label") and link.get("url")
        ]
        if not links:
            continue
        seen.add(task.category)
        results.append(
            {
                "category": task.category,
                "icon": entry.get("icon", "✨"),
                "service": enquiry.get("service") or entry.get("display_label") or "Local service",
                "prompt": enquiry.get("prompt") or "Ask nearby providers what they charge.",
                "links": links,
            }
        )
    return results
