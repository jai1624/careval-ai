import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.graph.state import LoggedTask
from src.utils.data_loader import load_wage_equivalents
from src.utils.valuation import (
    compute_mospi_comparisons,
    compute_valuation,
    get_dominant_category,
    get_ranked_categories,
    get_market_enquiries,
    get_micro_business_idea,
)

_WAGES = load_wage_equivalents(str(config.wage_equivalents_path))
RATE_311 = _WAGES["311"].base_hourly_rate_inr
RATE_313 = _WAGES["313"].base_hourly_rate_inr


def make_task(category, minutes, text="task"):
    return LoggedTask(
        task_id=str(uuid.uuid4()),
        raw_text=text,
        normalized_task=text,
        estimated_minutes=minutes,
        category=category,
    )


def test_valuation_basic_calculation():
    tasks = [make_task("311", 60)]
    result = compute_valuation(tasks)
    assert result.daily_value_inr == RATE_311
    assert result.annual_value_inr == RATE_311 * 365


def test_valuation_excludes_unclassified():
    tasks = [make_task("311", 60), make_task("unclassified", 45)]
    result = compute_valuation(tasks)
    assert result.classified_minutes == 60
    assert result.unclassified_minutes == 45
    assert result.daily_value_inr == RATE_311


def test_valuation_sums_multiple_tasks():
    tasks = [make_task("311", 30), make_task("313", 60)]
    result = compute_valuation(tasks)
    assert result.daily_value_inr == round(0.5 * RATE_311 + RATE_313, 2)


def test_mospi_comparison_domestic_and_caregiving_are_separate():
    tasks = [make_task("311", 100), make_task("412", 50)]  # domestic + caregiving
    comparisons = compute_mospi_comparisons(tasks)
    domains = {c.domain: c for c in comparisons}
    assert "domestic" in domains and "caregiving" in domains
    assert domains["domestic"].user_minutes == 100
    assert domains["caregiving"].user_minutes == 50
    # They must never be summed together into one figure
    assert domains["domestic"].user_minutes != domains["caregiving"].user_minutes + domains["domestic"].user_minutes


def test_dominant_category_picks_highest_minutes():
    tasks = [make_task("311", 20), make_task("313", 90)]
    assert get_dominant_category(tasks) == "313"


def test_dominant_category_excludes_unclassified():
    tasks = [make_task("unclassified", 500), make_task("311", 10)]
    assert get_dominant_category(tasks) == "311"


def test_dominant_category_none_when_all_unclassified():
    tasks = [make_task("unclassified", 100)]
    assert get_dominant_category(tasks) is None


def test_micro_business_idea_present_for_valid_category():
    idea = get_micro_business_idea("311")
    assert idea is not None
    assert "title" in idea and "step_one" in idea


def test_2026_urban_rates_are_above_old_demo_wages():
    assert RATE_311 >= 200
    assert RATE_313 >= 150
    assert all(info.base_hourly_rate_inr >= 150 for info in _WAGES.values())


def test_replacement_cost_maps_role_and_labels_synthetic_fallback():
    tasks = [make_task("311", 60)]
    result = compute_valuation(tasks)
    assert result.task_values[0].benchmark_role == "Cook / Maharaj"
    assert result.used_synthetic_fallback is True
    assert "illustrative replacement-cost" in result.disclaimer.lower()
    assert "Cooking" in result.skills


def test_mospi_comparison_does_not_change_rupees():
    tasks = [make_task("311", 60)]
    rupees = compute_valuation(tasks).daily_value_inr
    comparisons = compute_mospi_comparisons(tasks)
    assert rupees == RATE_311
    assert comparisons[0].female_avg_minutes == 289
    assert all(c.user_minutes != rupees for c in comparisons)


def test_market_enquiries_are_suggestion_links_not_wages():
    tasks = [make_task("311", 60), make_task("411", 30), make_task("311", 20)]
    before = compute_valuation(tasks).daily_value_inr
    enquiries = get_market_enquiries(tasks)
    after = compute_valuation(tasks).daily_value_inr
    assert before == after
    services = {item["service"] for item in enquiries}
    assert "Home cook / tiffin" in services
    assert "Home tuition" in services
    assert all(item["links"] for item in enquiries)
    assert all(link["url"].startswith("https://") for item in enquiries for link in item["links"])


def test_ranked_categories_orders_by_duration_desc():
    tasks = [
        make_task("311", 20),
        make_task("313", 90),
        make_task("311", 10),
        make_task("unclassified", 500),
    ]
    assert get_ranked_categories(tasks) == ["313", "311"]


def test_ranked_categories_empty_when_all_unclassified():
    tasks = [make_task("unclassified", 40)]
    assert get_ranked_categories(tasks) == []
