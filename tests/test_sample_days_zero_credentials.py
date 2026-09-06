"""
Verifies the sample-day path runs end-to-end (classification -> valuation
-> MoSPI comparison -> micro-business idea) with ZERO Gemini or GCP calls,
since activity codes are pre-assigned in sample_days.json.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure no credentials are present for this test module, proving the
# sample-day path truly needs none.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)
os.environ["ENVIRONMENT"] = "local"

from src.utils.sample_provider import build_sample_day_tasks, list_sample_days
from src.utils.valuation import compute_mospi_comparisons, compute_valuation, get_dominant_category, get_micro_business_idea


def test_list_sample_days_returns_three():
    samples = list_sample_days()
    assert len(samples) == 3


def test_each_sample_day_runs_full_pipeline_with_zero_credentials():
    samples = list_sample_days()
    for sample in samples:
        tasks, gender = build_sample_day_tasks(sample["id"])
        assert len(tasks) > 0

        valuation = compute_valuation(tasks)
        assert valuation.daily_value_inr >= 0
        assert valuation.annual_value_inr == round(valuation.daily_value_inr * 365, 2)

        comparisons = compute_mospi_comparisons(tasks)
        assert len(comparisons) == 2  # domestic + caregiving, always separate

        dominant = get_dominant_category(tasks)
        if dominant:
            idea = get_micro_business_idea(dominant)
            assert idea is None or "title" in idea


def test_sample_day_invalid_id_returns_empty():
    tasks, gender = build_sample_day_tasks("does_not_exist")
    assert tasks == []
    assert gender is None
