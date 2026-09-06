import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.utils.data_loader import (
    clear_caches,
    load_activity_codes,
    load_mospi_benchmarks,
    load_sample_days,
    load_taxonomy,
    load_wage_equivalents,
    valid_activity_codes,
)


def setup_function(_):
    clear_caches()


def test_activity_codes_load_and_parse_keywords():
    codes = load_activity_codes(str(config.activity_codes_path))
    assert len(codes) > 0
    for code, info in codes.items():
        assert code == info.code
        assert isinstance(info.keywords, tuple)
        assert all(kw == kw.lower() for kw in info.keywords)


def test_wage_equivalents_dynamic_lookup():
    wages = load_wage_equivalents(str(config.wage_equivalents_path))
    assert len(wages) > 0
    for code, info in wages.items():
        assert info.base_hourly_rate_inr > 0


def test_wage_codes_are_subset_of_activity_codes():
    codes = valid_activity_codes(str(config.activity_codes_path))
    wages = load_wage_equivalents(str(config.wage_equivalents_path))
    assert set(wages.keys()).issubset(codes)


def test_mospi_benchmarks_locked_values():
    benchmarks = load_mospi_benchmarks(str(config.mospi_benchmarks_path))
    assert benchmarks["domestic"]["female_minutes_per_day"] == 289
    assert benchmarks["domestic"]["male_minutes_per_day"] == 88
    assert benchmarks["caregiving"]["female_minutes_per_day"] == 137
    assert benchmarks["caregiving"]["male_minutes_per_day"] == 75


def test_taxonomy_has_entries_for_every_activity_code():
    codes = valid_activity_codes(str(config.activity_codes_path))
    taxonomy = load_taxonomy(str(config.taxonomy_path))
    for code in codes:
        assert code in taxonomy, f"Missing taxonomy entry for {code}"


def test_sample_days_load_and_have_tasks():
    samples = load_sample_days(str(config.sample_days_path))
    assert len(samples) == 3
    for s in samples:
        assert "tasks" in s and len(s["tasks"]) > 0


def test_missing_file_returns_empty_not_crash(tmp_path):
    fake_path = str(tmp_path / "does_not_exist.csv")
    result = load_activity_codes(fake_path)
    assert result == {}


def test_old_schema_rows_are_skipped(tmp_path):
    old_csv = tmp_path / "old_schema.csv"
    old_csv.write_text(
        "activity_code,activity_label,category,keywords\n"
        '311,Cooking,domestic,"cooking,food"\n',
        encoding="utf-8",
    )
    assert load_activity_codes(str(old_csv)) == {}
