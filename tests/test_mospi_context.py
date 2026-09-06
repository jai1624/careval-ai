import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config as app_config
from src.utils.mospi_context import load_mospi_time_use_benchmarks


def test_mospi_context_uses_cached_file_when_no_live_url():
    pack = load_mospi_time_use_benchmarks()
    assert pack.used_cached_fallback is True
    assert pack.domestic["female_minutes_per_day"] == 289
    assert pack.caregiving["male_minutes_per_day"] == 75
    assert "catalog" in pack.catalog_url or "mospi" in pack.catalog_url.lower() or "microdata" in pack.catalog_url


def test_mospi_context_falls_back_when_live_json_is_invalid():
    with patch("src.utils.mospi_context.config") as mock_config:
        mock_config.mospi_benchmarks_url = "https://example.invalid/tus.json"
        mock_config.mospi_catalog_url = "https://microdata.gov.in/nada/index.php/catalog/223"
        mock_config.mospi_benchmarks_path = app_config.mospi_benchmarks_path
        with patch("src.utils.mospi_context._fetch_live_document", return_value={"nope": True}):
            pack = load_mospi_time_use_benchmarks()
    assert pack.used_cached_fallback is True
    assert pack.domestic["female_minutes_per_day"] == 289


def test_mospi_context_accepts_valid_live_json():
    live = {
        "source": "MoSPI TUS live",
        "catalog_url": "https://example.test/catalog",
        "domestic": {"female_minutes_per_day": 200, "male_minutes_per_day": 50},
        "caregiving": {"female_minutes_per_day": 100, "male_minutes_per_day": 40},
    }
    with patch("src.utils.mospi_context.config") as mock_config:
        mock_config.mospi_benchmarks_url = "https://example.test/tus.json"
        mock_config.mospi_catalog_url = "https://ignored.example/"
        mock_config.mospi_benchmarks_path = "unused"
        with patch("src.utils.mospi_context._fetch_live_document", return_value=live):
            pack = load_mospi_time_use_benchmarks()
    assert pack.used_cached_fallback is False
    assert pack.domestic["female_minutes_per_day"] == 200
    assert pack.source_label == "MoSPI TUS live"
