"""
Central configuration for CareVal AI.

Loads environment variables and exposes a single Config object used
throughout the app. Designed to never crash if optional GCP settings
are missing -- the app must always be able to fall back to local /
sample mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op in GCP where real env vars are injected)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "local").strip().lower())
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    gcp_project_id: str | None = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    gcp_location: str = field(
        default_factory=lambda: (
            os.getenv("GCP_LOCATION")
            or os.getenv("VERTEX_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or "us-central1"
        )
    )
    bigquery_dataset: str = field(default_factory=lambda: os.getenv("BIGQUERY_DATASET", "careval_ai"))
    bigquery_table: str = field(default_factory=lambda: os.getenv("BIGQUERY_TABLE", "day_logs"))

    use_vertex_ai: bool = field(default_factory=lambda: _get_bool("USE_VERTEX_AI", False))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Data file paths (never hardcoded elsewhere -- all derived from DATA_DIR)
    activity_codes_path: Path = DATA_DIR / "tus_activity_codes.csv"
    wage_equivalents_path: Path = DATA_DIR / "wage_equivalents.csv"
    mospi_benchmarks_path: Path = DATA_DIR / "mospi_benchmarks.json"
    # Optional structured JSON with the same keys as mospi_benchmarks.json.
    # Used only for time-use minutes / source attribution — never for wages.
    mospi_benchmarks_url: str | None = field(
        default_factory=lambda: os.getenv("MOSPI_BENCHMARKS_URL") or None
    )
    mospi_catalog_url: str = field(
        default_factory=lambda: os.getenv(
            "MOSPI_CATALOG_URL",
            "https://microdata.gov.in/nada/index.php/catalog/223",
        )
    )
    taxonomy_path: Path = DATA_DIR / "taxonomy.json"
    sample_days_path: Path = DATA_DIR / "sample_days.json"

    @property
    def is_gcp(self) -> bool:
        return self.environment == "gcp"

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key) or self.is_gcp

    def describe(self) -> dict:
        """Safe, non-secret summary for debugging / status displays."""
        return {
            "environment": self.environment,
            "gemini_configured": self.has_gemini_key,
            "gcp_project_id_set": bool(self.gcp_project_id),
            "use_vertex_ai": self.use_vertex_ai,
            "gcp_location": self.gcp_location,
        }


config = Config()
