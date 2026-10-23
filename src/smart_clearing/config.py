"""Runtime configuration loaded from the environment.

Business rules do not read environment variables themselves. Callers obtain a
``Settings`` object and pass the values they need.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SIMULATION_DISCLAIMER = (
    "Synthetic simulation only. This software does not submit payment "
    "instructions, does not move money, and is not production-ready for real "
    "financial settlement."
)


class Settings(BaseSettings):
    """Process configuration.

    ``database_url`` is empty unless ``SCSE_DATABASE_URL`` is set. The example
    password in ``.env.example`` is a local placeholder, not an application secret
    baked into this module.
    """

    model_config = SettingsConfigDict(
        env_prefix="SCSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    log_level: str = "INFO"
    model_path: Path = Path("artifacts/shortfall_model.joblib")
    random_seed: int = 42
    allow_sqlite: bool = False
    currency_exponent: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, cached for the lifetime of the interpreter."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings so tests can change environment variables."""
    get_settings.cache_clear()


def configure_logging(level: str) -> None:
    """Configure root logging once for application entry points."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
