"""Runtime configuration. Everything overridable by environment, no secrets in code."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class DataMode(StrEnum):
    SNAPSHOT = "snapshot"
    """Frozen provider data. Offline, deterministic, the default for tests and evaluation."""
    LIVE = "live"
    """Real provider calls. Same adapter interface, non-reproducible by nature."""


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    FAKE = "fake"
    """Deterministic stub so the pipeline and its tests never depend on a model server."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAVEL_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_mode: DataMode = DataMode.SNAPSHOT
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    fixtures_dir: Path = FIXTURES_DIR
    artifacts_dir: Path = ARTIFACTS_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
