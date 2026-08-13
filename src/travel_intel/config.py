"""Runtime configuration. Everything overridable by environment, no secrets in code."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = PACKAGE_ROOT / "data" / "fixtures"
"""The snapshot ships *inside* the package, not beside it in the repository.

This started life as `<repo>/data/fixtures`, resolved by walking up from this file. That
works under an editable install, where this file really does sit in the source tree — and
breaks the moment the package is installed normally, because then it lives in
`site-packages/travel_intel/` and walking up two levels lands in the Python library
directory. The symptom was a deployed app reporting `Available: none` while every local test
passed.

Reference data the default configuration cannot run without belongs with the code it serves,
so it travels through wheels, containers and hosted deployments unchanged.
"""

ARTIFACTS_DIR = Path("artifacts")
"""Relative to the working directory: these are regenerable outputs, not shipped assets."""


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
    llm_timeout_s: float = Field(default=180.0, gt=0)
    """Measured: llama3.1:8b on CPU takes ~100 s to write the explanation. The 60 s default
    this started with timed out on every real run and silently exercised the fallback."""
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    fixtures_dir: Path = FIXTURES_DIR
    artifacts_dir: Path = ARTIFACTS_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
