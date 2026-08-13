"""Runtime configuration. Everything overridable by environment, no secrets in code."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent

FIXTURE_SEARCH_PATH: tuple[Path, ...] = (
    PACKAGE_ROOT / "data" / "fixtures",
    Path.cwd() / "src" / "travel_intel" / "data" / "fixtures",
)
"""Where to look for the snapshot, in order.

The first entry is the real answer: reference data the default configuration cannot run
without ships *inside* the package, so it travels through wheels, containers and hosted
deployments unchanged. This started life as `<repo>/data/fixtures`, resolved by walking up
from this file — which works under an editable install, where this file really does sit in
the source tree, and breaks the moment the package is installed normally into
`site-packages/travel_intel/`.

The second entry is a fallback for a source checkout whose installed copy is stale. A hosted
deploy can run the app script straight from the repository while `import travel_intel`
resolves to a cached older install; the code is then new and the data missing, which is a
confusing way to fail.
"""


def locate_fixtures() -> Path:
    """First existing entry in the search path, or the packaged location if none exist."""
    for candidate in FIXTURE_SEARCH_PATH:
        if candidate.is_dir():
            return candidate
    return FIXTURE_SEARCH_PATH[0]


FIXTURES_DIR = locate_fixtures()

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
