"""Selecting an interpreter and an explainer from configuration.

`TRAVEL_INTEL_LLM_PROVIDER=fake` returns the deterministic pair. It is the default in tests
and CI, so the suite exercises the real code paths without a model server and without
network access — and the two implementations satisfy the same Protocols, so nothing
downstream knows which one it got.
"""

from __future__ import annotations

from travel_intel.config import LLMProvider, Settings, get_settings
from travel_intel.llm.client import OllamaClient
from travel_intel.llm.explain import Explainer, LLMExplainer, TemplateExplainer
from travel_intel.llm.interpret import (
    KeywordPreferenceInterpreter,
    LLMPreferenceInterpreter,
    PreferenceInterpreter,
)


def _client(settings: Settings) -> OllamaClient:
    return OllamaClient(
        model=settings.llm_model,
        host=settings.ollama_host,
        timeout_s=settings.llm_timeout_s,
        temperature=settings.llm_temperature,
    )


def build_interpreter(settings: Settings | None = None) -> PreferenceInterpreter:
    resolved = settings or get_settings()
    if resolved.llm_provider is LLMProvider.FAKE:
        return KeywordPreferenceInterpreter()
    return LLMPreferenceInterpreter(_client(resolved))


def build_explainer(settings: Settings | None = None) -> Explainer:
    resolved = settings or get_settings()
    if resolved.llm_provider is LLMProvider.FAKE:
        return TemplateExplainer()
    return LLMExplainer(_client(resolved))
